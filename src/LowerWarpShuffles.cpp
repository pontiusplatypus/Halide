#include "LowerWarpShuffles.h"
#include "ExprUsesVar.h"
#include "IREquality.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Solve.h"
#include "Substitute.h"
#include "LICM.h"

namespace Halide {
namespace Internal {

using std::vector;
using std::string;
using std::map;
using std::pair;
using std::set;

namespace {

Expr reduce_expr_helper(Expr e, Expr modulus) {
    if (is_one(modulus)) {
        return make_zero(e.type());
    } else if (is_const(e)) {
        return simplify(e % modulus);
    } else if (const Add *add = e.as<Add>()) {
        return (reduce_expr_helper(add->a, modulus) + reduce_expr_helper(add->b, modulus));
    } else if (const Sub *sub = e.as<Sub>()) {
        return (reduce_expr_helper(sub->a, modulus) - reduce_expr_helper(sub->b, modulus));
    } else if (const Mul *mul = e.as<Mul>()) {
        if (is_const(mul->b) && can_prove(modulus % mul->b == 0)) {
            return reduce_expr_helper(mul->a, simplify(modulus / mul->b)) * mul->b;
        } else {
            return reduce_expr_helper(mul->a, modulus) * reduce_expr_helper(mul->b, modulus);
        }
    } else if (const Ramp *ramp = e.as<Ramp>()) {
        return Ramp::make(reduce_expr_helper(ramp->base, modulus), reduce_expr_helper(ramp->stride, modulus), ramp->lanes);
    } else if (const Broadcast *b = e.as<Broadcast>()) {
        return Broadcast::make(reduce_expr_helper(b->value, modulus), b->lanes);
    } else {
        return e;
    }
}

// Try to reduce all terms in an affine expression modulo a given
// modulus, making as many simplifications as possible.
Expr reduce_expr(Expr e, Expr modulus, const Scope<Interval> &bounds) {
    e = reduce_expr_helper(simplify(e, true, bounds), modulus);
    if (is_one(simplify(e >= 0 && e < modulus, true, bounds))) {
        return e;
    } else {
        return e % modulus;
    }
}


// Substitute the warp lane variable inwards to make future passes simpler
class SubstituteInWarpLane : public IRMutator {
    using IRMutator::visit;

    Scope<Expr> gpu_vars;

    void visit(const Let *op) {
        if (expr_uses_vars(op->value, gpu_vars)) {
            expr = mutate(substitute(op->name, op->value, op->body));
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const LetStmt *op) {
        if (expr_uses_vars(op->value, gpu_vars)) {
            stmt = mutate(substitute(op->name, op->value, op->body));
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const For *op) {
        if (op->for_type == ForType::GPUBlock ||
            op->for_type == ForType::GPUThread ||
            op->for_type == ForType::GPULane) {
            gpu_vars.push(op->name, 0);
            IRMutator::visit(op);
            gpu_vars.pop(op->name);
        } else {
            IRMutator::visit(op);
        }
    }
};

// Determine a good way to stripe each allocations over warps by
// analyzing all uses.
class DetermineAllocStride : public IRVisitor {

    using IRVisitor::visit;

    const string &alloc, &lane_var;
    Expr warp_size;
    bool single_thread = false;
    vector<Expr> loads, stores, single_stores;

    Scope<Interval> bounds;

    Expr warp_stride(Expr e) {
        // Get the derivative of an integer expression w.r.t the warp lane
        if (!expr_uses_var(e, lane_var)) {
            return 0;
        } else if (e.as<Variable>()) {
            return 1;
        } else if (const Add *add = e.as<Add>()) {
            Expr sa = warp_stride(add->a), sb = warp_stride(add->b);
            if (sa.defined() && sb.defined()) {
                return sa + sb;
            }
        } else if (const Sub *sub = e.as<Sub>()) {
            Expr sa = warp_stride(sub->a), sb = warp_stride(sub->b);
            if (sa.defined() && sb.defined()) {
                return sa - sb;
            }
        } else if (const Mul *mul = e.as<Mul>()) {
            Expr sa = warp_stride(mul->a), sb = warp_stride(mul->b);
            if (sa.defined() && sb.defined() && is_zero(sb)) {
                return sa * mul->b;
            }
        } else if (const Broadcast *b = e.as<Broadcast>()) {
            return warp_stride(b->value);
        } else if (const Ramp *r = e.as<Ramp>()) {
            Expr sb = warp_stride(r->base);
            Expr ss = warp_stride(r->stride);
            if (sb.defined() && ss.defined() && is_zero(ss)) {
                return sb;
            }
        } else if (const Let *let = e.as<Let>()) {
            Expr sv = warp_stride(let->value);
            if (is_zero(sv)) {
                return warp_stride(let->body);
            }
        }
        return Expr();
    }

    void visit(const Store *op) {
        if (op->name == alloc) {
            if (single_thread) {
                single_stores.push_back(op->index);
            } else {
                stores.push_back(op->index);
            }
        }
        IRVisitor::visit(op);
    }

    void visit(const Load *op) {
        if (op->name == alloc) {
            loads.push_back(op->index);
        }
        IRVisitor::visit(op);
    }

    void visit(const IfThenElse *op) {
        // When things drop down to a single thread, we have different constraints.
        if (equal(op->condition, Variable::make(Int(32), lane_var) < 1)) {
            bool old_single_thread = single_thread;
            single_thread = true;
            op->then_case.accept(this);
            single_thread = old_single_thread;
            if (op->else_case.defined()) {
                op->else_case.accept(this);
            }
        } else {
            IRVisitor::visit(op);
        }
    }

    void visit(const For *op) {
        if (is_const(op->min) && is_const(op->extent)) {
            bounds.push(op->name, Interval(op->min, simplify(op->min + op->extent - 1)));
        }
        IRVisitor::visit(op);
    }

    void fail(const vector<Expr> &bad) {
        std::ostringstream message;
        message
            << "Access pattern for " << alloc << " does not meet the requirements for its store_at location. "
            << "All access to an allocation scheduled inside a loop over GPU "
            << "threads and outside a loop over GPU lanes must obey the following constraint:\n"
            << "The index must be affine in the gpu_lane variable with a consistent linear "
            << "term across all stores, and a constant term which, when divided by the stride "
            << "(rounding down), becomes a multiple of the warp size.\n";
        if (!stores.empty()) {
            message << alloc << " is stored to at the following indices by multiple lanes:\n";
            for (Expr e : stores) {
                message << "  " << e << "\n";
            }
        }
        if (!single_stores.empty()) {
            message << "And the following indicies by lane zero:\n";
            for (Expr e : single_stores) {
                message << "  " << e << "\n";
            }
        }
        if (!loads.empty()) {
            message << "And loaded from at the following indices:\n";
            for (Expr e : loads) {
                message << "  " << e << "\n";
            }
        }
        message << "The problematic indices are:\n";
        for (Expr e : bad) {
            message << "  " << e << "\n";
        }
        user_error << message.str();
    }

public:
    DetermineAllocStride(const string &alloc, const string &lane_var, const Expr &warp_size) :
        alloc(alloc), lane_var(lane_var), warp_size(warp_size) {}

    bool can_prove(const Expr &e) {
        return is_one(simplify(e, true, bounds));
    }

    Expr get_stride() {
        bool ok = true;
        Expr stride;
        Expr var = Variable::make(Int(32), lane_var);
        vector<Expr> bad;
        for (Expr e : stores) {
            Expr s = warp_stride(e);
            if (!stride.defined()) {
                stride = s;
            }

            internal_assert(stride.defined());

            bool this_ok = (s.defined() &&
                            (can_prove(stride == s) &&
                             can_prove(reduce_expr(e / stride - var, warp_size, bounds) == 0)));
            if (!this_ok) {
                bad.push_back(e);
            }
            ok = ok && this_ok;
        }

        for (Expr e : loads) {
            // We can handle any access pattern for loads, but it's
            // better if the stride matches up because then it's just
            // a register access, not a warp shuffle.
            //
            // TODO: Can't do loads across different thread_id.{yz} without a sync
            Expr s = warp_stride(e);
            if (!stride.defined()) {
                stride = s;
            }
        }

        if (stride.defined()) {
            for (Expr e : single_stores) {
                Expr simpler = substitute(lane_var, 0, e);
                bool this_ok = can_prove(reduce_expr(simpler / stride, warp_size, bounds) == 0);
                if (!this_ok) {
                    bad.push_back(e);
                }
                ok = ok && this_ok;
            }
        }

        if (!ok) fail(bad);

        if (!stride.defined()) {
            // Only accessed via broadcasts and a single store.
            stride = 1;
        }

        return stride;
    }
};

// TODO: There should be a separate visitor for the stuff inside the loop and the locating of the loops. Avoid all this_lane.defined()
class LowerWarpShuffles : public IRMutator {
    using IRMutator::visit;

    Expr warp_size, this_lane;
    string this_lane_name;
    bool may_use_warp_shuffle;
    vector<Stmt> allocations;
    struct AllocInfo {
        int size;
        Expr stride;
    };
    Scope<AllocInfo> allocation_info;
    Scope<Interval> bounds;

    void visit(const For *op) {
        bool should_pop = false;
        if (is_const(op->min) && is_const(op->extent)) {
            should_pop = true;
            bounds.push(op->name, Interval(op->min, simplify(op->min + op->extent - 1)));
        }
        if (!this_lane.defined() &&
            (op->for_type == ForType::GPULane ||
             (op->for_type == ForType::GPUThread && !allocations.empty()))) {

            debug(0) << "Lowering warp shuffles in loop over " << op->name << ": " << allocations.size() << "\n";

            bool should_mask = false;
            Expr old_warp_size = warp_size;
            if (op->for_type == ForType::GPULane) {
                const int64_t *loop_size = as_const_int(op->extent);
                user_assert(loop_size && *loop_size <= 32)
                    << "CUDA gpu lanes loop must have constant extent of at most 32: " << op->extent << "\n";

                // Select a warp size - the smallest power of two that contains the loop size
                int64_t ws = 1;
                while (ws < *loop_size) {
                    ws *= 2;
                }
                should_mask = (ws != *loop_size);
                warp_size = make_const(Int(32), ws);
            } else {
                warp_size = op->extent;
            }
            this_lane_name = op->name;
            this_lane = Variable::make(Int(32), op->name);
            may_use_warp_shuffle = (op->for_type == ForType::GPULane);

            Stmt body = op->body;

            // Figure out the shrunken size of the hoisted allocations
            // and populate the scope.
            for (Stmt s : allocations) {
                const Allocate *alloc = s.as<Allocate>();
                internal_assert(alloc && alloc->extents.size() == 1);
                // The allocation has been moved into the lane loop,
                // with storage striped across the warp lanes, so the
                // size required per-lane is the old size divided by
                // the number of lanes (rounded up).
                Expr new_size = (alloc->extents[0] + op->extent - 1) / op->extent;
                new_size = simplify(new_size, true, bounds);
                new_size = find_constant_bound(new_size, Direction::Upper, bounds);
                const int64_t *sz = as_const_int(new_size);
                user_assert(sz) << "Warp-level allocation with non-constant size: "
                                << alloc->extents[0] << ". Use Func::bound_extent.";
                DetermineAllocStride stride(alloc->name, op->name, warp_size);
                body.accept(&stride);
                allocation_info.push(alloc->name, {(int)(*sz), stride.get_stride()});
            }

            body = mutate(op->body);

            if (should_mask) {
                // Mask off the excess lanes in the warp
                body = IfThenElse::make(this_lane < op->extent, body, Stmt());
            }

            // Wrap the hoisted warp-level allocations, at their new
            // reduced size.
            for (Stmt s : allocations) {
                const Allocate *alloc = s.as<Allocate>();
                internal_assert(alloc && alloc->extents.size() == 1);
                int new_size = allocation_info.get(alloc->name).size;
                allocation_info.pop(alloc->name);
                body = Allocate::make(alloc->name, alloc->type, alloc->memory_type,
                                      {new_size}, alloc->condition,
                                      body, alloc->new_expr, alloc->free_function);
            }
            allocations.clear();

            debug(0) << "Stmt after lowering warp shuffles in loop over: " << op->name << "\n" << body << "\n\n";

            this_lane = Expr();
            this_lane_name.clear();
            may_use_warp_shuffle = false;

            // Mutate the body once more to apply the same transformation to any inner loops
            body = mutate(body);

            // Rewrap any hoisted allocations that weren't placed outside some inner loop
            for (Stmt s : allocations) {
                const Allocate *alloc = s.as<Allocate>();
                body = Allocate::make(alloc->name, alloc->type, alloc->memory_type,
                                      alloc->extents, alloc->condition,
                                      body, alloc->new_expr, alloc->free_function);
            }
            allocations.clear();

            stmt = For::make(op->name, op->min, warp_size,
                             op->for_type, op->device_api, body);

            warp_size = old_warp_size;
        } else {
          IRMutator::visit(op);
        }
        if (should_pop) {
            bounds.pop(op->name);
        }
    }

    void visit(const IfThenElse *op) {
        // Consider lane-masking if-then-elses when determining the
        // active bounds of the lane index.

        const LT *lt = op->condition.as<LT>();
        if (lt && equal(lt->a, this_lane) && is_const(lt->b)) {
            Expr condition = mutate(op->condition);
            internal_assert(bounds.contains(this_lane_name));
            Interval interval = bounds.get(this_lane_name);
            interval.max = simplify(lt->b - 1);
            bounds.push(this_lane_name, interval);
            Stmt then_case = mutate(op->then_case);
            Stmt else_case = mutate(op->else_case);
            stmt = IfThenElse::make(condition, then_case, else_case);
            bounds.pop(this_lane_name);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const Store *op) {
        if (allocation_info.contains(op->name)) {
            Expr idx = mutate(op->index);
            Expr value = mutate(op->value);
            Expr stride = allocation_info.get(op->name).stride;
            internal_assert(stride.defined() && warp_size.defined());

            // Reduce the index to an index in my own stripe. We have already validated the legality of this.
            Expr in_warp_idx = simplify((idx / (warp_size * stride)) * stride + reduce_expr(idx, stride, bounds), true, bounds);
            stmt = Store::make(op->name, value, in_warp_idx, op->param, op->predicate);
        } else {
            IRMutator::visit(op);
        }
    }

    Expr make_warp_load(Type type, string name, Expr idx, Expr lane) {
        // idx: The index of the value within the local allocation
        // lane: Which thread's value we want. If it's our own, we can just use a load.

        // Do the equivalent load, and then ask for another lane's
        // value of the result. For this to work idx
        // must not depend on the thread ID.

        // We handle other cases by converting it to a select tree
        // that muxes between all possible values.

        if (expr_uses_var(idx, this_lane_name)) {
            Expr equiv = make_warp_load(type, name, make_zero(idx.type()), lane);
            int elems = allocation_info.get(name).size;
            debug(0) << "Making a monstrosity: " << elems << ": " << idx << "\n";
            for (int i = 1; i < elems; i++) {
                // Load the right lanes from stripe number i
                equiv = select(idx >= i, make_warp_load(type, name, make_const(idx.type(), i), lane), equiv);
            }
            return simplify(equiv, true, bounds);
        }

        // Load the value to be shuffled
        Expr base_val = Load::make(type, name, idx, Buffer<>(),
                                   Parameter(), const_true(idx.type().lanes()));

        Expr scalar_lane = lane;
        if (const Broadcast *b = scalar_lane.as<Broadcast>()) {
            scalar_lane = b->value;
        }
        if (equal(scalar_lane, this_lane)) {
            // This is a regular load. No shuffling required.
            return base_val;
        }

        internal_assert(may_use_warp_shuffle) << name << ", " << idx << ", " << lane << "\n";

        string intrin_suffix;
        if (type == Float(32)) {
            intrin_suffix = ".f32";
        } else if (type == Int(32) || type == UInt(32)) {
            intrin_suffix = ".i32";
        } else {
            // TODO: bools, vectors
            user_error << "Warp shuffles only supported for scalar (u)int32_t and float\n";
        }

        Expr wild = Variable::make(Int(32), "*");
        vector<Expr> result;
        int bits;

        // Move this_lane as far left as possible in the expression to
        // reduce the number of cases to check below.
        lane = solve_expression(lane, this_lane_name).result;

        if (expr_match(this_lane + wild, lane, result)) {
            // We know that 0 <= lane + wild < warp_size by how we
            // constructed it, so we can just do a shuffle down.
            Expr down = Call::make(type, "llvm.nvvm.shfl.down" + intrin_suffix,
                                   {base_val, result[0], 31}, Call::PureExtern);
            return down;
        } else if (expr_match((this_lane + wild) % wild, lane, result) &&
            is_const_power_of_two_integer(result[1], &bits) &&
            bits <= 5) {
            result[0] = simplify(result[0] % result[1], true, bounds);
            // Rotate. Mux a shuffle up and a shuffle down. Uses fewer
            // intermediate registers than using a general gather for
            // this.
            Expr mask = (1 << bits) - 1;
            Expr down = Call::make(type, "llvm.nvvm.shfl.down" + intrin_suffix,
                                   {base_val, result[0], (1 << bits) - 1}, Call::PureExtern);
            Expr up = Call::make(type, "llvm.nvvm.shfl.up" + intrin_suffix,
                                 {base_val, (1 << bits) - result[0], 0, mask}, Call::PureExtern);
            Expr cond = (this_lane >= (1 << bits) - result[0]);
            Expr equiv = select(cond, up, down);
            return simplify(equiv, true, bounds);
        } else {
            Expr mask = warp_size - 1;
            // The idx variant can do a general gather. Use it for all other cases.
            return Call::make(type, "llvm.nvvm.shfl.idx" + intrin_suffix,
                              {base_val, lane, mask}, Call::PureExtern);
        }
        // TODO: butterfly, clamp don't need to use the general gather

        internal_error << "Unsupported access pattern for warp shuffle: " << lane << "\n";
        return Expr();
    }

    void visit(const Load *op) {
        if (allocation_info.contains(op->name)) {
            Expr idx = mutate(op->index);
            Expr stride = allocation_info.get(op->name).stride;

            // Break the index into lane and stripe components
            Expr lane = simplify(reduce_expr(idx / stride, warp_size, bounds), true, bounds);
            idx = simplify((idx / (warp_size * stride)) * stride + reduce_expr(idx, stride, bounds), true, bounds);
            // We don't want the idx to depend on the lane var
            idx = simplify(solve_expression(idx, this_lane_name).result, true, bounds);
            debug(0) << "Making warp load: " << idx << ", " << lane << "\n";
            expr = make_warp_load(op->type, op->name, idx, lane);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const Allocate *op) {
        if (this_lane.defined() || op->name == "__shared") {
            // Not a warp-level allocation
            IRMutator::visit(op);
        } else {
            // Pick up this allocation and deposit it inside the loop over lanes at reduced size.
            allocations.push_back(Stmt(op));
            stmt = mutate(op->body);
        }
    }

public:
    LowerWarpShuffles() {}
};

class HoistWarpShufflesFromSingleIfStmt : public IRMutator {
    using IRMutator::visit;

    Scope<int> stored_to;
    vector<pair<string, Expr>> lifted_lets;

    void visit(const Call *op) {
        // If it was written outside this if clause but read inside of
        // it, we need to hoist it.
        //
        // TODO: What if it is written to both inside *and* outside of
        // this if clause. Is that possible?
        if (starts_with(op->name, "llvm.nvvm.shfl.") &&
            !expr_uses_vars(op, stored_to)) {
            string name = unique_name('t');
            lifted_lets.push_back({name, op});
            expr = Variable::make(op->type, name);
        } else {
            IRMutator::visit(op);
        }
    }

    template<typename ExprOrStmt, typename LetOrLetStmt>
    ExprOrStmt visit_let(const LetOrLetStmt *op) {
        Expr value = mutate(op->value);
        ExprOrStmt body = mutate(op->body);

        // If any of the lifted expressions use this, we also need to
        // lift this.
        bool should_lift = false;
        for (const auto &p : lifted_lets) {
            should_lift |= expr_uses_var(p.second, op->name);
        }

        if (should_lift) {
            lifted_lets.push_back({op->name, value});
            return body;
        } else {
            return LetOrLetStmt::make(op->name, value, body);
        }
    }

    void visit(const Let *op) {
        expr = visit_let<Expr>(op);
    }

    void visit(const LetStmt *op) {
        stmt = visit_let<Stmt>(op);
    }

    void visit(const Store *op) {
        stored_to.push(op->name, 0);
        IRMutator::visit(op);
    }
public:
    Stmt rewrap(Stmt s) {
        while (!lifted_lets.empty()) {
            const pair<string, Expr> &p = lifted_lets.back();
            s = LetStmt::make(p.first, p.second, s);
            lifted_lets.pop_back();
        }
        return s;
    }
};

// The destination *and source* for warp shuffles must be active
// threads, or the value is undefined, so we want to lift them out of
// if statements.
class HoistWarpShuffles : public IRMutator {
    using IRMutator::visit;

    void visit(const IfThenElse *op) {
        // Move all Exprs that contain a shuffle out of the body of
        // the if.
        Stmt then_case = mutate(op->then_case);
        Stmt else_case = mutate(op->else_case);

        HoistWarpShufflesFromSingleIfStmt hoister;
        then_case = hoister.mutate(then_case);
        else_case = hoister.mutate(else_case);
        Stmt s = IfThenElse::make(op->condition, then_case, else_case);
        stmt = hoister.rewrap(s);
    }
};

class LowerWarpShufflesInEachKernel : public IRMutator {
    using IRMutator::visit;

    void visit(const For *op) {
        if (op->device_api == DeviceAPI::CUDA) {
            Stmt s = op;
            debug(0) << "BEFORE:\n" << s << "\n\n";
            s = LowerWarpShuffles().mutate(s);
            debug(0) << "HOIST:\n" << s << "\n\n";
            s = HoistWarpShuffles().mutate(s);
            debug(0) << "AFTER:\n" << s << "\n\n";
            stmt = simplify(s);
        } else {
            IRMutator::visit(op);
        }
    }
};

}

Stmt lower_warp_shuffles(Stmt s) {
    s = loop_invariant_code_motion(s);
    s = SubstituteInWarpLane().mutate(s);
    s = simplify_exprs(s);
    s = LowerWarpShufflesInEachKernel().mutate(s);
    return s;
};

}
}
