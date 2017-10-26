#include "LowerWarpShuffles.h"
#include "ExprUsesVar.h"
#include "IREquality.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Solve.h"

namespace Halide {
namespace Internal {

using std::vector;
using std::string;
using std::map;
using std::pair;
using std::set;

class LowerWarpShuffles : public IRMutator {
    using IRMutator::visit;

    Expr warp_size, this_lane;
    string this_lane_name;
    vector<Stmt> allocations;
    Scope<int> allocation_size;
    Scope<Expr> varies_across_warp;
    Scope<Interval> bounds;
    map<string, Expr> alloc_stride;
    set<string> warp_allocation_names;

    Expr warp_stride(Expr e) {
        // Get the derivative of an integer expression w.r.t the warp lane
        if (is_const(e)) {
            return 0;
        } if (const Variable *var = e.as<Variable>()) {
            if (varies_across_warp.contains(var->name)) {
                return varies_across_warp.get(var->name);
            } else {
                return 0;
            }
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
            if (!is_zero(sv)) {
                varies_across_warp.push(let->name, sv);
                Expr sb = warp_stride(let->body);
                varies_across_warp.pop(let->name);
                return sb;
            }
        }

        return Expr();
    }

    void visit(const For *op) {
        bool should_pop = false;
        if (is_const(op->min) && is_const(op->extent)) {
            should_pop = true;
            bounds.push(op->name, Interval(op->min, simplify(op->min + op->extent - 1)));
        }
        if (op->for_type == ForType::GPULane) {
            const int64_t *loop_size = as_const_int(op->extent);
            user_assert(loop_size && *loop_size <= 32)
                << "CUDA gpu lanes loop must have constant extent of at most 32: " << op->extent << "\n";

            // Select a warp size - the smallest power of two that contains the loop size
            int64_t ws = 1;
            while (ws < *loop_size) {
                ws *= 2;
            }

            warp_size = make_const(Int(32), ws);
            this_lane_name = op->name;
            this_lane = Variable::make(Int(32), op->name);
            varies_across_warp.push(op->name, 1);

            // Figure out the shrunken size of the hoisted allocations
            // and population the scope.
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
                allocation_size.push(alloc->name, (int)(*sz));
            }

            Stmt body = mutate(op->body);

            if (ws != *loop_size) {
                // Mask off the excess lanes in the warp
                body = IfThenElse::make(this_lane < op->extent, body, Stmt());
            }

            // Wrap the hoisted warp-level allocations, at their new
            // reduced size.
            for (Stmt s : allocations) {
                const Allocate *alloc = s.as<Allocate>();
                internal_assert(alloc && alloc->extents.size() == 1);
                int new_size = allocation_size.get(alloc->name);
                allocation_size.pop(alloc->name);
                body = Allocate::make(alloc->name, alloc->type, {new_size}, alloc->condition,
                                      body, alloc->new_expr, alloc->free_function);
            }

            stmt = For::make(op->name, op->min, warp_size,
                             op->for_type, op->device_api, body);

            varies_across_warp.pop(op->name);
            warp_size = Expr();
            this_lane = Expr();
            this_lane_name.clear();
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

    void visit(const Let *op) {
        if (warp_size.defined() &&
            expr_uses_vars(op->value, varies_across_warp)) {
            varies_across_warp.push(op->name, warp_stride(op->value));
            IRMutator::visit(op);
            varies_across_warp.pop(op->name);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const LetStmt *op) {
        if (warp_size.defined() &&
            expr_uses_vars(op->value, varies_across_warp)) {
            varies_across_warp.push(op->name, warp_stride(op->value));
            IRMutator::visit(op);
            varies_across_warp.pop(op->name);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const Store *op) {
        if (warp_allocation_names.count(op->name)) {
            Expr idx = mutate(op->index);
            Expr value = mutate(op->value);

            // Simplify idx early to make the following error message better
            idx = simplify(idx, true, bounds);

            Expr stride = warp_stride(idx);

            user_assert(stride.defined())
                << "Stores to warp level storage must write in a strided pattern "
                << "across the warp lanes so that storage can be striped across the warp lanes. "
                << "Store index: " << idx << " does not follow such a pattern\n";

            stride = simplify(stride, true, bounds);

            auto it = alloc_stride.find(op->name);
            if (it != alloc_stride.end()) {
                Expr existing_stride = it->second;
                user_assert(equal(stride, existing_stride))
                    << "Different stores to " << op->name
                    << " write to inconsistent subsets of the elements of the allocation. "
                    << " One store writes with stride " << existing_stride
                    << " and another writes with stride " << stride << "\n";
            } else {
                alloc_stride[op->name] = stride;
            }

            user_assert(warp_size.defined())
                << "Can't use warp-level allocation for " << op->name
                << "; The gpu block width is not a power of 2 less than or equal to 32\n";

            // Check that the lane I'm writing to is my own
            Expr lane = simplify((idx / stride) % warp_size, true, bounds);
            if (const Broadcast *b = lane.as<Broadcast>()) {
                lane = b->value;
            }

            // Reduce the index to an index in my own stripe
            Expr in_warp_idx = simplify((idx / (warp_size * stride)) * stride + (idx % stride), true, bounds);

            // The index may not depend on the thread id, and the lane must be my own.
            user_assert(!expr_uses_vars(in_warp_idx, varies_across_warp) && equal(lane, this_lane))
                << "Stores to warp level storage must write in a strided pattern "
                << "across the warp lanes so that storage can be striped across the warp lanes. "
                << "Within each lane, a store may only write to one of its own lane's values. "
                << "Store index: " << idx << " does not follow such a pattern.\n";

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

        if (expr_uses_vars(idx, varies_across_warp)) {
            Expr equiv = make_warp_load(type, name, 0, lane);
            int elems = allocation_size.get(name);
            for (int i = 1; i < elems; i++) {
                // Load the right lanes from stripe number i
                equiv = select(idx >= i, make_warp_load(type, name, i, lane), equiv);
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

        string intrin_suffix;
        if (type == Float(32)) {
            intrin_suffix = ".f32";
        } else if (type == Int(32) || type == UInt(32)) {
            intrin_suffix = ".i32";
        } else {
            // TODO: bools
            user_error << "Warp shuffles only supported for scalar (u)int32_t and float\n";
        }

        // There are only a few patterns of lane access cuda can handle.
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
                                   {base_val, result[0], 31}, Call::Extern);
            return down;
        } else if (expr_match((this_lane + wild) % wild, lane, result) &&
            is_const_power_of_two_integer(result[1], &bits) &&
            bits <= 5) {
            result[0] = simplify(result[0] % result[1], true, bounds);
            // Rotate. Mux a shuffle up and a shuffle down
            Expr mask = (1 << bits) - 1;
            Expr down = Call::make(type, "llvm.nvvm.shfl.down" + intrin_suffix,
                                   {base_val, result[0], (1 << bits) - 1}, Call::Extern);
            Expr up = Call::make(type, "llvm.nvvm.shfl.up" + intrin_suffix,
                                 {base_val, (1 << bits) - result[0], 0, mask}, Call::Extern);
            Expr cond = (this_lane >= (1 << bits) - result[0]);
            Expr equiv = select(cond, up, down);
            return simplify(equiv, true, bounds);
        } else if (expr_match(wild % wild, lane, result) &&
                   !expr_uses_vars(result[0], varies_across_warp) &&
                   is_const_power_of_two_integer(result[1], &bits) &&
                   bits <= 5) {
            // Broadcast with modulo
            return Call::make(type, "llvm.nvvm.shfl.idx" + intrin_suffix,
                              {base_val, lane, 31}, Call::Extern);
        } else if (!expr_uses_vars(lane, varies_across_warp)) {
            // Broadcast
            return Call::make(type, "llvm.nvvm.shfl.idx" + intrin_suffix,
                              {base_val, lane, 31}, Call::Extern);
        }

        // TODO: butterfly, clamp
        user_error << "Unsupported access pattern for warp shuffle: " << lane << "\n";
        return Expr();
    }

    void visit(const Load *op) {
        if (warp_allocation_names.count(op->name)) {
            // Get the stride of the warp_size...
            auto it = alloc_stride.find(op->name);
            internal_assert(it != alloc_stride.end()) << "warp_load found before any warp_store\n";

            Expr idx = mutate(op->index);
            Expr stride = it->second;

            // Break the index into lane and stripe components
            Expr lane = simplify((idx / stride) % warp_size, true, bounds);
            idx = simplify((idx / (warp_size * stride)) * stride + (idx % stride), true, bounds);
            expr = make_warp_load(op->type, op->name, idx, lane);
        } else {
            IRMutator::visit(op);
        }
    }

    void visit(const Allocate *op) {
        if (op->name == "__shared") {
            IRMutator::visit(op);
        } else {
            // Pick up this allocation and deposit it inside the loop over lanes at reduced size.
            allocations.push_back(Stmt(op));
            warp_allocation_names.insert(op->name);
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
            LowerWarpShuffles shuffler;
            HoistWarpShuffles hoister;
            debug(0) << "BEFORE:\n" << Stmt(op) << "\n\n";
            stmt = hoister.mutate(shuffler.mutate(op));
            debug(0) << "AFTER:\n" << stmt << "\n\n";
        } else {
            IRMutator::visit(op);
        }
    }
};

Stmt lower_warp_shuffles(Stmt s) {
    return LowerWarpShufflesInEachKernel().mutate(s);
};

}
}
