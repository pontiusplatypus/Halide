#include "Halide.h"

using namespace Halide;

namespace {

void set_alignment_and_bounds(OutputImageParam p, int size) {
    p.set_host_alignment(16)
        .dim(0).set_bounds(0, size)
        .dim(1).set_stride(size);
}

class MatMul : public Halide::Generator<MatMul> {
public:

    GeneratorParam<int>   size {"size", 1024};
    Input<Buffer<float>>  A{"A", 2};
    Input<Buffer<float>>  B{"B", 2};

    Output<Buffer<float>> out{"out", 2};

    void generate() {
        Var x("x"), y("y"), p("p");

        Func prod("prod");
        RDom r(0, size);
        prod(x, y) += A(x, r) * B(r, y);

        Var xi, yi, xio, xii, yii, xo, warp, x_pair;
        RVar rxo, rxi;
        out(x, y) = prod(x, y);
        out.bound(x, 0, size)
            .bound(y, 0, size)
            .tile(x, y, xi, yi, 6*32, 8)
            .vectorize(xi, 2)
            .split(xi, xio, xii, 32)
            .reorder(xio, yi, xii, x, y)
            .unroll(xio)
            .unroll(yi)
            .gpu_blocks(x, y).split(xii, warp, xii, 32).gpu_threads(xii, warp);
        prod.compute_at(out, warp)
            .split(x, xo, xi, 64, TailStrategy::RoundUp)
            .vectorize(xi, 2)
            .gpu_threads(xi)
            .unroll(xo)
            .unroll(y)
            .update()
            .split(x, xo, xi, 64, TailStrategy::RoundUp)
            .vectorize(xi, 2)
            .gpu_threads(xi)
            .split(r.x, rxo, rxi, 32)
            .reorder(xi, y, xo, rxi, rxo)
            .unroll(xo)
            .unroll(y);

        Var Bx = B.in().args()[0], By = B.in().args()[1];
        Var Ax = A.in().args()[0], Ay = A.in().args()[1];
        B.in()
            .compute_at(prod, rxo)
            .split(Bx, xo, xi, 32)
            .gpu_threads(xi)
            .unroll(xo).unroll(By);

        set_alignment_and_bounds(A, size);
        set_alignment_and_bounds(B, size);
        set_alignment_and_bounds(out, size);
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(MatMul, mat_mul)
