#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    Target t = get_jit_target_from_environment();

    if (!t.features_any_of({Target::CUDACapability30,
                    Target::CUDACapability32,
                    Target::CUDACapability35,
                    Target::CUDACapability50,
                    Target::CUDACapability61})) {
        printf("This test requires cuda enabled with cuda capability 3.0 or greater\n");
        return 0;
    }

    {
        // Shuffle test to do a small convolution
        Func f, g;
        Var x, y;

        f(x, y) = x + y;
        g(x, y) = f(x-1, y) + f(x+1, y);

        Var xo, xi, yi, yo;
        g.gpu_tile(x, y, xi, yi, 32, 2, TailStrategy::RoundUp).gpu_lanes(xi);
        f.compute_root();
        f.in(g).compute_at(g, yi).split(x, xo, xi, 32, TailStrategy::RoundUp).gpu_lanes(xi).unroll(xo);

        Buffer<int> out = g.realize(32, 4);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int correct = 2*(x + y);
                int actual = out(x, y);
                if (correct != actual) {
                    printf("out(%d, %d) = %d instead of %d\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    {
        // Broadcast test - an outer product access pattern
        Func a, b, c;
        Var x, y;
        a(x) = cast<float>(x);
        b(y) = cast<float>(y);
        c(x, y) = a(x) + 100 * b(y);

        a.compute_root();
        b.compute_root();

        Var xi, yi, yii;

        // You may notice a redundant split below. We want to use a
        // serial for loop for the loop over y inside this kernel, but
        // there still needs to be a thread loop to compute_at to get
        // warp-level storage (TODO: Add store_in, so that the storage
        // memory space isn't implicitly defined by the
        // storage/compute level).
        c.tile(x, y, xi, yi, 32, 32, TailStrategy::RoundUp)
            .gpu_blocks(x, y)
            .split(yi, yi, yii, 32)
            .gpu_threads(yi)
            .gpu_lanes(xi);
        a.in(c).compute_at(c, yi).gpu_lanes(x);
        b.in(c).compute_at(c, yi).gpu_lanes(y);

        Buffer<float> out = c.realize(32, 32);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                float correct = x + 100 * y;
                float actual = out(x, y);
                // The floats are small integers, so they should be exact.
                if (correct != actual) {
                    printf("out(%d, %d) = %f instead of %f\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    {
        // Vectorized broadcast test. Each lane is responsible for a
        // 2-vector from 'a' and a 2-vector from 'b' instead of a single
        // value.
        Func a, b, c;
        Var x, y;
        a(x) = cast<float>(x);
        b(y) = cast<float>(y);
        c(x, y) = a(x) + 100 * b(y);

        a.compute_root();
        b.compute_root();

        Var xi, yi, yii;

        c.tile(x, y, xi, yi, 64, 64, TailStrategy::RoundUp)
            .gpu_blocks(x, y)
            .split(yi, yi, yii, 64).unroll(yii, 2).gpu_threads(yi)
            .vectorize(xi, 2).gpu_lanes(xi);
        a.in(c).compute_at(c, yi).vectorize(x, 2).gpu_lanes(x);
        b.in(c).compute_at(c, yi).vectorize(y, 2).gpu_lanes(y);

        Buffer<float> out = c.realize(64, 64);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                float correct = x + 100 * y;
                float actual = out(x, y);
                // The floats are small integers, so they should be exact.
                if (correct != actual) {
                    printf("out(%d, %d) = %f instead of %f\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    {
        // A stencil chain where many of the lanes will be masked
        Func a, b, c, d;
        Var x, y;

        a(x, y) = x + y;
        a.compute_root();

        b(x, y) = a(x-1, y) + a(x, y) + a(x+1, y);
        c(x, y) = b(x-1, y) + b(x, y) + b(x+1, y);
        d(x, y) = c(x-1, y) + c(x, y) + c(x+1, y);

        Var xi, yi;
        // Compute 24-wide pieces of output per block. Should use 32
        // warp lanes to do so. The footprint on the input is 30, so
        // the last two lanes are always inactive. 26-wide blocks
        // would be a more efficient use of the gpu, but a less
        // interesting test.
        d.gpu_tile(x, y, xi, yi, 24, 2).gpu_lanes(xi);
        for (Func stage : {a.in(), b, c}) {
            stage.compute_at(d, yi).gpu_lanes(x);
        }

        Buffer<int> out = d.realize(24, 2);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int correct = 27*(x + y);
                int actual = out(x, y);
                if (correct != actual) {
                    printf("out(%d, %d) = %d instead of %d\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    {
        // Same as above, but in half-warps
        Func a, b, c, d;
        Var x, y;

        a(x, y) = x + y;
        a.compute_root();

        b(x, y) = a(x-1, y) + a(x, y) + a(x+1, y);
        c(x, y) = b(x-1, y) + b(x, y) + b(x+1, y);
        d(x, y) = c(x-1, y) + c(x, y) + c(x+1, y);

        Var xi, yi;
        // Compute 10-wide pieces of output per block. Should use 16
        // warp lanes to do so.
        d.gpu_tile(x, y, xi, yi, 10, 2).gpu_lanes(xi);
        for (Func stage : {a.in(), b, c}) {
            stage.compute_at(d, yi).gpu_lanes(x);
        }

        Buffer<int> out = d.realize(24, 2);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int correct = 27*(x + y);
                int actual = out(x, y);
                if (correct != actual) {
                    printf("out(%d, %d) = %d instead of %d\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    if (1) { // TODO: Doesn't work yet.
        // A shuffle with a shift amount that depends on the y coord
        Func a, b;
        Var x, y;

        a(x, y) = x + y;
        b(x, y) = a(x + y, y);

        Var xi, yi;
        b.gpu_tile(x, y, xi, yi, 32, 8, TailStrategy::RoundUp).gpu_lanes(xi);
        a.compute_at(b, yi).split(x, x, xi, 32, TailStrategy::RoundUp).gpu_lanes(xi);

        Buffer<int> out = b.realize(32, 32);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                int correct = x + 2*y;
                int actual = out(x, y);
                if (correct != actual) {
                    printf("out(%d, %d) = %d instead of %d\n",
                           x, y, actual, correct);
                    return -1;
                }
            }
        }
    }

    if (0) { // TODO: This generates horrible IR and PTX
        // An upsample
        Func a, b;
        Var x, y;

        a(x, y) = x + y;
        b(x, y) = a(x/2, y/2);

        Var xi, yi;
        b.align_bounds(x, 2).unroll(x, 2).gpu_tile(x, y, xi, yi, 32, 2, TailStrategy::RoundUp).gpu_lanes(xi);
        a.compute_root().in().compute_at(b, yi).gpu_lanes(x);

        b.realize(64, 64);
    }

    {
        // Half-warp shuffle with a shift amount that depends on the y coord

    }

    // TODO add custom lowering pass that verifies no shared usage

    printf("Success!\n");
    return 0;
}
