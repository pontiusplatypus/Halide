#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    Target t = get_jit_target_from_environment();

    if (!t.has_feature(Target::CUDA)) {
        printf("This is a test of a cuda-specific feature.\n");
        return 0;
    }

    {
        // Shuffle test to do a small convolution
        Func f, g;
        Var x, y;

        f(x, y) = x + y;
        g(x, y) = f(x-1, y) + f(x+1, y);

        Var xo, xi, yi, yo;
        g.gpu_tile(x, y, xi, yi, 32, 2, TailStrategy::RoundUp);
        f.compute_root();
        f.in(g).compute_at(g, yi).split(x, xo, xi, 32, TailStrategy::RoundUp).gpu_threads(xi).unroll(xo);

        Buffer<int> out = g.realize(32, 4);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                printf("%d ", out(x, y));
            }
            printf("\n");
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

        Var xi, yi, z, zi;

        c.split(y, y, z, 1).reorder(x, y, z).gpu_tile(x, y, z, xi, yi, zi, 32, 32, 1, TailStrategy::RoundUp);

        a.in(c).compute_at(c, zi).gpu_threads(x);
        b.in(c).compute_at(c, zi).gpu_threads(y);

        Buffer<float> out = c.realize(32, 32);
        for (int y = 0; y < out.height(); y++) {
            for (int x = 0; x < out.width(); x++) {
                printf("%1.0f ", out(x, y));
            }
            printf("\n");
        }
    }

    // TODO: Handle if statements in kernels. Not safe to do a shuffle when some lanes are off.

    return 0;
}
