#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Tile shape of the x-y plane processed by one thread block.
static constexpr int BX = 32;
static constexpr int BY = 4;

// Both sweeps of the algorithm are 7-point stencils with clamped boundaries and
// only differ in the per-cell arithmetic, so they share one templated kernel.
// The kernel walks a chunk of z-planes: the x/y neighbours of the current plane
// live in shared memory while the z neighbours are kept in registers and rolled
// forward, so every cell is loaded from global memory essentially once.
// SCALE_MODE picks how the second differences are divided by the squared grid
// spacings. All three forms produce bit-identical results for the spacings they
// are selected for (see pickScaleMode), but double-precision division is by far
// the most expensive instruction in the kernel, so it is avoided when possible.
enum ScaleMode {
    SCALE_DIV = 0,    // general case: divide by d^2
    SCALE_RECIP = 1,  // d^2 is a power of two: multiply by the exact reciprocal
    SCALE_UNIT = 2    // d^2 == 1: no scaling at all
};

template <bool CHEM_POT, int SCALE_MODE>
__global__ void __launch_bounds__(BX* BY) chStencilKernel(
    const double* __restrict__ in,    // c for the chemical potential, mu for the update
    const double* __restrict__ cold,  // only used by the update sweep
    double* __restrict__ out,
    const int nx, const int ny, const int nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB,
    const double dtD, const int zChunk) {
    __shared__ double s[BY + 2][BX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BX + tx;
    const int y = blockIdx.y * BY + ty;

    // Threads outside the domain still participate in the loads (clamped) so
    // that the halo entries they produce for their neighbours stay valid.
    const int cx = (x < nx) ? x : nx - 1;
    const int cy = (y < ny) ? y : ny - 1;
    const bool active = (x < nx) && (y < ny);

    const int xn = (cx > 0) ? cx - 1 : 0;
    const int xp = (cx < nx - 1) ? cx + 1 : nx - 1;
    const int yn = (cy > 0) ? cy - 1 : 0;
    const int yp = (cy < ny - 1) ? cy + 1 : ny - 1;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t row = static_cast<size_t>(cy) * nx;

    const int z0 = blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, nz);
    if (z0 >= z1) return;

    const double dx2 = (SCALE_MODE == SCALE_RECIP) ? 1.0 / (dx * dx) : dx * dx;
    const double dy2 = (SCALE_MODE == SCALE_RECIP) ? 1.0 / (dy * dy) : dy * dy;
    const double dz2 = (SCALE_MODE == SCALE_RECIP) ? 1.0 / (dz * dz) : dz * dz;

    double prev = in[static_cast<size_t>((z0 > 0) ? z0 - 1 : 0) * plane + row + cx];
    double cur = in[static_cast<size_t>(z0) * plane + row + cx];

    for (int z = z0; z < z1; ++z) {
        const size_t zoff = static_cast<size_t>(z) * plane;
        const double next = in[static_cast<size_t>((z < nz - 1) ? z + 1 : nz - 1) * plane + row + cx];

        s[ty + 1][tx + 1] = cur;
        if (tx == 0) s[ty + 1][0] = in[zoff + row + xn];
        if (tx == BX - 1) s[ty + 1][BX + 1] = in[zoff + row + xp];
        if (ty == 0) s[0][tx + 1] = in[zoff + static_cast<size_t>(yn) * nx + cx];
        if (ty == BY - 1) s[BY + 1][tx + 1] = in[zoff + static_cast<size_t>(yp) * nx + cx];
        __syncthreads();

        if (active) {
            const double sx = s[ty + 1][tx + 2] + s[ty + 1][tx] - 2.0 * cur;
            const double sy = s[ty + 2][tx + 1] + s[ty][tx + 1] - 2.0 * cur;
            const double sz = next + prev - 2.0 * cur;
            const double cxx = (SCALE_MODE == SCALE_UNIT) ? sx : ((SCALE_MODE == SCALE_RECIP) ? sx * dx2 : sx / dx2);
            const double cyy = (SCALE_MODE == SCALE_UNIT) ? sy : ((SCALE_MODE == SCALE_RECIP) ? sy * dy2 : sy / dy2);
            const double czz = (SCALE_MODE == SCALE_UNIT) ? sz : ((SCALE_MODE == SCALE_RECIP) ? sz * dz2 : sz / dz2);
            const double lap = cxx + cyy + czz;

            const size_t idx = zoff + row + cx;
            if (CHEM_POT) {
                // cur * (2 * e_AB) is bit-identical to (2 * cur) * e_AB (scaling by
                // two is exact) and hoists the doubling out of the inner loop.
                out[idx] = 4.5 * ((cur + 1.0) * e_AA + (cur - 1.0) * e_BB - cur * (2.0 * e_AB)) + 3.0 * cur + cur * cur * cur - gamma * lap;
            } else {
                out[idx] = cold[idx] + dtD * lap;
            }
        }
        __syncthreads();

        prev = cur;
        cur = next;
    }
}

// Select the cheapest scaling variant that is bit-exact for the given squared
// grid spacings: dividing by 1.0 is the identity, and dividing by a power of two
// is exactly the same as multiplying by its (exactly representable) reciprocal.
int pickScaleMode(const double dx2, const double dy2, const double dz2) {
    auto isPowerOfTwo = [](const double v) {
        int exp = 0;
        return std::isnormal(v) && std::isnormal(1.0 / v) && std::fabs(std::frexp(v, &exp)) == 0.5;
    };
    if (dx2 == 1.0 && dy2 == 1.0 && dz2 == 1.0) return SCALE_UNIT;
    if (isPowerOfTwo(dx2) && isPowerOfTwo(dy2) && isPowerOfTwo(dz2)) return SCALE_RECIP;
    return SCALE_DIV;
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    size_t gridSize = nx * ny * nz;

    // Allocate arrays
    std::vector<double> cold(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Set up the GPU
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, std::max<size_t>(bytes, 1)));
    CUDA_CHECK(cudaMalloc(&d_cnew, std::max<size_t>(bytes, 1)));
    CUDA_CHECK(cudaMalloc(&d_mu, std::max<size_t>(bytes, 1)));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    // Every block sweeps a contiguous range of z-planes, which lets it reuse the
    // z neighbours it has already loaded. The range is kept short enough that
    // there are plenty of blocks to fill all SMs, and capped so that even very
    // deep grids keep enough blocks in flight to hide memory latency.
    const int blocksXY = std::max(((inx + BX - 1) / BX) * ((iny + BY - 1) / BY), 1);
    const int targetBlocks = prop.multiProcessorCount * 32;
    const int zBlocks = std::max(std::min((targetBlocks + blocksXY - 1) / blocksXY, inz), 1);
    const int zChunk = std::min(std::max((inz + zBlocks - 1) / zBlocks, 1), 16);

    const dim3 block(BX, BY, 1);
    const dim3 grid(std::max((inx + BX - 1) / BX, 1), std::max((iny + BY - 1) / BY, 1),
                    std::max((inz + zChunk - 1) / zChunk, 1));

    const double dtD = dt * D;
    const int scaleMode = pickScaleMode(dx * dx, dy * dy, dz * dz);

    // One time step: chemical potential from c, then the concentration update from mu.
    auto launchStep = [&](const dim3& g, const int chunk) {
        auto launch = [&](auto mode) {
            constexpr int M = decltype(mode)::value;
            chStencilKernel<true, M><<<g, block>>>(d_cold, nullptr, d_mu, inx, iny, inz,
                                                   dx, dy, dz, gamma, e_AA, e_BB, e_AB, dtD, chunk);
            chStencilKernel<false, M><<<g, block>>>(d_mu, d_cold, d_cnew, inx, iny, inz,
                                                    dx, dy, dz, gamma, e_AA, e_BB, e_AB, dtD, chunk);
        };
        switch (scaleMode) {
            case SCALE_UNIT: launch(std::integral_constant<int, SCALE_UNIT>{}); break;
            case SCALE_RECIP: launch(std::integral_constant<int, SCALE_RECIP>{}); break;
            default: launch(std::integral_constant<int, SCALE_DIV>{}); break;
        }
    };

    // Force module load / kernel setup up front so it does not pollute the timing.
    // zChunk == 0 makes every block exit immediately without touching the data.
    launchStep(dim3(1, 1, 1), 0);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential, then update the concentration
        launchStep(grid, zChunk);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Microsecond resolution keeps the rate meaningful for runs that take well
    // under a millisecond on the GPU; the reported time stays in whole ms.
    auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (std::max<long>(durationUs.count(), 1) / 1000000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
