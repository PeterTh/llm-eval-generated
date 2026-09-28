#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        const cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                                    \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,        \
                   cudaGetErrorString(err_));                                                         \
            exit(EXIT_FAILURE);                                                                       \
        }                                                                                             \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Launch tile: each thread owns one (x, y) column of the domain and marches it
// through a chunk of z slices, so the two z neighbours stay in registers and only
// the four in-plane neighbours are re-read (they hit the L1/L2 caches).
static constexpr int BX = 32;
static constexpr int BY = 4;

// Point-wise operation applied to the chemical potential: mu = f(c) - gamma * lap(c)
struct ChemicalPotentialOp {
    double gamma, e_AA, e_BB, e_AB;

    __device__ double operator()(const double cv, const double lap, const size_t) const {
        return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * lap;
    }
};

// Point-wise operation of the concentration update: cnew = cold + dt * D * lap(mu)
struct UpdateOp {
    const double* __restrict__ cold;
    double dtD;

    __device__ double operator()(const double, const double lap, const size_t idx) const {
        return cold[idx] + dtD * lap;
    }
};

// Register-sliding 7-point stencil sweep with clamped boundary conditions.
template <typename Op>
__global__ __launch_bounds__(BX* BY) void stencilKernel(const double* __restrict__ in, double* __restrict__ out,
                                                        const int nx, const int ny, const int nz, const int zchunk,
                                                        const double ihx2, const double ihy2, const double ihz2,
                                                        const Op op) {
    const int x = static_cast<int>(blockIdx.x) * BX + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * BY + static_cast<int>(threadIdx.y);
    const int zstart = static_cast<int>(blockIdx.z) * zchunk;
    if (x >= nx || y >= ny || zstart >= nz) return;

    const int zend = min(zstart + zchunk, nz);
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = static_cast<size_t>(y) * nx + x;

    // Clamped boundary conditions expressed as neighbour offsets (0 at a face,
    // which reproduces the mirrored center value of the original code).
    const int xpo = (x < nx - 1) ? 1 : 0;
    const int xno = (x > 0) ? -1 : 0;
    const int ypo = (y < ny - 1) ? nx : 0;
    const int yno = (y > 0) ? -nx : 0;

    double prev = __ldg(&in[static_cast<size_t>(max(zstart - 1, 0)) * plane + base]);
    double cur = __ldg(&in[static_cast<size_t>(zstart) * plane + base]);

    for (int z = zstart; z < zend; ++z) {
        const size_t idx = static_cast<size_t>(z) * plane + base;
        const double next = __ldg(&in[static_cast<size_t>(min(z + 1, nz - 1)) * plane + base]);

        const double cxx = (__ldg(&in[idx + xpo]) + __ldg(&in[idx + xno]) - 2.0 * cur) * ihx2;
        const double cyy = (__ldg(&in[idx + ypo]) + __ldg(&in[idx + yno]) - 2.0 * cur) * ihy2;
        const double czz = (next + prev - 2.0 * cur) * ihz2;

        out[idx] = op(cur, cxx + cyy + czz, idx);

        prev = cur;
        cur = next;
    }
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

    // Device setup
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    CUDA_CHECK(cudaMemcpy(d_cold, cold.data(), bytes, cudaMemcpyHostToDevice));

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    // Launch geometry: pick a z chunk size that keeps all SMs busy while
    // amortizing the halo reloads at the chunk boundaries.
    int smCount = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    const int blocksX = (inx + BX - 1) / BX;
    const int blocksY = (iny + BY - 1) / BY;
    const long long blocksXY = static_cast<long long>(blocksX) * blocksY;
    const long long wantZ = std::max(1ll, (8ll * smCount + blocksXY - 1) / blocksXY);
    int zchunk = static_cast<int>(std::max(4ll, (inz + wantZ - 1) / wantZ));
    zchunk = std::min(zchunk, inz);
    const int blocksZ = (inz + zchunk - 1) / zchunk;

    const dim3 block(BX, BY, 1);
    const dim3 grid(blocksX, blocksY, blocksZ);

    const double ihx2 = 1.0 / (dx * dx);
    const double ihy2 = 1.0 / (dy * dy);
    const double ihz2 = 1.0 / (dz * dz);

    const ChemicalPotentialOp muOp{gamma, e_AA, e_BB, e_AB};

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        stencilKernel<<<grid, block>>>(d_cold, d_mu, inx, iny, inz, zchunk, ihx2, ihy2, ihz2, muOp);

        // Update concentration
        const UpdateOp updOp{d_cold, dt * D};
        stencilKernel<<<grid, block>>>(d_mu, d_cnew, inx, iny, inz, zchunk, ihx2, ihy2, ihz2, updOp);

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
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
