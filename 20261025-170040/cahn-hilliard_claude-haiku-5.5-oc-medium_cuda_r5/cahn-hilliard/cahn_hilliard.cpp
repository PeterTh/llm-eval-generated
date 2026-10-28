#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Abort with a diagnostic on any CUDA runtime error.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Thread block: TILE_X threads along x (contiguous memory), TILE_Y along y.
// Each block works on one (x, y) tile of a z-plane, staging the tile plus a
// one-cell halo in shared memory.
static constexpr int TILE_X = 32;
static constexpr int TILE_Y = 8;
static constexpr int THREADS_PER_BLOCK = TILE_X * TILE_Y;
static constexpr unsigned int MAX_GRID_YZ = 65535;  // grid.y / grid.z hardware limit

// Physical parameters (compile-time constants so the device code folds them)
static constexpr double DX = 1.0;
static constexpr double DY = 1.0;
static constexpr double DZ = 1.0;
static constexpr double DT = 0.01;
static constexpr double E_AA = -(2.0 / 9.0);
static constexpr double E_BB = -(2.0 / 9.0);
static constexpr double E_AB = (2.0 / 9.0);
static constexpr double GAMMA = 0.5;
static constexpr double D_COEF = 1.0;

// 3D index calculation
__host__ __device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z,
                                                const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initial concentration field (pseudo-random in [-1, 1]), one thread per cell
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t nx, const size_t ny,
                                              const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t linear_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_id >= vol) return;

    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[linear_id] = -1.0 + 2.0 * pseudo;
}

// Chemical potential: mu = f(c) - gamma * lap(c)
struct ChemicalPotentialOp {
    __device__ __forceinline__ double operator()(const double cv, const double lap, const size_t /*idx*/) const {
        return 4.5 * ((cv + 1.0) * E_AA + (cv - 1.0) * E_BB - 2.0 * cv * E_AB)
             + 3.0 * cv + cv * cv * cv
             - GAMMA * lap;
    }
};

// Concentration update: cnew = cold + dt * D * lap(mu)
struct UpdateOp {
    const double* __restrict__ cold;
    __device__ __forceinline__ double operator()(const double /*muv*/, const double lap, const size_t idx) const {
        return cold[idx] + DT * D_COEF * lap;
    }
};

// Generic 7-point clamped-boundary stencil pass.
// For every cell: lap = Laplacian(src)(cell), dst[cell] = op(src[cell], lap, cell).
// The x/y neighbourhood comes from a shared-memory tile; z neighbours are read
// from global memory. Clamped boundaries are realised by clamping the halo
// coordinates, which yields exactly the same operands as the reference code.
template <typename Op>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
stencilKernel(const double* __restrict__ src, double* __restrict__ dst,
              const size_t nx, const size_t ny, const size_t nz, const Op op) {
    __shared__ double tile[TILE_Y + 2][TILE_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_X + tx;
    const size_t tilesX = (nx + TILE_X - 1) / TILE_X;
    const size_t tilesY = (ny + TILE_Y - 1) / TILE_Y;

    // Grid-stride over tiles so that any grid size is covered.
    for (size_t z = blockIdx.z; z < nz; z += gridDim.z) {
        const size_t zp = (z < nz - 1) ? z + 1 : z;
        const size_t zn = (z > 0) ? z - 1 : 0;

        for (size_t by = blockIdx.y; by < tilesY; by += gridDim.y) {
            for (size_t bx = blockIdx.x; bx < tilesX; bx += gridDim.x) {
                const long long x0 = static_cast<long long>(bx) * TILE_X;
                const long long y0 = static_cast<long long>(by) * TILE_Y;

                // Cooperative load of the (TILE_Y+2) x (TILE_X+2) halo tile.
                constexpr int HALO_CELLS = (TILE_Y + 2) * (TILE_X + 2);
                for (int k = tid; k < HALO_CELLS; k += THREADS_PER_BLOCK) {
                    const int r = k / (TILE_X + 2);
                    const int c = k % (TILE_X + 2);
                    long long gx = x0 + c - 1;
                    long long gy = y0 + r - 1;
                    gx = gx < 0 ? 0 : (gx > static_cast<long long>(nx) - 1 ? static_cast<long long>(nx) - 1 : gx);
                    gy = gy < 0 ? 0 : (gy > static_cast<long long>(ny) - 1 ? static_cast<long long>(ny) - 1 : gy);
                    tile[r][c] = src[idx3(static_cast<size_t>(gx), static_cast<size_t>(gy), z, nx, ny)];
                }
                __syncthreads();

                const size_t x = static_cast<size_t>(x0) + tx;
                const size_t y = static_cast<size_t>(y0) + ty;
                if (x < nx && y < ny) {
                    const double center = tile[ty + 1][tx + 1];
                    const double xp = tile[ty + 1][tx + 2];
                    const double xn = tile[ty + 1][tx];
                    const double yp = tile[ty + 2][tx + 1];
                    const double yn = tile[ty][tx + 1];
                    const double zpv = src[idx3(x, y, zp, nx, ny)];
                    const double znv = src[idx3(x, y, zn, nx, ny)];

                    const double cxx = (xp + xn - 2.0 * center) / (DX * DX);
                    const double cyy = (yp + yn - 2.0 * center) / (DY * DY);
                    const double czz = (zpv + znv - 2.0 * center) / (DZ * DZ);
                    const double lap = cxx + cyy + czz;

                    const size_t id = idx3(x, y, z, nx, ny);
                    dst[id] = op(center, lap, id);
                }
                __syncthreads();  // tile is reused by the next iteration
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

    const size_t gridSize = nx * ny * nz;

    // Device buffers (kept resident on the GPU for the whole run)
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, gridSize * sizeof(double)));

    // Launch configurations
    const unsigned int initBlocks = static_cast<unsigned int>((gridSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const dim3 stencilBlock(TILE_X, TILE_Y);
    const dim3 stencilGrid(static_cast<unsigned int>((nx + TILE_X - 1) / TILE_X),
                           static_cast<unsigned int>(std::min<size_t>((ny + TILE_Y - 1) / TILE_Y, MAX_GRID_YZ)),
                           static_cast<unsigned int>(std::min<size_t>(nz, MAX_GRID_YZ)));

    // Initialize concentration field on the GPU
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        initializeConcentrationKernel<<<initBlocks, THREADS_PER_BLOCK>>>(d_cold, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const ChemicalPotentialOp muOp{};
    for (int t = 0; t < iterations && gridSize > 0; ++t) {
        // Compute chemical potential: mu = f(c) - gamma * lap(c)
        stencilKernel<<<stencilGrid, stencilBlock>>>(d_cold, d_mu, nx, ny, nz, muOp);
        CUDA_CHECK(cudaGetLastError());

        // Update concentration: cnew = cold + dt * D * lap(mu)
        const UpdateOp updOp{d_cold};
        stencilKernel<<<stencilGrid, stencilBlock>>>(d_mu, d_cnew, nx, ny, nz, updOp);
        CUDA_CHECK(cudaGetLastError());

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Bring the final field back to the host only when it is needed
    std::vector<double> cold(gridSize);
    if ((printResults || validate) && gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

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
