#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Thread block shape (x fastest) and the interior tile it covers.
// Each block stages a (TILE + 2 halo) neighbourhood into shared memory.
constexpr size_t TILE_X = 32;
constexpr size_t TILE_Y = 4;
constexpr size_t TILE_Z = 4;
constexpr unsigned BLOCK_THREADS = TILE_X * TILE_Y * TILE_Z;
constexpr unsigned TILE_ELEMS = (TILE_X + 2) * (TILE_Y + 2) * (TILE_Z + 2);
// Upper bound for grid dimensions y and z (CUDA limit); larger extents are grid-strided
constexpr unsigned MAX_GRID_YZ = 65535;

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                    cudaGetErrorString(err_));                                                \
            exit(EXIT_FAILURE);                                                               \
        }                                                                                     \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct SimParams {
    size_t nx, ny, nz;
    double dx, dy, dz;
    double dt;
    double gamma, e_AA, e_BB, e_AB;
    double D;
};

// Clamp a signed index into [0, n-1]; implements the clamped boundary condition
__device__ inline size_t clampIndex(const long long i, const size_t n) {
    if (i < 0) return 0;
    if (i >= static_cast<long long>(n)) return n - 1;
    return static_cast<size_t>(i);
}

// Cooperatively load the tile (interior + 1-cell halo) starting at global (x0, y0, z0)
// from src into shared memory. Out-of-grid halo cells are clamped to the boundary.
__device__ inline void loadTile(double (&tile)[TILE_Z + 2][TILE_Y + 2][TILE_X + 2], const double* __restrict__ src,
                                const SimParams& p, const size_t x0, const size_t y0, const size_t z0) {
    const unsigned tid = (threadIdx.z * TILE_Y + threadIdx.y) * TILE_X + threadIdx.x;
    for (unsigned i = tid; i < TILE_ELEMS; i += BLOCK_THREADS) {
        const unsigned lx = i % (TILE_X + 2);
        const unsigned ly = (i / (TILE_X + 2)) % (TILE_Y + 2);
        const unsigned lz = i / ((TILE_X + 2) * (TILE_Y + 2));
        const size_t gx = clampIndex(static_cast<long long>(x0) + lx - 1, p.nx);
        const size_t gy = clampIndex(static_cast<long long>(y0) + ly - 1, p.ny);
        const size_t gz = clampIndex(static_cast<long long>(z0) + lz - 1, p.nz);
        tile[lz][ly][lx] = src[idx3(gx, gy, gz, p.nx, p.ny)];
    }
}

// Laplacian of the tile centred at local (lx, ly, lz); same operation order as the CPU reference
__device__ inline double laplacianTile(const double (&tile)[TILE_Z + 2][TILE_Y + 2][TILE_X + 2],
                                       const unsigned lx, const unsigned ly, const unsigned lz,
                                       const double dx, const double dy, const double dz) {
    const double cxx = (tile[lz][ly][lx + 1] + tile[lz][ly][lx - 1] -
                        2.0 * tile[lz][ly][lx]) / (dx * dx);
    const double cyy = (tile[lz][ly + 1][lx] + tile[lz][ly - 1][lx] -
                        2.0 * tile[lz][ly][lx]) / (dy * dy);
    const double czz = (tile[lz + 1][ly][lx] + tile[lz - 1][ly][lx] -
                        2.0 * tile[lz][ly][lx]) / (dz * dz);
    return cxx + cyy + czz;
}

// Compute chemical potential
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu, const SimParams p) {
    __shared__ double tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2];

    const size_t nby = (p.ny + TILE_Y - 1) / TILE_Y;
    const size_t nbz = (p.nz + TILE_Z - 1) / TILE_Z;

    for (size_t bz = blockIdx.z; bz < nbz; bz += gridDim.z) {
        for (size_t by = blockIdx.y; by < nby; by += gridDim.y) {
            const size_t bx = blockIdx.x;
            loadTile(tile, c, p, bx * TILE_X, by * TILE_Y, bz * TILE_Z);
            __syncthreads();

            const size_t x = bx * TILE_X + threadIdx.x;
            const size_t y = by * TILE_Y + threadIdx.y;
            const size_t z = bz * TILE_Z + threadIdx.z;
            if (x < p.nx && y < p.ny && z < p.nz) {
                const unsigned lx = threadIdx.x + 1;
                const unsigned ly = threadIdx.y + 1;
                const unsigned lz = threadIdx.z + 1;
                const double cv = tile[lz][ly][lx];
                const double lap = laplacianTile(tile, lx, ly, lz, p.dx, p.dy, p.dz);

                mu[idx3(x, y, z, p.nx, p.ny)] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                                              + 3.0 * cv + cv * cv * cv
                                              - p.gamma * lap;
            }
            __syncthreads();
        }
    }
}

// Cahn-Hilliard update step
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu, const SimParams p) {
    __shared__ double tile[TILE_Z + 2][TILE_Y + 2][TILE_X + 2];

    const size_t nby = (p.ny + TILE_Y - 1) / TILE_Y;
    const size_t nbz = (p.nz + TILE_Z - 1) / TILE_Z;

    for (size_t bz = blockIdx.z; bz < nbz; bz += gridDim.z) {
        for (size_t by = blockIdx.y; by < nby; by += gridDim.y) {
            const size_t bx = blockIdx.x;
            loadTile(tile, mu, p, bx * TILE_X, by * TILE_Y, bz * TILE_Z);
            __syncthreads();

            const size_t x = bx * TILE_X + threadIdx.x;
            const size_t y = by * TILE_Y + threadIdx.y;
            const size_t z = bz * TILE_Z + threadIdx.z;
            if (x < p.nx && y < p.ny && z < p.nz) {
                const size_t idx = idx3(x, y, z, p.nx, p.ny);
                const double lap = laplacianTile(tile, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1,
                                                 p.dx, p.dy, p.dz);
                cnew[idx] = cold[idx] + p.dt * p.D * lap;
            }
            __syncthreads();
        }
    }
}

// Initialize concentration field
__global__ void initializeConcentrationKernel(double* __restrict__ c, const SimParams p) {
    const size_t vol = p.nx * p.ny * p.nz;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linear_id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear_id < vol; linear_id += stride) {
        // Generate pseudo-random value in [-1, 1]
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[linear_id] = -1.0 + 2.0 * pseudo;
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
    SimParams p{};
    p.nx = nx;
    p.ny = ny;
    p.nz = nz;
    p.dx = 1.0;
    p.dy = 1.0;
    p.dz = 1.0;
    p.dt = 0.01;
    p.e_AA = -(2.0 / 9.0);
    p.e_BB = -(2.0 / 9.0);
    p.e_AB = (2.0 / 9.0);
    p.gamma = 0.5;
    p.D = 1.0;

    size_t gridSize = nx * ny * nz;

    // Allocate device arrays
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, gridSize * sizeof(double)));

    // Launch geometry for the stencil kernels
    const size_t nbx = (nx + TILE_X - 1) / TILE_X;
    const size_t nby = (ny + TILE_Y - 1) / TILE_Y;
    const size_t nbz = (nz + TILE_Z - 1) / TILE_Z;
    const dim3 stencilBlock(TILE_X, TILE_Y, TILE_Z);
    const dim3 stencilGrid(static_cast<unsigned>(nbx),
                           static_cast<unsigned>(std::min<size_t>(nby, MAX_GRID_YZ)),
                           static_cast<unsigned>(std::min<size_t>(nbz, MAX_GRID_YZ)));
    const unsigned initBlocks = static_cast<unsigned>((gridSize + BLOCK_THREADS - 1) / BLOCK_THREADS);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        initializeConcentrationKernel<<<initBlocks, BLOCK_THREADS>>>(cold, p);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0) {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotentialKernel<<<stencilGrid, stencilBlock>>>(cold, mu, p);
            CUDA_CHECK(cudaGetLastError());

            // Update concentration
            cahnHilliardUpdateKernel<<<stencilGrid, stencilBlock>>>(cnew, cold, mu, p);
            CUDA_CHECK(cudaGetLastError());

            // Swap buffers
            std::swap(cold, cnew);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int status = 0;

    // Bring the final field back to the host only when it is needed
    if (printResults || validate) {
        std::vector<double> result(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));

        // Print results for external validation
        if (printResults) {
            print_results(result, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(cold));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(mu));
    return status;
}
