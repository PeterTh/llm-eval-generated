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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

// Thread block tile. x is the fastest-varying dimension for coalesced access.
// Each block stages its tile plus a one-cell halo in shared memory, which is
// all the 7-point stencil needs.
constexpr int TILE_X = 32;
constexpr int TILE_Y = 4;
constexpr int TILE_Z = 4;
constexpr int SHARED_X = TILE_X + 2;
constexpr int SHARED_Y = TILE_Y + 2;
constexpr int SHARED_Z = TILE_Z + 2;
constexpr int SHARED_SIZE = SHARED_X * SHARED_Y * SHARED_Z;
constexpr unsigned int MAX_GRID_YZ = 65535;
constexpr int INIT_BLOCK = 256;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Clamp a (possibly negative) coordinate into [0, n-1]; this implements the
// clamped boundary conditions of the original code.
__device__ __forceinline__ size_t clampIndex(const long long v, const size_t n) {
    if (v < 0) return 0;
    if (v >= static_cast<long long>(n)) return n - 1;
    return static_cast<size_t>(v);
}

__device__ __forceinline__ int sharedIdx(const int sx, const int sy, const int sz) {
    return (sz * SHARED_Y + sy) * SHARED_X + sx;
}

// Cooperatively load the (TILE + 2*halo) region of `in` around the block origin
// into shared memory, using clamped global coordinates.
__device__ __forceinline__ void loadTile(double* __restrict__ s, const double* __restrict__ in,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const size_t x0, const size_t y0, const size_t z0) {
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int nThreads = blockDim.x * blockDim.y * blockDim.z;
    for (int t = tid; t < SHARED_SIZE; t += nThreads) {
        const int sx = t % SHARED_X;
        const int sy = (t / SHARED_X) % SHARED_Y;
        const int sz = t / (SHARED_X * SHARED_Y);
        const size_t gx = clampIndex(static_cast<long long>(x0) + sx - 1, nx);
        const size_t gy = clampIndex(static_cast<long long>(y0) + sy - 1, ny);
        const size_t gz = clampIndex(static_cast<long long>(z0) + sz - 1, nz);
        s[t] = in[idx3(gx, gy, gz, nx, ny)];
    }
}

// Laplacian at shared-memory center index c0 (same arithmetic as computeLaplacian)
__device__ __forceinline__ double laplacianShared(const double* __restrict__ s, const int sx, const int sy, const int sz,
                                                  const double dx, const double dy, const double dz) {
    const int c0 = sharedIdx(sx, sy, sz);
    const int strideY = SHARED_X;
    const int strideZ = SHARED_X * SHARED_Y;

    const double cxx = (s[c0 + 1] + s[c0 - 1] - 2.0 * s[c0]) / (dx * dx);
    const double cyy = (s[c0 + strideY] + s[c0 - strideY] - 2.0 * s[c0]) / (dy * dy);
    const double czz = (s[c0 + strideZ] + s[c0 - strideZ] - 2.0 * s[c0]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential: mu = f(c) - gamma * lap(c)
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    __shared__ double s[SHARED_SIZE];

    const size_t tilesY = (ny + TILE_Y - 1) / TILE_Y;
    const size_t tilesZ = (nz + TILE_Z - 1) / TILE_Z;
    const size_t x0 = static_cast<size_t>(blockIdx.x) * TILE_X;
    const size_t x = x0 + threadIdx.x;

    for (size_t tz = blockIdx.z; tz < tilesZ; tz += gridDim.z) {
        for (size_t ty = blockIdx.y; ty < tilesY; ty += gridDim.y) {
            const size_t y0 = ty * TILE_Y;
            const size_t z0 = tz * TILE_Z;

            loadTile(s, c, nx, ny, nz, x0, y0, z0);
            __syncthreads();

            const size_t y = y0 + threadIdx.y;
            const size_t z = z0 + threadIdx.z;
            if (x < nx && y < ny && z < nz) {
                const int sx = threadIdx.x + 1;
                const int sy = threadIdx.y + 1;
                const int sz = threadIdx.z + 1;
                const double cv = s[sharedIdx(sx, sy, sz)];
                const double lap = laplacianShared(s, sx, sy, sz, dx, dy, dz);

                mu[idx3(x, y, z, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                          + 3.0 * cv + cv * cv * cv
                                          - gamma * lap;
            }
            __syncthreads();
        }
    }
}

// Cahn-Hilliard update: cnew = cold + dt * D * lap(mu)
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt, const double dx, const double dy, const double dz) {
    __shared__ double s[SHARED_SIZE];

    const size_t tilesY = (ny + TILE_Y - 1) / TILE_Y;
    const size_t tilesZ = (nz + TILE_Z - 1) / TILE_Z;
    const size_t x0 = static_cast<size_t>(blockIdx.x) * TILE_X;
    const size_t x = x0 + threadIdx.x;

    for (size_t tz = blockIdx.z; tz < tilesZ; tz += gridDim.z) {
        for (size_t ty = blockIdx.y; ty < tilesY; ty += gridDim.y) {
            const size_t y0 = ty * TILE_Y;
            const size_t z0 = tz * TILE_Z;

            loadTile(s, mu, nx, ny, nz, x0, y0, z0);
            __syncthreads();

            const size_t y = y0 + threadIdx.y;
            const size_t z = z0 + threadIdx.z;
            if (x < nx && y < ny && z < nz) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double lap = laplacianShared(s, threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1, dx, dy, dz);
                cnew[idx] = cold[idx] + dt * D * lap;
            }
            __syncthreads();
        }
    }
}

// Initialize concentration field (grid-stride loop over linear indices)
__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t vol) {
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

// Launch configuration for the tiled stencil kernels. Blocks along y/z are
// capped at the CUDA grid limit; the kernels loop over any remaining tiles.
static dim3 stencilGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t tilesX = (nx + TILE_X - 1) / TILE_X;
    const size_t tilesY = (ny + TILE_Y - 1) / TILE_Y;
    const size_t tilesZ = (nz + TILE_Z - 1) / TILE_Z;
    return dim3(static_cast<unsigned int>(tilesX),
                static_cast<unsigned int>(std::min<size_t>(tilesY, MAX_GRID_YZ)),
                static_cast<unsigned int>(std::min<size_t>(tilesZ, MAX_GRID_YZ)));
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
    
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    
    // Allocate device arrays (the field stays resident on the GPU for the whole run)
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, bytes));
    CUDA_CHECK(cudaMalloc(&d_cnew, bytes));
    CUDA_CHECK(cudaMalloc(&d_mu, bytes));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    if (gridSize > 0) {
        const size_t initBlocks = (gridSize + INIT_BLOCK - 1) / INIT_BLOCK;
        initializeConcentrationKernel<<<static_cast<unsigned int>(initBlocks), INIT_BLOCK>>>(d_cold, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    const dim3 block(TILE_X, TILE_Y, TILE_Z);
    const dim3 grid = stencilGrid(nx, ny, nz);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations && gridSize > 0; ++t) {
        // Compute chemical potential
        chemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz,
                                                 gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());
        
        // Update concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
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
    
    // Copy the final concentration field back to the host, if needed
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        if (gridSize > 0) {
            CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
        }
    }
    
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
