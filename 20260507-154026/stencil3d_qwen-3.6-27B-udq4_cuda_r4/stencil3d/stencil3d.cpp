#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Block dimensions for stencil kernel (8x8x8 = 512 threads, good occupancy on Ampere)
constexpr int STENCIL_BX = 8;
constexpr int STENCIL_BY = 8;
constexpr int STENCIL_BZ = 8;

// Initialize grid kernel - each thread initializes one cell
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t idx = z * (nx * ny) + y * nx + x;
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil kernel with shared memory tiling and multi-GPU z-axis partitioning.
// z_offset shifts the thread's local z into global coordinates so each GPU covers
// a disjoint slice of the z-axis while still reading halo data from Unified Memory.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz,
                               const ptrdiff_t z_offset) {
    // Shared memory tile with 1-cell halo on each side
    __shared__ Real s[STENCIL_BX + 2][STENCIL_BY + 2][STENCIL_BZ + 2];

    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    const size_t gx = blockIdx.x * STENCIL_BX + tx;
    const size_t gy = blockIdx.y * STENCIL_BY + ty;
    const ptrdiff_t gz = z_offset + (ptrdiff_t)(blockIdx.z * STENCIL_BZ + tz);

    // Clamp global coordinates to valid range for loading
    const size_t cx = gx < nx ? gx : nx - 1;
    const size_t cy = gy < ny ? gy : ny - 1;
    const size_t cz = gz < 0 ? 0 : gz >= (ptrdiff_t)nz ? nz - 1 : (size_t)gz;

    // --- Load main value (coalesced across threads) ---
    s[tx+1][ty+1][tz+1] = input[cz * (nx * ny) + cy * nx + cx];

    // --- Load halo values (only boundary threads participate) ---
    if (tx == 0) {
        const size_t x0 = gx > 0 ? gx - 1 : 0;
        s[0][ty+1][tz+1] = input[cz * (nx * ny) + cy * nx + x0];
    }
    if (tx == STENCIL_BX - 1) {
        const size_t xp = gx < nx - 1 ? gx + 1 : nx - 1;
        s[STENCIL_BX+1][ty+1][tz+1] = input[cz * (nx * ny) + cy * nx + xp];
    }
    if (ty == 0) {
        const size_t y0 = gy > 0 ? gy - 1 : 0;
        s[tx+1][0][tz+1] = input[cz * (nx * ny) + y0 * nx + cx];
    }
    if (ty == STENCIL_BY - 1) {
        const size_t yp = gy < ny - 1 ? gy + 1 : ny - 1;
        s[tx+1][STENCIL_BY+1][tz+1] = input[cz * (nx * ny) + yp * nx + cx];
    }
    if (tz == 0) {
        const ptrdiff_t z0 = gz - 1;
        const size_t sz0 = z0 < 0 ? 0 : z0 >= (ptrdiff_t)nz ? nz - 1 : (size_t)z0;
        s[tx+1][ty+1][0] = input[sz0 * (nx * ny) + cy * nx + cx];
    }
    if (tz == STENCIL_BZ - 1) {
        const ptrdiff_t zp = gz + 1;
        const size_t szp = zp < 0 ? 0 : zp >= (ptrdiff_t)nz ? nz - 1 : (size_t)zp;
        s[tx+1][ty+1][STENCIL_BZ+2] = input[szp * (nx * ny) + cy * nx + cx];
    }

    __syncthreads();

    const size_t sgz = gz < 0 ? 0 : gz >= (ptrdiff_t)nz ? nz - 1 : (size_t)gz;
    const size_t outIdx = sgz * (nx * ny) + gy * nx + gx;

    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1 &&
        gz > 0 && gz < (ptrdiff_t)(nz - 1)) {
        // Interior point: compute 7-point averaging stencil from shared memory
        output[outIdx] = (s[tx+1][ty+1][tz+1] + s[tx][ty+1][tz+1] + s[tx+2][ty+1][tz+1] +
                          s[tx+1][ty][tz+1] + s[tx+1][ty+2][tz+1] +
                          s[tx+1][ty+1][tz] + s[tx+1][ty+1][tz+2]) / 7.0;
    } else if (gx < nx && gy < ny && gz >= 0 && gz < (ptrdiff_t)nz) {
        // Boundary point: copy from input (preserves boundary values)
        output[outIdx] = input[outIdx];
    }
}

// Per-GPU worker arguments for std::thread parallelism
struct GPUArgs {
    Real *din, *dout;
    size_t nx, ny, nz;
    ptrdiff_t z_off;
    int dev;
};

void gpuWork(void* p) {
    GPUArgs* a = static_cast<GPUArgs*>(p);
    CUDA_CHECK(cudaSetDevice(a->dev));

    dim3 blk(STENCIL_BX, STENCIL_BY, STENCIL_BZ);
    dim3 grd((a->nx + STENCIL_BX - 1) / STENCIL_BX,
             (a->ny + STENCIL_BY - 1) / STENCIL_BY,
             (a->nz + STENCIL_BZ - 1) / STENCIL_BZ);
    stencilKernel<<<grd, blk>>>(a->din, a->dout, a->nx, a->ny, a->nz, a->z_off);
    CUDA_CHECK(cudaGetLastError());
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Detect and initialize CUDA
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    printf("Using %d GPU(s)\n", numGPUs);

    size_t gridSize = nx * ny * nz;

    // Allocate Unified Memory - accessible from all GPUs with peer access
    Real *d_g1 = nullptr, *d_g2 = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_g1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocManaged(&d_g2, gridSize * sizeof(Real)));

    // Enable peer access between all GPU pairs for coherent multi-GPU access
    for (int i = 0; i < numGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < numGPUs; ++j) {
            if (i != j) {
                int canAccess = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
                if (canAccess) {
                    cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
                        printf("Warning: could not enable peer access GPU %d -> %d\n", i, j);
                    }
                }
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Initialize grid on GPU 0
    printf("Initializing grid...\n");
    {
        dim3 blk(8, 8, 8);
        dim3 grd((nx + blk.x - 1) / blk.x,
                 (ny + blk.y - 1) / blk.y,
                 (nz + blk.z - 1) / blk.z);
        initializeGridKernel<<<grd, blk>>>(d_g1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Partition z-axis across GPUs for parallel scalability
    std::vector<ptrdiff_t> z_off(numGPUs);
    std::vector<size_t> z_lnz(numGPUs);
    {
        size_t chunk = nz / numGPUs;
        size_t rem = nz % numGPUs;
        size_t pos = 0;
        for (int g = 0; g < numGPUs; ++g) {
            z_off[g] = (ptrdiff_t)pos;
            size_t my_chunk = chunk + (g < (int)rem ? 1 : 0);
            z_lnz[g] = my_chunk;
            pos += my_chunk;
        }
    }

    // Prefetch memory to GPU 0 so all GPUs can reach it via peer access
    CUDA_CHECK(cudaMemPrefetchAsync(d_g1, gridSize * sizeof(Real), 0));
    CUDA_CHECK(cudaMemPrefetchAsync(d_g2, gridSize * sizeof(Real), 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prepare per-GPU worker arguments
    std::vector<GPUArgs> args(numGPUs);
    for (int g = 0; g < numGPUs; ++g) {
        args[g].dev = g;
        args[g].nx = nx;
        args[g].ny = ny;
        args[g].nz = z_lnz[g];
        args[g].z_off = z_off[g];
    }

    // Run stencil iterations with multi-GPU parallelism
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *inp = (iter % 2 == 0) ? d_g1 : d_g2;
        Real *out = (iter % 2 == 0) ? d_g2 : d_g1;

        std::vector<std::thread> threads;
        threads.reserve(numGPUs);
        for (int g = 0; g < numGPUs; ++g) {
            args[g].din = inp;
            args[g].dout = out;
            threads.emplace_back(gpuWork, &args[g]);
        }
        for (auto& t : threads) t.join();
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy final result back to host
    std::vector<Real> finalGrid(gridSize);
    const Real* d_final = (iterations % 2 == 0) ? d_g2 : d_g1;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDefault));

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = true;

        // 1. No NaN or Inf values
        for (const auto& val : finalGrid) {
            if (std::isnan(val) || std::isinf(val)) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = false;
                break;
            }
        }

        // 2. Values should be reasonable (bounded)
        if (valid) {
            Real minVal = finalGrid[0];
            Real maxVal = finalGrid[0];
            for (const auto& val : finalGrid) {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 1e6 || minVal < -1e6) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    CUDA_CHECK(cudaFree(d_g1));
    CUDA_CHECK(cudaFree(d_g2));

    return 0;
}
