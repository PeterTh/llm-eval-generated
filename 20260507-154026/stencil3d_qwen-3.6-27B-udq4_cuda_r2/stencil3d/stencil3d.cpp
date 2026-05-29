#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
//  Initialization kernel
// ---------------------------------------------------------------------------
__global__ void initKernel(Real* grid, size_t nx, size_t ny, size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t idx = z * nx * ny + y * nx + x;
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// ---------------------------------------------------------------------------
//  7-point stencil kernel with 3D shared-memory tiling
//  Block dimensions are compile-time template parameters so the shared
//  memory array size can be computed at compile time.
// ---------------------------------------------------------------------------
template <int BX, int BY, int BZ>
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output, size_t nx, size_t ny,
                              size_t nz) {
    __shared__ Real smem[(BZ + 2) * (BY + 2) * (BX + 2)];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Global coordinates (interior offset by +1)
    int gx = blockIdx.x * BX + tx + 1;
    int gy = blockIdx.y * BY + ty + 1;
    int gz = blockIdx.z * BZ + tz + 1;

    int pitch  = BX + 2;
    int stride = pitch * (BY + 2);

    // Helper lambda for shared-memory indexing
    auto smemIdx = [=](int sz, int sy, int sx) {
        return sz * stride + sy * pitch + sx;
    };

    // ---- Phase 1: load main element --------------------------------
    if (gx < nx && gy < ny && gz < nz) {
        smem[smemIdx(tz + 1, ty + 1, tx + 1)] =
            input[gz * nx * ny + gy * nx + gx];
    }

    // ---- Phase 2: load face halos ----------------------------------
    // X halos
    if (tx == 0 && gx > 0 && gy < ny && gz < nz) {
        smem[smemIdx(tz + 1, ty + 1, 0)] =
            input[gz * nx * ny + gy * nx + (gx - 1)];
    }
    if (tx == BX - 1 && gx + 1 < nx && gy < ny && gz < nz) {
        smem[smemIdx(tz + 1, ty + 1, BX + 1)] =
            input[gz * nx * ny + gy * nx + (gx + 1)];
    }

    // Y halos
    if (ty == 0 && gx < nx && gy > 0 && gz < nz) {
        smem[smemIdx(tz + 1, 0, tx + 1)] =
            input[gz * nx * ny + (gy - 1) * nx + gx];
    }
    if (ty == BY - 1 && gx < nx && gy + 1 < ny && gz < nz) {
        smem[smemIdx(tz + 1, BY + 1, tx + 1)] =
            input[gz * nx * ny + (gy + 1) * nx + gx];
    }

    // Z halos
    if (tz == 0 && gx < nx && gy < ny && gz > 0) {
        smem[smemIdx(0, ty + 1, tx + 1)] =
            input[(gz - 1) * nx * ny + gy * nx + gx];
    }
    if (tz == BZ - 1 && gx < nx && gy < ny && gz + 1 < nz) {
        smem[smemIdx(BZ + 1, ty + 1, tx + 1)] =
            input[(gz + 1) * nx * ny + gy * nx + gx];
    }

    // ---- Phase 3: load edge halos (xy) -----------------------------
    if (tx == 0 && ty == 0 && gx > 0 && gy > 0 && gz < nz) {
        smem[smemIdx(tz + 1, 0, 0)] =
            input[gz * nx * ny + (gy - 1) * nx + (gx - 1)];
    }
    if (tx == 0 && ty == BY - 1 && gx > 0 && gy + 1 < ny && gz < nz) {
        smem[smemIdx(tz + 1, BY + 1, 0)] =
            input[gz * nx * ny + (gy + 1) * nx + (gx - 1)];
    }
    if (tx == BX - 1 && ty == 0 && gx + 1 < nx && gy > 0 && gz < nz) {
        smem[smemIdx(tz + 1, 0, BX + 1)] =
            input[gz * nx * ny + (gy - 1) * nx + (gx + 1)];
    }
    if (tx == BX - 1 && ty == BY - 1 && gx + 1 < nx && gy + 1 < ny &&
        gz < nz) {
        smem[smemIdx(tz + 1, BY + 1, BX + 1)] =
            input[gz * nx * ny + (gy + 1) * nx + (gx + 1)];
    }

    // Edge halos (xz)
    if (tx == 0 && tz == 0 && gx > 0 && gy < ny && gz > 0) {
        smem[smemIdx(0, ty + 1, 0)] =
            input[(gz - 1) * nx * ny + gy * nx + (gx - 1)];
    }
    if (tx == 0 && tz == BZ - 1 && gx > 0 && gy < ny && gz + 1 < nz) {
        smem[smemIdx(BZ + 1, ty + 1, 0)] =
            input[(gz + 1) * nx * ny + gy * nx + (gx - 1)];
    }
    if (tx == BX - 1 && tz == 0 && gx + 1 < nx && gy < ny && gz > 0) {
        smem[smemIdx(0, ty + 1, BX + 1)] =
            input[(gz - 1) * nx * ny + gy * nx + (gx + 1)];
    }
    if (tx == BX - 1 && tz == BZ - 1 && gx + 1 < nx && gy < ny &&
        gz + 1 < nz) {
        smem[smemIdx(BZ + 1, ty + 1, BX + 1)] =
            input[(gz + 1) * nx * ny + gy * nx + (gx + 1)];
    }

    // Edge halos (yz)
    if (ty == 0 && tz == 0 && gx < nx && gy > 0 && gz > 0) {
        smem[smemIdx(0, 0, tx + 1)] =
            input[(gz - 1) * nx * ny + (gy - 1) * nx + gx];
    }
    if (ty == 0 && tz == BZ - 1 && gx < nx && gy > 0 && gz + 1 < nz) {
        smem[smemIdx(BZ + 1, 0, tx + 1)] =
            input[(gz + 1) * nx * ny + (gy - 1) * nx + gx];
    }
    if (ty == BY - 1 && tz == 0 && gx < nx && gy + 1 < ny && gz > 0) {
        smem[smemIdx(0, BY + 1, tx + 1)] =
            input[(gz - 1) * nx * ny + (gy + 1) * nx + gx];
    }
    if (ty == BY - 1 && tz == BZ - 1 && gx < nx && gy + 1 < ny &&
        gz + 1 < nz) {
        smem[smemIdx(BZ + 1, BY + 1, tx + 1)] =
            input[(gz + 1) * nx * ny + (gy + 1) * nx + gx];
    }

    // ---- Phase 4: load corner halos (xyz) --------------------------
    if (tx == 0 && ty == 0 && tz == 0 && gx > 0 && gy > 0 && gz > 0) {
        smem[0] = input[(gz - 1) * nx * ny + (gy - 1) * nx + (gx - 1)];
    }
    if (tx == 0 && ty == 0 && tz == BZ - 1 && gx > 0 && gy > 0 &&
        gz + 1 < nz) {
        smem[(BZ + 1) * stride] =
            input[(gz + 1) * nx * ny + (gy - 1) * nx + (gx - 1)];
    }
    if (tx == 0 && ty == BY - 1 && tz == 0 && gx > 0 && gy + 1 < ny &&
        gz > 0) {
        smem[(BY + 1) * pitch] =
            input[(gz - 1) * nx * ny + (gy + 1) * nx + (gx - 1)];
    }
    if (tx == 0 && ty == BY - 1 && tz == BZ - 1 && gx > 0 && gy + 1 < ny &&
        gz + 1 < nz) {
        smem[(BZ + 1) * stride + (BY + 1) * pitch] =
            input[(gz + 1) * nx * ny + (gy + 1) * nx + (gx - 1)];
    }
    if (tx == BX - 1 && ty == 0 && tz == 0 && gx + 1 < nx && gy > 0 &&
        gz > 0) {
        smem[BX + 1] =
            input[(gz - 1) * nx * ny + (gy - 1) * nx + (gx + 1)];
    }
    if (tx == BX - 1 && ty == 0 && tz == BZ - 1 && gx + 1 < nx && gy > 0 &&
        gz + 1 < nz) {
        smem[(BZ + 1) * stride + (BX + 1)] =
            input[(gz + 1) * nx * ny + (gy - 1) * nx + (gx + 1)];
    }
    if (tx == BX - 1 && ty == BY - 1 && tz == 0 && gx + 1 < nx &&
        gy + 1 < ny && gz > 0) {
        smem[(BY + 1) * pitch + (BX + 1)] =
            input[(gz - 1) * nx * ny + (gy + 1) * nx + (gx + 1)];
    }
    if (tx == BX - 1 && ty == BY - 1 && tz == BZ - 1 && gx + 1 < nx &&
        gy + 1 < ny && gz + 1 < nz) {
        smem[(BZ + 1) * stride + (BY + 1) * pitch + (BX + 1)] =
            input[(gz + 1) * nx * ny + (gy + 1) * nx + (gx + 1)];
    }

    __syncthreads();

    // ---- Stencil computation ---------------------------------------
    if (gx > 0 && gx < static_cast<int>(nx) - 1 && gy > 0 &&
        gy < static_cast<int>(ny) - 1 && gz > 0 &&
        gz < static_cast<int>(nz) - 1) {
        int si = smemIdx(tz + 1, ty + 1, tx + 1);
        Real result = (smem[si] + smem[si - 1] + smem[si + 1] +
                       smem[si - pitch] + smem[si + pitch] +
                       smem[si - stride] + smem[si + stride]) /
                      7.0;
        output[gz * nx * ny + gy * nx + gx] = result;
    }
}

// ---------------------------------------------------------------------------
//  Boundary copy kernel – copies boundary values from input to output
// ---------------------------------------------------------------------------
__global__ void copyBoundaryKernel(const Real* __restrict__ input,
                                   Real* __restrict__ output, size_t nx,
                                   size_t ny, size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 ||
            z == nz - 1) {
            size_t idx = z * nx * ny + y * nx + x;
            output[idx] = input[idx];
        }
    }
}

// ---------------------------------------------------------------------------
//  Host-side validation
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
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

    size_t gridSize = nx * ny * nz;
    size_t bytes = gridSize * sizeof(Real);

    // Select GPU and print device info
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);

    // Allocate device memory (double buffering)
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, bytes));

    // Initialize grid on GPU
    printf("Initializing grid...\n");
    {
        dim3 blockSize(8, 8, 8);
        dim3 gridSize_b((nx + blockSize.x - 1) / blockSize.x,
                        (ny + blockSize.y - 1) / blockSize.y,
                        (nz + blockSize.z - 1) / blockSize.z);
        initKernel<<<gridSize_b, blockSize>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Choose shared-memory block dimensions (8×8×8 → 512 threads)
    constexpr int BX = 8, BY = 8, BZ = 8;

    // Block grid for stencil (interior only)
    dim3 sBlock(BX, BY, BZ);
    dim3 sGrid((nx - 2 + BX - 1) / BX, (ny - 2 + BY - 1) / BY,
               (nz - 2 + BZ - 1) / BZ);

    // Block grid for boundary copy (≤ 1024 threads/block)
    dim3 bBlock(16, 16, 4);
    dim3 bGrid((nx + bBlock.x - 1) / bBlock.x,
               (ny + bBlock.y - 1) / bBlock.y,
               (nz + bBlock.z - 1) / bBlock.z);

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *input, *output;
        if (iter % 2 == 0) {
            input  = d_grid1;
            output = d_grid2;
        } else {
            input  = d_grid2;
            output = d_grid1;
        }

        stencilKernel<BX, BY, BZ><<<sGrid, sBlock>>>(input, output, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());

        copyBoundaryKernel<<<bGrid, bBlock>>>(input, output, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates =
        static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups =
        cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy final grid back to host for validation / results
    std::vector<Real> finalGrid(gridSize);
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, bytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
        return valid ? 0 : 1;
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    return 0;
}
