#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void stencilKernel(const Real* in, Real* out, size_t nx, size_t ny,
                              size_t localNz, size_t globalZ, size_t zStart) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z + 1;
    if (x >= nx || y >= ny || lz > localNz) return;
    const size_t plane = nx * ny, q = lz * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || zStart + lz - 1 == 0 || zStart + lz - 1 + 1 == globalZ) {
        out[q] = in[q];
    } else {
        out[q] = (in[q] + in[q-1] + in[q+1] + in[q-nx] + in[q+nx] + in[q-plane] + in[q+plane]) / 7.0;
    }
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

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
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    if (rank == 0) {
      printf("3D Stencil Benchmark\n");
      printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
      printf("Iterations: %d\n", iterations);
      printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    size_t gridSize = nx * ny * nz;
    
    if (nz < 3 || nx < 3 || ny < 3 || iterations < 0 || static_cast<size_t>(ranks) > nz) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be >= 3, iterations nonnegative, and MPI ranks <= Z planes\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < rem);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny, localSize = (localNz + 2) * plane;
    std::vector<Real> grid1(localSize), grid2(localSize);
    
    // Initialize
    if (rank == 0) printf("Initializing grid...\n");
    #pragma omp parallel for
    for (long long q = 0; q < static_cast<long long>(localSize); ++q) {
        const size_t lz = static_cast<size_t>(q) / plane;
        const size_t global = (zStart + lz) * plane + static_cast<size_t>(q) % plane;
        grid1[static_cast<size_t>(q)] = (global % 19) * 1.0;
    }
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    Real *d1, *d2;
    CUDA_CHECK(cudaMalloc(&d1, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, localSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d1, grid1.data(), localSize*sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d2, grid2.data(), localSize*sizeof(Real), cudaMemcpyHostToDevice));
    
    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        Real *src = (iter % 2 == 0) ? d1 : d2;
        Real *dst = (iter % 2 == 0) ? d2 : d1;
        std::vector<Real>& host = (iter % 2 == 0) ? grid1 : grid2;
        CUDA_CHECK(cudaMemcpy(host.data() + plane, src + plane, plane*sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.data() + localNz*plane, src + localNz*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost));
        const int prev = rank ? rank - 1 : MPI_PROC_NULL, next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        MPI_Sendrecv(host.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                     host.data() + (localNz + 1)*plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(host.data() + localNz*plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                     host.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(src, host.data(), localSize*sizeof(Real), cudaMemcpyHostToDevice));
        dim3 block(32, 4, 1), grid((nx+31)/32, (ny+3)/4, static_cast<unsigned>(localNz));
        stencilKernel<<<grid, block>>>(src, dst, nx, ny, localNz, nz, zStart);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;
    Real *finalDevice = (iterations % 2 == 0) ? d1 : d2;
    CUDA_CHECK(cudaMemcpy(localFinal.data(), finalDevice, localSize*sizeof(Real), cudaMemcpyDeviceToHost));
    std::vector<Real> finalGrid(gridSize);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0; r<ranks; ++r) { size_t n=nz/ranks+(static_cast<size_t>(r)<nz%ranks); size_t s=static_cast<size_t>(r)*(nz/ranks)+std::min(static_cast<size_t>(r),nz%ranks); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>(s*plane); }
    MPI_Gatherv(localFinal.data()+plane, static_cast<int>(localNz*plane), MPI_DOUBLE, finalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0 && printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            CUDA_CHECK(cudaFree(d1)); CUDA_CHECK(cudaFree(d2)); MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d1)); CUDA_CHECK(cudaFree(d2)); MPI_Finalize(); return 1;
        }
    }
    
    CUDA_CHECK(cudaFree(d1)); CUDA_CHECK(cudaFree(d2)); MPI_Finalize(); return 0;
}
