#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// 3D index calculation
__device__ __host__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for 7-point stencil computation
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz_local) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    
    if (x < nx - 1 && y < ny - 1 && z < nz_local - 1) {
        const size_t idx = idx3(x, y, z, nx, ny);
        
        const Real center = input[idx];
        const Real left = input[idx3(x-1, y, z, nx, ny)];
        const Real right = input[idx3(x+1, y, z, nx, ny)];
        const Real front = input[idx3(x, y-1, z, nx, ny)];
        const Real back = input[idx3(x, y+1, z, nx, ny)];
        const Real bottom = input[idx3(x, y, z-1, nx, ny)];
        const Real top = input[idx3(x, y, z+1, nx, ny)];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// CUDA kernel for boundary copy (global boundaries only)
__global__ void copyBoundaryKernel(const Real* __restrict__ input,
                                   Real* __restrict__ output,
                                   const size_t nx, const size_t ny, const size_t nz_local,
                                   const int rank, const int nprocs) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1; // Start from z=1 (owned planes)
    
    if (x < nx && y < ny && z <= nz_local) {
        // Copy X and Y boundaries for all owned Z planes
        bool isBoundary = (x == 0 || x == nx-1 || y == 0 || y == ny-1);
        // Copy Z boundaries only for global boundaries
        if (rank == 0 && z == 1) isBoundary = true; // Global bottom
        if (rank == nprocs-1 && z == nz_local) isBoundary = true; // Global top
        
        if (isBoundary) {
            const size_t idx = idx3(x, y, z, nx, ny);
            output[idx] = input[idx];
        }
    }
}

// OpenMP parallelized initialization
void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz_local, const size_t z_offset) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z + z_offset;
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_idx = idx3(x, y, global_z, nx, ny);
                grid[idx] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Hybrid stencil iteration: MPI halo exchange + CUDA kernel + OpenMP for CPU work
void stencilIterationHybrid(Real* d_input, Real* d_output,
                            Real* h_input, Real* h_output,
                            const size_t nx, const size_t ny, const size_t nz_local,
                            const size_t z_start, const size_t global_nz,
                            int rank, int nprocs,
                            dim3 blockDim, dim3 gridDimInterior, dim3 gridDimBoundary) {
    const size_t planeSize = nx * ny;

    // --- MPI halo exchange ---
    // Exchange bottom halo: send z_local=1 plane down, receive into z_local=0
    // Exchange top halo: send z_local=nz_local plane up, receive into z_local=nz_local+1
    MPI_Status status;

    // Exchange bottom halo
    if (nprocs > 1) {
        int below = rank - 1;
        int above = rank + 1;

        if (rank % 2 == 0) {
            // Even ranks send first, then receive
            if (below >= 0) {
                // Send bottom owned plane (z_local=1) to rank below's top halo
                MPI_Send(h_input + 1 * planeSize, planeSize, MPI_DOUBLE, below, 0, MPI_COMM_WORLD);
            }
            if (above < nprocs) {
                // Receive into top halo (z_local=nz_local+1) from rank above's bottom owned
                MPI_Recv(h_input + (nz_local + 1) * planeSize, planeSize, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &status);
            }
        } else {
            // Odd ranks receive first, then send
            if (above < nprocs) {
                MPI_Recv(h_input + (nz_local + 1) * planeSize, planeSize, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &status);
            }
            if (below >= 0) {
                MPI_Send(h_input + 1 * planeSize, planeSize, MPI_DOUBLE, below, 0, MPI_COMM_WORLD);
            }
        }

        if (rank % 2 == 0) {
            if (above < nprocs) {
                // Send top owned plane (z_local=nz_local) to rank above's bottom halo
                MPI_Send(h_input + nz_local * planeSize, planeSize, MPI_DOUBLE, above, 1, MPI_COMM_WORLD);
            }
            if (below >= 0) {
                // Receive into bottom halo (z_local=0) from rank below's top owned
                MPI_Recv(h_input + 0 * planeSize, planeSize, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &status);
            }
        } else {
            if (below >= 0) {
                MPI_Recv(h_input + 0 * planeSize, planeSize, MPI_DOUBLE, below, 1, MPI_COMM_WORLD, &status);
            }
            if (above < nprocs) {
                MPI_Send(h_input + nz_local * planeSize, planeSize, MPI_DOUBLE, above, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        // Single process: set halo boundaries to 0 (or copy from owned)
        // For global boundary, the stencil won't be applied there anyway
        // Just zero out halos
        memset(h_input + 0 * planeSize, 0, planeSize * sizeof(Real));
        memset(h_input + (nz_local + 1) * planeSize, 0, planeSize * sizeof(Real));
    }

    // Upload halo-updated input to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input, (nz_local + 2) * planeSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Launch stencil kernel for interior points
    stencilKernel<<<gridDimInterior, blockDim>>>(d_input, d_output, nx, ny, nz_local + 2);

    // Launch boundary copy kernel
    copyBoundaryKernel<<<gridDimBoundary, blockDim>>>(d_input, d_output, nx, ny, nz_local, rank, nprocs);

    // Download result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, (nz_local + 2) * planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize CUDA (one GPU per node, or let each rank pick)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Finalize();
        return 1;
    }
    // Assign GPU: round-robin by local rank (simple approach: rank % numDevices)
    int local_rank = rank % numDevices;
    CUDA_CHECK(cudaSetDevice(local_rank));

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 parses, then broadcast)
    if (rank == 0) {
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
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parameters
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", nprocs);
        int numThreads = 1;
        #pragma omp parallel
        #pragma omp single
        numThreads = omp_get_num_threads();
        printf("OpenMP threads per rank: %d\n", numThreads);
        printf("CUDA devices: %d\n", numDevices);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 1D decomposition along Z axis
    // Each rank owns a contiguous slab of Z planes
    size_t z_start = 0; // global z index of first owned plane
    size_t nz_local = 0; // number of owned Z planes

    // Compute decomposition
    {
        size_t base = nz / nprocs;
        size_t remainder = nz % nprocs;
        if ((size_t)rank < remainder) {
            nz_local = base + 1;
            z_start = (size_t)rank * (base + 1);
        } else {
            nz_local = base;
            z_start = remainder * (base + 1) + ((size_t)rank - remainder) * base;
        }
    }

    if (rank == 0) {
        printf("Z decomposition: %zu planes per rank (approx), nz=%zu, nprocs=%d\n", nz / nprocs, nz, nprocs);
    }

    // Local grid dimensions: nx * ny * (nz_local + 2) for halo planes
    // z_local=0 is bottom halo, z_local=1..nz_local are owned, z_local=nz_local+1 is top halo
    const size_t local_nz = nz_local + 2;
    const size_t planeSize = nx * ny;
    const size_t localGridSize = planeSize * local_nz;

    // Allocate host memory
    std::vector<Real> h_grid1(localGridSize, 0.0);
    std::vector<Real> h_grid2(localGridSize, 0.0);

    // Initialize grid with OpenMP parallelism
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(h_grid1, nx, ny, nz_local, z_start);

    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, localGridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, localGridSize * sizeof(Real)));

    // Upload initial grid to GPU
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid1.data(), localGridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, h_grid2.data(), localGridSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Configure CUDA kernel dimensions
    dim3 block(16, 16, 1);
    dim3 gridInterior((unsigned)(nx + block.x - 1) / block.x,
                      (unsigned)(ny + block.y - 1) / block.y,
                      (unsigned)(nz_local + block.z - 1) / block.z);
    dim3 gridBoundary((unsigned)(nx + block.x - 1) / block.x,
                      (unsigned)(ny + block.y - 1) / block.y,
                      (unsigned)(local_nz + block.z - 1) / block.z);

    // Run stencil iterations
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationHybrid(d_grid1, d_grid2,
                                   h_grid1.data(), h_grid2.data(),
                                   nx, ny, nz_local, z_start, nz,
                                   rank, nprocs,
                                   block, gridInterior, gridBoundary);
        } else {
            stencilIterationHybrid(d_grid2, d_grid1,
                                   h_grid2.data(), h_grid1.data(),
                                   nx, ny, nz_local, z_start, nz,
                                   rank, nprocs,
                                   block, gridInterior, gridBoundary);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather timing (report max across all ranks)
    long long local_ms = duration.count();
    long long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/output
    // Each rank sends its owned planes (z_local=1..nz_local) to rank 0
    const size_t ownedSize = planeSize * nz_local;
    std::vector<Real> finalGrid;
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
    }

    // Determine which buffer has the final result
    // After iterations: if iterations is even, result is in grid1 (d_grid1/h_grid1); if odd, in grid2
    // But we need to download from GPU first
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    Real* h_final_ptr = (iterations % 2 == 0) ? h_grid1.data() : h_grid2.data();
    CUDA_CHECK(cudaMemcpy(h_final_ptr, d_final, localGridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gather owned data (z_local=1..nz_local) to rank 0
    // Use MPI_Gatherv since each rank may have different nz_local
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    {
        int dummy;
        size_t base = nz / nprocs;
        size_t remainder = nz % nprocs;
        for (int r = 0; r < nprocs; ++r) {
            size_t nz_r = (size_t)r < remainder ? base + 1 : base;
            recvcounts[r] = (int)(planeSize * nz_r);
        }
        displs[0] = 0;
        for (int r = 1; r < nprocs; ++r) {
            displs[r] = displs[r-1] + recvcounts[r-1];
        }
    }

    // Send owned portion (starting at z_local=1)
    MPI_Gatherv(h_final_ptr + planeSize, (int)ownedSize, MPI_DOUBLE,
                finalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results and validate on rank 0
    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    // Broadcast validation result
    int valid_int = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        valid_int = valid ? 0 : 1;
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    if (validate) {
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();
    return valid_int;
}
