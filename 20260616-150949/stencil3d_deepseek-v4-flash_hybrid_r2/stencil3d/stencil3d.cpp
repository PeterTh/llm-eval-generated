#include <algorithm>
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

// 3D index calculation - host and device
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y,
                                                  const size_t z, const size_t nx,
                                                  const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA stencil kernel for 7-point stencil
// Computes interior points only. Boundary copy is handled by
// cudaMemcpyDeviceToDevice (full grid copy) before kernel launch.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const int nx, const int ny, const int nz_local,
                               const int z_start, const int z_end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int z = blockIdx.z + z_start;

    if (x < nx - 1 && y < ny - 1 && z < z_end) {
        const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        const size_t row = static_cast<size_t>(nx);
        const size_t idx = static_cast<size_t>(z) * plane +
                           static_cast<size_t>(y) * row +
                           static_cast<size_t>(x);

        output[idx] = (input[idx] +
                       input[idx - 1] +        // left (x-1)
                       input[idx + 1] +        // right (x+1)
                       input[idx - row] +      // front (y-1)
                       input[idx + row] +      // back (y+1)
                       input[idx - plane] +    // bottom (z-1)
                       input[idx + plane]) *   // top (z+1)
                       0.14285714285714285;    // 1/7
    }
}

// Initialize local subdomain grid with OpenMP parallelism
void initializeGrid(Real* grid, const size_t nx, const size_t ny,
                    const size_t nz_local, const size_t z_offset,
                    const size_t nz_global) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Compute global Z coordinate for deterministic initialization
                int global_z = static_cast<int>(z_offset) + static_cast<int>(z) - 1;
                // Clamp ghost cells at global boundaries to the boundary value
                if (global_z < 0) global_z = 0;
                if (global_z >= static_cast<int>(nz_global))
                    global_z = static_cast<int>(nz_global) - 1;
                const size_t global_idx = idx3(x, y, static_cast<size_t>(global_z),
                                               nx, ny);
                grid[idx3(x, y, z, nx, ny)] = (global_idx % 19) * 1.0;
            }
        }
    }
}

// Validate result (operates on full grid assembled on rank 0)
bool validateResult(const std::vector<Real>& grid,
                    const size_t /*nx*/, const size_t /*ny*/,
                    const size_t /*nz*/) {
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Determine which GPU to use (round-robin across MPI ranks)
    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    if (ngpus == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpu_id = rank % ngpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // Default parameters
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 only, then broadcast)
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parsed parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Grid decomposition along Z dimension
    size_t nz_base = nz / static_cast<size_t>(nprocs);
    size_t nz_extra = nz % static_cast<size_t>(nprocs);
    size_t nz_local_owned = nz_base +
        (static_cast<size_t>(rank) < nz_extra ? 1 : 0);

    // Compute Z offset (global Z index of first owned layer)
    size_t z_offset = static_cast<size_t>(rank) * nz_base +
                      std::min(static_cast<size_t>(rank), nz_extra);

    // Local grid includes ghost cells: one below, one above
    size_t nz_local = nz_local_owned + 2;

    // Validate grid is large enough
    if (nz_local_owned == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: Grid too small for %d ranks (nz=%zu)\n",
                    nprocs, nz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t plane_size = nx * ny;
    size_t plane_bytes = plane_size * sizeof(Real);
    size_t local_grid_bytes = plane_size * nz_local * sizeof(Real);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", nprocs);
        printf("GPU devices available: %d\n", ngpus);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate pinned host memory for faster GPU<->host transfers
    Real *grid1_host, *grid2_host;
    CUDA_CHECK(cudaMallocHost(&grid1_host, local_grid_bytes));
    CUDA_CHECK(cudaMallocHost(&grid2_host, local_grid_bytes));

    // Allocate device memory
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_grid_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_grid_bytes));

    // Initialize local grid with ghost cells
    if (rank == 0) printf("Initializing grid...\n");
    initializeGrid(grid1_host, nx, ny, nz_local, z_offset, nz);

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1_host, local_grid_bytes,
                          cudaMemcpyHostToDevice));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // MPI neighbor ranks
    int up_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int down_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // MPI tag conventions
    const int TAG_UP = 0;    // Data flowing to lower rank (decreasing Z)
    const int TAG_DOWN = 1;  // Data flowing to higher rank (increasing Z)

    // Determine interior Z range for CUDA kernel.
    // Skip physical boundaries: global z=0 (local z=1 for rank 0) and
    // global z=nz-1 (local z=nz_local_owned for last rank).
    int z_start, z_end;
    if (nprocs == 1) {
        // Single rank: both global z=0 and z=nz-1 are boundaries
        z_start = 2;                                // skip owned z=1 (global z=0)
        z_end = static_cast<int>(nz_local_owned);   // skip last owned (global z=nz-1)
    } else if (rank == 0) {
        z_start = 2;                                // skip owned z=1 (global z=0)
        z_end = static_cast<int>(nz_local_owned) + 1; // through last owned
    } else if (rank == nprocs - 1) {
        z_start = 1;                                // first owned
        z_end = static_cast<int>(nz_local_owned);     // skip last (global z=nz-1)
    } else {
        z_start = 1;                                // first owned
        z_end = static_cast<int>(nz_local_owned) + 1; // all owned layers
    }
    int nz_interior = z_end - z_start;

    // CUDA launch configuration (at least 1 block per dimension for valid launch)
    dim3 blockDim(32, 8, 1);
    dim3 gridDim(
        std::max(1u, (static_cast<unsigned int>(nx) - 2u + blockDim.x - 1u) / blockDim.x),
        std::max(1u, (static_cast<unsigned int>(ny) - 2u + blockDim.y - 1u) / blockDim.y),
        std::max(1u, nz_interior > 0 ? static_cast<unsigned int>(nz_interior) : 1u)
    );

    // Host buffers for MPI ghost exchange
    std::vector<Real> send_up(plane_size);
    std::vector<Real> send_down(plane_size);
    std::vector<Real> recv_up(plane_size);
    std::vector<Real> recv_down(plane_size);

    // Synchronize all ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Main stencil iteration loop
    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in;
        Real* d_out;
        if (iter % 2 == 0) {
            d_in = d_grid1;
            d_out = d_grid2;
        } else {
            d_in = d_grid2;
            d_out = d_grid1;
        }

        // --- Step 1: Ghost cell exchange via MPI ---
        // Copy owned boundary planes from GPU to host
        if (up_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_up.data(),
                         d_in + plane_size,
                         plane_bytes, cudaMemcpyDeviceToHost, stream));
        }
        if (down_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(send_down.data(),
                         d_in + plane_size * nz_local_owned,
                         plane_bytes, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Exchange with up neighbor: send first owned plane, receive into ghost below
        if (up_rank != MPI_PROC_NULL) {
            MPI_Sendrecv(send_up.data(), plane_size, MPI_DOUBLE, up_rank, TAG_UP,
                         recv_up.data(), plane_size, MPI_DOUBLE, up_rank, TAG_DOWN,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Exchange with down neighbor: send last owned plane, receive into ghost above
        if (down_rank != MPI_PROC_NULL) {
            MPI_Sendrecv(send_down.data(), plane_size, MPI_DOUBLE, down_rank, TAG_DOWN,
                         recv_down.data(), plane_size, MPI_DOUBLE, down_rank, TAG_UP,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Copy received ghost planes from host to GPU
        if (up_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in,
                         recv_up.data(),
                         plane_bytes, cudaMemcpyHostToDevice, stream));
        }
        if (down_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(
                         d_in + plane_size * (nz_local_owned + 1),
                         recv_down.data(),
                         plane_bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Step 2: Copy d_in to d_out (preserves boundary & ghost values) ---
        CUDA_CHECK(cudaMemcpyAsync(d_out, d_in, local_grid_bytes,
                                   cudaMemcpyDeviceToDevice, stream));

        // --- Step 3: Launch CUDA stencil kernel for interior points ---
        if (nz_interior > 0) {
            stencilKernel<<<gridDim, blockDim, 0, stream>>>(
                d_in, d_out, nx, ny, nz_local, z_start, z_end);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Wait for GPU to finish before stopping the local timer
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", elapsed);

        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Determine final grid buffer
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    Real* h_final = (iterations % 2 == 0) ? grid1_host : grid2_host;

    // Copy final result from GPU to host
    CUDA_CHECK(cudaMemcpy(h_final, d_final, local_grid_bytes,
                          cudaMemcpyDeviceToHost));

    // Gather full grid on rank 0 for validation / output
    if (validate || printResults) {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        size_t offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t r_nz = nz_base + (static_cast<size_t>(r) < nz_extra ? 1 : 0);
            int count = static_cast<int>(plane_size * r_nz);
            recvcounts[r] = count;
            displs[r] = static_cast<int>(offset);
            offset += count;
        }

        std::vector<Real> full_grid;
        if (rank == 0)
            full_grid.resize(nx * ny * nz);

        // Send owned data (skip ghost below at local z=0)
        int send_count = static_cast<int>(plane_size * nz_local_owned);
        MPI_Gatherv(h_final + plane_size, send_count, MPI_DOUBLE,
                    rank == 0 ? full_grid.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults)
                print_results(full_grid, "Grid");

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full_grid, nx, ny, nz);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(grid1_host));
    CUDA_CHECK(cudaFreeHost(grid2_host));

    MPI_Finalize();
    return 0;
}
