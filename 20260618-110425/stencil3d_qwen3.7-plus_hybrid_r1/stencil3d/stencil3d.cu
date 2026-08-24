#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdlib>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

static_assert(std::is_same_v<Real, double>, "Real must be double for MPI_DOUBLE");

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char mpi_errstr[MPI_MAX_ERROR_STRING]; \
        int mpi_errlen; \
        MPI_Error_string(mpi_err, mpi_errstr, &mpi_errlen); \
        fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, mpi_errstr); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Combined 7-point stencil + boundary copy kernel
// Each thread handles one point in the local domain (local_z=1..nz_local, y=0..ny-1, x=0..nx-1)
// Boundary points are copied; interior points get the stencil computation
__global__ void __launch_bounds__(256)
stencil_kernel(const Real* __restrict__ input,
               Real* __restrict__ output,
               const size_t nx, const size_t ny,
               const size_t nz_local,
               const int global_z_start,
               const size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1; // +1 for bottom halo offset

    if (x >= nx || y >= ny || lz > nz_local) return;

    const size_t plane = nx * ny;
    const size_t idx = lz * plane + y * nx + x;
    const size_t global_z = (size_t)(global_z_start) + (lz - 1);

    const bool is_boundary = (x == 0 || x == nx - 1 ||
                               y == 0 || y == ny - 1 ||
                               global_z == 0 || global_z == global_nz - 1);

    if (is_boundary) {
        output[idx] = input[idx];
    } else {
        const Real center = input[idx];
        const Real left   = input[idx - 1];
        const Real right  = input[idx + 1];
        const Real front  = input[idx - nx];
        const Real back   = input[idx + nx];
        const Real bottom = input[idx - plane];
        const Real top    = input[idx + plane];
        output[idx] = (center + left + right + front + back + bottom + top) / (Real)7.0;
    }
}

void initializeGrid(Real* grid, size_t nx, size_t ny, size_t nz_local,
                    int global_z_start, size_t plane_size) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 1; lz <= nz_local; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            size_t gz = (size_t)(global_z_start) + (lz - 1);
            size_t base_local = lz * plane_size + y * nx;
            size_t base_global = gz * plane_size + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                grid[base_local + x] = (Real)(((base_global + x) % 19) * 1.0);
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid,
                    [[maybe_unused]] size_t nx,
                    [[maybe_unused]] size_t ny,
                    [[maybe_unused]] size_t nz) {
    size_t n = grid.size();
    if (n == 0) return true;

    bool has_nan_inf = false;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    #pragma omp parallel reduction(||:has_nan_inf) reduction(min:minVal) reduction(max:maxVal)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            Real val = grid[i];
            if (std::isnan(val) || std::isinf(val)) has_nan_inf = true;
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }
    }

    if (has_nan_inf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    // MPI initialization
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, nprocs;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    // GPU selection based on local rank (for multi-GPU nodes)
    int local_rank = 0;
    {
        char* env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (env) local_rank = atoi(env);
        env = std::getenv("SLURM_LOCALID");
        if (env) local_rank = atoi(env);
    }
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(local_rank % num_gpus));
    }

    // Parse command line arguments
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = (size_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d, GPUs per node: %d\n", nprocs, num_gpus);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
    }

    // 1D domain decomposition along Z axis
    size_t base_nz = nz / (size_t)nprocs;
    size_t extra = nz % (size_t)nprocs;
    size_t nz_local = base_nz + ((size_t)rank < extra ? 1 : 0);
    int global_z_start = (int)((size_t)rank * base_nz + std::min((size_t)rank, extra));

    // Determine neighbor ranks for halo exchange
    int rank_below, rank_above;
    if (nz_local == 0) {
        rank_below = MPI_PROC_NULL;
        rank_above = MPI_PROC_NULL;
    } else {
        rank_below = (global_z_start > 0) ? rank - 1 : MPI_PROC_NULL;
        rank_above = (global_z_start + (int)nz_local < (int)nz) ? rank + 1 : MPI_PROC_NULL;
    }

    const size_t plane_size = nx * ny;
    const size_t local_array_size = plane_size * (nz_local + 2); // +2 for halo layers

    // Allocate CPU buffer and initialize with OpenMP
    std::vector<Real> localGrid(local_array_size, 0.0);
    if (nz_local > 0) {
        initializeGrid(localGrid.data(), nx, ny, nz_local, global_z_start, plane_size);
    }

    // Allocate GPU double buffers
    Real *d_buf1 = nullptr, *d_buf2 = nullptr;
    if (nz_local > 0) {
        CUDA_CHECK(cudaMalloc(&d_buf1, local_array_size * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_buf2, local_array_size * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_buf1, localGrid.data(), local_array_size * sizeof(Real),
                              cudaMemcpyHostToDevice));
    }

    // Free CPU buffer (no longer needed until gather)
    localGrid.clear();
    localGrid.shrink_to_fit();

    // Halo exchange host buffers
    std::vector<Real> send_bottom(plane_size);
    std::vector<Real> recv_bottom(plane_size);
    std::vector<Real> send_top(plane_size);
    std::vector<Real> recv_top(plane_size);

    // CUDA kernel launch configuration
    // Block: (32, 4, 2) = 256 threads for good occupancy and coalesced access
    dim3 block(32, 4, 2);
    dim3 grid_dim(1, 1, 1);
    if (nz_local > 0) {
        grid_dim = dim3((unsigned int)((nx + block.x - 1) / block.x),
                        (unsigned int)((ny + block.y - 1) / block.y),
                        (unsigned int)((nz_local + block.z - 1) / block.z));
    }

    // Synchronize all ranks before timing
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start_time = std::chrono::high_resolution_clock::now();

    Real *d_input = d_buf1;
    Real *d_output = d_buf2;

    for (int iter = 0; iter < iterations; ++iter) {
        // === Halo exchange on d_input ===
        if (nz_local > 0) {
            // Copy bottom owned plane (local_z=1) and top owned plane (local_z=nz_local) to host
            CUDA_CHECK(cudaMemcpy(send_bottom.data(), d_input + plane_size,
                                  plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(send_top.data(), d_input + nz_local * plane_size,
                                  plane_size * sizeof(Real), cudaMemcpyDeviceToHost));
        }

        // Exchange with neighbors:
        // Send my bottom plane down, receive top plane from above
        MPI_CHECK(MPI_Sendrecv(send_bottom.data(), (int)plane_size, MPI_DOUBLE,
                               rank_below, 0,
                               recv_top.data(), (int)plane_size, MPI_DOUBLE,
                               rank_above, 0,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        // Send my top plane up, receive bottom plane from below
        MPI_CHECK(MPI_Sendrecv(send_top.data(), (int)plane_size, MPI_DOUBLE,
                               rank_above, 1,
                               recv_bottom.data(), (int)plane_size, MPI_DOUBLE,
                               rank_below, 1,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        // Copy received halo data to GPU
        if (nz_local > 0) {
            if (rank_below != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_input, recv_bottom.data(),
                                      plane_size * sizeof(Real), cudaMemcpyHostToDevice));
            }
            if (rank_above != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpy(d_input + (nz_local + 1) * plane_size, recv_top.data(),
                                      plane_size * sizeof(Real), cudaMemcpyHostToDevice));
            }
        }

        // === Stencil computation on GPU ===
        if (nz_local > 0) {
            stencil_kernel<<<grid_dim, block>>>(d_input, d_output, nx, ny, nz_local,
                                                global_z_start, nz);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Swap input/output buffers
        std::swap(d_input, d_output);
    }

    // Synchronize all ranks after computation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    long long duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_CHECK(MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
                         0, MPI_COMM_WORLD));

    // === Copy result from GPU to CPU ===
    size_t owned_size = nz_local * plane_size;
    std::vector<Real> ownedData(owned_size);
    if (nz_local > 0) {
        // d_input holds the final result (after last swap)
        // Owned data is at local_z=1..nz_local, offset by plane_size
        CUDA_CHECK(cudaMemcpy(ownedData.data(), d_input + plane_size,
                              owned_size * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // === Gather all data to rank 0 ===
    int local_count = (int)owned_size;
    std::vector<int> recv_counts(nprocs);
    MPI_CHECK(MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                         0, MPI_COMM_WORLD));

    std::vector<int> recv_displs(nprocs, 0);
    if (rank == 0) {
        for (int r = 1; r < nprocs; ++r) {
            recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
        }
    }

    std::vector<Real> globalGrid;
    if (rank == 0) {
        globalGrid.resize(nx * ny * nz);
    }

    MPI_CHECK(MPI_Gatherv(ownedData.data(), local_count, MPI_DOUBLE,
                          rank == 0 ? globalGrid.data() : nullptr,
                          recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                          0, MPI_COMM_WORLD));

    // === Output results on rank 0 ===
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * (double)iterations;
        double mcups = (max_duration_ms > 0) ? cellUpdates / (max_duration_ms / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    if (nz_local > 0) {
        cudaFree(d_buf1);
        cudaFree(d_buf2);
    }

    MPI_CHECK(MPI_Finalize());
    return 0;
}
