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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: 7-point stencil on owned rows + boundary copy
// Layout: local array has (local_nz+2) planes. Index 0 = top halo,
// indices 1..local_nz = owned, index local_nz+1 = bottom halo.
// This kernel processes only owned rows (z in [1, local_nz]).
// global_z_start: the global z-index of local index 1 (first owned row).
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny,
                              const size_t local_nz,
                              const size_t global_z_start,
                              const size_t global_nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z + 1; // owned rows: 1..local_nz

    if (x >= nx || y >= ny || z > local_nz) return;

    const size_t plane = nx * ny;
    const size_t idx = z * plane + y * nx + x;
    const size_t gz = global_z_start + (z - 1); // global z-index

    // Boundary check (matches original: x==0||nx-1, y==0||ny-1, z==0||nz-1)
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
        gz == 0 || gz == global_nz - 1) {
        output[idx] = input[idx];
    } else if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1) {
        // Interior stencil
        const Real center = input[idx];
        const Real left   = input[idx - 1];
        const Real right  = input[idx + 1];
        const Real front  = input[idx - nx];
        const Real back   = input[idx + nx];
        const Real bottom = input[idx - plane];
        const Real top    = input[idx + plane];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Initialize owned rows and halos using global indices
void initializeGridLocal(Real* grid, size_t nx, size_t ny, size_t local_nz,
                         size_t local_nz_halo, size_t z_offset,
                         int mpi_rank, int mpi_size) {
    const size_t plane = nx * ny;

    // Initialize owned rows (local z = 1..local_nz)
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            size_t gz = z_offset + (lz - 1);
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gz * plane + y * nx + x;
                grid[lz * plane + y * nx + x] = static_cast<Real>(global_idx % 19);
            }
        }
    }

    // Initialize top halo (local z = 0)
    if (mpi_rank == 0) {
        // Boundary: copy first owned row
        memcpy(&grid[0], &grid[plane], plane * sizeof(Real));
    } else {
        // Halo from neighbor (global z = z_offset - 1)
        size_t gz = z_offset - 1;
        #pragma omp parallel for schedule(static)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gz * plane + y * nx + x;
                grid[y * nx + x] = static_cast<Real>(global_idx % 19);
            }
        }
    }

    // Initialize bottom halo (local z = local_nz + 1)
    if (mpi_rank == mpi_size - 1) {
        // Boundary: copy last owned row
        memcpy(&grid[(local_nz + 1) * plane], &grid[local_nz * plane], plane * sizeof(Real));
    } else {
        // Halo from neighbor (global z = z_offset + local_nz)
        size_t gz = z_offset + local_nz;
        #pragma omp parallel for schedule(static)
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_idx = gz * plane + y * nx + x;
                grid[(local_nz + 1) * plane + y * nx + x] = static_cast<Real>(global_idx % 19);
            }
        }
    }
}

bool validateResult(const Real* grid, size_t total_size) {
    bool has_nan_inf = false;
    Real g_min = grid[0], g_max = grid[0];

    #pragma omp parallel
    {
        Real tmin = grid[0], tmax = grid[0];
        bool tnan = false;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < total_size; ++i) {
            Real v = grid[i];
            if (std::isnan(v) || std::isinf(v)) tnan = true;
            if (v < tmin) tmin = v;
            if (v > tmax) tmax = v;
        }
        #pragma omp critical
        {
            if (tnan) has_nan_inf = true;
            if (tmin < g_min) g_min = tmin;
            if (tmax > g_max) g_max = tmax;
        }
    }

    if (has_nan_inf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", g_min, g_max);

    if (g_max > 1e6 || g_min < -1e6) {
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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Assign one GPU per MPI rank
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(mpi_rank % num_gpus));

    // Parse arguments (all ranks identical)
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (mpi_rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs: %d\n", mpi_size, num_gpus);
    }

    // 1D domain decomposition along Z
    size_t base_z = nz / mpi_size;
    size_t rem = nz % mpi_size;
    size_t local_nz, z_offset;
    if ((size_t)mpi_rank < rem) {
        local_nz = base_z + 1;
        z_offset = (size_t)mpi_rank * (base_z + 1);
    } else {
        local_nz = base_z;
        z_offset = rem * (base_z + 1) + ((size_t)mpi_rank - rem) * base_z;
    }

    size_t local_nz_halo = local_nz + 2; // halos at index 0 and local_nz+1
    size_t plane = nx * ny;
    size_t local_size = local_nz_halo * plane;

    // Device buffers (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_size * sizeof(Real)));

    // Pinned host buffers
    Real *h_buf;
    CUDA_CHECK(cudaMallocHost(&h_buf, local_size * sizeof(Real)));

    // Pinned halo exchange buffers
    Real *h_send, *h_recv;
    CUDA_CHECK(cudaMallocHost(&h_send, plane * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv, plane * sizeof(Real)));

    // Initialize
    if (mpi_rank == 0) printf("Initializing grid...\n");
    size_t global_z_start = z_offset; // global z of local index 1
    initializeGridLocal(h_buf, nx, ny, local_nz, local_nz_halo, z_offset, mpi_rank, mpi_size);

    CUDA_CHECK(cudaMemcpy(d_grid1, h_buf, local_size * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, h_buf, local_size * sizeof(Real), cudaMemcpyHostToDevice));

    // MPI neighbors
    int rank_up = (mpi_rank > 0) ? mpi_rank - 1 : MPI_PROC_NULL;
    int rank_down = (mpi_rank < mpi_size - 1) ? mpi_rank + 1 : MPI_PROC_NULL;

    // Kernel launch config: 2D thread blocks, 1D grid in z over owned rows
    dim3 block(16, 8);
    dim3 grid_dim((unsigned)(nx + 15) / 16, (unsigned)(ny + 7) / 8, (unsigned)local_nz);

    if (mpi_rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // Halo exchange on d_in (prepare halos before kernel reads them)
        // Send first owned row (z=1) up, receive top halo (z=0) from up
        CUDA_CHECK(cudaMemcpy(h_send, &d_in[plane], plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h_send, (int)(plane * sizeof(Real)), MPI_BYTE, rank_up, 0,
                     h_recv, (int)(plane * sizeof(Real)), MPI_BYTE, rank_up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(&d_in[0], h_recv, plane * sizeof(Real), cudaMemcpyHostToDevice));

        // Send last owned row (z=local_nz) down, receive bottom halo (z=local_nz+1) from down
        CUDA_CHECK(cudaMemcpy(h_send, &d_in[local_nz * plane], plane * sizeof(Real), cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h_send, (int)(plane * sizeof(Real)), MPI_BYTE, rank_down, 1,
                     h_recv, (int)(plane * sizeof(Real)), MPI_BYTE, rank_down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(&d_in[(local_nz + 1) * plane], h_recv, plane * sizeof(Real), cudaMemcpyHostToDevice));

        // Launch stencil kernel
        stencilKernel<<<grid_dim, block>>>(d_in, d_out, nx, ny, local_nz, global_z_start, nz);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long global_ms;
    MPI_Allreduce(&local_ms, &global_ms, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Computation time: %lld ms\n", global_ms);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = (global_ms > 0) ? cellUpdates / (global_ms / 1000.0) / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0
    const size_t global_size = nx * ny * nz;
    std::vector<Real> global_grid;
    if (mpi_rank == 0) global_grid.resize(global_size);

    Real *d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    // Reuse h_buf for gathering owned rows
    CUDA_CHECK(cudaMemcpy(h_buf, &d_final[plane], local_nz * plane * sizeof(Real), cudaMemcpyDeviceToHost));

    std::vector<int> recvcounts(mpi_size), displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        size_t r_nz = ((size_t)r < rem) ? base_z + 1 : base_z;
        size_t r_off = ((size_t)r < rem) ? (size_t)r * (base_z + 1)
                                          : rem * (base_z + 1) + ((size_t)r - rem) * base_z;
        recvcounts[r] = (int)(r_nz * plane * sizeof(Real));
        displs[r] = (int)(r_off * plane * sizeof(Real));
    }

    MPI_Gatherv(h_buf, (int)(local_nz * plane * sizeof(Real)), MPI_BYTE,
                global_grid.data(), recvcounts.data(), displs.data(),
                MPI_BYTE, 0, MPI_COMM_WORLD);

    if (printResults && mpi_rank == 0) {
        print_results(global_grid, "Grid");
    }

    int rc = 0;
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        if (validateResult(global_grid.data(), global_size)) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    MPI_Finalize();
    return rc;
}
