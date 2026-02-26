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
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 7-point stencil kernel with boundary copy
// Each thread processes one (x, y, lz) point in the local grid.
// lz is 1-based (1..local_nz are owned planes; 0 and local_nz+1 are halos).
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx, const int ny,
                              const int local_nz,
                              const int gz_start,
                              const int gnz,
                              const int lz_begin,
                              const int lz_end)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int lz = blockIdx.z + lz_begin;

    if (x >= nx || y >= ny || lz > lz_end) return;

    int gz = gz_start + (lz - 1);
    size_t slab = (size_t)nx * ny;
    size_t idx = (size_t)lz * slab + (size_t)y * nx + x;

    bool is_boundary = (x == 0 || x == nx - 1 ||
                        y == 0 || y == ny - 1 ||
                        gz == 0 || gz == gnz - 1);

    if (is_boundary) {
        output[idx] = input[idx];
    } else {
        output[idx] = (input[idx] +
                       input[idx - 1] + input[idx + 1] +
                       input[idx - nx] + input[idx + nx] +
                       input[idx - slab] + input[idx + slab]) / 7.0;
    }
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

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU (round-robin across available devices)
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    // Domain decomposition along Z
    int inz = (int)nz;
    int base_nz = inz / nprocs;
    int rem = inz % nprocs;
    int local_nz = base_nz + (rank < rem ? 1 : 0);
    int gz_start = rank * base_nz + std::min(rank, rem);

    size_t slab = nx * ny;
    size_t local_total = slab * (size_t)(local_nz + 2); // owned + 2 halo planes

    // Allocate pinned host buffers for halo exchange
    Real *h_send_bot = nullptr, *h_send_top = nullptr;
    Real *h_recv_bot = nullptr, *h_recv_top = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send_bot, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bot, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, slab * sizeof(Real)));

    // Allocate device grids (double-buffered)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_grid1, 0, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_grid2, 0, local_total * sizeof(Real)));

    // Initialize owned planes on host using OpenMP, then copy to device
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> h_local(local_total, 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int lz = 1; lz <= local_nz; ++lz) {
        for (int yy = 0; yy < (int)ny; ++yy) {
            int gz = gz_start + (lz - 1);
            for (int xx = 0; xx < (int)nx; ++xx) {
                size_t global_idx = (size_t)gz * slab + (size_t)yy * (size_t)nx + (size_t)xx;
                size_t local_idx = (size_t)lz * slab + (size_t)yy * (size_t)nx + (size_t)xx;
                h_local[local_idx] = (Real)(global_idx % 19) * 1.0;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_grid1, h_local.data(), local_total * sizeof(Real),
                           cudaMemcpyHostToDevice));

    // CUDA streams for overlapping computation with communication
    cudaStream_t stream_comp, stream_halo;
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    CUDA_CHECK(cudaStreamCreate(&stream_halo));

    // Kernel launch configuration
    dim3 block(32, 8, 1);
    auto make_grid = [&](int nz_planes) -> dim3 {
        return dim3(((int)nx + block.x - 1) / block.x,
                    ((int)ny + block.y - 1) / block.y,
                    nz_planes > 0 ? nz_planes : 1);
    };

    int prev_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int next_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* d_in  = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        if (local_nz == 0) continue;

        // --- Phase 1: Start async halo D->H copy while launching interior kernel ---

        // Copy bottom owned plane (lz=1) and top owned plane (lz=local_nz) to host
        CUDA_CHECK(cudaMemcpyAsync(h_send_bot, d_in + slab,
                                    slab * sizeof(Real), cudaMemcpyDeviceToHost, stream_halo));
        CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_in + (size_t)local_nz * slab,
                                    slab * sizeof(Real), cudaMemcpyDeviceToHost, stream_halo));

        // Launch interior kernel (lz=2..local_nz-1) which doesn't need neighbor halos
        int int_begin = 2;
        int int_end = local_nz - 1;
        if (int_begin <= int_end) {
            dim3 g = make_grid(int_end - int_begin + 1);
            stencilKernel<<<g, block, 0, stream_comp>>>(
                d_in, d_out, (int)nx, (int)ny, local_nz, gz_start, (int)nz,
                int_begin, int_end);
        }

        // --- Phase 2: Complete halo exchange ---

        CUDA_CHECK(cudaStreamSynchronize(stream_halo));

        // MPI halo exchange with neighbors
        MPI_Sendrecv(h_send_bot, (int)slab, MPI_DOUBLE, prev_rank, 0,
                     h_recv_top, (int)slab, MPI_DOUBLE, next_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_top, (int)slab, MPI_DOUBLE, next_rank, 1,
                     h_recv_bot, (int)slab, MPI_DOUBLE, prev_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Copy received halos to device
        if (prev_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_recv_bot,
                                        slab * sizeof(Real), cudaMemcpyHostToDevice, stream_halo));
        }
        if (next_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in + (size_t)(local_nz + 1) * slab, h_recv_top,
                                        slab * sizeof(Real), cudaMemcpyHostToDevice, stream_halo));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_halo));

        // --- Phase 3: Compute boundary z-planes that required halos ---

        // Bottom owned plane (lz=1)
        {
            dim3 g = make_grid(1);
            stencilKernel<<<g, block, 0, stream_comp>>>(
                d_in, d_out, (int)nx, (int)ny, local_nz, gz_start, (int)nz, 1, 1);
        }
        // Top owned plane (lz=local_nz) if different from bottom
        if (local_nz > 1) {
            dim3 g = make_grid(1);
            stencilKernel<<<g, block, 0, stream_comp>>>(
                d_in, d_out, (int)nx, (int)ny, local_nz, gz_start, (int)nz,
                local_nz, local_nz);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_comp));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // --- Gather full grid to rank 0 for validation / output ---

    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;

    size_t owned_size = (size_t)local_nz * slab;
    std::vector<Real> h_owned(owned_size);
    if (owned_size > 0) {
        CUDA_CHECK(cudaMemcpy(h_owned.data(), d_final + slab,
                               owned_size * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    std::vector<int> recv_counts(nprocs);
    std::vector<int> displs(nprocs);
    int my_count = (int)owned_size;
    MPI_Gather(&my_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Real> fullGrid;
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
        }
        fullGrid.resize(nx * ny * nz);
    }

    MPI_Gatherv(h_owned.data(), my_count, MPI_DOUBLE,
                fullGrid.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int ret = 0;

    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = true;

            Real minVal = fullGrid[0], maxVal = fullGrid[0];
            #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
            for (size_t i = 0; i < fullGrid.size(); ++i) {
                if (std::isnan(fullGrid[i]) || std::isinf(fullGrid[i])) {
                    valid = false;
                }
                minVal = std::min(minVal, fullGrid[i]);
                maxVal = std::max(maxVal, fullGrid[i]);
            }

            if (!valid) {
                printf("Validation failed: found NaN or Inf value\n");
            } else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }

            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) ret = 1;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(h_send_bot));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bot));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
    CUDA_CHECK(cudaStreamDestroy(stream_halo));

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
