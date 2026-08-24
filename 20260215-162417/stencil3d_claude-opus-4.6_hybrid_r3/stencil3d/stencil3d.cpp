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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 7-point stencil CUDA kernel operating on a z-range of the local subdomain.
// Local array has ghost layers: lz=0 is bottom ghost, lz=nz_local+1 is top ghost.
// Owned data is at lz=1..nz_local, mapping to global z = z_start_global + (lz-1).
__global__ __launch_bounds__(256)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   int nx, int ny, int nz_global,
                   int z_start_global, int lz_start, int lz_end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int lz = blockIdx.z * blockDim.z + threadIdx.z + lz_start;

    if (x >= nx || y >= ny || lz > lz_end) return;

    int z_global = z_start_global + (lz - 1);
    int stride_xy = nx * ny;
    int idx = lz * stride_xy + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
        z_global == 0 || z_global == nz_global - 1) {
        output[idx] = input[idx];
        return;
    }

    output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                   input[idx - nx] + input[idx + nx] +
                   input[idx - stride_xy] + input[idx + stride_xy]) / 7.0;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Each MPI rank claims one GPU
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along Z axis
    size_t base_nz = nz / (size_t)nprocs;
    size_t rem = nz % (size_t)nprocs;
    size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, rem);
    size_t nz_local = base_nz + ((size_t)rank < rem ? 1 : 0);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t nz_ghost = nz_local + 2;
    size_t slice_size = nx * ny;
    size_t local_size = slice_size * nz_ghost;

    // Pinned host buffers for async halo exchange
    Real *h_send_bot, *h_send_top, *h_recv_bot, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bot, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bot, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, slice_size * sizeof(Real)));

    // Device double-buffers
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_size * sizeof(Real)));

    // Initialize local grid on host with OpenMP, then copy to device
    if (rank == 0) printf("Initializing grid...\n");
    std::vector<Real> h_init(local_size);

    #pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < nz_ghost; ++lz) {
        for (size_t yi = 0; yi < ny; ++yi) {
            for (size_t xi = 0; xi < nx; ++xi) {
                size_t lidx = lz * slice_size + yi * nx + xi;
                if (lz == 0 || lz == nz_ghost - 1) {
                    h_init[lidx] = 0.0; // ghost layers
                } else {
                    size_t z_g = z_start + (lz - 1);
                    size_t gidx = z_g * slice_size + yi * nx + xi;
                    h_init[lidx] = (Real)(gidx % 19);
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_grid1, h_init.data(), local_size * sizeof(Real), cudaMemcpyHostToDevice));
    h_init.clear();
    h_init.shrink_to_fit();

    // MPI neighbor ranks (MPI_PROC_NULL for boundary ranks → safe no-op in Sendrecv)
    int rank_below = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_above = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Two streams: one for interior compute, one for halo communication + boundary compute
    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_comm));

    dim3 block(32, 8, 1); // 256 threads per block

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    Real *d_in = d_grid1, *d_out = d_grid2;
    int inx = (int)nx, iny = (int)ny, inz_g = (int)nz;
    int iz_start = (int)z_start, inz_local = (int)nz_local;

    for (int iter = 0; iter < iterations; ++iter) {
        // --- Async copy halo slices from GPU to pinned host ---
        CUDA_CHECK(cudaMemcpyAsync(h_send_bot, d_in + 1 * slice_size,
                                    slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));
        CUDA_CHECK(cudaMemcpyAsync(h_send_top, d_in + inz_local * slice_size,
                                    slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));

        // --- Launch interior kernel (lz=2..nz_local-1) on compute stream ---
        // These layers don't touch ghost data, so they can run during halo exchange
        if (inz_local > 2) {
            int interior_count = inz_local - 2;
            dim3 grid_int((inx + (int)block.x - 1) / (int)block.x,
                          (iny + (int)block.y - 1) / (int)block.y,
                          interior_count);
            stencilKernel<<<grid_int, block, 0, stream_compute>>>(
                d_in, d_out, inx, iny, inz_g, iz_start, 2, inz_local - 1);
        }

        // --- Wait for D→H copies, then do MPI halo exchange on host ---
        CUDA_CHECK(cudaStreamSynchronize(stream_comm));

        MPI_Sendrecv(h_send_bot, (int)slice_size, MPI_DOUBLE, rank_below, 0,
                     h_recv_top, (int)slice_size, MPI_DOUBLE, rank_above, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(h_send_top, (int)slice_size, MPI_DOUBLE, rank_above, 1,
                     h_recv_bot, (int)slice_size, MPI_DOUBLE, rank_below, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // --- Copy received halos back to GPU ---
        if (rank_below != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_recv_bot,
                                        slice_size * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
        }
        if (rank_above != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpyAsync(d_in + ((size_t)inz_local + 1) * slice_size, h_recv_top,
                                        slice_size * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
        }

        // --- Launch boundary layer kernels on comm stream (after halos arrive) ---
        if (inz_local >= 1) {
            dim3 grid_b((inx + (int)block.x - 1) / (int)block.x,
                        (iny + (int)block.y - 1) / (int)block.y, 1);
            stencilKernel<<<grid_b, block, 0, stream_comm>>>(
                d_in, d_out, inx, iny, inz_g, iz_start, 1, 1);
        }
        if (inz_local >= 2) {
            dim3 grid_t((inx + (int)block.x - 1) / (int)block.x,
                        (iny + (int)block.y - 1) / (int)block.y, 1);
            stencilKernel<<<grid_t, block, 0, stream_comm>>>(
                d_in, d_out, inx, iny, inz_g, iz_start, inz_local, inz_local);
        }

        // Sync all streams before buffer swap
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(d_in, d_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy owned result back from GPU (skip ghost layers)
    std::vector<Real> local_result(nz_local * slice_size);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_in + slice_size,
                          nz_local * slice_size * sizeof(Real), cudaMemcpyDeviceToHost));

    // Gather full grid on rank 0 for printing / validation
    if (printResults || validate) {
        int local_count = (int)(nz_local * slice_size);
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<Real> finalGrid;
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            finalGrid.resize(nx * ny * nz);
        }

        MPI_Gatherv(local_result.data(), local_count, MPI_DOUBLE,
                     rank == 0 ? finalGrid.data() : nullptr,
                     recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = true;

                for (size_t i = 0; i < finalGrid.size(); ++i) {
                    if (std::isnan(finalGrid[i]) || std::isinf(finalGrid[i])) {
                        printf("Validation failed: found NaN or Inf value\n");
                        valid = false;
                        break;
                    }
                }

                Real minVal = finalGrid[0], maxVal = finalGrid[0];
                #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
                for (size_t i = 0; i < finalGrid.size(); ++i) {
                    if (finalGrid[i] < minVal) minVal = finalGrid[i];
                    if (finalGrid[i] > maxVal) maxVal = finalGrid[i];
                }

                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    valid = false;
                }

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFreeHost(h_send_bot));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bot));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_comm));

    MPI_Finalize();
    return 0;
}
