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

// 7-point stencil CUDA kernel operating on a local slab with ghost layers.
// z ranges from 1 to local_nz (owned layers); ghost layers at 0 and local_nz+1.
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               size_t nx, size_t ny, size_t local_nz,
                               bool is_first_rank, bool is_last_rank) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) return;

    size_t slab = nx * ny;
    size_t idx = z * slab + y * nx + x;

    bool is_boundary = (x == 0 || x == nx - 1 ||
                        y == 0 || y == ny - 1 ||
                        (z == 1 && is_first_rank) ||
                        (z == local_nz && is_last_rank));

    if (is_boundary) {
        output[idx] = input[idx];
    } else {
        output[idx] = (input[idx] +
                       input[idx - 1]    + input[idx + 1] +
                       input[idx - nx]   + input[idx + nx] +
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

    // Assign GPU round-robin across local devices
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

    // MPI domain decomposition along Z axis
    size_t base_nz = nz / nprocs;
    size_t rem_nz  = nz % nprocs;
    size_t local_nz = base_nz + ((size_t)rank < rem_nz ? 1 : 0);
    size_t z_start  = (size_t)rank * base_nz + std::min((size_t)rank, rem_nz);

    bool is_first = (rank == 0);
    bool is_last  = (rank == nprocs - 1);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", nprocs);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t slab = nx * ny;
    size_t local_total = slab * (local_nz + 2); // owned layers + 2 ghost layers

    // Initialize local grid on host with OpenMP
    std::vector<Real> h_grid(local_total, 0.0);

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t zg = z_start + (zl - 1);
                size_t global_idx = zg * slab + y * nx + x;
                h_grid[zl * slab + y * nx + x] = (global_idx % 19) * 1.0;
            }
        }
    }

    if (rank == 0) printf("Initializing grid...\n");

    // Device double buffers
    Real *d_buf1, *d_buf2;
    CUDA_CHECK(cudaMalloc(&d_buf1, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_buf2, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_buf1, h_grid.data(), local_total * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_buf2, 0, local_total * sizeof(Real)));

    // Pinned host buffers for halo exchange
    Real *h_send_lo, *h_send_hi, *h_recv_lo, *h_recv_hi;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, slab * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, slab * sizeof(Real)));

    // CUDA streams for overlapping halo transfer with interior computation
    cudaStream_t stream_comp, stream_halo;
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    CUDA_CHECK(cudaStreamCreate(&stream_halo));

    // Kernel config
    dim3 block(32, 8, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    Real *d_in = d_buf1, *d_out = d_buf2;

    for (int iter = 0; iter < iterations; ++iter) {
        // --- Halo exchange: D2H, MPI, H2D ---
        CUDA_CHECK(cudaMemcpy(h_send_lo, d_in + 1 * slab,
                              slab * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_send_hi, d_in + local_nz * slab,
                              slab * sizeof(Real), cudaMemcpyDeviceToHost));

        MPI_Request reqs[4];
        int nreq = 0;
        if (!is_first) {
            MPI_Isend(h_send_lo, (int)slab, MPI_DOUBLE, rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(h_recv_lo, (int)slab, MPI_DOUBLE, rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
        if (!is_last) {
            MPI_Isend(h_send_hi, (int)slab, MPI_DOUBLE, rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
            MPI_Irecv(h_recv_hi, (int)slab, MPI_DOUBLE, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        if (!is_first) {
            CUDA_CHECK(cudaMemcpy(d_in, h_recv_lo,
                                  slab * sizeof(Real), cudaMemcpyHostToDevice));
        }
        if (!is_last) {
            CUDA_CHECK(cudaMemcpy(d_in + (local_nz + 1) * slab, h_recv_hi,
                                  slab * sizeof(Real), cudaMemcpyHostToDevice));
        }

        // --- Launch stencil kernel ---
        if (local_nz > 0) {
            dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                      (unsigned)((ny + block.y - 1) / block.y),
                      (unsigned)((local_nz + block.z - 1) / block.z));
            stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, local_nz,
                                            is_first, is_last);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_in, d_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", dur.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (dur.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy owned layers back to host
    std::vector<Real> local_result(local_nz * slab);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_in + slab,
                          local_nz * slab * sizeof(Real), cudaMemcpyDeviceToHost));

    // Gather to rank 0 for printing / validation
    if (printResults || validate) {
        std::vector<int> rcounts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            size_t rnz = base_nz + ((size_t)r < rem_nz ? 1 : 0);
            rcounts[r] = (int)(rnz * slab);
            displs[r] = (r == 0) ? 0 : displs[r - 1] + rcounts[r - 1];
        }

        std::vector<Real> global_grid;
        if (rank == 0) global_grid.resize(nx * ny * nz);

        MPI_Gatherv(local_result.data(), (int)(local_nz * slab), MPI_DOUBLE,
                     rank == 0 ? global_grid.data() : nullptr,
                     rcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global_grid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                Real minVal = global_grid[0], maxVal = global_grid[0];
                int bad = 0;
                #pragma omp parallel for reduction(+:bad) reduction(min:minVal) reduction(max:maxVal)
                for (size_t i = 0; i < global_grid.size(); ++i) {
                    if (std::isnan(global_grid[i]) || std::isinf(global_grid[i])) bad++;
                    minVal = std::min(minVal, global_grid[i]);
                    maxVal = std::max(maxVal, global_grid[i]);
                }
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                bool valid = (bad == 0 && maxVal <= 1e6 && minVal >= -1e6);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
    CUDA_CHECK(cudaStreamDestroy(stream_halo));
    CUDA_CHECK(cudaFree(d_buf1));
    CUDA_CHECK(cudaFree(d_buf2));

    MPI_Finalize();
    return 0;
}
