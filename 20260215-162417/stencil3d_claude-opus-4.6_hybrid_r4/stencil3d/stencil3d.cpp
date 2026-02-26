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

// Unified stencil kernel: handles both interior computation and boundary copy.
// Processes local z-planes in [lz_start, lz_end].
// gz = gz_offset + lz maps local z to global z.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                               const int nx, const int ny,
                               const int lz_start, const int lz_end,
                               const int gz_offset, const int nz_global) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lz = lz_start + (int)(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= nx || y >= ny || lz > lz_end) return;

    const int slab = nx * ny;
    const size_t idx = (size_t)lz * slab + (size_t)y * nx + x;
    const int gz = gz_offset + lz;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
        gz == 0 || gz == nz_global - 1) {
        out[idx] = in[idx];
    } else {
        out[idx] = (in[idx] +
                    in[idx - 1] + in[idx + 1] +
                    in[idx - nx] + in[idx + nx] +
                    in[idx - slab] + in[idx + slab]) / 7.0;
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

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

    // Bind each MPI rank to a GPU (round-robin)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
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

    // Domain decomposition along Z axis
    size_t base_nz = nz / (size_t)nprocs;
    size_t rem = nz % (size_t)nprocs;
    size_t z_start = (size_t)rank * base_nz + std::min((size_t)rank, rem);
    size_t local_nz = base_nz + ((size_t)rank < rem ? 1 : 0);

    if (local_nz == 0) {
        fprintf(stderr, "Rank %d: no z-slices assigned (nz=%zu < nprocs=%d)\n",
                rank, nz, nprocs);
        MPI_Finalize();
        return 1;
    }

    const size_t plane_size = nx * ny;
    const size_t local_total = plane_size * (local_nz + 2);
    // gz = gz_offset + lz: maps local z index to global z
    // lz=1 is first owned plane with global z = z_start
    const int gz_offset = (int)z_start - 1;

    // Initialize local grid (with ghost zones) using OpenMP
    std::vector<Real> h_grid(local_total);

    #pragma omp parallel for schedule(static)
    for (size_t lz = 0; lz < local_nz + 2; ++lz) {
        int gz = gz_offset + (int)lz;
        for (size_t iy = 0; iy < ny; ++iy) {
            for (size_t ix = 0; ix < nx; ++ix) {
                size_t local_idx = lz * plane_size + iy * nx + ix;
                if (gz >= 0 && gz < (int)nz) {
                    size_t global_idx = (size_t)gz * plane_size + iy * nx + ix;
                    h_grid[local_idx] = (Real)(global_idx % 19);
                } else {
                    h_grid[local_idx] = 0.0;
                }
            }
        }
    }

    // Allocate device memory and copy initial data
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, local_total * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid.data(), local_total * sizeof(Real),
                          cudaMemcpyHostToDevice));

    // Pinned host buffers for asynchronous halo exchange
    Real *h_send_lo, *h_send_hi, *h_recv_lo, *h_recv_hi;
    CUDA_CHECK(cudaMallocHost(&h_send_lo, plane_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_hi, plane_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_lo, plane_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_hi, plane_size * sizeof(Real)));

    // Dual CUDA streams: computation overlaps with communication
    cudaStream_t stream_comp, stream_comm;
    CUDA_CHECK(cudaStreamCreate(&stream_comp));
    CUDA_CHECK(cudaStreamCreate(&stream_comm));

    // Kernel launch configuration
    const dim3 block(32, 8, 1);
    auto make_grid = [&](int z_range) -> dim3 {
        return dim3(((int)nx + (int)block.x - 1) / (int)block.x,
                    ((int)ny + (int)block.y - 1) / (int)block.y,
                    std::max(1, (z_range + (int)block.z - 1) / (int)block.z));
    };

    if (rank == 0) printf("Initializing grid...\n");
    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *d_in  = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real *d_out = (iter % 2 == 0) ? d_grid2 : d_grid1;

        // --- Phase 1: Extract halos to host (async) while computing interior ---

        // Copy halo planes from device to pinned host memory
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_in + plane_size,
                       plane_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));
        if (rank < nprocs - 1)
            CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_in + local_nz * plane_size,
                       plane_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));

        // Launch interior kernel on comp stream (z-planes that don't touch ghost zones)
        if ((int)local_nz > 2) {
            dim3 g = make_grid((int)local_nz - 2);
            stencilKernel<<<g, block, 0, stream_comp>>>(
                d_in, d_out, (int)nx, (int)ny,
                2, (int)local_nz - 1, gz_offset, (int)nz);
        }

        // --- Phase 2: MPI halo exchange (after halo data reaches host) ---
        CUDA_CHECK(cudaStreamSynchronize(stream_comm));

        MPI_Request reqs[4];
        int nreqs = 0;

        if (rank > 0) {
            MPI_Isend(h_send_lo, (int)plane_size, MPI_DOUBLE,
                      rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_lo, (int)plane_size, MPI_DOUBLE,
                      rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < nprocs - 1) {
            MPI_Isend(h_send_hi, (int)plane_size, MPI_DOUBLE,
                      rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_hi, (int)plane_size, MPI_DOUBLE,
                      rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }

        if (nreqs > 0)
            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // --- Phase 3: Copy received halos to device, then compute halo faces ---
        if (rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_in, h_recv_lo,
                       plane_size * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
        if (rank < nprocs - 1)
            CUDA_CHECK(cudaMemcpyAsync(d_in + (local_nz + 1) * plane_size, h_recv_hi,
                       plane_size * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));

        // Launch halo-dependent face kernels on comm stream (ordered after memcpy)
        dim3 g_face = make_grid(1);
        stencilKernel<<<g_face, block, 0, stream_comm>>>(
            d_in, d_out, (int)nx, (int)ny,
            1, 1, gz_offset, (int)nz);

        if (local_nz > 1) {
            stencilKernel<<<g_face, block, 0, stream_comm>>>(
                d_in, d_out, (int)nx, (int)ny,
                (int)local_nz, (int)local_nz, gz_offset, (int)nz);
        }

        // --- Phase 4: Synchronize all GPU work before next iteration ---
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0
    Real *d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;

    // Copy owned cells (skip ghost zones) from device to host
    std::vector<Real> h_local(local_nz * plane_size);
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_final + plane_size,
               local_nz * plane_size * sizeof(Real), cudaMemcpyDeviceToHost));

    int local_count = (int)(local_nz * plane_size);
    std::vector<int> recvcounts(nprocs), displs(nprocs);
    MPI_Gather(&local_count, 1, MPI_INT,
               recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i)
            displs[i] = displs[i-1] + recvcounts[i-1];
    }

    std::vector<Real> fullGrid;
    if (rank == 0) fullGrid.resize(nx * ny * nz);

    MPI_Gatherv(h_local.data(), local_count, MPI_DOUBLE,
                fullGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results and validate on rank 0 only
    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(fullGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream_comp));
    CUDA_CHECK(cudaStreamDestroy(stream_comm));
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
