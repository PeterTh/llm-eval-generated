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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__global__ void stencil7_kernel_xy_tiled(const Real* __restrict__ in,
                                        Real* __restrict__ out,
                                        int nx, int ny,
                                        int local_nz,
                                        long long z_start,
                                        int nz_global,
                                        int z_begin, int z_end) {
    extern __shared__ Real sh[]; // (blockDim.y+2) * (blockDim.x+2)

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_local = z_begin + static_cast<int>(blockIdx.z);

    if (z_local > z_end) return; // whole block out of range; safe to return

    const int tile_w = blockDim.x + 2;
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;

    const bool active = (x < nx) && (y < ny);
    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t base = active
        ? (static_cast<size_t>(z_local) * plane + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x))
        : 0;

    // Load center (inactive threads participate to avoid __syncthreads() divergence)
    sh[ty * tile_w + tx] = active ? in[base] : 0.0;

    // Load halos needed for left/right/front/back (only on tile border threads)
    if (threadIdx.x == 0) {
        sh[ty * tile_w + 0] = (active && x > 0) ? in[base - 1] : 0.0;
    }
    if (threadIdx.x == blockDim.x - 1) {
        sh[ty * tile_w + (tx + 1)] = (active && (x + 1) < nx) ? in[base + 1] : 0.0;
    }
    if (threadIdx.y == 0) {
        sh[0 * tile_w + tx] = (active && y > 0) ? in[base - nx] : 0.0;
    }
    if (threadIdx.y == blockDim.y - 1) {
        sh[(ty + 1) * tile_w + tx] = (active && (y + 1) < ny) ? in[base + nx] : 0.0;
    }

    __syncthreads();

    if (!active) return;

    const long long z_global = z_start + static_cast<long long>(z_local - 1);
    const bool is_boundary = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1) || (z_global == 0) || (z_global == (long long)(nz_global - 1));

    if (is_boundary) {
        out[base] = in[base];
        return;
    }

    const Real center = sh[ty * tile_w + tx];
    const Real left = sh[ty * tile_w + (tx - 1)];
    const Real right = sh[ty * tile_w + (tx + 1)];
    const Real front = sh[(ty - 1) * tile_w + tx];
    const Real back = sh[(ty + 1) * tile_w + tx];

    const Real bottom = in[base - plane];
    const Real top = in[base + plane];

    out[base] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
}

static bool validateResult(const std::vector<Real>& grid) {
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

static void printUsage(const char* progName) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;

        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Broadcast parameters
    unsigned long long nx_b = nx, ny_b = ny, nz_b = nz;
    MPI_Bcast(&nx_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_b, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    nx = static_cast<size_t>(nx_b);
    ny = static_cast<size_t>(ny_b);
    nz = static_cast<size_t>(nz_b);

    const long long nz_ll = static_cast<long long>(nz);
    const long long base = nz_ll / size;
    const long long rem = nz_ll % size;
    const long long local_nz_ll = base + (rank < rem ? 1 : 0);
    const long long z_start = base * rank + (rank < rem ? rank : rem);

    const int local_nz = static_cast<int>(local_nz_ll);
    const size_t plane = nx * ny;
    const size_t local_with_halo = static_cast<size_t>(local_nz + 2) * plane;

    // Host initialization (real cells only), including halos in allocation.
    std::vector<Real> h_init(local_with_halo, 0.0);

    #pragma omp parallel for collapse(2)
    for (int z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const long long z_global = z_start + z;
            const size_t z_off_local = static_cast<size_t>(z + 1) * plane;
            const size_t z_off_global = static_cast<size_t>(z_global) * plane;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gidx = z_off_global + y * nx + x;
                h_init[z_off_local + y * nx + x] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }

    int devCount = 0;
    cudaCheck(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount");
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    cudaCheck(cudaSetDevice(rank % devCount), "cudaSetDevice");

    Real *d_a = nullptr, *d_b = nullptr;
    cudaCheck(cudaMalloc(&d_a, local_with_halo * sizeof(Real)), "cudaMalloc d_a");
    cudaCheck(cudaMalloc(&d_b, local_with_halo * sizeof(Real)), "cudaMalloc d_b");
    cudaCheck(cudaMemcpy(d_a, h_init.data(), local_with_halo * sizeof(Real), cudaMemcpyHostToDevice), "H2D init");

    // Pinned host buffers for halo exchange
    Real *h_send_prev = nullptr, *h_send_next = nullptr, *h_recv_prev = nullptr, *h_recv_next = nullptr;
    cudaCheck(cudaMallocHost(&h_send_prev, plane * sizeof(Real)), "cudaMallocHost send_prev");
    cudaCheck(cudaMallocHost(&h_send_next, plane * sizeof(Real)), "cudaMallocHost send_next");
    cudaCheck(cudaMallocHost(&h_recv_prev, plane * sizeof(Real)), "cudaMallocHost recv_prev");
    cudaCheck(cudaMallocHost(&h_recv_next, plane * sizeof(Real)), "cudaMallocHost recv_next");

    const int prev = (rank == 0) ? MPI_PROC_NULL : (rank - 1);
    const int next = (rank == size - 1) ? MPI_PROC_NULL : (rank + 1);

    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    // Kernel configuration
    dim3 block(32, 8, 1);
    const int nx_i = static_cast<int>(nx);
    const int ny_i = static_cast<int>(ny);
    const int nz_i = static_cast<int>(nz);

    const size_t shmem_bytes = static_cast<size_t>(block.x + 2) * static_cast<size_t>(block.y + 2) * sizeof(Real);

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        // Post receives
        MPI_Irecv(h_recv_prev, static_cast<int>(plane), MPI_DOUBLE, prev, 100, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_recv_next, static_cast<int>(plane), MPI_DOUBLE, next, 101, MPI_COMM_WORLD, &reqs[1]);

        // Stage sends from device boundary planes
        if (prev != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(h_send_prev, d_a + 1 * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream), "D2H send_prev");
        }
        if (next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(h_send_next, d_a + static_cast<size_t>(local_nz) * plane, plane * sizeof(Real), cudaMemcpyDeviceToHost, stream), "D2H send_next");
        }
        cudaCheck(cudaStreamSynchronize(stream), "sync D2H sends");

        // Post sends
        MPI_Isend(h_send_prev, static_cast<int>(plane), MPI_DOUBLE, prev, 101, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(h_send_next, static_cast<int>(plane), MPI_DOUBLE, next, 100, MPI_COMM_WORLD, &reqs[3]);

        // Compute interior z-planes that don't depend on halos: z_local in [2, local_nz-1]
        if (local_nz >= 3) {
            const int z_begin = 2;
            const int z_end = local_nz - 1;
            const int z_count = z_end - z_begin + 1;
            dim3 grid((nx_i + block.x - 1) / block.x, (ny_i + block.y - 1) / block.y, z_count);
            stencil7_kernel_xy_tiled<<<grid, block, shmem_bytes, stream>>>(d_a, d_b, nx_i, ny_i, local_nz, z_start, nz_i, z_begin, z_end);
        }

        // Wait for halos
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Copy received halos to device
        if (prev != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(d_a + 0 * plane, h_recv_prev, plane * sizeof(Real), cudaMemcpyHostToDevice, stream), "H2D recv_prev");
        }
        if (next != MPI_PROC_NULL) {
            cudaCheck(cudaMemcpyAsync(d_a + static_cast<size_t>(local_nz + 1) * plane, h_recv_next, plane * sizeof(Real), cudaMemcpyHostToDevice, stream), "H2D recv_next");
        }

        // Compute border planes z_local = 1 and z_local = local_nz
        {
            // z_local = 1
            if (local_nz >= 1) {
                dim3 grid((nx_i + block.x - 1) / block.x, (ny_i + block.y - 1) / block.y, 1);
                stencil7_kernel_xy_tiled<<<grid, block, shmem_bytes, stream>>>(d_a, d_b, nx_i, ny_i, local_nz, z_start, nz_i, 1, 1);
            }
            // z_local = local_nz (if distinct)
            if (local_nz >= 2) {
                dim3 grid((nx_i + block.x - 1) / block.x, (ny_i + block.y - 1) / block.y, 1);
                stencil7_kernel_xy_tiled<<<grid, block, shmem_bytes, stream>>>(d_a, d_b, nx_i, ny_i, local_nz, z_start, nz_i, local_nz, local_nz);
            }
        }

        cudaCheck(cudaGetLastError(), "kernel launch");
        cudaCheck(cudaStreamSynchronize(stream), "sync iter");

        std::swap(d_a, d_b);
    }

    const double t1 = MPI_Wtime();
    double local_s = t1 - t0;
    double max_s = 0.0;
    MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_s / 1e6;
        printf("Computation time: %.3f s (max across ranks)\n", max_s);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 (real cells only)
    std::vector<Real> h_local(plane * static_cast<size_t>(local_nz));
    if (local_nz > 0) {
        cudaCheck(cudaMemcpy(h_local.data(), d_a + 1 * plane, h_local.size() * sizeof(Real), cudaMemcpyDeviceToHost), "D2H final");
    }

    std::vector<int> recvCounts;
    std::vector<int> displs;
    std::vector<Real> finalGrid;
    if (rank == 0) {
        recvCounts.resize(size);
        displs.resize(size);
        long long disp = 0;
        for (int r = 0; r < size; ++r) {
            const long long lz = base + (r < rem ? 1 : 0);
            const long long cnt = lz * static_cast<long long>(plane);
            recvCounts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(disp);
            disp += cnt;
        }
        finalGrid.resize(static_cast<size_t>(disp));
    }

    MPI_Gatherv(h_local.data(), static_cast<int>(h_local.size()), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            cudaCheck(cudaFree(d_a), "cudaFree d_a");
            cudaCheck(cudaFree(d_b), "cudaFree d_b");
            cudaCheck(cudaFreeHost(h_send_prev), "cudaFreeHost send_prev");
            cudaCheck(cudaFreeHost(h_send_next), "cudaFreeHost send_next");
            cudaCheck(cudaFreeHost(h_recv_prev), "cudaFreeHost recv_prev");
            cudaCheck(cudaFreeHost(h_recv_next), "cudaFreeHost recv_next");
            cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    cudaCheck(cudaFree(d_a), "cudaFree d_a");
    cudaCheck(cudaFree(d_b), "cudaFree d_b");
    cudaCheck(cudaFreeHost(h_send_prev), "cudaFreeHost send_prev");
    cudaCheck(cudaFreeHost(h_send_next), "cudaFreeHost send_next");
    cudaCheck(cudaFreeHost(h_recv_prev), "cudaFreeHost recv_prev");
    cudaCheck(cudaFreeHost(h_recv_next), "cudaFreeHost recv_next");
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");

    MPI_Finalize();
    return 0;
}
