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

namespace {

__host__ __device__ __forceinline__ size_t idx3d(const int x, const int y, const int z, const int nx, const int ny) {
    return static_cast<size_t>(z) * static_cast<size_t>(nx) * static_cast<size_t>(ny) + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

inline void mpi_abort(const char* msg, const int err = 1) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        fprintf(stderr, "%s\n", msg);
    }
    MPI_Abort(MPI_COMM_WORLD, err);
}

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t _err = (call);                                                  \
        if (_err != cudaSuccess) {                                                        \
            char buf[512];                                                                \
            snprintf(buf, sizeof(buf), "CUDA error %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            mpi_abort(buf, 2);                                                            \
        }                                                                                \
    } while (0)

inline int getenv_int(const char* name, const int default_val = -1) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_val;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return default_val;
    return static_cast<int>(x);
}

inline int get_local_rank_fallback() {
    // Common environment variables across MPI implementations.
    int lr = getenv_int("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (lr >= 0) return lr;
    lr = getenv_int("MV2_COMM_WORLD_LOCAL_RANK", -1);
    if (lr >= 0) return lr;
    lr = getenv_int("SLURM_LOCALID", -1);
    if (lr >= 0) return lr;
    lr = getenv_int("PMI_LOCAL_RANK", -1);
    if (lr >= 0) return lr;
    return -1;
}

struct DecompZ {
    int z_start = 0;   // global start (inclusive)
    int nz_local = 0;  // number of global planes owned by this rank
};

inline DecompZ decompose_z(const int nz_global, const int size, const int rank) {
    const int base = nz_global / size;
    const int rem = nz_global % size;
    const int nz_local = base + ((rank < rem) ? 1 : 0);
    const int z_start = rank * base + ((rank < rem) ? rank : rem);
    return DecompZ{z_start, nz_local};
}

__global__ void init_grid_kernel(Real* __restrict__ grid, const int nx, const int ny, const int nz_local, const int z_start_global) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    const int z_local = static_cast<int>(blockIdx.z) + 1;  // 1..nz_local (halo at 0)
    if (x >= nx || y >= ny || z_local > nz_local) return;

    const int z_global = z_start_global + (z_local - 1);
    const size_t gidx = static_cast<size_t>(z_global) * static_cast<size_t>(nx) * static_cast<size_t>(ny) + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
    grid[idx3d(x, y, z_local, nx, ny)] = static_cast<Real>(gidx % 19) * 1.0;
}

__global__ void stencil_kernel_zrange(const Real* __restrict__ in, Real* __restrict__ out, const int nx, const int ny, const int z_begin, const int z_end) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x) + 1;  // 1..nx-2
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y) + 1;  // 1..ny-2
    const int z = static_cast<int>(blockIdx.z) + z_begin;

    if (x >= nx - 1 || y >= ny - 1 || z > z_end) return;

    const size_t idx = idx3d(x, y, z, nx, ny);
    const Real center = in[idx];
    const Real left = in[idx3d(x - 1, y, z, nx, ny)];
    const Real right = in[idx3d(x + 1, y, z, nx, ny)];
    const Real front = in[idx3d(x, y - 1, z, nx, ny)];
    const Real back = in[idx3d(x, y + 1, z, nx, ny)];
    const Real bottom = in[idx3d(x, y, z - 1, nx, ny)];
    const Real top = in[idx3d(x, y, z + 1, nx, ny)];

    out[idx] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
}

__global__ void boundary_copy_kernel(const Real* __restrict__ in, Real* __restrict__ out, const int nx, const int ny, const int nz_local, const int z_start_global, const int nz_global) {
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    const int z_local = static_cast<int>(blockIdx.z) + 1;  // 1..nz_local
    if (x >= nx || y >= ny || z_local > nz_local) return;

    const int z_global = z_start_global + (z_local - 1);
    const bool is_boundary = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1) || (z_global == 0) || (z_global == nz_global - 1);
    if (!is_boundary) return;

    const size_t idx = idx3d(x, y, z_local, nx, ny);
    out[idx] = in[idx];
}

bool validateResult(const std::vector<Real>& grid) {
    // OpenMP-parallel sanity checks.
    bool ok = true;
    Real minVal = grid.empty() ? 0.0 : grid[0];
    Real maxVal = grid.empty() ? 0.0 : grid[0];

    #pragma omp parallel
    {
        bool local_ok = true;
        Real local_min = minVal;
        Real local_max = maxVal;

        #pragma omp for nowait
        for (size_t i = 0; i < grid.size(); ++i) {
            const Real v = grid[i];
            if (std::isnan(v) || std::isinf(v)) {
                local_ok = false;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }

        #pragma omp critical
        {
            ok = ok && local_ok;
            minVal = std::min(minVal, local_min);
            maxVal = std::max(maxVal, local_max);
        }
    }

    if (!ok) {
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

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    [[maybe_unused]] int omp_threads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        omp_threads = omp_get_num_threads();
    }

    size_t nx_u = 128;
    size_t ny_u = 0;
    size_t nz_u = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx_u = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny_u = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz_u = static_cast<size_t>(atoi(argv[++i]));
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

    if (ny_u == 0) ny_u = nx_u;
    if (nz_u == 0) nz_u = nx_u;

    const int nx = static_cast<int>(nx_u);
    const int ny = static_cast<int>(ny_u);
    const int nz_global = static_cast<int>(nz_u);

    if (nx <= 0 || ny <= 0 || nz_global <= 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    const DecompZ dec = decompose_z(nz_global, size, rank);
    const int nz_local = dec.nz_local;
    const int z_start = dec.z_start;

    if (nz_local <= 0) {
        mpi_abort("MPI decomposition resulted in empty slab (too many ranks for nz)", 4);
    }

    // Choose GPU per process (local-rank aware where possible).
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        mpi_abort("No CUDA devices found", 3);
    }
    int localRank = get_local_rank_fallback();
    if (localRank < 0) localRank = rank;
    const int dev = localRank % devCount;
    CUDA_CHECK(cudaSetDevice(dev));

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz_global);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t plane_elems = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t local_elems_with_halo = static_cast<size_t>(nz_local + 2) * plane_elems;  // 1 halo plane on each side

    Real* d_in = nullptr;
    Real* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, local_elems_with_halo * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_out, local_elems_with_halo * sizeof(Real)));

    CUDA_CHECK(cudaMemset(d_in, 0, local_elems_with_halo * sizeof(Real)));
    CUDA_CHECK(cudaMemset(d_out, 0, local_elems_with_halo * sizeof(Real)));

    cudaStream_t stream_compute{};
    cudaStream_t stream_comm{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm, cudaStreamNonBlocking));

    // Initialize local owned planes on GPU.
    {
        const dim3 block(32, 4, 1);
        const dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, static_cast<unsigned int>(nz_local));
        init_grid_kernel<<<grid, block, 0, stream_compute>>>(d_in, nx, ny, nz_local, z_start);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    }

    // Pinned host buffers for halo exchange (one Z-plane each).
    Real* h_send_prev = nullptr;
    Real* h_send_next = nullptr;
    Real* h_recv_prev = nullptr;
    Real* h_recv_next = nullptr;
    if (size > 1) {
        CUDA_CHECK(cudaMallocHost(&h_send_prev, plane_elems * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_send_next, plane_elems * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recv_prev, plane_elems * sizeof(Real)));
        CUDA_CHECK(cudaMallocHost(&h_recv_next, plane_elems * sizeof(Real)));
    }

    cudaEvent_t halo_ready{};
    cudaEvent_t iter_done{};
    CUDA_CHECK(cudaEventCreateWithFlags(&halo_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&iter_done, cudaEventDisableTiming));

    const int prev = (rank == 0) ? MPI_PROC_NULL : (rank - 1);
    const int next = (rank == size - 1) ? MPI_PROC_NULL : (rank + 1);
    constexpr int TAG_TO_PREV = 101;  // sends first plane to prev
    constexpr int TAG_TO_NEXT = 102;  // sends last plane to next

    auto exchange_halos = [&](Real* d_current) {
        if (size == 1) return;

        // Ensure previous iteration finished producing d_current before reading it for comm.
        CUDA_CHECK(cudaStreamWaitEvent(stream_comm, iter_done));

        MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};

        // Receive halo planes.
        MPI_Irecv(h_recv_prev, static_cast<int>(plane_elems), MPI_DOUBLE, prev, TAG_TO_NEXT, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(h_recv_next, static_cast<int>(plane_elems), MPI_DOUBLE, next, TAG_TO_PREV, MPI_COMM_WORLD, &reqs[1]);

        // Send our boundary planes (from device -> pinned host).
        const size_t pitch_xy = plane_elems;
        const Real* d_first = d_current + pitch_xy * 1;
        const Real* d_last = d_current + pitch_xy * static_cast<size_t>(nz_local);
        CUDA_CHECK(cudaMemcpyAsync(h_send_prev, d_first, plane_elems * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));
        CUDA_CHECK(cudaMemcpyAsync(h_send_next, d_last, plane_elems * sizeof(Real), cudaMemcpyDeviceToHost, stream_comm));

        // Wait only for D->H copies so MPI can use the host buffers.
        CUDA_CHECK(cudaStreamSynchronize(stream_comm));

        MPI_Isend(h_send_prev, static_cast<int>(plane_elems), MPI_DOUBLE, prev, TAG_TO_PREV, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(h_send_next, static_cast<int>(plane_elems), MPI_DOUBLE, next, TAG_TO_NEXT, MPI_COMM_WORLD, &reqs[3]);

        // Wait for halo data (and sends, so buffers can be reused next iteration).
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Copy received halos back to device halo planes.
        if (prev != MPI_PROC_NULL) {
            Real* d_halo0 = d_current + pitch_xy * 0;
            CUDA_CHECK(cudaMemcpyAsync(d_halo0, h_recv_prev, plane_elems * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
        }
        if (next != MPI_PROC_NULL) {
            Real* d_haloN = d_current + pitch_xy * static_cast<size_t>(nz_local + 1);
            CUDA_CHECK(cudaMemcpyAsync(d_haloN, h_recv_next, plane_elems * sizeof(Real), cudaMemcpyHostToDevice, stream_comm));
        }

        CUDA_CHECK(cudaEventRecord(halo_ready, stream_comm));
    };

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Mark initial iteration input as ready.
    CUDA_CHECK(cudaEventRecord(iter_done, stream_compute));

    for (int iter = 0; iter < iterations; ++iter) {
        // Start halo exchange for current input (overlapped with interior compute).
        exchange_halos(d_in);

        const dim3 block(32, 4, 1);
        const dim3 grid_xy((std::max(nx - 2, 0) + block.x - 1) / block.x, (std::max(ny - 2, 0) + block.y - 1) / block.y, 1);

        // Compute interior z-planes that don't require freshly exchanged halos.
        if (nx > 2 && ny > 2 && nz_local > 3) {
            const int z_begin = 2;
            const int z_end = nz_local - 1;
            const dim3 grid(grid_xy.x, grid_xy.y, static_cast<unsigned int>(z_end - z_begin + 1));
            stencil_kernel_zrange<<<grid, block, 0, stream_compute>>>(d_in, d_out, nx, ny, z_begin, z_end);
            CUDA_CHECK(cudaGetLastError());
        }

        // Wait for halos before computing boundary-owned planes.
        if (size > 1) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, halo_ready));
        }

        if (nx > 2 && ny > 2) {
            // First owned plane (z_local=1) if not a global boundary.
            if (z_start > 0 && nz_local >= 1) {
                const int z_begin = 1;
                const int z_end = 1;
                const dim3 grid(grid_xy.x, grid_xy.y, 1);
                stencil_kernel_zrange<<<grid, block, 0, stream_compute>>>(d_in, d_out, nx, ny, z_begin, z_end);
                CUDA_CHECK(cudaGetLastError());
            }
            // Last owned plane (z_local=nz_local) if not a global boundary.
            const int z_end_global = z_start + nz_local - 1;
            if (z_end_global < nz_global - 1 && nz_local >= 2) {
                const int z_begin = nz_local;
                const int z_end = nz_local;
                const dim3 grid(grid_xy.x, grid_xy.y, 1);
                stencil_kernel_zrange<<<grid, block, 0, stream_compute>>>(d_in, d_out, nx, ny, z_begin, z_end);
                CUDA_CHECK(cudaGetLastError());
            } else if (z_end_global < nz_global - 1 && nz_local == 1 && z_start > 0) {
                // Single-plane slab that is not a global boundary: compute it once.
                const dim3 grid(grid_xy.x, grid_xy.y, 1);
                stencil_kernel_zrange<<<grid, block, 0, stream_compute>>>(d_in, d_out, nx, ny, 1, 1);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // Preserve original semantics: boundaries are copied from input to output each iteration.
        {
            const dim3 b2(32, 4, 1);
            const dim3 g2((nx + b2.x - 1) / b2.x, (ny + b2.y - 1) / b2.y, static_cast<unsigned int>(nz_local));
            boundary_copy_kernel<<<g2, b2, 0, stream_compute>>>(d_in, d_out, nx, ny, nz_local, z_start, nz_global);
            CUDA_CHECK(cudaGetLastError());
        }

        // Mark iteration output ready and swap buffers.
        CUDA_CHECK(cudaEventRecord(iter_done, stream_compute));
        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const double local_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_ms);
        const double cellUpdates = static_cast<double>(std::max(nx - 2, 0)) * static_cast<double>(std::max(ny - 2, 0)) * static_cast<double>(std::max(nz_global - 2, 0)) * static_cast<double>(iterations);
        const double mcups = (max_ms > 0.0) ? (cellUpdates / (max_ms / 1000.0) / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Optional gather for validation / printing.
    int valid_int = 1;
    if (validate || printResults) {
        std::vector<Real> local_host(static_cast<size_t>(nz_local) * plane_elems);
        CUDA_CHECK(cudaMemcpy(local_host.data(), d_in + plane_elems, local_host.size() * sizeof(Real), cudaMemcpyDeviceToHost));

        std::vector<int> recvcounts;
        std::vector<int> displs;
        std::vector<Real> global;
        if (rank == 0) {
            recvcounts.resize(static_cast<size_t>(size));
            displs.resize(static_cast<size_t>(size));
            for (int r = 0; r < size; ++r) {
                const DecompZ d = decompose_z(nz_global, size, r);
                recvcounts[static_cast<size_t>(r)] = d.nz_local * static_cast<int>(plane_elems);
            }
            displs[0] = 0;
            for (int r = 1; r < size; ++r) {
                displs[static_cast<size_t>(r)] = displs[static_cast<size_t>(r - 1)] + recvcounts[static_cast<size_t>(r - 1)];
            }
            global.resize(static_cast<size_t>(nz_global) * plane_elems);
        }

        MPI_Gatherv(local_host.data(), static_cast<int>(local_host.size()), MPI_DOUBLE,
                    (rank == 0) ? global.data() : nullptr,
                    (rank == 0) ? recvcounts.data() : nullptr,
                    (rank == 0) ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(global, "Grid");
        }

        if (rank == 0 && validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(global);
            if (ok) {
                printf("Validation: PASSED\n");
                valid_int = 1;
            } else {
                printf("Validation: FAILED\n");
                valid_int = 0;
            }
        }
        if (validate) {
            MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    if (h_send_prev) CUDA_CHECK(cudaFreeHost(h_send_prev));
    if (h_send_next) CUDA_CHECK(cudaFreeHost(h_send_next));
    if (h_recv_prev) CUDA_CHECK(cudaFreeHost(h_recv_prev));
    if (h_recv_next) CUDA_CHECK(cudaFreeHost(h_recv_next));

    CUDA_CHECK(cudaEventDestroy(halo_ready));
    CUDA_CHECK(cudaEventDestroy(iter_done));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_comm));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return (validate && !valid_int) ? 1 : 0;
}
