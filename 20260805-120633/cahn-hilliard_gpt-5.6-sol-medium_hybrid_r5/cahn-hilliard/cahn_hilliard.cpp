#include <mpi.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t err_ = (call);                                                   \
    if (err_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(err_));                                  \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

struct Slab {
    size_t z0;
    size_t nz;
};

struct HaloTransport {
    bool device_direct = false;
    double* host = nullptr; // send-low, send-high, recv-low, recv-high
    size_t plane = 0;
};

static Slab decompose(size_t nz, int rank, int ranks) {
    const size_t q = nz / static_cast<size_t>(ranks);
    const size_t r = nz % static_cast<size_t>(ranks);
    const size_t local = q + (static_cast<size_t>(rank) < r ? 1 : 0);
    const size_t begin = static_cast<size_t>(rank) * q +
                         std::min(static_cast<size_t>(rank), r);
    return {begin, local};
}

__device__ __forceinline__ size_t didx(size_t x, size_t y, size_t z,
                                       size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

// z is local and includes two ghost planes. X/Y remain local clamped boundaries;
// Z clamping is represented by copied ghost planes on the two global end ranks.
__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                             size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t plane,
                                             double idx2, double idy2, double idz2) {
    const size_t xm = x == 0 ? x : x - 1;
    const size_t xp = x + 1 == nx ? x : x + 1;
    const size_t ym = y == 0 ? y : y - 1;
    const size_t yp = y + 1 == ny ? y : y + 1;
    const size_t i = didx(x, y, z, nx, plane);
    const double center = a[i];
    return (a[didx(xp, y, z, nx, plane)] + a[didx(xm, y, z, nx, plane)] - 2.0 * center) * idx2 +
           (a[didx(x, yp, z, nx, plane)] + a[didx(x, ym, z, nx, plane)] - 2.0 * center) * idy2 +
           (a[i + plane] + a[i - plane] - 2.0 * center) * idz2;
}

__global__ void chemical_potential(const double* __restrict__ c,
                                   double* __restrict__ mu,
                                   size_t nx, size_t ny, size_t plane,
                                   size_t z_begin, size_t z_end,
                                   double idx2, double idy2, double idz2,
                                   double gamma, double e_AA, double e_BB,
                                   double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z_begin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z_end) return;
    const size_t i = didx(x, y, z, nx, plane);
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
            3.0 * cv + cv * cv * cv -
            gamma * laplacian(c, x, y, z, nx, ny, plane, idx2, idy2, idz2);
}

__global__ void update_concentration(double* __restrict__ cnew,
                                     const double* __restrict__ cold,
                                     const double* __restrict__ mu,
                                     size_t nx, size_t ny, size_t plane,
                                     size_t z_begin, size_t z_end,
                                     double factor, double idx2,
                                     double idy2, double idz2) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z_begin + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z_end) return;
    const size_t i = didx(x, y, z, nx, plane);
    cnew[i] = cold[i] + factor *
        laplacian(mu, x, y, z, nx, ny, plane, idx2, idy2, idz2);
}

static void launch_mu(const double* c, double* mu, size_t nx, size_t ny,
                      size_t plane, size_t zb, size_t ze, double idx2,
                      double idy2, double idz2, double gamma, double e_AA,
                      double e_BB, double e_AB, cudaStream_t stream) {
    if (zb >= ze) return;
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    ze - zb);
    chemical_potential<<<grid, block, 0, stream>>>(c, mu, nx, ny, plane, zb, ze,
        idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

static void launch_update(double* cnew, const double* cold, const double* mu,
                          size_t nx, size_t ny, size_t plane, size_t zb,
                          size_t ze, double factor, double idx2, double idy2,
                          double idz2, cudaStream_t stream) {
    if (zb >= ze) return;
    const dim3 block(32, 4, 1);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    ze - zb);
    update_concentration<<<grid, block, 0, stream>>>(cnew, cold, mu, nx, ny,
        plane, zb, ze, factor, idx2, idy2, idz2);
    CUDA_CHECK(cudaGetLastError());
}

static void begin_halo(double* a, size_t local_nz, int rank, int ranks,
                       const HaloTransport& transport, MPI_Request req[4]) {
    const size_t plane = transport.plane;
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int count = static_cast<int>(plane);
    if (prev == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(a, a + plane, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    if (next == MPI_PROC_NULL)
        CUDA_CHECK(cudaMemcpy(a + (local_nz + 1) * plane, a + local_nz * plane,
                              plane * sizeof(double), cudaMemcpyDeviceToDevice));
    if (!transport.device_direct) {
        CUDA_CHECK(cudaMemcpy(transport.host, a + plane, plane * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(transport.host + plane, a + local_nz * plane,
                              plane * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double* recv_low = transport.device_direct ? a : transport.host + 2 * plane;
    double* recv_high = transport.device_direct ? a + (local_nz + 1) * plane
                                                : transport.host + 3 * plane;
    double* send_low = transport.device_direct ? a + plane : transport.host;
    double* send_high = transport.device_direct ? a + local_nz * plane
                                                : transport.host + plane;
    MPI_Irecv(recv_low, count, MPI_DOUBLE, prev, 101, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(recv_high, count, MPI_DOUBLE, next, 100, MPI_COMM_WORLD, &req[1]);
    MPI_Isend(send_low, count, MPI_DOUBLE, prev, 100, MPI_COMM_WORLD, &req[2]);
    MPI_Isend(send_high, count, MPI_DOUBLE, next, 101, MPI_COMM_WORLD, &req[3]);
}

static void finish_halo(double* a, size_t local_nz, int rank, int ranks,
                        const HaloTransport& transport, MPI_Request req[4]) {
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
    if (!transport.device_direct) {
        const size_t bytes = transport.plane * sizeof(double);
        if (rank != 0)
            CUDA_CHECK(cudaMemcpy(a, transport.host + 2 * transport.plane, bytes,
                                  cudaMemcpyHostToDevice));
        if (rank + 1 != ranks)
            CUDA_CHECK(cudaMemcpy(a + (local_nz + 1) * transport.plane,
                                  transport.host + 3 * transport.plane, bytes,
                                  cudaMemcpyHostToDevice));
    }
}

static void print_usage(const char* prog) {
    std::printf("Usage: %s [options]\n", prog);
    std::printf("  -x <num>  Grid size in X (default: 64)\n");
    std::printf("  -y <num>  Grid size in Y (default: same as X)\n");
    std::printf("  -z <num>  Grid size in Z (default: same as X)\n");
    std::printf("  -i <num>  Number of time steps (default: 20)\n");
    std::printf("  -v        Enable validation\n");
    std::printf("  -r        Print results for external validation\n");
    std::printf("  -h        Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_requested = false, bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_requested = true;
        else if (!std::strcmp(argv[i], "-h")) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else bad_args = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (bad_args || nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        static_cast<size_t>(ranks) > nz || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) {
            std::fprintf(stderr, "Invalid dimensions/iterations, too many ranks, or MPI plane count overflow\n");
            print_usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Assign ranks sharing a node round-robin to its visible accelerators.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA accelerator is visible\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    bool cuda_aware_mpi = false;
    #if defined(OPEN_MPI)
    cuda_aware_mpi = MPIX_Query_cuda_support() != 0;
    #endif

    const Slab slab = decompose(nz, rank, ranks);
    const size_t plane = nx * ny;
    const size_t local_elements = slab.nz * plane;
    const size_t allocated = (slab.nz + 2) * plane;
    if (slab.nz > (std::numeric_limits<size_t>::max() / plane) - 2) {
        if (rank == 0) std::fprintf(stderr, "Grid allocation overflows size_t\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<double> initial(local_elements);
    const size_t global_elements = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (long long li = 0; li < static_cast<long long>(local_elements); ++li) {
        const size_t global_id = slab.z0 * plane + static_cast<size_t>(li);
        const double pseudo = (((global_id + 1) * size_t{1299709}) % global_elements) /
                              static_cast<double>(global_elements);
        initial[static_cast<size_t>(li)] = -1.0 + 2.0 * pseudo;
    }

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, allocated * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew, allocated * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu, allocated * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, initial.data(), local_elements * sizeof(double),
                          cudaMemcpyHostToDevice));
    initial.clear();
    initial.shrink_to_fit();
    cudaStream_t interior_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&interior_stream, cudaStreamNonBlocking));
    HaloTransport transport;
    transport.device_direct = cuda_aware_mpi;
    transport.plane = plane;
    if (!transport.device_direct)
        CUDA_CHECK(cudaHostAlloc(&transport.host, 4 * plane * sizeof(double),
                                 cudaHostAllocPortable));

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Parallelism: %d MPI ranks, %d OpenMP threads/rank, CUDA (%s MPI halos)\n",
                    ranks, omp_get_max_threads(),
                    cuda_aware_mpi ? "device-direct" : "pinned-host staged");
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Running Cahn-Hilliard simulation...\n");
    }

    constexpr double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5, D = 1.0;
    const double idx2 = 1.0 / (dx * dx), idy2 = 1.0 / (dy * dy), idz2 = 1.0 / (dz * dz);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        MPI_Request req[4];
        begin_halo(cold, slab.nz, rank, ranks, transport, req);
        // Work independent of incoming ghosts overlaps the device-direct MPI transfer.
        launch_mu(cold, mu, nx, ny, plane, 2, slab.nz, idx2, idy2, idz2,
                  gamma, e_AA, e_BB, e_AB, interior_stream);
        finish_halo(cold, slab.nz, rank, ranks, transport, req);
        launch_mu(cold, mu, nx, ny, plane, 1, std::min<size_t>(2, slab.nz + 1),
                  idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB, nullptr);
        if (slab.nz > 1)
            launch_mu(cold, mu, nx, ny, plane, slab.nz, slab.nz + 1,
                      idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        begin_halo(mu, slab.nz, rank, ranks, transport, req);
        launch_update(cnew, cold, mu, nx, ny, plane, 2, slab.nz, dt * D,
                      idx2, idy2, idz2, interior_stream);
        finish_halo(mu, slab.nz, rank, ranks, transport, req);
        launch_update(cnew, cold, mu, nx, ny, plane, 1, std::min<size_t>(2, slab.nz + 1),
                      dt * D, idx2, idy2, idz2, nullptr);
        if (slab.nz > 1)
            launch_update(cnew, cold, mu, nx, ny, plane, slab.nz, slab.nz + 1,
                          dt * D, idx2, idy2, idz2, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double updates = static_cast<double>(global_elements) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    elapsed > 0.0 ? updates / elapsed / 1.0e6 : 0.0);
    }

    int exit_code = 0;
    if (validate || print_results_requested) {
        std::vector<double> local(local_elements);
        CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, local_elements * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::vector<int> counts(rank == 0 ? ranks : 0), displs(rank == 0 ? ranks : 0);
        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const Slab s = decompose(nz, r, ranks);
                counts[r] = static_cast<int>(s.nz * plane);
                displs[r] = static_cast<int>(s.z0 * plane);
            }
        }
        if (local_elements > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            global_elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI int counts\n");
            exit_code = 1;
        } else {
            std::vector<double> global(rank == 0 ? global_elements : 0);
            MPI_Gatherv(local.data(), static_cast<int>(local_elements), MPI_DOUBLE,
                        rank == 0 ? global.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (rank == 0 && print_results_requested) print_results(global, "Concentration");
            if (rank == 0 && validate) {
                int bad = 0;
                double min_value = std::numeric_limits<double>::infinity();
                double max_value = -std::numeric_limits<double>::infinity();
                #pragma omp parallel for reduction(+:bad) reduction(min:min_value) reduction(max:max_value)
                for (long long i = 0; i < static_cast<long long>(global_elements); ++i) {
                    const double v = global[static_cast<size_t>(i)];
                    bad += !std::isfinite(v);
                    min_value = std::min(min_value, v);
                    max_value = std::max(max_value, v);
                }
                std::printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
                if (bad || max_value > 10.0 || min_value < -10.0) {
                    std::printf("Validation: FAILED\n");
                    exit_code = 1;
                } else std::printf("Validation: PASSED\n");
            }
        }
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaStreamDestroy(interior_stream));
    if (transport.host) CUDA_CHECK(cudaFreeHost(transport.host));
    CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFree(cnew));
    CUDA_CHECK(cudaFree(cold));
    MPI_Finalize();
    return exit_code;
}
