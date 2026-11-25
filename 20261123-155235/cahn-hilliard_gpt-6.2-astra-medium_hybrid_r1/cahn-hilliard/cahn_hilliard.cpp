#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include "../common/results_output.hpp"

static int rank;
[[noreturn]] static void fail(const char* message) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

// Each slab has one ghost plane at either end. Physical boundaries use
// the center plane directly, so their ghost storage is never read.
template<bool potential>
__global__ void stencil(const double* __restrict__ field,
                        const double* __restrict__ concentration,
                        double* __restrict__ output, int nx, int ny, size_t plane,
                        int depth, bool lower, bool upper, int first, int count,
                        bool edges) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= static_cast<unsigned>(nx)) return;
    for (size_t row = blockIdx.z; row < static_cast<unsigned>(count); row += gridDim.z) {
        const int z = edges && row ? depth : first + static_cast<int>(row);
        for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
             y < static_cast<unsigned>(ny); y += size_t(blockDim.y) * gridDim.y) {
            const size_t i = size_t(z) * plane + y * nx + x;
            const double center = field[i];
            const double twice = 2.0 * center;
            const double xx = (field[i + (x + 1 < static_cast<unsigned>(nx))] +
                               field[i - (x > 0)]) - twice;
            const double yy = (field[i + (y + 1 < static_cast<unsigned>(ny) ? nx : 0)] +
                               field[i - (y > 0 ? nx : 0)]) - twice;
            const double zz = (field[i + (z < depth || upper ? plane : 0)] +
                               field[i - (z > 1 || lower ? plane : 0)]) - twice;
            const double lap = xx + yy + zz;  // Unit grid spacing in each dimension.
            if constexpr (potential) {
                constexpr double e = 2.0 / 9.0;
                double interaction = fma(center + 1.0, -e, __dmul_rn(center - 1.0, -e));
                interaction = fma(-twice, e, interaction);
                const double bulk = fma(center * center, center,
                                        fma(4.5, interaction, __dmul_rn(3.0, center)));
                output[i] = fma(-0.5, lap, bulk);
            } else {
                output[i] = fma(0.01, lap, concentration[i]);
            }
        }
    }
}

struct Slab {
    int nx, ny, depth, peers;
    size_t plane;
    bool direct = false;
    cudaStream_t compute{}, transfer{};
    cudaEvent_t ready{};
    double* staging = nullptr;
    std::vector<MPI_Request> requests;

    template<bool potential>
    void apply(double* field, const double* concentration, double* out) {
        const bool lower = rank > 0, upper = rank + 1 < peers;
        auto launch = [&](int first, int count, bool edges) {
            if (count <= 0) return;
            const unsigned width = nx >= 128 ? 128 : nx >= 32 ? 32 : std::min(nx, 16);
            const dim3 threads(width, std::min<unsigned>(128 / width, ny));
            const dim3 blocks((unsigned(nx) + width - 1) / width,
                              std::min((unsigned(ny) + threads.y - 1) / threads.y, 65535u),
                              std::min(count, 65535));
            stencil<potential><<<blocks, threads, 0, compute>>>(field, concentration, out,
                nx, ny, plane, depth, lower, upper, first, count, edges);
            check(cudaGetLastError());
        };
        if (peers == 1) {
            launch(1, depth, false);
            return;
        }
        // Publish the input before sending its faces. The interior reads
        // only owned planes and can run throughout the exchange.
        check(cudaEventRecord(ready, compute));
        check(cudaStreamWaitEvent(transfer, ready));
        double* send[2] = {field + plane, field + size_t(depth) * plane};
        double* recv[2] = {field, field + (size_t(depth) + 1) * plane};
        const int peer[2] = {lower ? rank - 1 : MPI_PROC_NULL, upper ? rank + 1 : MPI_PROC_NULL};
        if (!direct) {
            for (int d = 0; d < 2; ++d) {
                if (peer[d] == MPI_PROC_NULL) continue;
                double* host = staging + d * plane;
                check(cudaMemcpyAsync(host, send[d], plane * sizeof(double),
                                      cudaMemcpyDeviceToHost, transfer));
                send[d] = host;
                recv[d] = host + 2 * plane;
            }
        }
        const int interior = depth - int(lower) - int(upper);
        launch(1 + int(lower), interior, false);
        check(cudaStreamSynchronize(transfer));
        requests.clear();
        for (int d = 0; d < 2; ++d) {
            if (peer[d] == MPI_PROC_NULL) continue;
            for (size_t offset = 0; offset < plane; offset += INT_MAX) {
                int n = std::min<size_t>(INT_MAX, plane - offset);
                MPI_Irecv(recv[d] + offset, n, MPI_DOUBLE, peer[d], 1-d, MPI_COMM_WORLD,
                          &requests.emplace_back());
                MPI_Isend(send[d] + offset, n, MPI_DOUBLE, peer[d], d, MPI_COMM_WORLD,
                          &requests.emplace_back());
            }
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        if (!direct) {
            for (int d = 0; d < 2; ++d) {
                if (peer[d] == MPI_PROC_NULL) continue;
                double* ghost = d ? field + (size_t(depth) + 1) * plane : field;
                check(cudaMemcpyAsync(ghost, recv[d], plane * sizeof(double),
                                      cudaMemcpyHostToDevice, transfer));
            }
            check(cudaEventRecord(ready, transfer));
            check(cudaStreamWaitEvent(compute, ready));
        }
        launch(lower ? 1 : depth, lower && upper && depth > 1 ? 2 : 1, true);
    }
};

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx > limit / ny || nx * ny > limit / (nz + 2))
        fail("Invalid grid dimensions");

    const int peers = std::min<size_t>(ranks, nz);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < peers ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= peers) {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm node;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) fail("A CUDA device is required");
    check(cudaSetDevice(localRank % devices));

    const size_t plane = nx * ny, volume = plane * nz;
    auto startPlane = [&](int r) {
        return (nz / peers) * r + std::min<size_t>(r, nz % peers);
    };
    const int depth = startPlane(rank + 1) - startPlane(rank);
    const size_t offset = startPlane(rank) * plane, cells = size_t(depth) * plane;
    Slab slab{};
    slab.nx = static_cast<int>(nx);
    slab.ny = static_cast<int>(ny);
    slab.depth = depth;
    slab.peers = peers;
    slab.plane = plane;
    check(cudaStreamCreateWithFlags(&slab.compute, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&slab.transfer, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&slab.ready, cudaEventDisableTiming));
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    slab.direct = MPIX_Query_cuda_support();
#endif
    if (peers > 1 && !slab.direct) {
        if (plane > limit / 4) fail("Boundary planes exceed addressable memory");
        check(cudaMallocHost(&slab.staging, 4 * plane * sizeof(double)));
    }
    const size_t bytes = (cells + 2 * plane) * sizeof(double);
    double *cold, *cnew, *mu;
    check(cudaMalloc(&cold, bytes));
    check(cudaMalloc(&cnew, bytes));
    check(cudaMalloc(&mu, bytes));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(cells);
    const int threads = std::min<size_t>(omp_get_max_threads(), (cells + 8191) / 8192);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t i = 0; i < cells; ++i) {
        const double pseudo = (((offset + i + 1) * 1299709) % volume) / double(volume);
        host[i] = -1.0 + 2.0 * pseudo;
    }
    check(cudaMemcpyAsync(cold + plane, host.data(), cells * sizeof(double),
                          cudaMemcpyHostToDevice, slab.compute));
    check(cudaStreamSynchronize(slab.compute));
    if (!validate && !printResults) std::vector<double>().swap(host);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(active);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        slab.apply<true>(cold, nullptr, mu);
        slab.apply<false>(mu, cold, cnew);
        std::swap(cold, cnew);
    }
    check(cudaStreamSynchronize(slab.compute));
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / seconds / 1e6);
    }
    if (validate || printResults)
        check(cudaMemcpy(host.data(), cold + plane, cells * sizeof(double), cudaMemcpyDeviceToHost));

    if (printResults) {
        // Preserve the global order for the external hash and Kahan sum.
        // Chunk transfers so total sizes and displacements can exceed INT_MAX.
        if (rank == 0) {
            host.resize(volume);
            for (int r = 1; r < peers; ++r) {
                const size_t end = startPlane(r + 1) * plane;
                for (size_t i = startPlane(r) * plane; i < end; i += INT_MAX)
                    MPI_Recv(host.data() + i, std::min<size_t>(INT_MAX, end-i),
                             MPI_DOUBLE, r, 2, active, MPI_STATUS_IGNORE);
            }
            print_results(host, "Concentration");
        } else {
            for (size_t i = 0; i < cells; i += INT_MAX)
                MPI_Send(host.data() + i, std::min<size_t>(INT_MAX, cells-i), MPI_DOUBLE, 0, 2, active);
        }
    }
    int valid = 1;
    if (validate) {
        double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
        int invalid = 0;
        #pragma omp parallel for schedule(static) num_threads(threads) reduction(min:minimum) reduction(max:maximum) reduction(|:invalid)
        for (size_t i = 0; i < cells; ++i) {
            invalid |= !std::isfinite(host[i]);
            minimum = std::min(minimum, host[i]);
            maximum = std::max(maximum, host[i]);
        }
        // Negating the minimum allows one collective to compute all three values.
        double local[] = {-minimum, maximum, double(invalid)}, global[3];
        MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX, active);
        valid = !global[2] && -global[0] >= -10.0 && global[1] <= 10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (global[2]) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", -global[0], global[1]);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    check(cudaFree(cold));
    check(cudaFree(cnew));
    check(cudaFree(mu));
    if (slab.staging) check(cudaFreeHost(slab.staging));
    check(cudaEventDestroy(slab.ready));
    check(cudaStreamDestroy(slab.transfer));
    check(cudaStreamDestroy(slab.compute));
    MPI_Comm_free(&active);
    MPI_Finalize();
    return !valid;
}
