#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) checkCuda((call), #call)

struct Slab {
    int first, end;  // Half-open global z range
};

Slab partition(int nz, int ranks, int rank) {
    int length = nz / ranks;
    int extra = nz % ranks;
    int first = rank * length + std::min(rank, extra);
    return {first, first + length + (rank < extra)};
}

// Both stencil phases have unit grid spacing. Only global faces clamp.
template<bool potential>
__global__ void stencil(const double* __restrict__ field, const double* concentration,
                        double* __restrict__ out, int nx, int ny, int nz, int origin,
                        int first, int end) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    size_t plane = size_t(nx) * ny;
    // March through consecutive planes in each lane. Keep the axial
    // neighbors in registers while the warp loads contiguous rows in x.
    constexpr int planesPerBlock = 4;
    for (long long slab = first + (long long)blockIdx.z * planesPerBlock; slab < end;
         slab += gridDim.z * planesPerBlock) {
        for (long long y = blockIdx.y * blockDim.y + threadIdx.y; y < ny;
             y += gridDim.y * blockDim.y) {
            size_t i = size_t(slab - origin) * plane + y * nx + x;
            double previous = field[i - (slab > 0 ? plane : 0)];
            double v = field[i];
            for (int k = 0; k < planesPerBlock && slab + k < end; ++k, i += plane) {
                double following = field[i + (slab + k + 1 < nz ? plane : 0)];
                double xx = (field[i + (x + 1 < nx)] + field[i - (x > 0)]) - 2.0 * v;
                double yy = (field[i + (y + 1 < ny ? nx : 0)] +
                             field[i - (y > 0 ? nx : 0)]) - 2.0 * v;
                double zz = (following + previous) - 2.0 * v;
                double lap = xx + yy + zz;
                if constexpr (potential) {
                    constexpr double like = -2.0 / 9.0, unlike = 2.0 / 9.0;
                    double energy = fma(v + 1.0, like, (v - 1.0) * like);
                    energy = fma(-2.0 * v, unlike, energy);
                    double bulk = fma(v * v, v, fma(4.5, energy, 3.0 * v));
                    out[i] = fma(-0.5, lap, bulk);
                } else {
                    out[i] = fma(0.01, lap, concentration[i]);
                }
                previous = v;
                v = following;
            }
        }
    }
}

struct Grid {
    int nx, ny, nz, origin;
    template<bool potential>
    void launch(Slab range, const double* c, double* mu, double* next, cudaStream_t stream) {
        if (range.end <= range.first) return;
        dim3 threads(32, 4);
        dim3 blocks((nx - 1) / 32 + 1, std::min(65535, (ny - 1) / 4 + 1),
                    std::min(65535, (range.end - range.first - 1) / 4 + 1));
        stencil<potential><<<blocks, threads, 0, stream>>>(
            potential ? c : mu, c, potential ? mu : next, nx, ny, nz, origin,
            range.first, range.end);
        CUDA(cudaGetLastError());
    }
};

struct Transfer {
    int peer, count;
    size_t offset, staging;
};

struct Halos {
    std::vector<Transfer> sends, receives;
    std::vector<MPI_Request> requests;
    size_t storage = 0;
    double* host = nullptr;

    // Intersect the owned planes with each peer's extended slab. This also
    // handles one-plane ranks whose second halo belongs to a second neighbor.
    void add(std::vector<Transfer>& list, Slab source, Slab target, int peer,
             int origin, size_t plane) {
        int first = std::max(source.first, target.first - 2);
        int end = std::min<long long>(source.end, (long long)target.end + 2);
        if (end <= first) return;
        size_t count = size_t(end - first) * plane;
        size_t offset = size_t(first - origin) * plane;
        while (count) {
            int chunk = std::min<size_t>(count, INT_MAX);
            list.push_back({peer, chunk, offset, storage});
            count -= chunk;
            offset += chunk;
            storage += chunk;
        }
    }

    void prepare(int rank, int ranks, Grid grid, Slab owned) {
        size_t plane = size_t(grid.nx) * grid.ny;
        for (int peer = std::max(0, rank - 2); peer < ranks && peer - rank <= 2; ++peer) {
            if (peer == rank) continue;
            Slab other = partition(grid.nz, ranks, peer);
            add(sends, owned, other, peer, grid.origin, plane);
            add(receives, other, owned, peer, grid.origin, plane);
        }
        requests.resize(sends.size() + receives.size());
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        if (MPIX_Query_cuda_support()) return;
#endif
        if (storage) CUDA(cudaMallocHost(&host, storage * sizeof(double)));
    }

    void exchange(double* c, cudaStream_t transfer) {
        int n = 0;
        for (auto h : receives) {
            MPI_Irecv(host ? host + h.staging : c + h.offset, h.count, MPI_DOUBLE,
                      h.peer, 0, MPI_COMM_WORLD, &requests[n++]);
        }
        if (host) {
            for (auto h : sends)
                CUDA(cudaMemcpyAsync(host + h.staging, c + h.offset, h.count * sizeof(double),
                                     cudaMemcpyDeviceToHost, transfer));
            CUDA(cudaStreamSynchronize(transfer));
        }
        for (auto h : sends) {
            MPI_Isend(host ? host + h.staging : c + h.offset, h.count, MPI_DOUBLE,
                      h.peer, 0, MPI_COMM_WORLD, &requests[n++]);
        }
        MPI_Waitall(n, requests.data(), MPI_STATUSES_IGNORE);
        if (host) {
            for (auto h : receives)
                CUDA(cudaMemcpyAsync(c + h.offset, host + h.staging, h.count * sizeof(double),
                                     cudaMemcpyHostToDevice, transfer));
            CUDA(cudaStreamSynchronize(transfer));
        }
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

int run(int argc, char** argv, int rank, int ranks) {
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
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX ||
        nx > maxCells / ny || nx * ny > maxCells / nz) {
        if (!rank) fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    ranks = std::min<int>(ranks, nz);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= ranks) return 0;

    MPI_Comm node;
    MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", rank);
        MPI_Abort(active, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));

    Slab owned = partition(nz, ranks, rank);
    Grid grid{int(nx), int(ny), int(nz), std::max(0, owned.first - 2)};
    int extendedEnd = std::min<size_t>(nz, size_t(owned.end) + 2);
    size_t plane = nx * ny, volume = plane * nz;
    size_t cells = plane * (extendedEnd - grid.origin);
    size_t localCells = plane * (owned.end - owned.first);
    size_t localOffset = plane * (owned.first - grid.origin);

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    double *c, *mu, *next;
    CUDA(cudaMalloc(&c, cells * sizeof(double)));
    CUDA(cudaMalloc(&mu, cells * sizeof(double)));
    CUDA(cudaMalloc(&next, cells * sizeof(double)));
    {
        std::vector<double> initial(cells);
        size_t base = plane * grid.origin;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cells; ++i) {
            double pseudo = ((base + i + 1) * 1299709 % volume) / double(volume);
            initial[i] = -1.0 + 2.0 * pseudo;
        }
        CUDA(cudaMemcpy(c, initial.data(), cells * sizeof(double), cudaMemcpyHostToDevice));
    }
    cudaStream_t compute, transfer;
    CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    Halos halos;
    halos.prepare(rank, ranks, grid, owned);
    Slab potentials{std::max(0, owned.first - 1), owned.end + (owned.end < int(nz))};
    // A potential one plane away from a remote face is independent of the
    // exchange. An update two planes away can also run before halos arrive.
    auto independent = [&](int radius) {
        int length = owned.end - owned.first;
        int first = owned.first + (rank > 0 ? std::min(radius, length) : 0);
        int end = owned.end - (rank + 1 < ranks ? std::min(radius, length) : 0);
        return Slab{first, std::max(first, end)};
    };
    Slab innerMu = independent(1);
    Slab innerC = independent(2);

    cudaFuncAttributes attributes;
    CUDA(cudaFuncGetAttributes(&attributes, stencil<true>));
    CUDA(cudaFuncGetAttributes(&attributes, stencil<false>));
    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(active);
    double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        if (step && ranks > 1) {
            CUDA(cudaStreamSynchronize(compute));
            grid.launch<true>(innerMu, c, mu, next, compute);
            grid.launch<false>(innerC, c, mu, next, compute);
            halos.exchange(c, transfer);
            grid.launch<true>({potentials.first, innerMu.first}, c, mu, next, compute);
            grid.launch<true>({innerMu.end, potentials.end}, c, mu, next, compute);
            grid.launch<false>({owned.first, innerC.first}, c, mu, next, compute);
            grid.launch<false>({innerC.end, owned.end}, c, mu, next, compute);
        } else {
            grid.launch<true>(potentials, c, mu, next, compute);
            grid.launch<false>(owned, c, mu, next, compute);
        }
        std::swap(c, next);
    }
    CUDA(cudaStreamSynchronize(compute));
    double localTime = MPI_Wtime() - start, elapsed = 0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    if (!rank) {
        printf("Computation time: %lld ms\n", (long long)(elapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / elapsed / 1e6);
    }

    std::vector<double> local;
    if (validate || printResults) {
        local.resize(localCells);
        CUDA(cudaMemcpy(local.data(), c + localOffset, localCells * sizeof(double),
                        cudaMemcpyDeviceToHost));
    }
    CUDA(cudaFree(c));
    CUDA(cudaFree(next));
    CUDA(cudaFree(mu));
    if (halos.host) CUDA(cudaFreeHost(halos.host));
    CUDA(cudaStreamDestroy(compute));
    CUDA(cudaStreamDestroy(transfer));

    if (printResults && ranks == 1) {
        print_results(local, "Concentration");
    } else if (printResults) {
        // Window the gather so MPI's int counts and displacements fit even
        // when the global field contains more than INT_MAX elements.
        std::vector<double> global(rank ? 0 : volume);
        std::vector<int> counts(ranks), offsets(ranks);
        for (size_t base = 0; base < volume; base += INT_MAX) {
            size_t end = std::min(volume, base + INT_MAX);
            for (int r = 0; r < ranks; ++r) {
                Slab slab = partition(nz, ranks, r);
                size_t first = std::clamp(plane * slab.first, base, end);
                size_t last = std::clamp(plane * slab.end, base, end);
                counts[r] = last - first;
                offsets[r] = first - base;
            }
            size_t skip = counts[rank] ? base + offsets[rank] - owned.first * plane : 0;
            MPI_Gatherv(local.data() + skip, counts[rank], MPI_DOUBLE,
                        rank ? nullptr : global.data() + base, counts.data(), offsets.data(),
                        MPI_DOUBLE, 0, active);
        }
        if (!rank) print_results(global, "Concentration");
    }
    int valid = 1;
    if (validate) {
        double low = INFINITY, high = -INFINITY;
        int finite = 1;
        #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(&:finite)
        for (size_t i = 0; i < localCells; ++i) {
            finite &= std::isfinite(local[i]);
            low = std::min(low, local[i]);
            high = std::max(high, local[i]);
        }
        double minVal, maxVal;
        MPI_Reduce(&finite, &valid, 1, MPI_INT, MPI_MIN, 0, active);
        MPI_Reduce(&low, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, active);
        MPI_Reduce(&high, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, active);
        if (!rank) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                valid = minVal >= -10.0 && maxVal <= 10.0;
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, active);
    }
    MPI_Comm_free(&active);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return result;
}
