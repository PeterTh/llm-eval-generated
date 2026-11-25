#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

[[noreturn]] static void fail(const char* message) {
    fprintf(stderr, "cahn_hilliard: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void check(cudaError_t status) {
    if (status != cudaSuccess) fail(cudaGetErrorString(status));
}

// Two ghost planes precede and follow each owned slab. Clamp to the
// physical domain only, including when evaluating mu in a ghost plane.
struct Grid {
    int nx, ny, nz, first, depth;
    size_t plane;
};

__device__ double potential(const double* c, Grid g, int x, int y, int z) {
    z = min(max(z, -g.first), g.nz - 1 - g.first);
    size_t i = size_t(z + 2) * g.plane + size_t(y) * g.nx + x;
    double v = c[i];
    double xx = c[i + (x < g.nx-1)] + c[i - (x > 0)] - 2.0*v;
    double yy = c[i + (y < g.ny-1 ? g.nx : 0)] + c[i - (y > 0 ? g.nx : 0)] - 2.0*v;
    double zz = c[i + (z < g.nz-1-g.first ? g.plane : 0)]
              + c[i - (z > -g.first ? g.plane : 0)] - 2.0*v;
    constexpr double e = 2.0 / 9.0;
    double interaction = fma(-2.0*v, e, fma(v+1.0, -e, (v-1.0)*(-e)));
    double bulk = fma(v*v, v, fma(4.5, interaction, 3.0*v));
    return bulk - 0.5*(xx + yy + zz);
}

// One thread per tile point, including a one-point x/y apron. Share mu
// within the plane and carry its z neighbors in registers.
constexpr int bx = 32, by = 8, layers = 6;
__global__ void advance(const double* c, double* next, Grid g, int begin, int end,
                        int tilesX, int tilesY, size_t tiles) {
    __shared__ double mu[by][bx];
    int tx = threadIdx.x, ty = threadIdx.y;
    for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        int x = int(tile % tilesX)*(bx-2) + tx-1;
        size_t row = tile / tilesX;
        int y = int(row % tilesY)*(by-2) + ty-1;
        int z = begin + int(row / tilesY)*layers;
        bool write = tx > 0 && tx < bx-1 && ty > 0 && ty < by-1 && x < g.nx && y < g.ny;
        x = min(max(x, 0), g.nx-1);
        y = min(max(y, 0), g.ny-1);
        double prev = potential(c, g, x, y, z-1);
        double curr = potential(c, g, x, y, z);
        int count = min(layers, end-z);
        for (int k = 0; k < count; ++k) {
            double upper = potential(c, g, x, y, z+k+1);
            mu[ty][tx] = curr;
            __syncthreads();
            if (write) {
                double xx = mu[ty][tx+1] + mu[ty][tx-1] - 2.0*curr;
                double yy = mu[ty+1][tx] + mu[ty-1][tx] - 2.0*curr;
                double zz = upper + prev - 2.0*curr;
                size_t i = size_t(z+k+2)*g.plane + size_t(y)*g.nx + x;
                next[i] = fma(0.01, xx + yy + zz, c[i]);
            }
            __syncthreads();
            prev = curr;
            curr = upper;
        }
    }
}

static void update(const double* c, double* next, Grid g, int begin, int end,
                   cudaStream_t stream) {
    if (begin >= end) return;
    int tx = 1 + (g.nx-1)/(bx-2), ty = 1 + (g.ny-1)/(by-2);
    size_t tiles = size_t(tx)*ty*(1 + (end-begin-1)/layers);
    advance<<<std::min<size_t>(tiles, INT_MAX), dim3(bx, by), 0, stream>>>(
        c, next, g, begin, end, tx, ty, tiles);
    check(cudaGetLastError());
}

struct Transfer {
    int peer, count;
    size_t offset, staging;
};

// Intersect each owned interval with the peer's two-plane halo. Splitting
// long messages also supports MPI implementations with 32-bit counts.
static void addTransfer(std::vector<Transfer>& transfers, int peer, Grid g,
                        int begin, int end, size_t& staged) {
    if (begin >= end) return;
    size_t offset = size_t(begin-g.first+2)*g.plane;
    size_t length = size_t(end-begin)*g.plane;
    while (length) {
        int count = std::min<size_t>(INT_MAX, length);
        transfers.push_back({peer, count, offset, staged});
        offset += count;
        staged += count;
        length -= count;
    }
}

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
    int provided, rank, world;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int nx = 64;
    int ny = 0;
    int nz = 0;
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
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx <= 0 || ny <= 0 || nz <= 0 ||
        std::max({nx, ny, nz}) > INT_MAX-32)
        fail("invalid grid dimensions");
    size_t plane = size_t(nx)*ny;
    if (plane > std::numeric_limits<size_t>::max()/sizeof(double)/(size_t(nz)+4))
        fail("grid size overflow");
    size_t volume = plane*nz;

    // Restrict computation to nonempty slabs when nz is smaller than the job.
    int ranks = std::min(world, nz);
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (comm == MPI_COMM_NULL) {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return status;
    }

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) fail("a CUDA device is required");
    check(cudaSetDevice(localRank % devices));

    auto slabStart = [=](int r) { return r*(nz/ranks) + std::min(r, nz%ranks); };
    int first = slabStart(rank), last = slabStart(rank+1);
    Grid grid{nx, ny, nz, first, last-first, plane};
    size_t owned = size_t(grid.depth)*plane;
    size_t bytes = (size_t(grid.depth)+4)*plane*sizeof(double);
    double *c, *next;
    check(cudaMalloc(&c, bytes));
    check(cudaMalloc(&next, bytes));

    if (!rank) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(owned);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned; ++i) {
        size_t id = size_t(first)*plane+i;
        double pseudo = ((id+1)*1299709 % volume)/static_cast<double>(volume);
        host[i] = -1.0 + 2.0*pseudo;
    }
    check(cudaMemcpy(c+2*plane, host.data(), owned*sizeof(double), cudaMemcpyHostToDevice));
    if (!validate && !printResults) std::vector<double>().swap(host);

    std::vector<Transfer> sends, receives;
    size_t staged = 0;
    for (int peer = std::max(0, rank-2); peer <= std::min(ranks-1, rank+2); ++peer) {
        if (peer == rank) continue;
        int lo = slabStart(peer), hi = slabStart(peer+1);
        addTransfer(sends, peer, grid, std::max(first, lo-2), std::min(last, hi+2), staged);
        addTransfer(receives, peer, grid, std::max(lo, first-2), std::min(hi, last+2), staged);
    }

    bool direct = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    direct = MPIX_Query_cuda_support() != 0;
#endif
    double* staging = nullptr;
    if (staged && !direct) check(cudaMallocHost(&staging, staged*sizeof(double)));
    cudaStream_t compute, copy;
    cudaEvent_t ready;
    check(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    std::vector<MPI_Request> requests(receives.size()+sends.size());
    int coreBegin = rank ? std::min(2, grid.depth) : 0;
    int coreEnd = std::max(coreBegin, grid.depth - (rank+1 < ranks ? 2 : 0));

    if (!rank) printf("Running Cahn-Hilliard simulation...\n");
    check(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        if (ranks == 1) {
            update(c, next, grid, 0, grid.depth, compute);
        } else {
            // Start the export as soon as the previous field is ready.
            check(cudaEventRecord(ready, compute));
            check(cudaStreamWaitEvent(copy, ready, 0));
            for (const auto& s : sends) {
                if (!direct) check(cudaMemcpyAsync(staging+s.staging, c+s.offset,
                    size_t(s.count)*sizeof(double), cudaMemcpyDeviceToHost, copy));
            }
            update(c, next, grid, coreBegin, coreEnd, compute);
            check(cudaStreamSynchronize(copy));
            size_t n = 0;
            for (const auto& r : receives)
                MPI_Irecv(direct ? c+r.offset : staging+r.staging, r.count, MPI_DOUBLE,
                          r.peer, 0, comm, &requests[n++]);
            for (const auto& s : sends)
                MPI_Isend(direct ? c+s.offset : staging+s.staging, s.count, MPI_DOUBLE,
                          s.peer, 0, comm, &requests[n++]);
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            for (const auto& r : receives) {
                if (!direct) check(cudaMemcpyAsync(c+r.offset, staging+r.staging,
                    size_t(r.count)*sizeof(double), cudaMemcpyHostToDevice, copy));
            }
            check(cudaEventRecord(ready, copy));
            check(cudaStreamWaitEvent(compute, ready, 0));
            update(c, next, grid, 0, coreBegin, compute);
            update(c, next, grid, coreEnd, grid.depth, compute);
        }
        std::swap(c, next);
    }
    check(cudaStreamSynchronize(compute));
    double elapsed = MPI_Wtime()-start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(1000*seconds));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/seconds/1e6);
    }
    if (validate || printResults)
        check(cudaMemcpy(host.data(), c+2*plane, owned*sizeof(double), cudaMemcpyDeviceToHost));
    if (printResults) {
        // Preserve the global order for the bitwise hash and Kahan sum.
        if (!rank) host.resize(volume);
        for (int peer = 1; peer < ranks; ++peer) {
            if (rank != 0 && peer != rank) continue;
            size_t base = size_t(slabStart(peer))*plane;
            size_t count = size_t(slabStart(peer+1)-slabStart(peer))*plane;
            for (size_t i = 0; i < count; ) {
                int chunk = std::min<size_t>(INT_MAX, count-i);
                if (!rank) MPI_Recv(host.data()+base+i, chunk, MPI_DOUBLE, peer, 1, comm, MPI_STATUS_IGNORE);
                else MPI_Send(host.data()+i, chunk, MPI_DOUBLE, 0, 1, comm);
                i += chunk;
            }
        }
        if (!rank) print_results(host, "Concentration");
    }
    if (validate) {
        double low = host[0], high = host[0];
        int invalid = 0;
        #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:invalid)
        for (size_t i = 0; i < owned; ++i) {
            double v = host[i];
            invalid |= !std::isfinite(v);
            low = std::min(low, v);
            high = std::max(high, v);
        }
        double globalLow, globalHigh;
        int nonfinite;
        MPI_Reduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&invalid, &nonfinite, 1, MPI_INT, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Validating result...\n");
            if (nonfinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalLow, globalHigh);
                if (globalHigh > 10 || globalLow < -10)
                    printf("Validation failed: values out of expected range\n");
            }
            status = nonfinite || globalHigh > 10 || globalLow < -10;
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    if (staging) check(cudaFreeHost(staging));
    check(cudaFree(next));
    check(cudaFree(c));
    check(cudaEventDestroy(ready));
    check(cudaStreamDestroy(copy));
    check(cudaStreamDestroy(compute));
    MPI_Comm_free(&comm);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
