#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// One halo plane on either side. Only physical domain faces are clamped;
// internal slab faces use the values exchanged with the adjacent rank.
__device__ __forceinline__ double laplacian(const double* c, size_t i,
                                           size_t nx, size_t ny, size_t localZ,
                                           bool first, bool last) {
    const size_t plane = nx * ny;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / plane;
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t xn = x > 0 ? i - 1 : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t yn = y > 0 ? i - nx : i;
    const size_t zp = last && z == localZ ? i : i + plane;
    const size_t zn = first && z == 1 ? i : i - plane;
    const double center = c[i];
    const double cxx = c[xp] + c[xn] - 2.0 * center;
    const double cyy = c[yp] + c[yn] - 2.0 * center;
    const double czz = c[zp] + c[zn] - 2.0 * center;
    return cxx + cyy + czz;
}

template<bool Chemical>
__global__ void stencil(const double* field, const double* cold, double* out,
                        size_t nx, size_t ny, size_t localZ, bool first, bool last,
                        size_t begin, size_t count) {
    for (size_t j = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const size_t i = begin + j;
        const double lap = laplacian(field, i, nx, ny, localZ, first, last);
        if constexpr (Chemical) {
            const double cv = field[i];
            constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0);
            constexpr double e_AB = 2.0 / 9.0;
            out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * 1.0 * lap;
        }
    }
}

// Pinned staging keeps MPI portable (CUDA-aware MPI is not required). Two
// streams allow the GPU to compute the interior while MPI progresses halos.
struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    double* halos = nullptr;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, received;

    Slab(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x * y), rank(r), ranks(n), comm(c) {
        const size_t bytes = (nz + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&cold, bytes));
        CUDA(cudaMalloc(&cnew, bytes));
        CUDA(cudaMalloc(&mu, bytes));
        CUDA(cudaHostAlloc(&halos, 4 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
    }

    ~Slab() {
        CUDA(cudaFree(cold));
        CUDA(cudaFree(cnew));
        CUDA(cudaFree(mu));
        CUDA(cudaFreeHost(halos));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(received));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(transfer));
    }

    template<bool Chemical>
    void launch(const double* field, double* out, size_t begin, size_t count) {
        if (!count) return;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535));
        stencil<Chemical><<<blocks, 256, 0, compute>>>(field, cold, out, nx, ny, nz,
                                                     rank == 0, rank == ranks - 1, begin, count);
        CUDA(cudaGetLastError());
    }

    template<bool Chemical>
    void step(double* field, double* out) {
        if (ranks == 1) {
            launch<Chemical>(field, out, plane, nz * plane);
            return;
        }
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        // Interior cells never touch receive buffers or the remote halos.
        if (nz > 2) launch<Chemical>(field, out, 2 * plane, (nz - 2) * plane);
        const size_t bytes = plane * sizeof(double);
        MPI_Request requests[4];
        int nr = 0;
        if (rank > 0) {
            MPI_Irecv(halos + 2 * plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 1, comm, &requests[nr++]);
            CUDA(cudaMemcpyAsync(halos, field + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(halos + 3 * plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 0, comm, &requests[nr++]);
            CUDA(cudaMemcpyAsync(halos + plane, field + nz * plane, bytes,
                                 cudaMemcpyDeviceToHost, transfer));
        }
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0)
            MPI_Isend(halos, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[nr++]);
        if (rank + 1 < ranks)
            MPI_Isend(halos + plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 1, comm, &requests[nr++]);
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA(cudaMemcpyAsync(field, halos + 2 * plane, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA(cudaMemcpyAsync(field + (nz + 1) * plane, halos + 3 * plane, bytes,
                                 cudaMemcpyHostToDevice, transfer));
        CUDA(cudaEventRecord(received, transfer));
        CUDA(cudaStreamWaitEvent(compute, received, 0));
        launch<Chemical>(field, out, plane, plane);
        if (nz > 1) launch<Chemical>(field, out, nz * plane, plane);
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
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
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* text = argv[++i];
            unsigned long long value = strtoull(text, &end, 10);
            if (!*text || *end || *text == '-' || value > static_cast<unsigned long long>(INT_MAX) ||
                (option == 'x' && value == 0)) {
                if (!worldRank) fprintf(stderr, "Invalid numeric argument: %s\n", text);
                MPI_Finalize();
                return 1;
            }
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else {
            const bool help = !strcmp(argv[i], "-h");
            if (!worldRank) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx > static_cast<size_t>(INT_MAX) / ny || nx * ny > limit / (nz + 2)) {
        if (!worldRank) fprintf(stderr, "Grid exceeds supported allocation or MPI plane size\n");
        MPI_Finalize();
        return 1;
    }

    // Extra ranks sit out when there are fewer Z planes than ranks.
    const int activeRanks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    int result = 0;
    if (comm != MPI_COMM_NULL) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node);
        int nodeRank, devices;
        MPI_Comm_rank(node, &nodeRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(nodeRank % devices));
        MPI_Comm_free(&node);
        const size_t localZ = nz / activeRanks + (size_t(worldRank) < nz % activeRanks);
        const size_t offsetZ = size_t(worldRank) * (nz / activeRanks) +
                               std::min<size_t>(worldRank, nz % activeRanks);
        const size_t plane = nx * ny, count = localZ * plane, total = plane * nz;
        if (!worldRank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\nTime steps: %d\n", nx, ny, nz, iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d; CUDA acceleration; OpenMP threads: %d\n", activeRanks, omp_get_max_threads());
            printf("Initializing concentration field...\n");
        }
        {
            Slab slab(nx, ny, localZ, worldRank, activeRanks, comm);
            double* host;
            CUDA(cudaHostAlloc(&host, count * sizeof(double), cudaHostAllocDefault));
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < count; ++i) {
                const size_t linear = offsetZ * plane + i;
                const double pseudo = (((linear + 1) * 1299709) % total) / static_cast<double>(total);
                host[i] = -1.0 + 2.0 * pseudo;
            }
            CUDA(cudaMemcpyAsync(slab.cold + plane, host, count * sizeof(double),
                                 cudaMemcpyHostToDevice, slab.compute));
            CUDA(cudaStreamSynchronize(slab.compute));
            if (!worldRank) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                slab.step<true>(slab.cold, slab.mu);
                slab.step<false>(slab.mu, slab.cnew);
                std::swap(slab.cold, slab.cnew);
            }
            CUDA(cudaStreamSynchronize(slab.compute));
            const double elapsed = MPI_Wtime() - start;
            double seconds;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!worldRank) {
                printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
                printf("Performance: %.3f MCellUpdates/s\n", double(total) * iterations / seconds / 1e6);
            }
            if (validate || printResults)
                CUDA(cudaMemcpy(host, slab.cold + plane, count * sizeof(double), cudaMemcpyDeviceToHost));
            if (printResults) {
                // Preserve the original global ordering, Kahan sum and byte hash.
                // Chunked transfers avoid MPI's int count limit on the full volume.
                std::vector<double> all;
                if (!worldRank) {
                    all.resize(total);
                    std::copy(host, host + count, all.begin());
                }
                for (int r = 1; r < activeRanks; ++r) {
                    const size_t rz = nz / activeRanks + (size_t(r) < nz % activeRanks);
                    const size_t oz = size_t(r) * (nz / activeRanks) + std::min<size_t>(r, nz % activeRanks);
                    for (size_t p = 0; p < rz * plane;) {
                        const int chunk = static_cast<int>(std::min<size_t>(rz * plane - p, INT_MAX));
                        if (worldRank == r) MPI_Send(host + p, chunk, MPI_DOUBLE, 0, 2, comm);
                        if (!worldRank) MPI_Recv(all.data() + oz * plane + p, chunk, MPI_DOUBLE,
                                                r, 2, comm, MPI_STATUS_IGNORE);
                        p += chunk;
                    }
                }
                if (!worldRank) print_results(all, "Concentration");
            }
            if (validate) {
                int invalid = 0;
                double minimum = std::numeric_limits<double>::infinity();
                double maximum = -std::numeric_limits<double>::infinity();
                #pragma omp parallel for schedule(static) reduction(|:invalid) reduction(min:minimum) reduction(max:maximum)
                for (size_t i = 0; i < count; ++i) {
                    invalid |= !std::isfinite(host[i]);
                    minimum = std::min(minimum, host[i]);
                    maximum = std::max(maximum, host[i]);
                }
                int globalInvalid;
                double globalMin, globalMax;
                MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
                MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
                MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
                result = globalInvalid || globalMin < -10.0 || globalMax > 10.0;
                if (!worldRank) {
                    printf("Validating result...\n");
                    if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                        if (result) printf("Validation failed: values out of expected range\n");
                    }
                    printf("Validation: %s\n", result ? "FAILED" : "PASSED");
                }
            }
            CUDA(cudaFreeHost(host));
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
