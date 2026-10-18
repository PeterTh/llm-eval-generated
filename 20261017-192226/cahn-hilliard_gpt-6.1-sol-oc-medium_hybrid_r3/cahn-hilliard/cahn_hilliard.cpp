#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// The local array includes one halo plane on either side. Physical faces
// clamp to their own cell, exactly as in the serial stencil.
template<bool Potential>
__global__ void stencil(const double* __restrict__ in, double* __restrict__ out,
                        const double* __restrict__ cold, size_t nx, size_t ny,
                        size_t localZ, size_t first, size_t last,
                        bool physicalLow, bool physicalHigh) {
    const size_t plane = nx * ny;
    const size_t count = (last - first) * plane;
    for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         q < count; q += size_t(blockDim.x) * gridDim.x) {
        const size_t z = first + q / plane;
        const size_t xy = q % plane;
        const size_t x = xy % nx, y = xy / nx;
        const size_t i = z * plane + xy;
        const double cv = in[i];
        const double xx = in[i + (x + 1 < nx ? 1 : 0)] + in[i - (x ? 1 : 0)] - 2.0 * cv;
        const double yy = in[i + (y + 1 < ny ? nx : 0)] + in[i - (y ? nx : 0)] - 2.0 * cv;
        const double zz = in[i + (z == localZ && physicalHigh ? 0 : plane)]
                        + in[i - (z == 1 && physicalLow ? 0 : plane)] - 2.0 * cv;
        const double lap = xx + yy + zz;
        if (Potential) {
            constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            out[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                   + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            out[i] = cold[i] + 0.01 * lap;
        }
    }
}

struct Slab {
    MPI_Comm comm;
    int rank, ranks;
    size_t nx, ny, nz, plane;
    double *old, *next, *mu, *send, *recv;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, halos;

    Slab(MPI_Comm communicator, size_t x, size_t y, size_t z)
        : comm(communicator), nx(x), ny(y), nz(z), plane(x * y) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&halos, cudaEventDisableTiming));
        const size_t bytes = (nz + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&old, bytes));
        CUDA(cudaMalloc(&next, bytes));
        CUDA(cudaMalloc(&mu, bytes));
        // Pinned staging works with ordinary MPI, including non-CUDA-aware MPI.
        CUDA(cudaMallocHost(&send, 2 * plane * sizeof(double)));
        CUDA(cudaMallocHost(&recv, 2 * plane * sizeof(double)));
    }

    template<bool Potential>
    void launch(const double* in, double* out, size_t first, size_t last) {
        if (first >= last) return;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, ((last - first) * plane + 255) / 256));
        stencil<Potential><<<blocks, 256, 0, compute>>>(in, out, old, nx, ny, nz,
                                                     first, last, rank == 0, rank == ranks - 1);
        CUDA(cudaGetLastError());
    }

    template<bool Potential>
    void phase(double* in, double* out) {
        if (ranks == 1) {
            launch<Potential>(in, out, 1, nz + 1);
            return;
        }
        const size_t bytes = plane * sizeof(double);
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        if (rank > 0)
            CUDA(cudaMemcpyAsync(send, in + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks)
            CUDA(cudaMemcpyAsync(send + plane, in + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));

        MPI_Request requests[4];
        int n = 0;
        // Interior computation does not depend on received halo values.
        launch<Potential>(in, out, 2, nz);
        // This also completes the preceding phase's H2D reads of recv before
        // MPI may overwrite that pinned buffer with the next halo exchange.
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0)
            MPI_Irecv(recv, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[n++]);
        if (rank + 1 < ranks)
            MPI_Irecv(recv + plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[n++]);
        if (rank > 0)
            MPI_Isend(send, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[n++]);
        if (rank + 1 < ranks)
            MPI_Isend(send + plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[n++]);
        MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA(cudaMemcpyAsync(in, recv, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA(cudaMemcpyAsync(in + (nz + 1) * plane, recv + plane, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaEventRecord(halos, transfer));
        CUDA(cudaStreamWaitEvent(compute, halos, 0));
        launch<Potential>(in, out, 1, 2);
        if (nz > 1) launch<Potential>(in, out, nz, nz + 1);
    }

    ~Slab() {
        CUDA(cudaFree(old)); CUDA(cudaFree(next)); CUDA(cudaFree(mu));
        CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(recv));
        CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(halos));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    }
};

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided, worldRank, worldSize;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
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
            const long long value = strtoll(argv[++i], &end, 10);
            if (*end || value < 0 || (option != 'i' && value == 0) ||
                (option == 'i' && value > INT_MAX)) {
                if (!worldRank) fprintf(stderr, "Invalid value for -%c\n", option);
                MPI_Finalize(); return 1;
            }
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!worldRank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > size_t(INT_MAX) / ny || nz > std::numeric_limits<size_t>::max() / (nx * ny) / sizeof(double) - 2) {
        if (!worldRank) fprintf(stderr, "Grid exceeds supported allocation or MPI halo size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    // Empty slabs are excluded from simulation collectives.
    const int active = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    int status = 0;
    if (worldRank < active) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node);
        int localRank, devices;
        MPI_Comm_rank(node, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&node);
        const size_t localZ = nz / active + (size_t(worldRank) < nz % active);
        const size_t offsetZ = size_t(worldRank) * (nz / active) + std::min<size_t>(worldRank, nz % active);
        const size_t count = localZ * plane;
        if (!worldRank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        {
            Slab slab(comm, nx, ny, localZ);
            std::vector<double> host(count);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < count; ++i) {
                const size_t id = offsetZ * plane + i;
                const double pseudo = (((id + 1) * size_t(1299709)) % gridSize) / static_cast<double>(gridSize);
                host[i] = -1.0 + 2.0 * pseudo;
            }
            CUDA(cudaMemcpyAsync(slab.old + plane, host.data(), count * sizeof(double),
                                 cudaMemcpyHostToDevice, slab.compute));
            CUDA(cudaStreamSynchronize(slab.compute));
            // Two steps restore the ping-pong pointer layout. Replaying this
            // graph amortizes host launch overhead for small single-GPU grids.
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            if (active == 1 && iterations >= 2) {
                CUDA(cudaStreamBeginCapture(slab.compute, cudaStreamCaptureModeThreadLocal));
                for (int t = 0; t < 2; ++t) {
                    slab.phase<true>(slab.old, slab.mu);
                    slab.phase<false>(slab.mu, slab.next);
                    std::swap(slab.old, slab.next);
                }
                CUDA(cudaStreamEndCapture(slab.compute, &graph));
                CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            }
            if (!worldRank) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            int t = 0;
            if (executable) {
                for (; t + 1 < iterations; t += 2)
                    CUDA(cudaGraphLaunch(executable, slab.compute));
            }
            for (; t < iterations; ++t) {
                slab.phase<true>(slab.old, slab.mu);
                slab.phase<false>(slab.mu, slab.next);
                std::swap(slab.old, slab.next);
            }
            CUDA(cudaStreamSynchronize(slab.compute));
            const double elapsed = MPI_Wtime() - start;
            double seconds;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (executable) CUDA(cudaGraphExecDestroy(executable));
            if (graph) CUDA(cudaGraphDestroy(graph));
            if (!worldRank) {
                printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
                printf("Performance: %.3f MCellUpdates/s\n", double(gridSize) * iterations / seconds / 1e6);
            }
            if (validate || printResults)
                CUDA(cudaMemcpy(host.data(), slab.old + plane, count * sizeof(double), cudaMemcpyDeviceToHost));
            if (printResults) {
                // Chunked rank-ordered gathering avoids MPI's int displacement limit
                // and preserves the original global order for hashes and Kahan sums.
                std::vector<double> result;
                if (!worldRank) result.resize(gridSize);
                for (int r = 0; r < active; ++r) {
                    const size_t z = nz / active + (size_t(r) < nz % active);
                    const size_t offset = (size_t(r) * (nz / active) + std::min<size_t>(r, nz % active)) * plane;
                    for (size_t pos = 0; pos < z * plane;) {
                        const int chunk = static_cast<int>(std::min<size_t>(INT_MAX, z * plane - pos));
                        if (r == 0 && worldRank == 0)
                            std::copy_n(host.data() + pos, chunk, result.data() + offset + pos);
                        else if (worldRank == r)
                            MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                        else if (worldRank == 0)
                            MPI_Recv(result.data() + offset + pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                        pos += chunk;
                    }
                }
                if (!worldRank) print_results(result, "Concentration");
            }
            if (validate) {
                double low = host[0], high = host[0];
                int bad = 0;
                #pragma omp parallel for reduction(min:low) reduction(max:high) reduction(|:bad) schedule(static)
                for (size_t i = 0; i < count; ++i) {
                    bad |= !std::isfinite(host[i]);
                    low = std::min(low, host[i]); high = std::max(high, host[i]);
                }
                double globalLow, globalHigh;
                int globalBad;
                MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, comm);
                MPI_Allreduce(&low, &globalLow, 1, MPI_DOUBLE, MPI_MIN, comm);
                MPI_Allreduce(&high, &globalHigh, 1, MPI_DOUBLE, MPI_MAX, comm);
                status = globalBad || globalLow < -10.0 || globalHigh > 10.0;
                if (!worldRank) {
                    printf("Validating result...\n");
                    if (globalBad) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n", globalLow, globalHigh);
                        if (status) printf("Validation failed: values out of expected range\n");
                    }
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
