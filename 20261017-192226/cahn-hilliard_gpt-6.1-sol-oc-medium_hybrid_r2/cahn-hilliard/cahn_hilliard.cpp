#include <algorithm>
#include <cerrno>
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

// Each rank owns whole XY planes, with one ghost plane at either end.
// X/Y boundaries are clamped here; global Z boundaries are clamped by the
// halo exchange. All fields stay on the device throughout time integration.
__device__ __forceinline__ double laplacian(const double* a, size_t i,
                                           size_t x, size_t y,
                                           size_t nx, size_t ny,
                                           size_t below, size_t above) {
    const double v = a[i];
    const double xx = a[i + (x + 1 < nx)] + a[i - (x > 0)] - 2.0 * v;
    const double yy = a[i + (y + 1 < ny ? nx : 0)] +
                      a[i - (y > 0 ? nx : 0)] - 2.0 * v;
    const double zz = a[i + above] + a[i - below] - 2.0 * v;
    return xx + yy + zz;
}

template<bool chemical, bool boundary, bool whole = false>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ cold,
                        double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t plane = nx * ny;
    const size_t count = plane * (whole ? nz : (boundary ? (nz > 1 ? 2 : 1) : nz - 2));
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const size_t z = whole ? j / plane + 1 : (boundary ? (j < plane ? 1 : nz) : j / plane + 2);
        const size_t xy = j % plane;
        const size_t i = z * plane + xy;
        const double lap = laplacian(input, i, xy % nx, xy / nx, nx, ny,
                                     whole && z == 1 ? 0 : plane,
                                     whole && z == nz ? 0 : plane);
        if constexpr (chemical) {
            const double cv = input[i];
            constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0);
            constexpr double e_AB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            output[i] = cold[i] + 0.01 * 1.0 * lap;
        }
    }
}

struct Slab {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, halos;
    double* staging;

    Slab(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x * y), rank(r), ranks(n), comm(c) {
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&halos, cudaEventDisableTiming));
        CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(double)));
    }
    ~Slab() {
        CUDA(cudaFreeHost(staging));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(halos));
        CUDA(cudaStreamDestroy(compute));
        CUDA(cudaStreamDestroy(transfer));
    }

    template<bool chemical>
    void step(double* input, const double* cold, double* output) {
        if (ranks == 1) {
            // With no remote halos, clamp directly in one full-volume kernel.
            // Enqueue all timesteps without host/device synchronization.
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, (plane * nz + 255) / 256));
            stencil<chemical, false, true><<<blocks, 256, 0, compute>>>(input, cold, output, nx, ny, nz);
            CUDA(cudaGetLastError());
            return;
        }
        // The transfer stream reads only the completed preceding field, while
        // the compute stream evaluates points independent of incoming halos.
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        if (nz > 2) {
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, (plane * (nz - 2) + 255) / 256));
            stencil<chemical, false><<<blocks, 256, 0, compute>>>(input, cold, output, nx, ny, nz);
            CUDA(cudaGetLastError());
        }
        const size_t bytes = plane * sizeof(double);
        MPI_Request requests[4];
        int nr = 0;
        if (rank > 0) {
            MPI_Irecv(staging + 2 * plane, int(plane), MPI_DOUBLE, rank - 1, 1, comm, &requests[nr++]);
            CUDA(cudaMemcpyAsync(staging, input + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        } else {
            CUDA(cudaMemcpyAsync(input, input + plane, bytes, cudaMemcpyDeviceToDevice, transfer));
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(staging + 3 * plane, int(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[nr++]);
            CUDA(cudaMemcpyAsync(staging + plane, input + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        } else {
            CUDA(cudaMemcpyAsync(input + (nz + 1) * plane, input + nz * plane, bytes, cudaMemcpyDeviceToDevice, transfer));
        }
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0)
            MPI_Isend(staging, int(plane), MPI_DOUBLE, rank - 1, 0, comm, &requests[nr++]);
        if (rank + 1 < ranks)
            MPI_Isend(staging + plane, int(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[nr++]);
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
        if (rank > 0)
            CUDA(cudaMemcpyAsync(input, staging + 2 * plane, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks)
            CUDA(cudaMemcpyAsync(input + (nz + 1) * plane, staging + 3 * plane, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaEventRecord(halos, transfer));
        CUDA(cudaStreamWaitEvent(compute, halos, 0));
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(65535, (plane * (nz > 1 ? 2 : 1) + 255) / 256));
        stencil<chemical, true><<<blocks, 256, 0, compute>>>(input, cold, output, nx, ny, nz);
        CUDA(cudaGetLastError());
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseNumber(const char* text, size_t& value) {
    char* end = nullptr;
    if (!*text || *text == '-') return false;
    errno = 0;
    const unsigned long long v = strtoull(text, &end, 10);
    if (errno || *end || v > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(v);
    return true;
}

int main(int argc, char** argv) {
    int provided, worldRank, worldSize;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t nx = 64, ny = 0, nz = 0, iterations = 20;
    bool validate = false, printResults = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value = 0;
            bad |= !parseNumber(argv[++i], value);
            if (option == 'x') nx = value;
            if (option == 'y') ny = value;
            if (option == 'z') nz = value;
            if (option == 'i') iterations = value;
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    // MPI plane counts use int; all volume and device offsets use size_t.
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    bad |= !nx || !ny || !nz || nx > size_t(INT_MAX) / ny;
    if (!bad) bad |= limit / (nx * ny) < 3 || nz > limit / (nx * ny) - 2;
    if (help || bad) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }

    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    int status = 0;
    if (worldRank < ranks) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm local;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "A CUDA device is required on every participating node.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        const size_t plane = nx * ny, gridSize = plane * nz;
        const size_t localZ = nz / ranks + (size_t(rank) < nz % ranks);
        const size_t zStart = size_t(rank) * (nz / ranks) + std::min<size_t>(rank, nz % ranks);
        const size_t count = plane * localZ;
        std::vector<double> host(count);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            const size_t global = zStart * plane + i;
            const double pseudo = (((global + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            host[i] = -1.0 + 2.0 * pseudo;
        }
        double *cold, *cnew, *mu;
        const size_t bytes = (localZ + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&cold, bytes));
        CUDA(cudaMalloc(&cnew, bytes));
        CUDA(cudaMalloc(&mu, bytes));
        CUDA(cudaMemcpy(cold + plane, host.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\nTime steps: %zu\n", nx, ny, nz, iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d, OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
            printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
        }
        {
            Slab slab(nx, ny, localZ, rank, ranks, comm);
            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            for (size_t t = 0; t < iterations; ++t) {
                slab.step<true>(cold, nullptr, mu);
                slab.step<false>(mu, cold, cnew);
                std::swap(cold, cnew);
            }
            CUDA(cudaStreamSynchronize(slab.compute));
            const double elapsed = MPI_Wtime() - start;
            double seconds;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                printf("Performance: %.3f MCellUpdates/s\n", double(gridSize) * iterations / seconds / 1e6);
            }
        }
        if (printResults || validate)
            CUDA(cudaMemcpy(host.data(), cold + plane, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA(cudaFree(cold));
        CUDA(cudaFree(cnew));
        CUDA(cudaFree(mu));

        if (printResults) {
            // Gather only on request, in global Z order. Chunking avoids the
            // MPI_Gatherv int displacement/count limit for large fields.
            std::vector<double> global;
            if (rank == 0) {
                global.resize(gridSize);
                std::copy(host.begin(), host.end(), global.begin());
            }
            for (int r = 1; r < ranks; ++r) {
                const size_t rz = nz / ranks + (size_t(r) < nz % ranks);
                const size_t offset = (size_t(r) * (nz / ranks) + std::min<size_t>(r, nz % ranks)) * plane;
                if (rank == 0 || rank == r) {
                    for (size_t pos = 0; pos < rz * plane;) {
                        const int chunk = static_cast<int>(std::min<size_t>(INT_MAX, rz * plane - pos));
                        if (rank == 0)
                            MPI_Recv(global.data() + offset + pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                        else
                            MPI_Send(host.data() + pos, chunk, MPI_DOUBLE, 0, 2, comm);
                        pos += chunk;
                    }
                }
            }
            if (rank == 0) print_results(global, "Concentration");
        }
        if (validate) {
            double lo = host[0], hi = host[0];
            int invalid = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < count; ++i) {
                invalid |= !std::isfinite(host[i]);
                lo = std::min(lo, host[i]);
                hi = std::max(hi, host[i]);
            }
            double globalLo, globalHi;
            int globalInvalid;
            MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
            status = globalInvalid || globalLo < -10.0 || globalHi > 10.0;
            if (rank == 0) {
                printf("Validating result...\n");
                if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                    if (status) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
