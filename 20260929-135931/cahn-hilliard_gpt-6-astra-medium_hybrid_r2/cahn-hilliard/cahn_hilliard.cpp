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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// The physical mesh has unit spacing. Only MPI interfaces use ghost planes;
// global domain faces retain the original clamped (zero normal flux) stencil.
__device__ __forceinline__ double laplacian(const double* a, size_t i,
        size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz,
        bool lower, bool upper) {
    const size_t plane = nx * ny;
    double center = a[i];
    double xx = a[i + (x + 1 < nx ? 1 : 0)] + a[i - (x ? 1 : 0)] - 2.0 * center;
    double yy = a[i + (y + 1 < ny ? nx : 0)] + a[i - (y ? nx : 0)] - 2.0 * center;
    double zz = a[i + (z < nz || upper ? plane : 0)]
              + a[i - (z > 1 || lower ? plane : 0)] - 2.0 * center;
    return xx + yy + zz;
}

template<bool chemical>
__global__ void stencil(const double* __restrict__ input,
        const double* __restrict__ cold, double* __restrict__ output,
        size_t nx, size_t ny, size_t nz, size_t first, size_t count,
        bool lower, bool upper) {
    const size_t plane = nx * ny;
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count * plane; j += size_t(blockDim.x) * gridDim.x) {
        size_t z = first + j / plane;
        size_t xy = j % plane;
        size_t x = xy % nx, y = xy / nx, i = z * plane + xy;
        double lap = laplacian(input, i, x, y, z, nx, ny, nz, lower, upper);
        if (chemical) {
            const double cv = input[i];
            const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        } else {
            output[i] = cold[i] + 0.01 * lap;
        }
    }
}

struct Solver {
    size_t nx, ny, nz, plane;
    int rank, ranks;
    MPI_Comm comm;
    double *a, *b, *mu, *send, *recv;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, received;

    Solver(size_t x, size_t y, size_t z, int r, int n, MPI_Comm c)
        : nx(x), ny(y), nz(z), plane(x*y), rank(r), ranks(n), comm(c) {
        size_t bytes = (nz + 2) * plane * sizeof(double);
        CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes)); CUDA(cudaMalloc(&mu, bytes));
        CUDA(cudaMallocHost(&send, 2 * plane * sizeof(double)));
        CUDA(cudaMallocHost(&recv, 2 * plane * sizeof(double)));
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
    }
    ~Solver() {
        CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFree(mu));
        CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(recv));
        CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(received));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    }
    template<bool chemical>
    void launch(double* input, double* output, size_t first, size_t count) {
        if (!count) return;
        unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (count * plane + 255) / 256));
        stencil<chemical><<<blocks, 256, 0, compute>>>(input, a, output,
            nx, ny, nz, first, count, rank > 0, rank + 1 < ranks);
        CUDA(cudaGetLastError());
    }
    template<bool chemical>
    void phase(double* input, double* output) {
        if (ranks == 1) { launch<chemical>(input, output, 1, nz); return; }
        // Copies and nonblocking MPI traffic run concurrently with interior work.
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        size_t bytes = plane * sizeof(double);
        if (rank > 0) CUDA(cudaMemcpyAsync(send, input + plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(send + plane, input + nz * plane, bytes, cudaMemcpyDeviceToHost, transfer));
        if (nz > 2) launch<chemical>(input, output, 2, nz - 2);
        MPI_Request requests[4]; int num = 0;
        if (rank > 0) MPI_Irecv(recv, int(plane), MPI_DOUBLE, rank-1, 1, comm, &requests[num++]);
        if (rank + 1 < ranks) MPI_Irecv(recv + plane, int(plane), MPI_DOUBLE, rank+1, 0, comm, &requests[num++]);
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0) MPI_Isend(send, int(plane), MPI_DOUBLE, rank-1, 0, comm, &requests[num++]);
        if (rank + 1 < ranks) MPI_Isend(send + plane, int(plane), MPI_DOUBLE, rank+1, 1, comm, &requests[num++]);
        MPI_Waitall(num, requests, MPI_STATUSES_IGNORE);
        if (rank > 0) CUDA(cudaMemcpyAsync(input, recv, bytes, cudaMemcpyHostToDevice, transfer));
        if (rank + 1 < ranks) CUDA(cudaMemcpyAsync(input + (nz+1)*plane, recv+plane, bytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaEventRecord(received, transfer));
        CUDA(cudaStreamWaitEvent(compute, received, 0));
        launch<chemical>(input, output, 1, 1);
        if (nz > 1) launch<chemical>(input, output, nz, 1);
    }
    void step() {
        phase<true>(a, mu);
        phase<false>(mu, b);
        std::swap(a, b);
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
    int provided, rank, worldSize;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    

    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (!nx || nx > size_t(INT_MAX) || !ny || ny > size_t(INT_MAX) / nx ||
        !nz || nz > limit / (nx * ny) - 2 || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize(); return 1;
    }
    // Empty ranks do not participate in halo traffic, including nz < worldSize.
    int active = int(std::min(nz, size_t(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < active) {
        MPI_Comm local;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (devices == 0) { fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        size_t localZ = nz / active + (size_t(rank) < nz % active);
        size_t startZ = size_t(rank) * (nz / active) + std::min(size_t(rank), nz % active);
        size_t plane = nx * ny, volume = plane * nz, count = plane * localZ;
        Solver solver(nx, ny, localZ, rank, active, comm);
        std::vector<double> host(count);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i) {
            size_t id = startZ * plane + i;
            double pseudo = (((id + 1) * 1299709) % volume) / static_cast<double>(volume);
            host[i] = -1.0 + 2.0 * pseudo;
        }
        CUDA(cudaMemcpy(solver.a + plane, host.data(), count*sizeof(double), cudaMemcpyHostToDevice));
        // Replay pairs of steps to amortize launch overhead on a single GPU.
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        if (active == 1 && iterations >= 2) {
            CUDA(cudaStreamBeginCapture(solver.compute, cudaStreamCaptureModeThreadLocal));
            solver.step(); solver.step();
            CUDA(cudaStreamEndCapture(solver.compute, &graph));
            CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        }
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        double start = MPI_Wtime();
        int t = 0;
        if (executable) {
            for (; t + 1 < iterations; t += 2) CUDA(cudaGraphLaunch(executable, solver.compute));
        }
        for (; t < iterations; ++t) solver.step();
        CUDA(cudaStreamSynchronize(solver.compute));
        double elapsed = MPI_Wtime() - start, seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (executable) CUDA(cudaGraphExecDestroy(executable));
        if (graph) CUDA(cudaGraphDestroy(graph));
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations / seconds / 1e6);
        }
        if (validate || printResults)
            CUDA(cudaMemcpy(host.data(), solver.a + plane, count*sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            // Gather only on request, in global z/y/x order. Chunking avoids the
            // MPI_Gatherv int displacement/count limit for large global fields.
            const size_t chunk = size_t(INT_MAX);
            if (rank == 0) {
                std::vector<double> result(volume);
                std::copy(host.begin(), host.end(), result.begin());
                size_t offset = count;
                for (int r = 1; r < active; ++r) {
                    size_t n = (nz / active + (size_t(r) < nz % active)) * plane;
                    for (size_t j = 0; j < n; j += std::min(chunk, n-j))
                        MPI_Recv(result.data()+offset+j, int(std::min(chunk, n-j)), MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    offset += n;
                }
                print_results(result, "Concentration");
            } else {
                for (size_t j = 0; j < count; j += std::min(chunk, count-j))
                    MPI_Send(host.data()+j, int(std::min(chunk, count-j)), MPI_DOUBLE, 0, 2, comm);
            }
        }
        if (validate) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int invalid = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < count; ++i) {
                invalid |= !std::isfinite(host[i]);
                lo = std::min(lo, host[i]); hi = std::max(hi, host[i]);
            }
            double globalLo, globalHi;
            int globalInvalid;
            MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
            MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
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
