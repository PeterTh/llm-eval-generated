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

// Owned planes occupy [1, depth]; planes 0 and depth+1 are MPI ghosts.
// Unit spacings and physical constants are the original benchmark parameters.
__global__ void stencil(const double* __restrict__ input,
                        double* __restrict__ output,
                        const double* __restrict__ cold,
                        size_t nx, size_t ny, size_t depth,
                        bool lower, bool upper, bool chemical, int region) {
    const size_t plane = nx * ny;
    const size_t planes = region == 2 ? depth : (region == 1 ? (depth == 1 ? 1 : 2) : depth - 2);
    for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         j < planes * plane; j += (size_t)blockDim.x * gridDim.x) {
        const size_t p = j / plane;
        const size_t z = region == 2 ? p + 1 : (region == 1 ? (p == 0 ? 1 : depth) : p + 2);
        const size_t xy = j % plane, x = xy % nx, y = xy / nx;
        const size_t i = z * plane + xy;
        const size_t zm = (z == 1 && !lower) ? i : i - plane;
        const size_t zp = (z == depth && !upper) ? i : i + plane;
        const double v = input[i];
        const double xx = input[i + (x + 1 < nx ? 1 : 0)] + input[i - (x ? 1 : 0)] - 2.0 * v;
        const double yy = input[i + (y + 1 < ny ? nx : 0)] + input[i - (y ? nx : 0)] - 2.0 * v;
        const double zz = input[zp] + input[zm] - 2.0 * v;
        const double lap = (xx + yy) + zz;
        if (chemical) {
            constexpr double aa = -(2.0 / 9.0), bb = -(2.0 / 9.0), ab = 2.0 / 9.0;
            output[i] = 4.5 * ((v + 1.0) * aa + (v - 1.0) * bb - 2.0 * v * ab)
                        + 3.0 * v + v * v * v - 0.5 * lap;
        } else {
            output[i] = cold[i] + 0.01 * lap;
        }
    }
}

struct Solver {
    size_t nx, ny, depth, plane;
    int lower, upper;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, received;
    double *send, *recv;

    Solver(size_t x, size_t y, size_t z, int rank, int ranks, MPI_Comm communicator)
        : nx(x), ny(y), depth(z), plane(x*y),
          lower(rank ? rank-1 : MPI_PROC_NULL),
          upper(rank+1 < ranks ? rank+1 : MPI_PROC_NULL), comm(communicator) {
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        CUDA(cudaMallocHost(&send, 2*plane*sizeof(double)));
        CUDA(cudaMallocHost(&recv, 2*plane*sizeof(double)));
    }
    ~Solver() {
        CUDA(cudaFreeHost(send)); CUDA(cudaFreeHost(recv));
        CUDA(cudaEventDestroy(ready)); CUDA(cudaEventDestroy(received));
        CUDA(cudaStreamDestroy(compute)); CUDA(cudaStreamDestroy(transfer));
    }
    void launch(const double* in, double* out, const double* cold, bool chemical, int region) {
        const size_t count = plane * (region == 2 ? depth : (region == 1 ? std::min(size_t(2), depth) : depth-2));
        const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (count+255)/256));
        stencil<<<blocks, 256, 0, compute>>>(in, out, cold, nx, ny, depth,
                       lower != MPI_PROC_NULL, upper != MPI_PROC_NULL, chemical, region);
        CUDA(cudaGetLastError());
    }
    void step(double* in, double* out, const double* cold, bool chemical) {
        if (lower == MPI_PROC_NULL && upper == MPI_PROC_NULL) {
            launch(in, out, cold, chemical, 2);
            return;
        }
        // Pinned staging works with ordinary MPI as well as CUDA-aware MPI.
        // MPI progresses on the calling thread while the GPU computes interior planes.
        // The previous asynchronous upload must release the receive buffer
        // before MPI is allowed to overwrite it.
        if (lower != MPI_PROC_NULL || upper != MPI_PROC_NULL)
            CUDA(cudaStreamSynchronize(transfer));
        MPI_Request requests[4];
        int n = 0;
        const int count = static_cast<int>(plane);
        const size_t bytes = plane*sizeof(double);
        if (lower != MPI_PROC_NULL) MPI_Irecv(recv, count, MPI_DOUBLE, lower, 1, comm, &requests[n++]);
        if (upper != MPI_PROC_NULL) MPI_Irecv(recv+plane, count, MPI_DOUBLE, upper, 0, comm, &requests[n++]);
        if (lower != MPI_PROC_NULL || upper != MPI_PROC_NULL) {
            CUDA(cudaEventRecord(ready, compute));
            CUDA(cudaStreamWaitEvent(transfer, ready, 0));
            if (lower != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send, in+plane, bytes, cudaMemcpyDeviceToHost, transfer));
            if (upper != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send+plane, in+depth*plane, bytes, cudaMemcpyDeviceToHost, transfer));
        }
        if (depth > 2) launch(in, out, cold, chemical, false);
        if (lower != MPI_PROC_NULL || upper != MPI_PROC_NULL) {
            CUDA(cudaStreamSynchronize(transfer));
            if (lower != MPI_PROC_NULL) MPI_Isend(send, count, MPI_DOUBLE, lower, 0, comm, &requests[n++]);
            if (upper != MPI_PROC_NULL) MPI_Isend(send+plane, count, MPI_DOUBLE, upper, 1, comm, &requests[n++]);
            MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
            if (lower != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(in, recv, bytes, cudaMemcpyHostToDevice, transfer));
            if (upper != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(in+(depth+1)*plane, recv+plane, bytes, cudaMemcpyHostToDevice, transfer));
            CUDA(cudaEventRecord(received, transfer));
            CUDA(cudaStreamWaitEvent(compute, received, 0));
        }
        launch(in, out, cold, chemical, true);
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
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    

    if (!nx || nx > static_cast<size_t>(INT_MAX) / ny ||
        nz > std::numeric_limits<size_t>::max() / (nx*ny) / sizeof(double) - 2) {
        if (rank == 0) fprintf(stderr, "Grid dimensions are invalid or too large.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Extra ranks do not own empty slabs or participate in halo traffic.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) { MPI_Finalize(); return 0; }
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "A CUDA GPU is required on every active rank.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    const size_t plane = nx*ny, volume = plane*nz;
    const size_t depth = nz/active + (static_cast<size_t>(rank) < nz%active);
    const size_t first = rank*(nz/active) + std::min(static_cast<size_t>(rank), nz%active);
    const size_t owned = depth*plane, bytes = (depth+2)*plane*sizeof(double);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> host(owned);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < owned; ++i) {
        const size_t id = first*plane+i;
        const double pseudo = (((id+1)*size_t(1299709)) % volume) / static_cast<double>(volume);
        host[i] = -1.0 + 2.0*pseudo;
    }
    double *cold, *cnew, *mu;
    CUDA(cudaMalloc(&cold, bytes)); CUDA(cudaMalloc(&cnew, bytes)); CUDA(cudaMalloc(&mu, bytes));
    CUDA(cudaMemcpy(cold+plane, host.data(), owned*sizeof(double), cudaMemcpyHostToDevice));
    {
        Solver solver(nx, ny, depth, rank, active, comm);
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        CUDA(cudaDeviceSynchronize());
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            solver.step(cold, mu, cold, true);
            solver.step(mu, cnew, cold, false);
            std::swap(cold, cnew);
        }
        CUDA(cudaStreamSynchronize(solver.compute));
        double elapsed = MPI_Wtime()-start, total;
        MPI_Reduce(&elapsed, &total, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(total*1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/total/1e6);
        }
    }
    if (printResults || validate)
        CUDA(cudaMemcpy(host.data(), cold+plane, owned*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaFree(cold)); CUDA(cudaFree(cnew)); CUDA(cudaFree(mu));
    if (printResults) {
        // Gather only on request, preserving the serial order and its exact hash/sum.
        // Chunk transfers to avoid MPI's int count limit for large volumes.
        std::vector<double> result;
        if (rank == 0) { result.resize(volume); std::copy(host.begin(), host.end(), result.begin()); }
        for (int r = 1; r < active; ++r) {
            const size_t begin = (r*(nz/active)+std::min(size_t(r), nz%active))*plane;
            const size_t count = (nz/active+(size_t(r)<nz%active))*plane;
            for (size_t off = 0; off < count;) {
                const int chunk = static_cast<int>(std::min(count-off, size_t(INT_MAX)));
                if (rank == r) MPI_Send(host.data()+off, chunk, MPI_DOUBLE, 0, 2, comm);
                if (rank == 0) MPI_Recv(result.data()+begin+off, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                off += chunk;
            }
        }
        if (rank == 0) print_results(result, "Concentration");
    }
    int failed = 0;
    if (validate) {
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        int nonfinite = 0;
        #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:nonfinite) schedule(static)
        for (size_t i = 0; i < owned; ++i) {
            nonfinite |= !std::isfinite(host[i]);
            lo = std::min(lo, host[i]); hi = std::max(hi, host[i]);
        }
        double globalLo, globalHi;
        MPI_Allreduce(&nonfinite, &failed, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (failed) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                if (globalHi > 10.0 || globalLo < -10.0) printf("Validation failed: values out of expected range\n");
            }
        }
        failed |= globalHi > 10.0 || globalLo < -10.0;
        if (rank == 0) printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return failed ? 1 : 0;
}
