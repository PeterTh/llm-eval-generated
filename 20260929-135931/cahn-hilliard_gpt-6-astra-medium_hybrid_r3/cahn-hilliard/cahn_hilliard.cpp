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

// Z planes 1..depth are owned; planes 0 and depth+1 are MPI halos.
// At physical boundaries the original clamped stencil is used directly.
template<bool Chemical>
__global__ void stencil(const double* __restrict__ input,
                        const double* __restrict__ concentration,
                        double* __restrict__ output, size_t nx, size_t ny,
                        size_t depth, size_t begin, size_t end,
                        bool lowBoundary, bool highBoundary) {
    const size_t plane = nx * ny;
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    // Map warps along X for coalesced loads, without per-cell integer division.
    for (size_t z = begin + blockIdx.z; z < end; z += gridDim.z) {
      for (size_t y = size_t(blockIdx.y)*blockDim.y + threadIdx.y;
           y < ny; y += size_t(blockDim.y)*gridDim.y) {
        const size_t i = z * plane + y * nx + x;
        const size_t xp = x + 1 < nx ? i + 1 : i;
        const size_t xn = x > 0 ? i - 1 : i;
        const size_t yp = y + 1 < ny ? i + nx : i;
        const size_t yn = y > 0 ? i - nx : i;
        const size_t zp = z == depth && highBoundary ? i : i + plane;
        const size_t zn = z == 1 && lowBoundary ? i : i - plane;
        const double cv = input[i];
        // dx = dy = dz = 1, as in the original benchmark.
        const double cxx = input[xp] + input[xn] - 2.0 * cv;
        const double cyy = input[yp] + input[yn] - 2.0 * cv;
        const double czz = input[zp] + input[zn] - 2.0 * cv;
        const double laplacian = cxx + cyy + czz;
        if constexpr (Chemical) {
            constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            output[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * laplacian;
        } else {
            output[i] = concentration[i] + 0.01 * laplacian;
        }
      }
    }
}

struct Solver {
    size_t nx, ny, depth, plane;
    int rank, ranks;
    MPI_Comm comm;
    cudaStream_t compute, transfer;
    cudaEvent_t ready, received;
    double* staging;

    Solver(size_t x, size_t y, size_t d, int r, int n, MPI_Comm c)
        : nx(x), ny(y), depth(d), plane(x*y), rank(r), ranks(n), comm(c), staging(nullptr) {
        CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&received, cudaEventDisableTiming));
        if (ranks > 1) CUDA(cudaMallocHost(&staging, 4 * plane * sizeof(double)));
    }
    ~Solver() {
        if (staging) CUDA(cudaFreeHost(staging));
        CUDA(cudaEventDestroy(ready));
        CUDA(cudaEventDestroy(received));
        CUDA(cudaStreamDestroy(transfer));
        CUDA(cudaStreamDestroy(compute));
    }

    template<bool Chemical>
    void launch(const double* input, const double* c, double* output, size_t first, size_t last) {
        if (first >= last) return;
        const dim3 threads(32, 4);
        const dim3 blocks((nx+31)/32, std::min<size_t>(65535, (ny+3)/4),
                          std::min<size_t>(65535, last-first));
        stencil<Chemical><<<blocks, threads, 0, compute>>>(input, c, output, nx, ny, depth,
                                                    first, last, rank == 0, rank == ranks-1);
        CUDA(cudaGetLastError());
    }

    template<bool Chemical>
    void phase(double* input, const double* c, double* output) {
        if (ranks == 1) {
            launch<Chemical>(input, c, output, 1, depth+1);
            return;
        }
        MPI_Request requests[4];
        int count = 0;
        // The previous phase may still be reading the pinned receive buffers.
        CUDA(cudaStreamSynchronize(transfer));
        // Receives are posted before device-to-host copies to let MPI progress early.
        if (rank > 0) MPI_Irecv(staging+2*plane, int(plane), MPI_DOUBLE, rank-1, 1, comm, &requests[count++]);
        if (rank+1 < ranks) MPI_Irecv(staging+3*plane, int(plane), MPI_DOUBLE, rank+1, 0, comm, &requests[count++]);
        CUDA(cudaEventRecord(ready, compute));
        CUDA(cudaStreamWaitEvent(transfer, ready, 0));
        if (rank > 0) CUDA(cudaMemcpyAsync(staging, input+plane, plane*sizeof(double), cudaMemcpyDeviceToHost, transfer));
        if (rank+1 < ranks) CUDA(cudaMemcpyAsync(staging+plane, input+depth*plane, plane*sizeof(double), cudaMemcpyDeviceToHost, transfer));
        launch<Chemical>(input, c, output, 2, depth);
        CUDA(cudaStreamSynchronize(transfer));
        if (rank > 0) MPI_Isend(staging, int(plane), MPI_DOUBLE, rank-1, 0, comm, &requests[count++]);
        if (rank+1 < ranks) MPI_Isend(staging+plane, int(plane), MPI_DOUBLE, rank+1, 1, comm, &requests[count++]);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        if (rank > 0) CUDA(cudaMemcpyAsync(input, staging+2*plane, plane*sizeof(double), cudaMemcpyHostToDevice, transfer));
        if (rank+1 < ranks) CUDA(cudaMemcpyAsync(input+(depth+1)*plane, staging+3*plane, plane*sizeof(double), cudaMemcpyHostToDevice, transfer));
        CUDA(cudaEventRecord(received, transfer));
        CUDA(cudaStreamWaitEvent(compute, received, 0));
        launch<Chemical>(input, c, output, 1, 2);
        if (depth > 1) launch<Chemical>(input, c, output, depth, depth+1);
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -x <num>     Grid size in X dimension (default: 64)\n"
           "  -y <num>     Grid size in Y dimension (default: same as X)\n"
           "  -z <num>     Grid size in Z dimension (default: same as X)\n"
           "  -i <num>     Number of time steps (default: 20)\n"
           "  -v           Enable validation\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
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
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i+1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long n = strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end || n > INT_MAX || (option == 'x' && n == 0)) {
                if (!worldRank) fprintf(stderr, "Invalid numeric argument: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            if (option == 'x') nx = n;
            if (option == 'y') ny = n;
            if (option == 'z') nz = n;
            if (option == 'i') iterations = int(n);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!worldRank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (nx > size_t(INT_MAX)/ny || nx*ny > std::numeric_limits<size_t>::max()/sizeof(double)/(nz+2)) {
        if (!worldRank) fprintf(stderr, "Grid exceeds supported allocation or MPI plane size\n");
        MPI_Finalize();
        return 1;
    }
    // Extra ranks have no cells when nz is smaller than the MPI world.
    const int ranks = int(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    int status = 0;
    if (worldRank < ranks) {
        MPI_Comm local;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank, devices;
        MPI_Comm_rank(local, &localRank);
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "A CUDA device is required on every active rank\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        const size_t depth = nz/ranks + (size_t(worldRank) < nz%ranks);
        const size_t offset = size_t(worldRank)*(nz/ranks) + std::min<size_t>(worldRank, nz%ranks);
        const size_t plane = nx*ny, cells = depth*plane, total = nx*ny*nz;
        if (!worldRank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                   nx, ny, nz, iterations, validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        {
            Solver solver(nx, ny, depth, worldRank, ranks, comm);
            double *cold, *cnew, *mu;
            const size_t bytes = (depth+2)*plane*sizeof(double);
            CUDA(cudaMalloc(&cold, bytes));
            CUDA(cudaMalloc(&cnew, bytes));
            CUDA(cudaMalloc(&mu, bytes));
            std::vector<double> host(cells);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < cells; ++i) {
                const size_t linear = offset*plane+i;
                const double pseudo = (((linear+1)*1299709)%total)/static_cast<double>(total);
                host[i] = -1.0 + 2.0*pseudo;
            }
            CUDA(cudaMemcpy(cold+plane, host.data(), cells*sizeof(double), cudaMemcpyHostToDevice));
            if (!worldRank) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                solver.phase<true>(cold, cold, mu);
                solver.phase<false>(mu, cold, cnew);
                std::swap(cold, cnew);
            }
            CUDA(cudaStreamSynchronize(solver.compute));
            const double elapsed = MPI_Wtime()-start;
            double seconds;
            MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!worldRank) {
                printf("Computation time: %lld ms\n", static_cast<long long>(seconds*1000));
                printf("Performance: %.3f MCellUpdates/s\n", double(total)*iterations/seconds/1e6);
            }
            if (validate || printResults) CUDA(cudaMemcpy(host.data(), cold+plane, cells*sizeof(double), cudaMemcpyDeviceToHost));
            if (printResults) {
                // Gather in global linear order, in chunks to avoid MPI int-count overflow.
                std::vector<double> all;
                if (!worldRank) {
                    all.resize(total);
                    std::copy(host.begin(), host.end(), all.begin());
                }
                for (int r = 1; r < ranks; ++r) {
                    const size_t d = nz/ranks + (size_t(r) < nz%ranks);
                    const size_t base = (size_t(r)*(nz/ranks)+std::min<size_t>(r,nz%ranks))*plane;
                    for (size_t pos = 0; pos < d*plane;) {
                        const int chunk = int(std::min<size_t>(INT_MAX, d*plane-pos));
                        if (!worldRank) MPI_Recv(all.data()+base+pos, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                        if (worldRank == r) MPI_Send(host.data()+pos, chunk, MPI_DOUBLE, 0, 2, comm);
                        pos += chunk;
                    }
                }
                if (!worldRank) print_results(all, "Concentration");
            }
            if (validate) {
                double lo = host[0], hi = host[0];
                int invalid = 0;
                #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
                for (size_t i = 0; i < cells; ++i) {
                    invalid |= !std::isfinite(host[i]);
                    lo = std::min(lo, host[i]);
                    hi = std::max(hi, host[i]);
                }
                double globalLo, globalHi;
                int globalInvalid;
                MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
                MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
                MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
                status = globalInvalid || globalLo < -10 || globalHi > 10;
                if (!worldRank) {
                    printf("Validating result...\n");
                    if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                        if (status) printf("Validation failed: values out of expected range\n");
                    }
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            }
            CUDA(cudaFree(cold));
            CUDA(cudaFree(cnew));
            CUDA(cudaFree(mu));
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
