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

// MPI calls stay on the main thread (MPI_THREAD_FUNNELED).
static int worldRank;
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: MPI: %.*s\n", worldRank, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Owned planes are 1..depth; planes 0 and depth+1 are MPI halos.
// Global boundary planes clamp to their center, exactly as in the serial code.
__device__ __forceinline__ double laplacian(const double* __restrict__ a,
        size_t i, size_t x, size_t y, size_t z, size_t nx, size_t ny,
        size_t depth, bool first, bool last) {
    const size_t plane = nx * ny;
    const double center = a[i];
    const double twice = __dmul_rn(2.0, center);
    const double xx = __dsub_rn(__dadd_rn(a[i + (x + 1 < nx ? 1 : 0)], a[i - (x ? 1 : 0)]), twice);
    const double yy = __dsub_rn(__dadd_rn(a[i + (y + 1 < ny ? nx : 0)], a[i - (y ? nx : 0)]), twice);
    const double zz = __dsub_rn(__dadd_rn(a[i + (last && z == depth ? 0 : plane)],
                                      a[i - (first && z == 1 ? 0 : plane)]), twice);
    return __dadd_rn(__dadd_rn(xx, yy), zz); // Unit spacing.

}

template<bool Chemical, bool Boundary>
__global__ void stencil(const double* __restrict__ input, double* __restrict__ output,
        const double* old, size_t nx, size_t ny, size_t depth,
        bool first, bool last, size_t begin, size_t planes) {
    const size_t plane = nx * ny;
    const size_t count = plane * planes;
    for (size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         t < count; t += size_t(blockDim.x) * gridDim.x) {
        const size_t p = t / plane;
        const size_t z = Boundary ? (p == 0 ? 1 : depth) : begin + p;
        const size_t xy = t - p * plane;
        const size_t x = xy % nx, y = xy / nx;
        const size_t i = z * plane + xy;
        const double cv = input[i];
        const double lap = laplacian(input, i, x, y, z, nx, ny, depth, first, last);
        if (Chemical) {
            constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
            // Explicit rounding also prevents boundary/interior specializations
            // from applying different algebraic transformations.
            const double energy = __dsub_rn(
                __dadd_rn(__dmul_rn(__dadd_rn(cv, 1.0), eAA),
                          __dmul_rn(__dsub_rn(cv, 1.0), eBB)),
                __dmul_rn(__dmul_rn(2.0, cv), eAB));
            output[i] = __dsub_rn(__dadd_rn(
                __dadd_rn(__dmul_rn(4.5, energy), __dmul_rn(3.0, cv)),
                __dmul_rn(__dmul_rn(cv, cv), cv)), __dmul_rn(0.5, lap));
        } else {
            output[i] = __dadd_rn(old[i], __dmul_rn(0.01, lap));
        }
    }
}

struct Solver {
    size_t nx, ny, depth, plane;
    int rank, ranks;
    MPI_Comm comm;
    double *a = nullptr, *b = nullptr, *mu = nullptr, *halo = nullptr;
    cudaStream_t compute, transfer;

    Solver(size_t x, size_t y, size_t d, int r, int n, MPI_Comm c)
        : nx(x), ny(y), depth(d), plane(x*y), rank(r), ranks(n), comm(c) {
        const size_t bytes = (depth + 2) * plane * sizeof(double);
        cudaCheck(cudaMalloc(&a, bytes));
        cudaCheck(cudaMalloc(&b, bytes));
        cudaCheck(cudaMalloc(&mu, bytes));
        cudaCheck(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
        cudaCheck(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
        if (ranks > 1) cudaCheck(cudaMallocHost(&halo, 4 * plane * sizeof(double)));
    }
    ~Solver() {
        cudaCheck(cudaFree(a)); cudaCheck(cudaFree(b)); cudaCheck(cudaFree(mu));
        if (halo) cudaCheck(cudaFreeHost(halo));
        cudaCheck(cudaStreamDestroy(compute)); cudaCheck(cudaStreamDestroy(transfer));
    }
    template<bool Chemical, bool Boundary>
    void launch(const double* input, double* output, const double* old,
                size_t begin, size_t planes, cudaStream_t stream) {
        if (!planes) return;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>((planes * plane + 255) / 256, 65535));
        stencil<Chemical, Boundary><<<blocks, 256, 0, stream>>>(input, output, old,
                nx, ny, depth, rank == 0, rank == ranks - 1, begin, planes);
        cudaCheck(cudaGetLastError());
    }

    template<bool Chemical>
    void distributedStage(double* input, double* output, const double* old) {
        // Two outgoing and two incoming planes; only halos leave the GPU.
        const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
        const int upper = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
        MPI_Request requests[4];
        int count = 0;
        if (lower != MPI_PROC_NULL)
            mpiCheck(MPI_Irecv(halo + 2*plane, static_cast<int>(plane), MPI_DOUBLE,
                               lower, 1, comm, &requests[count++]));
        if (upper != MPI_PROC_NULL)
            mpiCheck(MPI_Irecv(halo + 3*plane, static_cast<int>(plane), MPI_DOUBLE,
                               upper, 0, comm, &requests[count++]));
        if (lower != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(halo, input + plane, plane*sizeof(double), cudaMemcpyDeviceToHost, transfer));
        if (upper != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(halo + plane, input + depth*plane, plane*sizeof(double), cudaMemcpyDeviceToHost, transfer));
        launch<Chemical, false>(input, output, old, 2, depth > 2 ? depth - 2 : 0, compute);
        cudaCheck(cudaStreamSynchronize(transfer));
        if (lower != MPI_PROC_NULL)
            mpiCheck(MPI_Isend(halo, static_cast<int>(plane), MPI_DOUBLE, lower, 0, comm, &requests[count++]));
        if (upper != MPI_PROC_NULL)
            mpiCheck(MPI_Isend(halo + plane, static_cast<int>(plane), MPI_DOUBLE, upper, 1, comm, &requests[count++]));
        // MPI makes progress while CUDA computes the interior.
        mpiCheck(MPI_Waitall(count, requests, MPI_STATUSES_IGNORE));
        if (lower != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(input, halo + 2*plane, plane*sizeof(double), cudaMemcpyHostToDevice, transfer));
        if (upper != MPI_PROC_NULL)
            cudaCheck(cudaMemcpyAsync(input + (depth+1)*plane, halo + 3*plane, plane*sizeof(double), cudaMemcpyHostToDevice, transfer));
        cudaCheck(cudaStreamSynchronize(transfer));
        launch<Chemical, true>(input, output, old, 1, depth == 1 ? 1 : 2, compute);
        // Both regions are ready before the next stencil consumes this field;
        // the incoming pinned buffers can also be reused safely.
        cudaCheck(cudaStreamSynchronize(compute));
    }

    void run(int iterations) {
        if (ranks == 1 && iterations >= 2) {
            // Capture two steps so both concentration buffers return to their
            // starting roles. Replay removes per-stencil host launch overhead.
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            cudaCheck(cudaStreamBeginCapture(compute, cudaStreamCaptureModeThreadLocal));
            for (int k = 0; k < 2; ++k) {
                launch<true, false>(a, mu, a, 1, depth, compute);
                launch<false, false>(mu, b, a, 1, depth, compute);
                std::swap(a, b);
            }
            cudaCheck(cudaStreamEndCapture(compute, &graph));
            cudaCheck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (int t = 0; t < iterations / 2; ++t)
                cudaCheck(cudaGraphLaunch(executable, compute));
            cudaCheck(cudaStreamSynchronize(compute));
            cudaCheck(cudaGraphExecDestroy(executable));
            cudaCheck(cudaGraphDestroy(graph));
            iterations %= 2;
        }
        for (int t = 0; t < iterations; ++t) {
            if (ranks == 1) {
                launch<true, false>(a, mu, a, 1, depth, compute);
                launch<false, false>(mu, b, a, 1, depth, compute);
            } else {
                distributedStage<true>(a, mu, a);
                distributedStage<false>(mu, b, a);
            }
            std::swap(a, b);
        }
        cudaCheck(cudaStreamSynchronize(compute));
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
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    int worldSize;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    bool bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                  !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i+1 < argc) {
            const char option = argv[i][1];
            char* end;
            const char* text = argv[++i];
            const long long value = strtoll(text, &end, 10);
            if (*end || end == text || value < 0 ||
                (option == 'x' && value == 0) ||
                (option == 'i' && value > INT_MAX)) { bad = true; break; }
            if (option == 'x') nx = static_cast<size_t>(value);
            if (option == 'y') ny = static_cast<size_t>(value);
            if (option == 'z') nz = static_cast<size_t>(value);
            if (option == 'i') iterations = static_cast<int>(value);
        } else { bad = true; break; }
    }
    if (help || bad) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad ? 1 : 0;
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx > limit / ny || nx*ny > limit / nz ||
        nx*ny > limit / 4 || (worldSize > 1 && nx*ny > INT_MAX)) {
        if (!worldRank) fprintf(stderr, "Grid dimensions exceed supported allocation or MPI message size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx*ny, volume = plane*nz;
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node));
    int nodeRank, nodeSize;
    mpiCheck(MPI_Comm_rank(node, &nodeRank));
    mpiCheck(MPI_Comm_size(node, &nodeSize));
    mpiCheck(MPI_Comm_free(&node));
    if (!getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs()/nodeSize)));

    // Empty slabs do not enter halo exchanges, but still participate in finalization.
    const int ranks = static_cast<int>(std::min<size_t>(worldSize, nz));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (comm == MPI_COMM_NULL) { MPI_Finalize(); return 0; }
    int devices;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(nodeRank % devices));
    const size_t depth = nz / ranks + (static_cast<size_t>(worldRank) < nz % ranks);
    const size_t offset = (nz / ranks)*worldRank + std::min<size_t>(worldRank, nz % ranks);
    if (depth > limit/plane - 2) MPI_Abort(MPI_COMM_WORLD, 1);
    int result = 0;
    {
        Solver solver(nx, ny, depth, worldRank, ranks, comm);
        std::vector<double> local(depth*plane);
        if (!worldRank) {
            printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local.size(); ++i) {
            const size_t id = offset*plane + i;
            const double pseudo = (((id + 1)*1299709) % volume) / static_cast<double>(volume);
            local[i] = -1.0 + 2.0*pseudo;
        }
        // Pageable H2D cudaMemcpy may return before the device transfer is
        // complete. Explicitly order initialization on our nonblocking stream.
        cudaCheck(cudaMemcpyAsync(solver.a + plane, local.data(), local.size()*sizeof(double),
                                  cudaMemcpyHostToDevice, solver.compute));
        cudaCheck(cudaStreamSynchronize(solver.compute));
        if (!worldRank) printf("Running Cahn-Hilliard simulation...\n");
        mpiCheck(MPI_Barrier(comm));
        const double start = MPI_Wtime();
        solver.run(iterations);
        double elapsed = MPI_Wtime() - start, maximum;
        mpiCheck(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (!worldRank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(maximum*1000));
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(volume)*iterations/maximum/1e6);
        }
        if (validate || printResults)
            cudaCheck(cudaMemcpy(local.data(), solver.a + plane, local.size()*sizeof(double), cudaMemcpyDeviceToHost));
        if (printResults) {
            // Ordered, chunked gathering preserves the original global result
            // hash and Kahan sum without imposing MPI_Gatherv's int offsets.
            std::vector<double> global;
            if (!worldRank) {
                global.resize(volume);
                std::copy(local.begin(), local.end(), global.begin());
            }
            for (int r = 1; r < ranks; ++r) {
                const size_t d = nz/ranks + (static_cast<size_t>(r) < nz%ranks);
                const size_t off = ((nz/ranks)*r + std::min<size_t>(r, nz%ranks))*plane;
                for (size_t pos = 0; pos < d*plane;) {
                    const int count = static_cast<int>(std::min<size_t>(d*plane-pos, INT_MAX));
                    if (!worldRank) mpiCheck(MPI_Recv(global.data()+off+pos, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE));
                    else if (worldRank == r) mpiCheck(MPI_Send(local.data()+pos, count, MPI_DOUBLE, 0, 2, comm));
                    pos += count;
                }
            }
            if (!worldRank) print_results(global, "Concentration");
        }
        if (validate) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int invalid = 0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:invalid) schedule(static)
            for (size_t i = 0; i < local.size(); ++i) {
                invalid |= !std::isfinite(local[i]);
                lo = std::min(lo, local[i]); hi = std::max(hi, local[i]);
            }
            double globalLo, globalHi;
            int globalInvalid;
            mpiCheck(MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm));
            mpiCheck(MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm));
            mpiCheck(MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm));
            result = globalInvalid || globalLo < -10.0 || globalHi > 10.0;
            if (!worldRank) {
                printf("Validating result...\n");
                if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                    if (result) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
    }
    mpiCheck(MPI_Comm_free(&comm));
    MPI_Finalize();
    return result;
}
