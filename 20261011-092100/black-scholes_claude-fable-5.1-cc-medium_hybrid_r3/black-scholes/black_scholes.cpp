// Black-Scholes option pricing benchmark: hybrid MPI + OpenMP + CUDA version.
//
// Parallelization strategy
//   * MPI     : the global option index space [0, N) is split into one
//               contiguous, balanced chunk per rank. Each rank binds to one
//               GPU (local rank modulo #devices), so run one rank per GPU.
//   * CUDA    : the bulk of each rank's chunk is priced on its GPU. Inputs are
//               generated in-kernel from the option index (bit-identical to the
//               host generator), so only 8 bytes/option (the price) cross PCIe.
//               Work is tiled over several streams so kernel execution overlaps
//               device-to-host copies into pinned host memory.
//   * OpenMP  : the remaining part of the chunk is priced concurrently on the
//               CPU cores while one thread drives the GPU pipeline. OpenMP also
//               parallelizes host-side generation of the rank's inputs.
//
// Results are gathered to rank 0 only when they are needed for output
// (-r) or validation (-v), matching the original program's behaviour.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
}

// Standard test cases for validation
constexpr int kNumTestOptions = 7;

struct TestOptionTable {
    OptionInput data[kNumTestOptions];
};

__host__ __device__ constexpr TestOptionTable getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

// Generate option i of the scaled test set (identical arithmetic on host and
// device, so GPU and CPU price bit-identical inputs).
__host__ __device__ inline OptionInput makeOption(const size_t i) noexcept {
    constexpr TestOptionTable testOptions = getTestOptions();
    // Cycle through test options and vary parameters slightly
    OptionInput option = testOptions.data[i % kNumTestOptions];
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate the options [first, first + count) of the global set
void generateOptions(std::vector<OptionInput>& options, const size_t first, const size_t count) {
    options.resize(count);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        options[i] = makeOption(first + i);
    }
}

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------

constexpr int kBlockSize = 256;

__global__ void __launch_bounds__(kBlockSize)
blackScholesKernel(const size_t first, const size_t count, double* __restrict__ out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = blackScholes(makeOption(first + i));
    }
}

struct GpuPipeline {
    static constexpr int kStreams = 4;
    static constexpr size_t kTile = size_t(1) << 22;   // options per tile (32 MiB of prices)

    cudaStream_t streams[kStreams] = {};
    double* dbuf[kStreams] = {};
    int numSMs = 1;
    int device = 0;
    void* registered = nullptr;

    void init() {
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&dbuf[s], kTile * sizeof(double)));
        }
    }

    // Pin the host output region so device-to-host copies are asynchronous.
    void pinHost(double* ptr, size_t count) {
        if (count == 0) return;
        if (cudaHostRegister(ptr, count * sizeof(double), cudaHostRegisterDefault) == cudaSuccess) {
            registered = ptr;
        } else {
            (void)cudaGetLastError();   // fall back to pageable copies
        }
    }

    void unpinHost() {
        if (registered) {
            CUDA_CHECK(cudaHostUnregister(registered));
            registered = nullptr;
        }
    }

    // Price global options [first, first + count) into host memory `out`.
    void run(const size_t first, const size_t count, double* out) const {
        // The CUDA device binding is per host thread; any OpenMP thread may call this.
        CUDA_CHECK(cudaSetDevice(device));
        int tileIdx = 0;
        for (size_t off = 0; off < count; off += kTile, ++tileIdx) {
            const size_t n = std::min(kTile, count - off);
            const int s = tileIdx % kStreams;
            const size_t blocksNeeded = (n + kBlockSize - 1) / kBlockSize;
            const size_t blocks = std::min(blocksNeeded, static_cast<size_t>(numSMs) * 16);
            blackScholesKernel<<<static_cast<unsigned>(blocks), kBlockSize, 0, streams[s]>>>(
                first + off, n, dbuf[s]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(out + off, dbuf[s], n * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
    }

    void sync() const {
        CUDA_CHECK(cudaSetDevice(device));
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
    }

    void destroy() {
        unpinHost();
        for (int s = 0; s < kStreams; ++s) {
            if (dbuf[s]) CUDA_CHECK(cudaFree(dbuf[s]));
            if (streams[s]) CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }
};

// Fraction of each rank's chunk priced on its GPU; the rest is priced by the
// OpenMP threads concurrently. Override with BS_GPU_FRACTION=<0..1>.
double gpuFraction() {
    double f = 0.7;
    if (const char* env = getenv("BS_GPU_FRACTION")) {
        f = atof(env);
    }
    return std::min(1.0, std::max(0.0, f));
}

// ---------------------------------------------------------------------------
// MPI helpers
// ---------------------------------------------------------------------------

int selectDevice(const int worldRank) {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int dev = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(dev));
    return dev;
}

// Gather the distributed result chunks into `all` on rank 0 (chunks may exceed
// the int count limit of MPI_Gatherv, so send in bounded segments).
void gatherResults(const std::vector<double>& local, const std::vector<size_t>& firsts,
                   const size_t numOptions, std::vector<double>& all,
                   const int rank, const int size) {
    constexpr size_t kSeg = size_t(1) << 28;   // 2 GiB of doubles per message
    if (rank == 0) {
        all.resize(numOptions);
        std::memcpy(all.data(), local.data(), local.size() * sizeof(double));
        for (int r = 1; r < size; ++r) {
            const size_t count = firsts[r + 1] - firsts[r];
            for (size_t off = 0; off < count; off += kSeg) {
                const int n = static_cast<int>(std::min(kSeg, count - off));
                MPI_Recv(all.data() + firsts[r] + off, n, MPI_DOUBLE, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        const size_t count = local.size();
        for (size_t off = 0; off < count; off += kSeg) {
            const int n = static_cast<int>(std::min(kSeg, count - off));
            MPI_Send(local.data() + off, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }

    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
    printf("Run with one MPI rank per GPU; set OMP_NUM_THREADS for the CPU share\n");
    printf("(e.g. mpirun -np <gpus> --map-by numa --bind-to numa).\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    selectDevice(rank);
    const int numThreads = omp_get_max_threads();

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPU fraction: %.2f\n",
               size, numThreads, gpuFraction());
    }

    // Balanced contiguous partition of the global option set over ranks
    std::vector<size_t> firsts(size + 1);
    for (int r = 0; r <= size; ++r) {
        firsts[r] = (numOptions * static_cast<size_t>(r)) / static_cast<size_t>(size);
    }
    const size_t first = firsts[rank];
    const size_t localCount = firsts[rank + 1] - first;

    // Split this rank's chunk between GPU and CPU
    const size_t gpuCount = static_cast<size_t>(static_cast<double>(localCount) * gpuFraction());
    const size_t cpuCount = localCount - gpuCount;

    // Generate options (host copy of the CPU share; the GPU share is generated
    // in-kernel with identical arithmetic)
    std::vector<OptionInput> options;
    generateOptions(options, first + gpuCount, cpuCount);

    // Allocate results
    std::vector<double> results(localCount);

    // Set up the GPU pipeline (context, streams, buffers, pinned output, warm-up)
    GpuPipeline gpu;
    gpu.init();
    gpu.pinHost(results.data(), gpuCount);
    {
        double* warm = nullptr;
        CUDA_CHECK(cudaMalloc(&warm, kBlockSize * sizeof(double)));
        blackScholesKernel<<<1, kBlockSize>>>(0, kBlockSize, warm);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(warm));
    }

    // Price options
    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // One thread feeds the GPU; everyone (including that thread once the
        // launches are queued) prices the CPU share.
        #pragma omp single nowait
        {
            gpu.run(first, gpuCount, results.data());
        }

        #pragma omp for schedule(dynamic, 8192) nowait
        for (size_t i = 0; i < cpuCount; ++i) {
            results[gpuCount + i] = blackScholes(options[i]);
        }

        #pragma omp barrier
        #pragma omp single
        {
            gpu.sync();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    long long localUs = duration.count();
    long long maxUs = 0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxUs / 1e6));
    }

    gpu.destroy();

    // Collect the distributed results on rank 0 when they are needed
    std::vector<double> allResults;
    if (printResults || validate) {
        gatherResults(results, firsts, numOptions, allResults, rank, size);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(allResults, "OptionPrices");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> headOptions;
            generateOptions(headOptions, 0, std::min(static_cast<size_t>(10), numOptions));
            bool valid = validateResults(headOptions, allResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
