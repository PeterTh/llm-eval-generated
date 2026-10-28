// Black-Scholes option pricing benchmark.
//
// Hybrid parallelization:
//   * MPI    - options are block-distributed across ranks. Each rank generates
//              its own options from their global index (no option data is
//              communicated); only results needed for output/validation are
//              gathered on rank 0.
//   * OpenMP - each rank runs one host thread per GPU it drives. Each thread
//              owns a CUDA device and prices its sub-range independently.
//   * CUDA   - a double-precision kernel prices options on the device. Work is
//              split into chunks that are double-buffered across two streams.
//              When results are observed (-r / -v) they are copied back through
//              pinned staging buffers, overlapped with compute; otherwise they
//              remain on the device and no transfers are performed.
//
// The same formula is evaluated in double precision for every option, and the
// option set, result ordering, and printed output format are unchanged. Device
// transcendental functions (erf/exp/log) may differ from host libm in the last
// bits, so results match the CPU reference to within a few ulps (relative
// error <~1e-13 for heavily cancelling cases), not necessarily bit-for-bit.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
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

constexpr size_t kTestCount = 7;

// Standard test cases, passed by value to kernels (device-visible copy).
struct TestSet {
    OptionInput opts[kTestCount];
};

constexpr TestSet makeTestSet() noexcept {
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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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

// Option i is derived from test case (i % 7), scaled by a factor that grows
// with i / 7. Evaluated identically on host (validation) and device (pricing).
__host__ __device__ inline OptionInput makeOption(const TestSet& tests, const size_t i) noexcept {
    OptionInput option = tests.opts[i % kTestCount];
    const double factor = 1.0 + 0.1 * (static_cast<double>(i) / static_cast<double>(kTestCount));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Grid-stride kernel: out[i] = price of option (offset + i).
__global__ void priceKernel(double* __restrict__ out, const size_t offset,
                            const size_t count, const TestSet tests) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = blackScholes(makeOption(tests, offset + i));
    }
}

// Contiguous block [begin, begin + count) of n items assigned to part idx of parts.
struct Range {
    size_t begin;
    size_t count;
};

inline Range blockRange(const size_t n, const int parts, const int idx) noexcept {
    const size_t p = static_cast<size_t>(parts);
    const size_t r = static_cast<size_t>(idx);
    const size_t base = n / p;
    const size_t rem = n % p;
    return {r * base + std::min(r, rem), base + (r < rem ? 1 : 0)};
}

// Per-thread GPU state: two streams, device buffers, and (only when results
// are needed on the host) pinned staging buffers.
constexpr size_t kChunk = size_t{1} << 22;  // options per chunk

struct GpuWorker {
    cudaStream_t stream[2] = {};
    double* dev[2] = {};
    double* pinned[2] = {};
    size_t cap = 0;
    bool keepResults = false;
};

void setupWorker(GpuWorker& w, const int device, const size_t maxCount, const bool keepResults) {
    CUDA_CHECK(cudaSetDevice(device));  // also creates the context before timing
    w.cap = std::min(kChunk, maxCount);
    w.keepResults = keepResults;
    if (w.cap == 0) return;
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaStreamCreate(&w.stream[k]));
        CUDA_CHECK(cudaMalloc(&w.dev[k], w.cap * sizeof(double)));
        if (keepResults) {
            CUDA_CHECK(cudaMallocHost(&w.pinned[k], w.cap * sizeof(double)));
        }
    }
}

void teardownWorker(GpuWorker& w) {
    if (w.cap == 0) return;
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaStreamDestroy(w.stream[k]));
        CUDA_CHECK(cudaFree(w.dev[k]));
        if (w.keepResults) CUDA_CHECK(cudaFreeHost(w.pinned[k]));
    }
}

// Price `count` options starting at global index `offset`.
// If the worker keeps results, out[0..count) receives the prices; otherwise
// the prices are computed on the device and not transferred to the host.
// Must be called from the thread that owns the worker's device.
void priceRange(GpuWorker& w, double* out, const size_t offset, const size_t count,
                const TestSet& tests) {
    if (count == 0) return;

    const size_t numChunks = (count + kChunk - 1) / kChunk;
    auto chunkStart = [](size_t c) { return c * kChunk; };
    auto chunkLen = [&](size_t c) { return std::min(kChunk, count - chunkStart(c)); };

    // Wait for chunk c to finish on its stream, then copy it into the output.
    auto drain = [&](size_t c) {
        const int k = static_cast<int>(c & 1);
        CUDA_CHECK(cudaStreamSynchronize(w.stream[k]));
        if (w.keepResults) {
            std::memcpy(out + chunkStart(c), w.pinned[k], chunkLen(c) * sizeof(double));
        }
    };

    for (size_t c = 0; c < numChunks; ++c) {
        const int k = static_cast<int>(c & 1);
        if (w.keepResults && c >= 2) drain(c - 2);  // slot k is free once chunk c-2 is drained

        const size_t len = chunkLen(c);
        const unsigned int blocks =
            static_cast<unsigned int>(std::min<size_t>((len + 255) / 256, 65535));
        priceKernel<<<blocks, 256, 0, w.stream[k]>>>(w.dev[k], offset + chunkStart(c), len, tests);
        CUDA_CHECK(cudaGetLastError());
        if (w.keepResults) {
            CUDA_CHECK(cudaMemcpyAsync(w.pinned[k], w.dev[k], len * sizeof(double),
                                       cudaMemcpyDeviceToHost, w.stream[k]));
        }
    }

    // Complete the last (up to two) chunks; this also synchronizes both streams.
    for (size_t c = (numChunks > 2 ? numChunks - 2 : 0); c < numChunks; ++c) {
        drain(c);
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
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Map ranks on this node onto GPUs. If a node has at least as many ranks
    // as GPUs, ranks share devices round-robin with one thread each. Otherwise
    // each rank drives devCount / localSize GPUs with one OpenMP thread per GPU.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int numThreads = 1;
    int firstDevice = 0;
    if (localSize <= devCount) {
        numThreads = devCount / localSize;
        firstDevice = localRank * numThreads;
    } else {
        firstDevice = localRank % devCount;
    }

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    const TestSet tests = makeTestSet();
    const Range myRange = blockRange(numOptions, size, rank);

    // Gather only what is needed on rank 0: all results for -r, otherwise the
    // first min(N, 10) results for validation. Results are only materialized on
    // the host when they will be observed; otherwise they stay on the devices.
    const bool needResults = printResults || validate;
    const size_t needed = printResults ? numOptions : std::min<size_t>(numOptions, 10);

    // Each rank's results for its own block (allocation is outside the timed region).
    std::vector<double> localResults(needResults ? myRange.count : 0);
    double* const localOut = needResults ? localResults.data() : nullptr;

    double tStart = 0.0;
    double tEnd = 0.0;
    #pragma omp parallel num_threads(numThreads)
    {
        const int t = omp_get_thread_num();
        const Range tr = blockRange(myRange.count, numThreads, t);
        const int device = (firstDevice + t) % devCount;

        // Device setup (context creation, buffers) is excluded from timing.
        GpuWorker worker;
        setupWorker(worker, device, tr.count, needResults);

        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            tStart = MPI_Wtime();
        }
        #pragma omp barrier

        priceRange(worker, localOut ? localOut + tr.begin : nullptr,
                   myRange.begin + tr.begin, tr.count, tests);

        #pragma omp barrier
        #pragma omp master
        {
            tEnd = MPI_Wtime();
        }
        #pragma omp barrier

        teardownWorker(worker);
    }
    const double localElapsed = tEnd - tStart;

    // Wall time of the slowest rank
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long long durationUs = std::llround(elapsed * 1e6);
        printf("Computation time: %.3f ms\n", durationUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (durationUs / 1e6));
    }

    std::vector<double> results;
    if (needResults) {
        if (needed > static_cast<size_t>(INT_MAX)) {
            if (root) fprintf(stderr, "Too many options to gather for output\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const Range rr = blockRange(numOptions, size, r);
            const size_t c = rr.begin < needed ? std::min(rr.count, needed - rr.begin) : 0;
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(std::min(rr.begin, needed));
        }
        if (root) results.resize(needed);
        MPI_Gatherv(localResults.data(), counts[rank], MPI_DOUBLE,
                    results.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (root) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> reference;
            const size_t numChecks = std::min<size_t>(numOptions, 10);
            reference.reserve(numChecks);
            for (size_t i = 0; i < numChecks; ++i) {
                reference.push_back(makeOption(tests, i));
            }
            bool valid = validateResults(reference, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
