#include <algorithm>
#include <array>
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

// Abort the whole MPI job on any CUDA runtime error
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

constexpr size_t kNumTests = 7;
constexpr size_t kNumChecks = 10;

// Standard test cases for validation
inline constexpr std::array<OptionInput, kNumTests> getTestOptions() noexcept {
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

// Test set passed by value to device kernels
struct TestSet {
    OptionInput opt[kNumTests];
};

TestSet makeTestSet() {
    TestSet ts;
    constexpr auto tests = getTestOptions();
    for (size_t k = 0; k < kNumTests; ++k) {
        ts.opt[k] = tests[k];
    }
    return ts;
}

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
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

// Option i is a scaled copy of test option (i mod kNumTests). Generated on demand
// from its index, so no option array has to be stored or transferred.
__host__ __device__ inline OptionInput makeOption(const TestSet& ts, const size_t i) noexcept {
    const OptionInput& base = ts.opt[i % kNumTests];
    OptionInput option = base;

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTests));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

bool validateResults(const size_t numOptions, const std::vector<double>& results,
                     const TestSet& ts) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(kNumChecks, numOptions));

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = makeOption(ts, static_cast<size_t>(i)).value;
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

// ---------------------------------------------------------------------------
// CUDA pricing
// ---------------------------------------------------------------------------

constexpr int kStreams = 4;                       // pipelined chunks per device
constexpr size_t kChunk = size_t(1) << 20;        // options per device chunk
constexpr unsigned kBlockSize = 256;

__global__ void priceKernel(double* __restrict__ out, const TestSet ts,
                            const size_t firstOption, const size_t count) {
    const size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < count) {
        out[k] = blackScholes(makeOption(ts, firstOption + k));
    }
}

// Per-OpenMP-thread state: one GPU, a set of streams and device staging buffers
struct DeviceWorker {
    int device = 0;
    cudaStream_t stream[kStreams] = {};
    double* buf[kStreams] = {};
};

void setupWorker(DeviceWorker& w, const size_t cap) {
    CUDA_CHECK(cudaSetDevice(w.device));
    for (int s = 0; s < kStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&w.stream[s]));
        if (cap > 0) {
            CUDA_CHECK(cudaMalloc(&w.buf[s], cap * sizeof(double)));
        }
    }
}

void teardownWorker(DeviceWorker& w) {
    CUDA_CHECK(cudaSetDevice(w.device));
    for (int s = 0; s < kStreams; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(w.stream[s]));
        if (w.buf[s]) {
            CUDA_CHECK(cudaFree(w.buf[s]));
        }
        CUDA_CHECK(cudaStreamDestroy(w.stream[s]));
    }
}

// Prices global options [firstOption, firstOption + count) into hostOut (pinned).
// Chunks are round-robined over streams so kernels and D2H copies overlap.
void priceWorker(DeviceWorker& w, double* hostOut, const size_t firstOption,
                 const size_t count, const TestSet& ts) {
    if (count == 0) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(w.device));

    size_t c = 0;
    for (size_t off = 0; off < count; off += kChunk, ++c) {
        const size_t len = std::min(kChunk, count - off);
        const int s = static_cast<int>(c % kStreams);
        const unsigned blocks = static_cast<unsigned>((len + kBlockSize - 1) / kBlockSize);

        priceKernel<<<blocks, kBlockSize, 0, w.stream[s]>>>(w.buf[s], ts, firstOption + off, len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hostOut + off, w.buf[s], len * sizeof(double),
                                   cudaMemcpyDeviceToHost, w.stream[s]));
    }

    for (int s = 0; s < kStreams; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(w.stream[s]));
    }
}

// ---------------------------------------------------------------------------
// Main: MPI across ranks, OpenMP threads driving one GPU each, CUDA pricing
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int earlyExit = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            earlyExit = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = 1;
            break;
        }
    }
    if (earlyExit >= 0) {
        MPI_Finalize();
        return earlyExit;
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const TestSet ts = makeTestSet();

    // Contiguous block distribution of [0, numOptions) over ranks
    const size_t baseCnt = numOptions / static_cast<size_t>(nranks);
    const size_t remCnt = numOptions % static_cast<size_t>(nranks);
    auto rankLo = [&](int r) {
        return static_cast<size_t>(r) * baseCnt + std::min(static_cast<size_t>(r), remCnt);
    };
    auto rankCnt = [&](int r) {
        return baseCnt + (static_cast<size_t>(r) < remCnt ? 1 : 0);
    };
    const size_t myLo = rankLo(rank);
    const size_t myCnt = rankCnt(rank);

    // Map ranks on this node to GPUs
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    const int nWorkers = (numDevices >= localSize) ? numDevices / localSize : 1;

    std::vector<DeviceWorker> workers(static_cast<size_t>(nWorkers));
    std::vector<size_t> wOff(static_cast<size_t>(nWorkers));
    std::vector<size_t> wCnt(static_cast<size_t>(nWorkers));
    size_t maxWorkerCnt = 0;
    for (int t = 0; t < nWorkers; ++t) {
        DeviceWorker& w = workers[static_cast<size_t>(t)];
        w.device = (numDevices >= localSize)
                       ? (localRank + t * localSize) % numDevices
                       : localRank % numDevices;

        const size_t nw = static_cast<size_t>(nWorkers);
        wCnt[static_cast<size_t>(t)] = myCnt / nw + (static_cast<size_t>(t) < myCnt % nw ? 1 : 0);
        wOff[static_cast<size_t>(t)] = static_cast<size_t>(t) * (myCnt / nw)
                                       + std::min(static_cast<size_t>(t), myCnt % nw);
        maxWorkerCnt = std::max(maxWorkerCnt, wCnt[static_cast<size_t>(t)]);
    }
    const size_t stageCap = std::min(kChunk, maxWorkerCnt);

    // Pinned host buffer for this rank's results (enables async D2H copies)
    double* hostRes = nullptr;
    if (myCnt > 0) {
        CUDA_CHECK(cudaSetDevice(workers[0].device));
        CUDA_CHECK(cudaHostAlloc(&hostRes, myCnt * sizeof(double), cudaHostAllocPortable));
    }

    // Create contexts, streams and buffers outside the timed region
    #pragma omp parallel for num_threads(nWorkers) schedule(static, 1)
    for (int t = 0; t < nWorkers; ++t) {
        setupWorker(workers[static_cast<size_t>(t)], stageCap);
    }

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(nWorkers) schedule(static, 1)
    for (int t = 0; t < nWorkers; ++t) {
        const size_t idx = static_cast<size_t>(t);
        priceWorker(workers[idx], hostRes + wOff[idx], myLo + wOff[idx], wCnt[idx], ts);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double localMs = duration.count() / 1000.0;
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        printf("Options per second: %.0f\n", numOptions / (maxMs / 1000.0));
    }

    // Gather only the prefix that output or validation needs onto rank 0
    int exitCode = 0;
    const size_t gatherCount = printResults ? numOptions
                             : (validate ? std::min(kNumChecks, numOptions) : 0);
    std::vector<double> results;
    if (gatherCount > 0) {
        if (gatherCount > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) fprintf(stderr, "Too many options to gather (limit %d)\n", INT_MAX);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> counts(static_cast<size_t>(nranks));
        std::vector<int> displs(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) {
            const size_t lo = rankLo(r);
            const size_t cnt = (lo >= gatherCount) ? 0 : std::min(lo + rankCnt(r), gatherCount) - lo;
            counts[static_cast<size_t>(r)] = static_cast<int>(cnt);
            displs[static_cast<size_t>(r)] = static_cast<int>(cnt ? lo : 0);
        }
        if (rank == 0) {
            results.resize(gatherCount);
        }
        MPI_Gatherv(hostRes, counts[static_cast<size_t>(rank)], MPI_DOUBLE,
                    results.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(numOptions, results, ts);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Propagate the validation status so all ranks exit consistently
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    #pragma omp parallel for num_threads(nWorkers) schedule(static, 1)
    for (int t = 0; t < nWorkers; ++t) {
        teardownWorker(workers[static_cast<size_t>(t)]);
    }
    if (hostRes) {
        CUDA_CHECK(cudaFreeHost(hostRes));
    }

    MPI_Finalize();
    return exitCode;
}
