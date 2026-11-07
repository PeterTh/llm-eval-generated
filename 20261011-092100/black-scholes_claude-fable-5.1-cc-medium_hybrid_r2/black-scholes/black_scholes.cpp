// Black-Scholes European option pricing benchmark.
//
// Hybrid parallelisation (always active):
//   * MPI    – the global option range is split into contiguous blocks, one per
//              rank. Each rank is pinned to one GPU of its node (local rank
//              modulo device count). Results are gathered on rank 0, which
//              produces exactly the same output as the serial program.
//   * CUDA   – the bulk of each rank's block is priced on the GPU. Option
//              inputs are regenerated on the device from the global index
//              (bit-identical to generateOptions()), so no input upload is
//              needed; results stream back to pinned host memory through a
//              double-buffered, multi-stream pipeline.
//   * OpenMP – the host CPU cores price options concurrently with the GPU.
//              GPU streams and CPU threads pull chunks of the rank's block from
//              one shared work counter, so the GPU/CPU split balances itself
//              dynamically on any hardware mix without calibration.

#include <algorithm>
#include <array>
#include <atomic>
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
#include <sched.h>
#include <unistd.h>

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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
inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
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

constexpr size_t kNumTestOptions = getTestOptions().size();

// Same table, resident in GPU constant memory for device-side generation.
__constant__ OptionInput c_testOptions[kNumTestOptions];

// Generate the i-th option of the (global) data set. Cycles through the test
// options and varies spot/strike slightly. Identical on host and device.
__host__ __device__ inline OptionInput makeOption(const OptionInput* table, const size_t i) noexcept {
    OptionInput o = table[i % kNumTestOptions];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    o.spot *= factor;
    o.strike *= factor;
    return o;
}

// Generate options [first, first + count) of the global data set (OpenMP).
void generateOptions(std::vector<OptionInput>& options, const size_t first, const size_t count) {
    static constexpr auto testOptions = getTestOptions();
    options.resize(count);
    OptionInput* out = options.data();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        out[i] = makeOption(testOptions.data(), first + i);
    }
}

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------

// Prices global options [first, first + count), count < 2^32. The table index
// (first + i) % 7 is derived from a 32-bit remainder to avoid a 64-bit modulo
// per option; the factor uses the same double arithmetic as makeOption().
__global__ void __launch_bounds__(256)
blackScholesKernel(double* __restrict__ out, const size_t first, const unsigned count) {
    const unsigned stride = gridDim.x * blockDim.x;
    const unsigned firstRem = static_cast<unsigned>(first % kNumTestOptions);
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t g = first + i;
        unsigned rem = firstRem + i % static_cast<unsigned>(kNumTestOptions);
        rem = (rem >= kNumTestOptions) ? rem - static_cast<unsigned>(kNumTestOptions) : rem;
        OptionInput o = c_testOptions[rem];
        const double factor = 1.0 + 0.1 * (g / static_cast<double>(kNumTestOptions));
        o.spot *= factor;
        o.strike *= factor;
        out[i] = blackScholes(o);
    }
}

// GPU pricing engine: several streams, each with its own device output
// buffer; results are streamed straight into pinned host memory so kernel
// execution of one chunk overlaps the D2H copy of another.
struct GpuPricer {
    static constexpr int kStreams = 4;
    static constexpr size_t kMaxChunk = size_t(1) << 21;  // 2M options, 16 MiB of output
    static constexpr size_t kMinChunk = size_t(1) << 16;

    cudaStream_t streams[kStreams];
    double* dbuf[kStreams];
    bool busy[kStreams] = {};
    int numSMs = 0;

    void init(int device) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        static constexpr auto testOptions = getTestOptions();
        CUDA_CHECK(cudaMemcpyToSymbol(c_testOptions, testOptions.data(), sizeof(OptionInput) * kNumTestOptions));
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&dbuf[s], kMaxChunk * sizeof(double)));
        }
        // Warm up: module load and first-launch costs stay out of the timed region.
        blackScholesKernel<<<numSMs, 256, 0, streams[0]>>>(dbuf[0], 0, static_cast<unsigned>(numSMs * 256));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }

    // Enqueue pricing of global options [first, first + n) (n <= kMaxChunk)
    // on stream s, with the results copied to host memory `out`.
    void launchChunk(int s, double* out, const size_t first, const size_t n) {
        const unsigned threads = 256;
        unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
        blocks = std::min<unsigned>(blocks, static_cast<unsigned>(numSMs) * 16u);
        blackScholesKernel<<<blocks, threads, 0, streams[s]>>>(dbuf[s], first, static_cast<unsigned>(n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(out, dbuf[s], n * sizeof(double), cudaMemcpyDeviceToHost, streams[s]));
        busy[s] = true;
    }

    // Drive the GPU until the shared work counter is exhausted. Chunks shrink
    // as the remaining work shrinks so the GPU never holds a large piece when
    // the CPU threads run dry (bounded tail imbalance).
    void feed(std::atomic<size_t>& next, double* out, const size_t globalFirst, const size_t total) {
        int s = 0;
        for (;;) {
            if (busy[s]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[s]));
                busy[s] = false;
            }
            const size_t seen = next.load(std::memory_order_relaxed);
            if (seen >= total) break;
            size_t chunk = (total - seen) / (2 * kStreams);
            chunk = std::min(std::max(chunk, kMinChunk), kMaxChunk);
            const size_t off = next.fetch_add(chunk, std::memory_order_relaxed);
            if (off >= total) break;
            const size_t n = std::min(chunk, total - off);
            launchChunk(s, out + off, globalFirst + off, n);
            s = (s + 1) % kStreams;
        }
        for (int i = 0; i < kStreams; ++i) {
            if (busy[i]) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                busy[i] = false;
            }
        }
    }

    void destroy() {
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaFree(dbuf[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }
};

// ---------------------------------------------------------------------------
// Hybrid pricing of the rank's block [globalFirst, globalFirst + total).
// OpenMP thread 0 feeds the GPU; all other threads price CPU-sized chunks.
// Inputs are regenerated on the fly from the global index (a handful of
// flops), so neither engine streams a 72-byte input record per option.
// ---------------------------------------------------------------------------
constexpr size_t kCpuChunk = 4096;

void priceHybrid(GpuPricer& gpu, double* out, const size_t globalFirst, const size_t total) {
    if (total == 0) return;
    static constexpr auto testOptions = getTestOptions();
    const OptionInput* table = testOptions.data();
    std::atomic<size_t> next{0};

#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            gpu.feed(next, out, globalFirst, total);
        } else {
            for (;;) {
                const size_t off = next.fetch_add(kCpuChunk, std::memory_order_relaxed);
                if (off >= total) break;
                const size_t end = std::min(off + kCpuChunk, total);
                for (size_t i = off; i < end; ++i) {
                    out[i] = blackScholes(makeOption(table, globalFirst + i));
                }
            }
        }
    }
}

// Choose the OpenMP thread count from the CPUs this rank may run on. Some
// launchers (e.g. Open MPI for jobs with <= 2 ranks) pin every rank to a
// single core by default, which would starve the host side of the hybrid
// scheme; in that case widen the mask to the node and share the cores among
// the node's ranks. An explicit OMP_NUM_THREADS always wins.
void configureHostThreads(const int rank, const int localSize) {
    if (getenv("OMP_NUM_THREADS") != nullptr) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return;
    const int bound = CPU_COUNT(&set);
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    const int fairShare = static_cast<int>(std::max(1L, online / std::max(1, localSize)));

    if (bound <= 2 && fairShare >= 4 * bound) {
        cpu_set_t all;
        CPU_ZERO(&all);
        for (long c = 0; c < online && c < CPU_SETSIZE; ++c) CPU_SET(c, &all);
        if (sched_setaffinity(0, sizeof(all), &all) == 0) {
            if (rank == 0) {
                fprintf(stderr,
                        "warning: rank was pinned to %d CPU(s) by the launcher; widening affinity to the "
                        "node and using %d OpenMP threads per rank (set OMP_NUM_THREADS or launcher "
                        "binding, e.g. mpirun --bind-to none / --map-by ...:PE=n, to override)\n",
                        bound, fairShare);
            }
            omp_set_num_threads(fairShare);
            return;
        }
    }
    omp_set_num_threads(std::max(1, bound));
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
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int exitCode = -1;

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
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    // --- GPU selection: one device per rank on each node ---------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);
    configureHostThreads(rank, localSize);

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available (%s)\n", rank,
                devErr == cudaSuccess ? "device count is 0" : cudaGetErrorString(devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;

    GpuPricer gpu;
    gpu.init(device);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // --- Domain decomposition: contiguous block per rank ---------------------
    const size_t base = numOptions / static_cast<size_t>(numRanks);
    const size_t rem = numOptions % static_cast<size_t>(numRanks);
    const size_t localCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t localFirst = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    if (localCount > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Too many options per rank for MPI_Gatherv (max %d)\n", INT_MAX);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate results. Rank 0 owns the full result vector and computes its own
    // block in place; other ranks use a local buffer. Both are page-locked so
    // the GPU can DMA straight into them.
    std::vector<double> results(rank == 0 ? numOptions : 0);
    std::vector<double> localResults(rank == 0 ? 0 : localCount);
    double* localOut = (rank == 0) ? results.data() : localResults.data();
    if (localCount > 0) {
        CUDA_CHECK(cudaHostRegister(localOut, localCount * sizeof(double), cudaHostRegisterDefault));
    }

    // Warm up the OpenMP thread pool outside the timed region.
#pragma omp parallel
    {
        volatile int sink = omp_get_thread_num();
        (void)sink;
    }

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    priceHybrid(gpu, localOut, localFirst, localCount);

    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all results on rank 0 (rank 0's block is already in place).
    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t d = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(d);
        }
    }
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, results.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localOut, static_cast<int>(localCount), MPI_DOUBLE, nullptr, nullptr, nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0) {
        const long long micros = static_cast<long long>(maxSeconds * 1e6);
        printf("Computation time: %.3f ms\n", micros / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (micros / 1e6));

        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            // Only the first few options are inspected; regenerate just those.
            std::vector<OptionInput> options;
            generateOptions(options, 0, std::min(static_cast<size_t>(10), numOptions));
            valid = validateResults(options, results) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (localCount > 0) {
        CUDA_CHECK(cudaHostUnregister(localOut));
    }
    gpu.destroy();
    MPI_Finalize();
    return (validate && !valid) ? 1 : 0;
}
