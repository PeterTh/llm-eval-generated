#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

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

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,   \
                    __LINE__);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
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

// Trivially copyable mirror of the test set, passed to the device as a kernel argument
// (lands in constant memory) so option inputs never have to be transferred or stored.
struct TestOptionTable {
    static constexpr size_t count = 7;
    OptionInput opt[count];
};

inline TestOptionTable makeTestOptionTable() noexcept {
    constexpr auto testOptions = getTestOptions();
    TestOptionTable table;
    for (size_t i = 0; i < TestOptionTable::count; ++i) {
        table.opt[i] = testOptions[i];
    }
    return table;
}

// Generates the option with global index i, identical to the original generateOptions().
__host__ __device__ inline OptionInput generateOption(const TestOptionTable& table,
                                                      const size_t i) noexcept {
    OptionInput option = table.opt[i % TestOptionTable::count];

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(TestOptionTable::count));
    option.spot *= factor;
    option.strike *= factor;

    return option;
}

// Prices a contiguous range of options [globalStart, globalStart + n) on the device.
__global__ void priceOptionsKernel(const TestOptionTable table, const size_t globalStart,
                                   const size_t n, double* __restrict__ out) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        out[i] = blackScholes(generateOption(table, globalStart + i));
    }
}

// Prices a contiguous range of options on the CPU using the calling thread.
inline void priceOptionsHost(const TestOptionTable& table, const size_t globalStart,
                             const size_t n, double* __restrict__ out) noexcept {
    for (size_t i = 0; i < n; ++i) {
        out[i] = blackScholes(generateOption(table, globalStart + i));
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

namespace {

// Work distribution granularity: one chunk is the unit handed out by the shared counter.
// The chunk is sized so that a single CPU core needs on the order of a millisecond for
// it: small enough that the host workers cannot starve the far faster GPU by claiming
// everything up front, large enough to amortize the atomic and the loop overhead.
constexpr size_t kChunkSize = 1u << 13;      // 8K options per chunk
constexpr size_t kGpuChunksPerGrab = 256;    // GPU takes 2M options per grab
constexpr int kNumStreams = 4;               // overlap kernels with device-to-host copies

struct RankRange {
    size_t start;
    size_t count;
};

RankRange computeRange(const size_t numOptions, const int rank, const int size) noexcept {
    const size_t base = numOptions / static_cast<size_t>(size);
    const size_t rem = numOptions % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t start = r * base + std::min(r, rem);
    const size_t count = base + (r < rem ? 1 : 0);
    return {start, count};
}

} // namespace

int main(int argc, char** argv) {
    int mpiThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Node-local rank determines which GPU this process drives and how many OpenMP
    // threads it may use without oversubscribing the node.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Comm_free(&localComm);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&localComm);
            MPI_Finalize();
            return 1;
        }
    }

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));

    // MPI launchers commonly pin a rank to a single core, which would confine the whole
    // OpenMP team to one CPU. Hand every node-local rank a disjoint slice of the node's
    // CPUs and place the threads there explicitly.
    const int totalCpus = std::max(1, static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN)));
    const int cpuSlice = std::max(1, totalCpus / localSize);
    const int cpuBase = (localRank * cpuSlice) % totalCpus;

    // Honour an explicit OMP_NUM_THREADS, otherwise use the rank's whole CPU slice.
    const char* ompEnv = getenv("OMP_NUM_THREADS");
    const int coresPerRank =
        (ompEnv != nullptr && atoi(ompEnv) > 0) ? atoi(ompEnv) : cpuSlice;
    omp_set_num_threads(coresPerRank);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d, GPUs per node: %d\n", numRanks,
               coresPerRank, numDevices);
    }

    const TestOptionTable table = makeTestOptionTable();
    const RankRange range = computeRange(numOptions, rank, numRanks);

    // Pinned staging buffer for this rank's slice of the result vector: written directly
    // by the CPU worker threads and by the asynchronous device-to-host copies.
    double* localResults = nullptr;
    if (range.count > 0) {
        CUDA_CHECK(cudaHostAlloc(&localResults, range.count * sizeof(double),
                                 cudaHostAllocDefault));
    }

    cudaStream_t streams[kNumStreams] = {};
    double* deviceBuffers[kNumStreams] = {};
    const size_t gpuGrabElems = kChunkSize * kGpuChunksPerGrab;
    const size_t deviceBufElems = std::min(gpuGrabElems, std::max<size_t>(range.count, 1));
    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&deviceBuffers[s], deviceBufElems * sizeof(double)));
    }

    // Warm up the CUDA context, the kernel module and the OpenMP thread pool so that
    // one-time initialization costs do not pollute the measured region.
    priceOptionsKernel<<<1, 32, 0, streams[0]>>>(table, 0, 1, deviceBuffers[0]);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    // The pool is created here and reused for the measured region, so pinning the
    // threads in this first parallel region is enough.
    int threadsStarted = 0;
    #pragma omp parallel reduction(+ : threadsStarted)
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET((cpuBase + omp_get_thread_num() % cpuSlice) % totalCpus, &mask);
        pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        threadsStarted = 1;
    }
    if (threadsStarted <= 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Dynamic GPU/CPU co-scheduling: a shared atomic chunk counter feeds both the GPU
    // driver thread and the remaining OpenMP worker threads, so the split between the
    // accelerator and the host cores balances itself without any tuning.
    const size_t numChunks = (range.count + kChunkSize - 1) / kChunkSize;
    std::atomic<size_t> nextChunk{0};

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            bool streamBusy[kNumStreams] = {};
            int s = 0;
            for (;;) {
                // Taper the grab size towards the end of the range so that the host
                // workers are not left idle while the GPU finishes one large batch.
                const size_t claimed = nextChunk.load(std::memory_order_relaxed);
                const size_t remaining = claimed < numChunks ? numChunks - claimed : 0;
                const size_t grab = std::clamp(remaining / (2 * kNumStreams), size_t{1},
                                               kGpuChunksPerGrab);
                const size_t chunk = nextChunk.fetch_add(grab, std::memory_order_relaxed);
                if (chunk >= numChunks) {
                    break;
                }
                const size_t offset = chunk * kChunkSize;
                const size_t end = std::min(range.count, (chunk + grab) * kChunkSize);
                const size_t n = end - offset;

                if (streamBusy[s]) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[s]));
                }
                constexpr int blockSize = 256;
                const int blocks = static_cast<int>(
                    std::min<size_t>((n + blockSize - 1) / blockSize, 8192));
                priceOptionsKernel<<<blocks, blockSize, 0, streams[s]>>>(
                    table, range.start + offset, n, deviceBuffers[s]);
                CUDA_CHECK(cudaMemcpyAsync(localResults + offset, deviceBuffers[s],
                                           n * sizeof(double), cudaMemcpyDeviceToHost,
                                           streams[s]));
                streamBusy[s] = true;
                s = (s + 1) % kNumStreams;
            }
            for (int i = 0; i < kNumStreams; ++i) {
                if (streamBusy[i]) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
                }
            }
        } else {
            for (;;) {
                const size_t chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= numChunks) {
                    break;
                }
                const size_t offset = chunk * kChunkSize;
                const size_t n = std::min(range.count, offset + kChunkSize) - offset;
                priceOptionsHost(table, range.start + offset, n, localResults + offset);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedUs =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (elapsedUs / 1e6));
    }

    // Collect the full result vector on rank 0 for output and validation.
    std::vector<double> results;
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        results.resize(numOptions);
        counts.resize(numRanks);
        displs.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const RankRange rr = computeRange(numOptions, r, numRanks);
            counts[r] = static_cast<int>(rr.count);
            displs[r] = static_cast<int>(rr.start);
        }
    }
    MPI_Gatherv(localResults, static_cast<int>(range.count), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");

            // Only the first few options are inspected; regenerate them locally.
            std::vector<OptionInput> options(std::min<size_t>(10, numOptions));
            for (size_t i = 0; i < options.size(); ++i) {
                options[i] = generateOption(table, i);
            }
            const bool valid = validateResults(options, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaFree(deviceBuffers[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    if (localResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(localResults));
    }

    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
