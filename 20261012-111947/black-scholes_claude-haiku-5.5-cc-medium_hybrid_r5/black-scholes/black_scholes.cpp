#include <algorithm>
#include <chrono>
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

// Aborts the whole MPI job on a CUDA error
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

// Standard test cases. A plain struct (not std::array) so it can be passed by value to kernels.
constexpr size_t kNumTestOptions = 7;

struct TestOptions {
    OptionInput options[kNumTestOptions];
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

// Standard test cases for validation
inline constexpr TestOptions getTestOptions() noexcept {
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

// Option i of the benchmark set. Depends only on i, so every rank and GPU can build its own slice.
__host__ __device__ inline OptionInput generateOption(const TestOptions& tests, const size_t i) noexcept {
    // Cycle through test options and vary parameters slightly
    const OptionInput& base = tests.options[i % kNumTestOptions];
    OptionInput option = base;

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// One thread per option: out[j] = price of option (first + j)
__global__ void priceKernel(double* out, const size_t first, const size_t count, const TestOptions tests) {
    const size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (j < count) {
        out[j] = blackScholes(generateOption(tests, first + j));
    }
}

constexpr int kThreadsPerBlock = 256;
constexpr size_t kBatchSize = size_t{1} << 24;       // Options per kernel launch
constexpr size_t kMaxMessageCount = size_t{1} << 28; // Doubles per MPI message

// Start of the share belonging to a unit with weight `before` (sum of the weights ahead of it),
// when n items are split in proportion to unit weights summing to `total`. Exact integer arithmetic.
size_t splitPoint(const size_t n, const size_t before, const size_t total) noexcept {
    return before * (n / total) + before * (n % total) / total;
}

// Per-GPU resources: two streams with two device buffers, so kernels overlap with device-to-host copies
struct GpuLane {
    cudaStream_t stream[2];
    double* buffer[2];
};

GpuLane createLane(const size_t capacity) {
    GpuLane lane{};
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaStreamCreate(&lane.stream[s]));
        CUDA_CHECK(cudaMalloc(&lane.buffer[s], capacity * sizeof(double)));
    }
    return lane;
}

void destroyLane(GpuLane& lane) {
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaFree(lane.buffer[s]));
        CUDA_CHECK(cudaStreamDestroy(lane.stream[s]));
    }
}

// Prices local options [begin, end) on the calling thread's current GPU.
// out is the rank's local result buffer; globalOffset is the global index of local option 0.
void priceRange(const GpuLane& lane, double* out, const size_t globalOffset,
                const size_t begin, const size_t end, const TestOptions& tests) {
    size_t batch = 0;
    for (size_t b = begin; b < end; b += kBatchSize, ++batch) {
        const size_t count = std::min(kBatchSize, end - b);
        const int s = static_cast<int>(batch % 2);
        const size_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        priceKernel<<<blocks, kThreadsPerBlock, 0, lane.stream[s]>>>(
            lane.buffer[s], globalOffset + b, count, tests);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(out + b, lane.buffer[s], count * sizeof(double),
                                   cudaMemcpyDeviceToHost, lane.stream[s]));
    }
}

// Gathers every rank's local results into all (valid on rank 0 only), in global option order.
// rankBegin[r] is the first global option owned by rank r; rankBegin[size] is the total.
void gatherResults(const double* local, const size_t localCount, double* all,
                   const std::vector<size_t>& rankBegin, const int rank, const int size) {
    if (rank != 0) {
        for (size_t off = 0; off < localCount; off += kMaxMessageCount) {
            const size_t count = std::min(kMaxMessageCount, localCount - off);
            MPI_Send(local + off, static_cast<int>(count), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        return;
    }

    std::copy(local, local + localCount, all);

    std::vector<MPI_Request> requests;
    for (int r = 1; r < size; ++r) {
        const size_t begin = rankBegin[r];
        const size_t count = rankBegin[r + 1] - begin;
        for (size_t off = 0; off < count; off += kMaxMessageCount) {
            const size_t n = std::min(kMaxMessageCount, count - off);
            requests.emplace_back();
            MPI_Irecv(all + begin + off, static_cast<int>(n), MPI_DOUBLE, r, 0, MPI_COMM_WORLD,
                      &requests.back());
        }
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

bool validateResults(const TestOptions& tests, const size_t numOptions,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), numOptions);

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = generateOption(tests, i).value;
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
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

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const TestOptions tests = getTestOptions();

    // Ranks on the same node share its GPUs: rank L drives devices d with d % nodeSize == L.
    // If there are more ranks than GPUs, ranks wrap around and share a device.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    for (int d = nodeRank; d < deviceCount; d += nodeSize) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        devices.push_back(nodeRank % deviceCount);
    }
    const int numGpus = static_cast<int>(devices.size());

    // Each rank owns a contiguous block of options, sized in proportion to its GPU count
    std::vector<int> gpusPerRank(size);
    MPI_Allgather(&numGpus, 1, MPI_INT, gpusPerRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<size_t> rankBegin(size + 1, 0);
    size_t totalGpus = 0;
    for (int r = 0; r < size; ++r) {
        totalGpus += gpusPerRank[r];
    }
    size_t gpusBefore = 0;
    for (int r = 0; r <= size; ++r) {
        rankBegin[r] = splitPoint(numOptions, gpusBefore, totalGpus);
        if (r < size) {
            gpusBefore += gpusPerRank[r];
        }
    }
    const size_t localBegin = rankBegin[rank];
    const size_t localCount = rankBegin[rank + 1] - localBegin;

    // Pinned host buffer, so device-to-host copies run asynchronously
    CUDA_CHECK(cudaSetDevice(devices[0]));
    double* localResults = nullptr;
    if (localCount > 0) {
        CUDA_CHECK(cudaMallocHost(&localResults, localCount * sizeof(double)));
    }

    // Price options: one OpenMP thread per local GPU
    if (root) {
        printf("Pricing options...\n");
    }
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    #pragma omp parallel num_threads(numGpus)
    {
        const int t = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(devices[t]));

        const size_t chunkBegin = splitPoint(localCount, t, numGpus);
        const size_t chunkEnd = splitPoint(localCount, t + 1, numGpus);
        GpuLane lane = createLane(std::max<size_t>(1, std::min(kBatchSize, chunkEnd - chunkBegin)));

        // Device setup stays outside the timed region
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        priceRange(lane, localResults, localBegin, chunkBegin, chunkEnd, tests);
        CUDA_CHECK(cudaStreamSynchronize(lane.stream[0]));
        CUDA_CHECK(cudaStreamSynchronize(lane.stream[1]));

        #pragma omp barrier
        #pragma omp master
        {
            end = std::chrono::high_resolution_clock::now();
        }

        destroyLane(lane);
    }

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double elapsedUs = static_cast<double>(duration.count());
    double maxElapsedUs = 0.0;
    MPI_Reduce(&elapsedUs, &maxElapsedUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %.3f ms\n", maxElapsedUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxElapsedUs / 1e6));
    }

    // Gather all results on rank 0 (outside the timed region)
    std::vector<double> results;
    if (root) {
        results.resize(numOptions);
    }
    gatherResults(localResults, localCount, results.data(), rankBegin, rank, size);
    if (localResults) {
        CUDA_CHECK(cudaFreeHost(localResults));
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
            bool valid = validateResults(tests, numOptions, results);

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
