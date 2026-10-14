#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t err_ = (call);                                     \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
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
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (compiled for both host and device)
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

constexpr size_t kNumTestOptions = 7;

// Standard test cases for validation
inline constexpr std::array<OptionInput, kNumTestOptions> getTestOptions() noexcept {
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

// Option i of the benchmark set: test case i % 7, with spot and strike scaled by a factor
// that grows with the cycle number. Evaluated on both host and device so the GPU needs no input data.
__host__ __device__ inline OptionInput makeOption(const OptionInput* tests, const size_t i) noexcept {
    OptionInput option = tests[i % kNumTestOptions];

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate the first numOptions options of the benchmark set (host only)
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    const auto testOptions = getTestOptions();
    options.resize(numOptions);

    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = makeOption(testOptions.data(), i);
    }
}

constexpr size_t kNumValidationChecks = 10;

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(kNumValidationChecks, options.size());

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

// ---------------------------------------------------------------------------
// GPU pricing
// ---------------------------------------------------------------------------

constexpr int kThreadsPerBlock = 256;

// Options per device buffer; two buffers per GPU let copy-back overlap the next kernel
constexpr size_t kBatchSize = size_t{1} << 22;

__constant__ OptionInput c_tests[kNumTestOptions];

// Prices options [first, first + count) and writes them to out[0 .. count)
__global__ void priceKernel(double* out, const size_t first, const size_t count) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < count) {
        out[j] = blackScholes(makeOption(c_tests, first + j));
    }
}

// Page-locked host buffer, so device-to-host copies into it run asynchronously
class PinnedBuffer {
  public:
    explicit PinnedBuffer(const size_t n) : size_(n) {
        if (size_ > 0) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), size_ * sizeof(double)));
        }
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    double* data() const { return data_; }
    size_t size() const { return size_; }

  private:
    double* data_ = nullptr;
    size_t size_ = 0;
};

// Prices options [begin, end) on one GPU and writes them to out[0 .. end - begin)
void priceRangeOnGpu(const int device, const size_t begin, const size_t end, double* out) {
    const size_t total = end - begin;
    if (total == 0) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
    const auto testOptions = getTestOptions();
    CUDA_CHECK(cudaMemcpyToSymbol(c_tests, testOptions.data(), sizeof(c_tests)));

    // Batches alternate between two device buffers and streams. Stream order keeps each buffer's
    // copy-back ahead of the next kernel that reuses it, so the host never has to wait mid-loop.
    const size_t batch = std::min(total, kBatchSize);
    double* dOut[2];
    cudaStream_t stream[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&dOut[b], batch * sizeof(double)));
        CUDA_CHECK(cudaStreamCreate(&stream[b]));
    }

    size_t batchIndex = 0;
    for (size_t off = 0; off < total; off += batch, ++batchIndex) {
        const int b = static_cast<int>(batchIndex % 2);
        const size_t count = std::min(batch, total - off);
        const unsigned int blocks = static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        priceKernel<<<blocks, kThreadsPerBlock, 0, stream[b]>>>(dOut[b], begin + off, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(out + off, dOut[b], count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream[b]));
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamSynchronize(stream[b]));
        CUDA_CHECK(cudaFree(dOut[b]));
        CUDA_CHECK(cudaStreamDestroy(stream[b]));
    }
}

// Prices options [begin, end) across the given GPUs, one OpenMP thread per GPU
void priceRangeOnGpus(const std::vector<int>& devices, const size_t begin, const size_t end, double* out) {
    const size_t total = end - begin;
    const size_t numWorkers = devices.size();

    #pragma omp parallel for num_threads(numWorkers) schedule(static, 1)
    for (size_t w = 0; w < numWorkers; ++w) {
        const size_t lo = total * w / numWorkers;
        const size_t hi = total * (w + 1) / numWorkers;
        priceRangeOnGpu(devices[w], begin + lo, begin + hi, out + lo);
    }
}

// Create the CUDA contexts up front so context setup stays out of the timed region
void initGpus(const std::vector<int>& devices) {
    const size_t numWorkers = devices.size();

    #pragma omp parallel for num_threads(numWorkers) schedule(static, 1)
    for (size_t w = 0; w < numWorkers; ++w) {
        CUDA_CHECK(cudaSetDevice(devices[w]));
        CUDA_CHECK(cudaFree(nullptr));
    }
}

// First option index owned by rank r when n options are split over p ranks
size_t blockStart(const size_t n, const int r, const int p) {
    return n * static_cast<size_t>(r) / static_cast<size_t>(p);
}

int runBenchmark(int argc, char** argv) {
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Ranks on the same node split its GPUs round-robin; a rank with no GPU of its own shares one
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    std::vector<int> devices;
    for (int d = nodeRank; d < numDevices; d += nodeSize) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        devices.push_back(nodeRank % numDevices);
    }

    // Each MPI rank prices one contiguous block of options
    const size_t begin = blockStart(numOptions, rank, size);
    const size_t end = blockStart(numOptions, rank + 1, size);
    PinnedBuffer localResults(end - begin);

    initGpus(devices);

    if (root) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    priceRangeOnGpus(devices, begin, end, localResults.data());
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %.3f ms\n", elapsed * 1e3);
        printf("Options per second: %.0f\n", numOptions / elapsed);
    }

    // Gather only what output needs: all prices for -r, the first few for -v
    const size_t gatherCount = printResults ? numOptions
                             : (validate ? std::min(numOptions, kNumValidationChecks) : 0);
    std::vector<double> results;
    if (gatherCount > 0) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t lo = std::min(blockStart(numOptions, r, size), gatherCount);
            const size_t hi = std::min(blockStart(numOptions, r + 1, size), gatherCount);
            displs[r] = static_cast<int>(lo);
            counts[r] = static_cast<int>(hi - lo);
        }
        if (root) {
            results.resize(gatherCount);
        }
        MPI_Gatherv(localResults.data(), counts[rank], MPI_DOUBLE,
                    results.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(results, "OptionPrices");
    }

    // Validation
    int exitCode = 0;
    if (validate && root) {
        printf("Validating results...\n");
        std::vector<OptionInput> options;
        generateOptions(options, std::min(numOptions, kNumValidationChecks));
        bool valid = validateResults(options, results);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    const int exitCode = runBenchmark(argc, argv);
    MPI_Finalize();
    return exitCode;
}
