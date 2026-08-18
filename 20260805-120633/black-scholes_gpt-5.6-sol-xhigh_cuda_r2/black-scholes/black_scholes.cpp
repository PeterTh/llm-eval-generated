#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

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

constexpr size_t kDoubleFields = 6;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) noexcept {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_result = (expression);                      \
        if (cuda_check_result != cudaSuccess) {                                  \
            cudaFailure(cuda_check_result, #expression, __FILE__, __LINE__);     \
        }                                                                        \
    } while (false)

class CudaBuffer {
public:
    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    ~CudaBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(const size_t bytes) {
        if (bytes != 0) {
            CUDA_CHECK(cudaMalloc(&data_, bytes));
        }
    }

    [[nodiscard]] void* get() const noexcept { return data_; }

private:
    void* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// Structure-of-arrays storage gives every warp coalesced loads for every field.
// The six double arrays share one allocation so they can also be transferred in
// one operation.
struct HostOptions {
    explicit HostOptions(const size_t size)
        : count(size), doubles(checkedDoubleCount(size)), types(size) {}

    static size_t checkedDoubleCount(const size_t count) {
        if (count > std::numeric_limits<size_t>::max() / kDoubleFields) {
            std::fprintf(stderr, "Requested option count is too large\n");
            std::exit(EXIT_FAILURE);
        }
        return count * kDoubleFields;
    }

    [[nodiscard]] double* strike() noexcept { return doubles.data(); }
    [[nodiscard]] double* spot() noexcept { return doubles.data() + count; }
    [[nodiscard]] double* q() noexcept { return doubles.data() + 2 * count; }
    [[nodiscard]] double* r() noexcept { return doubles.data() + 3 * count; }
    [[nodiscard]] double* t() noexcept { return doubles.data() + 4 * count; }
    [[nodiscard]] double* vol() noexcept { return doubles.data() + 5 * count; }

    size_t count;
    std::vector<double> doubles;
    std::vector<int> types;
};

// Standard test cases for validation.
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

// Generate the same contracts as the original implementation, directly into
// the layout consumed by the CUDA kernel.
void generateOptions(HostOptions& options) {
    constexpr auto testOptions = getTestOptions();

    if (options.count == 0) {
        return;
    }

    double* const strike = options.strike();
    double* const spot = options.spot();
    double* const q = options.q();
    double* const r = options.r();
    double* const t = options.t();
    double* const vol = options.vol();

    for (size_t i = 0; i < options.count; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        const double factor =
            1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));

        options.types[i] = base.type;
        strike[i] = base.strike * factor;
        spot[i] = base.spot * factor;
        q[i] = base.q;
        r[i] = base.r;
        t[i] = base.t;
        vol[i] = base.vol;
    }
}

__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(
    const int* __restrict__ types,
    const double* __restrict__ strike,
    const double* __restrict__ spot,
    const double* __restrict__ q,
    const double* __restrict__ r,
    const double* __restrict__ t,
    const double* __restrict__ vol,
    double* __restrict__ results,
    const size_t count) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < count; i += stride) {
        const double time = t[i];
        const double sigma = vol[i];

        if (time <= 0.0 || sigma <= 0.0) {
            results[i] = 0.0;
            continue;
        }

        const double stockPrice = spot[i];
        const double strikePrice = strike[i];
        const double rate = r[i];
        const double dividendYield = q[i];
        const double sqrtTime = sqrt(time);
        const double sigmaSqrtTime = sigma * sqrtTime;
        const double d1 =
            (log(stockPrice / strikePrice) +
             (rate - dividendYield + 0.5 * sigma * sigma) * time) /
            sigmaSqrtTime;
        const double d2 = d1 - sigmaSqrtTime;
        const double discount = exp(-rate * time);
        const double discountedStock = stockPrice * exp(-dividendYield * time);

        if (types[i] == CALL) {
            results[i] = discountedStock * cumulativeNormalDevice(d1) -
                         strikePrice * discount * cumulativeNormalDevice(d2);
        } else {
            results[i] = strikePrice * discount * cumulativeNormalDevice(-d2) -
                         discountedStock * cumulativeNormalDevice(-d1);
        }
    }
}

[[nodiscard]] size_t checkedBytes(const size_t count, const size_t elementSize) {
    if (count > std::numeric_limits<size_t>::max() / elementSize) {
        std::fprintf(stderr, "Requested option count is too large\n");
        std::exit(EXIT_FAILURE);
    }
    return count * elementSize;
}

// Transfers are deliberately outside the measured interval. The reported time
// is the GPU pricing computation, matching the original benchmark's compute-only
// timing boundary.
double priceOptionsCuda(const HostOptions& options, double* const hostResults) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    if (options.count == 0) {
        return 0.0;
    }

    const size_t doubleBytes = checkedBytes(options.doubles.size(), sizeof(double));
    const size_t typeBytes = checkedBytes(options.types.size(), sizeof(int));
    const size_t resultBytes = checkedBytes(options.count, sizeof(double));

    CudaBuffer deviceDoubles;
    CudaBuffer deviceTypes;
    CudaBuffer deviceResults;
    deviceDoubles.allocate(doubleBytes);
    deviceTypes.allocate(typeBytes);
    deviceResults.allocate(resultBytes);

    CUDA_CHECK(cudaMemcpy(deviceDoubles.get(), options.doubles.data(), doubleBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceTypes.get(), options.types.data(), typeBytes,
                          cudaMemcpyHostToDevice));

    int minimumGridSize = 0;
    int threadsPerBlock = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minimumGridSize, &threadsPerBlock, blackScholesKernel, 0, 0));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const size_t requiredBlocks =
        options.count / static_cast<size_t>(threadsPerBlock) +
        (options.count % static_cast<size_t>(threadsPerBlock) != 0);
    // A bounded grid avoids excessive block-scheduling overhead for very large
    // inputs; the grid-stride loop retains arbitrary-size scalability.
    const size_t residentWaves = std::max<size_t>(
        static_cast<size_t>(minimumGridSize),
        static_cast<size_t>(properties.multiProcessorCount) * 32);
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(requiredBlocks, residentWaves));

    const auto* const deviceInput = static_cast<const double*>(deviceDoubles.get());
    const auto* const deviceType = static_cast<const int*>(deviceTypes.get());
    auto* const deviceOutput = static_cast<double*>(deviceResults.get());

    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start.get()));
    blackScholesKernel<<<blocks, threadsPerBlock>>>(
        deviceType,
        deviceInput,
        deviceInput + options.count,
        deviceInput + 2 * options.count,
        deviceInput + 3 * options.count,
        deviceInput + 4 * options.count,
        deviceInput + 5 * options.count,
        deviceOutput,
        options.count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start.get(), stop.get()));

    if (hostResults != nullptr) {
        CUDA_CHECK(cudaMemcpy(hostResults, deviceOutput, resultBytes,
                              cudaMemcpyDeviceToHost));
    }

    return static_cast<double>(elapsedMilliseconds);
}

bool validateResults(const size_t numOptions, const std::vector<double>& results) {
    constexpr auto testOptions = getTestOptions();
    bool allPassed = true;
    const size_t numChecks = std::min<size_t>(10, numOptions);

    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % testOptions.size()].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);

        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relError);

        // Preserve the original validation behavior.
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) ||
            std::isinf(computed)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n",
                        i, computed);
            allPassed = false;
        }
    }

    return allPassed;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    std::printf("Black-Scholes Option Pricing Benchmark\n");
    std::printf("Number of options: %zu\n", numOptions);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    HostOptions options(numOptions);
    generateOptions(options);

    const bool copyResultsToHost = printResults || validate;
    std::vector<double> results;
    if (copyResultsToHost) {
        results.resize(numOptions);
    }

    std::printf("Pricing options...\n");
    const double elapsedMilliseconds = priceOptionsCuda(
        options, copyResultsToHost ? results.data() : nullptr);

    const double optionsPerSecond = elapsedMilliseconds > 0.0
        ? static_cast<double>(numOptions) * 1000.0 / elapsedMilliseconds
        : 0.0;
    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    std::printf("Options per second: %.0f\n", optionsPerSecond);

    if (printResults) {
        print_results(results, "OptionPrices");
    }

    if (validate) {
        std::printf("Validating results...\n");
        const bool valid = validateResults(numOptions, results);

        if (valid) {
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
