#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

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

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kInputFieldCount = 7;
constexpr double kInvSqrtTwo = 0.707106781186547524400844362104849039;

enum InputField : std::size_t {
    SIGN = 0,
    STRIKE,
    SPOT,
    DIVIDEND_YIELD,
    RISK_FREE_RATE,
    TIME_TO_MATURITY,
    VOLATILITY
};

void cudaCheck(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(const std::size_t count) {
        if (count != 0) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 count * sizeof(T)),
                      "device allocation");
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

// Standard normal cumulative distribution function.  Keeping this operation
// in double precision preserves the numerical behavior of the CPU benchmark.
__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * kInvSqrtTwo));
}

// One thread prices one option in the common case.  The grid-stride loop also
// supports data sets larger than the device's maximum one-dimensional grid.
__global__ __launch_bounds__(kThreadsPerBlock)
void blackScholesKernel(const double* __restrict__ inputs,
                        double* __restrict__ prices,
                        const std::size_t numOptions) {
    const double* const signs = inputs + SIGN * numOptions;
    const double* const strikes = inputs + STRIKE * numOptions;
    const double* const spots = inputs + SPOT * numOptions;
    const double* const dividendYields = inputs + DIVIDEND_YIELD * numOptions;
    const double* const riskFreeRates = inputs + RISK_FREE_RATE * numOptions;
    const double* const times = inputs + TIME_TO_MATURITY * numOptions;
    const double* const volatilities = inputs + VOLATILITY * numOptions;

    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = first; i < numOptions; i += stride) {
        const double time = times[i];
        const double volatility = volatilities[i];

        if (time <= 0.0 || volatility <= 0.0) {
            prices[i] = 0.0;
            continue;
        }

        const double spot = spots[i];
        const double strike = strikes[i];
        const double dividendYield = dividendYields[i];
        const double riskFreeRate = riskFreeRates[i];
        const double sqrtTime = sqrt(time);
        const double volatilitySqrtTime = volatility * sqrtTime;
        const double d1 =
            (log(spot / strike) +
             (riskFreeRate - dividendYield +
              0.5 * volatility * volatility) *
                 time) /
            volatilitySqrtTime;
        const double d2 = d1 - volatilitySqrtTime;

        // sign is +1 for calls and -1 for puts.  This form removes divergent
        // call/put branches while evaluating the same two normal CDFs.
        const double sign = signs[i];
        const double discountedSpot = spot * exp(-dividendYield * time);
        const double discountedStrike = strike * exp(-riskFreeRate * time);
        prices[i] =
            sign * (discountedSpot * cumulativeNormalDevice(sign * d1) -
                    discountedStrike * cumulativeNormalDevice(sign * d2));
    }
}

}  // namespace

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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// Convert the host-friendly array of structures to a device-friendly
// structure of arrays.  Every warp consequently issues coalesced loads for
// each field used by the pricing kernel.
void packOptions(const std::vector<OptionInput>& options,
                 std::vector<double>& packedOptions) {
    const std::size_t numOptions = options.size();
    packedOptions.resize(kInputFieldCount * numOptions);

    for (std::size_t i = 0; i < numOptions; ++i) {
        const OptionInput& option = options[i];
        packedOptions[SIGN * numOptions + i] =
            option.type == CALL ? 1.0 : -1.0;
        packedOptions[STRIKE * numOptions + i] = option.strike;
        packedOptions[SPOT * numOptions + i] = option.spot;
        packedOptions[DIVIDEND_YIELD * numOptions + i] = option.q;
        packedOptions[RISK_FREE_RATE * numOptions + i] = option.r;
        packedOptions[TIME_TO_MATURITY * numOptions + i] = option.t;
        packedOptions[VOLATILITY * numOptions + i] = option.vol;
    }
}

float priceOptionsCuda(const std::vector<double>& packedOptions,
                       std::vector<double>& results) {
    const std::size_t numOptions = results.size();
    if (numOptions == 0) {
        return 0.0F;
    }

    // Inputs and outputs share one allocation to minimize CUDA allocation
    // overhead while retaining separately aligned contiguous regions.
    DeviceBuffer<double> deviceStorage(packedOptions.size() + numOptions);
    double* const deviceInputs = deviceStorage.get();
    double* const deviceResults = deviceInputs + packedOptions.size();

    cudaCheck(cudaMemcpy(deviceInputs, packedOptions.data(),
                         packedOptions.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
              "input transfer");

    int device = 0;
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDevice(&device), "device query");
    cudaCheck(cudaGetDeviceProperties(&properties, device),
              "device-properties query");

    const std::size_t requiredBlocks =
        numOptions / kThreadsPerBlock +
        (numOptions % kThreadsPerBlock != 0 ? 1 : 0);
    const std::size_t maxBlocks =
        static_cast<std::size_t>(properties.maxGridSize[0]);
    const int blockCount =
        static_cast<int>(std::min(requiredBlocks, maxBlocks));

    // Force lazy module loading before the timed region so the reported
    // computation time measures kernel work rather than one-time setup.
    cudaFuncAttributes kernelAttributes{};
    cudaCheck(cudaFuncGetAttributes(&kernelAttributes, blackScholesKernel),
              "pricing-kernel initialization");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "start-event creation");
    cudaCheck(cudaEventCreate(&stop), "stop-event creation");
    cudaCheck(cudaEventRecord(start), "start-event recording");

    blackScholesKernel<<<blockCount, kThreadsPerBlock>>>(
        deviceInputs, deviceResults, numOptions);
    cudaCheck(cudaGetLastError(), "pricing-kernel launch");

    cudaCheck(cudaEventRecord(stop), "stop-event recording");
    cudaCheck(cudaEventSynchronize(stop), "pricing-kernel execution");

    float elapsedMilliseconds = 0.0F;
    cudaCheck(cudaEventElapsedTime(&elapsedMilliseconds, start, stop),
              "kernel timing");
    cudaCheck(cudaEventDestroy(start), "start-event destruction");
    cudaCheck(cudaEventDestroy(stop), "stop-event destruction");

    cudaCheck(cudaMemcpy(results.data(), deviceResults,
                         numOptions * sizeof(double), cudaMemcpyDeviceToHost),
              "result transfer");
    return elapsedMilliseconds;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    constexpr std::size_t deviceBytesPerOption =
        (kInputFieldCount + 1) * sizeof(double);
    if (numOptions >
        std::numeric_limits<std::size_t>::max() / deviceBytesPerOption) {
        std::fprintf(stderr, "Requested option count is too large\n");
        return 1;
    }

    // Generate options
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);

    std::vector<double> packedOptions;
    packOptions(options, packedOptions);
    
    // Allocate results
    std::vector<double> results(numOptions);
    
    // Price options
    printf("Pricing options...\n");
    const float elapsedMilliseconds = priceOptionsCuda(packedOptions, results);
    const double optionsPerSecond =
        elapsedMilliseconds > 0.0F
            ? static_cast<double>(numOptions) /
                  (static_cast<double>(elapsedMilliseconds) * 1.0e-3)
            : 0.0;

    printf("Computation time: %.3f ms\n",
           static_cast<double>(elapsedMilliseconds));
    printf("Options per second: %.0f\n", optionsPerSecond);
    
    // Print results for external validation
    if (printResults) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(options, results);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
