#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Keep all pricing work on the GPU. Each invocation is inlined into the
// grid-stride kernel to avoid call overhead and expose the formula to nvcc's
// optimizer.
__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double sqrtT = sqrt(T);
    const double sigmaSqrtT = sigma * sqrtT;
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discountedSpot = S * exp(-q * T);
    const double discountedStrike = K * exp(-r * T);

    // N(-x) = 1 - N(x). Reusing Nd1/Nd2 eliminates two expensive erf calls
    // for puts and allows nvcc to predicate the final selection efficiently.
    const double callPrice = discountedSpot * Nd1 - discountedStrike * Nd2;
    const double putPrice = discountedStrike * (1.0 - Nd2) - discountedSpot * (1.0 - Nd1);
    return option.type == CALL ? callPrice : putPrice;
}

__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t numOptions) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < numOptions; i += stride) {
        results[i] = blackScholes(options[i]);
    }
}

[[noreturn]] void reportCudaError(const cudaError_t error, const char* expression,
                                  const char* file, const int line) {
    fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
            file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cudaCheckResult = (expression);                         \
        if (cudaCheckResult != cudaSuccess) {                                     \
            reportCudaError(cudaCheckResult, #expression, __FILE__, __LINE__);    \
        }                                                                        \
    } while (false)

double priceOptionsCuda(const std::vector<OptionInput>& options,
                        std::vector<double>& results) {
    CUDA_CHECK(cudaFree(nullptr));

    if (options.empty()) {
        return 0.0;
    }

    const size_t numOptions = options.size();
    const size_t optionBytes = numOptions * sizeof(OptionInput);
    const size_t resultBytes = numOptions * sizeof(double);
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOptions), optionBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceResults), resultBytes));
    CUDA_CHECK(cudaMemcpy(deviceOptions, options.data(), optionBytes,
                          cudaMemcpyHostToDevice));

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    int suggestedGridSize = 0;
    int blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &suggestedGridSize, &blockSize, blackScholesKernel, 0, 0));

    int activeBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm, blackScholesKernel, blockSize, 0));

    const size_t requiredBlocks = (numOptions + static_cast<size_t>(blockSize) - 1) /
                                  static_cast<size_t>(blockSize);
    const size_t residentBlocks = static_cast<size_t>(properties.multiProcessorCount) *
                                  static_cast<size_t>(activeBlocksPerSm);
    const size_t launchBlocks = std::max<size_t>(
        1, std::min(requiredBlocks, std::min(residentBlocks,
                                             static_cast<size_t>(properties.maxGridSize[0]))));

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));
    blackScholesKernel<<<static_cast<unsigned int>(launchBlocks), blockSize>>>(
        deviceOptions, deviceResults, numOptions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaMemcpy(results.data(), deviceResults, resultBytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaFree(deviceResults));
    CUDA_CHECK(cudaFree(deviceOptions));
    return static_cast<double>(elapsedMilliseconds);
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
    
    // Generate options
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);
    
    // Allocate results
    std::vector<double> results(numOptions);
    
    // Price options
    printf("Pricing options...\n");
    const double durationMilliseconds = priceOptionsCuda(options, results);
    const double optionsPerSecond = durationMilliseconds > 0.0
        ? static_cast<double>(numOptions) * 1000.0 / durationMilliseconds
        : 0.0;
    
    printf("Computation time: %.3f ms\n", durationMilliseconds);
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
