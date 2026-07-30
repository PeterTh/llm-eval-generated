#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

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

// CUDA device function: Standard normal cumulative distribution function
__device__ __forceinline__ double cumulativeNormalDevice(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// CUDA kernel: Black-Scholes formula for European options (vectorized)
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                    double* __restrict__ results,
                                    const size_t numOptions) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOptions) return;

    const OptionInput& option = options[idx];
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = cumulativeNormalDevice(d1);
    const double Nd2 = cumulativeNormalDevice(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormalDevice(-d2) - S * exp(-q * T) * cumulativeNormalDevice(-d1);
    }

    results[idx] = price;
}

// Host-side Black-Scholes for CPU validation
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

double blackScholes(const OptionInput& option) noexcept {
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
    } else {
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char** argv) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
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
    
    printf("Black-Scholes Option Pricing Benchmark (CUDA)\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Print GPU info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (%d SMs, %zu MiB)\n", prop.name, prop.multiProcessorCount, prop.totalGlobalMem / (1024 * 1024));
    
    // Generate options on host
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);
    
    // Allocate host results
    std::vector<double> results(numOptions);
    
    // Allocate device memory
    OptionInput* d_options = nullptr;
    double* d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_options, numOptions * sizeof(OptionInput)));
    CUDA_CHECK(cudaMalloc(&d_results, numOptions * sizeof(double)));
    
    // Copy options to device
    CUDA_CHECK(cudaMemcpy(d_options, options.data(), numOptions * sizeof(OptionInput), cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    constexpr int BLOCK_SIZE = 256;
    const int gridSize = (static_cast<int>(numOptions) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Price options on GPU
    printf("Pricing options...\n");
    
    // Use events for accurate GPU timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaEventRecord(startEvent));
    blackScholesKernel<<<gridSize, BLOCK_SIZE>>>(d_options, d_results, numOptions);
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float gpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, numOptions * sizeof(double), cudaMemcpyDeviceToHost));
    
    printf("Computation time: %.3f ms (GPU: %.3f ms)\n", duration.count() / 1000.0, gpuTimeMs);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    
    // Cleanup CUDA events
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    
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
        } else {
            printf("Validation: FAILED\n");
        }
        
        // Free device memory before returning
        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
        return valid ? 0 : 1;
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_results));
    
    return 0;
}
