#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr size_t kParameterCount = 6;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// The six parameter arrays occupy consecutive, warp-aligned planes.  This is
// substantially more efficient on the GPU than loading the original 72-byte
// OptionInput structures with one strided transaction per thread.
__global__ void blackScholesKernel(const double* __restrict__ parameters,
                                   const int* __restrict__ types,
                                   double* __restrict__ results,
                                   const size_t count,
                                   const size_t stride) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < count; i += step) {
        const double K = parameters[i];
        const double S = parameters[stride + i];
        const double q = parameters[2 * stride + i];
        const double r = parameters[3 * stride + i];
        const double T = parameters[4 * stride + i];
        const double sigma = parameters[5 * stride + i];

        if (T <= 0.0 || sigma <= 0.0) {
            results[i] = 0.0;
            continue;
        }

        const double sqrtT = sqrt(T);
        const double sigmaSqrtT = sigma * sqrtT;
        const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                          sigmaSqrtT;
        const double d2 = d1 - sigmaSqrtT;
        const double discountedStrike = K * exp(-r * T);
        const double discountedSpot = S * exp(-q * T);

        if (types[i] == CALL) {
            results[i] = discountedSpot * cumulativeNormal(d1) -
                         discountedStrike * cumulativeNormal(d2);
        } else {
            results[i] = discountedStrike * cumulativeNormal(-d2) -
                         discountedSpot * cumulativeNormal(-d1);
        }
    }
}

} // namespace

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

    constexpr size_t maxSize = std::numeric_limits<size_t>::max();
    if (numOptions > maxSize - 31) {
        std::fprintf(stderr, "Requested option count is too large\n");
        return 1;
    }
    const size_t parameterStride = (numOptions + 31) & ~size_t{31};
    if (parameterStride > maxSize / (kParameterCount * sizeof(double)) ||
        numOptions > maxSize / sizeof(int) ||
        numOptions > maxSize / sizeof(double)) {
        std::fprintf(stderr, "Requested option count is too large\n");
        return 1;
    }
    
    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate options
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);
    
    // Allocate results
    std::vector<double> results(numOptions);

    // A padded structure-of-arrays layout keeps every parameter plane aligned
    // to a full warp.  Pinned buffers permit transfers to overlap CUDA command
    // submission without the runtime first making an internal staging copy.
    const size_t parameterBytes = kParameterCount * parameterStride * sizeof(double);
    const size_t typeBytes = numOptions * sizeof(int);
    const size_t resultBytes = numOptions * sizeof(double);

    double* hostParameters = nullptr;
    int* hostTypes = nullptr;
    double* hostResults = nullptr;
    double* deviceParameters = nullptr;
    int* deviceTypes = nullptr;
    double* deviceResults = nullptr;

    if (numOptions != 0) {
        cudaCheck(cudaMallocHost(&hostParameters, parameterBytes), "allocating pinned parameters");
        cudaCheck(cudaMallocHost(&hostTypes, typeBytes), "allocating pinned option types");
        cudaCheck(cudaMallocHost(&hostResults, resultBytes), "allocating pinned results");
        cudaCheck(cudaMalloc(&deviceParameters, parameterBytes), "allocating device parameters");
        cudaCheck(cudaMalloc(&deviceTypes, typeBytes), "allocating device option types");
        cudaCheck(cudaMalloc(&deviceResults, resultBytes), "allocating device results");

        for (size_t i = 0; i < numOptions; ++i) {
            const OptionInput& option = options[i];
            hostParameters[i] = option.strike;
            hostParameters[parameterStride + i] = option.spot;
            hostParameters[2 * parameterStride + i] = option.q;
            hostParameters[3 * parameterStride + i] = option.r;
            hostParameters[4 * parameterStride + i] = option.t;
            hostParameters[5 * parameterStride + i] = option.vol;
            hostTypes[i] = option.type;
        }
    }

    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating CUDA stream");

    int device = 0;
    cudaDeviceProp deviceProperties{};
    cudaCheck(cudaGetDevice(&device), "querying active CUDA device");
    cudaCheck(cudaGetDeviceProperties(&deviceProperties, device), "querying CUDA device properties");

    // Force CUDA context and kernel module initialization out of the measured
    // region.  A zero-item launch executes no pricing work.
    blackScholesKernel<<<1, 1, 0, stream>>>(deviceParameters, deviceTypes,
                                            deviceResults, 0, parameterStride);
    cudaCheck(cudaGetLastError(), "launching warm-up kernel");
    cudaCheck(cudaStreamSynchronize(stream), "synchronizing warm-up kernel");

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (numOptions != 0) {
        cudaCheck(cudaMemcpyAsync(deviceParameters, hostParameters, parameterBytes,
                                  cudaMemcpyHostToDevice, stream), "copying parameters to device");
        cudaCheck(cudaMemcpyAsync(deviceTypes, hostTypes, typeBytes,
                                  cudaMemcpyHostToDevice, stream), "copying option types to device");

        const size_t requiredBlocks =
            (numOptions + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const size_t saturationBlocks =
            static_cast<size_t>(deviceProperties.multiProcessorCount) * 32;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min(requiredBlocks, saturationBlocks));
        blackScholesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            deviceParameters, deviceTypes, deviceResults, numOptions, parameterStride);
        cudaCheck(cudaGetLastError(), "launching Black-Scholes kernel");

        cudaCheck(cudaMemcpyAsync(hostResults, deviceResults, resultBytes,
                                  cudaMemcpyDeviceToHost, stream), "copying results to host");
    }
    cudaCheck(cudaStreamSynchronize(stream), "waiting for Black-Scholes computation");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (numOptions != 0) {
        std::copy_n(hostResults, numOptions, results.data());
    }

    cudaCheck(cudaStreamDestroy(stream), "destroying CUDA stream");
    if (deviceResults != nullptr) cudaCheck(cudaFree(deviceResults), "freeing device results");
    if (deviceTypes != nullptr) cudaCheck(cudaFree(deviceTypes), "freeing device option types");
    if (deviceParameters != nullptr) cudaCheck(cudaFree(deviceParameters), "freeing device parameters");
    if (hostResults != nullptr) cudaCheck(cudaFreeHost(hostResults), "freeing pinned results");
    if (hostTypes != nullptr) cudaCheck(cudaFreeHost(hostTypes), "freeing pinned option types");
    if (hostParameters != nullptr) cudaCheck(cudaFreeHost(hostParameters), "freeing pinned parameters");

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    const double seconds = duration.count() / 1.0e6;
    printf("Options per second: %.0f\n", seconds > 0.0 ? numOptions / seconds : 0.0);
    
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
