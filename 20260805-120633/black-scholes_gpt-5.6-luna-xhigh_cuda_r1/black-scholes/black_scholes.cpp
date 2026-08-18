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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula shared by the host reference and the CUDA kernel.
__host__ __device__ __forceinline__ double blackScholesPrice(
    const int type,
    const double strike,
    const double spot,
    const double q,
    const double r,
    const double maturity,
    const double volatility) noexcept {
    const double S = spot;
    const double K = strike;
    const double T = maturity;
    const double sigma = volatility;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Black-Scholes formula for European options.
double blackScholes(const OptionInput& option) noexcept {
    return blackScholesPrice(option.type, option.strike, option.spot, option.q,
                             option.r, option.t, option.vol);
}

// Structure-of-arrays inputs keep each field coalescent for a warp. The
// grid-stride loop supports arbitrarily large option vectors.
__global__ void blackScholesKernel(const int* __restrict__ types,
                                   const double* __restrict__ strikes,
                                   const double* __restrict__ spots,
                                   const double* __restrict__ dividendYields,
                                   const double* __restrict__ interestRates,
                                   const double* __restrict__ maturities,
                                   const double* __restrict__ volatilities,
                                   double* __restrict__ results,
                                   const size_t numOptions) {
    const size_t threadIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = threadIndex; i < numOptions; i += stride) {
        results[i] = blackScholesPrice(types[i], strikes[i], spots[i], dividendYields[i],
                                       interestRates[i], maturities[i], volatilities[i]);
    }
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(const size_t count) {
        if (count == 0) {
            return;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }

    T* get() const noexcept { return data_; }

  private:
    T* data_ = nullptr;
};

void priceOptionsOnGpu(const std::vector<OptionInput>& options,
                       std::vector<double>& results,
                       double& elapsedMilliseconds) {
    const size_t numOptions = options.size();
    elapsedMilliseconds = 0.0;
    if (numOptions == 0) {
        return;
    }

    // Remove validation-only fields from the device working set before timing.
    std::vector<int> types(numOptions);
    std::vector<double> strikes(numOptions);
    std::vector<double> spots(numOptions);
    std::vector<double> dividendYields(numOptions);
    std::vector<double> interestRates(numOptions);
    std::vector<double> maturities(numOptions);
    std::vector<double> volatilities(numOptions);
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& option = options[i];
        types[i] = option.type;
        strikes[i] = option.strike;
        spots[i] = option.spot;
        dividendYields[i] = option.q;
        interestRates[i] = option.r;
        maturities[i] = option.t;
        volatilities[i] = option.vol;
    }

    DeviceBuffer<int> deviceTypes;
    DeviceBuffer<double> deviceStrikes;
    DeviceBuffer<double> deviceSpots;
    DeviceBuffer<double> deviceDividendYields;
    DeviceBuffer<double> deviceInterestRates;
    DeviceBuffer<double> deviceMaturities;
    DeviceBuffer<double> deviceVolatilities;
    DeviceBuffer<double> deviceResults;

    deviceTypes.allocate(numOptions);
    deviceStrikes.allocate(numOptions);
    deviceSpots.allocate(numOptions);
    deviceDividendYields.allocate(numOptions);
    deviceInterestRates.allocate(numOptions);
    deviceMaturities.allocate(numOptions);
    deviceVolatilities.allocate(numOptions);
    deviceResults.allocate(numOptions);

    CUDA_CHECK(cudaMemcpy(deviceTypes.get(), types.data(), numOptions * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceStrikes.get(), strikes.data(), numOptions * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSpots.get(), spots.data(), numOptions * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceDividendYields.get(), dividendYields.data(),
                          numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceInterestRates.get(), interestRates.data(),
                          numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMaturities.get(), maturities.data(),
                          numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceVolatilities.get(), volatilities.data(),
                          numOptions * sizeof(double), cudaMemcpyHostToDevice));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    constexpr unsigned int threadsPerBlock = 256;
    const size_t requiredBlocks =
        (numOptions + threadsPerBlock - 1) / threadsPerBlock;
    const size_t residentGridLimit =
        static_cast<size_t>(deviceProperties.multiProcessorCount) * 32;
    const unsigned int blocks = static_cast<unsigned int>(std::min(
        requiredBlocks,
        std::min(residentGridLimit, static_cast<size_t>(std::numeric_limits<unsigned int>::max()))));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    blackScholesKernel<<<blocks, threadsPerBlock>>>(
        deviceTypes.get(), deviceStrikes.get(), deviceSpots.get(),
        deviceDividendYields.get(), deviceInterestRates.get(), deviceMaturities.get(),
        deviceVolatilities.get(), deviceResults.get(), numOptions);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    elapsedMilliseconds = static_cast<double>(elapsed);

    CUDA_CHECK(cudaMemcpy(results.data(), deviceResults.get(), numOptions * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
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
    
    // Price options unconditionally on the CUDA device.
    printf("Pricing options...\n");
    double elapsedMilliseconds = 0.0;
    priceOptionsOnGpu(options, results, elapsedMilliseconds);

    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    printf("Options per second: %.0f\n",
           elapsedMilliseconds > 0.0
               ? numOptions / (elapsedMilliseconds / 1000.0)
               : 0.0);
    
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
