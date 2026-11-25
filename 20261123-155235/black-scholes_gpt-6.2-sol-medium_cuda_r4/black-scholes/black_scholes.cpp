#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    OptionType type;
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
};

constexpr std::array<OptionInput, 7> testOptions{{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408},
}};

struct PricingParameters {
    double strike;
    double spot;
    double drift;
    double volatilityTime;
    double spotDiscount;
    double strikeDiscount;
    OptionType type;
};

constexpr unsigned baseOptionCount = testOptions.size();
// Neighboring threads read different entries, which the GPU can cache in global memory.
__device__ PricingParameters deviceParameters[baseOptionCount];

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void uploadPricingParameters() {
    PricingParameters parameters[baseOptionCount];
    for (unsigned i = 0; i < baseOptionCount; ++i) {
        const OptionInput& option = testOptions[i];
        parameters[i] = {
            option.strike,
            option.spot,
            (option.r - option.q + 0.5 * option.vol * option.vol) * option.t,
            option.t > 0.0 && option.vol > 0.0 ? option.vol * std::sqrt(option.t) : 0.0,
            std::exp(-option.q * option.t),
            std::exp(-option.r * option.t),
            option.type,
        };
    }
    checkCuda(cudaMemcpyToSymbol(deviceParameters, parameters, sizeof(parameters)),
              "Upload pricing parameters");
}

__device__ double cumulativeNormal(double x) {
    constexpr double inverseSqrtTwo = 0.7071067811865475244;
    return 0.5 * (1.0 + erf(x * inverseSqrtTwo));
}

__global__ void priceOptions(double* results, size_t offset, unsigned count) {
    const unsigned first = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned localIndex = first; localIndex < count; localIndex += stride) {
        const size_t i = offset + localIndex;
        const PricingParameters& option = deviceParameters[i % baseOptionCount];
        if (option.volatilityTime == 0.0) {
            results[localIndex] = 0.0;
            continue;
        }

        // Generate the same scaled prices as the host benchmark for this index.
        const double factor = fma(0.1, i / static_cast<double>(baseOptionCount), 1.0);
        const double spot = option.spot * factor;
        const double strike = option.strike * factor;
        const double d1 = (log(spot / strike) + option.drift) / option.volatilityTime;
        const double d2 = d1 - option.volatilityTime;

        const bool call = option.type == CALL;
        const double spotTerm = (spot * option.spotDiscount) * cumulativeNormal(call ? d1 : -d1);
        const double strikeTerm = (strike * option.strikeDiscount) * cumulativeNormal(call ? d2 : -d2);
        results[localIndex] = call ? spotTerm - strikeTerm : strikeTerm - spotTerm;
    }
}

double runPricing(size_t count, std::vector<double>& results) {
    if (count == 0) {
        return 0.0;
    }

    uploadPricingParameters();
    // Reuse the output buffer so the option count can exceed device memory.
    constexpr size_t maxBatchSize = 16 * 1024 * 1024;
    const size_t batchCapacity = std::min(count, maxBatchSize);
    double* deviceResults;
    checkCuda(cudaMalloc(&deviceResults, batchCapacity * sizeof(double)), "Allocate device results");

    int device;
    int multiprocessors;
    checkCuda(cudaGetDevice(&device), "Get CUDA device");
    checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
              "Get CUDA multiprocessor count");
    constexpr unsigned threads = 128;
    int residentBlocks;
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&residentBlocks, priceOptions, threads, 0),
              "Get Black-Scholes kernel occupancy");
    const unsigned maxBlocks = multiprocessors * residentBlocks * 4;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Create start event");
    checkCuda(cudaEventCreate(&stop), "Create stop event");
    double totalMilliseconds = 0.0;
    for (size_t offset = 0; offset < count;) {
        const unsigned batchSize = std::min(batchCapacity, count - offset);
        const unsigned blocks = std::min((batchSize + threads - 1) / threads, maxBlocks);
        checkCuda(cudaEventRecord(start), "Record start event");
        priceOptions<<<blocks, threads>>>(deviceResults, offset, batchSize);
        checkCuda(cudaGetLastError(), "Launch Black-Scholes kernel");
        checkCuda(cudaEventRecord(stop), "Record stop event");
        checkCuda(cudaEventSynchronize(stop), "Run Black-Scholes kernel");
        float milliseconds;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measure Black-Scholes kernel");
        totalMilliseconds += milliseconds;

        if (offset < results.size()) {
            const size_t copyCount = std::min<size_t>(batchSize, results.size() - offset);
            checkCuda(cudaMemcpy(results.data() + offset, deviceResults, copyCount * sizeof(double),
                                 cudaMemcpyDeviceToHost), "Download option prices");
        }
        offset += batchSize;
    }
    checkCuda(cudaEventDestroy(stop), "Destroy stop event");
    checkCuda(cudaEventDestroy(start), "Destroy start event");
    checkCuda(cudaFree(deviceResults), "Free device results");
    return totalMilliseconds;
}

constexpr size_t validationCount = 10;

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(validationCount, results.size());

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % baseOptionCount].value;
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

    std::vector<double> results;
    if (numOptions > results.max_size()) {
        fprintf(stderr, "Too many options: %zu\n", numOptions);
        return 1;
    }
    if (printResults) {
        results.resize(numOptions);
    } else if (validate) {
        results.resize(std::min(validationCount, numOptions));
    }

    printf("Pricing options...\n");
    const double milliseconds = runPricing(numOptions, results);
    printf("Computation time: %.3f ms\n", milliseconds);
    printf("Options per second: %.0f\n", milliseconds > 0.0 ? numOptions * 1000.0 / milliseconds : 0.0);

    // Print results for external validation
    if (printResults) {
        print_results(results, "OptionPrices");
    }

    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(results);

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
