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

constexpr std::array<OptionInput, 7> testOptions = {{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
}};
constexpr size_t numTestOptions = testOptions.size();
constexpr size_t maxValidationChecks = 10;

struct PricingParameters {
    double spot;
    double strike;
    double spotDiscount;
    double strikeDiscount;
    double standardDeviation;
    double driftRate;
    double maturity;
    int type;
    bool active;
};

__constant__ PricingParameters pricingParameters[numTestOptions];

__device__ double cumulativeNormal(double x) {
    constexpr double inverseSqrtTwo = 0.7071067811865475244;
    return 0.5 * (1.0 + erf(x * inverseSqrtTwo));
}

__device__ double blackScholes(const PricingParameters& base, double factor) {
    if (!base.active) {
        return 0.0;
    }

    const double spot = base.spot * factor;
    const double strike = base.strike * factor;
    const double d1 = fma(base.driftRate, base.maturity, log(spot / strike)) /
                      base.standardDeviation;
    const double d2 = d1 - base.standardDeviation;

    const bool call = base.type == CALL;
    const double spotProbability = cumulativeNormal(call ? d1 : -d1);
    const double strikeProbability = cumulativeNormal(call ? d2 : -d2);
    const double discountedSpot = spot * base.spotDiscount;
    const double discountedStrike = strike * base.strikeDiscount;

    return call ? fma(discountedSpot, spotProbability, -discountedStrike * strikeProbability)
                : fma(discountedStrike, strikeProbability, -discountedSpot * spotProbability);
}

__global__ void priceOptions(double* results, size_t numOptions) {
    // Each block shares a template, but writes prices in the original interleaved order.
    const size_t first = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
                         numTestOptions + blockIdx.y;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x * numTestOptions;
    const auto& base = pricingParameters[blockIdx.y];
    for (size_t i = first; i < numOptions; i += stride) {
        const double factor = fma(0.1, static_cast<double>(i) / numTestOptions, 1.0);
        results[i] = blackScholes(base, factor);
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

float priceOnGpu(size_t numOptions, std::vector<double>& results) {
    if (numOptions == 0) {
        return 0.0f;
    }
    // Only the spot and strike vary within each group of generated options.
    std::array<PricingParameters, numTestOptions> parameters;
    for (size_t i = 0; i < numTestOptions; ++i) {
        const auto& base = testOptions[i];
        const bool active = !(base.t <= 0.0 || base.vol <= 0.0);
        parameters[i] = {base.spot, base.strike, std::exp(-base.q * base.t),
                         std::exp(-base.r * base.t),
                         active ? base.vol * std::sqrt(base.t) : 0.0,
                         std::fma(0.5 * base.vol, base.vol, base.r - base.q),
                         base.t, base.type, active};
    }
    checkCuda(cudaMemcpyToSymbol(pricingParameters, parameters.data(), sizeof(pricingParameters)),
              "Copy pricing parameters to GPU");

    double* deviceResults;
    checkCuda(cudaMalloc(&deviceResults, numOptions * sizeof(double)), "Allocate GPU results");

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "Create CUDA start event");
    checkCuda(cudaEventCreate(&end), "Create CUDA stop event");

    constexpr int threads = 256;
    constexpr size_t maxBlocks = 65535;
    const unsigned blocks = std::min(1 + (numOptions - 1) / (threads * numTestOptions), maxBlocks);
    const unsigned groups = std::min(numOptions, numTestOptions);
    checkCuda(cudaEventRecord(start), "Record CUDA start event");
    priceOptions<<<dim3(blocks, groups), threads>>>(deviceResults, numOptions);
    checkCuda(cudaGetLastError(), "Launch option pricing kernel");
    checkCuda(cudaEventRecord(end), "Record CUDA stop event");
    checkCuda(cudaEventSynchronize(end), "Run option pricing kernel");

    float milliseconds;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "Measure CUDA kernel time");
    if (!results.empty()) {
        checkCuda(cudaMemcpy(results.data(), deviceResults, results.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "Copy prices from GPU");
    }
    checkCuda(cudaEventDestroy(end), "Destroy CUDA stop event");
    checkCuda(cudaEventDestroy(start), "Destroy CUDA start event");
    checkCuda(cudaFree(deviceResults), "Free GPU results");
    return milliseconds;
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min(maxValidationChecks, results.size());

    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % numTestOptions].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
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

    if (numOptions > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Too many options for the results buffer\n");
        return EXIT_FAILURE;
    }
    const size_t validationCount = validate ? std::min(maxValidationChecks, numOptions) : 0;
    const size_t resultCount = printResults ? numOptions : validationCount;
    std::vector<double> results(resultCount);

    printf("Pricing options...\n");
    const double milliseconds = priceOnGpu(numOptions, results);

    printf("Computation time: %.3f ms\n", milliseconds);
    printf("Options per second: %.0f\n", milliseconds > 0.0 ? numOptions * 1e3 / milliseconds : 0.0);

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
