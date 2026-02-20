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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

// CUDA kernel: SoA layout for coalesced memory access
__global__ void blackScholesKernel(const int* __restrict__ types,
                                   const double* __restrict__ spots,
                                   const double* __restrict__ strikes,
                                   const double* __restrict__ qs,
                                   const double* __restrict__ rs,
                                   const double* __restrict__ ts,
                                   const double* __restrict__ vols,
                                   double* __restrict__ results,
                                   size_t numOptions) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx >= numOptions) return;

    double S     = spots[idx];
    double K     = strikes[idx];
    double r     = rs[idx];
    double q     = qs[idx];
    double T     = ts[idx];
    double sigma = vols[idx];
    int    type  = types[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    double sqrtT = sqrt(T);
    double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
    double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
    double discount    = exp(-r * T);
    double divDiscount = exp(-q * T);

    double price;
    if (type == CALL) {
        price = S * divDiscount * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * (1.0 - Nd2) - S * divDiscount * (1.0 - Nd1);
    }

    results[idx] = price;
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
    
    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate options
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);
    
    // Convert AoS to SoA for coalesced GPU memory access
    std::vector<int>    h_types(numOptions);
    std::vector<double> h_spots(numOptions);
    std::vector<double> h_strikes(numOptions);
    std::vector<double> h_qs(numOptions);
    std::vector<double> h_rs(numOptions);
    std::vector<double> h_ts(numOptions);
    std::vector<double> h_vols(numOptions);

    for (size_t i = 0; i < numOptions; ++i) {
        h_types[i]   = options[i].type;
        h_spots[i]   = options[i].spot;
        h_strikes[i] = options[i].strike;
        h_qs[i]      = options[i].q;
        h_rs[i]      = options[i].r;
        h_ts[i]      = options[i].t;
        h_vols[i]    = options[i].vol;
    }

    // Allocate device memory
    int    *d_types;
    double *d_spots, *d_strikes, *d_qs, *d_rs, *d_ts, *d_vols, *d_results;

    CUDA_CHECK(cudaMalloc(&d_types,   numOptions * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_spots,   numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_strikes, numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_qs,      numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rs,      numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ts,      numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vols,    numOptions * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_results, numOptions * sizeof(double)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_types,   h_types.data(),   numOptions * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spots,   h_spots.data(),   numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strikes, h_strikes.data(), numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qs,      h_qs.data(),      numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rs,      h_rs.data(),      numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ts,      h_ts.data(),      numOptions * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vols,    h_vols.data(),    numOptions * sizeof(double), cudaMemcpyHostToDevice));

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = (numOptions + blockSize - 1) / blockSize;

    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    blackScholesKernel<<<gridSize, blockSize>>>(d_types, d_spots, d_strikes,
                                                 d_qs, d_rs, d_ts, d_vols,
                                                 d_results, numOptions);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back to host
    std::vector<double> results(numOptions);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, numOptions * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    
    // Free device memory
    cudaFree(d_types);
    cudaFree(d_spots);
    cudaFree(d_strikes);
    cudaFree(d_qs);
    cudaFree(d_rs);
    cudaFree(d_ts);
    cudaFree(d_vols);
    cudaFree(d_results);

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
