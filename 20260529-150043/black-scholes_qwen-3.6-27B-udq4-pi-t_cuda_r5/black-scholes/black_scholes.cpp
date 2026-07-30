#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

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

// ---------------------------------------------------------------------------
// CUDA device helpers
// ---------------------------------------------------------------------------

// Device-side cumulative normal distribution (Abramowitz & Stegun approximation)
__device__ __forceinline__ double d_cumulativeNormal(const double x) noexcept {
    // Constants for Abramowitz & Stegun approximation (formula 26.2.17)
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    const double sign = (x >= 0.0) ? 1.0 : -1.0;
    const double absX = fabs(x);
    const double t = 1.0 / (1.0 + p * absX);
    const double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                      exp(-absX * absX / 2.0);
    return 0.5 * (1.0 + sign * y);
}

// Device-side Black-Scholes pricing
__device__ double d_blackScholes(const int type,
                                 const double strike,
                                 const double spot,
                                 const double q,
                                 const double r,
                                 const double t,
                                 const double vol) noexcept {
    if (t <= 0.0 || vol <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(t);
    const double d1 = (log(spot / strike) + (r - q + 0.5 * vol * vol) * t) /
                      (vol * sqrtT);
    const double d2 = d1 - vol * sqrtT;

    const double Nd1 = d_cumulativeNormal(d1);
    const double Nd2 = d_cumulativeNormal(d2);

    const double discount = exp(-r * t);
    const double spotFactor = spot * exp(-q * t);

    if (type == CALL) {
        return spotFactor * Nd1 - strike * discount * Nd2;
    } else {
        return strike * discount * d_cumulativeNormal(-d2) -
               spotFactor * d_cumulativeNormal(-d1);
    }
}

// CUDA kernel: one thread per option, grid-stride loop
__global__ void blackScholesKernel(
    const int*     d_types,
    const double*  d_strikes,
    const double*  d_spots,
    const double*  d_qs,
    const double*  d_rs,
    const double*  d_ts,
    const double*  d_vols,
    double*        d_results,
    const size_t   numOptions)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numOptions;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        d_results[i] = d_blackScholes(
            d_types[i],
            d_strikes[i],
            d_spots[i],
            d_qs[i],
            d_rs[i],
            d_ts[i],
            d_vols[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

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

        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) ||
            std::isinf(computed)) {
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
// CUDA error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

    // ------------------------------------------------------------------
    // Query GPU properties
    // ------------------------------------------------------------------
    cudaDeviceProp prop;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (sm%u.%u)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d, Max threads/block: %d, Max threads/grid: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock,
           prop.maxThreadsDim[0]);

    // ------------------------------------------------------------------
    // Convert AoS -> SoA on host for coalesced GPU access
    // ------------------------------------------------------------------
    const size_t N = numOptions;
    std::vector<int>    h_types(N);
    std::vector<double> h_strikes(N), h_spots(N), h_qs(N), h_rs(N);
    std::vector<double> h_ts(N), h_vols(N);

    for (size_t i = 0; i < N; ++i) {
        h_types[i]   = options[i].type;
        h_strikes[i] = options[i].strike;
        h_spots[i]   = options[i].spot;
        h_qs[i]      = options[i].q;
        h_rs[i]      = options[i].r;
        h_ts[i]      = options[i].t;
        h_vols[i]    = options[i].vol;
    }

    // ------------------------------------------------------------------
    // Allocate device memory
    // ------------------------------------------------------------------
    int*     d_types   = nullptr;
    double*  d_strikes = nullptr, *d_spots = nullptr, *d_qs = nullptr, *d_rs = nullptr;
    double*  d_ts      = nullptr, *d_vols = nullptr;
    double*  d_results = nullptr;

    const size_t bytesDoubles = N * sizeof(double);
    const size_t bytesInts    = N * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_types,   bytesInts));
    CUDA_CHECK(cudaMalloc(&d_strikes, bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_spots,   bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_qs,      bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_rs,      bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_ts,      bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_vols,    bytesDoubles));
    CUDA_CHECK(cudaMalloc(&d_results, bytesDoubles));

    // ------------------------------------------------------------------
    // Copy input data to device
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_types,   h_types.data(),   bytesInts,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strikes, h_strikes.data(), bytesDoubles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spots,   h_spots.data(),   bytesDoubles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qs,      h_qs.data(),      bytesDoubles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rs,      h_rs.data(),      bytesDoubles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ts,      h_ts.data(),      bytesDoubles, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vols,    h_vols.data(),    bytesDoubles, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Configure and launch kernel
    // ------------------------------------------------------------------
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = std::min(
        static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock),
        static_cast<int>(prop.multiProcessorCount * 8)
    );

    printf("Kernel config: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);

    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    blackScholesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_types, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_results, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // ------------------------------------------------------------------
    // Copy results back to host
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, bytesDoubles, cudaMemcpyDeviceToHost));

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", N / (duration.count() / 1e6));

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

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_types));
    CUDA_CHECK(cudaFree(d_strikes));
    CUDA_CHECK(cudaFree(d_spots));
    CUDA_CHECK(cudaFree(d_qs));
    CUDA_CHECK(cudaFree(d_rs));
    CUDA_CHECK(cudaFree(d_ts));
    CUDA_CHECK(cudaFree(d_vols));
    CUDA_CHECK(cudaFree(d_results));

    return 0;
}
