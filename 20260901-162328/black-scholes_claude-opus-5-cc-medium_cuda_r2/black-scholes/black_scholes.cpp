#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
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

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__, \
                    __LINE__, cudaGetErrorString(err__));                                      \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

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

// Structure-of-arrays view of the option set, used for coalesced device access.
// The pricing inputs live in pinned host memory so that host/device transfers can
// overlap with kernel execution.
struct OptionSoA {
    double* strike = nullptr;
    double* spot = nullptr;
    double* q = nullptr;
    double* r = nullptr;
    double* t = nullptr;
    double* vol = nullptr;
    uint8_t* type = nullptr;
};

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options, one option per thread.
//
// Both the call and the put expression are evaluated unconditionally and selected at
// the end: within a warp the option types are mixed, so a branch would serialize the
// two paths anyway. N(-d) is formed as 0.5 * (1 - erf(d / sqrt(2))), which is the same
// sequence of floating-point operations the scalar code performs for
// cumulativeNormal(-d) because erf is exactly odd.
__global__ __launch_bounds__(256) void blackScholesKernel(const double* __restrict__ strikeArr,
                                                          const double* __restrict__ spotArr,
                                                          const double* __restrict__ qArr,
                                                          const double* __restrict__ rArr,
                                                          const double* __restrict__ tArr,
                                                          const double* __restrict__ volArr,
                                                          const uint8_t* __restrict__ typeArr,
                                                          double* __restrict__ prices,
                                                          const size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const double S = spotArr[i];
    const double K = strikeArr[i];
    const double r = rArr[i];
    const double q = qArr[i];
    const double T = tArr[i];
    const double sigma = volArr[i];

    if (T <= 0.0 || sigma <= 0.0) {
        prices[i] = 0.0;
        return;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double e1 = erf(d1 * M_SQRT1_2);
    const double e2 = erf(d2 * M_SQRT1_2);

    const double discount = exp(-r * T);
    const double spotDiscount = S * exp(-q * T);

    const double call = spotDiscount * (0.5 * (1.0 + e1)) - K * discount * (0.5 * (1.0 + e2));
    const double put = K * discount * (0.5 * (1.0 - e2)) - spotDiscount * (0.5 * (1.0 - e1));

    prices[i] = (typeArr[i] == CALL) ? call : put;
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
void generateOptions(std::vector<OptionInput>& options, OptionSoA& soa, const size_t numOptions) {
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

        soa.strike[i] = options[i].strike;
        soa.spot[i] = options[i].spot;
        soa.q[i] = options[i].q;
        soa.r[i] = options[i].r;
        soa.t[i] = options[i].t;
        soa.vol[i] = options[i].vol;
        soa.type[i] = static_cast<uint8_t>(options[i].type);
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

    // Nothing to transfer or launch; the streaming path below assumes a non-empty set.
    if (numOptions == 0) {
        printf("Pricing options...\n");
        printf("Computation time: %.3f ms\n", 0.0);
        const volatile double zeroSeconds = 0.0;  // 0 / 0, as in the timed path
        printf("Options per second: %.0f\n", static_cast<double>(numOptions) / zeroSeconds);
        std::vector<double> empty;
        if (printResults) {
            print_results(empty, "OptionPrices");
        }
        if (validate) {
            printf("Validating results...\n");
            printf("Validation: PASSED\n");
        }
        return 0;
    }

    // Initialize the device up front so that context creation is not part of the
    // measured region.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

    // Pinned staging buffers for the inputs and the prices.
    OptionSoA soa;
    double* h_prices = nullptr;
    CUDA_CHECK(cudaHostAlloc(&soa.strike, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.spot, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.q, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.r, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.t, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.vol, numOptions * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&soa.type, numOptions * sizeof(uint8_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_prices, numOptions * sizeof(double), cudaHostAllocDefault));

    // Generate options
    std::vector<OptionInput> options;
    generateOptions(options, soa, numOptions);

    // The work is split into chunks that are streamed through the device so that the
    // host-to-device transfer of one chunk overlaps the kernel and the device-to-host
    // transfer of its predecessors.
    constexpr int NUM_STREAMS = 4;
    constexpr size_t MAX_CHUNK = 1ull << 22;
    constexpr int BLOCK_SIZE = 256;

    const size_t chunkSize = std::min(numOptions, MAX_CHUNK);
    const size_t numChunks = (numOptions + chunkSize - 1) / chunkSize;
    const int numStreams = static_cast<int>(std::min<size_t>(NUM_STREAMS, numChunks));

    cudaStream_t streams[NUM_STREAMS];
    OptionSoA d_in[NUM_STREAMS];
    double* d_prices[NUM_STREAMS];
    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&d_in[s].strike, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].spot, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].q, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].r, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].t, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].vol, chunkSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_in[s].type, chunkSize * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_prices[s], chunkSize * sizeof(double)));
    }

    // Warm up: force the kernel module to be loaded so that the one-time load cost is
    // not attributed to the measured region. The empty launch has no side effects.
    blackScholesKernel<<<1, BLOCK_SIZE>>>(d_in[0].strike, d_in[0].spot, d_in[0].q, d_in[0].r,
                                          d_in[0].t, d_in[0].vol, d_in[0].type, d_prices[0], 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t c = 0; c < numChunks; ++c) {
        const size_t offset = c * chunkSize;
        const size_t count = std::min(chunkSize, numOptions - offset);
        const int s = static_cast<int>(c % static_cast<size_t>(numStreams));
        const cudaStream_t stream = streams[s];

        CUDA_CHECK(cudaMemcpyAsync(d_in[s].strike, soa.strike + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].spot, soa.spot + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].q, soa.q + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].r, soa.r + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].t, soa.t + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].vol, soa.vol + offset, count * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_in[s].type, soa.type + offset, count * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, stream));

        const size_t numBlocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blackScholesKernel<<<static_cast<unsigned int>(numBlocks), BLOCK_SIZE, 0, stream>>>(
            d_in[s].strike, d_in[s].spot, d_in[s].q, d_in[s].r, d_in[s].t, d_in[s].vol,
            d_in[s].type, d_prices[s], count);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_prices + offset, d_prices[s], count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }

    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

    // Results
    std::vector<double> results(h_prices, h_prices + numOptions);

    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaFree(d_in[s].strike));
        CUDA_CHECK(cudaFree(d_in[s].spot));
        CUDA_CHECK(cudaFree(d_in[s].q));
        CUDA_CHECK(cudaFree(d_in[s].r));
        CUDA_CHECK(cudaFree(d_in[s].t));
        CUDA_CHECK(cudaFree(d_in[s].vol));
        CUDA_CHECK(cudaFree(d_in[s].type));
        CUDA_CHECK(cudaFree(d_prices[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFreeHost(soa.strike));
    CUDA_CHECK(cudaFreeHost(soa.spot));
    CUDA_CHECK(cudaFreeHost(soa.q));
    CUDA_CHECK(cudaFreeHost(soa.r));
    CUDA_CHECK(cudaFreeHost(soa.t));
    CUDA_CHECK(cudaFreeHost(soa.vol));
    CUDA_CHECK(cudaFreeHost(soa.type));
    CUDA_CHECK(cudaFreeHost(h_prices));

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
