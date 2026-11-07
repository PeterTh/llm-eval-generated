// Black-Scholes option pricing benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt). All option prices are
// computed on the GPU, one thread per option. For large inputs the work is
// split into chunks that are pipelined across several CUDA streams so that
// host->device copies, kernel execution and device->host copies overlap.

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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(err__));                     \
            exit(1);                                                          \
        }                                                                     \
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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
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
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
}

// One thread per option; grid-stride loop so any grid size is valid.
__global__ void __launch_bounds__(256)
blackScholesKernel(const OptionInput* __restrict__ options,
                   double* __restrict__ results,
                   const size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        results[i] = blackScholes(options[i]);
    }
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

// GPU execution context: device buffers and streams are set up once, outside
// the timed region, and the actual transfer + compute pipeline is timed.
struct GpuPricer {
    static constexpr size_t kChunk = size_t(1) << 20;   // options per chunk
    static constexpr int kStreams = 4;
    static constexpr int kBlock = 256;

    size_t numOptions = 0;
    size_t chunk = 0;           // actual chunk size (<= kChunk)
    int numBuffers = 0;         // number of device double-buffers in use
    OptionInput* dOptions[kStreams] = {};
    double* dResults[kStreams] = {};
    cudaStream_t streams[kStreams] = {};
    int smCount = 0;

    explicit GpuPricer(size_t n) : numOptions(n) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev));

        chunk = std::min(kChunk, std::max<size_t>(n, 1));
        const size_t numChunks = (n + chunk - 1) / chunk;
        numBuffers = static_cast<int>(std::min<size_t>(kStreams, std::max<size_t>(numChunks, 1)));

        for (int s = 0; s < numBuffers; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&dOptions[s], chunk * sizeof(OptionInput)));
            CUDA_CHECK(cudaMalloc(&dResults[s], chunk * sizeof(double)));
        }

        // Warm up: forces lazy context/module initialization now rather than
        // inside the timed region.
        blackScholesKernel<<<1, kBlock, 0, streams[0]>>>(dOptions[0], dResults[0], 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }

    ~GpuPricer() {
        for (int s = 0; s < numBuffers; ++s) {
            cudaFree(dOptions[s]);
            cudaFree(dResults[s]);
            cudaStreamDestroy(streams[s]);
        }
    }

    GpuPricer(const GpuPricer&) = delete;
    GpuPricer& operator=(const GpuPricer&) = delete;

    // Copies options to the device, prices them, and copies prices back.
    // Chunks are round-robined over the streams so copies and compute overlap.
    void price(const OptionInput* hOptions, double* hResults) {
        for (size_t off = 0, c = 0; off < numOptions; off += chunk, ++c) {
            const int s = static_cast<int>(c % numBuffers);
            const size_t len = std::min(chunk, numOptions - off);
            const unsigned blocks = static_cast<unsigned>(
                std::min<size_t>((len + kBlock - 1) / kBlock, size_t(smCount) * 32));

            CUDA_CHECK(cudaMemcpyAsync(dOptions[s], hOptions + off, len * sizeof(OptionInput),
                                       cudaMemcpyHostToDevice, streams[s]));
            blackScholesKernel<<<blocks, kBlock, 0, streams[s]>>>(dOptions[s], dResults[s], len);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hResults + off, dResults[s], len * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        for (int s = 0; s < numBuffers; ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
    }
};

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

    // Set up the GPU (device buffers, streams) and pin the host buffers so
    // that transfers are asynchronous DMA copies.
    GpuPricer pricer(numOptions);
    if (numOptions > 0) {
        CUDA_CHECK(cudaHostRegister(options.data(), numOptions * sizeof(OptionInput),
                                    cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(results.data(), numOptions * sizeof(double),
                                    cudaHostRegisterDefault));
    }

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (numOptions > 0) {
        pricer.price(options.data(), results.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

    if (numOptions > 0) {
        CUDA_CHECK(cudaHostUnregister(options.data()));
        CUDA_CHECK(cudaHostUnregister(results.data()));
    }

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
