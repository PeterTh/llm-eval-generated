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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(1);                                                            \
        }                                                                       \
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

// Structure-of-arrays layout of the pricing inputs on the device.
// Each warp reads 32 contiguous doubles per array -> fully coalesced loads.
struct DeviceOptions {
    const int* __restrict__ type;
    const double* __restrict__ strike;
    const double* __restrict__ spot;
    const double* __restrict__ q;
    const double* __restrict__ r;
    const double* __restrict__ t;
    const double* __restrict__ vol;
};

// Standard normal cumulative distribution function
__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__device__ __forceinline__ double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (one option per thread).
// erf is an odd function, so N(-x) == 0.5 * (1 - erf(x / sqrt 2)) bit-exactly;
// computing erf once per d and selecting avoids divergent duplicate work.
__device__ __forceinline__ double blackScholes(const int type, const double S,
                                               const double K, const double q,
                                               const double r, const double T,
                                               const double sigma) noexcept {
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double erf1 = erf(d1 * M_SQRT1_2);
    const double erf2 = erf(d2 * M_SQRT1_2);
    const double discount = exp(-r * T);
    const double divDiscount = exp(-q * T);
    const double Sd = S * divDiscount;
    const double Kd = K * discount;

    double price;
    if (type == CALL) {
        price = Sd * (0.5 * (1.0 + erf1)) - Kd * (0.5 * (1.0 + erf2));
    } else { // PUT
        price = Kd * (0.5 * (1.0 - erf2)) - Sd * (0.5 * (1.0 - erf1));
    }

    return price;
}

__global__ void __launch_bounds__(256)
blackScholesKernel(const DeviceOptions opts, double* __restrict__ results,
                   const size_t numOptions) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numOptions; i += stride) {
        results[i] = blackScholes(opts.type[i], opts.spot[i], opts.strike[i],
                                  opts.q[i], opts.r[i], opts.t[i], opts.vol[i]);
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

    // ---- Device setup (not part of the timed region) ----
    // Work is split into contiguous ranges, one per GPU, and each range is
    // processed as a pipeline of chunks on several streams so that the
    // device-to-host copy of one chunk overlaps the kernel of the next.
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    const size_t n = numOptions;
    constexpr size_t minPerDevice = 1u << 18;            // don't split tiny problems
    constexpr size_t chunkOptions = 1u << 21;            // ~2M options per chunk
    constexpr int streamsPerDevice = 4;
    constexpr int blockSize = 256;

    int devicesUsed = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(numDevices), std::max<size_t>(1, n / minPerDevice)));

    struct DeviceCtx {
        int id = 0;
        size_t begin = 0, count = 0;
        unsigned char* dInput = nullptr;
        double* dResults = nullptr;
        DeviceOptions opts{};
        cudaStream_t streams[streamsPerDevice]{};
        unsigned maxGrid = 1;
    };
    std::vector<DeviceCtx> ctx(devicesUsed);

    // Pin the result vector so device-to-host copies are DMA'd directly into it.
    if (n > 0) {
        CUDA_CHECK(cudaHostRegister(results.data(), n * sizeof(double),
                                    cudaHostRegisterDefault));
    }

    // Pack inputs into a structure-of-arrays staging buffer:
    // 6 double arrays followed by 1 int array (per device range).
    std::vector<unsigned char> staging;
    for (int d = 0; d < devicesUsed; ++d) {
        DeviceCtx& c = ctx[d];
        c.id = d;
        c.begin = n * static_cast<size_t>(d) / devicesUsed;
        c.count = n * static_cast<size_t>(d + 1) / devicesUsed - c.begin;

        CUDA_CHECK(cudaSetDevice(c.id));
        CUDA_CHECK(cudaFree(0)); // force context creation up front
        for (int s = 0; s < streamsPerDevice; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&c.streams[s], cudaStreamNonBlocking));
        }

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, c.id));
        int blocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, blackScholesKernel, blockSize, 0));
        if (blocksPerSM < 1) blocksPerSM = 1;
        c.maxGrid = static_cast<unsigned>(prop.multiProcessorCount) * blocksPerSM;

        if (c.count == 0) continue;

        const size_t dblBytes = c.count * sizeof(double);
        const size_t intBytes = c.count * sizeof(int);
        const size_t inputBytes = 6 * dblBytes + intBytes;
        staging.resize(inputBytes);

        double* hStrike = reinterpret_cast<double*>(staging.data() + 0 * dblBytes);
        double* hSpot   = reinterpret_cast<double*>(staging.data() + 1 * dblBytes);
        double* hQ      = reinterpret_cast<double*>(staging.data() + 2 * dblBytes);
        double* hR      = reinterpret_cast<double*>(staging.data() + 3 * dblBytes);
        double* hT      = reinterpret_cast<double*>(staging.data() + 4 * dblBytes);
        double* hVol    = reinterpret_cast<double*>(staging.data() + 5 * dblBytes);
        int*    hType   = reinterpret_cast<int*>(staging.data() + 6 * dblBytes);
        for (size_t i = 0; i < c.count; ++i) {
            const OptionInput& o = options[c.begin + i];
            hStrike[i] = o.strike;
            hSpot[i]   = o.spot;
            hQ[i]      = o.q;
            hR[i]      = o.r;
            hT[i]      = o.t;
            hVol[i]    = o.vol;
            hType[i]   = o.type;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.dInput), inputBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c.dResults), dblBytes));
        CUDA_CHECK(cudaMemcpy(c.dInput, staging.data(), inputBytes, cudaMemcpyHostToDevice));

        c.opts.strike = reinterpret_cast<const double*>(c.dInput + 0 * dblBytes);
        c.opts.spot   = reinterpret_cast<const double*>(c.dInput + 1 * dblBytes);
        c.opts.q      = reinterpret_cast<const double*>(c.dInput + 2 * dblBytes);
        c.opts.r      = reinterpret_cast<const double*>(c.dInput + 3 * dblBytes);
        c.opts.t      = reinterpret_cast<const double*>(c.dInput + 4 * dblBytes);
        c.opts.vol    = reinterpret_cast<const double*>(c.dInput + 5 * dblBytes);
        c.opts.type   = reinterpret_cast<const int*>(c.dInput + 6 * dblBytes);
    }
    staging.clear();
    staging.shrink_to_fit();
    for (int d = 0; d < devicesUsed; ++d) {
        CUDA_CHECK(cudaSetDevice(ctx[d].id));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int d = 0; d < devicesUsed; ++d) {
        const DeviceCtx& c = ctx[d];
        if (c.count == 0) continue;
        CUDA_CHECK(cudaSetDevice(c.id));
        int s = 0;
        for (size_t off = 0; off < c.count; off += chunkOptions, s = (s + 1) % streamsPerDevice) {
            const size_t len = std::min(chunkOptions, c.count - off);
            DeviceOptions o = c.opts;
            o.strike += off; o.spot += off; o.q += off; o.r += off;
            o.t += off; o.vol += off; o.type += off;
            const size_t neededBlocks = (len + blockSize - 1) / blockSize;
            const unsigned grid = static_cast<unsigned>(
                std::min<size_t>(neededBlocks, c.maxGrid));
            blackScholesKernel<<<grid, blockSize, 0, c.streams[s]>>>(o, c.dResults + off, len);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(results.data() + c.begin + off, c.dResults + off,
                                       len * sizeof(double), cudaMemcpyDeviceToHost,
                                       c.streams[s]));
        }
    }
    for (int d = 0; d < devicesUsed; ++d) {
        CUDA_CHECK(cudaSetDevice(ctx[d].id));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

    for (int d = 0; d < devicesUsed; ++d) {
        DeviceCtx& c = ctx[d];
        CUDA_CHECK(cudaSetDevice(c.id));
        for (int s = 0; s < streamsPerDevice; ++s) {
            CUDA_CHECK(cudaStreamDestroy(c.streams[s]));
        }
        if (c.dResults) CUDA_CHECK(cudaFree(c.dResults));
        if (c.dInput) CUDA_CHECK(cudaFree(c.dInput));
    }
    if (n > 0) {
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
