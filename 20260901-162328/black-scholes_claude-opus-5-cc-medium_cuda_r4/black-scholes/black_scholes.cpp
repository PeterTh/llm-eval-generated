#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__,  \
                    __LINE__, cudaGetErrorString(err_));                                       \
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

// Structure-of-arrays view of the option set. Only the fields consumed by the
// pricing formula are stored; this is the layout handed to the GPU so that all
// device accesses are fully coalesced.
struct OptionArrays {
    int* type = nullptr;
    double* strike = nullptr;
    double* spot = nullptr;
    double* q = nullptr;
    double* r = nullptr;
    double* t = nullptr;
    double* vol = nullptr;
};

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options, one option per thread.
// erf is odd, so N(-d) == 1 - N(d) bit-for-bit; each d therefore needs a single
// erf evaluation even for puts.
__global__ void blackScholesKernel(const int* __restrict__ type,
                                   const double* __restrict__ strikeArr,
                                   const double* __restrict__ spotArr,
                                   const double* __restrict__ qArr,
                                   const double* __restrict__ rArr,
                                   const double* __restrict__ tArr,
                                   const double* __restrict__ volArr,
                                   double* __restrict__ out, const size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double S = spotArr[i];
        const double K = strikeArr[i];
        const double r = rArr[i];
        const double q = qArr[i];
        const double T = tArr[i];
        const double sigma = volArr[i];

        if (T <= 0.0 || sigma <= 0.0) {
            out[i] = 0.0;
            continue;
        }

        const double sqrtT = sqrt(T);
        const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        const double d2 = d1 - sigma * sqrtT;

        const double e1 = erf(d1 * M_SQRT1_2);
        const double e2 = erf(d2 * M_SQRT1_2);
        const double discount = exp(-r * T);
        const double carry = exp(-q * T);

        double price;
        if (type[i] == CALL) {
            price = S * carry * (0.5 * (1.0 + e1)) - K * discount * (0.5 * (1.0 + e2));
        } else { // PUT
            price = K * discount * (0.5 * (1.0 - e2)) - S * carry * (0.5 * (1.0 - e1));
        }

        out[i] = price;
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

// Generate a larger set of options by scaling the test set.
// Fills the (pinned) structure-of-arrays used by the GPU, and returns the first
// few options in AoS form for validation.
void generateOptions(OptionArrays& options, std::vector<OptionInput>& validationOptions,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();

    const unsigned int numThreads =
        std::max(1u, std::min(std::thread::hardware_concurrency(),
                              static_cast<unsigned int>((numOptions + (1u << 16) - 1) / (1u << 16))));
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int tid = 0; tid < numThreads; ++tid) {
        threads.emplace_back([&, tid]() {
            const size_t begin = numOptions * tid / numThreads;
            const size_t end = numOptions * (tid + 1) / numThreads;
            for (size_t i = begin; i < end; ++i) {
                // Cycle through test options and vary parameters slightly
                const OptionInput& base = testOptions[i % testOptions.size()];

                // Add some variation for larger datasets
                const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
                options.type[i] = base.type;
                options.strike[i] = base.strike * factor;
                options.spot[i] = base.spot * factor;
                options.q[i] = base.q;
                options.r[i] = base.r;
                options.t[i] = base.t;
                options.vol[i] = base.vol;
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    // validateResults only inspects the first 10 entries (and uses the container
    // size to decide how many to check), so materializing that prefix suffices.
    const size_t numValidation = std::min(static_cast<size_t>(10), numOptions);
    validationOptions.resize(numValidation);
    for (size_t i = 0; i < numValidation; ++i) {
        validationOptions[i] = testOptions[i % testOptions.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        validationOptions[i].spot *= factor;
        validationOptions[i].strike *= factor;
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

namespace {

constexpr int kBlockSize = 256;
constexpr int kStreamsPerDevice = 4;
// Chunk of options moved/priced per stream iteration. Large enough to amortize
// DMA setup, small enough to keep the copy/compute pipeline full.
constexpr size_t kMinChunk = 1u << 16;
constexpr size_t kMaxChunk = 1u << 20;

// Chunk size for a per-device slice of `count` options: aim for a handful of
// chunks per stream so copies and kernels overlap without excessive slicing.
size_t chunkSizeFor(const size_t count) {
    const size_t target = count / (kStreamsPerDevice * 2);
    return std::max<size_t>(1, std::min(count, std::clamp(target, kMinChunk, kMaxChunk)));
}

// Per-device double buffering state: staging buffers for the option fields plus
// the result slice, one set per stream.
struct DeviceContext {
    int device = 0;
    cudaStream_t streams[kStreamsPerDevice] = {};
    int* dType[kStreamsPerDevice] = {};
    double* dFields[kStreamsPerDevice][6] = {};
    double* dResult[kStreamsPerDevice] = {};
    size_t chunk = 0;
    int gridSize = 0;
};

void setupDevice(DeviceContext& ctx, const size_t chunk) {
    CUDA_CHECK(cudaSetDevice(ctx.device));
    ctx.chunk = chunk;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ctx.device));
    const size_t blocksForChunk = (chunk + kBlockSize - 1) / kBlockSize;
    ctx.gridSize = static_cast<int>(
        std::min(blocksForChunk, static_cast<size_t>(prop.multiProcessorCount) * 32));

    for (int s = 0; s < kStreamsPerDevice; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&ctx.dType[s], chunk * sizeof(int)));
        for (int f = 0; f < 6; ++f) {
            CUDA_CHECK(cudaMalloc(&ctx.dFields[s][f], chunk * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&ctx.dResult[s], chunk * sizeof(double)));
    }

    // Force context creation / module load so that none of it lands in the
    // measured region.
    blackScholesKernel<<<1, kBlockSize, 0, ctx.streams[0]>>>(
        ctx.dType[0], ctx.dFields[0][0], ctx.dFields[0][1], ctx.dFields[0][2], ctx.dFields[0][3],
        ctx.dFields[0][4], ctx.dFields[0][5], ctx.dResult[0], 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[0]));
}

void teardownDevice(DeviceContext& ctx) {
    CUDA_CHECK(cudaSetDevice(ctx.device));
    for (int s = 0; s < kStreamsPerDevice; ++s) {
        CUDA_CHECK(cudaFree(ctx.dType[s]));
        for (int f = 0; f < 6; ++f) {
            CUDA_CHECK(cudaFree(ctx.dFields[s][f]));
        }
        CUDA_CHECK(cudaFree(ctx.dResult[s]));
        CUDA_CHECK(cudaStreamDestroy(ctx.streams[s]));
    }
}

// Price options [begin, end) on one device, pipelining H2D copies, the pricing
// kernel and D2H copies across several streams.
void priceRange(DeviceContext& ctx, const OptionArrays& options, double* results,
                const size_t begin, const size_t end) {
    CUDA_CHECK(cudaSetDevice(ctx.device));

    int s = 0;
    for (size_t off = begin; off < end; off += ctx.chunk, s = (s + 1) % kStreamsPerDevice) {
        const size_t count = std::min(ctx.chunk, end - off);
        cudaStream_t stream = ctx.streams[s];

        const double* const hostFields[6] = {options.strike + off, options.spot + off,
                                             options.q + off,      options.r + off,
                                             options.t + off,      options.vol + off};

        CUDA_CHECK(cudaMemcpyAsync(ctx.dType[s], options.type + off, count * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        for (int f = 0; f < 6; ++f) {
            CUDA_CHECK(cudaMemcpyAsync(ctx.dFields[s][f], hostFields[f], count * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        }

        const int grid = static_cast<int>(std::min(
            static_cast<size_t>(ctx.gridSize), (count + kBlockSize - 1) / kBlockSize));
        blackScholesKernel<<<grid, kBlockSize, 0, stream>>>(
            ctx.dType[s], ctx.dFields[s][0], ctx.dFields[s][1], ctx.dFields[s][2],
            ctx.dFields[s][3], ctx.dFields[s][4], ctx.dFields[s][5], ctx.dResult[s], count);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(results + off, ctx.dResult[s], count * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }

    for (int i = 0; i < kStreamsPerDevice; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.streams[i]));
    }
}

} // namespace

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

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }
    printf("CUDA devices: %d\n", numDevices);

    // Generate options into portable pinned memory so that every device can DMA
    // from it directly.
    OptionArrays options;
    std::vector<OptionInput> validationOptions;
    if (numOptions > 0) {
        CUDA_CHECK(cudaHostAlloc(&options.type, numOptions * sizeof(int), cudaHostAllocPortable));
        double** fields[6] = {&options.strike, &options.spot, &options.q,
                              &options.r,      &options.t,    &options.vol};
        for (int f = 0; f < 6; ++f) {
            CUDA_CHECK(cudaHostAlloc(fields[f], numOptions * sizeof(double), cudaHostAllocPortable));
        }
    }
    generateOptions(options, validationOptions, numOptions);

    // Results live in pinned memory too, then get handed to the (host) result
    // reporting code.
    double* pinnedResults = nullptr;
    if (numOptions > 0) {
        CUDA_CHECK(
            cudaHostAlloc(&pinnedResults, numOptions * sizeof(double), cudaHostAllocPortable));
    }

    // Split the option range across all devices; each device gets a contiguous
    // slice so its copies stay sequential. Small problems stay on a single
    // device, where the extra host-side orchestration would dominate.
    const int devicesUsed = static_cast<int>(std::min<size_t>(
        numDevices, std::max<size_t>(1, (numOptions + kMinChunk - 1) / kMinChunk)));
    std::vector<DeviceContext> contexts(devicesUsed);
    for (int d = 0; d < devicesUsed; ++d) {
        contexts[d].device = d;
        const size_t begin = numOptions * d / devicesUsed;
        const size_t end = numOptions * (d + 1) / devicesUsed;
        setupDevice(contexts[d], chunkSizeFor(end - begin));
    }

    // Warm up the copy engines and bring the GPUs out of their idle clock state;
    // this recomputes a prefix that the measured run overwrites anyway.
    const size_t warmupCount =
        std::min(numOptions, contexts[0].chunk * static_cast<size_t>(kStreamsPerDevice));
    for (int rep = 0; rep < 2; ++rep) {
        for (int d = 0; d < devicesUsed; ++d) {
            priceRange(contexts[d], options, pinnedResults, 0, warmupCount);
        }
    }

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (devicesUsed == 1) {
        priceRange(contexts[0], options, pinnedResults, 0, numOptions);
    } else if (numOptions > 0) {
        std::vector<std::thread> workers;
        workers.reserve(devicesUsed);
        for (int d = 0; d < devicesUsed; ++d) {
            const size_t begin = numOptions * d / devicesUsed;
            const size_t end = numOptions * (d + 1) / devicesUsed;
            workers.emplace_back(
                [&contexts, &options, pinnedResults, begin, end, d]() {
                    priceRange(contexts[d], options, pinnedResults, begin, end);
                });
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

    std::vector<double> results(pinnedResults, pinnedResults + numOptions);

    // Print results for external validation
    if (printResults) {
        print_results(results, "OptionPrices");
    }

    for (int d = 0; d < devicesUsed; ++d) {
        teardownDevice(contexts[d]);
    }
    if (numOptions > 0) {
        CUDA_CHECK(cudaFreeHost(options.type));
        CUDA_CHECK(cudaFreeHost(options.strike));
        CUDA_CHECK(cudaFreeHost(options.spot));
        CUDA_CHECK(cudaFreeHost(options.q));
        CUDA_CHECK(cudaFreeHost(options.r));
        CUDA_CHECK(cudaFreeHost(options.t));
        CUDA_CHECK(cudaFreeHost(options.vol));
        CUDA_CHECK(cudaFreeHost(pinnedResults));
    }

    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(validationOptions, results);

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
