#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

static_assert(std::is_trivially_copyable_v<OptionInput>,
              "OptionInput must be safe to copy to a CUDA device");

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Black-Scholes formula for European options.  This remains available on the
// host for validation and documents the identical calculation in the kernel.
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    }
    return K * discount * cumulativeNormal(-d2) -
           S * exp(-q * T) * cumulativeNormal(-d1);
}

// CUDA equivalents deliberately use double precision and the device math
// library so the benchmark preserves the original numerical semantics.
__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double blackScholesDevice(const OptionInput& option) {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = cumulativeNormalDevice(d1);
    const double Nd2 = cumulativeNormalDevice(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    }
    return K * discount * cumulativeNormalDevice(-d2) -
           S * exp(-q * T) * cumulativeNormalDevice(-d1);
}

// Each CUDA thread prices one option.  Inputs are generated in parallel by
// OpenMP on the rank's host, then transferred as a contiguous MPI-rank shard.
__global__ void priceOptionsKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        results[index] = blackScholesDevice(options[index]);
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

// Generate exactly the same global option sequence as the original program.
// MPI assigns a contiguous global range to each rank; OpenMP fills that range.
void generateOptions(std::vector<OptionInput>& options,
                     const size_t globalOffset,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    // The allocation itself bounds this loop to ptrdiff_t on supported hosts.
    if (numOptions > static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        std::fprintf(stderr, "Option count is too large for this host: %zu\n", numOptions);
        std::abort();
    }

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t localIndex = 0;
         localIndex < static_cast<std::ptrdiff_t>(numOptions);
         ++localIndex) {
        const size_t globalIndex = globalOffset + static_cast<size_t>(localIndex);
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        OptionInput option = base;

        // Preserve the original global-index scaling exactly.
        const double factor = 1.0 +
                              0.1 * (globalIndex /
                                     static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[static_cast<size_t>(localIndex)] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(options.size(), results.size()));

    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[static_cast<size_t>(i)];
        const double expected = options[static_cast<size_t>(i)].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relError);

        // Retain the original benchmark's relaxed validity criterion.
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) ||
            std::isinf(computed)) {
            std::printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

struct WorkPartition {
    size_t offset;
    size_t count;
};

WorkPartition partitionWork(const size_t total, const int worldSize, const int rank) {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t base = total / ranks;
    const size_t remainder = total % ranks;
    const size_t rankSize = static_cast<size_t>(rank);
    return {rankSize * base + std::min(rankSize, remainder),
            base + (rankSize < remainder ? 1U : 0U)};
}

[[noreturn]] void failCuda(const cudaError_t error, const char* expression,
                           const char* file, const int line, const int rank) {
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d for %s: %s\n", rank, file,
                 line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(rank, call)                                                    \
    do {                                                                          \
        const cudaError_t cudaStatus = (call);                                    \
        if (cudaStatus != cudaSuccess) {                                          \
            failCuda(cudaStatus, #call, __FILE__, __LINE__, (rank));              \
        }                                                                         \
    } while (false)

void gatherResultsOnRoot(const std::vector<double>& localResults,
                         const WorkPartition localWork,
                         const size_t totalOptions,
                         const int rank,
                         const int worldSize,
                         std::vector<double>& results) {
    constexpr int resultTag = 100;
    constexpr size_t maxMpiCount = static_cast<size_t>(std::numeric_limits<int>::max());

    if (rank == 0) {
        results.resize(totalOptions);
        if (!localResults.empty()) {
            std::copy(localResults.begin(), localResults.end(),
                      results.begin() + static_cast<std::ptrdiff_t>(localWork.offset));
        }

        for (int source = 1; source < worldSize; ++source) {
            const WorkPartition sourceWork = partitionWork(totalOptions, worldSize, source);
            size_t transferred = 0;
            while (transferred < sourceWork.count) {
                const size_t chunk = std::min(maxMpiCount, sourceWork.count - transferred);
                MPI_Recv(results.data() + sourceWork.offset + transferred,
                         static_cast<int>(chunk), MPI_DOUBLE, source, resultTag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                transferred += chunk;
            }
        }
        return;
    }

    size_t transferred = 0;
    while (transferred < localResults.size()) {
        const size_t chunk = std::min(maxMpiCount, localResults.size() - transferred);
        MPI_Send(localResults.data() + transferred, static_cast<int>(chunk), MPI_DOUBLE,
                 0, resultTag, MPI_COMM_WORLD);
        transferred += chunk;
    }
}

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool invalidArguments = false;

    // Parse command line arguments identically on every MPI rank.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            invalidArguments = true;
        }
    }

    if (showHelp || invalidArguments) {
        if (rank == 0) {
            if (invalidArguments) {
                std::printf("Unknown or incomplete option supplied\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return invalidArguments ? 1 : 0;
    }

    // Map ranks to GPUs locally, avoiding a global-rank-to-device assumption
    // when an accelerator cluster has multiple nodes.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(rank, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: no CUDA devices are visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(rank, cudaSetDevice(device));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(rank, cudaGetDeviceProperties(&deviceProperties, device));

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA pricing: %s (device %d on local rank %d)\n",
                    deviceProperties.name, device, localRank);
        if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
            std::printf("MPI thread level: SINGLE (MPI calls remain outside OpenMP regions)\n");
        }
    }

    const WorkPartition localWork = partitionWork(numOptions, worldSize, rank);
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localWork.offset, localWork.count);

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localWork.count > 0) {
        CUDA_CHECK(rank, cudaMalloc(&deviceOptions, localWork.count * sizeof(OptionInput)));
        CUDA_CHECK(rank, cudaMalloc(&deviceResults, localWork.count * sizeof(double)));
        CUDA_CHECK(rank, cudaMemcpy(deviceOptions, localOptions.data(),
                                    localWork.count * sizeof(OptionInput),
                                    cudaMemcpyHostToDevice));
    }

    // Match the original benchmark's timing scope: input construction and data
    // transfer are complete before the timed option-pricing calculation begins.
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Pricing options...\n");
    }

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CUDA_CHECK(rank, cudaEventCreate(&startEvent));
    CUDA_CHECK(rank, cudaEventCreate(&stopEvent));
    CUDA_CHECK(rank, cudaEventRecord(startEvent));

    constexpr int threadsPerBlock = 256;
    const size_t maxLaunchOptions =
        static_cast<size_t>(deviceProperties.maxGridSize[0]) * threadsPerBlock;
    for (size_t launchOffset = 0; launchOffset < localWork.count;) {
        const size_t launchCount = std::min(maxLaunchOptions, localWork.count - launchOffset);
        const unsigned int blocks = static_cast<unsigned int>(
            (launchCount + threadsPerBlock - 1) / threadsPerBlock);
        priceOptionsKernel<<<blocks, threadsPerBlock>>>(deviceOptions + launchOffset,
                                                         deviceResults + launchOffset,
                                                         launchCount);
        CUDA_CHECK(rank, cudaGetLastError());
        launchOffset += launchCount;
    }

    CUDA_CHECK(rank, cudaEventRecord(stopEvent));
    CUDA_CHECK(rank, cudaEventSynchronize(stopEvent));
    float localMilliseconds = 0.0F;
    CUDA_CHECK(rank, cudaEventElapsedTime(&localMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(rank, cudaEventDestroy(startEvent));
    CUDA_CHECK(rank, cudaEventDestroy(stopEvent));

    float maximumMilliseconds = 0.0F;
    MPI_Reduce(&localMilliseconds, &maximumMilliseconds, 1, MPI_FLOAT, MPI_MAX, 0,
               MPI_COMM_WORLD);

    std::vector<double> allResults;
    std::vector<double> localResults;
    if (printResults) {
        localResults.resize(localWork.count);
        if (localWork.count > 0) {
            CUDA_CHECK(rank, cudaMemcpy(localResults.data(), deviceResults,
                                        localWork.count * sizeof(double),
                                        cudaMemcpyDeviceToHost));
        }
        gatherResultsOnRoot(localResults, localWork, numOptions, rank, worldSize, allResults);
    }

    std::vector<double> validationResults;
    const size_t validationCount = std::min<size_t>(10, numOptions);
    if (validate && !printResults) {
        // Only the values validation reads need to cross PCIe and MPI.  A sum
        // reduction is correct because each global option belongs to one rank.
        std::vector<double> localValidation(validationCount, 0.0);
        const size_t firstGlobal = localWork.offset;
        const size_t lastGlobal = localWork.offset + localWork.count;
        if (firstGlobal < validationCount) {
            const size_t count = std::min(lastGlobal, validationCount) - firstGlobal;
            CUDA_CHECK(rank, cudaMemcpy(localValidation.data() + firstGlobal,
                                        deviceResults, count * sizeof(double),
                                        cudaMemcpyDeviceToHost));
        }
        if (rank == 0) {
            validationResults.resize(validationCount);
        }
        MPI_Reduce(localValidation.data(),
                   rank == 0 ? validationResults.data() : nullptr,
                   static_cast<int>(validationCount), MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    }

    if (localWork.count > 0) {
        CUDA_CHECK(rank, cudaFree(deviceOptions));
        CUDA_CHECK(rank, cudaFree(deviceResults));
    }
    MPI_Comm_free(&localComm);

    int status = EXIT_SUCCESS;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", maximumMilliseconds);
        const double optionsPerSecond = maximumMilliseconds > 0.0F
                                            ? static_cast<double>(numOptions) /
                                                  (static_cast<double>(maximumMilliseconds) / 1.0e3)
                                            : 0.0;
        std::printf("Options per second: %.0f\n", optionsPerSecond);

        if (printResults) {
            print_results(allResults, "OptionPrices");
        }

        if (validate) {
            std::printf("Validating results...\n");
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, 0, validationCount);
            const std::vector<double>& resultsToValidate =
                printResults ? allResults : validationResults;
            if (validateResults(validationOptions, resultsToValidate)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                status = EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
