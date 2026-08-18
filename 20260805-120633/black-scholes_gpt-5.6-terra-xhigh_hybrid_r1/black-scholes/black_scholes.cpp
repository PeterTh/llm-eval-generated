#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
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

// Only values used in the pricing equation are transferred to an accelerator.
// Keeping this compact (56-byte) layout minimizes host-to-device traffic.
struct DeviceOption {
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    int type;
};

[[noreturn]] void abortWithCudaError(const cudaError_t error, const char* expression,
                                     const int rank) {
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank, expression,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(expression, rank)                                                   \
    do {                                                                                \
        const cudaError_t cuda_status_ = (expression);                                 \
        if (cuda_status_ != cudaSuccess) {                                             \
            abortWithCudaError(cuda_status_, #expression, (rank));                     \
        }                                                                               \
    } while (false)

// Standard normal cumulative distribution function.  This is deliberately
// evaluated in double precision; --use_fast_math would reduce option accuracy.
__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const DeviceOption* __restrict__ options,
                                   double* __restrict__ results, const std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const DeviceOption option = options[index];
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[index] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = cumulativeNormalDevice(d1);
    const double Nd2 = cumulativeNormalDevice(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        results[index] = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        results[index] = K * discount * cumulativeNormalDevice(-d2) -
                         S * exp(-q * T) * cumulativeNormalDevice(-d1);
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

inline OptionInput makeOption(const std::size_t globalIndex) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[globalIndex % testOptions.size()];
    const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

inline DeviceOption makeDeviceOption(const std::size_t globalIndex) noexcept {
    const OptionInput option = makeOption(globalIndex);
    return {option.strike, option.spot, option.q, option.r, option.t, option.vol, option.type};
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min<std::size_t>(10, options.size()));

    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[static_cast<std::size_t>(i)];
        const double expected = options[static_cast<std::size_t>(i)].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);

        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relError);

        // Preserve the original benchmark's deliberately relaxed validation.
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
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

enum class ParseResult : int {
    Run = 0,
    Help = 1,
    Error = 2,
};

ParseResult parseArguments(const int argc, char** argv, std::size_t& numOptions,
                           bool& validate, bool& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return ParseResult::Help;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return ParseResult::Error;
        }
    }
    return ParseResult::Run;
}

// A blocked distribution preserves global output order and makes each rank's
// option generation fully independent, avoiding broadcasts of the input data.
void splitRange(const std::size_t total, const int rank, const int ranks,
                std::size_t& first, std::size_t& count) {
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t base = total / rankCount;
    const std::size_t remainder = total % rankCount;
    count = base + (rankIndex < remainder ? 1 : 0);
    first = rankIndex * base + std::min(rankIndex, remainder);
}

// MPI_Gatherv uses int counts.  Result printing is a diagnostic path and this
// chunked gather keeps it correct even for vectors larger than INT_MAX.
void gatherResultsToRoot(const double* localResults, const std::size_t localCount,
                         const std::size_t first, const int rank, const int ranks,
                         std::vector<double>& globalResults) {
    constexpr std::size_t mpiChunk = 1U << 20;
    constexpr int mpiTag = 731;

    if (rank == 0) {
        if (localCount != 0) {
            std::copy_n(localResults, localCount, globalResults.data() + first);
        }
        for (int source = 1; source < ranks; ++source) {
            std::size_t sourceFirst = 0;
            std::size_t sourceCount = 0;
            splitRange(globalResults.size(), source, ranks, sourceFirst, sourceCount);
            for (std::size_t offset = 0; offset < sourceCount; offset += mpiChunk) {
                const std::size_t elements = std::min(mpiChunk, sourceCount - offset);
                MPI_Recv(globalResults.data() + sourceFirst + offset, static_cast<int>(elements),
                         MPI_DOUBLE, source, mpiTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (std::size_t offset = 0; offset < localCount; offset += mpiChunk) {
            const std::size_t elements = std::min(mpiChunk, localCount - offset);
            MPI_Send(localResults + offset, static_cast<int>(elements), MPI_DOUBLE, 0, mpiTag,
                     MPI_COMM_WORLD);
        }
    }
}

// Validation observes only the leading ten prices.  Collecting just that small
// prefix avoids a full device-to-host copy and gather when -v is used without
// -r, while still handling cases where the prefix spans several MPI ranks.
void gatherValidationPrefixToRoot(const double* localResults, const std::size_t localCount,
                                 const std::size_t totalCount, const std::size_t prefixCount,
                                 const std::size_t first, const int rank, const int ranks,
                                 std::vector<double>& prefixResults) {
    constexpr int mpiTag = 732;
    if (rank == 0) {
        if (localCount != 0) {
            std::copy_n(localResults, localCount, prefixResults.data() + first);
        }
        for (int source = 1; source < ranks; ++source) {
            std::size_t sourceFirst = 0;
            std::size_t sourceCount = 0;
            splitRange(totalCount, source, ranks, sourceFirst, sourceCount);
            const std::size_t sourcePrefixCount =
                sourceFirst < prefixCount ? std::min(sourceCount, prefixCount - sourceFirst) : 0;
            if (sourcePrefixCount != 0) {
                MPI_Recv(prefixResults.data() + sourceFirst, static_cast<int>(sourcePrefixCount),
                         MPI_DOUBLE, source, mpiTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else if (localCount != 0) {
        MPI_Send(localResults, static_cast<int>(localCount), MPI_DOUBLE, 0, mpiTag,
                 MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    std::size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    ParseResult parseResult = ParseResult::Run;
    if (rank == 0) {
        parseResult = parseArguments(argc, argv, numOptions, validate, printResults);
    }

    int parseCode = static_cast<int>(parseResult);
    MPI_Bcast(&parseCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseCode != static_cast<int>(ParseResult::Run)) {
        MPI_Finalize();
        return parseCode == static_cast<int>(ParseResult::Help) ? 0 : 1;
    }

    unsigned long long broadcastCount = static_cast<unsigned long long>(numOptions);
    int optionFlags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&broadcastCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(optionFlags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    numOptions = static_cast<std::size_t>(broadcastCount);
    validate = optionFlags[0] != 0;
    printResults = optionFlags[1] != 0;

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select one visible GPU per local MPI rank.  This is topology-aware across
    // nodes and remains safe when a node intentionally runs more ranks than GPUs.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is visible to the MPI ranks.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount), rank);
    MPI_Comm_free(&localComm);

    std::size_t first = 0;
    std::size_t localCount = 0;
    splitRange(numOptions, rank, ranks, first, localCount);

    // Input construction is distributed across CPU cores, first-touching the
    // rank-local buffer before it is transferred to the rank's GPU.
    DeviceOption* hostOptions = nullptr;
    if (localCount != 0) {
        CUDA_CHECK(cudaMallocHost(&hostOptions, localCount * sizeof(*hostOptions)), rank);
#pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(localCount); ++i) {
            hostOptions[static_cast<std::size_t>(i)] =
                makeDeviceOption(first + static_cast<std::size_t>(i));
        }
    }

    DeviceOption* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    double* hostResults = nullptr;
    cudaStream_t stream = nullptr;
    if (localCount != 0) {
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)), rank);
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)), rank);
        if (printResults) {
            CUDA_CHECK(cudaMallocHost(&hostResults, localCount * sizeof(*hostResults)), rank);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), rank);
    }

    const std::size_t validationCount = std::min<std::size_t>(10, numOptions);
    std::vector<double> validationResults;
    if (rank == 0 && validate && !printResults) {
        validationResults.resize(validationCount);
    }
    const std::size_t localValidationCount =
        validate && !printResults && first < validationCount
            ? std::min(localCount, validationCount - first)
            : 0;
    std::vector<double> localValidationResults(localValidationCount);

    if (rank == 0) {
        std::printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localCount != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceOptions, hostOptions, localCount * sizeof(*hostOptions),
                                   cudaMemcpyHostToDevice, stream), rank);
        constexpr int threadsPerBlock = 256;
        const std::size_t blocks = (localCount + threadsPerBlock - 1) / threadsPerBlock;
        if (blocks > std::numeric_limits<unsigned int>::max()) {
            if (rank == 0) {
                std::fprintf(stderr, "The local CUDA grid exceeds the supported one-dimensional size.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        blackScholesKernel<<<static_cast<unsigned int>(blocks), threadsPerBlock, 0, stream>>>(
            deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError(), rank);

        if (printResults) {
            CUDA_CHECK(cudaMemcpyAsync(hostResults, deviceResults, localCount * sizeof(*hostResults),
                                       cudaMemcpyDeviceToHost, stream), rank);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream), rank);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double optionsPerSecond = elapsed > 0.0 ? numOptions / elapsed : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Options per second: %.0f\n", optionsPerSecond);
    }

    std::vector<double> globalResults;
    if (printResults) {
        if (rank == 0) {
            globalResults.resize(numOptions);
        }
        gatherResultsToRoot(hostResults, localCount, first, rank, ranks, globalResults);
    }

    if (rank == 0 && printResults) {
        print_results(globalResults, "OptionPrices");
    }

    if (validate && !printResults) {
        // This happens after the timed GPU work: validation is a diagnostic
        // action, and only at most ten values cross the PCIe/network boundary.
        if (localValidationCount != 0) {
            CUDA_CHECK(cudaMemcpy(localValidationResults.data(), deviceResults,
                                  localValidationCount * sizeof(double), cudaMemcpyDeviceToHost),
                       rank);
        }
        gatherValidationPrefixToRoot(localValidationResults.data(), localValidationCount, numOptions,
                                     validationCount, first, rank, ranks, validationResults);
    }

    int exitCode = 0;
    if (rank == 0 && validate) {
        if (printResults) {
            validationResults.assign(globalResults.begin(), globalResults.begin() + validationCount);
        }
        std::vector<OptionInput> validationOptions(validationCount);
        for (std::size_t i = 0; i < validationCount; ++i) {
            validationOptions[i] = makeOption(i);
        }
        std::printf("Validating results...\n");
        if (validateResults(validationOptions, validationResults)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(stream), rank);
    }
    if (hostResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostResults), rank);
    }
    if (deviceResults != nullptr) {
        CUDA_CHECK(cudaFree(deviceResults), rank);
    }
    if (deviceOptions != nullptr) {
        CUDA_CHECK(cudaFree(deviceOptions), rank);
    }
    if (hostOptions != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostOptions), rank);
    }

    MPI_Finalize();
    return exitCode;
}
