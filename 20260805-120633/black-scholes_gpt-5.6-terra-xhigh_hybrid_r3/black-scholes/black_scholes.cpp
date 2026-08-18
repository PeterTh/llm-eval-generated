#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
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

// Standard test cases for validation and procedural generation on the host.
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

// Keeping the seven source options in constant memory lets the GPU recreate the
// original generated input from its global index without an H2D input transfer.
__device__ __constant__ OptionInput deviceTestOptions[7] = {
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
};

// Standard normal cumulative distribution function.
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Black-Scholes formula for European options.  This is shared by the CPU-side
// validation representation and the CUDA pricing kernel.
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

// A grid-stride loop keeps each MPI rank efficient for both small and very
// large local partitions while the global index retains the original sequence.
__global__ void priceOptionsKernel(double* results,
                                   const unsigned long long globalFirst,
                                   const unsigned long long count) {
    unsigned long long localIndex =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;

    for (; localIndex < count; localIndex += stride) {
        const unsigned long long globalIndex = globalFirst + localIndex;
        OptionInput option = deviceTestOptions[globalIndex % 7ULL];
        const double factor = 1.0 + 0.1 * (static_cast<double>(globalIndex) / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        results[localIndex] = blackScholes(option);
    }
}

[[noreturn]] void failCuda(const cudaError_t error, const char* operation, const int rank) {
    fprintf(stderr, "Rank %d: CUDA call %s failed: %s\n", rank, operation,
            cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(rank, call)                                                        \
    do {                                                                              \
        const cudaError_t cudaStatus = (call);                                        \
        if (cudaStatus != cudaSuccess) {                                              \
            failCuda(cudaStatus, #call, (rank));                                      \
        }                                                                             \
    } while (false)

struct Partition {
    size_t first;
    size_t count;
};

Partition partitionForRank(const size_t total, const int rank, const int rankCount) {
    const size_t ranks = static_cast<size_t>(rankCount);
    const size_t base = total / ranks;
    const size_t remainder = total % ranks;
    const size_t rankIndex = static_cast<size_t>(rank);
    return {rankIndex * base + std::min(rankIndex, remainder),
            base + (rankIndex < remainder ? 1U : 0U)};
}

// Generate an arbitrary contiguous range with the same formula as the serial
// program.  This is used for rank-zero validation and deliberately uses OpenMP
// for the host-side generation stage.
void generateOptions(std::vector<OptionInput>& options,
                     const size_t globalFirst,
                     const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

#pragma omp parallel for schedule(static)
    for (size_t localIndex = 0; localIndex < count; ++localIndex) {
        const size_t globalIndex = globalFirst + localIndex;
        OptionInput option = testOptions[globalIndex % testOptions.size()];
        const double factor =
            1.0 + 0.1 * (static_cast<double>(globalIndex) / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[localIndex] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min(options.size(), results.size());

    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Preserve the deliberately relaxed validation used by the baseline.
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) ||
            std::isinf(computed)) {
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

bool parseCount(const char* text, size_t& value) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

// Gather the complete distributed vector only for the externally-visible -r
// mode.  The large-count path avoids MPI_Gatherv's int count limitation.
std::vector<double> gatherAllResults(const std::vector<double>& localResults,
                                     const size_t total,
                                     const int rank,
                                     const int rankCount) {
    std::vector<double> allResults;

    if (total <= static_cast<size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            allResults.resize(total);
            counts.resize(rankCount);
            displacements.resize(rankCount);
            for (int source = 0; source < rankCount; ++source) {
                const Partition sourcePartition = partitionForRank(total, source, rankCount);
                counts[source] = static_cast<int>(sourcePartition.count);
                displacements[source] = static_cast<int>(sourcePartition.first);
            }
        }

        MPI_Gatherv(localResults.data(), static_cast<int>(localResults.size()), MPI_DOUBLE,
                    rank == 0 ? allResults.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        return allResults;
    }

    constexpr size_t maxMessageElements = static_cast<size_t>(INT_MAX);
    constexpr int gatherTag = 311;
    if (rank == 0) {
        allResults.resize(total);
        if (!localResults.empty()) {
            std::copy(localResults.begin(), localResults.end(), allResults.begin());
        }
        for (int source = 1; source < rankCount; ++source) {
            const Partition sourcePartition = partitionForRank(total, source, rankCount);
            size_t copied = 0;
            while (copied < sourcePartition.count) {
                const size_t chunk = std::min(maxMessageElements, sourcePartition.count - copied);
                MPI_Recv(allResults.data() + sourcePartition.first + copied,
                         static_cast<int>(chunk), MPI_DOUBLE, source, gatherTag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                copied += chunk;
            }
        }
    } else {
        size_t sent = 0;
        while (sent < localResults.size()) {
            const size_t chunk = std::min(maxMessageElements, localResults.size() - sent);
            MPI_Send(localResults.data() + sent, static_cast<int>(chunk), MPI_DOUBLE, 0,
                     gatherTag, MPI_COMM_WORLD);
            sent += chunk;
        }
    }
    return allResults;
}

// Validation examines only the first ten values, so exchange just that prefix
// when -r is not requested.  Ranks own contiguous global ranges, allowing the
// prefix to be placed directly at its original global offsets.
std::vector<double> gatherValidationPrefix(const std::vector<double>& localPrefix,
                                           const size_t globalFirst,
                                           const size_t prefixCount,
                                           const int rank,
                                           const int rankCount) {
    std::vector<double> prefix;
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<unsigned long long> offsets;
    const int localCount = static_cast<int>(localPrefix.size());
    const unsigned long long localOffset = static_cast<unsigned long long>(globalFirst);

    if (rank == 0) {
        prefix.resize(prefixCount);
        counts.resize(rankCount);
        displacements.resize(rankCount);
        offsets.resize(rankCount);
    }
    MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&localOffset, 1, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? offsets.data() : nullptr, 1, MPI_UNSIGNED_LONG_LONG, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        for (int source = 0; source < rankCount; ++source) {
            // A rank with no prefix data has a don't-care displacement.  A
            // non-empty prefix always starts at its global option index.
            displacements[source] =
                counts[source] == 0 ? 0 : static_cast<int>(offsets[source]);
        }
    }

    MPI_Gatherv(localPrefix.data(), localCount, MPI_DOUBLE,
                rank == 0 ? prefix.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return prefix;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseFailed = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseCount(argv[++i], numOptions)) {
                parseFailed = true;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseFailed = true;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || parseFailed) {
        if (rank == 0) {
            if (parseFailed) {
                printf("Invalid option count or argument.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseFailed ? 1 : 0;
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Keep the host representation used by validation generated with OpenMP.
    // The device kernel reconstructs the same values procedurally, avoiding a
    // bulk host-to-device copy in the performance-critical path.
    const size_t validationCount = std::min<size_t>(10, numOptions);
    std::vector<OptionInput> validationOptions;
    if (rank == 0) {
        generateOptions(validationOptions, 0, validationCount);
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(rank, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices are visible.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(rank, cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(rank, cudaFree(nullptr));  // initialize the selected accelerator before timing

    const Partition localPartition = partitionForRank(numOptions, rank, rankCount);
    if (localPartition.count >
        std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "Rank %d: local result allocation is too large.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    double* deviceResults = nullptr;
    if (localPartition.count > 0) {
        CUDA_CHECK(rank, cudaMalloc(&deviceResults, localPartition.count * sizeof(*deviceResults)));
    }

    int minimumGridSize = 0;
    int blockSize = 0;
    if (localPartition.count > 0) {
        CUDA_CHECK(rank, cudaOccupancyMaxPotentialBlockSize(
                             &minimumGridSize, &blockSize, priceOptionsKernel, 0, 0));
    }

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const auto start = std::chrono::steady_clock::now();
    if (localPartition.count > 0) {
        const unsigned long long blocksNeeded =
            (static_cast<unsigned long long>(localPartition.count) + blockSize - 1ULL) /
            static_cast<unsigned long long>(blockSize);
        const unsigned long long blocksToLaunch = std::min(
            blocksNeeded, static_cast<unsigned long long>(std::max(1, minimumGridSize)));
        priceOptionsKernel<<<static_cast<unsigned int>(blocksToLaunch), blockSize>>>(
            deviceResults, static_cast<unsigned long long>(localPartition.first),
            static_cast<unsigned long long>(localPartition.count));
        CUDA_CHECK(rank, cudaGetLastError());
        CUDA_CHECK(rank, cudaDeviceSynchronize());
    }
    const auto end = std::chrono::steady_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        const double optionsPerSecond = elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0;
        printf("Options per second: %.0f\n", optionsPerSecond);
    }

    const bool needAllResults = printResults;
    const size_t localValidationCount =
        validate && localPartition.first < validationCount
            ? std::min(localPartition.count, validationCount - localPartition.first)
            : 0;
    const size_t localCopyCount = needAllResults ? localPartition.count : localValidationCount;

    std::vector<double> localResults(localCopyCount);
    if (localCopyCount > 0) {
        CUDA_CHECK(rank, cudaMemcpy(localResults.data(), deviceResults,
                                    localCopyCount * sizeof(*deviceResults),
                                    cudaMemcpyDeviceToHost));
    }

    std::vector<double> results;
    if (printResults) {
        results = gatherAllResults(localResults, numOptions, rank, rankCount);
        if (rank == 0) {
            print_results(results, "OptionPrices");
        }
    } else if (validate) {
        results = gatherValidationPrefix(localResults, localPartition.first, validationCount, rank,
                                         rankCount);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0 && validate) {
        printf("Validating results...\n");
        const bool valid = validateResults(validationOptions, results);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (deviceResults != nullptr) {
        CUDA_CHECK(rank, cudaFree(deviceResults));
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
