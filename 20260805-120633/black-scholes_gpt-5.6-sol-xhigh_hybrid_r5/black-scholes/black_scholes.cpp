#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double kInvSqrtTwo = 0.707106781186547524400844362104849039;
constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kPipelineThreshold = 1U << 20;
constexpr int kMaximumStreams = 4;

enum OptionType : int {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    double value;
    double tol;
};

// Only values consumed by the pricing kernel cross the PCIe/NVLink boundary.
struct alignas(8) OptionParameters {
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    int type;
};

static_assert(sizeof(OptionParameters) == 56,
              "Unexpected padding would waste accelerator bandwidth");

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

struct WorkRange {
    std::size_t begin;
    std::size_t count;
};

WorkRange rangeForRank(const std::size_t total, const int rank,
                       const int ranks) noexcept {
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t base = total / rankCount;
    const std::size_t remainder = total % rankCount;
    return {rankIndex * base + std::min(rankIndex, remainder),
            base + (rankIndex < remainder ? 1U : 0U)};
}

[[noreturn]] void abortAll(const char* operation, const char* detail,
                           const int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation, detail);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        abortAll(operation, cudaGetErrorString(status), rank);
    }
}

#define CUDA_CHECK(call, rank) checkCuda((call), #call, (rank))

// Input construction remains outside the measured pricing interval, as in the
// serial benchmark. OpenMP shares this host-side work among the cores assigned
// to each MPI rank.
void generateOptions(OptionParameters* const options,
                     const std::size_t localCount,
                     const std::size_t globalBegin) {
    constexpr auto testOptions = getTestOptions();

#pragma omp parallel for schedule(static)
    for (std::size_t localIndex = 0; localIndex < localCount; ++localIndex) {
        const std::size_t globalIndex = globalBegin + localIndex;
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        const double factor =
            1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[localIndex] = {base.strike * factor,
                               base.spot * factor,
                               base.q,
                               base.r,
                               base.t,
                               base.vol,
                               base.type};
    }
}

__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * kInvSqrtTwo));
}

__device__ __forceinline__ double blackScholes(
    const OptionParameters& option) noexcept {
    if (option.t <= 0.0 || option.vol <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(option.t);
    const double sigmaSqrtT = option.vol * sqrtT;
    const double d1 =
        (log(option.spot / option.strike) +
         (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
        sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    const double discountedStrike = option.strike * exp(-option.r * option.t);
    const double discountedSpot = option.spot * exp(-option.q * option.t);

    if (option.type == CALL) {
        return discountedSpot * cumulativeNormal(d1) -
               discountedStrike * cumulativeNormal(d2);
    }
    return discountedStrike * cumulativeNormal(-d2) -
           discountedSpot * cumulativeNormal(-d1);
}

__global__ void priceOptionsKernel(
    const OptionParameters* __restrict__ options,
    double* __restrict__ results, const std::size_t count) {
    std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                        threadIdx.x;
    const std::size_t stride =
        gridDim.x * static_cast<std::size_t>(blockDim.x);
    for (; index < count; index += stride) {
        results[index] = blackScholes(options[index]);
    }
}

void warmUpAccelerator(const int rank) {
    // Force CUDA context and lazy kernel-module initialization before timing.
    // A zero-length launch executes no pricing work and dereferences no data.
    priceOptionsKernel<<<1, 1>>>(nullptr, nullptr, 0);
    CUDA_CHECK(cudaPeekAtLastError(), rank);
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
}

struct AcceleratorStorage {
    OptionParameters* hostOptions = nullptr;
    double* hostResults = nullptr;
    OptionParameters* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    std::vector<cudaStream_t> streams;
};

void allocateStorage(AcceleratorStorage& storage, const std::size_t count,
                     const int streamCount, const int rank) {
    if (count == 0) {
        return;
    }
    if (count > std::numeric_limits<std::size_t>::max() /
                    sizeof(OptionParameters) ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        abortAll("buffer size calculation", "requested size overflows size_t",
                 rank);
    }

    const std::size_t optionBytes = count * sizeof(OptionParameters);
    const std::size_t resultBytes = count * sizeof(double);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&storage.hostOptions),
                              optionBytes),
               rank);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&storage.hostResults),
                              resultBytes),
               rank);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage.deviceOptions),
                          optionBytes),
               rank);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&storage.deviceResults),
                          resultBytes),
               rank);

    storage.streams.resize(static_cast<std::size_t>(streamCount));
    for (cudaStream_t& stream : storage.streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), rank);
    }
}

void releaseStorage(AcceleratorStorage& storage) noexcept {
    for (const cudaStream_t stream : storage.streams) {
        cudaStreamDestroy(stream);
    }
    if (storage.deviceResults != nullptr) {
        cudaFree(storage.deviceResults);
    }
    if (storage.deviceOptions != nullptr) {
        cudaFree(storage.deviceOptions);
    }
    if (storage.hostResults != nullptr) {
        cudaFreeHost(storage.hostResults);
    }
    if (storage.hostOptions != nullptr) {
        cudaFreeHost(storage.hostOptions);
    }
}

void priceOptions(AcceleratorStorage& storage, const std::size_t count,
                  const int multiprocessors, const int rank) {
    if (count == 0) {
        return;
    }

    const std::size_t streamCount = storage.streams.size();
    const std::size_t itemsPerStream = (count + streamCount - 1U) / streamCount;
    const std::size_t maximumBlocks =
        static_cast<std::size_t>(multiprocessors) * 32U;

    for (std::size_t streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
        const std::size_t offset = streamIndex * itemsPerStream;
        if (offset >= count) {
            break;
        }
        const std::size_t chunk = std::min(itemsPerStream, count - offset);
        cudaStream_t stream = storage.streams[streamIndex];

        CUDA_CHECK(cudaMemcpyAsync(storage.deviceOptions + offset,
                                   storage.hostOptions + offset,
                                   chunk * sizeof(OptionParameters),
                                   cudaMemcpyHostToDevice, stream),
                   rank);
        const std::size_t requiredBlocks =
            (chunk + kThreadsPerBlock - 1U) / kThreadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min(requiredBlocks, maximumBlocks));
        priceOptionsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            storage.deviceOptions + offset, storage.deviceResults + offset,
            chunk);
        CUDA_CHECK(cudaPeekAtLastError(), rank);
        CUDA_CHECK(cudaMemcpyAsync(storage.hostResults + offset,
                                   storage.deviceResults + offset,
                                   chunk * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream),
                   rank);
    }

    for (const cudaStream_t stream : storage.streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream), rank);
    }
}

// Gather is deliberately outside the timed interval and is skipped for normal
// benchmark runs. MPI_Gatherv is fastest for ordinary sizes; the chunked path
// preserves correctness beyond MPI's legacy signed-int count limit.
std::vector<double> gatherAllResults(const double* localResults,
                                     const WorkRange localRange,
                                     const std::size_t total, const int rank,
                                     const int ranks) {
    std::vector<double> globalResults;
    if (rank == 0) {
        globalResults.resize(total);
    }

    static double ignored = 0.0;
    const double* const sendBuffer =
        localRange.count == 0 ? &ignored : localResults;
    if (total <= static_cast<std::size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<std::size_t>(ranks));
            displacements.resize(static_cast<std::size_t>(ranks));
            for (int source = 0; source < ranks; ++source) {
                const WorkRange sourceRange = rangeForRank(total, source, ranks);
                counts[static_cast<std::size_t>(source)] =
                    static_cast<int>(sourceRange.count);
                displacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(sourceRange.begin);
            }
        }
        MPI_Gatherv(sendBuffer, static_cast<int>(localRange.count), MPI_DOUBLE,
                    rank == 0 ? globalResults.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        return globalResults;
    }

    constexpr int kGatherTag = 27182;
    const std::size_t maximumChunk = static_cast<std::size_t>(INT_MAX);
    if (rank == 0) {
        std::copy_n(localResults, localRange.count,
                    globalResults.data() + localRange.begin);
        for (int source = 1; source < ranks; ++source) {
            const WorkRange sourceRange = rangeForRank(total, source, ranks);
            std::size_t received = 0;
            while (received < sourceRange.count) {
                const int chunk = static_cast<int>(
                    std::min(maximumChunk, sourceRange.count - received));
                MPI_Recv(globalResults.data() + sourceRange.begin + received,
                         chunk, MPI_DOUBLE, source, kGatherTag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                received += static_cast<std::size_t>(chunk);
            }
        }
    } else {
        std::size_t sent = 0;
        while (sent < localRange.count) {
            const int chunk = static_cast<int>(
                std::min(maximumChunk, localRange.count - sent));
            MPI_Send(localResults + sent, chunk, MPI_DOUBLE, 0, kGatherTag,
                     MPI_COMM_WORLD);
            sent += static_cast<std::size_t>(chunk);
        }
    }
    return globalResults;
}

std::vector<double> gatherValidationPrefix(const double* localResults,
                                           const WorkRange localRange,
                                           const std::size_t total,
                                           const int rank, const int ranks) {
    const std::size_t prefixCount = std::min<std::size_t>(10, total);
    const std::size_t localPrefixCount =
        localRange.begin < prefixCount
            ? std::min(localRange.count, prefixCount - localRange.begin)
            : 0;
    static double ignored = 0.0;
    const double* const sendBuffer =
        localPrefixCount == 0 ? &ignored : localResults;

    std::vector<double> prefix;
    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        prefix.resize(prefixCount);
        counts.resize(static_cast<std::size_t>(ranks));
        displacements.resize(static_cast<std::size_t>(ranks));
        for (int source = 0; source < ranks; ++source) {
            const WorkRange sourceRange = rangeForRank(total, source, ranks);
            const std::size_t sourceCount =
                sourceRange.begin < prefixCount
                    ? std::min(sourceRange.count,
                               prefixCount - sourceRange.begin)
                    : 0;
            counts[static_cast<std::size_t>(source)] =
                static_cast<int>(sourceCount);
            displacements[static_cast<std::size_t>(source)] =
                static_cast<int>(std::min(sourceRange.begin, prefixCount));
        }
    }
    MPI_Gatherv(sendBuffer, static_cast<int>(localPrefixCount), MPI_DOUBLE,
                rank == 0 ? prefix.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
    return prefix;
}

bool validateResults(const std::vector<double>& results) {
    constexpr auto testOptions = getTestOptions();
    bool allPassed = true;

    std::printf("Checking computed option prices:\n");
    for (std::size_t i = 0; i < results.size(); ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % testOptions.size()].value;
        const double error = std::fabs(computed - expected);
        const double relativeError = error / (std::fabs(expected) + 1.0e-10);
        std::printf(
            "  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
            i, computed, expected, relativeError);

        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) ||
            std::isinf(computed)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n",
                        i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

enum class ParseStatus { Success, Help, Error };

ParseStatus parseArguments(const int argc, char** const argv,
                           std::size_t& numOptions, bool& validate,
                           bool& printResults, const int rank) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* const value = argv[++i];
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno == ERANGE || end == value || *end != '\0' ||
                value[0] == '-') {
                if (rank == 0) {
                    std::printf("Invalid number of options: %s\n", value);
                    printUsage(argv[0]);
                }
                return ParseStatus::Error;
            }
            if (parsed > std::numeric_limits<std::size_t>::max()) {
                if (rank == 0) {
                    std::printf("Number of options is too large: %s\n", value);
                }
                return ParseStatus::Error;
            }
            numOptions = static_cast<std::size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return ParseStatus::Help;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return ParseStatus::Error;
        }
    }
    return ParseStatus::Success;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                        &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortAll("MPI_Init_thread", "MPI_THREAD_FUNNELED is unavailable", rank);
    }

    std::size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    const ParseStatus parseStatus =
        parseArguments(argc, argv, numOptions, validate, printResults, rank);
    if (parseStatus != ParseStatus::Success) {
        MPI_Finalize();
        return parseStatus == ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    int localRanks = 1;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_size(localCommunicator, &localRanks);

    // Unless the launch explicitly sets OMP_NUM_THREADS, divide each node's
    // CPU cores among its resident MPI ranks to prevent oversubscription.
    omp_set_dynamic(0);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localRanks));
    }
    const int openmpThreads = omp_get_max_threads();

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        abortAll("CUDA device selection", "no CUDA-capable device is visible",
                 rank);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device), rank);

    const WorkRange localRange = rangeForRank(numOptions, rank, ranks);
    const std::size_t requestedStreams =
        localRange.count >= kPipelineThreshold
            ? (localRange.count + kPipelineThreshold - 1U) /
                  kPipelineThreshold
            : 1U;
    const int streamCount = static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(kMaximumStreams), requestedStreams));

    AcceleratorStorage storage;
    allocateStorage(storage, localRange.count, streamCount, rank);
    generateOptions(storage.hostOptions, localRange.count, localRange.begin);
    warmUpAccelerator(rank);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), %d OpenMP "
                    "thread(s)/rank, CUDA GPU(s)\n",
                    ranks, openmpThreads);
        std::printf("Pricing options...\n");
        std::fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    priceOptions(storage, localRange.count,
                 deviceProperties.multiProcessorCount, rank);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double throughput =
            duration > 0.0 ? static_cast<double>(numOptions) / duration : 0.0;
        std::printf("Options per second: %.0f\n", throughput);
    }

    std::vector<double> gatheredResults;
    if (printResults) {
        gatheredResults = gatherAllResults(storage.hostResults, localRange,
                                           numOptions, rank, ranks);
        if (rank == 0) {
            print_results(gatheredResults, "OptionPrices");
        }
    } else if (validate) {
        gatheredResults = gatherValidationPrefix(
            storage.hostResults, localRange, numOptions, rank, ranks);
    }

    int returnCode = EXIT_SUCCESS;
    if (validate && rank == 0) {
        std::printf("Validating results...\n");
        const std::size_t checks = std::min<std::size_t>(10, numOptions);
        if (printResults && gatheredResults.size() > checks) {
            gatheredResults.resize(checks);
        }
        const bool valid = validateResults(gatheredResults);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        returnCode = valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    releaseStorage(storage);
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return returnCode;
}
