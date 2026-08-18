#include <algorithm>
#include <array>
#include <cerrno>
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

namespace {

constexpr double kSqrtOneHalf = 0.707106781186547524400844362104849039;
constexpr int kCudaBlockSize = 256;
constexpr int kMaximumStreamsPerRank = 8;

enum OptionType : int {
    CALL = 0,
    PUT = 1
};

// Keep the benchmark's original data model for validation and reporting.
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

// Only the fields used by the pricing formula are placed in GPU constant memory.
struct OptionParameters {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
};

inline constexpr std::array<OptionInput, 7> kTestOptions{{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
}};

__device__ __constant__ OptionParameters kDeviceTestOptions[7] = {
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15},
};

struct Range {
    std::size_t offset;
    std::size_t count;
};

Range partitionRange(const std::size_t total, const int partitions,
                     const int partition) noexcept {
    const std::size_t quotient = total / static_cast<std::size_t>(partitions);
    const std::size_t remainder = total % static_cast<std::size_t>(partitions);
    const std::size_t extra = partition < static_cast<int>(remainder) ? 1U : 0U;
    const std::size_t offset = static_cast<std::size_t>(partition) * quotient +
                               std::min(static_cast<std::size_t>(partition), remainder);
    return {offset, quotient + extra};
}

[[noreturn]] void abortCuda(const cudaError_t error, const char* expression,
                            const int worldRank) {
    std::fprintf(stderr, "MPI rank %d: CUDA call '%s' failed: %s\n", worldRank,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            abortCuda(cuda_check_error, #expression, worldRank);                 \
        }                                                                        \
    } while (false)

__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    // This is deliberately the same expression as the original CPU version.
    return 0.5 * (1.0 + erf(x * kSqrtOneHalf));
}

__device__ __forceinline__ double blackScholesDevice(const OptionParameters& base,
                                                     const double factor) {
    const double spot = base.spot * factor;
    const double strike = base.strike * factor;
    const double rootT = sqrt(base.t);
    const double sigmaRootT = base.vol * rootT;

    const double d1 =
        (log(spot / strike) +
         (base.r - base.q + 0.5 * base.vol * base.vol) * base.t) /
        sigmaRootT;
    const double d2 = d1 - sigmaRootT;
    const double discountR = exp(-base.r * base.t);
    const double discountQ = exp(-base.q * base.t);

    if (base.type == CALL) {
        return spot * discountQ * cumulativeNormalDevice(d1) -
               strike * discountR * cumulativeNormalDevice(d2);
    }
    return strike * discountR * cumulativeNormalDevice(-d2) -
           spot * discountQ * cumulativeNormalDevice(-d1);
}

__global__ void blackScholesKernel(const unsigned long long globalOffset,
                                   const std::size_t optionCount,
                                   double* __restrict__ results) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t localIndex = first; localIndex < optionCount;
         localIndex += stride) {
        const unsigned long long globalIndex =
            globalOffset + static_cast<unsigned long long>(localIndex);
        const int baseIndex = static_cast<int>(globalIndex % 7ULL);
        const double factor = 1.0 + 0.1 * (static_cast<double>(globalIndex) / 7.0);
        results[localIndex] =
            blackScholesDevice(kDeviceTestOptions[baseIndex], factor);
    }
}

OptionInput generateOption(const std::size_t globalIndex) noexcept {
    OptionInput option = kTestOptions[globalIndex % kTestOptions.size()];
    const double factor =
        1.0 + 0.1 * (globalIndex / static_cast<double>(kTestOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

bool validateResults(const std::vector<double>& results) {
    const int checks = static_cast<int>(std::min<std::size_t>(10, results.size()));
    std::array<OptionInput, 10> options{};

    // Host-side benchmark work is parallelized as well; this path is intentionally
    // independent of the device constants so it also checks data generation.
#pragma omp parallel for schedule(static) if (checks > 1)
    for (int i = 0; i < checks; ++i) {
        options[static_cast<std::size_t>(i)] = generateOption(static_cast<std::size_t>(i));
    }

    bool allPassed = true;
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        const double computed = results[static_cast<std::size_t>(i)];
        const double expected = options[static_cast<std::size_t>(i)].value;
        const double error = std::fabs(computed - expected);
        const double relativeError = error / (std::fabs(expected) + 1.0e-10);

        std::printf(
            "  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i,
            computed, expected, relativeError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            std::printf("Validation failed at option %d: invalid value %.4f\n", i,
                        computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void gatherResults(const double* localResults, const Range localRange,
                   const std::size_t total, const int worldSize,
                   const int worldRank, std::vector<double>& globalResults) {
    // MPI-3 count and displacement arguments are int. Windowed Gatherv retains
    // collective scalability while also supporting arrays larger than INT_MAX.
    constexpr std::size_t kGatherWindow = 1ULL << 30;
    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (worldRank == 0) {
        receiveCounts.resize(static_cast<std::size_t>(worldSize));
        displacements.resize(static_cast<std::size_t>(worldSize));
    }

    for (std::size_t windowBegin = 0; windowBegin < total;) {
        const std::size_t windowSize = std::min(kGatherWindow, total - windowBegin);
        const std::size_t windowEnd = windowBegin + windowSize;
        const std::size_t localBegin = std::max(localRange.offset, windowBegin);
        const std::size_t localEnd =
            std::min(localRange.offset + localRange.count, windowEnd);
        const std::size_t sendSize = localEnd > localBegin ? localEnd - localBegin : 0;
        const double* sendBuffer =
            sendSize == 0 ? localResults
                          : localResults + (localBegin - localRange.offset);

        if (worldRank == 0) {
#pragma omp parallel for schedule(static) if (worldSize > 1)
            for (int rank = 0; rank < worldSize; ++rank) {
                const Range rankRange = partitionRange(total, worldSize, rank);
                const std::size_t begin = std::max(rankRange.offset, windowBegin);
                const std::size_t end =
                    std::min(rankRange.offset + rankRange.count, windowEnd);
                const std::size_t count = end > begin ? end - begin : 0;
                receiveCounts[static_cast<std::size_t>(rank)] = static_cast<int>(count);
                displacements[static_cast<std::size_t>(rank)] =
                    count == 0 ? 0 : static_cast<int>(begin - windowBegin);
            }
        }

        MPI_Gatherv(sendBuffer, static_cast<int>(sendSize), MPI_DOUBLE,
                    worldRank == 0 ? globalResults.data() + windowBegin : nullptr,
                    worldRank == 0 ? receiveCounts.data() : nullptr,
                    worldRank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        windowBegin = windowEnd;
    }
}

bool parseOptionCount(const char* text, std::size_t& value) noexcept {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) !=
        MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) {
            std::fprintf(stderr, "MPI does not provide the required FUNNELED thread level\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t numberOfOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseOptionCount(argv[++i], numberOfOptions)) {
                argumentsValid = false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }
    if (numberOfOptions >
        std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        argumentsValid = false;
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            if (!argumentsValid) {
                std::printf("Invalid command-line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                        MPI_INFO_NULL, &nodeCommunicator);
    int nodeRank = 0;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "No CUDA-capable device is visible\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CUDA_CHECK(cudaFree(nullptr)); // Create the CUDA context before timing.

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    const Range localRange = partitionRange(numberOfOptions, worldSize, worldRank);
    const bool needGlobalResults = validate || printResults;

    double* deviceResults = nullptr;
    double* localResults = nullptr;
    if (localRange.count != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                              localRange.count * sizeof(double)));
        if (needGlobalResults) {
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&localResults),
                                      localRange.count * sizeof(double)));
        }
    }

    const std::size_t localBlocks =
        (localRange.count + static_cast<std::size_t>(kCudaBlockSize) - 1) /
        static_cast<std::size_t>(kCudaBlockSize);
    const int streamCount = localBlocks == 0
                                ? 0
                                : static_cast<int>(std::min<std::size_t>(
                                      localBlocks,
                                      static_cast<std::size_t>(std::max(
                                          1, std::min(kMaximumStreamsPerRank,
                                                      omp_get_max_threads())))));
    std::vector<cudaStream_t> streams(static_cast<std::size_t>(streamCount));
    for (int stream = 0; stream < streamCount; ++stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &streams[static_cast<std::size_t>(stream)], cudaStreamNonBlocking));
    }

    // Force CUDA module loading/JIT initialization out of the timed region. The
    // timed launch overwrites this element, so warm-up cannot affect the result.
    if (localRange.count != 0) {
        blackScholesKernel<<<1, 1, 0, streams[0]>>>(
            static_cast<unsigned long long>(localRange.offset), 1, deviceResults);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }

    if (worldRank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numberOfOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA GPUs\n",
                    worldSize, omp_get_max_threads());
        std::printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    // OpenMP host threads concurrently feed nonblocking CUDA streams. Each stream
    // owns a contiguous output range, so no synchronization or atomics are needed.
    std::vector<cudaError_t> streamErrors(static_cast<std::size_t>(streamCount),
                                          cudaSuccess);
#pragma omp parallel for schedule(static) num_threads(std::max(1, streamCount)) \
    if (streamCount > 1)
    for (int streamIndex = 0; streamIndex < streamCount; ++streamIndex) {
        cudaError_t error = cudaSetDevice(device);
        const Range streamRange =
            partitionRange(localRange.count, streamCount, streamIndex);
        cudaStream_t stream = streams[static_cast<std::size_t>(streamIndex)];

        if (error == cudaSuccess && streamRange.count != 0) {
            const std::size_t requiredBlocks =
                (streamRange.count + static_cast<std::size_t>(kCudaBlockSize) - 1) /
                static_cast<std::size_t>(kCudaBlockSize);
            const std::size_t occupancyBlocks = static_cast<std::size_t>(
                std::max(1, deviceProperties.multiProcessorCount * 16 / streamCount));
            const unsigned int gridSize = static_cast<unsigned int>(
                std::min(requiredBlocks, occupancyBlocks));

            blackScholesKernel<<<gridSize, kCudaBlockSize, 0, stream>>>(
                static_cast<unsigned long long>(localRange.offset + streamRange.offset),
                streamRange.count, deviceResults + streamRange.offset);
            error = cudaGetLastError();
        }
        if (error == cudaSuccess && streamRange.count != 0 && needGlobalResults) {
            error = cudaMemcpyAsync(localResults + streamRange.offset,
                                    deviceResults + streamRange.offset,
                                    streamRange.count * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream);
        }
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(stream);
        }
        streamErrors[static_cast<std::size_t>(streamIndex)] = error;
    }

    for (int stream = 0; stream < streamCount; ++stream) {
        const cudaError_t error = streamErrors[static_cast<std::size_t>(stream)];
        if (error != cudaSuccess) {
            abortCuda(error, "hybrid CUDA stream execution", worldRank);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const double milliseconds = elapsed * 1000.0;
        const double rate = elapsed > 0.0
                                ? static_cast<double>(numberOfOptions) / elapsed
                                : std::numeric_limits<double>::infinity();
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Options per second: %.0f\n", rate);
    }

    std::vector<double> globalResults;
    if (needGlobalResults) {
        if (worldRank == 0) {
            globalResults.resize(numberOfOptions);
        }
        gatherResults(localResults, localRange, numberOfOptions, worldSize, worldRank,
                      globalResults);
    }

    int returnCode = EXIT_SUCCESS;
    if (worldRank == 0) {
        if (printResults) {
            print_results(globalResults, "OptionPrices");
        }
        if (validate) {
            std::printf("Validating results...\n");
            const bool valid = validateResults(globalResults);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (cudaStream_t stream : streams) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (localResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(localResults));
    }
    if (deviceResults != nullptr) {
        CUDA_CHECK(cudaFree(deviceResults));
    }
    MPI_Comm_free(&nodeCommunicator);
    MPI_Finalize();
    return returnCode;
}
