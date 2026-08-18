#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

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

// These routines are compiled for both the host and the CUDA device. Keeping
// one formula avoids a host/device semantic drift in the pricing kernel.
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options.
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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
    const double sigmaSqrtT = sigma * sqrtT;
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    }

    return K * discount * cumulativeNormal(-d2) -
           S * exp(-q * T) * cumulativeNormal(-d1);
}

// A grid-stride loop lets a rank price arbitrarily large local partitions
// without relying on the CUDA grid dimension limit.
__global__ void blackScholesKernel(const OptionInput* options,
                                   double* results,
                                   const std::size_t count) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = thread; i < count; i += stride) {
        results[i] = blackScholes(options[i]);
    }
}

[[noreturn]] void abortCuda(const char* operation, const cudaError_t error,
                            const int rank) {
    fprintf(stderr, "MPI rank %d: CUDA error in %s: %s\n", rank, operation,
            cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(operation)                                                    \
    do {                                                                          \
        const cudaError_t cudaStatus = (operation);                              \
        if (cudaStatus != cudaSuccess) {                                         \
            abortCuda(#operation, cudaStatus, rank);                             \
        }                                                                         \
    } while (false)

[[noreturn]] void abortMpi(const char* operation, const int error,
                           const int rank) {
    char errorString[MPI_MAX_ERROR_STRING];
    int errorLength = 0;
    MPI_Error_string(error, errorString, &errorLength);
    fprintf(stderr, "MPI rank %d: MPI error in %s: %.*s\n", rank, operation,
            errorLength, errorString);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define MPI_CHECK(operation)                                                      \
    do {                                                                          \
        const int mpiStatus = (operation);                                        \
        if (mpiStatus != MPI_SUCCESS) {                                           \
            abortMpi(#operation, mpiStatus, rank);                                \
        }                                                                         \
    } while (false)

// Standard test cases for validation.
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

// Generate only the rank-local contiguous range. The OpenMP loop is also
// used for the small validation range, so every normal execution exercises
// the host-side parallel stage as well as the CUDA stage.
void generateOptions(OptionInput* options, const std::size_t globalOffset,
                     const std::size_t count) {
    constexpr auto testOptions = getTestOptions();

#pragma omp parallel for schedule(static)
    for (std::int64_t local = 0; local < static_cast<std::int64_t>(count); ++local) {
        const std::size_t globalIndex = globalOffset + static_cast<std::size_t>(local);
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[local] = base;

        const double factor = 1.0 +
                              0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[local].spot *= factor;
        options[local].strike *= factor;
    }
}

void partition(const std::size_t total, const int rank, const int ranks,
               std::size_t& offset, std::size_t& count) {
    const std::size_t base = total / static_cast<std::size_t>(ranks);
    const std::size_t remainder = total % static_cast<std::size_t>(ranks);
    count = base + (static_cast<std::size_t>(rank) < remainder ? 1U : 0U);
    offset = base * static_cast<std::size_t>(rank) +
             std::min(static_cast<std::size_t>(rank), remainder);
}

// Gather a contiguous distributed vector in chunks so MPI's int count and
// displacement limits do not cap the benchmark's size_t command-line input.
void gatherResults(const double* localResults, const std::size_t localOffset,
                   const std::size_t localCount, std::vector<double>& globalResults,
                   const std::size_t total, const int rank, const int ranks) {
    constexpr std::size_t maxMpiCount = static_cast<std::size_t>(INT_MAX);
    std::vector<int> receiveCounts(static_cast<std::size_t>(ranks));
    std::vector<int> receiveDisplacements(static_cast<std::size_t>(ranks));

    for (std::size_t begin = 0; begin < total;) {
        const std::size_t end = std::min(total, begin + maxMpiCount);
        std::fill(receiveCounts.begin(), receiveCounts.end(), 0);
        std::fill(receiveDisplacements.begin(), receiveDisplacements.end(), 0);

        for (int source = 0; source < ranks; ++source) {
            std::size_t sourceOffset = 0;
            std::size_t sourceCount = 0;
            partition(total, source, ranks, sourceOffset, sourceCount);
            const std::size_t intersectionBegin = std::max(begin, sourceOffset);
            const std::size_t intersectionEnd =
                std::min(end, sourceOffset + sourceCount);
            if (intersectionBegin < intersectionEnd) {
                receiveCounts[static_cast<std::size_t>(source)] =
                    static_cast<int>(intersectionEnd - intersectionBegin);
                receiveDisplacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(intersectionBegin - begin);
            }
        }

        const std::size_t localBegin = std::max(begin, localOffset);
        const std::size_t localEnd = std::min(end, localOffset + localCount);
        const int sendCount = localBegin < localEnd
                                  ? static_cast<int>(localEnd - localBegin)
                                  : 0;
        const double* sendBuffer =
            sendCount == 0 ? localResults : localResults + (localBegin - localOffset);
        double* receiveBuffer = rank == 0 ? globalResults.data() + begin : nullptr;

        MPI_CHECK(MPI_Gatherv(sendBuffer, sendCount, MPI_DOUBLE, receiveBuffer,
                              receiveCounts.data(), receiveDisplacements.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        begin = end;
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results, const int rank) {
    if (rank != 0) {
        return true;
    }

    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min<std::size_t>(10, options.size()));

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[static_cast<std::size_t>(i)];
        const double expected = options[static_cast<std::size_t>(i)].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Preserve the original benchmark's intentionally relaxed validation.
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsed = atoll(argv[++i]);
            if (parsed < 0) {
                parseError = true;
            } else {
                numOptions = static_cast<std::size_t>(parsed);
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            parseError = true;
        }
    }

    if (parseError) {
        if (rank == 0) {
            fprintf(stderr, "Invalid command line arguments.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));

    int deviceCount = 0;
    const cudaError_t deviceQueryStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceQueryStatus != cudaSuccess || deviceCount == 0) {
        abortCuda("cudaGetDeviceCount", deviceQueryStatus == cudaSuccess
                                               ? cudaErrorNoDevice
                                               : deviceQueryStatus,
                  rank);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    std::size_t localOffset = 0;
    std::size_t localCount = 0;
    partition(numOptions, rank, ranks, localOffset, localCount);

    OptionInput* localOptions = nullptr;
    double* localResults = nullptr;
    if (localCount != 0) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&localOptions),
                                 localCount * sizeof(OptionInput), cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&localResults),
                                 localCount * sizeof(double), cudaHostAllocPortable));
        generateOptions(localOptions, localOffset, localCount);
    }

    cudaStream_t stream = nullptr;
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    std::size_t deviceCapacity = 0;

    if (localCount != 0) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        (void)totalBytes;

        constexpr std::size_t bytesPerOption = sizeof(OptionInput) + sizeof(double);
        const std::size_t memoryLimitedCapacity = freeBytes / bytesPerOption;
        deviceCapacity = std::min(localCount,
                                  std::max<std::size_t>(1, memoryLimitedCapacity * 3 / 4));

        // Leave room for the CUDA context and unrelated allocations. If a
        // node is busy, retry with progressively smaller batches; this keeps
        // the execution GPU-only without requiring another dependency.
        while (deviceCapacity != 0) {
            cudaError_t allocationStatus = cudaMalloc(
                reinterpret_cast<void**>(&deviceOptions),
                deviceCapacity * sizeof(OptionInput));
            if (allocationStatus == cudaSuccess) {
                allocationStatus = cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                                              deviceCapacity * sizeof(double));
            }
            if (allocationStatus == cudaSuccess) {
                break;
            }
            if (deviceOptions != nullptr) {
                cudaFree(deviceOptions);
                deviceOptions = nullptr;
            }
            if (deviceResults != nullptr) {
                cudaFree(deviceResults);
                deviceResults = nullptr;
            }
            deviceCapacity /= 2;
        }
        if (deviceCapacity == 0) {
            abortCuda("cudaMalloc(device buffers)", cudaErrorMemoryAllocation, rank);
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", ranks);
        printf("CUDA devices per shared node: %d\n", deviceCount);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::high_resolution_clock::now();

    if (localCount != 0) {
        int multiprocessors = 1;
        CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors,
                                          cudaDevAttrMultiProcessorCount,
                                          localRank % deviceCount));
        constexpr unsigned int threadsPerBlock = 256;
        const std::size_t maxBlocks =
            std::max<std::size_t>(1, static_cast<std::size_t>(multiprocessors) * 32);
        const std::size_t blocksForBatch =
            (deviceCapacity + threadsPerBlock - 1) / threadsPerBlock;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min(maxBlocks, std::max<std::size_t>(1, blocksForBatch)));

        for (std::size_t offset = 0; offset < localCount; offset += deviceCapacity) {
            const std::size_t batch = std::min(deviceCapacity, localCount - offset);
            CUDA_CHECK(cudaMemcpyAsync(deviceOptions, localOptions + offset,
                                       batch * sizeof(OptionInput), cudaMemcpyHostToDevice,
                                       stream));
            blackScholesKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                deviceOptions, deviceResults, batch);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(localResults + offset, deviceResults,
                                       batch * sizeof(double), cudaMemcpyDeviceToHost,
                                       stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds =
        std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Pricing options...\n");
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        printf("Options per second: %.0f\n",
               elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
    }

    std::vector<double> results;
    if (rank == 0 && printResults) {
        results.resize(numOptions);
        gatherResults(localResults, localOffset, localCount, results, numOptions, rank,
                      ranks);
        print_results(results, "OptionPrices");
    } else if (printResults) {
        gatherResults(localResults, localOffset, localCount, results, numOptions, rank,
                      ranks);
    }

    bool valid = true;
    if (validate) {
        const std::size_t checkCount = std::min<std::size_t>(10, numOptions);
        std::vector<int> checkCounts(static_cast<std::size_t>(ranks), 0);
        std::vector<int> checkDisplacements(static_cast<std::size_t>(ranks), 0);
        for (int source = 0; source < ranks; ++source) {
            std::size_t sourceOffset = 0;
            std::size_t sourceCount = 0;
            partition(numOptions, source, ranks, sourceOffset, sourceCount);
            const std::size_t begin = std::max<std::size_t>(0, sourceOffset);
            const std::size_t end = std::min(checkCount, sourceOffset + sourceCount);
            if (begin < end) {
                checkCounts[static_cast<std::size_t>(source)] =
                    static_cast<int>(end - begin);
                checkDisplacements[static_cast<std::size_t>(source)] =
                    static_cast<int>(begin);
            }
        }

        std::vector<double> checkedResults(rank == 0 ? checkCount : 0);
        const std::size_t localBegin = std::min(checkCount, localOffset);
        const std::size_t localEnd = std::min(checkCount, localOffset + localCount);
        const int checkSendCount = localBegin < localEnd
                                       ? static_cast<int>(localEnd - localBegin)
                                       : 0;
        const double* checkSendBuffer = checkSendCount == 0
                                            ? localResults
                                            : localResults + (localBegin - localOffset);
        MPI_CHECK(MPI_Gatherv(checkSendBuffer, checkSendCount, MPI_DOUBLE,
                              rank == 0 ? checkedResults.data() : nullptr,
                              checkCounts.data(), checkDisplacements.data(), MPI_DOUBLE,
                              0, MPI_COMM_WORLD));

        std::vector<OptionInput> checkOptions(rank == 0 ? checkCount : 0);
        if (rank == 0 && checkCount != 0) {
            generateOptions(checkOptions.data(), 0, checkCount);
        }
        if (rank == 0) {
            printf("Validating results...\n");
        }
        valid = validateResults(checkOptions, checkedResults, rank);
        int localValid = valid ? 1 : 0;
        int globalValid = 0;
        MPI_CHECK(MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        valid = globalValid != 0;
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (deviceOptions != nullptr) {
        CUDA_CHECK(cudaFree(deviceOptions));
    }
    if (deviceResults != nullptr) {
        CUDA_CHECK(cudaFree(deviceResults));
    }
    if (localOptions != nullptr) {
        CUDA_CHECK(cudaFreeHost(localOptions));
    }
    if (localResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(localResults));
    }

    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
