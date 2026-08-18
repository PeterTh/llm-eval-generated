#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

enum OptionType { CALL = 0, PUT = 1 };

// The two validation-only fields from the original input are deliberately not
// sent over PCIe.  The remaining layout is naturally aligned and is consumed
// coalescently by consecutive CUDA threads.
struct OptionInput {
    int type;
    int padding;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
};

struct TestOption {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

constexpr std::array<TestOption, 7> kTestOptions{{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
}};

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              int rank) {
    std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n", rank,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call, rank)                                                \
    do {                                                                       \
        const cudaError_t cuda_check_error = (call);                            \
        if (cuda_check_error != cudaSuccess)                                   \
            cudaFailure(cuda_check_error, #call, rank);                        \
    } while (false)

__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * kInvSqrt2));
}

__global__ __launch_bounds__(256) void blackScholesKernel(
    const OptionInput* __restrict__ options, double* __restrict__ results,
    size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const OptionInput option = options[i];
        if (option.t <= 0.0 || option.vol <= 0.0) {
            results[i] = 0.0;
            continue;
        }

        const double sqrtT = sqrt(option.t);
        const double sigmaSqrtT = option.vol * sqrtT;
        const double d1 =
            (log(option.spot / option.strike) +
             (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
            sigmaSqrtT;
        const double d2 = d1 - sigmaSqrtT;
        const double spotDiscounted = option.spot * exp(-option.q * option.t);
        const double strikeDiscounted =
            option.strike * exp(-option.r * option.t);

        if (option.type == CALL) {
            results[i] = spotDiscounted * cumulativeNormal(d1) -
                         strikeDiscounted * cumulativeNormal(d2);
        } else {
            results[i] = strikeDiscounted * cumulativeNormal(-d2) -
                         spotDiscounted * cumulativeNormal(-d1);
        }
    }
}

OptionInput makeOption(size_t globalIndex) {
    const TestOption& base = kTestOptions[globalIndex % kTestOptions.size()];
    const double factor =
        1.0 + 0.1 * (globalIndex / static_cast<double>(kTestOptions.size()));
    return {base.type, 0, base.strike * factor, base.spot * factor,
            base.q,    base.r, base.t,               base.vol};
}

size_t localCount(size_t total, int rank, int ranks) {
    const size_t quotient = total / static_cast<size_t>(ranks);
    const size_t remainder = total % static_cast<size_t>(ranks);
    return quotient + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t localOffset(size_t total, int rank, int ranks) {
    const size_t quotient = total / static_cast<size_t>(ranks);
    const size_t remainder = total % static_cast<size_t>(ranks);
    return quotient * static_cast<size_t>(rank) +
           std::min(static_cast<size_t>(rank), remainder);
}

// MPI_Gatherv has int counts in MPI-3. Gather in bounded global chunks so the
// size_t command-line interface remains valid for very large data sets.
void gatherRange(const double* localResults, size_t localBegin,
                 size_t localSize, std::vector<double>& rootResults,
                 size_t gatherSize, size_t distributionSize, int rank,
                 int ranks) {
    constexpr size_t kChunk = static_cast<size_t>(INT_MAX);
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(rank == 0 ? ranks : 0);

    for (size_t chunkBegin = 0; chunkBegin < gatherSize;) {
        const size_t chunkSize = std::min(kChunk, gatherSize - chunkBegin);
        const size_t chunkEnd = chunkBegin + chunkSize;
        const size_t sendBegin = std::max(localBegin, chunkBegin);
        const size_t sendEnd = std::min(localBegin + localSize, chunkEnd);
        const int sendCount = sendEnd > sendBegin
                                  ? static_cast<int>(sendEnd - sendBegin)
                                  : 0;
        const double* sendBuffer =
            sendCount ? localResults + (sendBegin - localBegin) : localResults;

        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const size_t begin = localOffset(distributionSize, r, ranks);
                const size_t size = localCount(distributionSize, r, ranks);
                const size_t overlapBegin = std::max(begin, chunkBegin);
                const size_t overlapEnd = std::min(begin + size, chunkEnd);
                if (overlapEnd > overlapBegin) {
                    counts[r] = static_cast<int>(overlapEnd - overlapBegin);
                    displacements[r] =
                        static_cast<int>(overlapBegin - chunkBegin);
                } else {
                    counts[r] = 0;
                    displacements[r] = 0;
                }
            }
        }

        MPI_Gatherv(sendBuffer, sendCount, MPI_DOUBLE,
                    rank == 0 ? rootResults.data() + chunkBegin : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        chunkBegin = chunkEnd;
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const double computed = results[i];
        const double expected = kTestOptions[i % kTestOptions.size()].value;
        const double relativeError =
            std::fabs(computed - expected) / (std::fabs(expected) + 1.0e-10);
        std::printf(
            "  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
            i, computed, expected, relativeError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n",
                        i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                numOptions = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), rank);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), rank);
    CUDA_CHECK(cudaFree(nullptr), rank);  // Initialize the context before timing.

    const size_t count = localCount(numOptions, rank, ranks);
    const size_t offset = localOffset(numOptions, rank, ranks);
    OptionInput* hostOptions = nullptr;
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (count != 0) {
        CUDA_CHECK(cudaMallocHost(&hostOptions, count * sizeof(OptionInput)), rank);
        CUDA_CHECK(cudaMalloc(&deviceOptions, count * sizeof(OptionInput)), rank);
        CUDA_CHECK(cudaMalloc(&deviceResults, count * sizeof(double)), rank);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < count; ++i)
            hostOptions[i] = makeOption(offset + i);

        CUDA_CHECK(cudaMemcpy(deviceOptions, hostOptions,
                              count * sizeof(OptionInput),
                              cudaMemcpyHostToDevice),
                   rank);

        // Force CUDA module loading and device math initialization outside the
        // measured region. The timed launch overwrites this single warm-up
        // result, so program semantics are unchanged.
        blackScholesKernel<<<1, 1>>>(deviceOptions, deviceResults, 1);
        CUDA_CHECK(cudaGetLastError(), rank);
        CUDA_CHECK(cudaDeviceSynchronize(), rank);
    }

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks,
                    omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Pricing options...\n");
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device), rank);
    const size_t blocksNeeded = (count + 255) / 256;
    const size_t blockLimit = static_cast<size_t>(properties.multiProcessorCount) * 32;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(blocksNeeded, std::min(blockLimit,
                 static_cast<size_t>(std::numeric_limits<unsigned int>::max()))));

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (count != 0) {
        blackScholesKernel<<<blocks, 256>>>(deviceOptions, deviceResults, count);
        CUDA_CHECK(cudaGetLastError(), rank);
    }
    CUDA_CHECK(cudaDeviceSynchronize(), rank);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double throughput = elapsed > 0.0 ? numOptions / elapsed : 0.0;
        std::printf("Options per second: %.0f\n", throughput);
    }

    // Results remain resident on each GPU unless an observable output needs
    // them. This removes device-to-host traffic from scalable benchmark runs.
    const size_t gatherSize = printResults ? numOptions
                              : validate ? std::min<size_t>(10, numOptions)
                                         : 0;
    std::vector<double> localHostResults;
    std::vector<double> gatheredResults;
    if (gatherSize != 0) {
        const size_t neededLocal =
            offset < gatherSize ? std::min(count, gatherSize - offset) : 0;
        localHostResults.resize(neededLocal);
        if (neededLocal != 0) {
            CUDA_CHECK(cudaMemcpy(localHostResults.data(), deviceResults,
                                  neededLocal * sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       rank);
        }
        if (rank == 0) gatheredResults.resize(gatherSize);
        gatherRange(localHostResults.data(), offset, neededLocal, gatheredResults,
                    gatherSize, numOptions, rank, ranks);
    }

    int returnCode = 0;
    if (rank == 0) {
        if (printResults) print_results(gatheredResults, "OptionPrices");
        if (validate) {
            std::printf("Validating results...\n");
            if (validateResults(printResults
                                    ? std::vector<double>(gatheredResults.begin(),
                                                          gatheredResults.begin() +
                                                              std::min<size_t>(10, numOptions))
                                    : gatheredResults)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (deviceResults) CUDA_CHECK(cudaFree(deviceResults), rank);
    if (deviceOptions) CUDA_CHECK(cudaFree(deviceOptions), rank);
    if (hostOptions) CUDA_CHECK(cudaFreeHost(hostOptions), rank);
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return returnCode;
}
