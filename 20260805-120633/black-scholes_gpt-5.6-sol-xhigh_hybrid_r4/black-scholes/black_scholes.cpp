#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

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

// Standard test cases used both to generate the benchmark and for validation.
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

__constant__ OptionInput deviceTestOptions[7];

__host__ __device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    constexpr double inverseSqrtTwo = 0.707106781186547524400844362104849039;
    return 0.5 * (1.0 + erf(x * inverseSqrtTwo));
}

// Black-Scholes formula for European options. Keeping one host/device function
// ensures that the accelerator follows the original scalar implementation.
__host__ __device__ __forceinline__ double blackScholes(const OptionInput& option) noexcept {
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
    const double discountedSpot = S * exp(-q * T);

    if (option.type == CALL) {
        return discountedSpot * Nd1 - K * discount * Nd2;
    }
    return K * discount * cumulativeNormal(-d2) - discountedSpot * cumulativeNormal(-d1);
}

__global__ void blackScholesKernel(const uint64_t globalOffset,
                                   const uint64_t count,
                                   double* __restrict__ results) {
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t localIndex = first; localIndex < count; localIndex += stride) {
        const uint64_t globalIndex = globalOffset + localIndex;
        OptionInput option = deviceTestOptions[globalIndex % 7U];

        // Generate the same scaled input as the original host implementation.
        // Doing this in registers eliminates a large host-to-device input copy.
        const double factor = 1.0 + 0.1 * (static_cast<double>(globalIndex) / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        results[localIndex] = blackScholes(option);
    }
}

struct DeviceWorker {
    int device = 0;
    uint64_t globalOffset = 0;
    uint64_t localOffset = 0;
    uint64_t count = 0;
    double* deviceResults = nullptr;
    int blockSize = 0;
    int gridSize = 0;
    cudaError_t error = cudaSuccess;
    const char* operation = nullptr;
};

void setWorkerError(DeviceWorker& worker, const cudaError_t error, const char* operation) {
    if (worker.error == cudaSuccess && error != cudaSuccess) {
        worker.error = error;
        worker.operation = operation;
    }
}

void initializeWorker(DeviceWorker& worker,
                      const std::array<OptionInput, 7>& testOptions) {
    setWorkerError(worker, cudaSetDevice(worker.device), "selecting CUDA device");
    if (worker.error == cudaSuccess) {
        // Force context creation before the timed region.
        setWorkerError(worker, cudaFree(nullptr), "creating CUDA context");
    }
    if (worker.error == cudaSuccess) {
        setWorkerError(worker,
                       cudaMemcpyToSymbol(deviceTestOptions, testOptions.data(),
                                          sizeof(testOptions)),
                       "uploading option templates");
    }

    if (worker.error != cudaSuccess || worker.count == 0) {
        return;
    }

    const size_t allocationSize = static_cast<size_t>(worker.count) * sizeof(double);
    setWorkerError(worker,
                   cudaMalloc(reinterpret_cast<void**>(&worker.deviceResults), allocationSize),
                   "allocating device results");

    int minimumGridSize = 0;
    if (worker.error == cudaSuccess) {
        setWorkerError(worker,
                       cudaOccupancyMaxPotentialBlockSize(&minimumGridSize, &worker.blockSize,
                                                          blackScholesKernel, 0, 0),
                       "calculating kernel occupancy");
    }
    if (worker.error != cudaSuccess) {
        return;
    }

    const uint64_t blocksNeeded =
        (worker.count + static_cast<uint64_t>(worker.blockSize) - 1U) /
        static_cast<uint64_t>(worker.blockSize);
    const uint64_t occupancyWaves = static_cast<uint64_t>(minimumGridSize) * 4U;
    worker.gridSize = static_cast<int>(std::min(blocksNeeded, occupancyWaves));
}

void runWorker(DeviceWorker& worker) {
    if (worker.error != cudaSuccess || worker.count == 0) {
        return;
    }

    setWorkerError(worker, cudaSetDevice(worker.device), "selecting CUDA device for pricing");
    if (worker.error != cudaSuccess) {
        return;
    }

    blackScholesKernel<<<worker.gridSize, worker.blockSize>>>(
        worker.globalOffset, worker.count, worker.deviceResults);
    setWorkerError(worker, cudaGetLastError(), "launching pricing kernel");
    if (worker.error == cudaSuccess) {
        setWorkerError(worker, cudaDeviceSynchronize(), "executing pricing kernel");
    }
}

void copyWorkerResults(DeviceWorker& worker, std::vector<double>& localResults) {
    if (worker.error != cudaSuccess || worker.count == 0) {
        return;
    }

    setWorkerError(worker, cudaSetDevice(worker.device), "selecting CUDA device for result copy");
    if (worker.error == cudaSuccess) {
        setWorkerError(
            worker,
            cudaMemcpy(localResults.data() + static_cast<size_t>(worker.localOffset),
                       worker.deviceResults,
                       static_cast<size_t>(worker.count) * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "copying results to host");
    }
}

void releaseWorker(DeviceWorker& worker) {
    if (worker.deviceResults == nullptr) {
        return;
    }

    const cudaError_t selectError = cudaSetDevice(worker.device);
    if (selectError != cudaSuccess) {
        setWorkerError(worker, selectError, "selecting CUDA device for cleanup");
        return;
    }
    const cudaError_t freeError = cudaFree(worker.deviceResults);
    setWorkerError(worker, freeError, "freeing device results");
    if (freeError == cudaSuccess) {
        worker.deviceResults = nullptr;
    }
}

[[noreturn]] void abortCudaFailure(const std::vector<DeviceWorker>& workers, const int rank) {
    for (const DeviceWorker& worker : workers) {
        if (worker.error != cudaSuccess) {
            std::fprintf(stderr, "MPI rank %d, CUDA device %d: %s failed: %s\n",
                         rank, worker.device,
                         worker.operation == nullptr ? "CUDA operation" : worker.operation,
                         cudaGetErrorString(worker.error));
            break;
        }
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

bool workersSucceeded(const std::vector<DeviceWorker>& workers) {
    return std::all_of(workers.begin(), workers.end(), [](const DeviceWorker& worker) {
        return worker.error == cudaSuccess;
    });
}

OptionInput makeOption(const uint64_t index) {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[index % testOptions.size()];
    const double factor = 1.0 + 0.1 * (static_cast<double>(index) /
                                      static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

bool validateResults(const uint64_t numOptions, const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min<uint64_t>(10U, numOptions));

    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const OptionInput option = makeOption(static_cast<uint64_t>(i));
        const double computed = results[static_cast<size_t>(i)];
        const double expected = option.value;
        const double error = std::fabs(computed - expected);
        const double relativeError = error / (std::fabs(expected) + 1.0e-10);

        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relativeError);

        // Preserve the original benchmark's relaxed validation criteria.
        if (computed < 0.0 || computed > 1000.0 ||
            std::isnan(computed) || std::isinf(computed)) {
            std::printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void gatherResults(const std::vector<double>& localResults,
                   const std::vector<uint64_t>& rankOffsets,
                   const int rank,
                   const int worldSize,
                   std::vector<double>& results) {
    const uint64_t numOptions = rankOffsets.back();
    const uint64_t rankBegin = rankOffsets[rank];
    const uint64_t rankEnd = rankOffsets[rank + 1];
    constexpr uint64_t maxChunk = static_cast<uint64_t>(std::numeric_limits<int>::max());
    double unused = 0.0;

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        receiveCounts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
    }

    // Chunking keeps every MPI count and displacement within the MPI-3 int
    // interface while still supporting data sets larger than INT_MAX.
    for (uint64_t windowBegin = 0; windowBegin < numOptions; windowBegin += maxChunk) {
        const uint64_t windowEnd = std::min(numOptions, windowBegin + maxChunk);
        const uint64_t sendBegin = std::max(rankBegin, windowBegin);
        const uint64_t sendEnd = std::min(rankEnd, windowEnd);
        const uint64_t sendElements = sendEnd > sendBegin ? sendEnd - sendBegin : 0U;
        const size_t localIndex = static_cast<size_t>(sendBegin - rankBegin);
        const double* sendBuffer = sendElements == 0 ? &unused : localResults.data() + localIndex;

        if (rank == 0) {
            for (int source = 0; source < worldSize; ++source) {
                const uint64_t sourceBegin = std::max(rankOffsets[source], windowBegin);
                const uint64_t sourceEnd = std::min(rankOffsets[source + 1], windowEnd);
                const uint64_t elements = sourceEnd > sourceBegin ? sourceEnd - sourceBegin : 0U;
                receiveCounts[source] = static_cast<int>(elements);
                displacements[source] = elements == 0
                    ? 0
                    : static_cast<int>(sourceBegin - windowBegin);
            }
        }

        double* receiveBuffer = rank == 0
            ? results.data() + static_cast<size_t>(windowBegin)
            : nullptr;
        MPI_Gatherv(sendBuffer, static_cast<int>(sendElements), MPI_DOUBLE,
                    receiveBuffer,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "The MPI implementation does not provide MPI_THREAD_FUNNELED.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    uint64_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool argumentsValid = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || argv[i][0] == '-') {
                argumentsValid = false;
            } else {
                numOptions = static_cast<uint64_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (numOptions > static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                                           sizeof(double))) {
        argumentsValid = false;
    }
    if (!argumentsValid || showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %" PRIu64 "\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_size(localCommunicator, &localSize);

    int deviceCount = 0;
    const cudaError_t deviceCountError = cudaGetDeviceCount(&deviceCount);
    if (deviceCountError != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr, "MPI rank %d: no usable CUDA device: %s\n", rank,
                     cudaGetErrorString(deviceCountError));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Split all visible GPUs between ranks on a node. This supports both one
    // rank per GPU and one rank driving several GPUs without oversubscription.
    std::vector<int> assignedDevices;
    if (localSize <= deviceCount) {
        for (int device = localRank; device < deviceCount; device += localSize) {
            assignedDevices.push_back(device);
        }
    } else {
        // A scheduler may expose one logical CUDA device per rank. In that case
        // logical device zero can identify a different physical GPU per process.
        assignedDevices.push_back(localRank % deviceCount);
    }

    const uint64_t localWorkerCount = assignedDevices.size();
    std::vector<uint64_t> workerCounts(static_cast<size_t>(worldSize));
    MPI_Allgather(&localWorkerCount, 1, MPI_UINT64_T,
                  workerCounts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    std::vector<uint64_t> workerOffsets(static_cast<size_t>(worldSize) + 1U, 0U);
    for (int process = 0; process < worldSize; ++process) {
        workerOffsets[process + 1] = workerOffsets[process] + workerCounts[process];
    }
    const uint64_t totalWorkerCount = workerOffsets.back();
    const uint64_t optionsPerWorker = numOptions / totalWorkerCount;
    const uint64_t workersWithExtraOption = numOptions % totalWorkerCount;
    const auto optionsBeforeWorker = [&](const uint64_t worker) {
        return optionsPerWorker * worker + std::min(worker, workersWithExtraOption);
    };

    std::vector<uint64_t> rankOffsets(static_cast<size_t>(worldSize) + 1U, 0U);
    for (int process = 0; process <= worldSize; ++process) {
        rankOffsets[process] = optionsBeforeWorker(workerOffsets[process]);
    }

    std::vector<DeviceWorker> workers(assignedDevices.size());
    for (size_t i = 0; i < workers.size(); ++i) {
        const uint64_t globalWorker = workerOffsets[rank] + static_cast<uint64_t>(i);
        workers[i].device = assignedDevices[i];
        workers[i].globalOffset = optionsBeforeWorker(globalWorker);
        workers[i].localOffset = workers[i].globalOffset - rankOffsets[rank];
        workers[i].count = optionsBeforeWorker(globalWorker + 1U) - workers[i].globalOffset;
    }

    constexpr auto testOptions = getTestOptions();
    const int ompWorkers = static_cast<int>(workers.size());
    omp_set_dynamic(0);
#pragma omp parallel for schedule(static) num_threads(ompWorkers)
    for (int i = 0; i < ompWorkers; ++i) {
        initializeWorker(workers[static_cast<size_t>(i)], testOptions);
    }
    if (!workersSucceeded(workers)) {
#pragma omp parallel for schedule(static) num_threads(ompWorkers)
        for (int i = 0; i < ompWorkers; ++i) {
            releaseWorker(workers[static_cast<size_t>(i)]);
        }
        abortCudaFailure(workers, rank);
    }

    if (rank == 0) {
        std::printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

#pragma omp parallel for schedule(static) num_threads(ompWorkers)
    for (int i = 0; i < ompWorkers; ++i) {
        runWorker(workers[static_cast<size_t>(i)]);
    }

    const double localDuration = MPI_Wtime() - start;
    if (!workersSucceeded(workers)) {
#pragma omp parallel for schedule(static) num_threads(ompWorkers)
        for (int i = 0; i < ompWorkers; ++i) {
            releaseWorker(workers[static_cast<size_t>(i)]);
        }
        abortCudaFailure(workers, rank);
    }

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double milliseconds = duration * 1.0e3;
        const double optionsPerSecond = duration > 0.0
            ? static_cast<double>(numOptions) / duration
            : 0.0;
        std::printf("Computation time: %.3f ms\n", milliseconds);
        std::printf("Options per second: %.0f\n", optionsPerSecond);
    }

    const bool needHostResults = validate || printResults;
    std::vector<double> localResults;
    if (needHostResults) {
        const uint64_t localOptionCount = rankOffsets[rank + 1] - rankOffsets[rank];
        localResults.resize(static_cast<size_t>(localOptionCount));
#pragma omp parallel for schedule(static) num_threads(ompWorkers)
        for (int i = 0; i < ompWorkers; ++i) {
            copyWorkerResults(workers[static_cast<size_t>(i)], localResults);
        }
    }

#pragma omp parallel for schedule(static) num_threads(ompWorkers)
    for (int i = 0; i < ompWorkers; ++i) {
        releaseWorker(workers[static_cast<size_t>(i)]);
    }
    if (!workersSucceeded(workers)) {
        abortCudaFailure(workers, rank);
    }

    std::vector<double> results;
    if (needHostResults) {
        if (rank == 0) {
            results.resize(static_cast<size_t>(numOptions));
        }
        gatherResults(localResults, rankOffsets, rank, worldSize, results);
    }

    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }

    int valid = 1;
    if (rank == 0 && validate) {
        std::printf("Validating results...\n");
        valid = validateResults(numOptions, results) ? 1 : 0;
        std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return valid != 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
