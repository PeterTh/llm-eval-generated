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
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerSm = 8;
constexpr int kMpiResultTag = 741;
constexpr std::size_t kMinimumOptionsPerGpu = 131072;

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

enum OptionField : std::size_t {
    STRIKE = 0,
    SPOT,
    DIVIDEND_YIELD,
    RISK_FREE_RATE,
    TIME_TO_MATURITY,
    VOLATILITY,
    FIELD_COUNT
};

constexpr std::array<OptionInput, 7> kTestOptions{{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
}};

struct RankRange {
    std::size_t offset;
    std::size_t count;
};

struct DeviceWork {
    int device = 0;
    int blocks = 1;
    std::size_t localOffset = 0;
    std::size_t count = 0;
    double* fields = nullptr;
    int* types = nullptr;
    double* results = nullptr;
};

RankRange rankRange(const std::size_t total, const int rank, const int ranks) noexcept {
    const std::size_t rankCount = static_cast<std::size_t>(ranks);
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t quotient = total / rankCount;
    const std::size_t remainder = total % rankCount;
    return {quotient * rankIndex + std::min(rankIndex, remainder),
            quotient + (rankIndex < remainder ? 1U : 0U)};
}

__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * kInvSqrt2));
}

__global__ __launch_bounds__(kThreadsPerBlock)
void blackScholesKernel(const double* __restrict__ fields,
                        const int* __restrict__ types,
                        double* __restrict__ results,
                        const std::size_t count) {
    const double* const strike = fields + STRIKE * count;
    const double* const spot = fields + SPOT * count;
    const double* const q = fields + DIVIDEND_YIELD * count;
    const double* const r = fields + RISK_FREE_RATE * count;
    const double* const t = fields + TIME_TO_MATURITY * count;
    const double* const vol = fields + VOLATILITY * count;

    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = first; i < count; i += stride) {
        const double maturity = t[i];
        const double sigma = vol[i];
        if (maturity <= 0.0 || sigma <= 0.0) {
            results[i] = 0.0;
            continue;
        }

        const double rootT = sqrt(maturity);
        const double sigmaRootT = sigma * rootT;
        const double d1 = (log(spot[i] / strike[i]) +
                           (r[i] - q[i] + 0.5 * sigma * sigma) * maturity) /
                          sigmaRootT;
        const double d2 = d1 - sigmaRootT;
        const double discountR = exp(-r[i] * maturity);
        const double discountQ = exp(-q[i] * maturity);

        if (types[i] == CALL) {
            results[i] = spot[i] * discountQ * cumulativeNormal(d1) -
                         strike[i] * discountR * cumulativeNormal(d2);
        } else {
            results[i] = strike[i] * discountR * cumulativeNormal(-d2) -
                         spot[i] * discountQ * cumulativeNormal(-d1);
        }
    }
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, std::size_t& value) noexcept {
    if (text == nullptr || text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return value <= std::numeric_limits<std::size_t>::max() / FIELD_COUNT;
}

bool reportCudaError(const cudaError_t status, const char* operation,
                     const int rank, const int device) noexcept {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "MPI rank %d, CUDA device %d: %s failed: %s\n",
                 rank, device, operation, cudaGetErrorString(status));
    return false;
}

bool prepareDevice(DeviceWork& work, const double* hostFields,
                   const int* hostTypes, const std::size_t rankCount,
                   const int rank) {
    if (!reportCudaError(cudaSetDevice(work.device), "cudaSetDevice", rank, work.device) ||
        !reportCudaError(cudaFree(nullptr), "CUDA context initialization", rank, work.device)) {
        return false;
    }

    int multiprocessors = 1;
    if (!reportCudaError(cudaDeviceGetAttribute(&multiprocessors,
                                                 cudaDevAttrMultiProcessorCount,
                                                 work.device),
                         "cudaDeviceGetAttribute", rank, work.device)) {
        return false;
    }
    const std::size_t neededBlocks =
        (work.count + static_cast<std::size_t>(kThreadsPerBlock) - 1U) /
        static_cast<std::size_t>(kThreadsPerBlock);
    work.blocks = static_cast<int>(std::max<std::size_t>(
        1U, std::min<std::size_t>(neededBlocks,
                                  static_cast<std::size_t>(multiprocessors) * kBlocksPerSm)));

    if (work.count == 0) {
        return true;
    }

    const std::size_t fieldBytes = work.count * FIELD_COUNT * sizeof(double);
    if (!reportCudaError(cudaMalloc(reinterpret_cast<void**>(&work.fields), fieldBytes),
                         "cudaMalloc(option fields)", rank, work.device) ||
        !reportCudaError(cudaMalloc(reinterpret_cast<void**>(&work.types),
                                    work.count * sizeof(int)),
                         "cudaMalloc(option types)", rank, work.device) ||
        !reportCudaError(cudaMalloc(reinterpret_cast<void**>(&work.results),
                                    work.count * sizeof(double)),
                         "cudaMalloc(results)", rank, work.device)) {
        return false;
    }

    for (std::size_t field = 0; field < FIELD_COUNT; ++field) {
        const double* source = hostFields + field * rankCount + work.localOffset;
        double* destination = work.fields + field * work.count;
        if (!reportCudaError(cudaMemcpy(destination, source,
                                        work.count * sizeof(double),
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy(option fields)", rank, work.device)) {
            return false;
        }
    }
    return reportCudaError(cudaMemcpy(work.types,
                                      hostTypes + work.localOffset,
                                      work.count * sizeof(int),
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(option types)", rank, work.device);
}

bool runDevice(DeviceWork& work, const int rank) {
    if (!reportCudaError(cudaSetDevice(work.device), "cudaSetDevice", rank, work.device)) {
        return false;
    }
    if (work.count != 0) {
        blackScholesKernel<<<work.blocks, kThreadsPerBlock>>>(
            work.fields, work.types, work.results, work.count);
        if (!reportCudaError(cudaGetLastError(), "Black-Scholes kernel launch",
                             rank, work.device)) {
            return false;
        }
    }
    return reportCudaError(cudaDeviceSynchronize(), "Black-Scholes kernel execution",
                           rank, work.device);
}

void releaseDevice(DeviceWork& work) noexcept {
    cudaSetDevice(work.device);
    cudaFree(work.results);
    cudaFree(work.types);
    cudaFree(work.fields);
    work.results = nullptr;
    work.types = nullptr;
    work.fields = nullptr;
}

void gatherResults(const std::vector<double>& local, std::vector<double>& global,
                   const std::size_t total, const int rank, const int ranks) {
    const RankRange own = rankRange(total, rank, ranks);
    if (total <= static_cast<std::size_t>(INT_MAX)) {
        std::vector<int> counts;
        std::vector<int> displacements;
        if (rank == 0) {
            counts.resize(static_cast<std::size_t>(ranks));
            displacements.resize(static_cast<std::size_t>(ranks));
            for (int source = 0; source < ranks; ++source) {
                const RankRange range = rankRange(total, source, ranks);
                counts[static_cast<std::size_t>(source)] = static_cast<int>(range.count);
                displacements[static_cast<std::size_t>(source)] = static_cast<int>(range.offset);
            }
        }
        MPI_Gatherv(local.data(), static_cast<int>(own.count), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return;
    }

    constexpr std::size_t kMaxMessage = static_cast<std::size_t>(INT_MAX);
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin() + own.offset);
        for (int source = 1; source < ranks; ++source) {
            const RankRange range = rankRange(total, source, ranks);
            for (std::size_t received = 0; received < range.count;) {
                const std::size_t chunk = std::min(kMaxMessage, range.count - received);
                MPI_Recv(global.data() + range.offset + received, static_cast<int>(chunk),
                         MPI_DOUBLE, source, kMpiResultTag, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                received += chunk;
            }
        }
    } else {
        for (std::size_t sent = 0; sent < own.count;) {
            const std::size_t chunk = std::min(kMaxMessage, own.count - sent);
            MPI_Send(local.data() + sent, static_cast<int>(chunk), MPI_DOUBLE, 0,
                     kMpiResultTag, MPI_COMM_WORLD);
            sent += chunk;
        }
    }
}

bool validateResults(const std::size_t numOptions,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const std::size_t numChecks = std::min<std::size_t>(10, numOptions);

    std::printf("Checking computed option prices:\n");
    for (std::size_t i = 0; i < numChecks; ++i) {
        const OptionInput& base = kTestOptions[i % kTestOptions.size()];
        const double computed = results[i];
        const double expected = base.value;
        const double error = std::fabs(computed - expected);
        const double relativeError = error / (std::fabs(expected) + 1.0e-10);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
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

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required FUNNELED thread support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    std::size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], numOptions)) {
                argumentsValid = false;
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
    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    int localRanks = 1;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_size(localCommunicator, &localRanks);

    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    int localCudaFailure = (deviceStatus != cudaSuccess || deviceCount == 0) ? 1 : 0;
    if (localCudaFailure != 0) {
        std::fprintf(stderr, "MPI rank %d: no usable CUDA device: %s\n", rank,
                     deviceStatus == cudaSuccess ? "no devices visible"
                                                 : cudaGetErrorString(deviceStatus));
    }
    int anyCudaFailure = 0;
    MPI_Allreduce(&localCudaFailure, &anyCudaFailure, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (anyCudaFailure != 0) {
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    std::vector<int> assignedDevices;
    if (deviceCount >= localRanks) {
        for (int device = localRank; device < deviceCount; device += localRanks) {
            assignedDevices.push_back(device);
        }
    } else {
        assignedDevices.push_back(localRank % deviceCount);
    }

    const RankRange range = rankRange(numOptions, rank, ranks);
    const std::size_t usefulDevices = range.count == 0
                                          ? 1U
                                          : 1U + (range.count - 1U) /
                                                     kMinimumOptionsPerGpu;
    const std::size_t activeDevices =
        std::min<std::size_t>(assignedDevices.size(), usefulDevices);
    assignedDevices.resize(activeDevices);

    int hostThreads = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        hostThreads = std::max(1, omp_get_num_procs() / localRanks);
    }
    if (range.count != 0) {
        hostThreads = std::min<int>(hostThreads, static_cast<int>(
            std::min<std::size_t>(range.count, static_cast<std::size_t>(INT_MAX))));
    } else {
        hostThreads = 1;
    }
    omp_set_dynamic(0);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
                    ranks, hostThreads);
    }

    // Default-initialized arrays avoid a serial zero-fill, allowing the OpenMP
    // loop below to establish NUMA-local first-touch placement.
    std::unique_ptr<double[]> hostFields(new double[range.count * FIELD_COUNT]);
    std::unique_ptr<int[]> hostTypes(new int[range.count]);
#pragma omp parallel for schedule(static) num_threads(hostThreads)
    for (std::int64_t localIndex = 0;
         localIndex < static_cast<std::int64_t>(range.count); ++localIndex) {
        const std::size_t local = static_cast<std::size_t>(localIndex);
        const std::size_t global = range.offset + local;
        const OptionInput& base = kTestOptions[global % kTestOptions.size()];
        const double factor =
            1.0 + 0.1 * (global / static_cast<double>(kTestOptions.size()));
        hostTypes[local] = base.type;
        hostFields[STRIKE * range.count + local] = base.strike * factor;
        hostFields[SPOT * range.count + local] = base.spot * factor;
        hostFields[DIVIDEND_YIELD * range.count + local] = base.q;
        hostFields[RISK_FREE_RATE * range.count + local] = base.r;
        hostFields[TIME_TO_MATURITY * range.count + local] = base.t;
        hostFields[VOLATILITY * range.count + local] = base.vol;
    }

    std::vector<DeviceWork> work(activeDevices);
    for (std::size_t worker = 0; worker < activeDevices; ++worker) {
        work[worker].device = assignedDevices[worker];
        work[worker].localOffset = range.count * worker / activeDevices;
        const std::size_t end = range.count * (worker + 1U) / activeDevices;
        work[worker].count = end - work[worker].localOffset;
    }

    int preparationFailed = 0;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices)) reduction(| : preparationFailed)
    for (std::int64_t worker = 0; worker < static_cast<std::int64_t>(activeDevices);
         ++worker) {
        preparationFailed |= !prepareDevice(work[static_cast<std::size_t>(worker)],
                                            hostFields.get(), hostTypes.get(),
                                            range.count, rank);
    }
    hostFields.reset();
    hostTypes.reset();

    MPI_Allreduce(&preparationFailed, &anyCudaFailure, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (anyCudaFailure != 0) {
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices))
        for (std::int64_t worker = 0;
             worker < static_cast<std::int64_t>(activeDevices); ++worker) {
            releaseDevice(work[static_cast<std::size_t>(worker)]);
        }
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int executionFailed = 0;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices)) reduction(| : executionFailed)
    for (std::int64_t worker = 0; worker < static_cast<std::int64_t>(activeDevices);
         ++worker) {
        executionFailed |= !runDevice(work[static_cast<std::size_t>(worker)], rank);
    }
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&executionFailed, &anyCudaFailure, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (anyCudaFailure != 0) {
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices))
        for (std::int64_t worker = 0;
             worker < static_cast<std::int64_t>(activeDevices); ++worker) {
            releaseDevice(work[static_cast<std::size_t>(worker)]);
        }
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Options per second: %.0f\n",
                    elapsed > 0.0 ? static_cast<double>(numOptions) / elapsed : 0.0);
    }

    std::vector<double> localResults;
    if (printResults) {
        localResults.resize(range.count);
        int copyFailed = 0;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices)) reduction(| : copyFailed)
        for (std::int64_t worker = 0;
             worker < static_cast<std::int64_t>(activeDevices); ++worker) {
            DeviceWork& item = work[static_cast<std::size_t>(worker)];
            if (!reportCudaError(cudaSetDevice(item.device), "cudaSetDevice", rank,
                                 item.device) ||
                (item.count != 0 &&
                 !reportCudaError(cudaMemcpy(localResults.data() + item.localOffset,
                                             item.results,
                                             item.count * sizeof(double),
                                             cudaMemcpyDeviceToHost),
                                  "cudaMemcpy(results)", rank, item.device))) {
                copyFailed = 1;
            }
        }
        MPI_Allreduce(&copyFailed, &anyCudaFailure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
    }

    const std::size_t validationCount = std::min<std::size_t>(10, numOptions);
    std::vector<double> validationValues(validationCount, 0.0);
    if (validate && !printResults) {
        int copyFailed = 0;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices)) reduction(| : copyFailed)
        for (std::int64_t worker = 0;
             worker < static_cast<std::int64_t>(activeDevices); ++worker) {
            DeviceWork& item = work[static_cast<std::size_t>(worker)];
            const std::size_t itemGlobalStart = range.offset + item.localOffset;
            const std::size_t itemGlobalEnd = itemGlobalStart + item.count;
            const std::size_t copyStart = itemGlobalStart;
            const std::size_t copyEnd = std::min(itemGlobalEnd, validationCount);
            if (copyStart < copyEnd) {
                if (!reportCudaError(cudaSetDevice(item.device), "cudaSetDevice", rank,
                                     item.device) ||
                    !reportCudaError(cudaMemcpy(validationValues.data() + copyStart,
                                                item.results + (copyStart - itemGlobalStart),
                                                (copyEnd - copyStart) * sizeof(double),
                                                cudaMemcpyDeviceToHost),
                                     "cudaMemcpy(validation results)", rank,
                                     item.device)) {
                    copyFailed = 1;
                }
            }
        }
        MPI_Allreduce(&copyFailed, &anyCudaFailure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        std::vector<double> reducedValidation(validationCount, 0.0);
        MPI_Reduce(validationValues.data(), reducedValidation.data(),
                   static_cast<int>(validationCount), MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
        if (rank == 0) {
            validationValues.swap(reducedValidation);
        }
    }

    std::vector<double> globalResults;
    if (printResults && anyCudaFailure == 0) {
        if (rank == 0) {
            globalResults.resize(numOptions);
        }
        gatherResults(localResults, globalResults, numOptions, rank, ranks);
        if (rank == 0) {
            print_results(globalResults, "OptionPrices");
        }
    }

    bool valid = true;
    if (validate && anyCudaFailure == 0 && rank == 0) {
        std::printf("Validating results...\n");
        valid = validateResults(numOptions,
                                printResults ? globalResults : validationValues);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    int validInt = valid ? 1 : 0;
    MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(activeDevices))
    for (std::int64_t worker = 0; worker < static_cast<std::int64_t>(activeDevices);
         ++worker) {
        releaseDevice(work[static_cast<std::size_t>(worker)]);
    }
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return (anyCudaFailure == 0 && validInt != 0) ? 0 : 1;
}
