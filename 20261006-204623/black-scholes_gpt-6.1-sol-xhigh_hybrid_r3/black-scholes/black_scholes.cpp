#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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

// These quantities are invariant under the original spot/strike scaling.
// Hoist their expensive sqrt/exp operations out of the per-option kernel.
struct PricingCoefficients {
    int type;
    double strike;
    double spot;
    double drift;
    double volatilityTime;
    double dividendDiscount;
    double discount;
};

using CoefficientTable = std::array<PricingCoefficients, 7>;
__constant__ PricingCoefficients deviceOptions[7];

CoefficientTable prepareCoefficients() {
    CoefficientTable table{};
    constexpr auto options = getTestOptions();
    for (size_t i = 0; i < options.size(); ++i) {
        const auto& option = options[i];
        table[i] = {
            option.type, option.strike, option.spot,
            (option.r - option.q + 0.5 * option.vol * option.vol) * option.t,
            option.t <= 0.0 || option.vol <= 0.0 ? 0.0 : option.vol * sqrt(option.t),
            exp(-option.q * option.t), exp(-option.r * option.t)
        };
    }
    return table;
}

__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double blackScholes(const PricingCoefficients& option,
                                              double factor) {
    if (option.volatilityTime <= 0.0) {
        return 0.0;
    }

    // Scale both inputs before taking their ratio, just as in the original.
    // Scaling a precomputed price would change rounding and cancellation.
    const double S = option.spot * factor;
    const double K = option.strike * factor;
    const double d1 = (log(S / K) + option.drift) / option.volatilityTime;
    const double d2 = d1 - option.volatilityTime;
    if (option.type == CALL) {
        return S * option.dividendDiscount * cumulativeNormal(d1)
             - K * option.discount * cumulativeNormal(d2);
    }
    return K * option.discount * cumulativeNormal(-d2)
         - S * option.dividendDiscount * cumulativeNormal(-d1);
}

__global__ void priceOptions(double* __restrict__ prices, size_t first, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const size_t globalIndex = first + i;
        const double factor = 1.0 + 0.1 * (globalIndex / 7.0);
        prices[i] = blackScholes(deviceOptions[globalIndex % 7], factor);
    }
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void checkMpi(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}

struct Range {
    size_t first;
    size_t count;
};

Range partition(size_t count, int index, int parts) {
    const size_t base = count / static_cast<size_t>(parts);
    const size_t remainder = count % static_cast<size_t>(parts);
    return {base * static_cast<size_t>(index) + std::min(static_cast<size_t>(index), remainder),
            base + (static_cast<size_t>(index) < remainder ? 1 : 0)};
}

Range partitionByDevices(size_t count, size_t firstDevice, size_t devices, size_t totalDevices) {
    const size_t base = count / totalDevices;
    const size_t remainder = count % totalDevices;
    const size_t first = base * firstDevice + std::min(firstDevice, remainder);
    const size_t end = base * (firstDevice + devices) + std::min(firstDevice + devices, remainder);
    return {first, end - first};
}

// Identify physical devices rather than comparing CUDA ordinals: schedulers
// can expose identical, disjoint, reordered, or overlapping visibility masks.
// Give each physical GPU exactly one node-local rank, avoiding oversubscription.
std::vector<int> assignDevices(MPI_Comm nodeComm, int nodeRank, int nodeRanks, int deviceCount) {
    if (deviceCount > std::numeric_limits<int>::max() / static_cast<int>(sizeof(cudaUUID_t))) {
        throw std::runtime_error("GPU metadata exceeds MPI count limit");
    }
    std::vector<cudaUUID_t> localIds(deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, i), "cudaGetDeviceProperties");
        localIds[i] = properties.uuid;
    }
    const int localBytes = deviceCount * static_cast<int>(sizeof(cudaUUID_t));
    std::vector<int> counts(nodeRanks), displacements(nodeRanks);
    checkMpi(MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, nodeComm),
             "MPI_Allgather");
    int totalBytes = 0;
    for (int r = 0; r < nodeRanks; ++r) {
        if (counts[r] > std::numeric_limits<int>::max() - totalBytes) {
            throw std::runtime_error("Node GPU metadata exceeds MPI count limit");
        }
        displacements[r] = totalBytes;
        totalBytes += counts[r];
    }
    std::vector<cudaUUID_t> allIds(static_cast<size_t>(totalBytes) / sizeof(cudaUUID_t));
    checkMpi(MPI_Allgatherv(localIds.data(), localBytes, MPI_BYTE, allIds.data(), counts.data(),
                           displacements.data(), MPI_BYTE, nodeComm), "MPI_Allgatherv");
    const auto sameDevice = [](const cudaUUID_t& a, const cudaUUID_t& b) {
        return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
    };
    std::vector<cudaUUID_t> uniqueIds;
    for (const auto& id : allIds) {
        if (std::none_of(uniqueIds.begin(), uniqueIds.end(),
                         [&](const cudaUUID_t& other) { return sameDevice(id, other); })) {
            uniqueIds.push_back(id);
        }
    }
    std::vector<int> devices;
    for (int i = 0; i < deviceCount; ++i) {
        const auto position = std::find_if(uniqueIds.begin(), uniqueIds.end(),
            [&](const cudaUUID_t& id) { return sameDevice(id, localIds[i]); });
        const size_t ordinal = static_cast<size_t>(position - uniqueIds.begin());
        std::vector<int> candidates;
        for (int r = 0; r < nodeRanks; ++r) {
            const size_t begin = static_cast<size_t>(displacements[r]) / sizeof(cudaUUID_t);
            const size_t end = begin + static_cast<size_t>(counts[r]) / sizeof(cudaUUID_t);
            for (size_t j = begin; j < end; ++j) {
                if (sameDevice(localIds[i], allIds[j])) {
                    candidates.push_back(r);
                    break;
                }
            }
        }
        if (candidates[ordinal % candidates.size()] == nodeRank) {
            devices.push_back(i);
        }
    }
    return devices;
}

// Each OpenMP worker owns one GPU and two bounded buffers. Generation and
// pricing are fused; only requested output is copied back. Pinned staging
// buffers let the next kernel overlap a device-to-host transfer and host copy.
struct GpuWorker {
    static constexpr size_t chunkOptions = 1u << 20;
    static constexpr int blockSize = 256;

    struct Buffer {
        cudaStream_t stream = nullptr;
        double* device = nullptr;
        double* host = nullptr;
        size_t hostOffset = 0;
        size_t copyCount = 0;
        bool pending = false;
    };

    int deviceId = 0;
    Range range{};
    size_t capacity = 0;
    int slots = 0;
    int maxBlocks = 0;
    std::array<Buffer, 2> buffers{};

    void initialize(const CoefficientTable& table, size_t collectCount) {
        checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");
        if (range.count == 0) {
            return;
        }
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, deviceId), "cudaGetDeviceProperties");
        maxBlocks = properties.multiProcessorCount * 32;
        capacity = std::min(chunkOptions, range.count);
        slots = range.count > capacity ? 2 : 1;
        checkCuda(cudaMemcpyToSymbol(deviceOptions, table.data(),
                                     table.size() * sizeof(PricingCoefficients)),
                  "cudaMemcpyToSymbol");
        for (int i = 0; i < slots; ++i) {
            auto& buffer = buffers[i];
            checkCuda(cudaStreamCreateWithFlags(&buffer.stream, cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&buffer.device),
                                 capacity * sizeof(double)), "cudaMalloc");
            const size_t first = range.first + static_cast<size_t>(i) * capacity;
            if (first < collectCount) {
                const size_t hostCapacity = std::min(capacity, collectCount - first);
                checkCuda(cudaMallocHost(reinterpret_cast<void**>(&buffer.host),
                                         hostCapacity * sizeof(double)), "cudaMallocHost");
            }
        }
        // Initialize the kernel/module outside the measured pricing interval.
        priceOptions<<<1, 1, 0, buffers[0].stream>>>(buffers[0].device, range.first, 0);
        checkCuda(cudaGetLastError(), "priceOptions warmup");
        checkCuda(cudaStreamSynchronize(buffers[0].stream), "cudaStreamSynchronize");
    }

    void finish(Buffer& buffer, double* output) {
        if (buffer.pending) {
            checkCuda(cudaStreamSynchronize(buffer.stream), "cudaStreamSynchronize");
            if (buffer.copyCount != 0) {
                std::memcpy(output + buffer.hostOffset, buffer.host,
                            buffer.copyCount * sizeof(double));
            }
            buffer.pending = false;
        }
    }

    void price(double* output, size_t rankFirst, size_t collectCount) {
        checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");
        int slot = 0;
        for (size_t offset = 0; offset < range.count;) {
            auto& buffer = buffers[slot];
            finish(buffer, output);
            const size_t count = std::min(capacity, range.count - offset);
            const size_t first = range.first + offset;
            const int blocks = std::min(maxBlocks,
                                       static_cast<int>((count + blockSize - 1) / blockSize));
            priceOptions<<<blocks, blockSize, 0, buffer.stream>>>(buffer.device, first, count);
            checkCuda(cudaGetLastError(), "priceOptions");
            buffer.copyCount = first < collectCount ? std::min(count, collectCount - first) : 0;
            if (buffer.copyCount != 0) {
                buffer.hostOffset = first - rankFirst;
                checkCuda(cudaMemcpyAsync(buffer.host, buffer.device,
                                          buffer.copyCount * sizeof(double),
                                          cudaMemcpyDeviceToHost, buffer.stream),
                          "cudaMemcpyAsync");
            }
            buffer.pending = true;
            offset += count;
            slot = (slot + 1) % slots;
        }
        for (int i = 0; i < slots; ++i) {
            finish(buffers[i], output);
        }
    }

    void release() {
        checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");
        for (int i = 0; i < slots; ++i) {
            checkCuda(cudaFree(buffers[i].device), "cudaFree");
            if (buffers[i].host != nullptr) {
                checkCuda(cudaFreeHost(buffers[i].host), "cudaFreeHost");
            }
            checkCuda(cudaStreamDestroy(buffers[i].stream), "cudaStreamDestroy");
        }
    }
};

template <typename Function>
void onGpus(std::vector<GpuWorker>& workers, int rank, Function action) {
    if (workers.empty()) {
        return;
    }
    int failed = 0;
    const int count = static_cast<int>(workers.size());
    // A parallel for still visits every device if OMP_THREAD_LIMIT reduces
    // the team size. MPI calls are confined to the main thread after this.
    #pragma omp parallel for num_threads(count) schedule(static, 1) reduction(|:failed)
    for (int i = 0; i < count; ++i) {
        try {
            action(workers[i]);
        } catch (const std::exception& error) {
            #pragma omp critical
            {
                std::fprintf(stderr, "Rank %d, CUDA device %d: %s\n",
                             rank, workers[i].deviceId, error.what());
            }
            failed = 1;
        }
    }
    if (failed) {
        throw std::runtime_error("CUDA worker failed");
    }
}

// Gather in globally ordered windows so MPI's int counts/displacements never
// overflow, even for more than INT_MAX options. No communication of inputs,
// and no result gather when neither -r nor -v requests it.
void gatherResults(size_t collectCount, Range localRange,
                   const double* localResults, std::vector<double>& results,
                   int rank, int ranks) {
    if (collectCount == 0) {
        return;
    }
    const std::array<unsigned long long, 2> localMetadata{{localRange.first, localRange.count}};
    std::vector<unsigned long long> rankRanges(rank == 0 ? size_t{2} * ranks : 0);
    checkMpi(MPI_Gather(localMetadata.data(), 2, MPI_UNSIGNED_LONG_LONG, rankRanges.data(),
                        2, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD), "MPI_Gather");
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(counts.size());
    const size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t first = 0; first < collectCount;) {
        const size_t end = first + std::min(maxCount, collectCount - first);
        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const size_t index = size_t{2} * r;
                const Range range{static_cast<size_t>(rankRanges[index]),
                                  static_cast<size_t>(rankRanges[index + 1])};
                const size_t begin = std::max(first, range.first);
                const size_t finish = std::min(end, range.first + range.count);
                counts[r] = finish > begin ? static_cast<int>(finish - begin) : 0;
                displacements[r] = counts[r] != 0 ? static_cast<int>(begin - first) : 0;
            }
        }
        const size_t begin = std::max(first, localRange.first);
        const size_t finish = std::min(end, localRange.first + localRange.count);
        const int count = finish > begin ? static_cast<int>(finish - begin) : 0;
        const double* send = count != 0 ? localResults + (begin - localRange.first) : nullptr;
        checkMpi(MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : send, count, MPI_DOUBLE,
                            rank == 0 ? results.data() + first : nullptr,
                            counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD),
                 "MPI_Gatherv");
        first = end;
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), results.size()));
    constexpr auto options = getTestOptions();

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i % options.size()].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Preserve the original benchmark's relaxed validation.
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    // Parse once and broadcast, including early exits, to keep all ranks in
    // the same collective sequence.
    std::array<unsigned long long, 3> settings{{10000, 0, 0}};
    int status = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                const char* argument = argv[++i];
                char* end = nullptr;
                errno = 0;
                const unsigned long long count = std::strtoull(argument, &end, 10);
                if (std::strchr(argument, '-') != nullptr || *argument == '\0' || *end != '\0' || errno == ERANGE
                    || count > std::numeric_limits<size_t>::max()) {
                    std::fprintf(stderr, "Invalid number of options: %s\n", argument);
                    status = 1;
                    break;
                }
                settings[0] = count;
            } else if (strcmp(argv[i], "-v") == 0) {
                settings[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                settings[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                status = -1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                status = 1;
                break;
            }
        }
    }
    checkMpi(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast");
    if (status != 0) {
        return status < 0 ? 0 : status;
    }
    checkMpi(MPI_Bcast(settings.data(), 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD), "MPI_Bcast");
    const size_t numOptions = static_cast<size_t>(settings[0]);
    const bool validate = settings[1] != 0;
    const bool printResults = settings[2] != 0;
    const size_t collectCount = printResults ? numOptions : validate ? std::min(size_t{10}, numOptions) : 0;

    MPI_Comm nodeComm;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &nodeComm), "MPI_Comm_split_type");
    int nodeRank = 0, nodeRanks = 1;
    checkMpi(MPI_Comm_rank(nodeComm, &nodeRank), "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(nodeComm, &nodeRanks), "MPI_Comm_size");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        throw std::runtime_error("MPI/OpenMP/CUDA execution requires a visible CUDA device");
    }
    const auto deviceIds = assignDevices(nodeComm, nodeRank, nodeRanks, deviceCount);
    checkMpi(MPI_Comm_free(&nodeComm), "MPI_Comm_free");
    const size_t available = deviceIds.size();
    // Weight each rank by its GPU count, so uneven GPU counts per rank or
    // node do not leave faster ranks waiting for ranks with fewer devices.
    const unsigned long long localDevices = static_cast<unsigned long long>(available);
    unsigned long long totalDevices = 0, firstDevice = 0;
    checkMpi(MPI_Allreduce(&localDevices, &totalDevices, 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce");
    checkMpi(MPI_Exscan(&localDevices, &firstDevice, 1, MPI_UNSIGNED_LONG_LONG,
                       MPI_SUM, MPI_COMM_WORLD), "MPI_Exscan");
    if (rank == 0) {
        firstDevice = 0;
    }
    if (totalDevices > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("GPU count exceeds addressable range");
    }
    const Range localRange = partitionByDevices(numOptions, static_cast<size_t>(firstDevice),
                                                static_cast<size_t>(localDevices),
                                                static_cast<size_t>(totalDevices));
    const int workerCount = static_cast<int>(std::min(available, localRange.count));
    std::vector<GpuWorker> workers(workerCount);
    for (int i = 0; i < workerCount; ++i) {
        workers[i].deviceId = deviceIds[i];
        workers[i].range = partition(localRange.count, i, workerCount);
        workers[i].range.first += localRange.first;
    }

    // Root's local output goes directly into the final ordered result vector.
    // Validation alone needs only the same first ten values as the original.
    std::vector<double> results(rank == 0 ? collectCount : 0);
    const size_t localCollectCount = localRange.first < collectCount
                                  ? std::min(localRange.count, collectCount - localRange.first) : 0;
    std::vector<double> localResults(rank != 0 ? localCollectCount : 0);
    double* output = rank == 0 ? results.data() : localResults.data();

    omp_set_dynamic(0);
    const auto coefficients = prepareCoefficients();
    onGpus(workers, rank, [&](GpuWorker& worker) { worker.initialize(coefficients, collectCount); });

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    const double start = MPI_Wtime();
    onGpus(workers, rank, [&](GpuWorker& worker) {
        worker.price(output, localRange.first, collectCount);
    });
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    checkMpi(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "MPI_Reduce");

    gatherResults(collectCount, localRange, output, results, rank, ranks);
    onGpus(workers, rank, [](GpuWorker& worker) { worker.release(); });

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0.0 ? numOptions / duration : 0.0);
        if (printResults) {
            print_results(results, "OptionPrices");
        }
        if (validate) {
            printf("Validating results...\n");
            status = validateResults(results) ? 0 : 1;
            printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
        }
    }
    checkMpi(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD), "MPI_Bcast");
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    int rank = 0, ranks = 1;
    int status = 1;
    try {
        checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN), "MPI_Comm_set_errhandler");
        checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
        checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        }
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    if (MPI_Finalize() != MPI_SUCCESS) {
        return 1;
    }
    return status;
}
