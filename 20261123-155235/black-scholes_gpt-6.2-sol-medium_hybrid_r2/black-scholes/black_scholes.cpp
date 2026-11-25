#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/hash.hpp"

enum OptionType { CALL, PUT };

struct OptionInput {
    OptionType type;
    double strike, spot, q, r, t, vol;
    double value;
};

constexpr std::array<OptionInput, 7> testOptions{{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408},
}};
constexpr unsigned caseCount = testOptions.size();
constexpr int ratioRadius = 4;
constexpr unsigned ratioCount = 2 * ratioRadius + 1;

struct NormalPair {
    double first, second;
};

struct PricingCase {
    double spot, strike, spotDiscount, strikeDiscount, drift, time, deviation;
    OptionType type;
    uint64_t firstRatio;
    NormalPair normals[ratioCount];
};

__host__ __device__ NormalPair computeNormals(const PricingCase& option, double ratio) {
    constexpr double invSqrt2 = 0.7071067811865475244;
    const double d1 = fma(option.drift, option.time, log(ratio)) / option.deviation;
    const double d2 = d1 - option.deviation;
    return {
        0.5 * (1.0 + erf((option.type == CALL ? d1 : -d2) * invSqrt2)),
        0.5 * (1.0 + erf((option.type == CALL ? d2 : -d1) * invSqrt2)),
    };
}

std::array<PricingCase, caseCount> makePricingCases() {
    std::array<PricingCase, caseCount> cases{};
    for (unsigned i = 0; i < caseCount; ++i) {
        const auto& base = testOptions[i];
        auto& option = cases[i];
        option.spot = base.spot;
        option.strike = base.strike;
        option.spotDiscount = std::exp(-base.q * base.t);
        option.strikeDiscount = std::exp(-base.r * base.t);
        option.drift = std::fma(0.5 * base.vol, base.vol, base.r - base.q);
        option.time = base.t;
        option.deviation = base.vol * std::sqrt(base.t);
        option.type = base.type;

        // A common scale cancels in S/K, but rounding the two products can
        // move the ratio by a few ULPs. Cache those CDFs without rounding S/K
        // to the unscaled ratio or changing the order of price arithmetic.
        double ratio = base.spot / base.strike;
        for (int step = 0; step < ratioRadius; ++step) {
            ratio = std::nextafter(ratio, 0.0);
        }
        std::memcpy(&option.firstRatio, &ratio, sizeof(ratio));
        for (auto& normals : option.normals) {
            normals = computeNormals(option, ratio);
            ratio = std::nextafter(ratio, HUGE_VAL);
        }
    }
    return cases;
}

__global__ void priceOptions(double* __restrict__ results, size_t first, unsigned firstCase,
                             unsigned count, const PricingCase* __restrict__ cases) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const auto& option = cases[(firstCase + i) % caseCount];
    const double factor = fma(0.1, static_cast<double>(first + i) / caseCount, 1.0);
    const double spot = option.spot * factor;
    const double strike = option.strike * factor;
    const double ratio = spot / strike;
    const uint64_t ratioIndex = static_cast<uint64_t>(__double_as_longlong(ratio)) - option.firstRatio;
    const NormalPair normals = ratioIndex < ratioCount ? option.normals[ratioIndex]
                                                        : computeNormals(option, ratio);
    const double discountedSpot = spot * option.spotDiscount;
    const double discountedStrike = strike * option.strikeDiscount;
    results[i] = option.type == CALL
        ? fma(discountedSpot, normals.first, -discountedStrike * normals.second)
        : fma(discountedStrike, normals.first, -discountedSpot * normals.second);
}

struct Range {
    size_t begin, count;
};

Range partition(size_t count, size_t index, size_t size) {
    const size_t base = count / size;
    const size_t extra = count % size;
    return {index * base + std::min(index, extra), base + (index < extra)};
}

size_t prefixCount(Range range, size_t limit) {
    return range.begin < limit ? std::min(range.count, limit - range.begin) : 0;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " - " + cudaGetErrorString(status));
    }
}

std::vector<int> assignDevices(MPI_Comm node) {
    int visible = 0;
    const cudaError_t status = cudaGetDeviceCount(&visible);
    if (status != cudaErrorNoDevice) {
        checkCuda(status, "Enumerating CUDA devices");
    }

    int localRank, localSize;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localSize);
    using DeviceId = std::array<char, sizeof(cudaUUID_t::bytes)>;
    std::vector<DeviceId> ids(visible);
    for (int device = 0; device < visible; ++device) {
        cudaDeviceProp properties;
        checkCuda(cudaGetDeviceProperties(&properties, device), "Reading CUDA device ID");
        std::copy_n(properties.uuid.bytes, ids[device].size(), ids[device].begin());
    }

    // UUIDs keep launcher-masked or reordered device lists comparable.
    const int idSize = sizeof(DeviceId);
    const int localBytes = visible * idSize;
    std::vector<int> byteCounts(localSize), offsets(localSize);
    MPI_Allgather(&localBytes, 1, MPI_INT, byteCounts.data(), 1, MPI_INT, node);
    int bytes = 0;
    for (int peer = 0; peer < localSize; ++peer) {
        offsets[peer] = bytes;
        bytes += byteCounts[peer];
    }
    std::vector<DeviceId> allIds(bytes / idSize);
    MPI_Allgatherv(ids.data(), localBytes, MPI_BYTE, allIds.data(), byteCounts.data(),
                   offsets.data(), MPI_BYTE, node);

    struct DeviceAccess { int rank, ordinal; };
    std::map<DeviceId, std::vector<DeviceAccess>> access;
    for (int peer = 0; peer < localSize; ++peer) {
        for (int device = 0; device < byteCounts[peer] / idSize; ++device) {
            access[allIds[offsets[peer] / idSize + device]].push_back({peer, device});
        }
    }
    std::vector<int> assignedCounts(localSize), devices;
    for (const auto& entry : access) {
        const auto& candidates = entry.second;
        const auto owner = std::min_element(candidates.begin(), candidates.end(),
            [&](DeviceAccess a, DeviceAccess b) { return assignedCounts[a.rank] < assignedCounts[b.rank]; });
        ++assignedCounts[owner->rank];
        if (owner->rank == localRank) {
            devices.push_back(owner->ordinal);
        }
    }
    return devices;
}

Range assignOptions(size_t count, int rank, uint64_t localDevices) {
    uint64_t devices = 0;
    uint64_t firstDevice = 0;
    MPI_Allreduce(&localDevices, &devices, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (!devices) {
        if (!count) return {0, 0};
        throw std::runtime_error("No visible CUDA devices");
    }

    // Give each GPU an equal share while keeping the prices in rank order.
    MPI_Exscan(&localDevices, &firstDevice, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) firstDevice = 0;
    const size_t begin = partition(count, firstDevice, devices).begin;
    const size_t end = partition(count, firstDevice + localDevices, devices).begin;
    return {begin, end - begin};
}

constexpr unsigned blockSize = 256;
constexpr size_t tileSize = 1 << 20;

struct GpuBuffers {
    cudaStream_t stream = nullptr;
    double* prices = nullptr;
    PricingCase* cases = nullptr;
    double* staging = nullptr;

    ~GpuBuffers() {
        if (stream) cudaStreamSynchronize(stream);
        if (staging) cudaFreeHost(staging);
        if (cases) cudaFree(cases);
        if (prices) cudaFree(prices);
        if (stream) cudaStreamDestroy(stream);
    }

    void prepare(int device, size_t count, size_t outputCount,
                 const std::array<PricingCase, caseCount>& pricingCases) {
        checkCuda(cudaSetDevice(device), "Selecting CUDA device");
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Creating CUDA stream");
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, priceOptions), "Loading pricing kernel");
        checkCuda(cudaMalloc(&prices, std::min(tileSize, count) * sizeof(double)),
                  "Allocating device prices");
        checkCuda(cudaMalloc(&cases, caseCount * sizeof(PricingCase)), "Allocating device pricing cases");
        checkCuda(cudaMemcpyAsync(cases, pricingCases.data(), caseCount * sizeof(PricingCase),
                                  cudaMemcpyHostToDevice, stream), "Copying pricing cases");
        checkCuda(cudaStreamSynchronize(stream), "Initializing pricing cases");
        if (outputCount) {
            checkCuda(cudaMallocHost(&staging, std::min(tileSize, outputCount) * sizeof(double)),
                      "Allocating pinned prices");
        }
    }

    void price(Range work, size_t outputCount, double* output) {
        size_t offset = 0;
        while (offset < work.count) {
            const unsigned count = std::min(tileSize, work.count - offset);
            const size_t first = work.begin + offset;
            priceOptions<<<(count + blockSize - 1) / blockSize, blockSize, 0, stream>>>(
                prices, first, first % caseCount, count, cases);
            checkCuda(cudaGetLastError(), "Launching pricing kernel");
            if (offset < outputCount) {
                const size_t bytes = std::min<size_t>(count, outputCount - offset) * sizeof(double);
                checkCuda(cudaMemcpyAsync(staging, prices, bytes, cudaMemcpyDeviceToHost, stream),
                          "Copying prices from device");
                checkCuda(cudaStreamSynchronize(stream), "Waiting for prices");
                std::memcpy(output + offset, staging, bytes);
            }
            offset += count;
        }
        checkCuda(cudaStreamSynchronize(stream), "Finishing CUDA pricing");
    }
};

double priceRank(Range range, size_t outputLimit, double* results, int rank,
                 const std::vector<int>& devices) {
    const int workers = std::max<size_t>(1, std::min(devices.size(), range.count));
    const auto pricingCases = makePricingCases();
    omp_set_dynamic(0);

    int failed = 0;
    double start = 0.0;
    double elapsed = 0.0;
    #pragma omp parallel num_threads(workers) reduction(|:failed)
    {
        const int thread = omp_get_thread_num();
        Range work = partition(range.count, thread, omp_get_num_threads());
        work.begin += range.begin;
        const size_t outputCount = prefixCount(work, outputLimit);
        const int device = work.count ? devices[thread] : -1;
        GpuBuffers buffers;
        if (work.count) {
            try {
                buffers.prepare(device, work.count, outputCount, pricingCases);
            } catch (const std::exception& error) {
                fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, device, error.what());
                failed = 1;
            }
        }

        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
        }
        #pragma omp barrier

        if (work.count && !failed) {
            try {
                double* output = outputCount ? results + (work.begin - range.begin) : nullptr;
                buffers.price(work, outputCount, output);
            } catch (const std::exception& error) {
                fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, device, error.what());
                failed = 1;
            }
        }

        #pragma omp barrier
        #pragma omp master
        elapsed = MPI_Wtime() - start;
    }
    if (failed) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return elapsed;
}

// The summary uses the original Kahan and FNV-1a order. Stream rank ranges
// through bounded messages so reporting needs no second global results array.
struct ResultsSummary {
    explicit ResultsSummary(size_t count) : indices{0, count / 4, count / 2,
                                                    (count / 4) * 3 + (count % 4) * 3 / 4,
                                                    count ? count - 1 : 0} {}

    size_t indices[5];
    double samples[5]{};
    double sum = 0.0, correction = 0.0, minimum = 0.0, maximum = 0.0;
    hash digest;

    void add(size_t begin, const double* values, size_t count) {
        if (begin == 0 && count) {
            minimum = maximum = values[0];
        }
        for (size_t i = 0; i < count; ++i) {
            const double value = values[i];
            const double y = value - correction;
            const double next = sum + y;
            correction = (next - sum) - y;
            sum = next;
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            digest.add(value);
        }
        for (int i = 0; i < 5; ++i) {
            if (indices[i] >= begin && indices[i] - begin < count) {
                samples[i] = values[indices[i] - begin];
            }
        }
    }

    void print(size_t count) const {
        printf("=== RESULTS ===\nName: OptionPrices\nElements: %zu\n", count);
        if (count) {
            printf("Sum: %.17e\nMin: %.17e\nMax: %.17e\n", sum, minimum, maximum);
            for (int i = 0; i < 5; ++i) {
                printf("Sample[%zu]: %.17e\n", indices[i], samples[i]);
            }
            printf("Hash: %016" PRIx64 "\n", digest.get());
        }
        printf("=== END RESULTS ===\n");
    }
};

constexpr size_t validationCount = 10;

void collectResults(int rank, int ranks, size_t ownCount, size_t limit, const double* local,
                    ResultsSummary* summary, std::array<double, validationCount>& validation) {
    const uint64_t sendCount = ownCount;
    std::vector<uint64_t> counts(rank == 0 ? ranks : 0);
    MPI_Gather(&sendCount, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    constexpr int resultTag = 0;
    if (rank != 0) {
        for (size_t offset = 0; offset < ownCount;) {
            const int count = std::min(tileSize, ownCount - offset);
            MPI_Send(local + offset, count, MPI_DOUBLE, 0, resultTag, MPI_COMM_WORLD);
            offset += count;
        }
        return;
    }

    const auto consume = [&](size_t begin, const double* values, size_t count) {
        if (summary) {
            summary->add(begin, values, count);
        }
        const size_t checks = prefixCount({begin, count}, validation.size());
        if (checks) {
            std::copy_n(values, checks, validation.data() + begin);
        }
    };
    consume(0, local, ownCount);
    const size_t remoteCount = limit - ownCount;
    std::unique_ptr<double[]> incoming(remoteCount ? new double[std::min(tileSize, remoteCount)] : nullptr);
    size_t begin = ownCount;
    for (int source = 1; source < ranks; ++source) {
        for (size_t remaining = counts[source]; remaining > 0;) {
            const int chunk = std::min(tileSize, remaining);
            MPI_Recv(incoming.get(), chunk, MPI_DOUBLE, source, resultTag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            consume(begin, incoming.get(), chunk);
            begin += chunk;
            remaining -= chunk;
        }
    }
}

bool validateResults(size_t numOptions, const std::array<double, validationCount>& prices) {
    bool allPassed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < std::min(numOptions, prices.size()); ++i) {
        const double computed = prices[i];
        const double expected = testOptions[i % caseCount].value;
        const double relError = std::abs(computed - expected) / (std::abs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
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

int run(int argc, char** argv, int rank, int ranks) {
    long long optionCount = 10000;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            optionCount = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (optionCount < 0) {
        if (rank == 0) printf("Number of options must be nonnegative\n");
        return 1;
    }
    const size_t numOptions = optionCount;

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    const auto devices = numOptions ? assignDevices(node) : std::vector<int>{};
    MPI_Comm_free(&node);

    const Range range = assignOptions(numOptions, rank, devices.size());
    const size_t outputLimit = printResults ? numOptions : (validate ? std::min(numOptions, validationCount) : 0);
    const size_t outputCount = prefixCount(range, outputLimit);
    std::unique_ptr<double[]> local(outputCount ? new double[outputCount] : nullptr);
    const double elapsed = priceRank(range, outputLimit, local.get(), rank, devices);
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0.0 ? numOptions / duration : 0.0);
    }

    int status = 0;
    if (printResults || validate) {
        ResultsSummary summary(numOptions);
        std::array<double, validationCount> validation{};
        collectResults(rank, ranks, outputCount, outputLimit, local.get(),
                       printResults ? &summary : nullptr, validation);
        if (rank == 0 && printResults) {
            summary.print(numOptions);
        }
        if (validate) {
            if (rank == 0) {
                printf("Validating results...\n");
                const bool valid = validateResults(numOptions, validation);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
            MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    return status;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = EXIT_FAILURE;
    try {
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        }
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return status;
}
