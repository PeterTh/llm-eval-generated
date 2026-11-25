#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

struct OptionInput {
    bool call;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    double value;
};

constexpr OptionInput testOptions[] = {
    {true,   40.00,  42.00, 0.04, 0.08, 0.75, 0.35, 5.0975},
    {true,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15, 0.0205},
    {true,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {true,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413},
    {false, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15, 9.9210},
    {false, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734},
    {false, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408},
};
constexpr unsigned optionKinds = sizeof(testOptions) / sizeof(testOptions[0]);
constexpr int threadsPerBlock = 256;
constexpr size_t batchSize = 1 << 20;

struct PricingInputs {
    struct Parameters {
        double strike;
        double spot;
        double drift;
        double volSqrtTime;
        double strikeDiscount;
        double spotDiscount;
        double sign;
    } options[optionKinds];
};

PricingInputs makePricingInputs() {
    // The generated options vary only in spot and strike. Compute the
    // rate, maturity, and volatility terms once for each base option.
    PricingInputs inputs{};
    for (unsigned i = 0; i < optionKinds; ++i) {
        const auto& option = testOptions[i];
        inputs.options[i] = {
            option.strike,
            option.spot,
            (option.r - option.q + 0.5 * option.vol * option.vol) * option.t,
            option.vol * std::sqrt(option.t),
            std::exp(-option.r * option.t),
            std::exp(-option.q * option.t),
            option.call ? 1.0 : -1.0,
        };
    }
    return inputs;
}

__device__ double cumulativeNormal(double x) {
    constexpr double sqrtHalf = 0.7071067811865475244;
    return 0.5 * (1.0 + erf(x * sqrtHalf));
}

__global__ void priceOptions(PricingInputs inputs, size_t first, unsigned firstKind,
                             unsigned count, double* results) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const auto& option = inputs.options[(firstKind + i) % optionKinds];
    const double factor = 1.0 + 0.1 * (static_cast<double>(first + i) / optionKinds);
    const double spot = option.spot * factor;
    const double strike = option.strike * factor;
    const double d1 = (log(spot / strike) + option.drift) / option.volSqrtTime;
    const double d2 = d1 - option.volSqrtTime;
    const double spotTerm = spot * option.spotDiscount * cumulativeNormal(option.sign * d1);
    const double strikeTerm = strike * option.strikeDiscount * cumulativeNormal(option.sign * d2);
    results[i] = option.sign > 0.0 ? spotTerm - strikeTerm : strikeTerm - spotTerm;
}

// The boundary after part partitions of a balanced contiguous split.
size_t boundary(size_t size, size_t parts, size_t part) {
    return (size / parts) * part + std::min(size % parts, part);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<int> assignedDevices(MPI_Comm node, int nodeRank, int nodeRanks) {
    int visibleDevices = 0;
    const auto status = cudaGetDeviceCount(&visibleDevices);
    if (status != cudaErrorNoDevice) {
        checkCuda(status, "cudaGetDeviceCount");
    }

    // CUDA ordinals are process-local. Match UUIDs across ranks so both a
    // node-wide device list and scheduler-provided per-rank masks work.
    using DeviceId = std::array<char, sizeof(cudaUUID_t::bytes)>;
    std::vector<DeviceId> localIds(visibleDevices);
    for (int i = 0; i < visibleDevices; ++i) {
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, i), "cudaGetDeviceProperties");
        std::memcpy(localIds[i].data(), properties.uuid.bytes, localIds[i].size());
    }
    const int localBytes = visibleDevices * sizeof(DeviceId);
    std::vector<int> sizes(nodeRanks), offsets(nodeRanks);
    MPI_Allgather(&localBytes, 1, MPI_INT, sizes.data(), 1, MPI_INT, node);
    int totalBytes = 0;
    for (int i = 0; i < nodeRanks; ++i) {
        offsets[i] = totalBytes;
        totalBytes += sizes[i];
    }
    std::vector<DeviceId> allIds(totalBytes / sizeof(DeviceId));
    MPI_Allgatherv(localIds.data(), localBytes, MPI_BYTE, allIds.data(),
                   sizes.data(), offsets.data(), MPI_BYTE, node);

    struct Location { int rank; int ordinal; };
    std::map<DeviceId, std::vector<Location>> locations;
    for (int i = 0; i < nodeRanks; ++i) {
        const size_t first = offsets[i] / sizeof(DeviceId);
        const int count = sizes[i] / sizeof(DeviceId);
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            locations[allIds[first + ordinal]].push_back({i, ordinal});
        }
    }

    std::vector<int> devices;
    std::vector<int> assigned(nodeRanks);
    for (const auto& entry : locations) {
        const auto& candidates = entry.second;
        const auto owner = std::min_element(candidates.begin(), candidates.end(),
            [&assigned](Location a, Location b) { return assigned[a.rank] < assigned[b.rank]; });
        ++assigned[owner->rank];
        if (owner->rank == nodeRank) {
            devices.push_back(owner->ordinal);
        }
    }
    // Ranks without a GPU participate in MPI with an empty option range.
    return devices;
}

struct DeviceBuffer {
    cudaStream_t stream = nullptr;
    double* device = nullptr;
    double* staging = nullptr;
    double* destination = nullptr;
    size_t pending = 0;

    void allocate(size_t capacity, size_t copyCapacity) {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        checkCuda(cudaMalloc(&device, capacity * sizeof(double)), "cudaMalloc");
        if (copyCapacity) {
            checkCuda(cudaMallocHost(&staging, copyCapacity * sizeof(double)), "cudaMallocHost");
        }
    }

    void finishCopy() {
        if (pending) {
            checkCuda(cudaStreamSynchronize(stream), "copy option prices from GPU");
            std::memcpy(destination, staging, pending * sizeof(double));
            pending = 0;
        }
    }

    ~DeviceBuffer() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
        if (device) {
            cudaFree(device);
        }
        if (staging) {
            cudaFreeHost(staging);
        }
    }
};

struct DeviceWork {
    int ordinal = -1;
    size_t first = 0;
    size_t count = 0;
    size_t reportCount = 0;
    double* output = nullptr;
    std::exception_ptr error;
    std::array<DeviceBuffer, 2> buffers;

    void prepare() {
        checkCuda(cudaSetDevice(ordinal), "cudaSetDevice");
        cudaFuncAttributes attributes{};
        checkCuda(cudaFuncGetAttributes(&attributes, priceOptions), "load pricing kernel");
        buffers[0].allocate(std::min(count, batchSize), std::min(reportCount, batchSize));
        // A second stream lets the next kernel run while the preceding batch is copied.
        if (reportCount > batchSize) {
            const size_t secondCapacity = std::min(count - batchSize, batchSize);
            buffers[1].allocate(secondCapacity, std::min(reportCount - batchSize, batchSize));
        }
    }

    void run(const PricingInputs& inputs) {
        checkCuda(cudaSetDevice(ordinal), "cudaSetDevice");
        size_t slot = 0;
        for (size_t offset = 0; offset < count;) {
            auto& buffer = buffers[slot];
            buffer.finishCopy();
            const unsigned size = std::min(count - offset, batchSize);
            const size_t index = first + offset;
            const unsigned blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
            priceOptions<<<blocks, threadsPerBlock, 0, buffer.stream>>>(
                inputs, index, index % optionKinds, size, buffer.device);
            checkCuda(cudaGetLastError(), "launch pricing kernel");

            if (offset < reportCount) {
                buffer.pending = std::min<size_t>(size, reportCount - offset);
                buffer.destination = output + offset;
                checkCuda(cudaMemcpyAsync(buffer.staging, buffer.device,
                                          buffer.pending * sizeof(double),
                                          cudaMemcpyDeviceToHost, buffer.stream),
                          "queue option price copy");
            }
            offset += size;
            if (buffers[1].stream) {
                slot ^= 1;
            }
        }

        for (auto& buffer : buffers) {
            if (buffer.pending) {
                buffer.finishCopy();
            } else if (buffer.stream) {
                checkCuda(cudaStreamSynchronize(buffer.stream), "run pricing kernel");
            }
        }
    }

    ~DeviceWork() {
        if (count) {
            cudaSetDevice(ordinal);
        }
    }
};

[[noreturn]] void abortBenchmark(int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkWorkers(const std::vector<DeviceWork>& work, int rank) {
    for (const auto& device : work) {
        if (device.error) {
            try {
                std::rethrow_exception(device.error);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "CUDA device %d on ", device.ordinal);
                abortBenchmark(rank, error.what());
            }
        }
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min<size_t>(10, results.size());

    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % optionKinds].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);

        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
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

struct Settings {
    enum Mode : uint64_t { Validate = 1, Print = 2, Help = 4, Invalid = 8 };

    uint64_t count = 10000;
    uint64_t mode = 0;
};

Settings parseArguments(int argc, char** argv) {
    Settings settings;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            errno = 0;
            const long long count = std::strtoll(value, nullptr, 10);
            if (errno == ERANGE || count < 0 ||
                static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max()) {
                std::printf("Invalid number of options: %s\n", value);
                settings.mode |= Settings::Invalid;
                break;
            }
            settings.count = count;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            settings.mode |= Settings::Validate;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            settings.mode |= Settings::Print;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            settings.mode |= Settings::Help;
            break;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            settings.mode |= Settings::Invalid;
            break;
        }
    }
    return settings;
}

void gatherResults(std::vector<double>& results, size_t reportCount, size_t localFirst,
                   uint64_t localCount, int rank, int ranks) {
    if (!reportCount) {
        return;
    }

    std::vector<uint64_t> counts(rank == 0 ? ranks : 0);
    MPI_Gather(&localCount, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    std::vector<int> lengths(counts.size()), offsets(counts.size());

    // Use windows so MPI implementations with int counts can also report large runs.
    const auto windowSize = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t first = 0; first < reportCount;) {
        const size_t last = first + std::min(reportCount - first, windowSize);
        if (rank == 0) {
            size_t sourceFirst = 0;
            for (int i = 0; i < ranks; ++i) {
                const size_t sourceLast = sourceFirst + counts[i];
                const size_t begin = std::max(sourceFirst, first);
                const size_t end = std::min(sourceLast, last);
                lengths[i] = begin < end ? end - begin : 0;
                offsets[i] = lengths[i] ? begin - first : 0;
                sourceFirst = sourceLast;
            }
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, results.data() + first,
                        lengths.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            const size_t begin = std::max(localFirst, first);
            const size_t end = std::min(localFirst + localCount, last);
            const int count = begin < end ? end - begin : 0;
            const double* source = count ? results.data() + (begin - localFirst) : nullptr;
            MPI_Gatherv(source, count, MPI_DOUBLE, nullptr, nullptr, nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        first = last;
    }
}

int runBenchmark(Settings settings, int rank, int ranks) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nodeRank, nodeRanks;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_size(node, &nodeRanks);
    const auto devices = assignedDevices(node, nodeRank, nodeRanks);
    MPI_Comm_free(&node);
    const uint64_t deviceCount = devices.size();

    uint64_t precedingDevices = 0;
    uint64_t totalDevices = 0;
    MPI_Exscan(&deviceCount, &precedingDevices, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        precedingDevices = 0;
    }
    MPI_Allreduce(&deviceCount, &totalDevices, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (!totalDevices) {
        throw std::runtime_error("No CUDA devices are visible to the MPI job");
    }
    const size_t numOptions = settings.count;
    const size_t reportCount = settings.mode & Settings::Print ? numOptions
        : settings.mode & Settings::Validate ? std::min<size_t>(numOptions, 10) : 0;
    const size_t localFirst = boundary(numOptions, totalDevices, precedingDevices);
    const size_t localLast = boundary(numOptions, totalDevices, precedingDevices + deviceCount);
    const size_t localReport = std::min(localLast, reportCount) - std::min(localFirst, reportCount);
    std::vector<double> results(rank == 0 ? reportCount : localReport);

    std::vector<DeviceWork> work(deviceCount);
    for (size_t i = 0; i < deviceCount; ++i) {
        auto& device = work[i];
        device.ordinal = devices[i];
        device.first = boundary(numOptions, totalDevices, precedingDevices + i);
        const size_t last = boundary(numOptions, totalDevices, precedingDevices + i + 1);
        device.count = last - device.first;
        device.reportCount = std::min(last, reportCount) - std::min(device.first, reportCount);
        if (device.reportCount) {
            device.output = results.data() + (device.first - localFirst);
        }
    }
    const auto inputs = makePricingInputs();
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", settings.mode & Settings::Validate ? "enabled" : "disabled");
        std::printf("Pricing options...\n");
    }

    double start = 0.0, elapsed = 0.0;
    const int workers = std::max<size_t>(1, std::count_if(work.begin(), work.end(),
        [](const DeviceWork& device) { return device.count != 0; }));
    #pragma omp parallel num_threads(workers)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < work.size(); ++i) {
            auto& device = work[i];
            if (device.count) {
                try {
                    device.prepare();
                } catch (...) {
                    device.error = std::current_exception();
                }
            }
        }

        #pragma omp master
        {
            checkWorkers(work, rank);
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (size_t i = 0; i < work.size(); ++i) {
            auto& device = work[i];
            if (device.count) {
                try {
                    device.run(inputs);
                } catch (...) {
                    device.error = std::current_exception();
                }
            }
        }

        #pragma omp master
        elapsed = MPI_Wtime() - start;
    }
    checkWorkers(work, rank);

    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        std::printf("Options per second: %.0f\n", duration > 0.0 ? numOptions / duration : 0.0);
    }
    gatherResults(results, reportCount, localFirst, localReport, rank, ranks);

    int status = 0;
    if (rank == 0) {
        if (settings.mode & Settings::Print) {
            print_results(results, "OptionPrices");
        }
        if (settings.mode & Settings::Validate) {
            std::printf("Validating results...\n");
            const bool valid = validateResults(results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    if (settings.mode & Settings::Validate) {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortBenchmark(rank, "MPI_THREAD_FUNNELED is required");
    }

    std::array<uint64_t, 2> args{};
    if (rank == 0) {
        const auto settings = parseArguments(argc, argv);
        args = {settings.count, settings.mode};
    }
    MPI_Bcast(args.data(), args.size(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const Settings settings{args[0], args[1]};

    int status = 0;
    try {
        if (settings.mode & (Settings::Help | Settings::Invalid)) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            status = settings.mode & Settings::Invalid ? 1 : 0;
        } else {
            status = runBenchmark(settings, rank, ranks);
        }
    } catch (const std::exception& error) {
        abortBenchmark(rank, error.what());
    }
    MPI_Finalize();
    return status;
}
