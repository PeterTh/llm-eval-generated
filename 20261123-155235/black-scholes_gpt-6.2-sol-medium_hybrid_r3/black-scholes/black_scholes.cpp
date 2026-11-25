#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

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
constexpr int optionTypes = testOptions.size();

struct PricingTerms {
    double spot;
    double strike;
    double drift;
    double volTime;
    double spotDiscount;
    double strikeDiscount;
    bool put;
};

std::array<PricingTerms, optionTypes> makePricingTerms() {
    std::array<PricingTerms, optionTypes> terms;
    for (int i = 0; i < optionTypes; ++i) {
        const auto& option = testOptions[i];
        terms[i] = {
            option.spot,
            option.strike,
            (option.r - option.q + 0.5 * option.vol * option.vol) * option.t,
            option.vol * std::sqrt(option.t),
            std::exp(-option.q * option.t),
            std::exp(-option.r * option.t),
            option.type == PUT,
        };
    }
    return terms;
}

__constant__ PricingTerms deviceTerms[optionTypes];

__device__ double cumulativeNormal(double x) {
    constexpr double invSqrtTwo = 0.7071067811865475244;
    return 0.5 * (1.0 + erf(x * invSqrtTwo));
}

__global__ void priceOptions(size_t first, int count, double* results) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t index = first + i;
        const auto& option = deviceTerms[index % optionTypes];
        const double factor = 1.0 + 0.1 * (index / static_cast<double>(optionTypes));
        const double spot = option.spot * factor;
        const double strike = option.strike * factor;
        const double d1 = (log(spot / strike) + option.drift) / option.volTime;
        const double d2 = d1 - option.volTime;
        const double spotTerm = spot * option.spotDiscount * cumulativeNormal(option.put ? -d1 : d1);
        const double strikeTerm = strike * option.strikeDiscount * cumulativeNormal(option.put ? -d2 : d2);
        results[i] = option.put ? strikeTerm - spotTerm : spotTerm - strikeTerm;
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;

    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < std::min<size_t>(10, results.size()); ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % optionTypes].value;
        const double error = std::abs(computed - expected);
        const double relError = error / (std::abs(expected) + 1e-10);

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

bool anyRankFailed(bool failed) {
    const int local = failed;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return global != 0;
}

bool checkCuda(cudaError_t status, int rank, int device, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "Rank %d, CUDA device %d: %s: %s\n",
            rank, device, operation, cudaGetErrorString(status));
    return false;
}

std::vector<int> assignDevices(MPI_Comm node, int deviceCount, int rank, bool& failed) {
    using DeviceId = std::array<char, sizeof(cudaUUID_t::bytes)>;
    struct VisibleDevice {
        int rank;
        int ordinal;
    };

    int nodeRank = 0;
    int nodeRanks = 0;
    MPI_Comm_rank(node, &nodeRank);
    MPI_Comm_size(node, &nodeRanks);
    if (nodeRanks == 1) {
        std::vector<int> owned(deviceCount);
        std::iota(owned.begin(), owned.end(), 0);
        return owned;
    }

    std::vector<DeviceId> ids(deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp properties;
        if (checkCuda(cudaGetDeviceProperties(&properties, i), rank, i, "query device")) {
            memcpy(ids[i].data(), properties.uuid.bytes, ids[i].size());
        } else {
            failed = true;
        }
    }

    const int bytes = deviceCount * sizeof(DeviceId);
    std::vector<int> counts(nodeRanks);
    MPI_Allgather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, node);
    std::vector<int> displacements(nodeRanks);
    int totalBytes = 0;
    for (int r = 0; r < nodeRanks; ++r) {
        displacements[r] = totalBytes;
        totalBytes += counts[r];
    }
    std::vector<DeviceId> allIds(totalBytes / sizeof(DeviceId));
    MPI_Allgatherv(ids.data(), bytes, MPI_BYTE, allIds.data(), counts.data(),
                   displacements.data(), MPI_BYTE, node);

    // CUDA ordinals can differ when the launcher gives ranks different masks.
    // Assign each visible physical GPU once and balance among ranks that can use it.
    std::map<DeviceId, std::vector<VisibleDevice>> visibility;
    for (int r = 0; r < nodeRanks; ++r) {
        const int devices = counts[r] / sizeof(DeviceId);
        for (int i = 0; i < devices; ++i) {
            visibility[allIds[displacements[r] / sizeof(DeviceId) + i]].push_back({r, i});
        }
    }

    std::vector<int> assigned(nodeRanks);
    std::vector<int> owned;
    for (const auto& entry : visibility) {
        const auto& candidates = entry.second;
        const auto owner = std::min_element(candidates.begin(), candidates.end(),
            [&](const VisibleDevice& a, const VisibleDevice& b) {
                return assigned[a.rank] < assigned[b.rank];
            });
        ++assigned[owner->rank];
        if (owner->rank == nodeRank) {
            owned.push_back(owner->ordinal);
        }
    }
    return owned;
}

struct GpuWork {
    int device;
    size_t first;
    size_t end;
    int maxBlocks = 0;
    double* prices = nullptr;
};

constexpr int tileSize = 4 * 1024 * 1024;
constexpr int threadsPerBlock = 256;

bool priceOnGpus(std::vector<GpuWork>& work, size_t rankFirst, size_t outputCount,
                 std::vector<double>& results, int rank, double& elapsed) {
    const auto terms = makePricingTerms();
    const int workers = static_cast<int>(work.size());
    const int threads = std::max(1, std::min(workers, omp_get_max_threads()));
    std::atomic<bool> failed{false};
    bool ready = false;
    double start = 0.0;

    // Only the initial thread calls MPI (MPI_THREAD_FUNNELED).
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < workers; ++i) {
            auto& gpu = work[i];
            int multiprocessors = 0;
            cudaFuncAttributes attributes;
            const size_t capacity = std::min<size_t>(tileSize, gpu.end - gpu.first);
            if (!checkCuda(cudaSetDevice(gpu.device), rank, gpu.device, "select device") ||
                !checkCuda(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                                                  gpu.device), rank, gpu.device, "query device") ||
                !checkCuda(cudaFuncGetAttributes(&attributes, priceOptions),
                           rank, gpu.device, "load pricing kernel") ||
                !checkCuda(cudaMemcpyToSymbol(deviceTerms, terms.data(), sizeof(deviceTerms)),
                           rank, gpu.device, "upload pricing terms") ||
                !checkCuda(cudaMalloc(&gpu.prices, capacity * sizeof(double)),
                           rank, gpu.device, "allocate prices")) {
                failed = true;
            }
            gpu.maxBlocks = 32 * multiprocessors;
        }

        #pragma omp master
        {
            ready = !anyRankFailed(failed);
            start = MPI_Wtime();
        }
        #pragma omp barrier

        if (ready) {
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < workers; ++i) {
                const auto& gpu = work[i];
                bool ok = checkCuda(cudaSetDevice(gpu.device), rank, gpu.device, "select device");
                for (size_t first = gpu.first; ok && first < gpu.end;) {
                    const int count = std::min<size_t>(tileSize, gpu.end - first);
                    const int blocks = std::min(gpu.maxBlocks,
                                                (count + threadsPerBlock - 1) / threadsPerBlock);
                    priceOptions<<<blocks, threadsPerBlock>>>(first, count, gpu.prices);
                    ok = checkCuda(cudaGetLastError(), rank, gpu.device, "launch pricing kernel");

                    if (ok && first < outputCount) {
                        const size_t copied = std::min<size_t>(count, outputCount - first);
                        ok = checkCuda(cudaMemcpy(results.data() + (first - rankFirst), gpu.prices,
                                                  copied * sizeof(double), cudaMemcpyDeviceToHost),
                                       rank, gpu.device, "download prices");
                    }
                    first += count;
                }
                if (!ok) {
                    failed = true;
                }
            }

            // Submit work to all of this thread's devices before waiting for them.
            #pragma omp for schedule(static)
            for (int i = 0; i < workers; ++i) {
                const auto& gpu = work[i];
                if (!checkCuda(cudaSetDevice(gpu.device), rank, gpu.device, "select device") ||
                    !checkCuda(cudaDeviceSynchronize(), rank, gpu.device, "finish pricing")) {
                    failed = true;
                }
            }
        }

        #pragma omp master
        elapsed = MPI_Wtime() - start;

        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int i = 0; i < workers; ++i) {
            const auto& gpu = work[i];
            if (gpu.prices &&
                (!checkCuda(cudaSetDevice(gpu.device), rank, gpu.device, "select device") ||
                 !checkCuda(cudaFree(gpu.prices), rank, gpu.device, "free prices"))) {
                failed = true;
            }
        }
    }

    return !anyRankFailed(failed);
}

void gatherResults(std::vector<double>& results, uint64_t rankFirst, uint64_t rankEnd,
                   size_t outputCount, int rank, int ranks) {
    if (ranks == 1 || outputCount == 0) {
        return;
    }

    std::vector<uint64_t> rankOffsets(rank == 0 ? ranks + size_t{1} : 0);
    std::vector<int> counts(rank == 0 ? ranks : 0);
    std::vector<int> displacements(rank == 0 ? ranks : 0);
    MPI_Gather(&rankEnd, 1, MPI_UINT64_T, rank == 0 ? rankOffsets.data() + 1 : nullptr,
               1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // MPI_Gatherv uses int counts and displacements. Window the result for large runs.
    for (size_t first = 0; first < outputCount;) {
        const size_t end = first + std::min<size_t>(outputCount - first, INT_MAX);
        const size_t localFirst = std::max<uint64_t>(first, rankFirst);
        const size_t localEnd = std::min<uint64_t>(end, rankEnd);
        const int localCount = localEnd > localFirst ? localEnd - localFirst : 0;
        if (rank == 0) {
            for (int r = 0; r < ranks; ++r) {
                const size_t begin = std::max<uint64_t>(first, rankOffsets[r]);
                const size_t stop = std::min<uint64_t>(end, rankOffsets[r + 1]);
                counts[r] = stop > begin ? stop - begin : 0;
                displacements[r] = counts[r] ? begin - first : 0;
            }
        }

        void* send = nullptr;
        if (rank == 0) {
            send = MPI_IN_PLACE;
        } else if (localCount > 0) {
            send = results.data() + (localFirst - rankFirst);
        }
        MPI_Gatherv(send, localCount, MPI_DOUBLE,
                    rank == 0 ? results.data() + first : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        first = end;
    }
}

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            const auto count = std::strtoll(argv[++i], nullptr, 10);
            if (errno == ERANGE || count < 0) {
                if (rank == 0) {
                    fprintf(stderr, "Number of options is out of range\n");
                }
                return 1;
            }
            numOptions = count;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int deviceCount = 0;
    const bool queried = checkCuda(cudaGetDeviceCount(&deviceCount), rank, -1, "enumerate devices");
    if (queried && deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices are visible\n", rank);
    }
    if (anyRankFailed(!queried || deviceCount == 0)) {
        return 1;
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    bool assignmentFailed = false;
    const auto devices = assignDevices(node, deviceCount, rank, assignmentFailed);
    MPI_Comm_free(&node);
    if (anyRankFailed(assignmentFailed)) {
        return 1;
    }

    const uint64_t gpuCount = devices.size();
    uint64_t totalGpus = 0;
    uint64_t gpuOffset = 0;
    MPI_Exscan(&gpuCount, &gpuOffset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        gpuOffset = 0;
    }
    MPI_Allreduce(&gpuCount, &totalGpus, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // Partition by GPU so ranks with different GPU counts receive proportional work.
    const auto boundary = [=](uint64_t worker) {
        return worker * (numOptions / totalGpus) + std::min<uint64_t>(worker, numOptions % totalGpus);
    };

    std::vector<GpuWork> work;
    for (size_t i = 0; i < devices.size(); ++i) {
        const size_t first = boundary(gpuOffset + i);
        const size_t end = boundary(gpuOffset + i + 1);
        if (first < end) {
            work.push_back({devices[i], first, end});
        }
    }

    const size_t outputCount = printResults ? numOptions : std::min(numOptions, validate ? size_t{10} : 0);
    const size_t rankFirst = boundary(gpuOffset);
    const size_t rankEnd = boundary(gpuOffset + gpuCount);
    const size_t localOutput = rankFirst < outputCount
        ? std::min(rankEnd, outputCount) - rankFirst : 0;
    std::vector<double> results(rank == 0 ? outputCount : localOutput);

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    double elapsed = 0.0;
    if (!priceOnGpus(work, rankFirst, outputCount, results, rank, elapsed)) {
        return 1;
    }

    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    gatherResults(results, rankFirst, rankEnd, outputCount, rank, ranks);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0.0 ? numOptions / duration : 0.0);

        if (printResults) {
            print_results(results, "OptionPrices");
        }
        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    if (validate) {
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 1;
    }

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
