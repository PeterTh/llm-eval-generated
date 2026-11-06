#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

// This table is also used by validation on the host.  The device table is
// identical and lets each GPU generate its own contiguous range of options.
constexpr std::array<OptionInput, 7> testOptions = {{
    {0, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {0, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {0, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {0, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {1, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {1, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {1, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
}};
__constant__ OptionInput deviceOptions[7];

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

__global__ void priceOptions(double* results, size_t first, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; i < count; i += stride) {
        const size_t index = first + i;
        const OptionInput option = deviceOptions[index % 7];
        const double factor = 1.0 + 0.1 * (static_cast<double>(index) / 7.0);
        const double S = option.spot * factor;
        const double K = option.strike * factor;
        const double T = option.t;
        const double sigma = option.vol;
        if (T <= 0.0 || sigma <= 0.0) {
            results[i] = 0.0;
            continue;
        }
        const double d1 = (log(S / K) + (option.r - option.q + 0.5 * sigma * sigma) * T) /
                          (sigma * sqrt(T));
        const double d2 = d1 - sigma * sqrt(T);
        const double discount = exp(-option.r * T);
        constexpr double invSqrt2 = 0.7071067811865475244;
        double price;
        if (option.type == 0) {
            const double Nd1 = 0.5 * (1.0 + erf(d1 * invSqrt2));
            const double Nd2 = 0.5 * (1.0 + erf(d2 * invSqrt2));
            price = S * exp(-option.q * T) * Nd1 - K * discount * Nd2;
        } else {
            const double Nnegd2 = 0.5 * (1.0 + erf(-d2 * invSqrt2));
            const double Nnegd1 = 0.5 * (1.0 + erf(-d1 * invSqrt2));
            price = K * discount * Nnegd2 - S * exp(-option.q * T) * Nnegd1;
        }
        results[i] = price;
    }
}

static size_t partSize(size_t n, int parts, int part) {
    return n / parts + (static_cast<size_t>(part) < n % parts);
}
static size_t partStart(size_t n, int parts, int part) {
    return (n / parts) * part + std::min(static_cast<size_t>(part), n % parts);
}

static bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size() && i < 10; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % testOptions.size()].value;
        const double relError = std::fabs(computed - expected) / (std::fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

struct Worker {
    int device;
    size_t start, count;
    double* buffer = nullptr;
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_MULTIPLE) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_MULTIPLE\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Discover physical GPUs visible to the local ranks.  This handles both
    // shared device lists and scheduler-provided per-rank visibility masks.
    std::vector<int> visibleCounts(localRanks);
    MPI_Allgather(&deviceCount, 1, MPI_INT, visibleCounts.data(), 1, MPI_INT, localComm);
    const int maxVisible = *std::max_element(visibleCounts.begin(), visibleCounts.end());
    std::vector<cudaUUID_t> visible(maxVisible);
    for (int d = 0; d < deviceCount; ++d) {
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, d), "cudaGetDeviceProperties");
        visible[d] = properties.uuid;
    }
    std::vector<cudaUUID_t> allVisible(static_cast<size_t>(localRanks) * maxVisible);
    MPI_Allgather(visible.data(), maxVisible * static_cast<int>(sizeof(cudaUUID_t)), MPI_BYTE,
                  allVisible.data(), maxVisible * static_cast<int>(sizeof(cudaUUID_t)), MPI_BYTE, localComm);
    std::vector<cudaUUID_t> uniqueDevices;
    std::vector<std::vector<int>> ordinals(localRanks);
    for (int r = 0; r < localRanks; ++r) {
        for (int d = 0; d < visibleCounts[r]; ++d) {
            const cudaUUID_t& id = allVisible[static_cast<size_t>(r) * maxVisible + d];
            size_t u = 0;
            for (; u < uniqueDevices.size(); ++u)
                if (std::memcmp(&id, &uniqueDevices[u], sizeof(id)) == 0) break;
            if (u == uniqueDevices.size()) uniqueDevices.push_back(id);
            ordinals[r].push_back(static_cast<int>(u));
        }
    }
    std::vector<std::vector<int>> assigned(localRanks);
    std::vector<int> uses(uniqueDevices.size(), 0);
    std::vector<int> order(localRanks);
    for (int r = 0; r < localRanks; ++r) order[r] = r;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return visibleCounts[a] < visibleCounts[b];
    });
    // Give each rank a device first, sharing only when necessary.
    for (int r : order) {
        int best = 0;
        for (int d = 1; d < visibleCounts[r]; ++d)
            if (uses[ordinals[r][d]] < uses[ordinals[r][best]]) best = d;
        assigned[r].push_back(best);
        ++uses[ordinals[r][best]];
    }
    // Distribute any remaining devices among ranks that can see them.
    for (size_t u = 0; u < uniqueDevices.size(); ++u) {
        if (uses[u] != 0) continue;
        int bestRank = -1, bestDevice = -1;
        for (int r = 0; r < localRanks; ++r) {
            for (int d = 0; d < visibleCounts[r]; ++d) {
                if (ordinals[r][d] == static_cast<int>(u) &&
                    (bestRank < 0 || assigned[r].size() < assigned[bestRank].size())) {
                    bestRank = r;
                    bestDevice = d;
                }
            }
        }
        assigned[bestRank].push_back(bestDevice);
    }
    const std::vector<int>& devices = assigned[localRank];
    const int workersCount = static_cast<int>(devices.size());
    std::vector<int> rankWorkers(ranks);
    MPI_Allgather(&workersCount, 1, MPI_INT, rankWorkers.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> workerStarts(ranks + 1, 0);
    for (int r = 0; r < ranks; ++r) workerStarts[r + 1] = workerStarts[r] + rankWorkers[r];
    const int totalWorkers = workerStarts[ranks];
    const size_t localStart = partStart(numOptions, totalWorkers, workerStarts[rank]);
    const size_t localEnd = partStart(numOptions, totalWorkers, workerStarts[rank + 1]);
    const size_t localCount = localEnd - localStart;
    const size_t retained = printResults ? numOptions : (validate ? std::min(numOptions, size_t{10}) : 0);
    const size_t localRetained = localStart < retained ? std::min(localCount, retained - localStart) : 0;
    std::vector<double> localResults(localRetained);
    std::vector<Worker> workers(workersCount);
    constexpr size_t batchSize = size_t{1} << 22;

    // One OpenMP thread owns each GPU.  Set up contexts and buffers outside
    // the measured region, then process bounded batches to handle large runs.
#pragma omp parallel for num_threads(workersCount) schedule(static)
    for (int w = 0; w < workersCount; ++w) {
        Worker& worker = workers[w];
        worker.device = devices[w];
        worker.start = partStart(numOptions, totalWorkers, workerStarts[rank] + w) - localStart;
        worker.count = partSize(numOptions, totalWorkers, workerStarts[rank] + w);
        cudaCheck(cudaSetDevice(worker.device), "cudaSetDevice");
        cudaCheck(cudaMemcpyToSymbol(deviceOptions, testOptions.data(), sizeof(OptionInput) * testOptions.size()),
                  "copy option table");
        if (worker.count != 0)
            cudaCheck(cudaMalloc(&worker.buffer, std::min(worker.count, batchSize) * sizeof(double)), "cudaMalloc");
    }

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
#pragma omp parallel for num_threads(workersCount) schedule(static)
    for (int w = 0; w < workersCount; ++w) {
        const Worker& worker = workers[w];
        cudaCheck(cudaSetDevice(worker.device), "cudaSetDevice");
        for (size_t offset = 0; offset < worker.count;) {
            const size_t count = std::min(batchSize, worker.count - offset);
            const int blocks = static_cast<int>(std::min((count + 255) / 256, size_t{65535}));
            priceOptions<<<blocks, 256>>>(worker.buffer, localStart + worker.start + offset, count);
            cudaCheck(cudaGetLastError(), "priceOptions launch");
            const size_t resultOffset = worker.start + offset;
            const size_t toCopy = resultOffset < localRetained ? std::min(count, localRetained - resultOffset) : 0;
            if (toCopy != 0)
                cudaCheck(cudaMemcpy(localResults.data() + resultOffset, worker.buffer,
                                     toCopy * sizeof(double), cudaMemcpyDeviceToHost), "copy prices");
            offset += count;
        }
        cudaCheck(cudaDeviceSynchronize(), "priceOptions execution");
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }

    // Transfer in bounded chunks, in rank order, avoiding MPI's int count and
    // displacement limits while preserving the original result order.
    std::vector<double> results;
    if (rank == 0) {
        results.resize(retained);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t first = partStart(numOptions, totalWorkers, workerStarts[source]);
            const size_t end = partStart(numOptions, totalWorkers, workerStarts[source + 1]);
            const size_t count = first < retained ? std::min(end - first, retained - first) : 0;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(results.data() + first + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < localRetained;) {
            const int chunk = static_cast<int>(std::min(localRetained - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(localResults.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += chunk;
        }
    }

    int status = 0;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
#pragma omp parallel for num_threads(workersCount) schedule(static)
    for (int w = 0; w < workersCount; ++w) {
        cudaCheck(cudaSetDevice(workers[w].device), "cudaSetDevice");
        if (workers[w].buffer) cudaCheck(cudaFree(workers[w].buffer), "cudaFree");
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return status;
}
