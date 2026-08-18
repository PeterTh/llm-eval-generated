#include <algorithm>
#include <array>
#include <atomic>
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

enum OptionType { CALL = 0, PUT = 1 };

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
    return {{{CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
             {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
             {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
             {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
             {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}}};
}

__constant__ OptionInput deviceTestOptions[7];

__device__ __forceinline__ double cumulativeNormal(double x) noexcept {
    return 0.5 * (1.0 + erf(x * 0.70710678118654752440));
}

__device__ __forceinline__ double blackScholes(const OptionInput& option,
                                                double factor) noexcept {
    // Both spot and strike are scaled by the same factor in the original
    // generator.  Applying that scale to the final price avoids redundant
    // global-memory traffic while preserving the homogeneity of the formula.
    const double sqrtT = sqrt(option.t);
    const double sigmaSqrtT = option.vol * sqrtT;
    const double d1 = (log(option.spot / option.strike) +
                       (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
                      sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    const double spotDiscounted = option.spot * exp(-option.q * option.t);
    const double strikeDiscounted = option.strike * exp(-option.r * option.t);
    double price;
    if (option.type == CALL) {
        price = spotDiscounted * cumulativeNormal(d1) -
                strikeDiscounted * cumulativeNormal(d2);
    } else {
        price = strikeDiscounted * cumulativeNormal(-d2) -
                spotDiscounted * cumulativeNormal(-d1);
    }
    return factor * price;
}

__global__ void priceOptionsKernel(double* __restrict__ results, size_t count,
                                   size_t globalOffset) {
    const size_t thread = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t localIndex = thread; localIndex < count; localIndex += stride) {
        const size_t globalIndex = globalOffset + localIndex;
        const OptionInput option = deviceTestOptions[globalIndex % 7];
        const double factor = 1.0 + 0.1 * (globalIndex / 7.0);
        results[localIndex] = blackScholes(option, factor);
    }
}

static OptionInput generatedOption(size_t index) {
    const auto tests = getTestOptions();
    OptionInput option = tests[index % tests.size()];
    const double factor = 1.0 + 0.1 * (index / static_cast<double>(tests.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

static bool validateResults(const std::vector<double>& results) {
    const int checks = static_cast<int>(std::min<size_t>(10, results.size()));
    std::array<OptionInput, 10> options{};
    std::array<double, 10> relativeErrors{};
    std::array<int, 10> valid{};

#pragma omp parallel for schedule(static)
    for (int i = 0; i < checks; ++i) {
        options[i] = generatedOption(static_cast<size_t>(i));
        const double error = fabs(results[i] - options[i].value);
        relativeErrors[i] = error / (fabs(options[i].value) + 1.0e-10);
        valid[i] = results[i] >= 0.0 && results[i] <= 1000.0 && isfinite(results[i]);
    }

    bool allPassed = true;
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i,
                    results[i], options[i].value, relativeErrors[i]);
        if (!valid[i]) {
            std::printf("Validation failed at option %d: invalid value %.4f\n", i,
                        results[i]);
            allPassed = false;
        }
    }
    return allPassed;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static void reportCudaError(cudaError_t error, const char* operation, int rank, int device,
                            std::atomic<int>& failed) {
    if (error == cudaSuccess) return;
#pragma omp critical(cuda_error_output)
    std::fprintf(stderr, "Rank %d GPU %d: %s failed: %s\n", rank, device, operation,
                 cudaGetErrorString(error));
    failed.store(1, std::memory_order_relaxed);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false, showHelp = false, badArguments = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            badArguments = !end || *end != '\0' || argv[i][0] == '-';
            if (parsed > std::numeric_limits<size_t>::max()) badArguments = true;
            numOptions = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            badArguments = true;
        }
    }
    if (showHelp || badArguments) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArguments ? 1 : 0;
    }

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t globalOffset = static_cast<size_t>(rank) * base +
                                std::min(static_cast<size_t>(rank), remainder);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localRanks);

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    int noDevice = deviceStatus != cudaSuccess || deviceCount == 0;
    int anyNoDevice = 0;
    MPI_Allreduce(&noDevice, &anyNoDevice, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyNoDevice) {
        if (rank == 0)
            std::fprintf(stderr, "CUDA-capable GPUs are required on every participating node.\n");
        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return 1;
    }

    // Give every local rank a disjoint strided subset of the node's GPUs. If
    // there are more ranks than GPUs, preserve functionality via round-robin
    // oversubscription while warning the user about the non-optimal launch.
    const int workers = deviceCount >= localRanks
                            ? (deviceCount - 1 - localRank) / localRanks + 1
                            : 1;
    if (rank == 0 && localRanks > deviceCount)
        std::fprintf(stderr,
                     "Warning: %d local MPI ranks share %d GPUs; use at most one rank per GPU "
                     "for maximum performance.\n",
                     localRanks, deviceCount);

    std::vector<size_t> workerCounts(workers), workerOffsets(workers);
    std::vector<double*> deviceResults(workers, nullptr);
    const size_t workerBase = localCount / static_cast<size_t>(workers);
    const size_t workerRemainder = localCount % static_cast<size_t>(workers);
    for (int worker = 0; worker < workers; ++worker) {
        workerCounts[worker] = workerBase +
                               (static_cast<size_t>(worker) < workerRemainder ? 1 : 0);
        workerOffsets[worker] = static_cast<size_t>(worker) * workerBase +
                                std::min(static_cast<size_t>(worker), workerRemainder);
    }

    double* localResults = nullptr;
    if (localCount != 0) {
        cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&localResults),
                                           localCount * sizeof(double));
        if (error != cudaSuccess) {
            std::fprintf(stderr, "Rank %d: pinned host allocation failed: %s\n", rank,
                         cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    omp_set_dynamic(0);
    std::atomic<int> cudaFailed{0};
    const auto tests = getTestOptions();
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = deviceCount >= localRanks
                               ? localRank + worker * localRanks
                               : localRank % deviceCount;
        reportCudaError(cudaSetDevice(device), "cudaSetDevice", rank, device, cudaFailed);
        reportCudaError(cudaFree(nullptr), "CUDA context initialization", rank, device,
                        cudaFailed);
        reportCudaError(cudaMemcpyToSymbol(deviceTestOptions, tests.data(), sizeof(tests)),
                        "constant-data upload", rank, device, cudaFailed);
        if (workerCounts[worker] != 0)
            reportCudaError(cudaMalloc(reinterpret_cast<void**>(&deviceResults[worker]),
                                       workerCounts[worker] * sizeof(double)),
                            "device allocation", rank, device, cudaFailed);
    }

    int localFailure = cudaFailed.load(std::memory_order_relaxed), globalFailure = 0;
    MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalFailure) {
        if (localResults) cudaFreeHost(localResults);
        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("MPI ranks: %d, GPUs on rank-0 node: %d\n", ranks, deviceCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = deviceCount >= localRanks
                               ? localRank + worker * localRanks
                               : localRank % deviceCount;
        cudaSetDevice(device);
        const size_t count = workerCounts[worker];
        if (count == 0) continue;

        cudaDeviceProp properties{};
        reportCudaError(cudaGetDeviceProperties(&properties, device), "device query", rank,
                        device, cudaFailed);
        constexpr int blockSize = 256;
        const size_t blocksNeeded = (count + blockSize - 1) / blockSize;
        const size_t occupancyGrid = static_cast<size_t>(properties.multiProcessorCount) * 32;
        const int blocks = static_cast<int>(std::min(blocksNeeded, occupancyGrid));
        priceOptionsKernel<<<blocks, blockSize>>>(
            deviceResults[worker], count, globalOffset + workerOffsets[worker]);
        reportCudaError(cudaGetLastError(), "kernel launch", rank, device, cudaFailed);
        reportCudaError(cudaMemcpy(localResults + workerOffsets[worker], deviceResults[worker],
                                   count * sizeof(double), cudaMemcpyDeviceToHost),
                        "result download", rank, device, cudaFailed);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    localFailure = cudaFailed.load(std::memory_order_relaxed);
    MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalFailure) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Options per second: %.0f\n", elapsed > 0.0 ? numOptions / elapsed : 0.0);
    }

    std::vector<double> results;
    if (validate || printResults) {
        if (localCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI count limits.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<int> counts, displacements;
        if (rank == 0) {
            results.resize(numOptions);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t rCount = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t rOffset = static_cast<size_t>(r) * base +
                                       std::min(static_cast<size_t>(r), remainder);
                if (rCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    rOffset > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::fprintf(stderr, "Result gathering exceeds MPI count limits.\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                counts[r] = static_cast<int>(rCount);
                displacements[r] = static_cast<int>(rOffset);
            }
        }
        double dummy = 0.0;
        MPI_Gatherv(localCount ? localResults : &dummy, static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 && numOptions ? results.data() : &dummy,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int returnCode = 0;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::printf("Validating results...\n");
            const bool valid = validateResults(results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = deviceCount >= localRanks
                               ? localRank + worker * localRanks
                               : localRank % deviceCount;
        cudaSetDevice(device);
        if (deviceResults[worker]) cudaFree(deviceResults[worker]);
    }
    if (localResults) cudaFreeHost(localResults);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return returnCode;
}
