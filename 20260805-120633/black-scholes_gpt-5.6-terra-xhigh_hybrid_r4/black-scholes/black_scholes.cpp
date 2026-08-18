#include <algorithm>
#include <array>
#include <chrono>
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

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
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

// Only the values used by the pricing kernel are transferred to the GPU.
// Keeping this compact avoids moving validation-only data over PCIe/NVLink.
struct DeviceOption {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
};

#define HOST_DEVICE __host__ __device__

// Standard normal cumulative distribution function.
HOST_DEVICE inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Black-Scholes formula for European options.
HOST_DEVICE inline double blackScholes(const DeviceOption& option) noexcept {
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
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    }
    return K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
}

__global__ void blackScholesKernel(const DeviceOption* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t numOptions) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numOptions;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        results[i] = blackScholes(options[i]);
    }
}

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

inline OptionInput makeOption(const size_t index) noexcept {
    constexpr auto testOptions = getTestOptions();
    const OptionInput& base = testOptions[index % testOptions.size()];
    OptionInput option = base;
    const double factor = 1.0 + 0.1 * (index / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

inline DeviceOption makeDeviceOption(const size_t index) noexcept {
    const OptionInput option = makeOption(index);
    return {option.type, option.strike, option.spot, option.q, option.r, option.t, option.vol};
}

// Generate a contiguous global range.  The range form lets every MPI rank
// construct exactly its assigned inputs without replicating the full problem.
void generateOptions(std::vector<OptionInput>& options,
                     const size_t numOptions,
                     const size_t firstIndex = 0) {
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = makeOption(firstIndex + i);
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), options.size()));

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[static_cast<size_t>(i)];
        const double expected = options[static_cast<size_t>(i)].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Keep the original benchmark's deliberately relaxed validity check.
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

[[noreturn]] void cudaCheck(const cudaError_t status, const char* operation, const int rank) {
    fprintf(stderr, "Rank %d: CUDA failure during %s: %s\n", rank, operation,
            cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

inline void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        cudaCheck(status, operation, rank);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, rank)

void getLocalRange(const size_t globalCount, const int rank, const int worldSize,
                   size_t& firstIndex, size_t& localCount) {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t quotient = globalCount / ranks;
    const size_t remainder = globalCount % ranks;

    localCount = quotient + (rankIndex < remainder ? 1 : 0);
    firstIndex = rankIndex * quotient + std::min(rankIndex, remainder);
}

void gatherResults(const double* localResults, const size_t localCount,
                   const size_t numOptions, const int rank, const int worldSize,
                   std::vector<double>& results) {
    // MPI_Gatherv uses int counts/displacements.  The benchmark's result-output
    // path needs a root-resident vector anyway, so reject only infeasible output
    // requests rather than constraining the normal distributed compute path.
    if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Result collection supports at most %d options.\n",
                    std::numeric_limits<int>::max());
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        for (int process = 0; process < worldSize; ++process) {
            size_t firstIndex = 0;
            size_t processCount = 0;
            getLocalRange(numOptions, process, worldSize, firstIndex, processCount);
            counts[static_cast<size_t>(process)] = static_cast<int>(processCount);
            displacements[static_cast<size_t>(process)] = static_cast<int>(firstIndex);
        }
        results.resize(numOptions);
    }

    const int sendCount = static_cast<int>(localCount);
    const int mpiStatus = MPI_Gatherv(localResults, sendCount, MPI_DOUBLE,
                                      rank == 0 ? results.data() : nullptr,
                                      rank == 0 ? counts.data() : nullptr,
                                      rank == 0 ? displacements.data() : nullptr,
                                      MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpiStatus != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "MPI_Gatherv failed while collecting results.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, mpiStatus);
    }
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (initStatus != MPI_SUCCESS) {
        fprintf(stderr, "Unable to initialize MPI.\n");
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide the MPI_THREAD_FUNNELED support required by this benchmark.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool invalidArguments = false;

    // Parse command line arguments identically on every rank.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            invalidArguments = true;
        }
    }

    if (showHelp || invalidArguments) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return invalidArguments ? 1 : 0;
    }

    // Select one accelerator per local MPI rank.  CUDA_VISIBLE_DEVICES is
    // respected, so schedulers can bind ranks to GPUs without code changes.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA accelerator is visible to this MPI job.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, localRank % deviceCount));
    MPI_Comm_free(&nodeComm);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t firstIndex = 0;
    size_t localCount = 0;
    getLocalRange(numOptions, rank, worldSize, firstIndex, localCount);

    DeviceOption* hostOptions = nullptr;
    double* localResults = nullptr;
    DeviceOption* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    cudaStream_t stream = nullptr;

    if (localCount != 0) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostOptions),
                                  localCount * sizeof(*hostOptions)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&localResults),
                                  localCount * sizeof(*localResults)));

        // CPU cores create the rank-local batch in parallel while the GPU is
        // reserved for the transcendental-heavy pricing computation.
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localCount; ++i) {
            hostOptions[i] = makeDeviceOption(firstIndex + i);
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOptions),
                              localCount * sizeof(*deviceOptions)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                              localCount * sizeof(*deviceResults)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    // Synchronize only at the distributed timing boundary.  Each rank has an
    // independent GPU batch, and rank 0 reports the cluster critical path.
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();

    if (localCount != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceOptions, hostOptions,
                                   localCount * sizeof(*hostOptions),
                                   cudaMemcpyHostToDevice, stream));

        constexpr int threadsPerBlock = 256;
        const size_t blocksNeeded = (localCount + threadsPerBlock - 1) / threadsPerBlock;
        const size_t residentBlocks = static_cast<size_t>(
            std::max(1, deviceProperties.multiProcessorCount * 32));
        const int blocks = static_cast<int>(std::min(blocksNeeded, residentBlocks));
        blackScholesKernel<<<blocks, threadsPerBlock, 0, stream>>>(deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(localResults, deviceResults,
                                   localCount * sizeof(*localResults),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const auto end = std::chrono::steady_clock::now();
    const double localMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();
    double clusterMilliseconds = 0.0;
    MPI_Reduce(&localMilliseconds, &clusterMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", clusterMilliseconds);
        const double optionsPerSecond = clusterMilliseconds > 0.0
            ? static_cast<double>(numOptions) / (clusterMilliseconds / 1000.0)
            : 0.0;
        printf("Options per second: %.0f\n", optionsPerSecond);
    }

    std::vector<double> results;
    if (printResults || validate) {
        gatherResults(localResults, localCount, numOptions, rank, worldSize, results);
    }

    int exitCode = 0;
    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }

    if (rank == 0 && validate) {
        std::vector<OptionInput> validationOptions;
        generateOptions(validationOptions, std::min(static_cast<size_t>(10), numOptions));
        printf("Validating results...\n");
        const bool valid = validateResults(validationOptions, results);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? 0 : 1;
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    if (deviceResults != nullptr) {
        CUDA_CHECK(cudaFree(deviceResults));
    }
    if (deviceOptions != nullptr) {
        CUDA_CHECK(cudaFree(deviceOptions));
    }
    if (localResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(localResults));
    }
    if (hostOptions != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostOptions));
    }

    MPI_Finalize();
    return exitCode;
}
