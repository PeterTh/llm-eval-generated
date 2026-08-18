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

// A structure of arrays gives adjacent CUDA threads coalesced loads for every
// Black-Scholes input field.  Each MPI rank owns one contiguous global range.
struct LocalOptions {
    std::vector<int> type;
    std::vector<double> strike;
    std::vector<double> spot;
    std::vector<double> q;
    std::vector<double> r;
    std::vector<double> t;
    std::vector<double> vol;

    void resize(const size_t count) {
        type.resize(count);
        strike.resize(count);
        spot.resize(count);
        q.resize(count);
        r.resize(count);
        t.resize(count);
        vol.resize(count);
    }
};

struct DeviceOptions {
    int* type = nullptr;
    double* strike = nullptr;
    double* spot = nullptr;
    double* q = nullptr;
    double* r = nullptr;
    double* t = nullptr;
    double* vol = nullptr;
    double* results = nullptr;
};

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation, const int rank) {
    std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

inline void checkCuda(const cudaError_t error, const char* operation, const int rank) {
    if (error != cudaSuccess) {
        cudaFailure(error, operation, rank);
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

// OpenMP parallelizes host-side synthesis while MPI distributes non-overlapping
// contiguous global ranges.  The formula deliberately matches the original
// generator, including the global rather than rank-local variation factor.
void generateOptions(LocalOptions& options, const size_t globalStart, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

#pragma omp parallel for schedule(static)
    for (long long localIndex = 0; localIndex < static_cast<long long>(count); ++localIndex) {
        const size_t index = static_cast<size_t>(localIndex);
        const size_t globalIndex = globalStart + index;
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));

        options.type[index] = base.type;
        options.strike[index] = base.strike * factor;
        options.spot[index] = base.spot * factor;
        options.q[index] = base.q;
        options.r[index] = base.r;
        options.t[index] = base.t;
        options.vol[index] = base.vol;
    }
}

// This small host-only representation is retained for the existing validation
// output.  It is built only for the first ten values rather than for every rank.
void generateValidationOptions(std::vector<OptionInput>& options, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);
    for (size_t i = 0; i < count; ++i) {
        options[i] = testOptions[i % testOptions.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * 0.707106781186547524400844362104849039));
}

// Each rank launches enough blocks to saturate its assigned GPU, then uses a
// grid-stride loop to cover arbitrarily large MPI partitions.
__global__ void blackScholesKernel(const size_t count, const int* __restrict__ type,
                                   const double* __restrict__ strike,
                                   const double* __restrict__ spot,
                                   const double* __restrict__ q,
                                   const double* __restrict__ r,
                                   const double* __restrict__ t,
                                   const double* __restrict__ vol,
                                   double* __restrict__ results) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < count; i += stride) {
        const double maturity = t[i];
        const double sigma = vol[i];
        if (maturity <= 0.0 || sigma <= 0.0) {
            results[i] = 0.0;
            continue;
        }

        const double sqrtMaturity = sqrt(maturity);
        const double sigmaSqrtMaturity = sigma * sqrtMaturity;
        const double d1 = (log(spot[i] / strike[i]) +
                           (r[i] - q[i] + 0.5 * sigma * sigma) * maturity) /
                          sigmaSqrtMaturity;
        const double d2 = d1 - sigmaSqrtMaturity;
        const double discount = exp(-r[i] * maturity);
        const double dividendDiscount = exp(-q[i] * maturity);

        if (type[i] == CALL) {
            results[i] = spot[i] * dividendDiscount * cumulativeNormalDevice(d1) -
                         strike[i] * discount * cumulativeNormalDevice(d2);
        } else {
            results[i] = strike[i] * discount * cumulativeNormalDevice(-d2) -
                         spot[i] * dividendDiscount * cumulativeNormalDevice(-d1);
        }
    }
}

void allocateDeviceOptions(DeviceOptions& device, const size_t count, const int rank) {
    const size_t doubleBytes = count * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.type), count * sizeof(int)),
              "allocation of option types", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.strike), doubleBytes),
              "allocation of strike prices", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.spot), doubleBytes),
              "allocation of spot prices", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.q), doubleBytes),
              "allocation of dividend yields", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.r), doubleBytes),
              "allocation of interest rates", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.t), doubleBytes),
              "allocation of maturities", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.vol), doubleBytes),
              "allocation of volatilities", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.results), doubleBytes),
              "allocation of results", rank);
}

void releaseDeviceOptions(DeviceOptions& device, const int rank) {
    checkCuda(cudaFree(device.type), "release of option types", rank);
    checkCuda(cudaFree(device.strike), "release of strike prices", rank);
    checkCuda(cudaFree(device.spot), "release of spot prices", rank);
    checkCuda(cudaFree(device.q), "release of dividend yields", rank);
    checkCuda(cudaFree(device.r), "release of interest rates", rank);
    checkCuda(cudaFree(device.t), "release of maturities", rank);
    checkCuda(cudaFree(device.vol), "release of volatilities", rank);
    checkCuda(cudaFree(device.results), "release of results", rank);
}

void copyOptionsToDevice(const LocalOptions& host, const DeviceOptions& device,
                         const size_t count, const int rank) {
    const size_t doubleBytes = count * sizeof(double);
    checkCuda(cudaMemcpy(device.type, host.type.data(), count * sizeof(int), cudaMemcpyHostToDevice),
              "copy of option types", rank);
    checkCuda(cudaMemcpy(device.strike, host.strike.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of strike prices", rank);
    checkCuda(cudaMemcpy(device.spot, host.spot.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of spot prices", rank);
    checkCuda(cudaMemcpy(device.q, host.q.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of dividend yields", rank);
    checkCuda(cudaMemcpy(device.r, host.r.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of interest rates", rank);
    checkCuda(cudaMemcpy(device.t, host.t.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of maturities", rank);
    checkCuda(cudaMemcpy(device.vol, host.vol.data(), doubleBytes, cudaMemcpyHostToDevice),
              "copy of volatilities", rank);
}

void priceOptionsOnGpu(const LocalOptions& options, std::vector<double>& results,
                       DeviceOptions& device, const int deviceMultiprocessors,
                       const int rank) {
    const size_t count = results.size();
    if (count == 0) {
        return;
    }

    copyOptionsToDevice(options, device, count, rank);

    constexpr int blockSize = 256;
    // More blocks than SMs hide the latency of the transcendental operations,
    // without launching a grid proportional to an arbitrarily large batch.
    const size_t saturationBlocks = static_cast<size_t>(deviceMultiprocessors) * 32;
    const size_t requiredBlocks = (count + blockSize - 1) / blockSize;
    const unsigned int gridSize = static_cast<unsigned int>(
        std::min(requiredBlocks, std::min(saturationBlocks,
                                          static_cast<size_t>(std::numeric_limits<unsigned int>::max()))));

    blackScholesKernel<<<gridSize, blockSize>>>(count, device.type, device.strike, device.spot,
                                                 device.q, device.r, device.t, device.vol,
                                                 device.results);
    checkCuda(cudaGetLastError(), "kernel launch", rank);
    checkCuda(cudaMemcpy(results.data(), device.results, count * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "copy of results", rank);
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min(static_cast<size_t>(10), options.size());

    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Retain the original benchmark's deliberately relaxed validity check.
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
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

size_t localCountForRank(const size_t total, const int worldSize, const int rank) {
    const size_t quotient = total / static_cast<size_t>(worldSize);
    const size_t remainder = total % static_cast<size_t>(worldSize);
    return quotient + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t localStartForRank(const size_t total, const int worldSize, const int rank) {
    const size_t quotient = total / static_cast<size_t>(worldSize);
    const size_t remainder = total % static_cast<size_t>(worldSize);
    return static_cast<size_t>(rank) * quotient +
           std::min(static_cast<size_t>(rank), remainder);
}

// Gather a global prefix. Validation needs only its first ten prices, while
// external result reporting preserves the original all-results behavior.
bool gatherPrefix(const std::vector<double>& localResults, const size_t totalCount,
                  const size_t localStart, const int worldSize, const int rank,
                  const size_t prefixCount, std::vector<double>& rootResults) {
    if (prefixCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::fprintf(stderr, "Cannot gather more than %d results with MPI_Gatherv.\n",
                         std::numeric_limits<int>::max());
        }
        return false;
    }

    const size_t localAvailable = localStart < prefixCount
                                      ? std::min(localResults.size(), prefixCount - localStart)
                                      : 0;
    const int sendCount = static_cast<int>(localAvailable);
    std::vector<int> counts;
    std::vector<int> displacements;

    if (rank == 0) {
        rootResults.resize(prefixCount);
        counts.resize(worldSize);
        displacements.resize(worldSize);
        for (int sourceRank = 0; sourceRank < worldSize; ++sourceRank) {
            const size_t sourceStart = localStartForRank(totalCount, worldSize, sourceRank);
            const size_t sourceCount = localCountForRank(totalCount, worldSize, sourceRank);
            const size_t available = sourceStart < prefixCount
                                         ? std::min(sourceCount, prefixCount - sourceStart)
                                         : 0;
            counts[sourceRank] = static_cast<int>(available);
            displacements[sourceRank] = static_cast<int>(sourceStart);
        }
    }

    const int mpiStatus = MPI_Gatherv(localResults.data(), sendCount, MPI_DOUBLE,
                                      rank == 0 ? rootResults.data() : nullptr,
                                      rank == 0 ? counts.data() : nullptr,
                                      rank == 0 ? displacements.data() : nullptr,
                                      MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (mpiStatus != MPI_SUCCESS) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI_Gatherv failed while collecting results.\n");
        }
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI.\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int parseStatus = EXIT_SUCCESS;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = EXIT_FAILURE;
            break;
        }
    }

    if (parseStatus != EXIT_SUCCESS) {
        MPI_Finalize();
        return parseStatus;
    }

    // Assign one visible accelerator per local MPI rank.  MPI launchers usually
    // set CUDA_VISIBLE_DEVICES per rank; modulo also handles an unmasked node.
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device discovery", rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA accelerator is visible to this MPI job.\n");
        }
        MPI_Comm_free(&nodeComm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "device selection", rank);

    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, localRank % deviceCount),
              "device query", rank);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localStart = localStartForRank(numOptions, worldSize, rank);
    const size_t localCount = localCountForRank(numOptions, worldSize, rank);
    LocalOptions localOptions;
    generateOptions(localOptions, localStart, localCount);
    std::vector<double> localResults(localCount);

    DeviceOptions deviceOptions;
    if (localCount != 0) {
        allocateDeviceOptions(deviceOptions, localCount, rank);
    }

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    priceOptionsOnGpu(localOptions, localResults, deviceOptions,
                      deviceProperties.multiProcessorCount, rank);

    const auto end = std::chrono::high_resolution_clock::now();
    const double localMilliseconds =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double globalMilliseconds = 0.0;
    MPI_Reduce(&localMilliseconds, &globalMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (localCount != 0) {
        releaseDeviceOptions(deviceOptions, rank);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", globalMilliseconds);
        printf("Options per second: %.0f\n", numOptions / (globalMilliseconds / 1.0e3));
    }

    int resultStatus = EXIT_SUCCESS;
    if (printResults || validate) {
        const size_t requiredResults = printResults
                                           ? numOptions
                                           : std::min(static_cast<size_t>(10), numOptions);
        std::vector<double> gatheredResults;
        if (!gatherPrefix(localResults, numOptions, localStart, worldSize, rank,
                          requiredResults, gatheredResults)) {
            resultStatus = EXIT_FAILURE;
        } else if (rank == 0) {
            if (printResults) {
                print_results(gatheredResults, "OptionPrices");
            }
            if (validate) {
                printf("Validating results...\n");
                std::vector<OptionInput> validationOptions;
                generateValidationOptions(validationOptions,
                                          std::min(static_cast<size_t>(10), numOptions));
                if (validateResults(validationOptions, gatheredResults)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    resultStatus = EXIT_FAILURE;
                }
            }
        }
    }

    int globalResultStatus = EXIT_SUCCESS;
    MPI_Allreduce(&resultStatus, &globalResultStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return globalResultStatus;
}
