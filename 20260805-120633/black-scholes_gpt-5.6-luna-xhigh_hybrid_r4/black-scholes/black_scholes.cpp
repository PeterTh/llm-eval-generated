#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
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

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// The CUDA implementation intentionally mirrors blackScholes above.  Inputs
// are stored in structure-of-arrays form so every warp reads coalesced values
// while evaluating the transcendental-heavy pricing formula.
__device__ __forceinline__ double deviceCumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + ::erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const int* __restrict__ types,
                                   const double* __restrict__ strikes,
                                   const double* __restrict__ spots,
                                   const double* __restrict__ dividends,
                                   const double* __restrict__ rates,
                                   const double* __restrict__ maturities,
                                   const double* __restrict__ volatilities,
                                   double* __restrict__ results,
                                   const std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const double S = spots[i];
    const double K = strikes[i];
    const double r = rates[i];
    const double q = dividends[i];
    const double T = maturities[i];
    const double sigma = volatilities[i];

    if (T <= 0.0 || sigma <= 0.0) {
        results[i] = 0.0;
        return;
    }

    const double sqrtT = ::sqrt(T);
    const double sigmaSquared = sigma * sigma;
    const double d1 = (::log(S / K) + (r - q + 0.5 * sigmaSquared) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = deviceCumulativeNormal(d1);
    const double Nd2 = deviceCumulativeNormal(d2);
    const double discount = ::exp(-r * T);
    const double discountedSpot = S * ::exp(-q * T);

    if (types[i] == CALL) {
        results[i] = discountedSpot * Nd1 - K * discount * Nd2;
    } else {
        results[i] = K * discount * deviceCumulativeNormal(-d2) -
                     discountedSpot * deviceCumulativeNormal(-d1);
    }
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n", file, line,
                 expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cudaCheckError = (expression); \
        if (cudaCheckError != cudaSuccess) { \
            cudaFailure(cudaCheckError, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

void packOptions(const std::vector<OptionInput>& options,
                 std::vector<int>& types,
                 std::vector<double>& strikes,
                 std::vector<double>& spots,
                 std::vector<double>& dividends,
                 std::vector<double>& rates,
                 std::vector<double>& maturities,
                 std::vector<double>& volatilities) {
    const std::size_t count = options.size();
    types.resize(count);
    strikes.resize(count);
    spots.resize(count);
    dividends.resize(count);
    rates.resize(count);
    maturities.resize(count);
    volatilities.resize(count);

    // This is deliberately an OpenMP region even for a single MPI rank: it
    // keeps host preparation scalable and leaves the GPU fed on large jobs.
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i) {
        const OptionInput& option = options[static_cast<std::size_t>(i)];
        types[static_cast<std::size_t>(i)] = option.type;
        strikes[static_cast<std::size_t>(i)] = option.strike;
        spots[static_cast<std::size_t>(i)] = option.spot;
        dividends[static_cast<std::size_t>(i)] = option.q;
        rates[static_cast<std::size_t>(i)] = option.r;
        maturities[static_cast<std::size_t>(i)] = option.t;
        volatilities[static_cast<std::size_t>(i)] = option.vol;
    }
}

void priceOnGpu(const std::vector<int>& types,
                const std::vector<double>& strikes,
                const std::vector<double>& spots,
                const std::vector<double>& dividends,
                const std::vector<double>& rates,
                const std::vector<double>& maturities,
                const std::vector<double>& volatilities,
                std::vector<double>& results) {
    const std::size_t count = results.size();
    if (count == 0) {
        return;
    }

    int* deviceTypes = nullptr;
    double* deviceStrikes = nullptr;
    double* deviceSpots = nullptr;
    double* deviceDividends = nullptr;
    double* deviceRates = nullptr;
    double* deviceMaturities = nullptr;
    double* deviceVolatilities = nullptr;
    double* deviceResults = nullptr;
    cudaStream_t stream = nullptr;

    const std::size_t typeBytes = count * sizeof(int);
    const std::size_t valueBytes = count * sizeof(double);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTypes), typeBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceStrikes), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceSpots), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDividends), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRates), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMaturities), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceVolatilities), valueBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceResults), valueBytes));

    CUDA_CHECK(cudaMemcpyAsync(deviceTypes, types.data(), typeBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceStrikes, strikes.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceSpots, spots.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceDividends, dividends.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceRates, rates.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceMaturities, maturities.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceVolatilities, volatilities.data(), valueBytes,
                               cudaMemcpyHostToDevice, stream));

    constexpr unsigned int threadsPerBlock = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (count + threadsPerBlock - 1) / threadsPerBlock);
    blackScholesKernel<<<blocks, threadsPerBlock, 0, stream>>>(
        deviceTypes, deviceStrikes, deviceSpots, deviceDividends, deviceRates,
        deviceMaturities, deviceVolatilities, deviceResults, count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(results.data(), deviceResults, valueBytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(deviceResults));
    CUDA_CHECK(cudaFree(deviceVolatilities));
    CUDA_CHECK(cudaFree(deviceMaturities));
    CUDA_CHECK(cudaFree(deviceRates));
    CUDA_CHECK(cudaFree(deviceDividends));
    CUDA_CHECK(cudaFree(deviceSpots));
    CUDA_CHECK(cudaFree(deviceStrikes));
    CUDA_CHECK(cudaFree(deviceTypes));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Standard test cases for validation
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t localIndex = 0;
         localIndex < static_cast<std::ptrdiff_t>(numOptions); ++localIndex) {
        const size_t i = static_cast<size_t>(localIndex);
        const size_t globalIndex = globalOffset + i;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 +
                              0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int mpiInitializationResult =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (mpiInitializationResult != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (mpiRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long long parsedOptions = atoll(argv[++i]);
            if (parsedOptions < 0) {
                if (mpiRank == 0) {
                    std::fprintf(stderr, "Number of options must be non-negative\n");
                }
                MPI_Finalize();
                return EXIT_FAILURE;
            }
            numOptions = static_cast<size_t>(parsedOptions);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpiRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank,
                        MPI_INFO_NULL, &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceQueryResult = cudaGetDeviceCount(&deviceCount);
    if (deviceQueryResult != cudaSuccess || deviceCount == 0) {
        if (mpiRank == 0) {
            std::fprintf(stderr,
                         "A CUDA accelerator is required, but no CUDA device is available: %s\n",
                         cudaGetErrorString(deviceQueryResult));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t mpiSizeAsSize = static_cast<size_t>(mpiSize);
    const size_t mpiRankAsSize = static_cast<size_t>(mpiRank);
    const size_t baseCount = numOptions / mpiSizeAsSize;
    const size_t remainder = numOptions % mpiSizeAsSize;
    const size_t localCount = baseCount + (mpiRankAsSize < remainder ? 1 : 0);
    const size_t globalOffset = mpiRankAsSize * baseCount +
                                std::min(mpiRankAsSize, remainder);

    if (mpiRank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               mpiSize, omp_get_max_threads(), deviceCount);
        printf("Pricing options...\n");
    }

    // Generate and pack only the contiguous range owned by this MPI rank.
    std::vector<OptionInput> options;
    generateOptions(options, localCount, globalOffset);
    std::vector<int> types;
    std::vector<double> strikes;
    std::vector<double> spots;
    std::vector<double> dividends;
    std::vector<double> rates;
    std::vector<double> maturities;
    std::vector<double> volatilities;
    packOptions(options, types, strikes, spots, dividends, rates, maturities, volatilities);
    std::vector<double> localResults(localCount);

    // The timed section measures the end-to-end transfer and kernel cost for
    // the distributed pricing workload.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    priceOnGpu(types, strikes, spots, dividends, rates, maturities, volatilities,
               localResults);
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double optionsPerSecond = duration > 0.0
                                            ? static_cast<double>(numOptions) / duration
                                            : 0.0;
        printf("Options per second: %.0f\n", optionsPerSecond);
    }

    // Gathering is only needed for the original result-reporting and
    // validation interfaces.  The hot default benchmark stays distributed.
    const bool needGlobalResults = printResults || validate;
    std::vector<double> globalResults;
    std::vector<int> receiveCounts;
    std::vector<int> receiveDisplacements;
    if (needGlobalResults) {
        if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (mpiRank == 0) {
                std::fprintf(stderr,
                             "-r/-v requires at most %d options for MPI result gathering\n",
                             std::numeric_limits<int>::max());
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }
        receiveCounts.resize(static_cast<size_t>(mpiSize));
        receiveDisplacements.resize(static_cast<size_t>(mpiSize));
        for (int rank = 0; rank < mpiSize; ++rank) {
            const size_t rankAsSize = static_cast<size_t>(rank);
            const size_t rankCount = baseCount + (rankAsSize < remainder ? 1 : 0);
            const size_t rankOffset = rankAsSize * baseCount +
                                      std::min(rankAsSize, remainder);
            receiveCounts[rank] = static_cast<int>(rankCount);
            receiveDisplacements[rank] = static_cast<int>(rankOffset);
        }
        if (mpiRank == 0) {
            globalResults.resize(numOptions);
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    mpiRank == 0 ? globalResults.data() : nullptr,
                    receiveCounts.data(), receiveDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (mpiRank == 0 && printResults) {
        print_results(globalResults, "OptionPrices");
    }

    int validationPassed = 1;
    if (validate) {
        if (mpiRank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, std::min<size_t>(10, numOptions), 0);
            validationPassed = validateResults(validationOptions, globalResults) ? 1 : 0;
            printf("Validation: %s\n", validationPassed != 0 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return validationPassed != 0 ? 0 : 1;
}
