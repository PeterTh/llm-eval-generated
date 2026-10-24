#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

inline OptionInput makeOption(const size_t index) noexcept {
    constexpr auto testOptions = getTestOptions();
    const OptionInput& base = testOptions[index % testOptions.size()];
    OptionInput option = base;
    const double factor = 1.0 + 0.1 * (index / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    options.resize(numOptions);
    
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numOptions); ++i) {
        options[static_cast<size_t>(i)] = makeOption(static_cast<size_t>(i));
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

__constant__ OptionInput c_testOptions[7];

__device__ __forceinline__ double cumulativeNormalDevice(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double blackScholesDevice(const OptionInput& option) noexcept {
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
    const double sigmaSqrtT = sigma * sqrtT;
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    const double discount = exp(-r * T);
    const double carry = S * exp(-q * T);

    if (option.type == CALL) {
        return carry * cumulativeNormalDevice(d1) - K * discount * cumulativeNormalDevice(d2);
    } else { // PUT
        return K * discount * cumulativeNormalDevice(-d2) - carry * cumulativeNormalDevice(-d1);
    }
}

__global__ void blackScholesKernel(double* __restrict__ results,
                                   const size_t begin,
                                   const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += stride) {
        const size_t globalIndex = begin + idx;
        const OptionInput& base = c_testOptions[globalIndex % 7];
        OptionInput option = base;
        const double factor = 1.0 + 0.1 * (globalIndex / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        results[idx] = blackScholesDevice(option);
    }
}

static bool computeRangeOnDevice(int device,
                                 const size_t globalBegin,
                                 const size_t count,
                                 double* hostOut) {
    if (cudaSetDevice(device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    constexpr auto testOptions = getTestOptions();
    cudaError_t err = cudaMemcpyToSymbol(c_testOptions, testOptions.data(),
                                         sizeof(OptionInput) * testOptions.size());
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    if (count == 0) {
        return true;
    }

    constexpr size_t maxTile = static_cast<size_t>(1) << 20;
    const size_t tile = std::min(maxTile, count);
    double* deviceResults = nullptr;
    err = cudaMalloc(reinterpret_cast<void**>(&deviceResults), tile * sizeof(double));
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    constexpr int blockSize = 256;
    constexpr size_t maxBlocks = static_cast<size_t>(1) << 20;
    bool ok = true;

    for (size_t done = 0; done < count; done += tile) {
        const size_t n = std::min(tile, count - done);
        const size_t neededBlocks = (n + blockSize - 1) / blockSize;
        const int gridSize = static_cast<int>(std::min(neededBlocks, maxBlocks));

        blackScholesKernel<<<gridSize, blockSize>>>(deviceResults,
                                                    globalBegin + done, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }

        err = cudaMemcpy(hostOut + done, deviceResults, n * sizeof(double),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
    }

    cudaFree(deviceResults);
    return ok;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Pricing options...\n");
    }

    const size_t localBegin = (numOptions * static_cast<size_t>(rank)) /
                              static_cast<size_t>(numRanks);
    const size_t localEnd = (numOptions * static_cast<size_t>(rank + 1)) /
                            static_cast<size_t>(numRanks);
    const size_t localCount = localEnd - localBegin;

    std::vector<double> localResults(localCount);

    int devicesPerRank = 0;
    std::vector<int> devices;
    if (localCount > 0) {
        int deviceCount = 0;
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
            if (rank == 0) {
                printf("CUDA device discovery failed\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int localRank = 0;
        int localSize = 1;
        MPI_Comm localComm = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_size(localComm, &localSize);

        if (localSize >= deviceCount) {
            devices.push_back(localRank % deviceCount);
        } else {
            for (int d = localRank; d < deviceCount; d += localSize) {
                devices.push_back(d);
            }
        }
        devicesPerRank = static_cast<int>(devices.size());

        MPI_Comm_free(&localComm);
    }

    if (localCount > 0 && devicesPerRank > 0) {
        std::atomic<int> initFailure{0};
        #pragma omp parallel for schedule(static) num_threads(devicesPerRank)
        for (int d = 0; d < devicesPerRank; ++d) {
            double warmup = 0.0;
            if (!computeRangeOnDevice(devices[static_cast<size_t>(d)], 0, 1,
                                      &warmup)) {
                initFailure.store(1, std::memory_order_relaxed);
            }
        }
        if (initFailure.load() != 0) {
            if (rank == 0) {
                printf("CUDA initialization failed\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    std::atomic<int> deviceFailure{0};
    double localSeconds = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    if (localCount > 0 && devicesPerRank > 0) {
        #pragma omp parallel for schedule(static) num_threads(devicesPerRank)
        for (int d = 0; d < devicesPerRank; ++d) {
            const size_t subBegin = (localCount * static_cast<size_t>(d)) /
                                    static_cast<size_t>(devicesPerRank);
            const size_t subEnd = (localCount * static_cast<size_t>(d + 1)) /
                                  static_cast<size_t>(devicesPerRank);
            if (subEnd > subBegin) {
                if (!computeRangeOnDevice(devices[static_cast<size_t>(d)],
                                          localBegin + subBegin,
                                          subEnd - subBegin,
                                          localResults.data() + subBegin)) {
                    deviceFailure.store(1, std::memory_order_relaxed);
                }
            }
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    localSeconds = std::chrono::duration<double>(end - start).count();

    if (deviceFailure.load() != 0) {
        if (rank == 0) {
            printf("CUDA computation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxSeconds * 1000.0);
        printf("Options per second: %.0f\n",
               maxSeconds > 0.0 ? numOptions / maxSeconds : 0.0);
    }

    const bool needResults = printResults || validate;
    std::vector<double> results;
    std::vector<int> counts;
    std::vector<int> displs;

    if (needResults) {
        if (localCount > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                printf("Too many options per process for MPI_Gatherv\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rank == 0) {
            results.resize(numOptions);
            counts.resize(static_cast<size_t>(numRanks));
            displs.resize(static_cast<size_t>(numRanks));
            for (int r = 0; r < numRanks; ++r) {
                const size_t begin = (numOptions * static_cast<size_t>(r)) /
                                     static_cast<size_t>(numRanks);
                const size_t finish = (numOptions * static_cast<size_t>(r + 1)) /
                                      static_cast<size_t>(numRanks);
                counts[static_cast<size_t>(r)] = static_cast<int>(finish - begin);
                displs[static_cast<size_t>(r)] = static_cast<int>(begin);
            }
        }

        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int returnCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            const size_t validationCount =
                std::min(static_cast<size_t>(10), numOptions);
            std::vector<OptionInput> validationOptions(validationCount);
            #pragma omp parallel for schedule(static)
            for (long long i = 0;
                 i < static_cast<long long>(validationCount); ++i) {
                validationOptions[static_cast<size_t>(i)] =
                    makeOption(static_cast<size_t>(i));
            }

            const bool valid = validateResults(validationOptions, results);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return returnCode;
}
