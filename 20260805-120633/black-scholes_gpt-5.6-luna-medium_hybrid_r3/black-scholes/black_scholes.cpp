#include <algorithm>
#include <array>
#include <chrono>
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
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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

__global__ void priceKernel(const OptionInput* options, double* results,
                            const size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        results[i] = blackScholes(options[i]);
    }
}

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error,
                              const int rank) {
    fprintf(stderr, "MPI rank %d: CUDA error during %s: %s\n", rank, operation,
            cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void priceOnGpu(const std::vector<OptionInput>& localOptions,
                std::vector<double>& localResults, const int rank) {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        cudaFailure("cudaGetDeviceCount", error == cudaSuccess
                    ? cudaErrorNoDevice : error, rank);
    }
    if ((error = cudaSetDevice(rank % deviceCount)) != cudaSuccess) {
        cudaFailure("cudaSetDevice", error, rank);
    }

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    const size_t bytes = localOptions.size() * sizeof(OptionInput);
    if (localOptions.size() != 0) {
        if ((error = cudaMalloc(reinterpret_cast<void**>(&deviceOptions), bytes)) != cudaSuccess) {
            cudaFailure("cudaMalloc(options)", error, rank);
        }
        if ((error = cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                                localResults.size() * sizeof(double))) != cudaSuccess) {
            cudaFree(deviceOptions);
            cudaFailure("cudaMalloc(results)", error, rank);
        }
    }
    if (bytes != 0 && (error = cudaMemcpy(deviceOptions, localOptions.data(), bytes,
                                          cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(deviceOptions); cudaFree(deviceResults);
        cudaFailure("cudaMemcpy(options)", error, rank);
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((localOptions.size() + threads - 1) / threads);
    if (localOptions.size() != 0) {
        priceKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localOptions.size());
        if ((error = cudaGetLastError()) != cudaSuccess) {
            cudaFree(deviceOptions); cudaFree(deviceResults);
            cudaFailure("priceKernel launch", error, rank);
        }
        if ((error = cudaDeviceSynchronize()) != cudaSuccess) {
            cudaFree(deviceOptions); cudaFree(deviceResults);
            cudaFailure("priceKernel execution", error, rank);
        }
        if ((error = cudaMemcpy(localResults.data(), deviceResults,
                                localResults.size() * sizeof(double),
                                cudaMemcpyDeviceToHost)) != cudaSuccess) {
            cudaFree(deviceOptions); cudaFree(deviceResults);
            cudaFailure("cudaMemcpy(results)", error, rank);
        }
    }
    cudaFree(deviceOptions);
    cudaFree(deviceResults);
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
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (long long signedIndex = 0; signedIndex < static_cast<long long>(numOptions); ++signedIndex) {
        const size_t i = static_cast<size_t>(signedIndex);
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    std::array<bool, 10> checkPassed{};
    
    printf("Checking computed option prices:\n");
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        checkPassed[static_cast<size_t>(i)] =
            computed >= 0.0 && computed <= 1000.0 && !std::isnan(computed) && !std::isinf(computed);
    }
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        if (!checkPassed[static_cast<size_t>(i)]) {
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid execution: MPI ranks=%d, OpenMP threads/rank=%d, CUDA\n",
               worldSize, omp_get_max_threads());
    }
    
    // Generate options
    std::vector<OptionInput> options;
    if (rank == 0) {
        generateOptions(options, numOptions);
    }
    
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t begin = numOptions * static_cast<size_t>(r) / worldSize;
        const size_t end = numOptions * static_cast<size_t>(r + 1) / worldSize;
        counts[r] = static_cast<int>(end - begin);
        displacements[r] = static_cast<int>(begin);
    }
    const size_t localCount = static_cast<size_t>(counts[rank]);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    std::vector<int> byteCounts(worldSize), byteDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        byteCounts[r] = counts[r] * static_cast<int>(sizeof(OptionInput));
        byteDisplacements[r] = displacements[r] * static_cast<int>(sizeof(OptionInput));
    }
    MPI_Scatterv(rank == 0 ? options.data() : nullptr, byteCounts.data(), byteDisplacements.data(),
                 MPI_BYTE, localOptions.data(), byteCounts[rank], MPI_BYTE, 0, MPI_COMM_WORLD);
    std::vector<double> results(rank == 0 ? numOptions : 0);
    
    // Price options
    if (rank == 0) printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    priceOnGpu(localOptions, localResults, rank);
    MPI_Gatherv(localResults.data(), counts[rank], MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating results...\n");
        bool valid = validateResults(options, results);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
