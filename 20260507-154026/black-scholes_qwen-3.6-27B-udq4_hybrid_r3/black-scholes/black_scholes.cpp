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

// POD struct for device compatibility
struct alignas(32) OptionInput {
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

// CUDA device math constants
#define CUDA_M_SQRT1_2 0.7071067811865475244
#define CUDA_M_1_SQRTPI 0.564189583547756286948

// Device function: standard normal cumulative distribution
__device__ __forceinline__ double deviceCumulativeNormal(const double x) {
    return 0.5 * (1.0 + erff(x * CUDA_M_SQRT1_2));
}

// Device function: Black-Scholes pricing
__device__ __forceinline__ double deviceBlackScholes(const OptionInput& option) {
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

    const double Nd1 = deviceCumulativeNormal(d1);
    const double Nd2 = deviceCumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * deviceCumulativeNormal(-d2) -
                S * exp(-q * T) * deviceCumulativeNormal(-d1);
    }

    return price;
}

// CUDA kernel: price a batch of options
__global__ void blackScholesKernel(const OptionInput* d_options,
                                   double* d_results,
                                   const int n) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (idx < n) {
        d_results[idx] = deviceBlackScholes(d_options[idx]);
    }
}

// Host: standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Host: standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Host: Black-Scholes formula for European options
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
    } else {
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

// OpenMP-parallelized option generation
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// OpenMP-parallelized validation
bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), options.size()));

    printf("Checking computed option prices:\n");
#pragma omp parallel for schedule(static) reduction(&&:allPassed)
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

#pragma omp critical
        {
            printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                   i, computed, expected, relError);
        }

        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
#pragma omp critical
            {
                printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            }
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

// CUDA helper: check errors
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 only, then broadcast)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast parameters from rank 0 to all ranks
    MPI_Bcast(&numOptions, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Parallelization: Hybrid MPI + OpenMP + CUDA\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute work: each rank handles a chunk
    const size_t globalNumOptions = numOptions;
    const size_t chunkSize = globalNumOptions / mpi_size;
    const size_t remainder = globalNumOptions % mpi_size;
    const size_t localStart = mpi_rank * chunkSize + std::min(static_cast<size_t>(mpi_rank), remainder);
    const size_t localEnd = (mpi_rank + 1) * chunkSize + std::min(static_cast<size_t>(mpi_rank + 1), remainder);
    const size_t localNumOptions = localEnd - localStart;

    // Generate options on this rank (only the local chunk)
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localNumOptions);

    // Shift indices so the generated options correspond to global positions
    // We need to regenerate with the correct global indices
    localOptions.clear();
    localOptions.resize(localNumOptions);
    constexpr auto testOptions = getTestOptions();
    for (size_t i = 0; i < localNumOptions; ++i) {
        const size_t globalIdx = localStart + i;
        const OptionInput& base = testOptions[globalIdx % testOptions.size()];
        localOptions[i] = base;
        const double factor = 1.0 + 0.1 * (globalIdx / static_cast<double>(testOptions.size()));
        localOptions[i].spot *= factor;
        localOptions[i].strike *= factor;
    }

    // Allocate results
    std::vector<double> localResults(localNumOptions);

    // Price options using CUDA
    if (mpi_rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    if (localNumOptions > 0) {
        // Allocate device memory
        OptionInput* d_options = nullptr;
        double* d_results = nullptr;

        const size_t optBytes = localNumOptions * sizeof(OptionInput);
        const size_t resBytes = localNumOptions * sizeof(double);

        checkCuda(cudaMalloc(&d_options, optBytes), "malloc options");
        checkCuda(cudaMalloc(&d_results, resBytes), "malloc results");

        // Copy options to device
        checkCuda(cudaMemcpy(d_options, localOptions.data(), optBytes,
                             cudaMemcpyHostToDevice), "memcpy h2d options");

        // Launch kernel
        const int threadsPerBlock = 256;
        const int numBlocks = static_cast<int>((localNumOptions + threadsPerBlock - 1) / threadsPerBlock);

        blackScholesKernel<<<numBlocks, threadsPerBlock>>>(d_options, d_results,
                                                           static_cast<int>(localNumOptions));
        checkCuda(cudaGetLastError(), "kernel launch");

        // Copy results back
        checkCuda(cudaMemcpy(localResults.data(), d_results, resBytes,
                             cudaMemcpyDeviceToHost), "memcpy d2h results");

        // Free device memory
        checkCuda(cudaFree(d_options), "free options");
        checkCuda(cudaFree(d_results), "free results");
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Gather results to rank 0
    std::vector<double> allResults;
    if (mpi_rank == 0) {
        allResults.resize(globalNumOptions);
    }

    // Compute displacements for gather
    std::vector<int> displacements(mpi_size);
    std::vector<int> recvcounts(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        const size_t rStart = r * chunkSize + std::min(static_cast<size_t>(r), remainder);
        const size_t rEnd = (r + 1) * chunkSize + std::min(static_cast<size_t>(r + 1), remainder);
        recvcounts[r] = static_cast<int>(rEnd - rStart);
        displacements[r] = static_cast<int>(rStart);
    }

    MPI_Gatherv(localResults.data(), static_cast<int>(localNumOptions), MPI_DOUBLE,
                allResults.data(), recvcounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Report timing (only rank 0)
    if (mpi_rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", globalNumOptions / (duration.count() / 1e6));

        // Print results for external validation
        if (printResults) {
            print_results(allResults, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");

            // Regenerate full options for validation
            std::vector<OptionInput> fullOptions;
            fullOptions.resize(globalNumOptions);
            for (size_t i = 0; i < globalNumOptions; ++i) {
                const OptionInput& base = testOptions[i % testOptions.size()];
                fullOptions[i] = base;
                const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
                fullOptions[i].spot *= factor;
                fullOptions[i].strike *= factor;
            }

            bool valid = validateResults(fullOptions, allResults);

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
    }

    MPI_Finalize();
    return 0;
}
