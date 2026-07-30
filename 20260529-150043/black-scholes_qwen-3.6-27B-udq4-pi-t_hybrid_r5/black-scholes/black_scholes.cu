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

// Standard normal cumulative distribution function (host)
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Black-Scholes formula for European options (host reference)
double blackScholesHost(const OptionInput& option) noexcept {
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

// ============================================================================
// CUDA device code
// ============================================================================

// Device-side cumulative normal using erff (device overload)
__device__ __forceinline__ double deviceCumulativeNormal(double x) {
    return 0.5 * (1.0 + erff(x * M_SQRT1_2));
}

// Device-side Black-Scholes pricing
__device__ __forceinline__ double deviceBlackScholes(
    int type, double strike, double spot,
    double q, double r, double t, double vol)
{
    const double S = spot;
    const double K = strike;
    const double T = t;
    const double sigma = vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = deviceCumulativeNormal(d1);
    const double Nd2 = deviceCumulativeNormal(d2);
    const double discount = exp(-r * T);

    if (type == 0) { // CALL
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        return K * discount * deviceCumulativeNormal(-d2)
             - S * exp(-q * T) * deviceCumulativeNormal(-d1);
    }
}

// CUDA kernel: one thread per option, uses Structure-of-Arrays layout
// for coalesced memory access
__global__ void blackScholesKernel(
    const int* __restrict__ types,
    const double* __restrict__ strikes,
    const double* __restrict__ spots,
    const double* __restrict__ qs,
    const double* __restrict__ rs,
    const double* __restrict__ ts,
    const double* __restrict__ vols,
    double* __restrict__ results,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = deviceBlackScholes(
            types[idx], strikes[idx], spots[idx],
            qs[idx], rs[idx], ts[idx], vols[idx]
        );
    }
}

// ============================================================================
// Host code
// ============================================================================

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

// Generate options with OpenMP parallelization
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

// Generate options with a global offset (for MPI distribution)
void generateOptionsOffset(std::vector<OptionInput>& options,
                           const size_t numOptions, const size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        size_t globalIdx = globalOffset + i;
        const OptionInput& base = testOptions[globalIdx % testOptions.size()];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (globalIdx / static_cast<double>(testOptions.size()));
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

// CUDA helper: choose optimal grid/block dimensions
static void getLaunchConfig(int n, int& blockSize, int& gridSize) {
    blockSize = 256;
    gridSize = (n + blockSize - 1) / blockSize;
}

// CUDA device count
static int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 only, then broadcast)
    if (rank == 0) {
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
    }

    // Broadcast parameters from rank 0 to all ranks
    MPI_Bcast(&numOptions, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int validateInt = validate;
    int printResultsInt = printResults;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt;
    printResults = printResultsInt;

    // Print configuration (rank 0 only)
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Parallelization: Hybrid MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute work across MPI ranks: each rank gets a contiguous chunk
    size_t baseChunk = numOptions / numRanks;
    size_t remainder = numOptions % numRanks;
    size_t localNumOptions = baseChunk + (rank < static_cast<int>(remainder) ? 1 : 0);

    // Compute global offset for this rank
    size_t globalOffset = baseChunk * rank + std::min(static_cast<size_t>(rank), remainder);

    // Generate local options with correct global indices (OpenMP parallelized)
    std::vector<OptionInput> localOptions;
    generateOptionsOffset(localOptions, localNumOptions, globalOffset);

    // Allocate local results
    std::vector<double> localResults(localNumOptions, 0.0);

    // Extract fields for CUDA kernel (Structure of Arrays for coalesced access)
    std::vector<int> h_types(localNumOptions);
    std::vector<double> h_strikes(localNumOptions);
    std::vector<double> h_spots(localNumOptions);
    std::vector<double> h_qs(localNumOptions);
    std::vector<double> h_rs(localNumOptions);
    std::vector<double> h_ts(localNumOptions);
    std::vector<double> h_vols(localNumOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localNumOptions; ++i) {
        h_types[i]   = localOptions[i].type;
        h_strikes[i] = localOptions[i].strike;
        h_spots[i]   = localOptions[i].spot;
        h_qs[i]      = localOptions[i].q;
        h_rs[i]      = localOptions[i].r;
        h_ts[i]      = localOptions[i].t;
        h_vols[i]    = localOptions[i].vol;
    }

    // Determine which GPU to use (round-robin across available devices)
    int numDevices = getDeviceCount();
    int deviceIdx = (numDevices > 0) ? (rank % numDevices) : 0;
    cudaSetDevice(deviceIdx);

    // CUDA device memory allocation
    int* dev_types = nullptr;
    double* dev_strikes = nullptr;
    double* dev_spots = nullptr;
    double* dev_qs = nullptr;
    double* dev_rs = nullptr;
    double* dev_ts = nullptr;
    double* dev_vols = nullptr;
    double* dev_results = nullptr;

    cudaMalloc(&dev_types,   localNumOptions * sizeof(int));
    cudaMalloc(&dev_strikes, localNumOptions * sizeof(double));
    cudaMalloc(&dev_spots,   localNumOptions * sizeof(double));
    cudaMalloc(&dev_qs,      localNumOptions * sizeof(double));
    cudaMalloc(&dev_rs,      localNumOptions * sizeof(double));
    cudaMalloc(&dev_ts,      localNumOptions * sizeof(double));
    cudaMalloc(&dev_vols,    localNumOptions * sizeof(double));
    cudaMalloc(&dev_results, localNumOptions * sizeof(double));

    // Copy data to device
    cudaMemcpyAsync(dev_types,   h_types.data(),   localNumOptions * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_strikes, h_strikes.data(), localNumOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_spots,   h_spots.data(),   localNumOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_qs,      h_qs.data(),      localNumOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_rs,      h_rs.data(),      localNumOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_ts,      h_ts.data(),      localNumOptions * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_vols,    h_vols.data(),    localNumOptions * sizeof(double), cudaMemcpyHostToDevice);

    // Launch CUDA kernel with optimal configuration
    int blockSize, gridSize;
    getLaunchConfig(static_cast<int>(localNumOptions), blockSize, gridSize);

    // Synchronize all ranks before timing starts
    MPI_Barrier(MPI_COMM_WORLD);

    // Use host-side timing with CUDA device synchronization for accuracy
    cudaDeviceSynchronize();
    auto cudaStart = std::chrono::high_resolution_clock::now();

    blackScholesKernel<<<gridSize, blockSize>>>(
        dev_types, dev_strikes, dev_spots,
        dev_qs, dev_rs, dev_ts, dev_vols,
        dev_results, static_cast<int>(localNumOptions)
    );

    cudaDeviceSynchronize();
    auto cudaEnd = std::chrono::high_resolution_clock::now();
    double gpuTimeMs = std::chrono::duration_cast<std::chrono::microseconds>(cudaEnd - cudaStart).count() / 1000.0;

    // Copy results back from device
    cudaMemcpy(localResults.data(), dev_results, localNumOptions * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_types);
    cudaFree(dev_strikes);
    cudaFree(dev_spots);
    cudaFree(dev_qs);
    cudaFree(dev_rs);
    cudaFree(dev_ts);
    cudaFree(dev_vols);
    cudaFree(dev_results);

    // Synchronize all ranks before gathering
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather all results to rank 0 using MPI_Gatherv
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);

    for (int r = 0; r < numRanks; ++r) {
        recvCounts[r] = baseChunk + (r < static_cast<int>(remainder) ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < numRanks; ++r) {
        displs[r] = displs[r - 1] + recvCounts[r - 1];
    }

    std::vector<double> globalResults;
    if (rank == 0) {
        globalResults.resize(numOptions);
    }

    MPI_Gatherv(localResults.data(), static_cast<int>(localNumOptions), MPI_DOUBLE,
                rank == 0 ? globalResults.data() : nullptr,
                recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Compute global timing via MPI reduction
    double maxTimeMs = 0.0;
    MPI_Reduce(&gpuTimeMs, &maxTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // All output and validation on rank 0 only
    if (rank == 0) {
        if (numRanks > 1) {
            printf("Pricing options...\n");
        }
        printf("Computation time: %.3f ms\n", maxTimeMs);
        printf("Options per second: %.0f\n", numOptions / (maxTimeMs / 1000.0));

        // Print results for external validation
        if (printResults) {
            print_results(globalResults, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            // Regenerate full options for validation
            std::vector<OptionInput> fullOptions;
            generateOptions(fullOptions, numOptions);

            bool valid = validateResults(fullOptions, globalResults);

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
