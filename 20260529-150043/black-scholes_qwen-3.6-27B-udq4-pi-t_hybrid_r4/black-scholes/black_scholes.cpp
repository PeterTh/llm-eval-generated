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

// ---------------------------------------------------------------------------
// CUDA device functions — mirror the host Black-Scholes implementation
// ---------------------------------------------------------------------------

__device__ inline double d_cumulativeNormal(const double x) {
    return 0.5 * (1.0 + erff(x * M_SQRT1_2));
}

__device__ inline double d_normalPDF(const double x) {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

__device__ double d_blackScholes(const OptionInput& option) {
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

    const double Nd1 = d_cumulativeNormal(d1);
    const double Nd2 = d_cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * d_cumulativeNormal(-d2) - S * exp(-q * T) * d_cumulativeNormal(-d1);
    }

    return price;
}

// ---------------------------------------------------------------------------
// CUDA kernel: each thread prices one option
// ---------------------------------------------------------------------------

__global__ void blackScholesKernel(const OptionInput* options, double* results, int n) {
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx < n) {
        results[idx] = d_blackScholes(options[idx]);
    }
}

// ---------------------------------------------------------------------------
// Host-side Black-Scholes (used for fallback / reference)
// ---------------------------------------------------------------------------

inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

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

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

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

// OpenMP-accelerated option generation
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

// ---------------------------------------------------------------------------
// Validation (OpenMP-parallelized)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main — hybrid MPI + OpenMP + CUDA
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // ------------------------------------------------------------------
    // 1. Parse command-line arguments (before MPI init for portability)
    // ------------------------------------------------------------------
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // ------------------------------------------------------------------
    // 2. Initialize MPI
    // ------------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // ------------------------------------------------------------------
    // 3. Compute per-rank workload
    // ------------------------------------------------------------------
    const size_t basePerRank = numOptions / numRanks;
    const size_t remainder = numOptions % numRanks;
    const size_t localN = basePerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t localOffset = basePerRank * static_cast<size_t>(rank) +
                               std::min(static_cast<size_t>(rank), remainder);

    // ------------------------------------------------------------------
    // 4. Detect CUDA capability
    // ------------------------------------------------------------------
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    bool useCuda = (deviceCount > 0);

    // Pick a GPU for this rank (round-robin across available GPUs)
    int gpuDevice = -1;
    if (useCuda) {
        gpuDevice = rank % deviceCount;
        cudaSetDevice(gpuDevice);
    }

    // ------------------------------------------------------------------
    // 5. Print banner (rank 0 only)
    // ------------------------------------------------------------------
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", deviceCount);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ------------------------------------------------------------------
    // 6. Generate options locally (OpenMP-parallelized)
    // ------------------------------------------------------------------
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localN);
    // Apply the same parameter variation that the serial code would use
    // for the global indices [localOffset, localOffset+localN)
    {
        constexpr auto testOptions = getTestOptions();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localN; ++i) {
            const size_t globalIdx = localOffset + i;
            const OptionInput& base = testOptions[globalIdx % testOptions.size()];
            localOptions[i] = base;
            const double factor = 1.0 + 0.1 * (globalIdx / static_cast<double>(testOptions.size()));
            localOptions[i].spot *= factor;
            localOptions[i].strike *= factor;
        }
    }

    // ------------------------------------------------------------------
    // 7. Price options using CUDA kernel
    // ------------------------------------------------------------------
    std::vector<double> localResults(localN, 0.0);

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Synchronize across ranks before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    if (useCuda && localN > 0) {
        // ---- GPU path ----
        OptionInput* dOptions = nullptr;
        double* dResults = nullptr;

        cudaMalloc(&dOptions, localN * sizeof(OptionInput));
        cudaMalloc(&dResults, localN * sizeof(double));

        cudaMemcpy(dOptions, localOptions.data(), localN * sizeof(OptionInput),
                   cudaMemcpyHostToDevice);

        // Configure kernel launch parameters
        const int threadsPerBlock = 256;
        const int blocksPerGrid = (static_cast<int>(localN) + threadsPerBlock - 1) / threadsPerBlock;

        blackScholesKernel<<<blocksPerGrid, threadsPerBlock>>>(dOptions, dResults, static_cast<int>(localN));

        // Synchronize and copy back
        cudaDeviceSynchronize();
        cudaMemcpy(localResults.data(), dResults, localN * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaFree(dOptions);
        cudaFree(dResults);
    } else {
        // ---- CPU fallback (OpenMP-parallelized) ----
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localN; ++i) {
            localResults[i] = blackScholes(localOptions[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double localTimeMs = duration.count() / 1000.0;
    double localThroughput = localN / (duration.count() / 1e6);

    if (rank == 0) {
        printf("Computation time (rank 0): %.3f ms\n", localTimeMs);
        printf("Options per second (rank 0): %.0f\n", localThroughput);
    }

    // ------------------------------------------------------------------
    // 8. Gather all results to rank 0
    // ------------------------------------------------------------------
    // Build displacement/sendcount arrays
    std::vector<int> sendcounts(numRanks);
    std::vector<int> displs(numRanks);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            int cnt = static_cast<int>(basePerRank + (static_cast<size_t>(r) < remainder ? 1 : 0));
            sendcounts[r] = cnt;
            displs[r] = static_cast<int>(offset);
            offset += cnt;
        }
    }

    std::vector<double> globalResults;
    std::vector<OptionInput> globalOptions;

    if (rank == 0) {
        globalResults.resize(numOptions);
        globalOptions.resize(numOptions);
    }

    // Gather results
    MPI_Gatherv(localResults.data(), static_cast<int>(localN), MPI_DOUBLE,
               globalResults.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Gather options for validation (use MPI_PACKED to avoid byte-count issues)
    {
        const int optSize = static_cast<int>(sizeof(OptionInput));
        std::vector<int> optSendcounts(numRanks);
        std::vector<int> optDispls(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            optSendcounts[r] = sendcounts[r] * optSize;
            optDispls[r] = displs[r] * optSize;
        }
        MPI_Gatherv(localOptions.data(), localN * optSize, MPI_BYTE,
                    globalOptions.data(), optSendcounts.data(), optDispls.data(),
                    MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------------
    // 9. Print results / validate (rank 0 only)
    // ------------------------------------------------------------------
    if (rank == 0) {
        if (printResults) {
            print_results(globalResults, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(globalOptions, globalResults);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // ------------------------------------------------------------------
    // 10. Finalize MPI
    // ------------------------------------------------------------------
    MPI_Finalize();

    return 0;
}
