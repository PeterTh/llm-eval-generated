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

// ============================================================================
// CUDA device code
// ============================================================================

// Device-side Black-Scholes kernel
// Each thread prices one option independently — fully parallel, no shared state.
__global__ void blackScholesKernel(const OptionInput* options, double* results, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const OptionInput opt = options[idx];
    const double S = opt.spot;
    const double K = opt.strike;
    const double r = opt.r;
    const double q = opt.q;
    const double T = opt.t;
    const double sigma = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    // Device-side cumulative normal using erf() (double precision)
    const double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
    const double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
    const double discount = exp(-r * T);

    double price;
    if (opt.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * (0.5 * (1.0 + erf(-d2 * M_SQRT1_2)))
              - S * exp(-q * T) * (0.5 * (1.0 + erf(-d1 * M_SQRT1_2)));
    }

    results[idx] = price;
}

// ============================================================================
// Host-side helpers (for validation / fallback)
// ============================================================================

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
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
// Test data generation
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

// Generate options — parallelized with OpenMP
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// ============================================================================
// Validation
// ============================================================================

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

// ============================================================================
// Usage
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main — hybrid MPI + OpenMP + CUDA
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on root for display, all ranks parse)
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

    // Detect CUDA device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    int deviceId = 0;
    if (deviceCount > 0) {
        // Each MPI rank uses a unique GPU if available
        deviceId = rank % deviceCount;
        cudaSetDevice(deviceId);
    }

    int numThreads = 1;
    if (rank == 0) {
        // Print banner from root only
        printf("Black-Scholes Option Pricing Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", deviceCount);
        numThreads = omp_get_max_threads();
        printf("OpenMP threads: %d\n", numThreads);
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribute options across MPI ranks
    size_t localCount = numOptions / numRanks;
    size_t remainder = numOptions % numRanks;
    if (rank < remainder) {
        localCount++;
    }
    size_t localStart = (numOptions / numRanks) * rank + std::min(static_cast<size_t>(rank), remainder);

    // Generate full dataset on root, then scatter to all ranks
    std::vector<OptionInput> localOptions;
    if (rank == 0) {
        // Root generates all options in parallel (OpenMP)
        std::vector<OptionInput> allOptions;
        generateOptions(allOptions, numOptions);

        // Scatter to other ranks
        for (int r = 1; r < numRanks; ++r) {
            size_t rCount = numOptions / numRanks;
            size_t rRemainder = numOptions % numRanks;
            if (r < rRemainder) rCount++;
            size_t rStart = (numOptions / numRanks) * r + std::min(static_cast<size_t>(r), rRemainder);

            int msgSize = static_cast<int>(rCount * sizeof(OptionInput));
            MPI_Send(&allOptions[rStart], msgSize, MPI_BYTE, r, 0, MPI_COMM_WORLD);
        }

        // Keep local chunk
        localOptions.assign(allOptions.begin() + localStart, allOptions.begin() + localStart + localCount);
    } else {
        // Receive chunk from root
        localOptions.resize(localCount);
        int msgSize = static_cast<int>(localCount * sizeof(OptionInput));
        MPI_Recv(&localOptions[0], msgSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Allocate results
    std::vector<double> localResults(localCount);

    // ---- CUDA pricing phase ----
    printf("Rank %d: Pricing %zu options on GPU device %d...\n", rank, localCount, deviceId);
    fflush(stdout);

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate device memory
    OptionInput* dOptions = nullptr;
    double* dResults = nullptr;

    if (localCount > 0 && deviceCount > 0) {
        cudaMalloc(&dOptions, localCount * sizeof(OptionInput));
        cudaMalloc(&dResults, localCount * sizeof(double));

        // Copy options to device
        cudaMemcpy(dOptions, localOptions.data(), localCount * sizeof(OptionInput), cudaMemcpyHostToDevice);

        // Configure kernel launch parameters
        const int blockSize = 256;
        const int gridSize = static_cast<int>((localCount + blockSize - 1) / blockSize);

        // Launch Black-Scholes kernel on GPU
        blackScholesKernel<<<gridSize, blockSize>>>(dOptions, dResults, localCount);

        // Synchronize and copy results back
        cudaDeviceSynchronize();
        cudaMemcpy(localResults.data(), dResults, localCount * sizeof(double), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(dOptions);
        cudaFree(dResults);
    } else {
        // Fallback to OpenMP parallel CPU computation if no GPU or empty
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localCount; ++i) {
            localResults[i] = blackScholesHost(localOptions[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // ---- Gather results back to root for global reporting ----
    // Build displacement and count arrays for MPI_Gatherv
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    {
        int total = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t rCount = numOptions / numRanks;
            size_t rRemainder = numOptions % numRanks;
            if (r < rRemainder) rCount++;
            recvCounts[r] = static_cast<int>(rCount * sizeof(double));
            displs[r] = total;
            total += recvCounts[r];
        }
    }

    std::vector<double> globalResults;
    if (rank == 0) {
        globalResults.resize(numOptions);
    }

    MPI_Gatherv(localResults.data(), static_cast<int>(localCount * sizeof(double)), MPI_BYTE,
                globalResults.data(), recvCounts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Root prints timing and results
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

        // Print results for external validation
        if (printResults) {
            print_results(globalResults, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");

            // Re-generate reference options for comparison
            std::vector<OptionInput> refOptions;
            generateOptions(refOptions, numOptions);

            // Validate: check values are positive and reasonable (original semantics)
            bool valid = validateResults(refOptions, globalResults);

            // Verify CUDA kernel correctness against host reference for a sample
            bool kernelCorrect = true;
            const int numVerify = std::min(static_cast<int>(refOptions.size()), 1000);
            for (int i = 0; i < numVerify; ++i) {
                double refPrice = blackScholesHost(refOptions[i]);
                double computed = globalResults[i];
                if (fabs(computed - refPrice) > 1e-6) {
                    printf("  Kernel mismatch at option %d: gpu=%.10f, cpu=%.10f\n",
                           i, computed, refPrice);
                    kernelCorrect = false;
                    break;
                }
            }

            if (valid && kernelCorrect) {
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
