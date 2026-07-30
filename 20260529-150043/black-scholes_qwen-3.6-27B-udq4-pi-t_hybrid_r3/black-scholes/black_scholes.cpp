#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>
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
// CUDA device-side math helpers (must be __device__)
// ---------------------------------------------------------------------------
__device__ __forceinline__ double d_cumulativeNormal(const double x) {
    return 0.5 * (1.0 + erff(x * M_SQRT1_2));
}

__device__ __forceinline__ double d_normalPDF(const double x) {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// ---------------------------------------------------------------------------
// CUDA kernel: one thread per option
// ---------------------------------------------------------------------------
__global__ void blackScholesKernel(const OptionInput* options,
                                   double* results,
                                   const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const OptionInput opt = options[idx];

    const double S     = opt.spot;
    const double K     = opt.strike;
    const double r     = opt.r;
    const double q     = opt.q;
    const double T     = opt.t;
    const double sigma = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1  = d_cumulativeNormal(d1);
    const double Nd2  = d_cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (opt.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * d_cumulativeNormal(-d2)
              - S * exp(-q * T) * d_cumulativeNormal(-d1);
    }

    results[idx] = price;
}

// ---------------------------------------------------------------------------
// Host-side helpers (kept for validation / small data paths)
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

// Generate options with OpenMP parallelism
// globalOffset lets each MPI rank produce the correct global-indexed options
void generateOptions(std::vector<OptionInput>& options,
                     const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t g = globalOffset + i;  // global index
        const OptionInput& base = testOptions[g % testOptions.size()];
        options[i] = base;

        const double factor = 1.0 + 0.1 * (g / static_cast<double>(testOptions.size()));
        options[i].spot   *= factor;
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

// ---------------------------------------------------------------------------
// CUDA device selection: each MPI rank picks a unique GPU
// ---------------------------------------------------------------------------
static void selectCudaDevice(int rank, int /*numRanks*/) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Map rank to device in a round-robin fashion
    const int device = rank % deviceCount;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("[Rank %d] Using CUDA device %d: %s (%.1f GB)\n",
           rank, device, prop.name, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Initialize MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (!mpiInitialized) {
        MPI_Init(&argc, &argv);
    }

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Parse command line arguments (only on rank 0 for defaults, broadcast)
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

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
                if (rank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters from rank 0 to all ranks
    MPI_Bcast(&numOptions, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select CUDA device for this rank
    selectCudaDevice(rank, numRanks);

    // -----------------------------------------------------------------------
    // Distribute work across MPI ranks
    // -----------------------------------------------------------------------
    // Compute local chunk size using floor division with remainder distribution
    const size_t baseChunk = numOptions / numRanks;
    const size_t remainder = numOptions % numRanks;
    const size_t localN = baseChunk + (rank < static_cast<int>(remainder) ? 1 : 0);
    const size_t globalOffset = rank * baseChunk + std::min(static_cast<size_t>(rank), remainder);

    // -----------------------------------------------------------------------
    // Generate options (OpenMP parallelized) — each rank generates its chunk
    // -----------------------------------------------------------------------
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localN, globalOffset);

    // For validation, rank 0 needs the full option set
    std::vector<OptionInput> fullOptions;
    if (validate && rank == 0) {
        fullOptions.resize(numOptions);
        // Gather all options to rank 0
        std::vector<int> recvcounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t rc = baseChunk + (r < static_cast<int>(remainder) ? 1 : 0);
            recvcounts[r] = static_cast<int>(rc * sizeof(OptionInput));
            const size_t off = r * baseChunk + std::min(static_cast<size_t>(r), remainder);
            displs[r] = static_cast<int>(off * sizeof(OptionInput));
        }
        MPI_Gatherv(localOptions.data(), static_cast<int>(localN * sizeof(OptionInput)),
                    MPI_BYTE,
                    fullOptions.data(), recvcounts.data(), displs.data(),
                    MPI_BYTE, 0, MPI_COMM_WORLD);
    } else if (validate) {
        MPI_Gatherv(localOptions.data(), static_cast<int>(localN * sizeof(OptionInput)),
                    MPI_BYTE,
                    nullptr, nullptr, nullptr,
                    MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // -----------------------------------------------------------------------
    // CUDA pricing on GPU
    // -----------------------------------------------------------------------
    // Allocate device memory
    OptionInput* d_options = nullptr;
    double* d_results = nullptr;
    const size_t optBytes = localN * sizeof(OptionInput);
    const size_t resBytes = localN * sizeof(double);

    cudaMalloc(&d_options, optBytes);
    cudaMalloc(&d_results, resBytes);

    // Copy options to device
    cudaMemcpy(d_options, localOptions.data(), optBytes, cudaMemcpyHostToDevice);

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = (static_cast<int>(localN) + blockSize - 1) / blockSize;

    std::vector<double> localResults(localN);

    auto start = std::chrono::high_resolution_clock::now();

    blackScholesKernel<<<gridSize, blockSize>>>(d_options, d_results, static_cast<int>(localN));

    // Synchronize and copy results back
    cudaDeviceSynchronize();
    cudaMemcpy(localResults.data(), d_results, resBytes, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Clean up device memory
    cudaFree(d_options);
    cudaFree(d_results);

    // -----------------------------------------------------------------------
    // Gather results to rank 0 for printing / validation
    // -----------------------------------------------------------------------
    std::vector<double> globalResults;
    if (rank == 0) {
        globalResults.resize(numOptions);
    }

    // Gather counts and displacements for double arrays
    std::vector<int> recvcounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t rc = baseChunk + (r < static_cast<int>(remainder) ? 1 : 0);
        recvcounts[r] = static_cast<int>(rc);
        const size_t off = r * baseChunk + std::min(static_cast<size_t>(r), remainder);
        displs[r] = static_cast<int>(off);
    }

    if (rank == 0) {
        MPI_Gatherv(localResults.data(), static_cast<int>(localN),
                    MPI_DOUBLE,
                    globalResults.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(localResults.data(), static_cast<int>(localN),
                    MPI_DOUBLE,
                    nullptr, nullptr, nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // -----------------------------------------------------------------------
    // Print timing and results (rank 0 only)
    // -----------------------------------------------------------------------
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
            bool valid = validateResults(fullOptions, globalResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
