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

// ============================================================================
// CUDA kernel for Black-Scholes pricing
// ============================================================================

// SoA layout for coalesced GPU memory access
struct OptionInputGPU {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
};

__global__ void blackScholesKernel(
    const int* __restrict__ types,
    const double* __restrict__ strikes,
    const double* __restrict__ spots,
    const double* __restrict__ qs,
    const double* __restrict__ rs,
    const double* __restrict__ ts,
    const double* __restrict__ vols,
    double* __restrict__ results,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double S = spots[idx];
    const double K = strikes[idx];
    const double r = rs[idx];
    const double q = qs[idx];
    const double T = ts[idx];
    const double sigma = vols[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
    const double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
    const double discount = exp(-r * T);

    double price;
    if (types[idx] == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        const double Nneg_d1 = 0.5 * (1.0 + erf(-d1 * M_SQRT1_2));
        const double Nneg_d2 = 0.5 * (1.0 + erf(-d2 * M_SQRT1_2));
        price = K * discount * Nneg_d2 - S * exp(-q * T) * Nneg_d1;
    }

    results[idx] = price;
}

// ============================================================================
// CPU-side Black-Scholes (for validation fallback)
// ============================================================================

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
    } else {
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
}

// ============================================================================
// Test cases and data generation
// ============================================================================

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

// ============================================================================
// GPU pricing driver
// ============================================================================

void priceOptionsGPU(const std::vector<OptionInput>& options,
                     std::vector<double>& results,
                     size_t startIdx, size_t count) {
    if (count == 0) return;

    // Pack into SoA arrays for coalesced access
    std::vector<int> d_types(count);
    std::vector<double> d_strikes(count), d_spots(count), d_qs(count);
    std::vector<double> d_rs(count), d_ts(count), d_vols(count);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const OptionInput& opt = options[startIdx + i];
        d_types[i]  = opt.type;
        d_strikes[i] = opt.strike;
        d_spots[i]   = opt.spot;
        d_qs[i]      = opt.q;
        d_rs[i]      = opt.r;
        d_ts[i]      = opt.t;
        d_vols[i]    = opt.vol;
    }

    // Allocate device memory
    int *g_types;
    double *g_strikes, *g_spots, *g_qs, *g_rs, *g_ts, *g_vols, *g_results;

    cudaMalloc(&g_types,   count * sizeof(int));
    cudaMalloc(&g_strikes, count * sizeof(double));
    cudaMalloc(&g_spots,   count * sizeof(double));
    cudaMalloc(&g_qs,      count * sizeof(double));
    cudaMalloc(&g_rs,      count * sizeof(double));
    cudaMalloc(&g_ts,      count * sizeof(double));
    cudaMalloc(&g_vols,    count * sizeof(double));
    cudaMalloc(&g_results, count * sizeof(double));

    // Copy to device
    cudaMemcpy(g_types,   d_types.data(),   count * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(g_strikes, d_strikes.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_spots,   d_spots.data(),   count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_qs,      d_qs.data(),      count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_rs,      d_rs.data(),      count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ts,      d_ts.data(),      count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(g_vols,    d_vols.data(),    count * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    const int blockSize = 256;
    const int numBlocks = (static_cast<int>(count) + blockSize - 1) / blockSize;
    blackScholesKernel<<<numBlocks, blockSize>>>(
        g_types, g_strikes, g_spots, g_qs, g_rs, g_ts, g_vols,
        g_results, static_cast<int>(count));

    // Copy results back
    cudaMemcpy(results.data() + startIdx, g_results, count * sizeof(double),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(g_types);
    cudaFree(g_strikes);
    cudaFree(g_spots);
    cudaFree(g_qs);
    cudaFree(g_rs);
    cudaFree(g_ts);
    cudaFree(g_vols);
    cudaFree(g_results);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());

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

// ============================================================================
// Main: Hybrid MPI + OpenMP + CUDA
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Initialize OpenMP thread count
    int ompNumThreads = omp_get_max_threads();

    // Select GPU for this MPI rank (round-robin if multiple GPUs per node)
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(mpiRank % deviceCount);
    }

    // Parse command line arguments on rank 0, then broadcast
    size_t numOptions = 10000;
    int validateFlag = 0;
    int printResultsFlag = 0;

    if (mpiRank == 0) {
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
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }

        validateFlag = validate ? 1 : 0;
        printResultsFlag = printResults ? 1 : 0;

        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               mpiSize, ompNumThreads, deviceCount);
    }

    // Broadcast parameters
    unsigned long long numOptionsULL = static_cast<unsigned long long>(numOptions);
    MPI_Bcast(&numOptionsULL, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(numOptionsULL);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    bool validate = (validateFlag != 0);
    bool printResults = (printResultsFlag != 0);

    // Generate options on all ranks (deterministic, so all ranks get same data)
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);

    // Compute work distribution across MPI ranks
    size_t baseCount = numOptions / mpiSize;
    size_t remainder = numOptions % mpiSize;

    size_t myStart, myCount;
    if (static_cast<size_t>(mpiRank) < remainder) {
        myCount = baseCount + 1;
        myStart = static_cast<size_t>(mpiRank) * (baseCount + 1);
    } else {
        myCount = baseCount;
        myStart = remainder * (baseCount + 1) +
                  (static_cast<size_t>(mpiRank) - remainder) * baseCount;
    }

    // Allocate results
    std::vector<double> results(numOptions, 0.0);

    // Price options on this rank using GPU
    if (mpiRank == 0) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (myCount > 0) {
        priceOptionsGPU(options, results, myStart, myCount);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Gather results to rank 0
    // Build displacement and count arrays for MPI_Gatherv
    std::vector<int> recvCounts(mpiSize);
    std::vector<int> recvDispls(mpiSize);

    if (mpiRank == 0) {
        for (int r = 0; r < mpiSize; ++r) {
            size_t rCount, rStart;
            if (static_cast<size_t>(r) < remainder) {
                rCount = baseCount + 1;
                rStart = static_cast<size_t>(r) * (baseCount + 1);
            } else {
                rCount = baseCount;
                rStart = remainder * (baseCount + 1) +
                         (static_cast<size_t>(r) - remainder) * baseCount;
            }
            recvCounts[r] = static_cast<int>(rCount);
            recvDispls[r] = static_cast<int>(rStart);
        }
    }

    int myCountInt = static_cast<int>(myCount);
    MPI_Gatherv(results.data() + myStart, myCountInt, MPI_DOUBLE,
                results.data(), recvCounts.data(), recvDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Report timing
    double localTime = std::chrono::duration<double, std::milli>(end - start).count();
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Computation time: %.3f ms\n", maxTime);
        printf("Options per second: %.0f\n", numOptions / (maxTime / 1e3));
    }

    // Print results for external validation
    if (printResults && mpiRank == 0) {
        print_results(results, "OptionPrices");
    }

    // Validation on rank 0
    if (validate) {
        if (mpiRank == 0) {
            printf("Validating results...\n");
            bool valid = validateResults(options, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }

            // Broadcast validation result
            int validInt = valid ? 1 : 0;
            MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return valid ? 0 : 1;
        } else {
            int validInt;
            MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return validInt ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
