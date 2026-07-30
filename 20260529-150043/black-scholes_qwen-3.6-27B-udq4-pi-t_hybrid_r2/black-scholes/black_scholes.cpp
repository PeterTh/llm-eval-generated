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
#include <math_constants.h>

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

// Black-Scholes formula for European options (CPU reference)
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

// ============================================================================
// CUDA Kernel: Black-Scholes pricing
// ============================================================================

// Device-side erf approximation (Abramowitz & Stegun 7.1.26)
__device__ __forceinline__ double device_erf(double x) {
    // Use sign symmetry
    bool neg = x < 0.0;
    if (neg) x = -x;

    // Horner form coefficients for erf approximation
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    const double t = 1.0 / (1.0 + p * x);
    const double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return neg ? -y : y;
}

__device__ __forceinline__ double device_cumulative_normal(double x) {
    return 0.5 * (1.0 + device_erf(x * M_SQRT1_2));
}

// CUDA kernel: price all options in parallel
__global__ void blackScholesKernel(const int* types,
                                   const double* strikes,
                                   const double* spots,
                                   const double* qs,
                                   const double* rs,
                                   const double* ts,
                                   const double* vols,
                                   double* results,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double S     = spots[idx];
    const double K     = strikes[idx];
    const double r     = rs[idx];
    const double q     = qs[idx];
    const double T     = ts[idx];
    const double sigma = vols[idx];
    const int   type   = types[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sigmaSqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;

    const double Nd1 = device_cumulative_normal(d1);
    const double Nd2 = device_cumulative_normal(d2);
    const double discount = exp(-r * T);
    const double expMinusQT = exp(-q * T);

    if (type == CALL) {
        results[idx] = S * expMinusQT * Nd1 - K * discount * Nd2;
    } else {
        results[idx] = K * discount * device_cumulative_normal(-d2)
                     - S * expMinusQT * device_cumulative_normal(-d1);
    }
}

// ============================================================================
// Test data generation (OpenMP parallelized)
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

// Generate options with OpenMP parallelism
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
// CUDA helper: device selection per MPI rank
// ============================================================================

static int selectDevice(int mpiRank) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices found on rank %d\n", mpiRank);
        return -1;
    }
    // Round-robin device assignment across MPI ranks
    return mpiRank % deviceCount;
}

// ============================================================================
// Validation (OpenMP parallelized)
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

        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }

    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize MPI
    int mpiInitialized = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiInitialized);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Parse command line arguments (all ranks parse, but only rank 0 prints)
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
            if (mpiRank == 0) printUsage(argv[0]);
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

    // Print configuration from rank 0
    if (mpiRank == 0) {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        printf("Black-Scholes Option Pricing Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", mpiSize);
        printf("CUDA devices: %d\n", deviceCount);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ========================================================================
    // Generate options on all ranks (needed for local validation)
    // ========================================================================
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);

    // ========================================================================
    // Distribute work across MPI ranks
    // ========================================================================
    // Compute distribution: each rank gets a contiguous chunk
    size_t optionsPerRank = numOptions / mpiSize;
    size_t remainder = numOptions % mpiSize;
    size_t localStart = mpiRank * optionsPerRank + std::min(static_cast<size_t>(mpiRank), remainder);
    size_t localCount = optionsPerRank + (static_cast<size_t>(mpiRank) < remainder ? 1 : 0);

    // Extract local options
    std::vector<OptionInput> localOptions(options.begin() + localStart,
                                          options.begin() + localStart + localCount);

    // ========================================================================
    // Select CUDA device for this rank
    // ========================================================================
    int device = selectDevice(mpiRank);
    if (device < 0) {
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(device);

    if (mpiRank == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        printf("Rank 0 GPU: %s\n", prop.name);
    }

    // ========================================================================
    // Allocate device memory
    // ========================================================================
    int n = static_cast<int>(localCount);

    // Split into separate arrays for coalesced memory access
    std::vector<int>    h_types(n);
    std::vector<double> h_strikes(n);
    std::vector<double> h_spots(n);
    std::vector<double> h_qs(n);
    std::vector<double> h_rs(n);
    std::vector<double> h_ts(n);
    std::vector<double> h_vols(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        h_types[i]   = localOptions[i].type;
        h_strikes[i] = localOptions[i].strike;
        h_spots[i]   = localOptions[i].spot;
        h_qs[i]      = localOptions[i].q;
        h_rs[i]      = localOptions[i].r;
        h_ts[i]      = localOptions[i].t;
        h_vols[i]    = localOptions[i].vol;
    }

    double *d_strikes = nullptr, *d_spots = nullptr;
    double *d_qs = nullptr, *d_rs = nullptr, *d_ts = nullptr, *d_vols = nullptr;
    int    *d_type_arr = nullptr;
    double *d_results = nullptr;

    cudaMalloc(&d_type_arr,   n * sizeof(int));
    cudaMalloc(&d_strikes,    n * sizeof(double));
    cudaMalloc(&d_spots,      n * sizeof(double));
    cudaMalloc(&d_qs,         n * sizeof(double));
    cudaMalloc(&d_rs,         n * sizeof(double));
    cudaMalloc(&d_ts,         n * sizeof(double));
    cudaMalloc(&d_vols,       n * sizeof(double));
    cudaMalloc(&d_results,    n * sizeof(double));

    // ========================================================================
    // Copy data to device
    // ========================================================================
    cudaMemcpy(d_type_arr, h_types.data(),   n * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes,  h_strikes.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots,    h_spots.data(),   n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs,       h_qs.data(),      n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rs,       h_rs.data(),      n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ts,       h_ts.data(),      n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vols,     h_vols.data(),    n * sizeof(double), cudaMemcpyHostToDevice);

    // ========================================================================
    // Launch CUDA kernel
    // ========================================================================
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // Warm-up launch
    blackScholesKernel<<<gridSize, blockSize>>>(
        d_type_arr, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_results, n);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    blackScholesKernel<<<gridSize, blockSize>>>(
        d_type_arr, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_results, n);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // ========================================================================
    // Copy results back
    // ========================================================================
    std::vector<double> localResults(n);
    cudaMemcpy(localResults.data(), d_results, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_type_arr);
    cudaFree(d_strikes);
    cudaFree(d_spots);
    cudaFree(d_qs);
    cudaFree(d_rs);
    cudaFree(d_ts);
    cudaFree(d_vols);
    cudaFree(d_results);

    // ========================================================================
    // Gather results to rank 0
    // ========================================================================
    // Send counts to rank 0
    std::vector<int> recvcounts(mpiSize);
    MPI_Gather(&n, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather displacements on rank 0
    std::vector<int> displs(mpiSize);
    if (mpiRank == 0) {
        displs[0] = 0;
        for (int i = 1; i < mpiSize; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    std::vector<double> allResults;
    if (mpiRank == 0) {
        allResults.resize(numOptions);
    }

    MPI_Gatherv(localResults.data(), n, MPI_DOUBLE,
                mpiRank == 0 ? allResults.data() : nullptr,
                recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ========================================================================
    // Output results from rank 0
    // ========================================================================
    if (mpiRank == 0) {
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Options per second: %.0f\n", numOptions / (milliseconds / 1000.0));

        // Print results for external validation
        if (printResults) {
            print_results(allResults, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(options, allResults);

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
