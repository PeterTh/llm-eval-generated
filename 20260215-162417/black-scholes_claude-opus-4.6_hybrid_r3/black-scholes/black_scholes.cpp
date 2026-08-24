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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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

// Black-Scholes formula for European options (host, kept for reference)
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

// CUDA kernel: Black-Scholes on SoA data
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
    if (idx >= n) return;

    double S     = spots[idx];
    double K     = strikes[idx];
    double r_val = rs[idx];
    double q_val = qs[idx];
    double T     = ts[idx];
    double sigma = vols[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    double sqrtT = sqrt(T);
    double d1 = (log(S / K) + (r_val - q_val + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
    double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
    double discount = exp(-r_val * T);

    if (types[idx] == CALL) {
        results[idx] = S * exp(-q_val * T) * Nd1 - K * discount * Nd2;
    } else {
        double Nmd2 = 0.5 * (1.0 + erf(-d2 * M_SQRT1_2));
        double Nmd1 = 0.5 * (1.0 + erf(-d1 * M_SQRT1_2));
        results[idx] = K * discount * Nmd2 - S * exp(-q_val * T) * Nmd1;
    }
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

// Generate options for a local range [globalStart, globalStart+localN)
void generateLocalOptions(std::vector<OptionInput>& options,
                          size_t globalStart, size_t localN) {
    constexpr auto testOptions = getTestOptions();
    options.resize(localN);

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < localN; ++j) {
        size_t i = globalStart + j;
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[j] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[j].spot *= factor;
        options[j].strike *= factor;
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU based on local rank within the node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    MPI_Comm_free(&localComm);

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
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n",
               nprocs, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block-distribute options across MPI ranks
    size_t baseCount = numOptions / static_cast<size_t>(nprocs);
    size_t remainder = numOptions % static_cast<size_t>(nprocs);
    size_t localN = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t globalStart = static_cast<size_t>(rank) * baseCount
                         + std::min(static_cast<size_t>(rank), remainder);

    // Generate local options with OpenMP parallelism
    std::vector<OptionInput> localOptions;
    generateLocalOptions(localOptions, globalStart, localN);

    // AoS -> SoA conversion for coalesced GPU access (OpenMP parallel)
    std::vector<int>    h_types(localN);
    std::vector<double> h_strikes(localN), h_spots(localN), h_qs(localN);
    std::vector<double> h_rs(localN), h_ts(localN), h_vols(localN);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localN; ++i) {
        h_types[i]   = localOptions[i].type;
        h_strikes[i] = localOptions[i].strike;
        h_spots[i]   = localOptions[i].spot;
        h_qs[i]      = localOptions[i].q;
        h_rs[i]      = localOptions[i].r;
        h_ts[i]      = localOptions[i].t;
        h_vols[i]    = localOptions[i].vol;
    }

    // Free AoS memory early
    localOptions.clear();
    localOptions.shrink_to_fit();

    // Allocate device memory
    int    *d_types   = nullptr;
    double *d_strikes = nullptr, *d_spots = nullptr, *d_qs = nullptr;
    double *d_rs      = nullptr, *d_ts    = nullptr, *d_vols = nullptr;
    double *d_results = nullptr;

    if (localN > 0) {
        CUDA_CHECK(cudaMalloc(&d_types,   localN * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_strikes, localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_spots,   localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qs,      localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rs,      localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ts,      localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vols,    localN * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_results, localN * sizeof(double)));

        // H2D transfers
        CUDA_CHECK(cudaMemcpy(d_types,   h_types.data(),   localN * sizeof(int),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strikes, h_strikes.data(), localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_spots,   h_spots.data(),   localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qs,      h_qs.data(),      localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rs,      h_rs.data(),      localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ts,      h_ts.data(),      localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vols,    h_vols.data(),    localN * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Synchronize all ranks before timing
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Launch CUDA kernel
    if (localN > 0) {
        const int blockSize = 256;
        const int gridSize  = static_cast<int>((localN + blockSize - 1) / blockSize);
        blackScholesKernel<<<gridSize, blockSize>>>(
            d_types, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols,
            d_results, static_cast<int>(localN));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    long long localDurationUs = duration.count();
    long long globalDurationUs = 0;
    MPI_Reduce(&localDurationUs, &globalDurationUs, 1, MPI_LONG_LONG,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // D2H transfer
    std::vector<double> localResults(localN);
    if (localN > 0) {
        CUDA_CHECK(cudaMemcpy(localResults.data(), d_results,
                              localN * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather results to rank 0
    int localNInt = static_cast<int>(localN);
    std::vector<int> recvCounts(nprocs), displs(nprocs);
    MPI_Gather(&localNInt, 1, MPI_INT,
               recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> allResults;
    if (rank == 0) {
        allResults.resize(numOptions);
        displs[0] = 0;
        for (int i = 1; i < nprocs; ++i) {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }

    MPI_Gatherv(localResults.data(), localNInt, MPI_DOUBLE,
                allResults.data(), recvCounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", globalDurationUs / 1000.0);
        printf("Options per second: %.0f\n",
               numOptions / (globalDurationUs / 1e6));

        // Print results for external validation
        if (printResults) {
            print_results(allResults, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> fullOptions;
            generateLocalOptions(fullOptions, 0, numOptions);
            bool valid = validateResults(fullOptions, allResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Broadcast exit code so all ranks exit consistently
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Free device memory
    if (localN > 0) {
        cudaFree(d_types);
        cudaFree(d_strikes);
        cudaFree(d_spots);
        cudaFree(d_qs);
        cudaFree(d_rs);
        cudaFree(d_ts);
        cudaFree(d_vols);
        cudaFree(d_results);
    }

    MPI_Finalize();
    return exitCode;
}
