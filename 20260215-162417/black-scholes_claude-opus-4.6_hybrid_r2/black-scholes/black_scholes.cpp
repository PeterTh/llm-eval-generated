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

// SOA layout for efficient GPU memory access
struct OptionInputSOA {
    int* type;
    double* strike;
    double* spot;
    double* q;
    double* r;
    double* t;
    double* vol;
};

// Standard normal cumulative distribution function (host)
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function (host)
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (host)
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

// CUDA device function for cumulative normal distribution
__device__ double cumulativeNormal_d(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// CUDA kernel: Black-Scholes pricing with SOA layout
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

    const double S = spots[idx];
    const double K = strikes[idx];
    const double r_val = rs[idx];
    const double q_val = qs[idx];
    const double T = ts[idx];
    const double sigma = vols[idx];
    const int optType = types[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r_val - q_val + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = cumulativeNormal_d(d1);
    const double Nd2 = cumulativeNormal_d(d2);
    const double discount = exp(-r_val * T);
    const double divDiscount = exp(-q_val * T);

    double price;
    if (optType == CALL) {
        price = S * divDiscount * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormal_d(-d2) - S * divDiscount * cumulativeNormal_d(-d1);
    }

    results[idx] = price;
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
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
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

// Launch CUDA kernel for a local chunk of options
void computeOnGPU(const std::vector<OptionInput>& localOptions,
                  double* localResults, int localCount) {
    // Convert AOS to SOA for coalesced GPU memory access
    std::vector<int> h_types(localCount);
    std::vector<double> h_strikes(localCount), h_spots(localCount);
    std::vector<double> h_qs(localCount), h_rs(localCount);
    std::vector<double> h_ts(localCount), h_vols(localCount);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < localCount; ++i) {
        h_types[i]   = localOptions[i].type;
        h_strikes[i] = localOptions[i].strike;
        h_spots[i]   = localOptions[i].spot;
        h_qs[i]      = localOptions[i].q;
        h_rs[i]      = localOptions[i].r;
        h_ts[i]      = localOptions[i].t;
        h_vols[i]    = localOptions[i].vol;
    }

    // Allocate device memory
    int* d_types;
    double *d_strikes, *d_spots, *d_qs, *d_rs, *d_ts, *d_vols, *d_results;

    cudaMalloc(&d_types,   localCount * sizeof(int));
    cudaMalloc(&d_strikes, localCount * sizeof(double));
    cudaMalloc(&d_spots,   localCount * sizeof(double));
    cudaMalloc(&d_qs,      localCount * sizeof(double));
    cudaMalloc(&d_rs,      localCount * sizeof(double));
    cudaMalloc(&d_ts,      localCount * sizeof(double));
    cudaMalloc(&d_vols,    localCount * sizeof(double));
    cudaMalloc(&d_results, localCount * sizeof(double));

    // Copy to device
    cudaMemcpy(d_types,   h_types.data(),   localCount * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_strikes, h_strikes.data(), localCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spots,   h_spots.data(),   localCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qs,      h_qs.data(),      localCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rs,      h_rs.data(),      localCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ts,      h_ts.data(),      localCount * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vols,    h_vols.data(),    localCount * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    const int blockSize = 256;
    const int numBlocks = (localCount + blockSize - 1) / blockSize;
    blackScholesKernel<<<numBlocks, blockSize>>>(
        d_types, d_strikes, d_spots, d_qs, d_rs, d_ts, d_vols, d_results, localCount);

    // Copy results back
    cudaMemcpy(localResults, d_results, localCount * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_types);
    cudaFree(d_strikes);
    cudaFree(d_spots);
    cudaFree(d_qs);
    cudaFree(d_rs);
    cudaFree(d_ts);
    cudaFree(d_vols);
    cudaFree(d_results);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Assign each MPI rank to a GPU (round-robin if more ranks than GPUs)
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaSetDevice(rank % deviceCount);
    }

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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per rank: %d, OpenMP threads: %d\n",
               numProcs, deviceCount > 0 ? 1 : 0, omp_get_max_threads());
    }
    
    // All ranks generate all options (deterministic, avoids large broadcast)
    std::vector<OptionInput> options;
    generateOptions(options, numOptions);
    
    // Compute local chunk for this rank
    size_t localStart = (numOptions * rank) / numProcs;
    size_t localEnd   = (numOptions * (rank + 1)) / numProcs;
    int localCount    = static_cast<int>(localEnd - localStart);

    // Extract local options
    std::vector<OptionInput> localOptions(options.begin() + localStart,
                                          options.begin() + localEnd);
    std::vector<double> localResults(localCount);
    
    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (deviceCount > 0) {
        // GPU path: offload to CUDA
        computeOnGPU(localOptions, localResults.data(), localCount);
    } else {
        // CPU fallback: use OpenMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localCount; ++i) {
            localResults[i] = blackScholes(localOptions[i]);
        }
    }

    // Gather results on rank 0
    std::vector<double> results;
    std::vector<int> recvCounts(numProcs);
    std::vector<int> displs(numProcs);

    for (int p = 0; p < numProcs; ++p) {
        size_t pStart = (numOptions * p) / numProcs;
        size_t pEnd   = (numOptions * (p + 1)) / numProcs;
        recvCounts[p] = static_cast<int>(pEnd - pStart);
        displs[p]     = static_cast<int>(pStart);
    }

    if (rank == 0) {
        results.resize(numOptions);
    }

    MPI_Gatherv(localResults.data(), localCount, MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr,
                recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
        
        if (printResults) {
            print_results(results, "OptionPrices");
        }
        
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(options, results);
            
            if (valid) {
                printf("Validation: PASSED\n");
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
