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

// CUDA kernel for Black-Scholes computation
__global__ void blackScholesKernel(const OptionInput* options, double* results, int numOptions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numOptions) {
        const OptionInput& option = options[idx];
        const double S = option.spot;
        const double K = option.strike;
        const double r = option.r;
        const double q = option.q;
        const double T = option.t;
        const double sigma = option.vol;
        
        double price = 0.0;
        
        if (T > 0.0 && sigma > 0.0) {
            const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
            const double d2 = d1 - sigma * sqrt(T);
            
            const double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
            const double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
            const double discount = exp(-r * T);
            
            if (option.type == CALL) {
                price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
            } else { // PUT
                const double Nminus_d1 = 0.5 * (1.0 + erf(-d1 * M_SQRT1_2));
                const double Nminus_d2 = 0.5 * (1.0 + erf(-d2 * M_SQRT1_2));
                price = K * discount * Nminus_d2 - S * exp(-q * T) * Nminus_d1;
            }
        }
        
        results[idx] = price;
    }
}

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (CPU version)
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

// Generate a larger set of options by scaling the test set (OpenMP parallelized)
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Set CUDA device based on MPI rank (one GPU per rank, round-robin if needed)
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int deviceId = mpiRank % deviceCount;
        cudaSetDevice(deviceId);
    }
    
    // Parse command line arguments (all ranks parse same args)
    size_t numOptions = 10000;
    bool validate = false;
    bool printResultsFlag = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
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
    
    if (mpiRank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", mpiSize);
        printf("CUDA devices: %d\n", deviceCount);
        int ompThreads = omp_get_max_threads();
        printf("OpenMP threads per rank: %d\n", ompThreads);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute local work distribution
    size_t localN = numOptions / mpiSize;
    size_t remainder = numOptions % mpiSize;
    size_t localStart = mpiRank * localN + std::min<size_t>(mpiRank, remainder);
    if (static_cast<size_t>(mpiRank) < remainder) {
        localN++;
    }
    
    // Generate options on rank 0, then scatter
    std::vector<OptionInput> allOptions;
    std::vector<OptionInput> localOptions(localN);
    
    if (mpiRank == 0) {
        generateOptions(allOptions, numOptions);
    }
    
    // Scatter option data using MPI_Scatterv
    std::vector<int> recvCounts(mpiSize);
    std::vector<int> displs(mpiSize);
    {
        size_t offset = 0;
        for (int r = 0; r < mpiSize; ++r) {
            size_t count = numOptions / mpiSize;
            if (static_cast<size_t>(r) < remainder) count++;
            recvCounts[r] = static_cast<int>(count * sizeof(OptionInput));
            displs[r] = static_cast<int>(offset * sizeof(OptionInput));
            offset += count;
        }
    }
    
    // Use MPI_Scatterv with byte-level distribution for the struct array
    std::vector<char> sendBuf;
    if (mpiRank == 0) {
        sendBuf.resize(numOptions * sizeof(OptionInput));
        memcpy(sendBuf.data(), allOptions.data(), numOptions * sizeof(OptionInput));
    }
    
    MPI_Scatterv(
        sendBuf.data(), recvCounts.data(), displs.data(), MPI_BYTE,
        localOptions.data(), static_cast<int>(localN * sizeof(OptionInput)), MPI_BYTE,
        0, MPI_COMM_WORLD
    );
    
    // Allocate local results
    std::vector<double> localResults(localN);
    
    // ===== CUDA computation =====
    // Timer for the compute portion only
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    if (deviceCount > 0 && localN > 0) {
        // Allocate device memory
        OptionInput* d_options = nullptr;
        double* d_results = nullptr;
        cudaMalloc(&d_options, localN * sizeof(OptionInput));
        cudaMalloc(&d_results, localN * sizeof(double));
        
        // Copy input to device
        cudaMemcpy(d_options, localOptions.data(), localN * sizeof(OptionInput), cudaMemcpyHostToDevice);
        
        // Launch kernel
        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(localN) + blockSize - 1) / blockSize;
        blackScholesKernel<<<numBlocks, blockSize>>>(d_options, d_results, static_cast<int>(localN));
        
        // Copy results back
        cudaMemcpy(localResults.data(), d_results, localN * sizeof(double), cudaMemcpyDeviceToHost);
        
        cudaFree(d_options);
        cudaFree(d_results);
    } else if (localN > 0) {
        // CPU fallback with OpenMP
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localN; ++i) {
            localResults[i] = blackScholes(localOptions[i]);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Compute max time across all ranks for consistent timing
    double localDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather results to rank 0
    std::vector<double> allResults;
    std::vector<char> recvResultsBuf;
    
    // Prepare counts for gathering results (doubles)
    std::vector<int> resultRecvCounts(mpiSize);
    std::vector<int> resultDispls(mpiSize);
    {
        size_t offset = 0;
        for (int r = 0; r < mpiSize; ++r) {
            size_t count = numOptions / mpiSize;
            if (static_cast<size_t>(r) < remainder) count++;
            resultRecvCounts[r] = static_cast<int>(count * sizeof(double));
            resultDispls[r] = static_cast<int>(offset * sizeof(double));
            offset += count;
        }
    }
    
    if (mpiRank == 0) {
        allResults.resize(numOptions);
        recvResultsBuf.resize(numOptions * sizeof(double));
    }
    
    MPI_Gatherv(
        localResults.data(), static_cast<int>(localN * sizeof(double)), MPI_BYTE,
        recvResultsBuf.data(), resultRecvCounts.data(), resultDispls.data(), MPI_BYTE,
        0, MPI_COMM_WORLD
    );
    
    if (mpiRank == 0) {
        memcpy(allResults.data(), recvResultsBuf.data(), numOptions * sizeof(double));
        
        printf("Computation time: %.3f ms\n", maxDuration / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxDuration / 1e6));
        
        // Print results for external validation
        if (printResultsFlag) {
            print_results(allResults, "OptionPrices");
        }
        
        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(allOptions, allResults);
            
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
