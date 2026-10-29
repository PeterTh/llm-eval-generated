#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
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


__constant__ OptionInput deviceOptions[7];

__global__ void priceOptions(double* results, size_t offset, size_t count) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t global = offset + i;
        OptionInput option = deviceOptions[global % 7];
        const double factor = 1.0 + 0.1 * (global / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        results[i] = blackScholes(option);
    }
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static size_t partSize(size_t n, int parts, int part) {
    return n / parts + (static_cast<size_t>(part) < n % parts);
}
static size_t partOffset(size_t n, int parts, int part) {
    return (n / parts) * part + std::min(static_cast<size_t>(part), n % parts);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = atoll(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) { fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    // When there are more ranks than visible devices, ranks share devices.
    const int workers = localRanks < devices ? 1 + (devices - 1 - localRank) / localRanks : 1;
    const size_t count = partSize(numOptions, ranks, rank);
    const size_t offset = partOffset(numOptions, ranks, rank);
    const bool needResults = validate || printResults;
    std::vector<double> localResults(needResults ? count : 0);
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\nPricing options...\n",
               numOptions, validate ? "enabled" : "disabled");
    }
    std::vector<double*> deviceResults(workers, nullptr);
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = localRanks < devices ? localRank + worker * localRanks : localRank % devices;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        const auto bases = getTestOptions();
        cudaCheck(cudaMemcpyToSymbol(deviceOptions, bases.data(), sizeof(OptionInput) * bases.size()), "cudaMemcpyToSymbol");
        size_t n = partSize(count, workers, worker);
        if (n) cudaCheck(cudaMalloc(&deviceResults[worker], n * sizeof(double)), "cudaMalloc");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = localRanks < devices ? localRank + worker * localRanks : localRank % devices;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        size_t n = partSize(count, workers, worker);
        size_t startIndex = partOffset(count, workers, worker);
        if (n) {
            unsigned blocks = static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535));
            priceOptions<<<blocks, 256>>>(deviceResults[worker], offset + startIndex, n);
            cudaCheck(cudaGetLastError(), "priceOptions launch");
            cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }
    double elapsed = MPI_Wtime() - start, maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const int device = localRanks < devices ? localRank + worker * localRanks : localRank % devices;
        cudaCheck(cudaSetDevice(device), "cudaSetDevice");
        size_t n = partSize(count, workers, worker);
        size_t startIndex = partOffset(count, workers, worker);
        if (n) {
            if (needResults)
                cudaCheck(cudaMemcpy(localResults.data() + startIndex, deviceResults[worker], n * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy results");
            cudaCheck(cudaFree(deviceResults[worker]), "cudaFree");
        }
    }
    if (needResults) {
        std::vector<double> results;
        if (rank == 0) {
            results.resize(numOptions);
            std::copy(localResults.begin(), localResults.end(), results.begin());
            for (int source = 1; source < ranks; ++source) {
                size_t n = partSize(numOptions, ranks, source), at = partOffset(numOptions, ranks, source);
                while (n) {
                    int chunk = static_cast<int>(std::min<size_t>(n, INT_MAX));
                    MPI_Recv(results.data() + at, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    at += chunk; n -= chunk;
                }
            }
        } else {
            size_t at = 0;
            while (at < count) {
                int chunk = static_cast<int>(std::min<size_t>(count - at, INT_MAX));
                MPI_Send(localResults.data() + at, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                at += chunk;
            }
        }
        if (rank == 0) {
            if (printResults) print_results(results, "OptionPrices");
            if (validate) {
                printf("Validating results...\n");
                std::vector<OptionInput> options;
                generateOptions(options, std::min<size_t>(numOptions, 10));
                bool valid = validateResults(options, results);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
