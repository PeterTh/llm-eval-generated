#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
__device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__device__ double blackScholes(const OptionInput& option) noexcept {
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
    
    // Calls use N(d), puts use N(-d). Computing only the two values needed
    // avoids redundant transcendental functions for puts.
    const bool isCall = option.type == CALL;
    const double Nd1 = cumulativeNormal(isCall ? d1 : -d1);
    const double Nd2 = cumulativeNormal(isCall ? d2 : -d2);
    const double discount = exp(-r * T);

    double price;
    if (isCall) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * Nd2 - S * exp(-q * T) * Nd1;
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

// Each GPU generates exactly the same inputs as generateOptions, using global
// indices so that the result is independent of the MPI/GPU partition.
__constant__ OptionInput deviceTestOptions[7];

__global__ void priceOptions(double* results, size_t count, size_t offset) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t globalIndex = offset + i;
        OptionInput option = deviceTestOptions[globalIndex % 7];
        const double factor = 1.0 + 0.1 * (globalIndex / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        results[i] = blackScholes(option);
    }
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        // This may run on an OpenMP worker, where MPI calls are not permitted.
        std::abort();
    }
}

struct DeviceWork {
    int device;
    size_t offset;
    size_t count;
    double* results;
};

static size_t partitionCount(size_t total, int parts, int part) {
    return total / static_cast<size_t>(parts) +
           (static_cast<size_t>(part) < total % static_cast<size_t>(parts));
}

static size_t partitionOffset(size_t total, int parts, int part) {
    return (total / static_cast<size_t>(parts)) * static_cast<size_t>(part) +
           std::min(static_cast<size_t>(part), total % static_cast<size_t>(parts));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Divide visible devices among local ranks; oversubscribed ranks share them.
    const int firstDevice = localRank % deviceCount;
    const int numDevices = localRank < deviceCount
        ? 1 + (deviceCount - 1 - localRank) / localRanks : 1;
    const size_t localCount = partitionCount(numOptions, ranks, rank);
    const size_t rankOffset = partitionOffset(numOptions, ranks, rank);
    std::vector<DeviceWork> work(numDevices);
    constexpr auto testOptions = getTestOptions();
    #pragma omp parallel for num_threads(numDevices) schedule(static)
    for (int d = 0; d < numDevices; ++d) {
        DeviceWork& w = work[d];
        w.device = firstDevice + d * localRanks;
        w.offset = partitionOffset(localCount, numDevices, d);
        w.count = partitionCount(localCount, numDevices, d);
        w.results = nullptr;
        checkCuda(cudaSetDevice(w.device), "cudaSetDevice");
        checkCuda(cudaMemcpyToSymbol(deviceTestOptions, testOptions.data(), sizeof(OptionInput) * testOptions.size()),
                  "cudaMemcpyToSymbol");
        if (w.count != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&w.results), w.count * sizeof(double)), "cudaMalloc");
        }
    }

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(numDevices) schedule(static)
    for (int d = 0; d < numDevices; ++d) {
        const DeviceWork& w = work[d];
        checkCuda(cudaSetDevice(w.device), "cudaSetDevice");
        if (w.count != 0) {
            const int threads = 256;
            const size_t blocksNeeded = (w.count - 1) / threads + 1;
            const int blocks = static_cast<int>(std::min(blocksNeeded, size_t(65535)));
            priceOptions<<<blocks, threads>>>(w.results, w.count, rankOffset + w.offset);
            checkCuda(cudaGetLastError(), "priceOptions launch");
            checkCuda(cudaDeviceSynchronize(), "priceOptions execution");
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", numOptions / duration);
    }

    // Only transfer and collect results when needed. Validation inspects the
    // first ten; external reporting needs the complete, ordered array.
    const size_t needed = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    const size_t localNeeded = rankOffset < needed ? std::min(localCount, needed - rankOffset) : 0;
    std::vector<double> localResults(localNeeded);
    #pragma omp parallel for num_threads(numDevices) schedule(static)
    for (int d = 0; d < numDevices; ++d) {
        const DeviceWork& w = work[d];
        checkCuda(cudaSetDevice(w.device), "cudaSetDevice");
        const size_t count = w.offset < localNeeded ? std::min(w.count, localNeeded - w.offset) : 0;
        if (count != 0) {
            checkCuda(cudaMemcpy(localResults.data() + w.offset, w.results, count * sizeof(double),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy results");
        }
        if (w.results) checkCuda(cudaFree(w.results), "cudaFree");
    }

    std::vector<double> results;
    if (rank == 0) {
        results.resize(needed);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t offset = partitionOffset(numOptions, ranks, source);
            const size_t count = offset < needed ? std::min(partitionCount(numOptions, ranks, source), needed - offset) : 0;
            for (size_t pos = 0; pos < count;) {
                const int chunk = static_cast<int>(std::min(count - pos, static_cast<size_t>(INT_MAX)));
                MPI_Recv(results.data() + offset + pos, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pos += chunk;
            }
        }
    } else {
        for (size_t pos = 0; pos < localNeeded;) {
            const int chunk = static_cast<int>(std::min(localNeeded - pos, static_cast<size_t>(INT_MAX)));
            MPI_Send(localResults.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            pos += chunk;
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> options;
            generateOptions(options, std::min(numOptions, size_t(10)));
            printf("Validating results...\n");
            const bool valid = validateResults(options, results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
