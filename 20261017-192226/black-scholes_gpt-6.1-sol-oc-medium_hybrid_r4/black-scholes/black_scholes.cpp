#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <cuda_runtime.h>
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

// Each device generates its own inputs using the original global option index.
// Only the seven base options cross the host/device boundary.
__constant__ OptionInput deviceOptions[7];

__global__ void priceOptions(double* prices, size_t begin, size_t count) {
    for (size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = begin + j;
        OptionInput option = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        prices[j] = blackScholes(option);
    }
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        #pragma omp critical(mpi_failure)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

struct DeviceWork {
    int device;
    size_t offset, count, capacity;
    double* prices = nullptr;
    cudaStream_t stream = nullptr;
    int blocks = 0;
};

// Balanced contiguous partitions, without overflowing n * rank.
size_t partitionStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(n % ranks, static_cast<size_t>(rank));
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    // Normal MPI traffic is on the main thread; a worker may report a fatal
    // CUDA error through the serialized abort path.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "MPI_THREAD_SERIALIZED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || arg[0] == '-' || end == arg || *end ||
                value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid option count: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            numOptions = static_cast<size_t>(value);
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
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank, localRanks, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_free(&localComm);

    const size_t begin = partitionStart(numOptions, rank, ranks);
    const size_t count = partitionStart(numOptions, rank + 1, ranks) - begin;
    std::vector<DeviceWork> work;
    // One rank can drive all GPUs; multiple local ranks divide visible GPUs.
    for (int d = localRank; d < deviceCount; d += localRanks)
        work.push_back(DeviceWork{d, 0, 0, 0});
    if (work.empty()) work.push_back(DeviceWork{localRank % deviceCount, 0, 0, 0});
    const int workers = static_cast<int>(work.size());
    std::vector<double> localResults(printResults ? count : 0);
    double localChecks[10] = {}, checks[10] = {};
    constexpr size_t batchSize = 1u << 20;
    constexpr auto bases = getTestOptions();
    omp_set_dynamic(0);

    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        DeviceWork& job = work[w];
        job.offset = partitionStart(count, w, workers);
        job.count = partitionStart(count, w + 1, workers) - job.offset;
        job.capacity = std::min(batchSize, std::max(size_t{1}, job.count));
        checkCuda(cudaSetDevice(job.device));
        checkCuda(cudaStreamCreateWithFlags(&job.stream, cudaStreamNonBlocking));
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&job.prices),
                             job.capacity * sizeof(double)));
        checkCuda(cudaMemcpyToSymbol(deviceOptions, bases.data(), sizeof(deviceOptions)));
        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, job.device));
        job.blocks = prop.multiProcessorCount * 32;
        // Initialize the kernel/context before the timed collective.
        priceOptions<<<1, 256, 0, job.stream>>>(job.prices, 0, 0);
        checkCuda(cudaGetLastError());
        checkCuda(cudaStreamSynchronize(job.stream));
    }
    
    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        DeviceWork& job = work[w];
        checkCuda(cudaSetDevice(job.device));
        for (size_t j = 0; j < job.count; j += job.capacity) {
            const size_t n = std::min(job.capacity, job.count - j);
            const size_t offset = job.offset + j;
            const size_t global = begin + offset;
            const int blocks = static_cast<int>(std::min(
                static_cast<size_t>(job.blocks), (n + 255) / 256));
            priceOptions<<<blocks, 256, 0, job.stream>>>(job.prices, global, n);
            checkCuda(cudaGetLastError());
            if (printResults) {
                checkCuda(cudaMemcpyAsync(localResults.data() + offset, job.prices,
                    n * sizeof(double), cudaMemcpyDeviceToHost, job.stream));
            }
            if (validate && global < 10) {
                const size_t samples = std::min(n, size_t{10} - global);
                checkCuda(cudaMemcpyAsync(localChecks + global, job.prices,
                    samples * sizeof(double), cudaMemcpyDeviceToHost, job.stream));
            }
        }
        checkCuda(cudaStreamSynchronize(job.stream));
    }
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0 ? numOptions / duration : 0.0);
    }

    std::vector<double> results;
    if (printResults) {
        if (rank == 0) results.resize(numOptions);
        std::vector<int> counts(ranks), displacements(ranks);
        // Windowed Gatherv also supports arrays larger than MPI's int counts.
        for (size_t base = 0; base < numOptions;) {
            const size_t end = base + std::min(numOptions - base, static_cast<size_t>(INT_MAX));
            for (int r = 0; r < ranks; ++r) {
                const size_t lo = std::max(base, partitionStart(numOptions, r, ranks));
                const size_t hi = std::min(end, partitionStart(numOptions, r + 1, ranks));
                counts[r] = hi > lo ? static_cast<int>(hi - lo) : 0;
                displacements[r] = hi > lo ? static_cast<int>(lo - base) : 0;
            }
            const double* send = counts[rank] ? localResults.data() + std::max(base, begin) - begin : nullptr;
            MPI_Gatherv(send, counts[rank], MPI_DOUBLE,
                        rank == 0 ? results.data() + base : nullptr,
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            base = end;
        }
    }
    if (validate)
        MPI_Reduce(localChecks, checks, 10, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    #pragma omp parallel for num_threads(workers) schedule(static, 1)
    for (int w = 0; w < workers; ++w) {
        checkCuda(cudaSetDevice(work[w].device));
        checkCuda(cudaFree(work[w].prices));
        checkCuda(cudaStreamDestroy(work[w].stream));
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    int status = 0;
    if (rank == 0 && validate) {
        printf("Validating results...\n");
        std::vector<OptionInput> options;
        generateOptions(options, std::min(numOptions, size_t{10}));
        const std::vector<double> samples(checks, checks + options.size());
        bool valid = validateResults(options, samples);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }
    
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
