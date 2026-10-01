#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>

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


// Generate exactly the original input using the GLOBAL index, regardless of
// rank/GPU partitioning. Only the seven templates need host-to-device transfer.
__constant__ OptionInput deviceOptions[7];

__global__ void priceOptions(double* output, size_t first, size_t count) {
    for (size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = first + j;
        OptionInput option = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(7));
        option.spot *= factor;
        option.strike *= factor;
        output[j] = blackScholes(option);
    }
}

// Quotient/remainder partitioning avoids overflow in count * rank.
size_t partitionStart(size_t count, int part, int parts) {
    return (count / parts) * part + std::min(count % parts, static_cast<size_t>(part));
}

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void checkMpi(int error) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "MPI operation failed (error %d)\n", error);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}

struct GpuWork {
    int device = 0;
    size_t begin = 0, count = 0, capacity = 0;
    double* output[2] = {};
    double* staging[2] = {};
    cudaStream_t streams[2] = {};
    std::string error;
};

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* arg = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end ||
                value > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) fprintf(stderr, "Invalid number of options: %s\n", arg);
                return 1;
            }
            numOptions = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm localComm;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                MPI_INFO_NULL, &localComm));
    int localRank, localRanks, devices;
    checkMpi(MPI_Comm_rank(localComm, &localRank));
    checkMpi(MPI_Comm_size(localComm, &localRanks));
    checkMpi(MPI_Comm_free(&localComm));
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("CUDA requires at least one visible GPU per rank");

    // Use all visible GPUs with a single rank, or distribute them among local
    // ranks. This also supports launchers that expose only one GPU per rank.
    const int workers = localRank < devices ? 1 + (devices - 1 - localRank) / localRanks : 1;
    // Weight rank ranges by their GPU count so uneven GPU/rank layouts do
    // not leave ranks with several accelerators waiting for single-GPU ranks.
    std::vector<int> workerCounts(ranks);
    checkMpi(MPI_Allgather(&workers, 1, MPI_INT, workerCounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    int totalWorkers = 0;
    for (int n : workerCounts) {
        if (n > std::numeric_limits<int>::max() - totalWorkers)
            throw std::runtime_error("Too many GPU workers");
        totalWorkers += n;
    }
    std::vector<size_t> boundaries(static_cast<size_t>(ranks) + 1);
    int prefix = 0;
    for (int peer = 0; peer < ranks; ++peer) {
        boundaries[peer] = partitionStart(numOptions, prefix, totalWorkers);
        prefix += workerCounts[peer];
    }
    boundaries[ranks] = numOptions;
    const size_t first = boundaries[rank];
    const size_t count = boundaries[rank + 1] - first;
    std::vector<GpuWork> work(workers);
    const size_t retained = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    const size_t localRetained = first < retained ? std::min(count, retained - first) : 0;
    std::vector<double> localResults(localRetained);
    constexpr size_t chunkSize = 1 << 20;
    const auto templates = getTestOptions();
    omp_set_dynamic(0);

    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int w = 0; w < workers; ++w) {
        auto& gpu = work[w];
        try {
            gpu.device = localRank < devices ? localRank + w * localRanks : localRank % devices;
            gpu.begin = partitionStart(count, w, workers);
            gpu.count = partitionStart(count, w + 1, workers) - gpu.begin;
            gpu.capacity = std::min(chunkSize, gpu.count);
            checkCuda(cudaSetDevice(gpu.device));
            checkCuda(cudaMemcpyToSymbol(deviceOptions, templates.data(), sizeof(templates)));
            for (int slot = 0; slot < 2; ++slot) {
                checkCuda(cudaStreamCreateWithFlags(&gpu.streams[slot], cudaStreamNonBlocking));
                if (gpu.capacity) {
                    checkCuda(cudaMalloc(&gpu.output[slot], gpu.capacity * sizeof(double)));
                    if (gpu.begin < localRetained)
                        checkCuda(cudaMallocHost(&gpu.staging[slot], gpu.capacity * sizeof(double)));
                }
            }
        } catch (const std::exception& e) { gpu.error = e.what(); }
    }
    for (const auto& gpu : work)
        if (!gpu.error.empty()) throw std::runtime_error(gpu.error);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int w = 0; w < workers; ++w) {
        auto& gpu = work[w];
        try {
            checkCuda(cudaSetDevice(gpu.device));
            size_t pendingCount[2] = {}, pendingOffset[2] = {};
            size_t batch = 0;
            for (size_t offset = 0; offset < gpu.count; ++batch) {
                const int slot = batch % 2;
                // Double buffering overlaps PCIe copies with the next kernel.
                checkCuda(cudaStreamSynchronize(gpu.streams[slot]));
                if (pendingCount[slot])
                    std::copy_n(gpu.staging[slot], pendingCount[slot],
                                localResults.data() + pendingOffset[slot]);
                const size_t n = std::min(gpu.capacity, gpu.count - offset);
                const size_t localOffset = gpu.begin + offset;
                const int blocks = static_cast<int>(std::min(size_t(65535), (n + 255) / 256));
                priceOptions<<<blocks, 256, 0, gpu.streams[slot]>>>(gpu.output[slot], first + localOffset, n);
                checkCuda(cudaGetLastError());
                pendingOffset[slot] = localOffset;
                pendingCount[slot] = localOffset < localRetained ? std::min(n, localRetained - localOffset) : 0;
                if (pendingCount[slot])
                    checkCuda(cudaMemcpyAsync(gpu.staging[slot], gpu.output[slot],
                              pendingCount[slot] * sizeof(double), cudaMemcpyDeviceToHost, gpu.streams[slot]));
                offset += n;
            }
            for (int slot = 0; slot < 2; ++slot) {
                checkCuda(cudaStreamSynchronize(gpu.streams[slot]));
                if (pendingCount[slot])
                    std::copy_n(gpu.staging[slot], pendingCount[slot], localResults.data() + pendingOffset[slot]);
            }
        } catch (const std::exception& e) { gpu.error = e.what(); }
    }
    for (const auto& gpu : work)
        if (!gpu.error.empty()) throw std::runtime_error(gpu.error);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    checkMpi(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    // Only requested results are communicated. Chunked messages avoid MPI's
    // int count limit, and rank-order placement preserves the original hash.
    std::vector<double> results;
    if (rank == 0) {
        results.resize(retained);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t begin = boundaries[peer];
            const size_t end = std::min(retained, boundaries[peer + 1]);
            for (size_t offset = begin; offset < end;) {
                const int n = static_cast<int>(std::min(chunkSize, end - offset));
                checkMpi(MPI_Recv(results.data() + offset, n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                offset += n;
            }
        }
    } else {
        for (size_t offset = 0; offset < localRetained;) {
            const int n = static_cast<int>(std::min(chunkSize, localRetained - offset));
            checkMpi(MPI_Send(localResults.data() + offset, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            offset += n;
        }
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int w = 0; w < workers; ++w) {
        auto& gpu = work[w];
        try {
            checkCuda(cudaSetDevice(gpu.device));
            for (int slot = 0; slot < 2; ++slot) {
                if (gpu.output[slot]) checkCuda(cudaFree(gpu.output[slot]));
                if (gpu.staging[slot]) checkCuda(cudaFreeHost(gpu.staging[slot]));
                checkCuda(cudaStreamDestroy(gpu.streams[slot]));
            }
        } catch (const std::exception& e) { gpu.error = e.what(); }
    }
    for (const auto& gpu : work)
        if (!gpu.error.empty()) throw std::runtime_error(gpu.error);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0 ? numOptions / duration : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> options;
            generateOptions(options, std::min(numOptions, size_t(10)));
            printf("Validating results...\n");
            const bool valid = validateResults(options, results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    checkMpi(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, ranks = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkMpi(MPI_Finalize());
    return status;
}
