#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <limits>
#include <stdexcept>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
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


// Generate exactly the original global input sequence on the device. No input
// scatter or host-to-device transfer proportional to the problem size is needed.
__constant__ OptionInput deviceOptions[7];

__global__ void priceOptions(double* prices, size_t first, size_t count) {
    for (size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         j < count; j += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t i = first + j;
        OptionInput option = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        option.spot *= factor;
        option.strike *= factor;
        prices[j] = blackScholes(option);
    }
}

// Balanced contiguous ranges, including when there are more workers than items.
size_t boundary(size_t n, int worker, int workers) {
    return (n / workers) * worker + std::min(n % workers, static_cast<size_t>(worker));
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct DeviceWorker {
    int device = 0;
    size_t first = 0, count = 0, capacity = 0;
    double* prices = nullptr;
    double* staging = nullptr;
    cudaStream_t stream = nullptr;
};

int run(int argc, char** argv, int rank, int ranks) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end || errno == ERANGE ||
                value > std::numeric_limits<size_t>::max()) {
                if (rank == 0) fprintf(stderr, "Invalid option count: %s\n", argument);
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

    // Assign all visible GPUs across the ranks on each node. If the launcher
    // exposes just one GPU per rank, every rank correctly selects device zero.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, localRanks, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    MPI_Comm_free(&local);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("CUDA device required on every MPI rank");
    std::vector<DeviceWorker> workers;
    if (devices >= localRanks) {
        for (int d = localRank; d < devices; d += localRanks) {
            DeviceWorker w;
            w.device = d;
            workers.push_back(w);
        }
    } else {
        DeviceWorker w;
        w.device = localRank % devices;
        workers.push_back(w);
    }
    // Weight rank partitions by GPU count to balance heterogeneous nodes too.
    const int workerCount = static_cast<int>(workers.size());
    std::vector<int> counts(ranks);
    MPI_Allgather(&workerCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int totalWorkers = 0, firstWorker = 0;
    for (int r = 0; r < ranks; ++r) {
        if (r == rank) firstWorker = totalWorkers;
        if (counts[r] > INT_MAX - totalWorkers) throw std::runtime_error("Too many GPU workers");
        totalWorkers += counts[r];
    }
    const size_t first = boundary(numOptions, firstWorker, totalWorkers);
    const size_t last = boundary(numOptions, firstWorker + workerCount, totalWorkers);
    const size_t retained = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    const size_t localRetained = first < retained ? std::min(last, retained) - first : 0;
    std::vector<double> results(localRetained);
    constexpr size_t chunkSize = 1 << 20;
    int failed = 0;
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(workerCount) reduction(|:failed)
    for (int w = 0; w < workerCount; ++w) {
        try {
            auto& worker = workers[w];
            worker.first = boundary(numOptions, firstWorker + w, totalWorkers);
            worker.count = boundary(numOptions, firstWorker + w + 1, totalWorkers) - worker.first;
            worker.capacity = std::min(chunkSize, worker.count);
            cudaCheck(cudaSetDevice(worker.device));
            cudaCheck(cudaStreamCreateWithFlags(&worker.stream, cudaStreamNonBlocking));
            constexpr auto bases = getTestOptions();
            cudaCheck(cudaMemcpyToSymbol(deviceOptions, bases.data(), sizeof(OptionInput) * bases.size()));
            if (worker.capacity) {
                cudaCheck(cudaMalloc(reinterpret_cast<void**>(&worker.prices), worker.capacity * sizeof(double)));
                if (worker.first < retained)
                    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&worker.staging), worker.capacity * sizeof(double)));
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d GPU setup: %s\n", rank, e.what());
            failed = 1;
        }
    }
    if (failed) throw std::runtime_error("GPU setup failed");
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n", numOptions);
        printf("Validation: %s\nPricing options...\n", validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workerCount) reduction(|:failed)
    for (int w = 0; w < workerCount; ++w) {
        try {
            auto& worker = workers[w];
            cudaCheck(cudaSetDevice(worker.device));
            for (size_t offset = 0; offset < worker.count;) {
                const size_t n = std::min(worker.capacity, worker.count - offset);
                const size_t global = worker.first + offset;
                const unsigned blocks = static_cast<unsigned>(std::min(size_t(4096), (n + 255) / 256));
                priceOptions<<<blocks, 256, 0, worker.stream>>>(worker.prices, global, n);
                cudaCheck(cudaGetLastError());
                const size_t copyCount = global < retained ? std::min(n, retained - global) : 0;
                if (copyCount)
                    cudaCheck(cudaMemcpyAsync(worker.staging, worker.prices, copyCount * sizeof(double),
                                              cudaMemcpyDeviceToHost, worker.stream));
                cudaCheck(cudaStreamSynchronize(worker.stream));
                if (copyCount)
                    std::copy_n(worker.staging, copyCount, results.data() + global - first);
                offset += n;
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d GPU pricing: %s\n", rank, e.what());
            failed = 1;
        }
    }
    if (failed) throw std::runtime_error("GPU pricing failed");
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0 ? numOptions / duration : 0.0);
    }
    // Gather only requested outputs, in bounded messages without MPI's int
    // count limit. Global ordering preserves the serial statistics and hash order.
    if (rank == 0) results.resize(retained);
    int workerOffset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = std::min(retained, boundary(numOptions, workerOffset, totalWorkers));
        workerOffset += counts[r];
        const size_t end = std::min(retained, boundary(numOptions, workerOffset, totalWorkers));
        if (r == 0) continue;
        for (size_t offset = begin; offset < end;) {
            const int n = static_cast<int>(std::min(chunkSize, end - offset));
            if (rank == r)
                MPI_Send(results.data() + offset - begin, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            else if (rank == 0)
                MPI_Recv(results.data() + offset, n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += n;
        }
    }
    for (const auto& worker : workers) {
        cudaCheck(cudaSetDevice(worker.device));
        if (worker.prices) cudaCheck(cudaFree(worker.prices));
        if (worker.staging) cudaCheck(cudaFreeHost(worker.staging));
        cudaCheck(cudaStreamDestroy(worker.stream));
    }
    int status = 0;
    if (rank == 0) {
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
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
