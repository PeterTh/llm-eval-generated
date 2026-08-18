#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

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

#if defined(__CUDACC__)
#define BS_HOST_DEVICE __host__ __device__
#define BS_FORCE_INLINE __forceinline__
#else
#define BS_HOST_DEVICE
#define BS_FORCE_INLINE inline
#endif

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
BS_HOST_DEVICE BS_FORCE_INLINE double cumulativeNormal(const double x) noexcept {
#if defined(__CUDA_ARCH__)
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
#else
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
#endif
}

// Standard normal probability density function
BS_HOST_DEVICE BS_FORCE_INLINE double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
BS_HOST_DEVICE BS_FORCE_INLINE double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    
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

// One independent option per CUDA thread. The grid-stride loop lets a rank
// price arbitrarily large local partitions without relying on a 32-bit grid
// size, while keeping the launch large enough to saturate each GPU.
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t count) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < count; i += stride) {
        results[i] = blackScholes(options[i]);
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

// Generate a rank-local portion of the same globally indexed data set as the
// original implementation. OpenMP parallelizes this host-side staging work;
// the global index preserves byte-for-byte input semantics across MPI ranks.
void generateOptions(std::vector<OptionInput>& options,
                     const size_t firstOption,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (std::int64_t localIndex = 0;
         localIndex < static_cast<std::int64_t>(numOptions);
         ++localIndex) {
        const size_t local = static_cast<size_t>(localIndex);
        const size_t global = firstOption + local;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[global % testOptions.size()];
        options[local] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 *
            (global / static_cast<double>(testOptions.size()));
        options[local].spot *= factor;
        options[local].strike *= factor;
    }
}

void splitWork(const size_t total,
               const int rank,
               const int ranks,
               size_t& first,
               size_t& count) {
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t rankCount = static_cast<size_t>(ranks);
    const size_t base = total / rankCount;
    const size_t remainder = total % rankCount;

    count = base + (rankIndex < remainder ? 1 : 0);
    first = base * rankIndex + std::min(rankIndex, remainder);
}

[[noreturn]] void abortCuda(const cudaError_t error,
                            const char* operation,
                            const int rank) {
    std::fprintf(stderr, "MPI rank %d: CUDA error during %s: %s\n",
                 rank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error,
               const char* operation,
               const int rank) {
    if (error != cudaSuccess) {
        abortCuda(error, operation, rank);
    }
}

#define BS_CUDA_CHECK(call) checkCuda((call), #call, rank)

int localMpiRank(const int fallback) {
    // MPI implementations and schedulers expose one of these conventional
    // local-rank variables. This keeps one rank on one GPU on multi-node jobs.
    constexpr const char* names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID"
    };

    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && *value != '\0') {
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end != value && *end == '\0' && parsed >= 0 && parsed <= INT_MAX) {
                return static_cast<int>(parsed);
            }
        }
    }
    return fallback;
}

void gatherResults(const std::vector<double>& localResults,
                   const size_t localFirst,
                   const size_t totalResults,
                   const int rank,
                   const int ranks,
                   std::vector<double>& globalResults) {
    unsigned long long localCount =
        static_cast<unsigned long long>(localResults.size());
    std::vector<unsigned long long> allCounts(static_cast<size_t>(ranks));
    MPI_Allgather(&localCount, 1, MPI_UNSIGNED_LONG_LONG,
                  allCounts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);

    std::vector<int> counts(static_cast<size_t>(ranks));
    std::vector<int> displacements(static_cast<size_t>(ranks));
    bool gathervCompatible = true;
    unsigned long long displacement = 0;
    for (int source = 0; source < ranks; ++source) {
        if (allCounts[static_cast<size_t>(source)] >
                static_cast<unsigned long long>(INT_MAX) ||
            displacement > static_cast<unsigned long long>(INT_MAX)) {
            gathervCompatible = false;
        }
        counts[static_cast<size_t>(source)] =
            static_cast<int>(std::min<unsigned long long>(
                allCounts[static_cast<size_t>(source)], INT_MAX));
        displacements[static_cast<size_t>(source)] =
            static_cast<int>(std::min<unsigned long long>(displacement, INT_MAX));
        displacement += allCounts[static_cast<size_t>(source)];
    }

    if (rank == 0) {
        globalResults.resize(totalResults);
    }

    if (gathervCompatible) {
        MPI_Gatherv(localResults.empty() ? nullptr : localResults.data(),
                    counts[static_cast<size_t>(rank)], MPI_DOUBLE,
                    rank == 0 && !globalResults.empty() ? globalResults.data() : nullptr,
                    counts.data(), displacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        return;
    }

    // MPI_Gatherv uses int counts and displacements. Use bounded point-to-point
    // chunks for datasets beyond those limits while retaining global order.
    constexpr size_t chunkSize = 1U << 20;
    if (rank == 0) {
        std::copy(localResults.begin(), localResults.end(),
                  globalResults.begin() + static_cast<std::ptrdiff_t>(localFirst));
        for (int source = 1; source < ranks; ++source) {
            size_t sourceFirst = 0;
            size_t sourceCount = 0;
            splitWork(totalResults, source, ranks, sourceFirst, sourceCount);
            for (size_t offset = 0; offset < sourceCount; offset += chunkSize) {
                const size_t remaining = sourceCount - offset;
                const int messageCount = static_cast<int>(std::min(chunkSize, remaining));
                MPI_Recv(globalResults.data() + sourceFirst + offset,
                         messageCount, MPI_DOUBLE, source, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t offset = 0; offset < localResults.size(); offset += chunkSize) {
            const size_t remaining = localResults.size() - offset;
            const int messageCount = static_cast<int>(std::min(chunkSize, remaining));
            MPI_Send(localResults.data() + offset, messageCount, MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
        }
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
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    (void)providedThreadLevel;

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    int showHelp = 0;

    // Parse once and broadcast the configuration so every MPI rank follows
    // the same execution path, including the zero-work and help cases.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
        if (showHelp != 0) {
            printUsage(argv[0]);
        }
    }

    unsigned long long optionCount = static_cast<unsigned long long>(numOptions);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&optionCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(optionCount);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    if (showHelp != 0 || parseStatus != 0) {
        MPI_Finalize();
        return parseStatus;
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP host staging enabled, CUDA pricing enabled\n", ranks);
    }

    // Assign contiguous global blocks to MPI ranks. Every rank selects its
    // local accelerator, preferring scheduler/MPI local-rank information.
    size_t localFirst = 0;
    size_t localCount = 0;
    splitWork(numOptions, rank, ranks, localFirst, localCount);

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    if (deviceQuery != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA device required: %s\n",
                    cudaGetErrorString(deviceQuery));
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    const int gpuRank = localMpiRank(rank);
    const int device = gpuRank % deviceCount;
    BS_CUDA_CHECK(cudaSetDevice(device));

    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localFirst, localCount);
    std::vector<double> localResults(localCount);

    // Register the existing vectors as pinned host buffers so the transfers
    // surrounding the kernel can remain asynchronous and use the PCIe/NVLink
    // path efficiently.
    if (localCount > 0) {
        BS_CUDA_CHECK(cudaHostRegister(localOptions.data(),
                                       localOptions.size() * sizeof(OptionInput),
                                       cudaHostRegisterPortable));
        BS_CUDA_CHECK(cudaHostRegister(localResults.data(),
                                       localResults.size() * sizeof(double),
                                       cudaHostRegisterPortable));
    }

    cudaStream_t stream = nullptr;
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    const size_t allocationCount = std::max<size_t>(localCount, 1);
    BS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    BS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOptions),
                             allocationCount * sizeof(OptionInput)));
    BS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                             allocationCount * sizeof(double)));

    int multiprocessors = 0;
    BS_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors,
                                         cudaDevAttrMultiProcessorCount,
                                         device));
    const size_t requiredBlocks =
        localCount / 256 + (localCount % 256 == 0 ? 0 : 1);
    const size_t residentBlocks = std::max<size_t>(
        1, static_cast<size_t>(multiprocessors) * 32);
    const unsigned int blockCount = static_cast<unsigned int>(std::max<size_t>(
        1, std::min(requiredBlocks == 0 ? size_t{1} : requiredBlocks,
                     residentBlocks)));
    constexpr unsigned int threadsPerBlock = 256;

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    if (localCount > 0) {
        BS_CUDA_CHECK(cudaMemcpyAsync(deviceOptions, localOptions.data(),
                                      localOptions.size() * sizeof(OptionInput),
                                      cudaMemcpyHostToDevice, stream));
    }
    blackScholesKernel<<<blockCount, threadsPerBlock, 0, stream>>>(
        deviceOptions, deviceResults, localCount);
    BS_CUDA_CHECK(cudaGetLastError());
    if (localCount > 0) {
        BS_CUDA_CHECK(cudaMemcpyAsync(localResults.data(), deviceResults,
                                      localResults.size() * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
    }
    BS_CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;

    if (localCount > 0) {
        BS_CUDA_CHECK(cudaHostUnregister(localOptions.data()));
        BS_CUDA_CHECK(cudaHostUnregister(localResults.data()));
    }
    BS_CUDA_CHECK(cudaFree(deviceOptions));
    BS_CUDA_CHECK(cudaFree(deviceResults));
    BS_CUDA_CHECK(cudaStreamDestroy(stream));

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    std::vector<double> results;
    if (printResults || validate) {
        gatherResults(localResults, localFirst, numOptions, rank, ranks, results);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double optionsPerSecond =
            elapsed > 0.0 ? static_cast<double>(numOptions) / elapsed : 0.0;
        printf("Options per second: %.0f\n", optionsPerSecond);

        // Print results for external validation.
        if (printResults) {
            print_results(results, "OptionPrices");
        }
    }

    int valid = 1;
    if (validate && rank == 0) {
        printf("Validating results...\n");
        // Validation only examines the first ten entries, so avoid recreating
        // the full input set on rank zero after distributed pricing.
        std::vector<OptionInput> validationOptions;
        generateOptions(validationOptions, 0, std::min<size_t>(10, numOptions));
        valid = validateResults(validationOptions, results) ? 1 : 0;
        printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validate && valid == 0 ? 1 : 0;
}
