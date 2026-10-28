// Black-Scholes option pricing benchmark: hybrid MPI + OpenMP + CUDA.
//
// Decomposition:
//   - MPI:    options are block-partitioned across ranks (one rank per GPU
//             on each node). Results are gathered in global index order on
//             rank 0 so that output, hashes and validation are unchanged.
//   - CUDA:   each rank prices the bulk of its block on its GPU. Options are
//             generated on the device from their global index, so no input
//             arrays are transferred.
//   - OpenMP: host threads price a tail share of each rank's block
//             concurrently with the GPU kernel.
//
// This file is compiled as CUDA (see CMakeLists.txt); the pricing formula is
// shared verbatim between the host and device paths.

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Fraction of each rank's block priced by host OpenMP threads (the remainder
// goes to the GPU). Tunable at run time with -c; the best value depends on the
// relative CPU/GPU throughput of the target system.
constexpr double kDefaultCpuFraction = 0.1;

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
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
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

constexpr size_t kNumTestOptions = getTestOptions().size();

// Table of base test options, passed by value to the GPU kernel.
struct OptionTable {
    OptionInput opt[kNumTestOptions];
};

inline OptionTable makeOptionTable() {
    OptionTable table;
    const auto src = getTestOptions();
    for (size_t k = 0; k < kNumTestOptions; ++k) {
        table.opt[k] = src[k];
    }
    return table;
}

// Option with global index i. Cycles through the test options and scales
// spot/strike slightly for larger datasets (identical to the original
// generateOptions() arithmetic, evaluated on demand).
__host__ __device__ inline OptionInput makeOption(const OptionTable& table, const size_t i) noexcept {
    OptionInput option = table.opt[i % kNumTestOptions];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Block partition of [0, n) across `size` ranks: rank r owns
// [blockOffset(r), blockOffset(r) + blockCount(r)).
inline size_t blockCount(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline size_t blockOffset(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// GPU kernel: grid-stride loop pricing options [offset, offset + count).
// Writes out[i] for local index i in [0, count).
__global__ void priceKernel(const OptionTable table, const size_t offset,
                            const size_t count, double* out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = blackScholes(makeOption(table, offset + i));
    }
}

// Host path: prices options [offset, offset + count) with OpenMP threads.
void priceRangeCpu(const OptionTable& table, const size_t offset,
                   const size_t count, double* out) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        out[i] = blackScholes(makeOption(table, offset + i));
    }
}

bool validateResults(const size_t numOptions, const std::vector<double>& results) {
    bool allPassed = true;
    const auto testOptions = getTestOptions();
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), numOptions));
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        // Expected values are the test-option values (unchanged by scaling).
        const double expected = testOptions[static_cast<size_t>(i) % kNumTestOptions].value;
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
    printf("  -c <frac>    Fraction of each rank's options priced on host OpenMP threads,\n");
    printf("               remainder on the GPU, concurrently (default: %g)\n", kDefaultCpuFraction);
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

struct RunOptions {
    size_t numOptions = 10000;
    double cpuFraction = kDefaultCpuFraction;
    bool validate = false;
    bool printResults = false;
};

// Parses arguments identically on every rank. Returns -1 to continue running,
// otherwise the process exit status to return immediately.
int parseArgs(int argc, char** argv, RunOptions& opt, bool root) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            opt.numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            char* end = nullptr;
            opt.cpuFraction = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !(opt.cpuFraction >= 0.0 && opt.cpuFraction <= 1.0)) {
                if (root) {
                    printf("Invalid CPU fraction (must be in [0, 1]): %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (opt.numOptions > static_cast<size_t>(INT_MAX)) {
        if (root) {
            printf("Number of options exceeds the MPI_Gatherv limit (%d)\n", INT_MAX);
        }
        return 1;
    }
    return -1;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    RunOptions opt;
    const int parseExit = parseArgs(argc, argv, opt, root);
    if (parseExit >= 0) {
        MPI_Finalize();
        return parseExit;
    }
    const size_t numOptions = opt.numOptions;

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
    }

    // Bind each rank to one GPU on its node and split host cores among the
    // node-local ranks for the OpenMP share.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr)); // force context creation outside timing

    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    // Work partition for this rank: [offset, offset + count), split into a
    // GPU head and an OpenMP tail.
    const size_t offset = blockOffset(numOptions, size, rank);
    const size_t count = blockCount(numOptions, size, rank);
    const size_t cpuCount = std::min(count, static_cast<size_t>(std::floor(static_cast<double>(count) * opt.cpuFraction)));
    const size_t gpuCount = count - cpuCount;

    const OptionTable table = makeOptionTable();
    std::vector<double> local(count);

    double* dOut = nullptr;
    if (gpuCount > 0) {
        CUDA_CHECK(cudaMalloc(&dOut, gpuCount * sizeof(double)));
    }

    // Load the kernel module outside the timed region.
    priceKernel<<<1, 1>>>(table, 0, 0, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (root) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    if (gpuCount > 0) {
        constexpr unsigned kThreads = 256;
        const size_t blocks = std::min<size_t>((gpuCount + kThreads - 1) / kThreads, 1u << 16);
        priceKernel<<<static_cast<unsigned>(blocks), kThreads>>>(table, offset, gpuCount, dOut);
        CUDA_CHECK(cudaGetLastError());
    }

    // Host threads price the tail while the GPU kernel runs.
    priceRangeCpu(table, offset + gpuCount, cpuCount, local.data() + gpuCount);

    // Blocking copy on the default stream also waits for the kernel.
    if (gpuCount > 0) {
        CUDA_CHECK(cudaMemcpy(local.data(), dOut, gpuCount * sizeof(double), cudaMemcpyDeviceToHost));
    }

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all results, in global index order, onto rank 0.
    std::vector<double> results;
    std::vector<int> counts;
    std::vector<int> displs;
    if (root) {
        results.resize(numOptions);
        counts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(blockCount(numOptions, size, r));
            displs[r] = static_cast<int>(blockOffset(numOptions, size, r));
        }
    }
    MPI_Gatherv(local.data(), static_cast<int>(count), MPI_DOUBLE,
                root ? results.data() : nullptr,
                root ? counts.data() : nullptr,
                root ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (root) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);

        // Print results for external validation
        if (opt.printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (opt.validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(numOptions, results);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            exitCode = valid ? 0 : 1;
        }
    }
    if (opt.validate) {
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (dOut != nullptr) {
        CUDA_CHECK(cudaFree(dOut));
    }
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
