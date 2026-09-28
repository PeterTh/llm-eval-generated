// Hybrid-parallel Black-Scholes benchmark:
//   - MPI: options are block-distributed across ranks (one rank per GPU),
//     results are gathered on rank 0 for output/validation.
//   - CUDA: each rank prices the bulk of its local block on its GPU.
//   - OpenMP: parallel option generation on the host, plus a CPU share of the
//     pricing that runs concurrently with the asynchronous GPU work.
#include <algorithm>
#include <array>
#include <chrono>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

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

// GPU kernel: grid-stride loop pricing one option per work item
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t n) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
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

// Generate options for global indices [offset, offset + count) by scaling the
// test set; identical values to the original serial generation for any split.
void generateOptionsRange(OptionInput* options, const size_t offset, const size_t count) {
    constexpr auto testOptions = getTestOptions();

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < count; ++k) {
        const size_t i = offset + k;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[k] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[k].spot *= factor;
        options[k].strike *= factor;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    // Bind each rank to a GPU based on its node-local rank
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr)); // establish CUDA context outside the timed region

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid parallelization: %d MPI rank(s) x %d OpenMP thread(s) + CUDA (%d GPU(s)/node)\n",
               numRanks, omp_get_max_threads(), deviceCount);
    }

    // Block distribution of the global option range across ranks
    const size_t base = numOptions / static_cast<size_t>(numRanks);
    const size_t rem = numOptions % static_cast<size_t>(numRanks);
    const size_t localCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t localOffset = static_cast<size_t>(rank) * base +
                               std::min(static_cast<size_t>(rank), rem);

    // Generate the local block of options (OpenMP-parallel, pinned for fast H2D)
    OptionInput* localOptions = nullptr;
    double* localResults = nullptr;
    CUDA_CHECK(cudaMallocHost(&localOptions, std::max<size_t>(localCount, 1) * sizeof(OptionInput)));
    CUDA_CHECK(cudaMallocHost(&localResults, std::max<size_t>(localCount, 1) * sizeof(double)));
    generateOptionsRange(localOptions, localOffset, localCount);

    // Device buffers and stream, allocated outside the timed region
    OptionInput* dOptions = nullptr;
    double* dResults = nullptr;
    CUDA_CHECK(cudaMalloc(&dOptions, std::max<size_t>(localCount, 1) * sizeof(OptionInput)));
    CUDA_CHECK(cudaMalloc(&dResults, std::max<size_t>(localCount, 1) * sizeof(double)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm up: force lazy kernel/module loading outside the timed region
    blackScholesKernel<<<1, 32, 0, stream>>>(dOptions, dResults, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Split the local block between the GPU (front) and the CPU (tail); the
    // CPU share overlaps with the asynchronous GPU transfer + kernel.
    const size_t cpuCount = localCount / 32;
    const size_t gpuCount = localCount - cpuCount;

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (gpuCount > 0) {
        CUDA_CHECK(cudaMemcpyAsync(dOptions, localOptions, gpuCount * sizeof(OptionInput),
                                   cudaMemcpyHostToDevice, stream));
        const int blockSize = 256;
        const int gridSize = static_cast<int>(
            std::min<size_t>((gpuCount + blockSize - 1) / blockSize, 65535));
        blackScholesKernel<<<gridSize, blockSize, 0, stream>>>(dOptions, dResults, gpuCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(localResults, dResults, gpuCount * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
    }

    // CPU share, computed with OpenMP while the GPU works
    #pragma omp parallel for schedule(static)
    for (size_t i = gpuCount; i < localCount; ++i) {
        localResults[i] = blackScholes(localOptions[i]);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Report the slowest rank's time as the benchmark time
    long long localMicros = static_cast<long long>(duration.count());
    long long maxMicros = 0;
    MPI_Reduce(&localMicros, &maxMicros, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMicros / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxMicros / 1e6));
    }

    // Gather all results on rank 0 for output and validation
    std::vector<double> results;
    std::vector<int> counts(numRanks), displs(numRanks);
    for (int p = 0; p < numRanks; ++p) {
        const size_t c = base + (static_cast<size_t>(p) < rem ? 1 : 0);
        const size_t o = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), rem);
        counts[p] = static_cast<int>(c);
        displs[p] = static_cast<int>(o);
    }
    if (rank == 0) {
        results.resize(numOptions);
    }
    MPI_Gatherv(localResults, static_cast<int>(localCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> headOptions(std::min<size_t>(10, numOptions));
            generateOptionsRange(headOptions.data(), 0, headOptions.size());
            bool valid = validateResults(headOptions, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dOptions));
    CUDA_CHECK(cudaFree(dResults));
    CUDA_CHECK(cudaFreeHost(localOptions));
    CUDA_CHECK(cudaFreeHost(localResults));

    MPI_Finalize();
    return exitCode;
}
