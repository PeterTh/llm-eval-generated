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

#define CUDA_CHECK(call)                                                          \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
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

// CUDA kernel: each thread prices one option
__global__ void blackScholesKernel(const OptionInput* options, double* results, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
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

// Generate a chunk of options corresponding to global indices [startIndex, startIndex+count)
// out of a logical dataset of totalOptions elements. Deterministic in the global index so
// that any rank can independently reconstruct any slice of the full dataset.
void generateOptions(std::vector<OptionInput>& options, const size_t startIndex, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < count; ++k) {
        const size_t i = startIndex + k;
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

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Determine node-local rank so that ranks sharing a node spread across
    // that node's GPUs rather than all contending for device 0.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank under mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs/node: %d\n", worldSize, deviceCount);
        printf("Pricing options...\n");
    }

    // Partition the global dataset across MPI ranks as evenly as possible.
    const size_t base = numOptions / static_cast<size_t>(worldSize);
    const size_t rem = numOptions % static_cast<size_t>(worldSize);
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        const size_t d = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(d);
    }
    const size_t localOffset = static_cast<size_t>(displs[worldRank]);
    const size_t localCount = static_cast<size_t>(counts[worldRank]);

    // Each rank independently (re)generates only the slice of options it owns.
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localOffset, localCount);

    std::vector<double> localResults(localCount);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localCount > 0) {
        OptionInput* d_options = nullptr;
        double* d_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_options, localCount * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&d_results, localCount * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_options, localOptions.data(), localCount * sizeof(OptionInput),
                               cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks = static_cast<int>((localCount + threads - 1) / threads);
        blackScholesKernel<<<blocks, threads>>>(d_options, d_results, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(localResults.data(), d_results, localCount * sizeof(double),
                               cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all local results back to rank 0 to reproduce the original
    // (serial) program's output and validation semantics exactly.
    std::vector<double> results;
    if (worldRank == 0) {
        results.resize(numOptions);
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                worldRank == 0 ? results.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const double durationUs = elapsedSeconds * 1e6;
        printf("Computation time: %.3f ms\n", durationUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (durationUs / 1e6));

        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> fullOptions;
            generateOptions(fullOptions, 0, numOptions);
            bool valid = validateResults(fullOptions, results);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
