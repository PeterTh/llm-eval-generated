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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
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

// Standard normal cumulative distribution function (host)
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function (host)
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (host reference)
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    return price;
}

// --- CUDA device functions ---

__device__ inline double cudaCumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ double cudaBlackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cudaCumulativeNormal(d1);
    const double Nd2 = cudaCumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cudaCumulativeNormal(-d2) - S * exp(-q * T) * cudaCumulativeNormal(-d1);
    }
    return price;
}

__global__ void priceOptionsKernel(const OptionInput* options, double* results, const size_t N) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        results[i] = cudaBlackScholes(options[i]);
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

// Generate a range of options (each MPI rank generates its own portion independently)
void generateOptions(std::vector<OptionInput>& options, const size_t startIdx, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    constexpr size_t numBaseOpts = testOptions.size();
    options.resize(count);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const size_t globalIdx = startIdx + i;
        const size_t baseIdx = globalIdx % numBaseOpts;
        const double factor = 1.0 + 0.1 * (globalIdx / static_cast<double>(numBaseOpts));
        options[i] = {
            testOptions[baseIdx].type,
            testOptions[baseIdx].strike * factor,
            testOptions[baseIdx].spot * factor,
            testOptions[baseIdx].q,
            testOptions[baseIdx].r,
            testOptions[baseIdx].t,
            testOptions[baseIdx].vol,
            testOptions[baseIdx].value,
            testOptions[baseIdx].tol
        };
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
    printf("\n");
    printf("Hybrid MPI+OpenMP+CUDA parallel implementation.\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // CUDA setup: select GPU in round-robin fashion across MPI ranks
    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    if (numGpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % numGpus));
    } else {
        fprintf(stderr, "Rank %d: No CUDA-capable device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse command line arguments on rank 0
    size_t numOptions = 10000;
    bool doValidate = false;
    bool doPrintResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = atoll(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                doValidate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                doPrintResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast numOptions to all ranks
    MPI_Bcast(&numOptions, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // Distribute work across MPI ranks
    const size_t baseCount = numOptions / nRanks;
    const size_t remainder = numOptions % nRanks;
    const size_t localStart = rank * baseCount + std::min(static_cast<size_t>(rank), remainder);
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Generate local options (parallelized with OpenMP)
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localStart, localCount);

    // Allocate local results
    std::vector<double> localResults(localCount);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", nRanks);
        printf("CUDA devices available: %d\n", numGpus);
        printf("Pricing options (MPI+OpenMP+CUDA hybrid)...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // --- CUDA computation ---
    if (localCount > 0) {
        OptionInput* d_options = nullptr;
        double* d_results = nullptr;

        CUDA_CHECK(cudaMalloc(&d_options, localCount * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&d_results, localCount * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_options, localOptions.data(),
                              localCount * sizeof(OptionInput),
                              cudaMemcpyHostToDevice));

        const int blockSize = 256;
        const int gridSize = static_cast<int>((localCount + blockSize - 1) / blockSize);

        priceOptionsKernel<<<gridSize, blockSize>>>(d_options, d_results, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(localResults.data(), d_results,
                              localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
    }
    // --- end CUDA computation ---

    auto end = std::chrono::high_resolution_clock::now();

    // Gather results at rank 0
    std::vector<int> recvCounts(nRanks);
    std::vector<int> displs(nRanks);
    std::vector<double> allResults;

    const int localCountInt = static_cast<int>(localCount);
    MPI_Gather(&localCountInt, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        allResults.resize(numOptions);
        displs[0] = 0;
        for (int i = 1; i < nRanks; ++i) {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }

    MPI_Gatherv(localResults.data(), localCountInt, MPI_DOUBLE,
                allResults.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0: output, validate, print results
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

        if (doPrintResults) {
            print_results(allResults, "OptionPrices");
        }

        if (doValidate) {
            printf("Validating results...\n");
            // Reconstruct full options on rank 0 for validation
            std::vector<OptionInput> allOptions;
            generateOptions(allOptions, 0, numOptions);
            bool valid = validateResults(allOptions, allResults);
            if (valid) {
                printf("Validation: PASSED\n");
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
