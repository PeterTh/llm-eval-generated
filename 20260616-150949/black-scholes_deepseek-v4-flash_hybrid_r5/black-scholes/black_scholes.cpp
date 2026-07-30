#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        MPI_Abort(MPI_COMM_WORLD, 1);                                  \
    }                                                                   \
} while(0)

enum OptionType { CALL = 0, PUT = 1 };

struct OptionInput {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    double value;
    double tol;
};

// ---------- Host-side math ----------

inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

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

// ---------- Test data ----------

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

// Generate a contiguous range of options (rank-local portion)
void generateOptionsRange(std::vector<OptionInput>& options,
                          size_t globalStart, size_t count) {
    constexpr auto testOptions = getTestOptions();
    constexpr size_t numTestOpts = testOptions.size();
    options.resize(count);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        size_t gIdx = globalStart + i;
        const OptionInput& base = testOptions[gIdx % numTestOpts];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (static_cast<double>(gIdx)
                                           / static_cast<double>(numTestOpts));
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

        if (computed < 0.0 || computed > 1000.0 ||
            std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n",
                   i, computed);
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

// ======================= CUDA DEVICE CODE =======================

__device__ double dev_blackScholes(int type,
                                   double S, double K,
                                   double r, double q,
                                   double T, double sigma) noexcept {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
    double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
    double discount = exp(-r * T);

    double price;
    if (type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * (0.5 * (1.0 + erf(-d2 * M_SQRT1_2)))
              - S * exp(-q * T) * (0.5 * (1.0 + erf(-d1 * M_SQRT1_2)));
    }
    return price;
}

__global__ void priceOptionsKernel(
    const int*  __restrict__ types,
    const double* __restrict__ strikes,
    const double* __restrict__ spots,
    const double* __restrict__ divs,
    const double* __restrict__ rates,
    const double* __restrict__ times,
    const double* __restrict__ vols,
    double* __restrict__ results,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        results[i] = dev_blackScholes(types[i],
                                      spots[i], strikes[i],
                                      rates[i], divs[i],
                                      times[i], vols[i]);
    }
}

// ======================= MAIN =======================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // -------- Parse args (rank 0 only, then broadcast) --------
    size_t numOptions = 10000;
    int  validate    = 0;
    int  printResult = 0;

    int shouldExit = 0;
    int exitCode    = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = atoll(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResult = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode  = 1;
            }
        }
    }

    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode,   1, MPI_INT, 0, MPI_COMM_WORLD);
    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    MPI_Bcast(&numOptions,  1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,    1, MPI_INT,           0, MPI_COMM_WORLD);
    MPI_Bcast(&printResult, 1, MPI_INT,           0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI processes: %d\n", nRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // -------- Work distribution --------
    const size_t base  = numOptions / static_cast<size_t>(nRanks);
    const size_t rem   = numOptions % static_cast<size_t>(nRanks);

    std::vector<int> counts(static_cast<size_t>(nRanks));
    std::vector<int> disps(static_cast<size_t>(nRanks));

    for (int r = 0; r < nRanks; ++r) {
        counts[static_cast<size_t>(r)] =
            static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
        disps[static_cast<size_t>(r)] =
            (r == 0) ? 0
                     : (disps[static_cast<size_t>(r - 1)]
                        + counts[static_cast<size_t>(r - 1)]);
    }

    const size_t localCount = static_cast<size_t>(counts[static_cast<size_t>(rank)]);
    const size_t startIdx   = static_cast<size_t>(disps[static_cast<size_t>(rank)]);

    // -------- Generate local options --------
    std::vector<OptionInput> localOpts;
    if (numOptions > 0) {
        generateOptionsRange(localOpts, startIdx, localCount);
    }

    // -------- Extract flat host arrays (OpenMP-parallel) --------
    std::vector<int>    h_types;
    std::vector<double> h_strikes, h_spots, h_divs, h_rates, h_times, h_vols;
    std::vector<double> h_results;

    if (localCount > 0) {
        h_types.resize(localCount);
        h_strikes.resize(localCount);
        h_spots.resize(localCount);
        h_divs.resize(localCount);
        h_rates.resize(localCount);
        h_times.resize(localCount);
        h_vols.resize(localCount);
        h_results.resize(localCount);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localCount; ++i) {
            h_types[i]   = localOpts[i].type;
            h_strikes[i] = localOpts[i].strike;
            h_spots[i]   = localOpts[i].spot;
            h_divs[i]    = localOpts[i].q;
            h_rates[i]   = localOpts[i].r;
            h_times[i]   = localOpts[i].t;
            h_vols[i]    = localOpts[i].vol;
        }
    }

    // -------- CUDA setup (round-robin GPU assignment) --------
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);
    if (nDevices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % nDevices));
    }

    // -------- Timings --------
    double kernelMs  = 0.0;
    double xferMs    = 0.0;
    double totalCompMs = 0.0;
    double commMs    = 0.0;

    // -------- GPU pricing --------
    if (localCount > 0 && nDevices > 0) {
        auto t0 = std::chrono::high_resolution_clock::now();

        int *d_types   = nullptr;
        double *d_strikes = nullptr, *d_spots  = nullptr;
        double *d_divs = nullptr, *d_rates = nullptr;
        double *d_times = nullptr, *d_vols = nullptr;
        double *d_results = nullptr;

        cudaEvent_t xferStart, xferStop, kerStart, kerStop;
        cudaEventCreate(&xferStart); cudaEventCreate(&xferStop);
        cudaEventCreate(&kerStart);  cudaEventCreate(&kerStop);

        // ---- H2D ----
        cudaEventRecord(xferStart);

        CUDA_CHECK(cudaMalloc(&d_types,   localCount * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_strikes, localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_spots,   localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_divs,    localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rates,   localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_times,   localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vols,    localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_results, localCount * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_types,   h_types.data(),
                              localCount * sizeof(int),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strikes, h_strikes.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_spots,   h_spots.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_divs,    h_divs.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rates,   h_rates.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_times,   h_times.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vols,    h_vols.data(),
                              localCount * sizeof(double), cudaMemcpyHostToDevice));

        float xferMsFloat = 0.0f;
        cudaEventRecord(xferStop);
        cudaEventSynchronize(xferStop);
        cudaEventElapsedTime(&xferMsFloat, xferStart, xferStop);
        xferMs = static_cast<double>(xferMsFloat);

        // ---- Kernel ----
        cudaEventRecord(kerStart);

        const int blockSize = 256;
        const int gridSize  = static_cast<int>(
            (localCount + blockSize - 1) / blockSize);
        priceOptionsKernel<<<gridSize, blockSize>>>(
            d_types, d_strikes, d_spots, d_divs, d_rates,
            d_times, d_vols, d_results,
            static_cast<int>(localCount));

        float kernelMsFloat = 0.0f;
        cudaEventRecord(kerStop);
        cudaEventSynchronize(kerStop);
        cudaEventElapsedTime(&kernelMsFloat, kerStart, kerStop);
        kernelMs = static_cast<double>(kernelMsFloat);

        // ---- D2H ----
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results,
                              localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));

        auto t1 = std::chrono::high_resolution_clock::now();
        totalCompMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // cleanup
        CUDA_CHECK(cudaFree(d_types));
        CUDA_CHECK(cudaFree(d_strikes));
        CUDA_CHECK(cudaFree(d_spots));
        CUDA_CHECK(cudaFree(d_divs));
        CUDA_CHECK(cudaFree(d_rates));
        CUDA_CHECK(cudaFree(d_times));
        CUDA_CHECK(cudaFree(d_vols));
        CUDA_CHECK(cudaFree(d_results));
        cudaEventDestroy(xferStart); cudaEventDestroy(xferStop);
        cudaEventDestroy(kerStart);  cudaEventDestroy(kerStop);
    }

    // -------- MPI gather results --------
    std::vector<double> allResults;
    if (numOptions > 0) {
        if (rank == 0) allResults.resize(numOptions);

        auto c0 = std::chrono::high_resolution_clock::now();
        MPI_Gatherv(
            (localCount > 0) ? h_results.data() : nullptr,
            static_cast<int>(localCount),
            MPI_DOUBLE,
            (rank == 0) ? allResults.data() : nullptr,
            counts.data(), disps.data(),
            MPI_DOUBLE, 0, MPI_COMM_WORLD);
        auto c1 = std::chrono::high_resolution_clock::now();
        commMs = std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    // -------- Rank 0: output & validation --------
    if (rank == 0) {
        const double totalMs = totalCompMs + commMs;

        printf("\n--- Performance ---\n");
        printf("GPU kernel:         %8.3f ms\n", kernelMs);
        printf("Data transfer:      %8.3f ms\n", xferMs);
        printf("MPI communication:  %8.3f ms\n", commMs);
        printf("Total (compute):    %8.3f ms\n", totalCompMs);
        printf("Options/sec:        %8.0f\n",
               numOptions / (totalMs / 1000.0));

        if (printResult && !allResults.empty()) {
            print_results(allResults, "OptionPrices");
        }

        if (validate) {
            printf("\nValidating results...\n");
            // Re-generate first N options on rank 0 for validation
            std::vector<OptionInput> checkOpts;
            generateOptionsRange(
                checkOpts, 0,
                std::min(static_cast<size_t>(10), numOptions));
            bool valid = validateResults(checkOpts, allResults);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");

            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
