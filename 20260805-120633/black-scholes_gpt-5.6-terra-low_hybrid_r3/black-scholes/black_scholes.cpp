#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

enum OptionType { CALL = 0, PUT = 1 };

struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

#define HD __host__ __device__

HD inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

HD inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot, K = option.strike, r = option.r;
    const double q = option.q, T = option.t, sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    const double rootT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double discount = exp(-r * T);
    if (option.type == CALL)
        return S * exp(-q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    return K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
}

__global__ void priceOptionsKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = blackScholes(options[i]);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL,40,42,.04,.08,.75,.35,5.0975,1e-3},
             {CALL,100,90,.10,.10,.10,.15,.0205,1e-3},
             {CALL,100,100,.10,.10,.10,.15,1.8734,1e-3},
             {CALL,100,110,.10,.10,.10,.15,9.9413,1e-3},
             {PUT,100,90,.10,.10,.10,.15,9.9210,1e-3},
             {PUT,100,100,.10,.10,.10,.15,1.8734,1e-3},
             {PUT,100,110,.10,.10,.10,.15,.0408,1e-3}}};
}

void generateOptions(std::vector<OptionInput>& options, size_t first, size_t count) {
    constexpr auto base = getTestOptions();
    options.resize(count);
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(count); ++local) {
        const size_t i = first + static_cast<size_t>(local);
        OptionInput value = base[i % base.size()];
        const double factor = 1.0 + .1 * (i / static_cast<double>(base.size()));
        value.spot *= factor;
        value.strike *= factor;
        options[static_cast<size_t>(local)] = value;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool passed = true;
    const size_t checks = std::min<size_t>(10, options.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const double computed = results[i], expected = options[i].value;
        const double error = fabs(computed - expected);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i,
               computed, expected, error / (fabs(expected) + 1e-10));
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            passed = false;
        }
    }
    return passed;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n  -n <num>     Number of options to price (default: 10000)\n"
           "  -v           Enable validation against known values\n"
           "  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA failure at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(e)); } } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numOptions = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) fprintf(stderr, "-n must not exceed INT_MAX for MPI_Gatherv\n");
        MPI_Finalize(); return 1;
    }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, first, localCount);
    std::vector<double> localResults(localCount);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (!rank) {
        results.resize(numOptions); counts.resize(ranks); displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
            displacements[r] = static_cast<int>(static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder));
        }
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n"
               "MPI ranks: %d, OpenMP threads/rank: %d\nPricing options...\n", numOptions,
               validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    OptionInput* deviceOptions = nullptr; double* deviceResults = nullptr;
    if (localCount) {
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(OptionInput), cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        priceOptionsKernel<<<static_cast<unsigned>((localCount + threads - 1) / threads), threads>>>(deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceOptions)); CUDA_CHECK(cudaFree(deviceResults));
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                rank ? nullptr : results.data(), rank ? nullptr : counts.data(), rank ? nullptr : displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (!rank) {
        printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsed * 1000.0,
               elapsed > 0.0 ? numOptions / elapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, 0, numOptions);
            printf("Validating results...\n");
            if (validateResults(validationOptions, results)) printf("Validation: PASSED\n");
            else { printf("Validation: FAILED\n"); exitCode = 1; }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
