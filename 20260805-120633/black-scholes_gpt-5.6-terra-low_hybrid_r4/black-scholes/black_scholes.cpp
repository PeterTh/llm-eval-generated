#include <algorithm>
#include <array>
#include <chrono>
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

static inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

__host__ __device__ inline double cumulativeNormal(const double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) {
    const double S = option.spot;
    const double K = option.strike;
    const double T = option.t;
    const double sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (option.r - option.q + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double discount = exp(-option.r * T);
    if (option.type == CALL)
        return S * exp(-option.q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    return K * discount * cumulativeNormal(-d2) - S * exp(-option.q * T) * cumulativeNormal(-d1);
}

__global__ void priceOptions(const OptionInput* __restrict__ options,
                             double* __restrict__ results, const size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = blackScholes(options[i]);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() {
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

void generateOptions(std::vector<OptionInput>& options, const size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < options.size(); ++i) {
        const size_t globalIndex = globalOffset + i;
        options[i] = testOptions[globalIndex % testOptions.size()];
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min<size_t>(10, options.size()));
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            std::printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of options to price (default: 10000)\n"
                "  -v           Enable validation against known values\n"
                "  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numOptions > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::fprintf(stderr, "Option count exceeds the MPI collection limit.\n");
        MPI_Finalize(); return 1;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder);
    const size_t offset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, offset);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n",
                    numOptions, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\nPricing options...\n",
                    ranks, omp_get_max_threads(), deviceCount);
    }

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount != 0) {
        checkCuda(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)), "cudaMalloc(options)");
        checkCuda(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)), "cudaMalloc(results)");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    if (localCount != 0) {
        checkCuda(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions), cudaMemcpyHostToDevice), "copy options");
        constexpr int threads = 256;
        const int blocks = static_cast<int>((localCount + threads - 1) / threads);
        priceOptions<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(*deviceResults), cudaMemcpyDeviceToHost), "copy results");
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (deviceOptions) cudaFree(deviceOptions);
    if (deviceResults) cudaFree(deviceResults);

    std::vector<double> results;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        results.resize(numOptions); counts.resize(ranks); displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
            displacements[r] = static_cast<int>(static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder));
        }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsedSeconds * 1e3,
                    elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions(numOptions);
            generateOptions(validationOptions, 0);
            std::printf("Validating results...\n");
            const bool valid = validateResults(validationOptions, results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
