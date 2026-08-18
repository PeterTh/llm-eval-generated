#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType { CALL = 0, PUT = 1 };

// This structure is copied unchanged to the GPU, so the CPU and CUDA paths
// use exactly the same input representation.
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

#define HD __host__ __device__

HD inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

HD inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double discount = exp(-r * T);
    if (option.type == CALL)
        return S * exp(-q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    return K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
}

__global__ void priceOptionsKernel(const OptionInput* options, double* results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = blackScholes(options[i]);
}

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

// Generates only this MPI rank's contiguous portion.  The global index keeps
// the input sequence bit-for-bit equivalent to generating the full vector.
void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
    const long long count = static_cast<long long>(options.size());
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < count; ++local) {
        const size_t global = globalOffset + static_cast<size_t>(local);
        const OptionInput& base = testOptions[global % testOptions.size()];
        OptionInput option = base;
        const double factor = 1.0 + 0.1 * (global / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[static_cast<size_t>(local)] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min<size_t>(10, options.size()));
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
    printf("Options:\n  -n <num>     Number of options to price (default: 10000)\n"
           "  -v           Enable validation against known values\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    bool parseOK = true, showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = atoll(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) showHelp = true;
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); parseOK = false; }
    }
    if (showHelp || !parseOK) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseOK ? 0 : 1;
    }
    if (numOptions > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "-n must not exceed %d (MPI_Gatherv count limit)\n", INT_MAX);
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount", rank);
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice", rank);
    MPI_Comm_free(&localComm);

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t offset = static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions(localCount);
    generateOptions(localOptions, offset);
    std::vector<double> localResults(localCount);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n",
               numOptions, validate ? "enabled" : "disabled");
        printf("Pricing options with MPI + OpenMP + CUDA (%d ranks)...\n", ranks);
    }

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    // Establish the CUDA context and allocate persistent buffers before the
    // timed region, matching the original benchmark's preallocated vectors.
    cudaCheck(cudaFree(nullptr), "initializing CUDA context", rank);
    if (localCount != 0) {
        cudaCheck(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)), "cudaMalloc(options)", rank);
        cudaCheck(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)), "cudaMalloc(results)", rank);
        // Warm the module and math-library paths before measuring the batch.
        // This excludes one-time CUDA loading/JIT work from the benchmark just
        // as host-side option allocation is excluded in the original program.
        cudaCheck(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions),
                             cudaMemcpyHostToDevice), "warming option upload", rank);
        constexpr int threads = 256;
        const int blocks = static_cast<int>((localCount + threads - 1) / threads);
        priceOptionsKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        cudaCheck(cudaGetLastError(), "warming pricing kernel", rank);
        cudaCheck(cudaDeviceSynchronize(), "synchronizing warm-up", rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localCount != 0) {
        cudaCheck(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions),
                             cudaMemcpyHostToDevice), "copying options to device", rank);
        constexpr int threads = 256;
        const int blocks = static_cast<int>((localCount + threads - 1) / threads);
        priceOptionsKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        cudaCheck(cudaGetLastError(), "launching pricing kernel", rank);
        cudaCheck(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(*deviceResults),
                             cudaMemcpyDeviceToHost), "copying results from device", rank);
    }
    const auto end = std::chrono::steady_clock::now();
    cudaFree(deviceOptions);
    cudaFree(deviceResults);

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> results;
    std::vector<int> counts, displacements;
    if (rank == 0) {
        results.resize(numOptions);
        counts.resize(ranks);
        displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder ? 1 : 0));
            displacements[r] = static_cast<int>(static_cast<size_t>(r) * base +
                                                std::min(static_cast<size_t>(r), remainder));
        }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        printf("Options per second: %.0f\n", elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions(numOptions);
            generateOptions(validationOptions, 0);
            printf("Validating results...\n");
            if (validateResults(validationOptions, results)) printf("Validation: PASSED\n");
            else { printf("Validation: FAILED\n"); result = 1; }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
