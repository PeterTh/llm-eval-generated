#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
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

constexpr double kInvSqrt2 = 0.7071067811865475244;

__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * kInvSqrt2));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
    if (option.t <= 0.0 || option.vol <= 0.0) return 0.0;
    const double sqrtT = sqrt(option.t);
    const double d1 = (log(option.spot / option.strike) +
                       (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
                      (option.vol * sqrtT);
    const double d2 = d1 - option.vol * sqrtT;
    const double discount = exp(-option.r * option.t);
    if (option.type == CALL) {
        return option.spot * exp(-option.q * option.t) * cumulativeNormal(d1) -
               option.strike * discount * cumulativeNormal(d2);
    }
    return option.strike * discount * cumulativeNormal(-d2) -
           option.spot * exp(-option.q * option.t) * cumulativeNormal(-d1);
}

__global__ void priceOptionsKernel(const OptionInput* options, double* results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = blackScholes(options[i]);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL,40.,42.,.04,.08,.75,.35,5.0975,1.e-3},
             {CALL,100.,90.,.10,.10,.10,.15,.0205,1.e-3},
             {CALL,100.,100.,.10,.10,.10,.15,1.8734,1.e-3},
             {CALL,100.,110.,.10,.10,.10,.15,9.9413,1.e-3},
             {PUT,100.,90.,.10,.10,.10,.15,9.9210,1.e-3},
             {PUT,100.,100.,.10,.10,.10,.15,1.8734,1.e-3},
             {PUT,100.,110.,.10,.10,.10,.15,.0408,1.e-3}}};
}

void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < options.size(); ++i) {
        const size_t globalIndex = globalOffset + i;
        OptionInput option = testOptions[globalIndex % testOptions.size()];
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[i] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min<size_t>(10, options.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double relError = fabs(computed - expected) / (fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of options to price (default: 10000)\n  -v           Enable validation against known values\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

void checkCuda(cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numOptions = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numOptions > static_cast<size_t>(INT_MAX)) {
        if (!rank) fprintf(stderr, "Number of options exceeds MPI count limit\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "device discovery", rank);
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    checkCuda(cudaSetDevice(localRank % devices), "device selection", rank);

    if (!rank) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n",
               numOptions, validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder);
    const size_t offset = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, offset);

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount) {
        checkCuda(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)), "option allocation", rank);
        checkCuda(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)), "result allocation", rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localCount) {
        checkCuda(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions), cudaMemcpyHostToDevice), "option upload", rank);
        constexpr int threads = 256;
        const size_t blocks = (localCount + threads - 1) / threads;
        priceOptionsKernel<<<static_cast<unsigned int>(blocks), threads>>>(deviceOptions, deviceResults, localCount);
        checkCuda(cudaGetLastError(), "kernel launch", rank);
        checkCuda(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(*deviceResults), cudaMemcpyDeviceToHost), "result download", rank);
    }
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (!rank) {
        counts.resize(ranks); displacements.resize(ranks); results.resize(numOptions);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
            displacements[r] = static_cast<int>(static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder));
        }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE, rank ? nullptr : results.data(),
                rank ? nullptr : counts.data(), rank ? nullptr : displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (deviceOptions) cudaFree(deviceOptions);
    if (deviceResults) cudaFree(deviceResults);

    int exitCode = 0;
    if (!rank) {
        printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsedSeconds * 1000.0, numOptions / elapsedSeconds);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> options(numOptions);
            generateOptions(options, 0);
            printf("Validating results...\n");
            if (validateResults(options, results)) printf("Validation: PASSED\n");
            else { printf("Validation: FAILED\n"); exitCode = 1; }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
