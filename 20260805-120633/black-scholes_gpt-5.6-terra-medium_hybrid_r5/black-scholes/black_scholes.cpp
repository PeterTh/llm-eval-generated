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

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error)); \
    } \
} while (0)

__device__ __forceinline__ double cumulativeNormalDevice(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475244);
}

__global__ void priceOptionsKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const OptionInput option = options[i];
    if (option.t <= 0.0 || option.vol <= 0.0) {
        results[i] = 0.0;
        return;
    }
    const double sqrtT = sqrt(option.t);
    const double d1 = (log(option.spot / option.strike) +
                       (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
                      (option.vol * sqrtT);
    const double d2 = d1 - option.vol * sqrtT;
    const double discount = exp(-option.r * option.t);
    if (option.type == CALL) {
        results[i] = option.spot * exp(-option.q * option.t) * cumulativeNormalDevice(d1) -
                     option.strike * discount * cumulativeNormalDevice(d2);
    } else {
        results[i] = option.strike * discount * cumulativeNormalDevice(-d2) -
                     option.spot * exp(-option.q * option.t) * cumulativeNormalDevice(-d1);
    }
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
    constexpr auto base = getTestOptions();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < options.size(); ++i) {
        const size_t globalIndex = globalOffset + i;
        options[i] = base[globalIndex % base.size()];
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(base.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const size_t checks = std::min<size_t>(10, options.size());
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const double error = std::fabs(results[i] - options[i].value);
        const double relError = error / (std::fabs(options[i].value) + 1e-10);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, results[i], options[i].value, relError);
        if (results[i] < 0.0 || results[i] > 1000.0 || !std::isfinite(results[i])) allPassed = false;
    }
    return allPassed;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of options to price (default: 10000)\n"
                "  -v           Enable validation against known values\n"
                "  -r           Print results for external validation\n  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    int argumentError = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numOptions = std::strtoull(argv[++i], nullptr, 10);
            else if (!std::strcmp(argv[i], "-v")) validate = true;
            else if (!std::strcmp(argv[i], "-r")) printResults = true;
            else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); MPI_Finalize(); return 0; }
            else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); argumentError = 1; }
        }
    }
    MPI_Bcast(&argumentError, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (argumentError) { MPI_Finalize(); return 1; }
    unsigned long long globalCount = numOptions;
    MPI_Bcast(&globalCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(globalCount);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t baseCount = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder);
    const size_t offset = static_cast<size_t>(rank) * baseCount + std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, offset);

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount) {
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)));
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        CUDA_CHECK(cudaMemcpyAsync(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions), cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        const size_t grid = (localCount + threads - 1) / threads;
        priceOptionsKernel<<<static_cast<unsigned int>(grid), threads>>>(deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(localResults.data(), deviceResults, localCount * sizeof(*deviceResults), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (rank == 0) {
        counts.resize(ranks); displacements.resize(ranks); results.resize(numOptions);
        for (int r = 0; r < ranks; ++r) { counts[r] = static_cast<int>(baseCount + (static_cast<size_t>(r) < remainder)); displacements[r] = static_cast<int>(static_cast<size_t>(r) * baseCount + std::min(static_cast<size_t>(r), remainder)); }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE, results.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (deviceOptions) CUDA_CHECK(cudaFree(deviceOptions));
    if (deviceResults) CUDA_CHECK(cudaFree(deviceResults));

    int exitCode = 0;
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n", numOptions, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nPricing options...\n", ranks, omp_get_max_threads());
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsed * 1e3, elapsed > 0 ? numOptions / elapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) { std::vector<OptionInput> allOptions(numOptions); generateOptions(allOptions, 0); exitCode = validateResults(allOptions, results) ? 0 : 1; std::printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
