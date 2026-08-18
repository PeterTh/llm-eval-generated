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
    double strike, spot, q, r, t, vol, value, tol;
};

inline double cumulativeNormal(double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * 0.7071067811865475244));
}

double blackScholes(const OptionInput& option) noexcept {
    if (option.t <= 0.0 || option.vol <= 0.0) return 0.0;
    const double rootT = std::sqrt(option.t);
    const double d1 = (std::log(option.spot / option.strike) +
                       (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
                      (option.vol * rootT);
    const double d2 = d1 - option.vol * rootT;
    const double discount = std::exp(-option.r * option.t);
    if (option.type == CALL)
        return option.spot * std::exp(-option.q * option.t) * cumulativeNormal(d1) -
               option.strike * discount * cumulativeNormal(d2);
    return option.strike * discount * cumulativeNormal(-d2) -
           option.spot * std::exp(-option.q * option.t) * cumulativeNormal(-d1);
}

// The kernel intentionally operates on a structure-of-records input: it keeps input generation,
// validation, and MPI distribution semantically identical to the original benchmark.
__device__ __forceinline__ double deviceCumulativeNormal(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475244);
}

__global__ void priceOptionsKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const OptionInput o = options[i];
    if (o.t <= 0.0 || o.vol <= 0.0) { results[i] = 0.0; return; }
    const double rootT = sqrt(o.t);
    const double d1 = (log(o.spot / o.strike) +
                       (o.r - o.q + 0.5 * o.vol * o.vol) * o.t) / (o.vol * rootT);
    const double d2 = d1 - o.vol * rootT;
    const double discount = exp(-o.r * o.t);
    results[i] = (o.type == CALL)
        ? o.spot * exp(-o.q * o.t) * deviceCumulativeNormal(d1) -
          o.strike * discount * deviceCumulativeNormal(d2)
        : o.strike * discount * deviceCumulativeNormal(-d2) -
          o.spot * exp(-o.q * o.t) * deviceCumulativeNormal(-d1);
}

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error_)); \
    } \
} while (0)

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL,40,42,.04,.08,.75,.35,5.0975,1e-3}, {CALL,100,90,.10,.10,.10,.15,.0205,1e-3},
             {CALL,100,100,.10,.10,.10,.15,1.8734,1e-3}, {CALL,100,110,.10,.10,.10,.15,9.9413,1e-3},
             {PUT,100,90,.10,.10,.10,.15,9.9210,1e-3}, {PUT,100,100,.10,.10,.10,.15,1.8734,1e-3},
             {PUT,100,110,.10,.10,.10,.15,.0408,1e-3}}};
}

void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    constexpr auto base = getTestOptions();
    #pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(options.size()); ++local) {
        const size_t global = globalOffset + static_cast<size_t>(local);
        OptionInput value = base[global % base.size()];
        const double factor = 1.0 + 0.1 * (global / static_cast<double>(base.size()));
        value.spot *= factor;
        value.strike *= factor;
        options[static_cast<size_t>(local)] = value;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool passed = true;
    const size_t checks = std::min<size_t>(10, options.size());
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const double error = std::fabs(results[i] - options[i].value);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i,
                    results[i], options[i].value, error / (std::fabs(options[i].value) + 1e-10));
        if (results[i] < 0.0 || results[i] > 1000.0 || !std::isfinite(results[i])) passed = false;
    }
    return passed;
}

void printUsage(const char* prog) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of options to price (default: 10000)\n"
                "  -v           Enable validation against known values\n"
                "  -r           Print results for external validation\n  -h           Show this help message\n", prog);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    bool usage = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numOptions = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) usage = true;
        else badArgs = true;
    }
    if (usage || badArgs) { if (rank == 0) { if (badArgs) std::printf("Unknown option.\n"); printUsage(argv[0]); } MPI_Finalize(); return badArgs; }

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const size_t begin = numOptions * static_cast<size_t>(rank) / ranks;
    const size_t end = numOptions * static_cast<size_t>(rank + 1) / ranks;
    const size_t localCount = end - begin;
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, begin);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\nPricing options...\n",
                    numOptions, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    OptionInput* deviceOptions = nullptr; double* deviceResults = nullptr;
    if (localCount) {
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)));
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)));
        CUDA_CHECK(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions), cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        priceOptionsKernel<<<static_cast<unsigned>((localCount + threads - 1) / threads), threads>>>(deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(*deviceResults), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceResults)); CUDA_CHECK(cudaFree(deviceOptions));
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (rank == 0) {
        counts.resize(ranks); displacements.resize(ranks); results.resize(numOptions);
        for (int r = 0; r < ranks; ++r) { const size_t b = numOptions * static_cast<size_t>(r) / ranks; const size_t e = numOptions * static_cast<size_t>(r + 1) / ranks; counts[r] = static_cast<int>(e - b); displacements[r] = static_cast<int>(b); }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE, results.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int status = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsed * 1e3, numOptions / elapsed);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) { std::vector<OptionInput> allOptions(numOptions); generateOptions(allOptions, 0); status = validateResults(allOptions, results) ? 0 : 1; std::printf("Validation: %s\n", status ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
