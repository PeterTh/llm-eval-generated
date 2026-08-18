#include <algorithm>
#include <array>
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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * erfc(-x * 0.7071067811865475244008443621048490);
}

__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const OptionInput option = options[i];
        if (option.t <= 0.0 || option.vol <= 0.0) {
            results[i] = 0.0;
            continue;
        }
        const double rootT = sqrt(option.t);
        const double sigmaRootT = option.vol * rootT;
        const double d1 = (log(option.spot / option.strike) +
                          (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
                          sigmaRootT;
        const double d2 = d1 - sigmaRootT;
        const double discountedSpot = option.spot * exp(-option.q * option.t);
        const double discountedStrike = option.strike * exp(-option.r * option.t);
        results[i] = option.type == CALL
            ? discountedSpot * cumulativeNormal(d1) - discountedStrike * cumulativeNormal(d2)
            : discountedStrike * cumulativeNormal(-d2) - discountedSpot * cumulativeNormal(-d1);
    }
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

void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    constexpr auto tests = getTestOptions();
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < options.size(); ++j) {
        const size_t i = globalOffset + j;
        options[j] = tests[i % tests.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(tests.size()));
        options[j].spot *= factor;
        options[j].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    const int checks = static_cast<int>(std::min<size_t>(10, options.size()));
    int invalid = 0;
    #pragma omp parallel for reduction(+:invalid)
    for (int i = 0; i < checks; ++i) {
        const double x = results[i];
        invalid += x < 0.0 || x > 1000.0 || !std::isfinite(x);
    }
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        const double error = std::fabs(results[i] - options[i].value);
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, results[i], options[i].value,
                    error / (std::fabs(options[i].value) + 1e-10));
    }
    return invalid == 0;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of options (default: 10000)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false, badArgs = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long n = std::strtoull(argv[++i], &end, 10);
            badArgs = !end || *end != '\0' || n > std::numeric_limits<size_t>::max();
            numOptions = static_cast<size_t>(n);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else badArgs = true;
    }
    if (help || badArgs) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArgs ? 1 : 0;
    }
    if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "-n exceeds MPI gather limit\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA devices found\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    const size_t begin = numOptions * static_cast<size_t>(rank) / ranks;
    const size_t end = numOptions * static_cast<size_t>(rank + 1) / ranks;
    const size_t localCount = end - begin;
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, begin);

    OptionInput* dOptions = nullptr;
    double* dResults = nullptr;
    if (localCount) {
        CUDA_CHECK(cudaMalloc(&dOptions, localCount * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&dResults, localCount * sizeof(double)));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        CUDA_CHECK(cudaMemcpy(dOptions, localOptions.data(), localCount * sizeof(OptionInput), cudaMemcpyHostToDevice));
        int smCount = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, localRank % devices));
        constexpr int threads = 256;
        const int blocks = std::min<size_t>((localCount + threads - 1) / threads,
                                            static_cast<size_t>(smCount * 16));
        blackScholesKernel<<<blocks, threads>>>(dOptions, dResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localResults.data(), dResults, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (rank == 0) {
        counts.resize(ranks); displacements.resize(ranks); results.resize(numOptions);
        for (int r = 0; r < ranks; ++r) {
            const size_t b = numOptions * static_cast<size_t>(r) / ranks;
            const size_t e = numOptions * static_cast<size_t>(r + 1) / ranks;
            counts[r] = static_cast<int>(e - b); displacements[r] = static_cast<int>(b);
        }
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dOptions)); CUDA_CHECK(cudaFree(dResults));

    int status = 0;
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n"
                    "Validation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\nPricing options...\n",
                    numOptions, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n",
                    maxElapsed * 1000.0, maxElapsed > 0.0 ? numOptions / maxElapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> checkOptions(std::min<size_t>(10, numOptions));
            generateOptions(checkOptions, 0);
            const bool valid = validateResults(checkOptions, results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
