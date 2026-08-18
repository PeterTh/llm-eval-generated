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

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(comm, EXIT_FAILURE);
    }
}

__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * erfc(-x * 0.707106781186547524400844362105);
}

__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const OptionInput o = options[i];
        if (o.t <= 0.0 || o.vol <= 0.0) {
            results[i] = 0.0;
            continue;
        }
        const double rootT = sqrt(o.t);
        const double d1 = (log(o.spot / o.strike) +
                           (o.r - o.q + 0.5 * o.vol * o.vol) * o.t) /
                          (o.vol * rootT);
        const double d2 = d1 - o.vol * rootT;
        const double discountedSpot = o.spot * exp(-o.q * o.t);
        const double discountedStrike = o.strike * exp(-o.r * o.t);
        results[i] = o.type == CALL
            ? discountedSpot * cumulativeNormal(d1) - discountedStrike * cumulativeNormal(d2)
            : discountedStrike * cumulativeNormal(-d2) - discountedSpot * cumulativeNormal(-d1);
    }
}

inline constexpr std::array<OptionInput, 7> testOptions{{
    {CALL, 40., 42., .04, .08, .75, .35, 5.0975, 1.e-3},
    {CALL, 100., 90., .10, .10, .10, .15, .0205, 1.e-3},
    {CALL, 100., 100., .10, .10, .10, .15, 1.8734, 1.e-3},
    {CALL, 100., 110., .10, .10, .10, .15, 9.9413, 1.e-3},
    {PUT, 100., 90., .10, .10, .10, .15, 9.9210, 1.e-3},
    {PUT, 100., 100., .10, .10, .10, .15, 1.8734, 1.e-3},
    {PUT, 100., 110., .10, .10, .10, .15, .0408, 1.e-3}
}};

static void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    const long long n = static_cast<long long>(options.size());
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < n; ++local) {
        const size_t i = globalOffset + static_cast<size_t>(local);
        OptionInput option = testOptions[i % testOptions.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[local] = option;
    }
}

static bool validateResults(const std::vector<OptionInput>& options,
                            const std::vector<double>& results) {
    const int checks = static_cast<int>(std::min<size_t>(10, options.size()));
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid)
    for (int i = 0; i < checks; ++i) {
        const double v = results[i];
        invalid |= (!std::isfinite(v) || v < 0.0 || v > 1000.0);
    }
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        const double error = std::fabs(results[i] - options[i].value);
        const double relative = error / (std::fabs(options[i].value) + 1e-10);
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, results[i], options[i].value, relative);
    }
    return invalid == 0;
}

static void usage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of options (default: 10000)\n"
                "  -v        Validate values\n  -r        Print results\n  -h        Show help\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t total = 10000;
    bool validate = false, printResults = false;
    int parseError = 0, help = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || parsed > std::numeric_limits<size_t>::max()) parseError = 1;
            else total = static_cast<size_t>(parsed);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = 1;
        else parseError = 1;
    }
    if (help || parseError) {
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return parseError;
    }

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);
    cudaDeviceProp deviceProperties{};
    cudaCheck(cudaGetDeviceProperties(&deviceProperties, localRank % deviceCount),
              "cudaGetDeviceProperties", MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    const size_t base = total / static_cast<size_t>(ranks);
    const size_t extra = total % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < extra);
    const size_t offset = base * static_cast<size_t>(rank) +
                          std::min(static_cast<size_t>(rank), extra);
    if (localCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Per-rank option count exceeds MPI gather limit\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::vector<OptionInput> options(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(options, offset);
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount) {
        cudaCheck(cudaMalloc(&deviceOptions, localCount * sizeof(OptionInput)), "cudaMalloc(options)", MPI_COMM_WORLD);
        cudaCheck(cudaMalloc(&deviceResults, localCount * sizeof(double)), "cudaMalloc(results)", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceOptions, options.data(), localCount * sizeof(OptionInput),
                             cudaMemcpyHostToDevice), "copy options to GPU", MPI_COMM_WORLD);
    }
    cudaCheck(cudaDeviceSynchronize(), "initial CUDA synchronization", MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        constexpr int threads = 256;
        const size_t neededBlocks = (localCount + threads - 1) / threads;
        const unsigned blocks = static_cast<unsigned>(std::min<size_t>(
            neededBlocks, static_cast<size_t>(deviceProperties.multiProcessorCount) * 32));
        blackScholesKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        cudaCheck(cudaGetLastError(), "Black-Scholes kernel launch", MPI_COMM_WORLD);
        cudaCheck(cudaDeviceSynchronize(), "Black-Scholes kernel", MPI_COMM_WORLD);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        if (localCount)
            cudaCheck(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copy results from GPU", MPI_COMM_WORLD);
    }
    cudaCheck(cudaFree(deviceOptions), "cudaFree(options)", MPI_COMM_WORLD);
    cudaCheck(cudaFree(deviceResults), "cudaFree(results)", MPI_COMM_WORLD);

    std::vector<double> results;
    std::vector<int> counts, displacements;
    if (validate || printResults) {
        if (rank == 0) {
            results.resize(total);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int p = 0; p < ranks; ++p) {
                const size_t pc = base + (static_cast<size_t>(p) < extra);
                const size_t po = base * static_cast<size_t>(p) + std::min(static_cast<size_t>(p), extra);
                if (po > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::fprintf(stderr, "Global option count exceeds MPI gather limit\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                counts[p] = static_cast<int>(pc);
                displacements[p] = static_cast<int>(po);
            }
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr, counts.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int status = 0;
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n"
                    "MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n"
                    "Validation: %s\nPricing options...\n", total, ranks, omp_get_max_threads(),
                    deviceCount, validate ? "enabled" : "disabled");
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n",
                    maxElapsed * 1000.0, maxElapsed > 0.0 ? total / maxElapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions(total);
            generateOptions(validationOptions, 0);
            status = validateResults(validationOptions, results) ? 0 : 1;
            std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
