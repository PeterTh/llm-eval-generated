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

__host__ __device__ inline double cumulativeNormal(double x) noexcept {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot, K = option.strike, r = option.r;
    const double q = option.q, T = option.t, sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    const double rootT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double discountedSpot = S * exp(-q * T);
    const double discountedStrike = K * exp(-r * T);
    return option.type == CALL
        ? discountedSpot * cumulativeNormal(d1) - discountedStrike * cumulativeNormal(d2)
        : discountedStrike * cumulativeNormal(-d2) - discountedSpot * cumulativeNormal(-d1);
}

__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
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

static void cudaCheck(cudaError_t status, const char* operation, MPI_Comm comm) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(comm, EXIT_FAILURE);
}

void generateOptions(std::vector<OptionInput>& options, size_t globalBegin) {
    constexpr auto test = getTestOptions();
#pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(options.size()); ++j) {
        const size_t i = globalBegin + static_cast<size_t>(j);
        OptionInput option = test[i % test.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(test.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[static_cast<size_t>(j)] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int checks = static_cast<int>(std::min<size_t>(10, options.size()));
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        const double computed = results[i], expected = options[i].value;
        const double relative = std::fabs(computed - expected) / (std::fabs(expected) + 1e-10);
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relative);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) allPassed = false;
    }
    return allPassed;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of options to price (default: 10000)\n"
                "  -v           Enable validation against known values\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false, badArguments = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            badArguments |= (*end != '\0' || value > std::numeric_limits<size_t>::max());
            numOptions = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else badArguments = true;
    }
    if (help || badArguments) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badArguments ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    int localRank = 0;
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    MPI_Comm_rank(shared, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", MPI_COMM_WORLD);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", MPI_COMM_WORLD);
    MPI_Comm_free(&shared);

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t extra = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < extra);
    const size_t globalBegin = static_cast<size_t>(rank) * base +
                               std::min(static_cast<size_t>(rank), extra);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, globalBegin);

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount) {
        cudaCheck(cudaMalloc(&deviceOptions, localCount * sizeof(OptionInput)), "cudaMalloc options", MPI_COMM_WORLD);
        cudaCheck(cudaMalloc(&deviceResults, localCount * sizeof(double)), "cudaMalloc results", MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        cudaCheck(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(OptionInput),
                             cudaMemcpyHostToDevice), "copy options to device", MPI_COMM_WORLD);
        constexpr int blockSize = 256;
        const unsigned blocks = static_cast<unsigned>((localCount + blockSize - 1) / blockSize);
        blackScholesKernel<<<blocks, blockSize>>>(deviceOptions, deviceResults, localCount);
        cudaCheck(cudaGetLastError(), "kernel launch", MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy results to host", MPI_COMM_WORLD);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> results;
    std::vector<int> counts, displacements;
    if (validate || printResults) {
        if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) std::fprintf(stderr, "Result collection exceeds MPI int count limit\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (rank == 0) {
            results.resize(numOptions); counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < extra));
                displacements[r] = static_cast<int>(static_cast<size_t>(r) * base +
                                                     std::min(static_cast<size_t>(r), extra));
            }
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    cudaFree(deviceOptions); cudaFree(deviceResults);

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n"
                    "Validation: %s\nPricing options...\n", numOptions,
                    validate ? "enabled" : "disabled");
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n",
                    elapsed * 1000.0, elapsed > 0.0 ? numOptions / elapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions(numOptions);
            generateOptions(validationOptions, 0);
            const bool valid = validateResults(validationOptions, results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
