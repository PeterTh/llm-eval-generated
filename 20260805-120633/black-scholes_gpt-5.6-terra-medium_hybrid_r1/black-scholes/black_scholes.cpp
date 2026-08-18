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

constexpr double kInvSqrt2 = 0.7071067811865475244;
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

static void cudaCheck(cudaError_t status, const char* call, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure at %s:%d (%s): %s\n", file, line, call,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
}

__host__ __device__ inline double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * kInvSqrt2));
}

__host__ __device__ inline double priceOption(const OptionInput& option) {
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
    if (option.type == CALL) {
        return S * exp(-option.q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    }
    return K * discount * cumulativeNormal(-d2) - S * exp(-option.q * T) * cumulativeNormal(-d1);
}

__global__ void blackScholesKernel(const OptionInput* options, double* results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = priceOption(options[i]);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() {
    return {{{CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
             {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
             {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
             {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
             {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}}};
}

// Generate directly into each rank's contiguous global range; no input scatter is needed.
static void generateOptions(std::vector<OptionInput>& options, size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(options.size()); ++local) {
        const size_t global = globalOffset + static_cast<size_t>(local);
        OptionInput value = testOptions[global % testOptions.size()];
        const double factor = 1.0 + 0.1 * (global / static_cast<double>(testOptions.size()));
        value.spot *= factor;
        value.strike *= factor;
        options[static_cast<size_t>(local)] = value;
    }
}

static bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool passed = true;
    const size_t checks = std::min<size_t>(10, options.size());
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const double error = std::fabs(results[i] - options[i].value);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i, results[i],
                    options[i].value, error / (std::fabs(options[i].value) + 1e-10));
        if (results[i] < 0.0 || results[i] > 1000.0 || std::isnan(results[i]) || std::isinf(results[i])) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n", i, results[i]);
            passed = false;
        }
    }
    return passed;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of options to price (default: 10000)\n"
                "  -v           Enable validation against known values\n"
                "  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide the required thread support.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    int argumentError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            argumentError = 1;
        }
    }
    if (argumentError || numOptions > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            if (argumentError) std::printf("Unknown or incomplete option.\n");
            else std::printf("Number of options exceeds the MPI count limit.\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t offset = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);
    std::vector<OptionInput> localOptions(localCount);
    std::vector<double> localResults(localCount);
    generateOptions(localOptions, offset);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n",
                    numOptions, validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\nPricing options...\n", ranks, omp_get_max_threads());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount != 0) {
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)));
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)));
        CUDA_CHECK(cudaMemcpy(deviceOptions, localOptions.data(), localCount * sizeof(*deviceOptions), cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        const int blocks = static_cast<int>((localCount + threads - 1) / threads);
        blackScholesKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localResults.data(), deviceResults, localCount * sizeof(*deviceResults), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceOptions));
        CUDA_CHECK(cudaFree(deviceResults));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\nOptions per second: %.0f\n", elapsedSeconds * 1000.0,
                    elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
    }

    std::vector<double> results;
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            results.resize(numOptions);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
                displacements[r] = static_cast<int>(base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder));
            }
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0 && printResults) print_results(results, "OptionPrices");
        if (rank == 0 && validate) {
            std::vector<OptionInput> options(numOptions);
            generateOptions(options, 0);
            if (!validateResults(options, results)) {
                MPI_Comm_free(&localComm);
                MPI_Finalize();
                return 1;
            }
            std::printf("Validation: PASSED\n");
        }
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return 0;
}
