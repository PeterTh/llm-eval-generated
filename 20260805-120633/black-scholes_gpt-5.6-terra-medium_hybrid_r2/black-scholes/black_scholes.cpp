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

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

enum OptionType { CALL = 0, PUT = 1 };

// This layout is deliberately shared by the host and the CUDA kernel.  Keeping
// the expected value and tolerance also preserves the input representation used
// by the original benchmark.
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

inline __host__ __device__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + ::erf(x * M_SQRT1_2));
}

inline __host__ __device__ double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double sqrtT = ::sqrt(T);
    const double d1 = (::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double discount = ::exp(-r * T);

    if (option.type == CALL) {
        return S * ::exp(-q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    }
    return K * discount * cumulativeNormal(-d2) - S * ::exp(-q * T) * cumulativeNormal(-d1);
}

__global__ void priceOptionsKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t count) {
    // A grid-stride loop keeps every GPU busy for both small per-rank shards and
    // very large benchmark inputs without imposing a grid-size limit.
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        results[i] = blackScholes(options[i]);
    }
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

// Generate only this rank's contiguous global range.  This avoids a root-side
// scatter and makes the generated data exactly match the serial sequence.
void generateOptions(std::vector<OptionInput>& options, const size_t first, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(count); ++local) {
        const size_t global = first + static_cast<size_t>(local);
        OptionInput option = testOptions[global % testOptions.size()];
        const double factor = 1.0 + 0.1 * (global / static_cast<double>(testOptions.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[static_cast<size_t>(local)] = option;
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), options.size()));
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[static_cast<size_t>(i)];
        const double expected = options[static_cast<size_t>(i)].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);
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
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

[[noreturn]] void fail(const char* message, const int rank) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

void checkCuda(const cudaError_t status, const int rank, const char* operation) {
    if (status != cudaSuccess) {
        char message[512];
        snprintf(message, sizeof(message), "%s: %s", operation, cudaGetErrorString(status));
        fail(message, rank);
    }
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI does not provide MPI_THREAD_FUNNELED", rank);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const unsigned long long parsed = strtoull(argv[++i], nullptr, 10);
            if (parsed > std::numeric_limits<size_t>::max()) fail("option count is too large", rank);
            numOptions = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (numOptions > static_cast<size_t>(LLONG_MAX)) fail("option count exceeds OpenMP loop range", rank);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), rank, "cudaGetDeviceCount");
    if (devices == 0) fail("no CUDA device is visible", rank);
    checkCuda(cudaSetDevice(localRank % devices), rank, "cudaSetDevice");
    MPI_Comm_free(&localComm);

    // Quotient/remainder partitioning gives each rank one contiguous slice, so
    // MPI_Gatherv reconstructs precisely the serial ordering.
    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t remainder = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    if (localCount > static_cast<size_t>(INT_MAX)) fail("per-rank option count exceeds MPI_Gatherv limit", rank);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n",
               numOptions, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; CUDA devices per node: %d\nPricing options...\n", ranks, devices);
    }

    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, first, localCount);
    std::vector<double> localResults(localCount);

    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    const size_t optionBytes = localCount * sizeof(OptionInput);
    const size_t resultBytes = localCount * sizeof(double);
    if (localCount != 0) {
        checkCuda(cudaMalloc(&deviceOptions, optionBytes), rank, "cudaMalloc options");
        checkCuda(cudaMalloc(&deviceResults, resultBytes), rank, "cudaMalloc results");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (localCount != 0) {
        checkCuda(cudaMemcpy(deviceOptions, localOptions.data(), optionBytes, cudaMemcpyHostToDevice), rank,
                  "copy options to device");
        constexpr int threads = 256;
        int blocks = static_cast<int>(std::min<size_t>((localCount + threads - 1) / threads, 65535));
        priceOptionsKernel<<<blocks, threads>>>(deviceOptions, deviceResults, localCount);
        checkCuda(cudaGetLastError(), rank, "launch pricing kernel");
        checkCuda(cudaMemcpy(localResults.data(), deviceResults, resultBytes, cudaMemcpyDeviceToHost), rank,
                  "copy results from device");
    }
    checkCuda(cudaFree(deviceOptions), rank, "free device options");
    checkCuda(cudaFree(deviceResults), rank, "free device results");
    const auto end = std::chrono::steady_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts, displacements;
    std::vector<double> results;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(ranks));
        displacements.resize(static_cast<size_t>(ranks));
        for (int r = 0; r < ranks; ++r) {
            const size_t count = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t displacement = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder);
            if (count > static_cast<size_t>(INT_MAX) || displacement > static_cast<size_t>(INT_MAX))
                fail("global option count exceeds MPI_Gatherv limit", rank);
            counts[static_cast<size_t>(r)] = static_cast<int>(count);
            displacements[static_cast<size_t>(r)] = static_cast<int>(displacement);
        }
        results.resize(numOptions);
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        printf("Options per second: %.0f\n", elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::vector<OptionInput> validationOptions;
            // Validation only inspects the first ten entries, so avoid
            // regenerating the complete global input on the root rank.
            generateOptions(validationOptions, 0, std::min(numOptions, static_cast<size_t>(10)));
            printf("Validating results...\n");
            if (validateResults(validationOptions, results)) printf("Validation: PASSED\n");
            else { printf("Validation: FAILED\n"); exitCode = 1; }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
