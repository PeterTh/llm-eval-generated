#include <algorithm>
#include <array>
#include <cerrno>
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
    double strike, spot, q, r, t, vol, value, tol;
};

// Kept as one source file so the CUDA compiler can inline the complete pricing
// expression into the kernel.
__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__device__ __forceinline__ double blackScholes(const OptionInput& option) {
    const double S = option.spot;
    const double K = option.strike;
    const double T = option.t;
    const double sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double rootT = sqrt(T);
    const double d1 = (log(S / K) +
                       (option.r - option.q + 0.5 * sigma * sigma) * T) /
                      (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double discountR = exp(-option.r * T);
    const double discountQ = exp(-option.q * T);
    if (option.type == CALL) {
        return S * discountQ * cumulativeNormal(d1) -
               K * discountR * cumulativeNormal(d2);
    }
    return K * discountR * cumulativeNormal(-d2) -
           S * discountQ * cumulativeNormal(-d1);
}

__global__ void priceOptions(const OptionInput* __restrict__ options,
                             double* __restrict__ results, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) results[i] = blackScholes(options[i]);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
             {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
             {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
             {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
             {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}}};
}

void generateOptions(OptionInput* options, size_t count, size_t globalOffset) {
    constexpr auto tests = getTestOptions();
#pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(count); ++j) {
        const size_t i = globalOffset + static_cast<size_t>(j);
        OptionInput option = tests[i % tests.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(tests.size()));
        option.spot *= factor;
        option.strike *= factor;
        options[j] = option;
    }
}

bool validateResults(const std::vector<double>& results, size_t numOptions) {
    const int checks = static_cast<int>(std::min<size_t>(10, numOptions));
    int invalid = 0;
    std::vector<OptionInput> options(checks);
    generateOptions(options.data(), options.size(), 0);
    std::printf("Checking computed option prices:\n");
    for (int i = 0; i < checks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double relative = std::fabs(computed - expected) /
                                (std::fabs(expected) + 1e-10);
        std::printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, expected, relative);
        invalid += computed < 0.0 || computed > 1000.0 || !std::isfinite(computed);
    }
    return invalid == 0;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

[[noreturn]] void failCuda(cudaError_t error, const char* operation, int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
    failCuda(e_, #call, rank); } while (0)

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false, help = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno || !end || *end || value > SIZE_MAX) parseError = 1;
            else numOptions = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else parseError = 1;
    }
    if (help || parseError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    // Select GPUs independently on each node, allowing the same rank layout on
    // every node without requiring launcher-specific environment variables.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t base = numOptions / static_cast<size_t>(ranks);
    const size_t extra = numOptions % static_cast<size_t>(ranks);
    const size_t localCount = base + (static_cast<size_t>(rank) < extra);
    const size_t offset = static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), extra);

    OptionInput* hostOptions = nullptr;
    double* hostResults = nullptr;
    OptionInput* deviceOptions = nullptr;
    double* deviceResults = nullptr;
    if (localCount) {
        CUDA_CHECK(cudaMallocHost(&hostOptions, localCount * sizeof(*hostOptions)));
        CUDA_CHECK(cudaMallocHost(&hostResults, localCount * sizeof(*hostResults)));
        CUDA_CHECK(cudaMalloc(&deviceOptions, localCount * sizeof(*deviceOptions)));
        CUDA_CHECK(cudaMalloc(&deviceResults, localCount * sizeof(*deviceResults)));
        generateOptions(hostOptions, localCount, offset);
    }

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", ranks,
                    omp_get_max_threads());
        std::printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        CUDA_CHECK(cudaMemcpyAsync(deviceOptions, hostOptions,
                                   localCount * sizeof(*hostOptions),
                                   cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        const size_t blocks64 = (localCount + threads - 1) / threads;
        if (blocks64 > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "Rank %d: local problem is too large for one launch\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        priceOptions<<<static_cast<unsigned>(blocks64), threads>>>(
            deviceOptions, deviceResults, localCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hostResults, deviceResults,
                                   localCount * sizeof(*hostResults),
                                   cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> results;
    if (printResults || validate) {
        if (numOptions > static_cast<size_t>(INT_MAX) || localCount > INT_MAX) {
            if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI int counts\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        std::vector<int> counts, displacements;
        if (rank == 0) {
            results.resize(numOptions);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t c = base + (static_cast<size_t>(r) < extra);
                const size_t d = static_cast<size_t>(r) * base +
                                 std::min(static_cast<size_t>(r), extra);
                counts[r] = static_cast<int>(c);
                displacements[r] = static_cast<int>(d);
            }
        }
        MPI_Gatherv(hostResults, static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int status = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Options per second: %.0f\n", elapsed > 0.0 ? numOptions / elapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::printf("Validating results...\n");
            const bool valid = validateResults(results, numOptions);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (deviceResults) CUDA_CHECK(cudaFree(deviceResults));
    if (deviceOptions) CUDA_CHECK(cudaFree(deviceOptions));
    if (hostResults) CUDA_CHECK(cudaFreeHost(hostResults));
    if (hostOptions) CUDA_CHECK(cudaFreeHost(hostOptions));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return status;
}
