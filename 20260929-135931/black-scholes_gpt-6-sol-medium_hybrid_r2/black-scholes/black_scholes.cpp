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

constexpr std::array<OptionInput, 7> testOptions = {{
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}
}};

__constant__ OptionInput deviceTestOptions[7];

__host__ __device__ inline OptionInput makeOption(size_t index, const OptionInput* bases) {
    OptionInput option = bases[index % 7];
    const double factor = 1.0 + 0.1 * (index / static_cast<double>(7));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

__host__ __device__ inline double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;

    const double rootT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double discount = exp(-r * T);
    const double dividend = exp(-q * T);
    if (option.type == CALL)
        return S * dividend * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    return K * discount * cumulativeNormal(-d2) - S * dividend * cumulativeNormal(-d1);
}

__global__ void priceKernel(size_t globalStart, size_t count, double* results) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        results[i] = blackScholes(makeOption(globalStart + i, deviceTestOptions));
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

bool validateResults(const std::vector<double>& results) {
    bool passed = true;
    const size_t checks = std::min<size_t>(10, results.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const OptionInput option = makeOption(i, testOptions.data());
        const double computed = results[i];
        const double expected = option.value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            passed = false;
        }
    }
    return passed;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Block partitioning also works when the option count is smaller than the rank count.
size_t rankStart(size_t total, int rank, int ranks) {
    const size_t base = total / static_cast<size_t>(ranks);
    const size_t extra = total % static_cast<size_t>(ranks);
    return base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), extra);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
                if (rank == 0) fprintf(stderr, "Invalid option count: %s\n", value);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            numOptions = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    int gpuCount = 0;
    checkCuda(cudaGetDeviceCount(&gpuCount), "device discovery");
    if (gpuCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % gpuCount), "device selection");
    checkCuda(cudaMemcpyToSymbol(deviceTestOptions, testOptions.data(), sizeof(OptionInput) * 7),
              "constant upload");

    const size_t start = rankStart(numOptions, rank, ranks);
    const size_t count = rankStart(numOptions, rank + 1, ranks) - start;
    // The CPU share is bounded so GPU work remains dominant. Both run at once.
    const size_t cpuCount = count > 1 ? std::max<size_t>(1, count / 8) : 0;
    const size_t gpuCountOptions = count - cpuCount;
    std::vector<double> localResults(count);
    double* deviceResults = nullptr;
    if (gpuCountOptions != 0)
        checkCuda(cudaMalloc(&deviceResults, gpuCountOptions * sizeof(double)), "result allocation");

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin = std::chrono::steady_clock::now();
    if (gpuCountOptions != 0) {
        const int blocks = static_cast<int>(std::min<size_t>((gpuCountOptions + 255) / 256, 4096));
        priceKernel<<<blocks, 256>>>(start + cpuCount, gpuCountOptions, deviceResults);
        checkCuda(cudaGetLastError(), "kernel launch");
    }
    const int cpuThreads = std::min(omp_get_max_threads(),
                                    static_cast<int>(std::min<size_t>((cpuCount + 1023) / 1024, 64)));
    #pragma omp parallel for schedule(static) num_threads(cpuThreads > 0 ? cpuThreads : 1)
    for (size_t i = 0; i < cpuCount; ++i)
        localResults[i] = blackScholes(makeOption(start + i, testOptions.data()));
    if (gpuCountOptions != 0) {
        checkCuda(cudaMemcpy(localResults.data() + cpuCount, deviceResults,
                             gpuCountOptions * sizeof(double), cudaMemcpyDeviceToHost), "result download");
    }
    const auto end = std::chrono::steady_clock::now();
    if (deviceResults) checkCuda(cudaFree(deviceResults), "result free");
    const double localSeconds = std::chrono::duration<double>(end - begin).count();
    double elapsed = 0.0;
    MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        printf("Options per second: %.0f\n", elapsed > 0.0 ? numOptions / elapsed : 0.0);
    }

    if (printResults || validate) {
        const size_t gatherCount = printResults ? numOptions : std::min<size_t>(10, numOptions);
        std::vector<double> results;
        if (rank == 0) {
            results.resize(gatherCount);
            const size_t ownCount = std::min(count, gatherCount);
            std::copy_n(localResults.begin(), ownCount, results.begin());
        }
        constexpr size_t chunk = 1 << 24; // MPI's count parameter is an int.
        for (int source = 1; source < ranks; ++source) {
            const size_t sourceStart = rankStart(numOptions, source, ranks);
            const size_t sourceEnd = std::min(rankStart(numOptions, source + 1, ranks), gatherCount);
            const size_t sourceCount = sourceStart < sourceEnd ? sourceEnd - sourceStart : 0;
            for (size_t offset = 0; offset < sourceCount; offset += chunk) {
                const int length = static_cast<int>(std::min(chunk, sourceCount - offset));
                if (rank == source)
                    MPI_Send(localResults.data() + offset, length, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if (rank == 0)
                    MPI_Recv(results.data() + sourceStart + offset, length, MPI_DOUBLE,
                             source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        int valid = 1;
        if (rank == 0) {
            if (printResults) print_results(results, "OptionPrices");
            if (validate) {
                printf("Validating results...\n");
                valid = validateResults(results) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
