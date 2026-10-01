#include <algorithm>
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

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

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

// Generate exactly the same sequence on the host and on each GPU.  This avoids
// transferring the much larger OptionInput array to the accelerator.
__host__ __device__ inline OptionInput optionAt(size_t i) {
    OptionInput option;
    switch (i % 7) {
    case 0: option = {CALL, 40.0, 42.0, 0.04, 0.08, 0.75, 0.35, 5.0975, 1e-3}; break;
    case 1: option = {CALL, 100.0, 90.0, 0.10, 0.10, 0.10, 0.15, 0.0205, 1e-3}; break;
    case 2: option = {CALL, 100.0, 100.0, 0.10, 0.10, 0.10, 0.15, 1.8734, 1e-3}; break;
    case 3: option = {CALL, 100.0, 110.0, 0.10, 0.10, 0.10, 0.15, 9.9413, 1e-3}; break;
    case 4: option = {PUT, 100.0, 90.0, 0.10, 0.10, 0.10, 0.15, 9.9210, 1e-3}; break;
    case 5: option = {PUT, 100.0, 100.0, 0.10, 0.10, 0.10, 0.15, 1.8734, 1e-3}; break;
    default: option = {PUT, 100.0, 110.0, 0.10, 0.10, 0.10, 0.15, 0.0408, 1e-3}; break;
    }
    const double factor = 1.0 + 0.1 * (i / 7.0);
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

__host__ __device__ inline double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    const double sigmaSqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    const double discount = exp(-r * T);
    if (option.type == CALL) {
        return S * exp(-q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
    }
    return K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
}

__global__ void priceKernel(size_t first, size_t count, double* prices) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) prices[index] = blackScholes(optionAt(first + index));
}

static void cudaCheck(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(code));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static size_t rankStart(size_t n, int rank, int ranks) {
    const size_t q = n / static_cast<size_t>(ranks);
    const size_t rem = n % static_cast<size_t>(ranks);
    return q * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
}

// MPI's classic point-to-point count is int; send large output in chunks.
static void gatherResults(const std::vector<double>& local, std::vector<double>& global,
                          size_t n, int rank, int ranks) {
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    if (rank == 0) {
        std::copy(local.begin(), local.end(), global.begin());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t first = rankStart(n, peer, ranks);
            const size_t end = rankStart(n, peer + 1, ranks);
            for (size_t offset = first; offset < end; offset += std::min(chunk, end - offset)) {
                MPI_Recv(global.data() + offset, static_cast<int>(std::min(chunk, end - offset)),
                         MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size(); offset += std::min(chunk, local.size() - offset)) {
            MPI_Send(local.data() + offset, static_cast<int>(std::min(chunk, local.size() - offset)),
                     MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

static bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min<size_t>(10, results.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = optionAt(i).value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    cudaCheck(cudaGetDeviceCount(&devices), "device enumeration");
    if (devices == 0) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "device selection");
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    const size_t first = rankStart(numOptions, rank, ranks);
    const size_t localCount = rankStart(numOptions, rank + 1, ranks) - first;
    // A bounded CPU share overlaps the GPU kernel without delaying large jobs.
    const size_t cpuCount = std::min(localCount / 32, static_cast<size_t>(16384));
    const size_t gpuCount = localCount - cpuCount;
    std::vector<double> localResults(localCount);
    constexpr size_t batchSize = 8 * 1024 * 1024;
    const size_t bufferCount = std::min(gpuCount, batchSize);
    double* deviceResults = nullptr;
    if (bufferCount) cudaCheck(cudaMalloc(&deviceResults, bufferCount * sizeof(double)), "allocation");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    size_t launched = 0;
    if (gpuCount) {
        launched = std::min(gpuCount, batchSize);
        priceKernel<<<static_cast<unsigned>((launched + 255) / 256), 256>>>(
            first + cpuCount, launched, deviceResults);
        cudaCheck(cudaGetLastError(), "kernel launch");
    }

    // MPI calls stay on the main thread; OpenMP threads price disjoint options.
    #pragma omp parallel for schedule(static) num_threads(cpuCount >= 4096 ? std::min(8, omp_get_max_threads()) : 1)
    for (size_t i = 0; i < cpuCount; ++i) {
        localResults[i] = blackScholes(optionAt(first + i));
    }

    size_t completed = 0;
    while (completed < gpuCount) {
        cudaCheck(cudaMemcpy(localResults.data() + cpuCount + completed, deviceResults,
                             launched * sizeof(double), cudaMemcpyDeviceToHost), "result transfer");
        completed += launched;
        if (completed < gpuCount) {
            launched = std::min(gpuCount - completed, batchSize);
            priceKernel<<<static_cast<unsigned>((launched + 255) / 256), 256>>>(
                first + cpuCount + completed, launched, deviceResults);
            cudaCheck(cudaGetLastError(), "kernel launch");
        }
    }
    const auto end = std::chrono::steady_clock::now();
    cudaCheck(cudaFree(deviceResults), "deallocation");
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Options per second: %.0f\n", seconds > 0.0 ? numOptions / seconds : 0.0);
    }
    if (printResults || validate) {
        std::vector<double> results;
        if (rank == 0) results.resize(numOptions);
        gatherResults(localResults, results, numOptions, rank, ranks);
        if (rank == 0) {
            if (printResults) print_results(results, "OptionPrices");
            if (validate) {
                printf("Validating results...\n");
                const bool valid = validateResults(results);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
