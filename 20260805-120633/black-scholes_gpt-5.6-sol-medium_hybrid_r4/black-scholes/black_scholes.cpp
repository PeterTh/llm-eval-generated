#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

enum OptionType { CALL = 0, PUT = 1 };

struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

__host__ __device__ inline double cumulativeNormal(double x) noexcept {
    return 0.5 * (1.0 + erf(x * 0.70710678118654752440));
}

__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
    if (option.t <= 0.0 || option.vol <= 0.0) return 0.0;
    const double rootT = sqrt(option.t);
    const double d1 = (log(option.spot / option.strike) +
        (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
        (option.vol * rootT);
    const double d2 = d1 - option.vol * rootT;
    const double discountR = exp(-option.r * option.t);
    const double discountQ = exp(-option.q * option.t);
    if (option.type == CALL) {
        return option.spot * discountQ * cumulativeNormal(d1) -
               option.strike * discountR * cumulativeNormal(d2);
    }
    return option.strike * discountR * cumulativeNormal(-d2) -
           option.spot * discountQ * cumulativeNormal(-d1);
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

// Constructing inputs in the kernel avoids transferring an input array over PCIe.
__host__ __device__ inline OptionInput optionAt(unsigned long long index) noexcept {
    OptionInput o;
    switch (index % 7ULL) {
        case 0: o = {CALL, 40., 42., .04, .08, .75, .35, 5.0975, 1.e-3}; break;
        case 1: o = {CALL, 100., 90., .10, .10, .10, .15, .0205, 1.e-3}; break;
        case 2: o = {CALL, 100., 100., .10, .10, .10, .15, 1.8734, 1.e-3}; break;
        case 3: o = {CALL, 100., 110., .10, .10, .10, .15, 9.9413, 1.e-3}; break;
        case 4: o = {PUT, 100., 90., .10, .10, .10, .15, 9.9210, 1.e-3}; break;
        case 5: o = {PUT, 100., 100., .10, .10, .10, .15, 1.8734, 1.e-3}; break;
        default:o = {PUT, 100., 110., .10, .10, .10, .15, .0408, 1.e-3}; break;
    }
    const double factor = 1.0 + 0.1 * (static_cast<double>(index) / 7.0);
    o.spot *= factor;
    o.strike *= factor;
    return o;
}

__global__ void priceKernel(double* __restrict__ output, unsigned long long first,
                            unsigned long long count) {
    unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    for (; i < count; i += stride) output[i] = blackScholes(optionAt(first + i));
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\nOptions:\n", program);
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool parseCount(const char* text, unsigned long long& value) {
    if (!text[0] || text[0] == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || *end != '\0' || parsed > static_cast<unsigned long long>(SIZE_MAX)) return false;
    value = parsed;
    return true;
}

static bool validateResults(const std::vector<double>& results) {
    bool passed = true;
    const size_t checks = std::min<size_t>(10, results.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const OptionInput option = optionAt(i);
        const double computed = results[i];
        const double error = fabs(computed - option.value);
        const double relative = error / (fabs(option.value) + 1.e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, option.value, relative);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            passed = false;
        }
    }
    return passed;
}

static void cudaFailure(cudaError_t error, int device, int rank, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "MPI rank %d, CUDA device %d: %s failed: %s\n",
                rank, device, operation, cudaGetErrorString(error));
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI initialization with thread support failed\n");
        return 1;
    }

    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    unsigned long long numOptions = 10000;
    bool validate = false, printResults = false, help = false, argsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) argsValid = parseCount(argv[++i], numOptions);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); argsValid = false; }
    }
    if (help || !argsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argsValid ? 0 : 1;
    }

    const unsigned long long quotient = numOptions / static_cast<unsigned long long>(ranks);
    const unsigned long long remainder = numOptions % static_cast<unsigned long long>(ranks);
    const unsigned long long localCount = quotient + (static_cast<unsigned long long>(rank) < remainder);
    const unsigned long long first = quotient * rank + std::min<unsigned long long>(rank, remainder);

    int deviceCount = 0;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess || deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "CUDA-capable GPU required: %s\n", cudaGetErrorString(cudaStatus));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    MPI_Comm_free(&localComm);

    int firstDevice, usedDevices;
    if (deviceCount >= localRanks) {
        firstDevice = deviceCount * localRank / localRanks;
        const int endDevice = deviceCount * (localRank + 1) / localRanks;
        usedDevices = endDevice - firstDevice;
    } else {
        firstDevice = localRank % deviceCount;
        usedDevices = 1;
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %llu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel configuration: %d MPI rank(s), OpenMP, CUDA (%d visible GPU(s) on rank 0 node)\n",
               ranks, deviceCount);
        printf("Pricing options...\n");
    }

    const bool copyToHost = validate || printResults;
    std::vector<double> localResults;
    if (copyToHost) localResults.resize(static_cast<size_t>(localCount));
    int localFailure = 0;
    omp_set_dynamic(0);

    // Establish contexts and allocate before timing, just as the original benchmark
    // allocated its result storage before its computation timer.
    std::vector<double*> deviceResults(static_cast<size_t>(usedDevices), nullptr);
    std::vector<unsigned long long> deviceCounts(static_cast<size_t>(usedDevices));
    std::vector<unsigned long long> deviceOffsets(static_cast<size_t>(usedDevices));
#pragma omp parallel num_threads(usedDevices) reduction(|:localFailure)
    {
        const int thread = omp_get_thread_num();
        const int threads = omp_get_num_threads();
        const int device = firstDevice + thread;
        const unsigned long long count = localCount / threads +
            (static_cast<unsigned long long>(thread) < localCount % threads);
        const unsigned long long offset = localCount / threads * thread +
            std::min<unsigned long long>(thread, localCount % threads);
        deviceCounts[thread] = count;
        deviceOffsets[thread] = offset;
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess) error = cudaFree(nullptr); // Create the CUDA context.
        if (error == cudaSuccess && count)
            error = cudaMalloc(&deviceResults[thread], count * sizeof(double));
        if (error != cudaSuccess) {
            cudaFailure(error, device, rank, "initialization");
            localFailure = 1;
        }
    }
    int setupFailure = 0;
    MPI_Allreduce(&localFailure, &setupFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (setupFailure) {
        if (rank == 0) fprintf(stderr, "CUDA initialization failed\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
#pragma omp parallel num_threads(usedDevices) reduction(|:localFailure)
    {
        const int thread = omp_get_thread_num();
        const int device = firstDevice + thread;
        const unsigned long long threadCount = deviceCounts[thread];
        const unsigned long long threadOffset = deviceOffsets[thread];
        double* const output = deviceResults[thread];
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess && threadCount) {
            constexpr int blockSize = 256;
            const unsigned long long wanted = (threadCount + blockSize - 1) / blockSize;
            const int blocks = static_cast<int>(std::min<unsigned long long>(wanted, 65535));
            priceKernel<<<blocks, blockSize>>>(output, first + threadOffset, threadCount);
            error = cudaGetLastError();
        }
        if (error == cudaSuccess && threadCount && copyToHost) {
            error = cudaMemcpy(localResults.data() + threadOffset, output,
                               threadCount * sizeof(double), cudaMemcpyDeviceToHost);
        } else if (error == cudaSuccess && threadCount) {
            error = cudaDeviceSynchronize();
        }
        if (error != cudaSuccess) { cudaFailure(error, device, rank, "pricing"); localFailure = 1; }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - startTime;
    double globalElapsed = 0.0;
    MPI_Reduce(&elapsed, &globalElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

#pragma omp parallel num_threads(usedDevices)
    {
        const int thread = omp_get_thread_num();
        cudaSetDevice(firstDevice + thread);
        if (deviceResults[thread]) cudaFree(deviceResults[thread]);
    }

    int globalFailure = 0;
    MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalFailure) {
        if (rank == 0) fprintf(stderr, "CUDA pricing failed\n");
        MPI_Finalize();
        return 2;
    }

    std::vector<double> results;
    if (copyToHost) {
        if (numOptions > static_cast<unsigned long long>(INT_MAX) || localCount > INT_MAX) {
            if (rank == 0) fprintf(stderr, "Result collection currently supports at most INT_MAX options\n");
            MPI_Finalize();
            return 2;
        }
        std::vector<int> counts, displacements;
        if (rank == 0) {
            results.resize(static_cast<size_t>(numOptions));
            counts.resize(ranks); displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                counts[r] = static_cast<int>(quotient + (static_cast<unsigned long long>(r) < remainder));
                displacements[r] = static_cast<int>(quotient * r + std::min<unsigned long long>(r, remainder));
            }
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", globalElapsed * 1000.0);
        printf("Options per second: %.0f\n", globalElapsed > 0.0 ? numOptions / globalElapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
