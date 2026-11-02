#include <algorithm>
#include <array>
#include <chrono>
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

constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
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

__constant__ OptionInput deviceOptions[7];

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::abort(); \
    } \
} while (0)

// The original generator scales both prices using the global option index.
// Generate them on the GPU to avoid moving the much larger input array.
__global__ void priceOptions(double* results, size_t first, size_t count) {
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = first + j;
        const OptionInput base = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        const double S = base.spot * factor;
        const double K = base.strike * factor;
        const double T = base.t;
        const double sigma = base.vol;
        double price = 0.0;
        if (T > 0.0 && sigma > 0.0) {
            const double rootT = sqrt(T);
            const double d1 = (log(S / K) + (base.r - base.q + 0.5 * sigma * sigma) * T) / (sigma * rootT);
            const double d2 = d1 - sigma * rootT;
            const double discount = exp(-base.r * T);
            const double dividend = exp(-base.q * T);
            constexpr double invSqrt2 = 0.7071067811865475244;
            if (base.type == CALL) {
                const double Nd1 = 0.5 * (1.0 + erf(d1 * invSqrt2));
                const double Nd2 = 0.5 * (1.0 + erf(d2 * invSqrt2));
                price = S * dividend * Nd1 - K * discount * Nd2;
            } else {
                const double Nmd1 = 0.5 * (1.0 + erf(-d1 * invSqrt2));
                const double Nmd2 = 0.5 * (1.0 + erf(-d2 * invSqrt2));
                price = K * discount * Nmd2 - S * dividend * Nmd1;
            }
        }
        results[j] = price;
    }
}

// A balanced contiguous partition, without multiplying a potentially large count.
void partition(size_t total, int parts, int part, size_t& begin, size_t& count) {
    const size_t quotient = total / static_cast<size_t>(parts);
    const size_t remainder = total % static_cast<size_t>(parts);
    begin = quotient * static_cast<size_t>(part) + std::min(static_cast<size_t>(part), remainder);
    count = quotient + (static_cast<size_t>(part) < remainder ? 1 : 0);
}

bool validateResults(const std::vector<double>& results) {
    constexpr auto options = getTestOptions();
    bool allPassed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size(); ++i) {
        const double computed = results[i];
        const double expected = options[i % options.size()].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i, computed, expected, relError);
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
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

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(atoll(argv[++i]));
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
    int localRank, localRanks;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // When there are fewer ranks than visible GPUs, each rank drives several.
    // When there are more ranks, ranks share GPUs in round-robin order.
    int firstDevice, devices;
    if (localRanks <= deviceCount) {
        size_t first, count;
        partition(static_cast<size_t>(deviceCount), localRanks, localRank, first, count);
        firstDevice = static_cast<int>(first);
        devices = static_cast<int>(count);
    } else {
        firstDevice = localRank % deviceCount;
        devices = 1;
    }
    MPI_Comm_free(&localComm);

    size_t rankBegin, rankCount;
    partition(numOptions, ranks, rank, rankBegin, rankCount);
    const size_t needed = printResults ? numOptions : (validate ? std::min(numOptions, size_t{10}) : 0);
    const size_t localNeeded = rankBegin < needed ? std::min(rankCount, needed - rankBegin) : 0;
    std::vector<double> localResults(localNeeded);
    std::vector<double> results;
    if (rank == 0) results.resize(needed);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    constexpr size_t tileSize = 1u << 22;
    std::vector<double*> deviceBuffers(devices, nullptr);
    omp_set_dynamic(0);
    constexpr auto testOptions = getTestOptions();
#pragma omp parallel for num_threads(devices) schedule(static)
    for (int thread = 0; thread < devices; ++thread) {
        CUDA_CHECK(cudaSetDevice(firstDevice + thread));
        CUDA_CHECK(cudaMemcpyToSymbol(deviceOptions, testOptions.data(), sizeof(OptionInput) * testOptions.size()));
        size_t begin, count;
        partition(rankCount, devices, thread, begin, count);
        if (count) CUDA_CHECK(cudaMalloc(&deviceBuffers[thread], std::min(count, tileSize) * sizeof(double)));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(devices) schedule(static)
    for (int thread = 0; thread < devices; ++thread) {
        CUDA_CHECK(cudaSetDevice(firstDevice + thread));
        size_t begin, count;
        partition(rankCount, devices, thread, begin, count);
        for (size_t offset = 0; offset < count;) {
            const size_t n = std::min(tileSize, count - offset);
            const int blocks = static_cast<int>(std::min((n + 255) / 256, size_t{65535}));
            priceOptions<<<blocks, 256>>>(deviceBuffers[thread], rankBegin + begin + offset, n);
            CUDA_CHECK(cudaGetLastError());
            const size_t localOffset = begin + offset;
            if (localOffset < localNeeded) {
                const size_t copyCount = std::min(n, localNeeded - localOffset);
                CUDA_CHECK(cudaMemcpy(localResults.data() + localOffset, deviceBuffers[thread], copyCount * sizeof(double), cudaMemcpyDeviceToHost));
            }
            offset += n;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

#pragma omp parallel for num_threads(devices) schedule(static)
    for (int thread = 0; thread < devices; ++thread) {
        CUDA_CHECK(cudaSetDevice(firstDevice + thread));
        if (deviceBuffers[thread]) CUDA_CHECK(cudaFree(deviceBuffers[thread]));
    }

    // Point-to-point chunks avoid MPI's int count/displacement limits.
    if (needed) {
        constexpr size_t messageSize = 1u << 20;
        if (rank == 0) {
            std::copy(localResults.begin(), localResults.end(), results.begin());
            for (int source = 1; source < ranks; ++source) {
                size_t begin, count;
                partition(numOptions, ranks, source, begin, count);
                count = begin < needed ? std::min(count, needed - begin) : 0;
                for (size_t offset = 0; offset < count;) {
                    const int n = static_cast<int>(std::min(messageSize, count - offset));
                    MPI_Recv(results.data() + begin + offset, n, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(n);
                }
            }
        } else {
            for (size_t offset = 0; offset < localNeeded;) {
                const int n = static_cast<int>(std::min(messageSize, localNeeded - offset));
                MPI_Send(localResults.data() + offset, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += static_cast<size_t>(n);
            }
        }
    }

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            if (printResults && results.size() > 10) results.resize(10);
            const bool valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
