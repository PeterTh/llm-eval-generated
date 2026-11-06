#include <algorithm>
#include <array>
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

// The same seven inputs are used by every index, with only spot and strike
// scaled. Keeping them on the device avoids distributing and copying inputs.
#define TEST_OPTIONS \
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3}, \
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3}, \
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3}, \
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3}, \
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3}, \
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3}, \
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}
static constexpr std::array<OptionInput, 7> testOptions = {{TEST_OPTIONS}};
__device__ __constant__ OptionInput deviceOptions[7] = {TEST_OPTIONS};
#undef TEST_OPTIONS

__device__ inline double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__global__ void priceOptions(double* results, size_t start, size_t count) {
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = start + j;
        const OptionInput option = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        const double S = option.spot * factor;
        const double K = option.strike * factor;
        const double T = option.t;
        const double sigma = option.vol;
        if (T <= 0.0 || sigma <= 0.0) {
            results[j] = 0.0;
            continue;
        }
        const double d1 = (log(S / K) + (option.r - option.q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        const double d2 = d1 - sigma * sqrt(T);
        const double Nd1 = cumulativeNormal(d1);
        const double Nd2 = cumulativeNormal(d2);
        const double discount = exp(-option.r * T);
        if (option.type == CALL) {
            results[j] = S * exp(-option.q * T) * Nd1 - K * discount * Nd2;
        } else {
            results[j] = K * discount * cumulativeNormal(-d2) - S * exp(-option.q * T) * cumulativeNormal(-d1);
        }
    }
}

struct Range { size_t start, count; };
static Range partition(size_t count, int parts, int part) {
    const size_t base = count / static_cast<size_t>(parts);
    const size_t extra = count % static_cast<size_t>(parts);
    return {base * static_cast<size_t>(part) + std::min(static_cast<size_t>(part), extra),
            base + (static_cast<size_t>(part) < extra)};
}

static bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % 7].value;
        const double error = std::fabs(computed - expected);
        const double relError = error / (std::fabs(expected) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i, computed, expected, relError);
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

struct DeviceWork {
    int device;
    Range range;
    double* buffer = nullptr;
    cudaError_t error = cudaSuccess;
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not support the required thread level\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = atoll(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { parseStatus = 2; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1; break;
        }
    }
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
        fflush(stdout);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // With fewer ranks than visible GPUs, give each rank a disjoint group.
    // With more ranks than GPUs, ranks share devices in round-robin order.
    Range devices = localRanks <= deviceCount ? partition(deviceCount, localRanks, localRank)
                                              : Range{static_cast<size_t>(localRank % deviceCount), 1};
    const Range rankRange = partition(numOptions, ranks, rank);
    const size_t wanted = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    const size_t localWanted = rankRange.start < wanted ? std::min(rankRange.count, wanted - rankRange.start) : 0;
    std::vector<double> localResults(localWanted);
    std::vector<DeviceWork> work(devices.count);
    constexpr size_t chunkSize = 8 * 1024 * 1024;
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(devices.count))
    for (size_t d = 0; d < devices.count; ++d) {
        DeviceWork& w = work[d];
        w.device = static_cast<int>(devices.start + d);
        w.range = partition(rankRange.count, static_cast<int>(devices.count), static_cast<int>(d));
        w.range.start += rankRange.start;
        w.error = cudaSetDevice(w.device);
        if (w.error == cudaSuccess && w.range.count)
            w.error = cudaMalloc(reinterpret_cast<void**>(&w.buffer), std::min(chunkSize, w.range.count) * sizeof(double));
    }
    int localError = 0;
    for (const auto& w : work) if (w.error != cudaSuccess) {
        fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, w.device, cudaGetErrorString(w.error));
        localError = 1;
    }
    int anyError = 0;
    MPI_Allreduce(&localError, &anyError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyError) MPI_Abort(MPI_COMM_WORLD, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(devices.count))
    for (size_t d = 0; d < devices.count; ++d) {
        DeviceWork& w = work[d];
        w.error = cudaSetDevice(w.device);
        for (size_t offset = 0; w.error == cudaSuccess && offset < w.range.count;) {
            const size_t count = std::min(chunkSize, w.range.count - offset);
            const size_t globalStart = w.range.start + offset;
            const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (count + 255) / 256));
            priceOptions<<<blocks, 256>>>(w.buffer, globalStart, count);
            w.error = cudaGetLastError();
            if (w.error == cudaSuccess && globalStart < wanted) {
                const size_t copyCount = std::min(count, wanted - globalStart);
                w.error = cudaMemcpy(localResults.data() + (globalStart - rankRange.start), w.buffer,
                                     copyCount * sizeof(double), cudaMemcpyDeviceToHost);
            }
            offset += count;
        }
        if (w.error == cudaSuccess) w.error = cudaDeviceSynchronize();
    }
    const double elapsed = MPI_Wtime() - start;
    for (const auto& w : work) if (w.error != cudaSuccess) {
        fprintf(stderr, "Rank %d, CUDA device %d: %s\n", rank, w.device, cudaGetErrorString(w.error));
        localError = 1;
    }
    MPI_Allreduce(&localError, &anyError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyError) MPI_Abort(MPI_COMM_WORLD, 1);
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }

    // MPI counts are int, so transfer large result ranges in bounded chunks.
    std::vector<double> results;
    if (rank == 0) {
        results.resize(wanted);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int source = 1; source < ranks; ++source) {
            const Range r = partition(numOptions, ranks, source);
            const size_t n = r.start < wanted ? std::min(r.count, wanted - r.start) : 0;
            for (size_t offset = 0; offset < n;) {
                const int count = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
                MPI_Recv(results.data() + r.start + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += count;
            }
        }
    } else {
        for (size_t offset = 0; offset < localWanted;) {
            const int count = static_cast<int>(std::min(localWanted - offset, static_cast<size_t>(INT_MAX)));
            MPI_Send(localResults.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += count;
        }
    }
    int valid = 1;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static) num_threads(static_cast<int>(devices.count))
    for (size_t d = 0; d < devices.count; ++d) {
        cudaSetDevice(work[d].device);
        if (work[d].buffer) cudaFree(work[d].buffer);
    }
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
