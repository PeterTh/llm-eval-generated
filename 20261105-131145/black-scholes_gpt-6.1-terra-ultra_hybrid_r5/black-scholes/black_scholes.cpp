#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
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

// The generated inputs are the same seven cases and scaling as the original.
__constant__ OptionInput testOptions[7] = {
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}
};

__device__ inline double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__global__ void priceOptions(double* output, size_t offset, size_t count) {
    for (size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += size_t(blockDim.x) * gridDim.x) {
        const size_t i = offset + j;
        const OptionInput option = testOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        const double S = option.spot * factor;
        const double K = option.strike * factor;
        const double T = option.t;
        const double sigma = option.vol;
        if (T <= 0.0 || sigma <= 0.0) {
            output[j] = 0.0;
            continue;
        }
        const double d1 = (log(S / K) + (option.r - option.q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        const double d2 = d1 - sigma * sqrt(T);
        const double discount = exp(-option.r * T);
        if (option.type == CALL) {
            output[j] = S * exp(-option.q * T) * cumulativeNormal(d1) - K * discount * cumulativeNormal(d2);
        } else {
            output[j] = K * discount * cumulativeNormal(-d2) - S * exp(-option.q * T) * cumulativeNormal(-d1);
        }
    }
}

static void partition(size_t n, int parts, int part, size_t& begin, size_t& count) {
    const size_t base = n / parts, remainder = n % parts;
    count = base + (size_t(part) < remainder);
    begin = base * part + std::min(size_t(part), remainder);
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static bool validateResults(const std::vector<double>& results) {
    constexpr double expected[] = {5.0975, 0.0205, 1.8734, 9.9413, 9.9210, 1.8734, 0.0408};
    bool passed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size() && i < 10; ++i) {
        const double computed = results[i], value = expected[i % 7];
        const double relError = std::fabs(computed - value) / (std::fabs(value) + 1e-10);
        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n", i, computed, value, relError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            printf("Validation failed at option %zu: invalid value %.4f\n", i, computed);
            passed = false;
        }
    }
    return passed;
}

struct DeviceWork {
    int device;
    size_t offset, count;
    double* output = nullptr;
};

int main(int argc, char** argv) {
    int provided = 0;
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
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = atoll(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); parseStatus = 1; break; }
        else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            parseStatus = 2; break;
        }
    }
    if (parseStatus) { MPI_Finalize(); return parseStatus == 1 ? 0 : 1; }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no usable CUDA device: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    if (deviceCount >= localRanks) {
        for (int d = localRank; d < deviceCount; d += localRanks) devices.push_back(d);
    } else devices.push_back(localRank % deviceCount);

    size_t rankOffset, rankCount;
    partition(numOptions, ranks, rank, rankOffset, rankCount);
    std::vector<DeviceWork> work(devices.size());
    std::atomic<int> failed{0};
    auto checkCuda = [&](cudaError_t err, const char* operation) {
        if (err != cudaSuccess) {
            fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(err));
            failed.store(1, std::memory_order_relaxed);
            return false;
        }
        return true;
    };
    // Each OpenMP worker controls one assigned GPU. Setup is excluded from timing.
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int d = 0; d < static_cast<int>(devices.size()); ++d) {
        size_t offset, count;
        partition(rankCount, static_cast<int>(devices.size()), d, offset, count);
        work[d] = {devices[d], rankOffset + offset, count, nullptr};
        if (!checkCuda(cudaSetDevice(devices[d]), "cudaSetDevice")) continue;
        if (count > std::numeric_limits<size_t>::max() / sizeof(double)) {
            fprintf(stderr, "Rank %d: result allocation size overflows\n", rank);
            failed.store(1, std::memory_order_relaxed);
            continue;
        }
        if (count && !checkCuda(cudaMalloc(&work[d].output, count * sizeof(double)), "cudaMalloc")) continue;
        if (count) {
            // Load the kernel and initialize the CUDA context before measurement.
            priceOptions<<<1, 1>>>(work[d].output, work[d].offset, 1);
            if (!checkCuda(cudaGetLastError(), "warm-up launch")) continue;
            checkCuda(cudaDeviceSynchronize(), "warm-up execution");
        }

    }
    if (failed.load()) MPI_Abort(MPI_COMM_WORLD, 1);
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\nPricing options...\n",
               numOptions, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int d = 0; d < static_cast<int>(devices.size()); ++d) {
        const auto& w = work[d];
        if (!checkCuda(cudaSetDevice(w.device), "cudaSetDevice")) continue;
        if (w.count) {
            const int blocks = static_cast<int>(std::min<size_t>((w.count - 1) / 256 + 1, 65535));
            priceOptions<<<blocks, 256>>>(w.output, w.offset, w.count);
            if (!checkCuda(cudaGetLastError(), "kernel launch")) continue;
            checkCuda(cudaDeviceSynchronize(), "kernel execution");
        }
    }
    const double elapsed = MPI_Wtime() - start;
    if (failed.load()) MPI_Abort(MPI_COMM_WORLD, 1);
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", numOptions / duration);
    }

    // Only retrieve prices that the original output or validation will consume.
    const size_t wanted = printResults ? numOptions : (validate ? std::min<size_t>(10, numOptions) : 0);
    const size_t localWanted = rankOffset < wanted ? std::min(rankCount, wanted - rankOffset) : 0;
    std::vector<double> localResults(localWanted);
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int d = 0; d < static_cast<int>(devices.size()); ++d) {
        const auto& w = work[d];
        if (!checkCuda(cudaSetDevice(w.device), "cudaSetDevice")) continue;
        const size_t count = w.offset < wanted ? std::min(w.count, wanted - w.offset) : 0;
        if (count) checkCuda(cudaMemcpy(localResults.data() + (w.offset - rankOffset), w.output,
                                        count * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
        if (w.output) checkCuda(cudaFree(w.output), "cudaFree");
    }
    if (failed.load()) MPI_Abort(MPI_COMM_WORLD, 1);
    std::vector<double> results;
    if (rank == 0) {
        results.resize(wanted);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int source = 1; source < ranks; ++source) {
            size_t offset, count;
            partition(numOptions, ranks, source, offset, count);
            count = offset < wanted ? std::min(count, wanted - offset) : 0;
            while (count) {
                int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
                MPI_Recv(results.data() + offset, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk; count -= chunk;
            }
        }
    } else {
        size_t offset = 0;
        while (offset < localWanted) {
            int chunk = static_cast<int>(std::min<size_t>(localWanted - offset, INT_MAX));
            MPI_Send(localResults.data() + offset, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            offset += chunk;
        }
    }
    int result = 0;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            result = validateResults(results) ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return result;
}
