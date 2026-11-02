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

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

enum OptionType { CALL = 0, PUT = 1 };

struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

// Keep these inputs identical to the original benchmark. Each GPU generates its
// own options from the global index, avoiding both input storage and PCIe traffic.
#define TEST_OPTIONS \
    {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3}, \
    {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3}, \
    {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3}, \
    {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3}, \
    {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3}, \
    {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3}, \
    {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}

constexpr std::array<OptionInput, 7> testOptions = {{ TEST_OPTIONS }};
__device__ __constant__ OptionInput deviceOptions[7] = { TEST_OPTIONS };
#undef TEST_OPTIONS

__global__ void priceOptions(double* results, size_t first, size_t count) {
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = first + j;
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
        const double discount = exp(-option.r * T);
        const double dividend = exp(-option.q * T);
        if (option.type == CALL) {
            const double Nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
            const double Nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
            results[j] = S * dividend * Nd1 - K * discount * Nd2;
        } else {
            const double Nminusd1 = 0.5 * (1.0 + erf(-d1 * M_SQRT1_2));
            const double Nminusd2 = 0.5 * (1.0 + erf(-d2 * M_SQRT1_2));
            results[j] = K * discount * Nminusd2 - S * dividend * Nminusd1;
        }
    }
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        // This may be called by an OpenMP worker; MPI calls are restricted to
        // the main thread by MPI_THREAD_FUNNELED.
        std::abort();
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const size_t numChecks = std::min(size_t(10), results.size());
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % testOptions.size()].value;
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Divide a range evenly, including when there are more workers than elements.
size_t rangeStart(size_t n, int workers, int worker) {
    return (n / workers) * worker + std::min(n % workers, static_cast<size_t>(worker));
}

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
    int exitCode = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
            break;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            exitCode = 1;
            break;
        }
    }
    if (help || exitCode) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return exitCode;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // If there are fewer ranks than visible GPUs, use every GPU. Otherwise,
    // assign one GPU per rank, sharing only when ranks outnumber devices.
    std::vector<int> devices;
    if (localRanks < deviceCount) {
        for (int d = localRank; d < deviceCount; d += localRanks) devices.push_back(d);
    } else {
        devices.push_back(localRank % deviceCount);
    }
    MPI_Comm_free(&localComm);

    const size_t first = rangeStart(numOptions, ranks, rank);
    const size_t localCount = rangeStart(numOptions, ranks, rank + 1) - first;
    std::vector<double> localResults(localCount);
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    const int workers = static_cast<int>(devices.size());
    constexpr size_t chunkSize = 1u << 22;
    const size_t capacity = std::min(chunkSize, localCount);
    std::vector<double*> deviceBuffers(workers, nullptr);
    omp_set_dynamic(0);
    // Context creation and allocation correspond to the original benchmark's
    // setup phase, before the timed pricing loop.
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        if (capacity != 0) {
            checkCuda(cudaSetDevice(devices[worker]), "cudaSetDevice");
            checkCuda(cudaMalloc(&deviceBuffers[worker], capacity * sizeof(double)), "cudaMalloc");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const size_t begin = rangeStart(localCount, workers, worker);
        const size_t end = rangeStart(localCount, workers, worker + 1);
        if (end > begin) {
            checkCuda(cudaSetDevice(devices[worker]), "cudaSetDevice");
            for (size_t offset = begin; offset < end;) {
                const size_t count = std::min(capacity, end - offset);
                constexpr int threads = 256;
                const int blocks = static_cast<int>(std::min((count + threads - 1) / threads, size_t(65535)));
                priceOptions<<<blocks, threads>>>(deviceBuffers[worker], first + offset, count);
                checkCuda(cudaGetLastError(), "priceOptions launch");
                checkCuda(cudaMemcpy(localResults.data() + offset, deviceBuffers[worker], count * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
                offset += count;
            }
        }
    }
    const double elapsed = MPI_Wtime() - start;
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        if (deviceBuffers[worker]) {
            checkCuda(cudaSetDevice(devices[worker]), "cudaSetDevice");
            checkCuda(cudaFree(deviceBuffers[worker]), "cudaFree");
        }
    }
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }

    // Gather the full ordered vector only for -r; validation needs its prefix.
    const size_t needed = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    std::vector<double> results;
    if (rank == 0) results.resize(needed);
    for (int source = 0; source < ranks && rangeStart(numOptions, ranks, source) < needed; ++source) {
        const size_t sourceFirst = rangeStart(numOptions, ranks, source);
        const size_t sourceEnd = std::min(needed, rangeStart(numOptions, ranks, source + 1));
        for (size_t offset = sourceFirst; offset < sourceEnd;) {
            const int count = static_cast<int>(std::min(sourceEnd - offset, static_cast<size_t>(INT_MAX)));
            if (source == 0 && rank == 0) {
                std::copy_n(localResults.data() + offset, count, results.data() + offset);
            } else if (rank == source) {
                MPI_Send(localResults.data() + (offset - sourceFirst), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            } else if (rank == 0) {
                MPI_Recv(results.data() + offset, count, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            offset += count;
        }
    }
    if (rank == 0) {
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
