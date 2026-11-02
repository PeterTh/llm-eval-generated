#include <algorithm>
#include <array>
#include <atomic>
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

// Each rank and GPU reconstructs exactly the same options as the serial generator.
__constant__ OptionInput deviceOptions[7];

__global__ void priceOptions(double* results, size_t begin, size_t count) {
    for (size_t j = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         j < count; j += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t i = begin + j;
        const OptionInput base = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        const double S = base.spot * factor;
        const double K = base.strike * factor;
        const double rootT = sqrt(base.t);
        const double d1 = (log(S / K) + (base.r - base.q + 0.5 * base.vol * base.vol) * base.t)
                          / (base.vol * rootT);
        const double d2 = d1 - base.vol * rootT;
        const double discount = exp(-base.r * base.t);
        const double dividend = exp(-base.q * base.t);
        double price = 0.0;
        if (base.t > 0.0 && base.vol > 0.0) {
            if (base.type == CALL) {
                const double nd1 = 0.5 * (1.0 + erf(d1 * M_SQRT1_2));
                const double nd2 = 0.5 * (1.0 + erf(d2 * M_SQRT1_2));
                price = S * dividend * nd1 - K * discount * nd2;
            } else {
                const double nmd1 = 0.5 * (1.0 + erf(-d1 * M_SQRT1_2));
                const double nmd2 = 0.5 * (1.0 + erf(-d2 * M_SQRT1_2));
                price = K * discount * nmd2 - S * dividend * nmd1;
            }
        }
        results[j] = price;
    }
}

// Balanced contiguous partitions preserve the global option ordering.
struct Slice { size_t begin, count; };
Slice partition(size_t n, int parts, int part) {
    const size_t width = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    return {width * static_cast<size_t>(part) + std::min(static_cast<size_t>(part), extra),
            width + (static_cast<size_t>(part) < extra ? 1 : 0)};
}

bool validateResults(const std::vector<double>& results) {
    constexpr auto testOptions = getTestOptions();
    bool allPassed = true;
    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < results.size(); ++i) {
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

struct GPUWork {
    int device;
    Slice slice;
    double* buffer = nullptr;
    size_t capacity = 0;
};

bool cudaCheck(cudaError_t status, const char* operation, int device) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA device %d: %s: %s\n", device, operation, cudaGetErrorString(status));
    return false;
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
    int argumentStatus = -1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numOptions = atoll(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { argumentStatus = 0; break; }
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            argumentStatus = 1;
            break;
        }
    }
    if (argumentStatus >= 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentStatus;
    }
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, localRanks;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localRanks);
    int deviceCount = 0;
    if (!cudaCheck(cudaGetDeviceCount(&deviceCount), "enumerating devices", -1) || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    if (localRanks < deviceCount) {
        for (int d = localRank; d < deviceCount; d += localRanks) devices.push_back(d);
    } else {
        devices.push_back(localRank % deviceCount);
    }
    MPI_Comm_free(&localComm);

    const Slice rankSlice = partition(numOptions, ranks, rank);
    const size_t collectCount = printResults ? numOptions : (validate ? std::min(numOptions, size_t(10)) : 0);
    const size_t localCount = rankSlice.begin < collectCount
        ? std::min(rankSlice.count, collectCount - rankSlice.begin) : 0;
    std::vector<double> localResults(localCount);
    std::vector<GPUWork> work(devices.size());
    constexpr size_t chunkSize = 4 * 1024 * 1024;
    constexpr auto testOptions = getTestOptions();
    std::atomic<int> failed{0};
    omp_set_dynamic(0);
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int k = 0; k < static_cast<int>(devices.size()); ++k) {
        GPUWork& w = work[k];
        w.device = devices[k];
        Slice part = partition(rankSlice.count, static_cast<int>(devices.size()), k);
        w.slice = {rankSlice.begin + part.begin, part.count};
        w.capacity = std::min(chunkSize, w.slice.count);
        if (!cudaCheck(cudaSetDevice(w.device), "selecting device", w.device) ||
            !cudaCheck(cudaMemcpyToSymbol(deviceOptions, testOptions.data(), sizeof(OptionInput) * testOptions.size()),
                       "copying option parameters", w.device) ||
            (w.capacity && !cudaCheck(cudaMalloc(&w.buffer, w.capacity * sizeof(double)), "allocating results", w.device))) {
            failed.store(1);
        }
    }
    int localFailed = failed.load(), anyFailed = 0;
    MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyFailed) MPI_Abort(MPI_COMM_WORLD, 1);

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int k = 0; k < static_cast<int>(devices.size()); ++k) {
        GPUWork& w = work[k];
        if (!cudaCheck(cudaSetDevice(w.device), "selecting device", w.device)) { failed.store(1); continue; }
        for (size_t offset = 0; offset < w.slice.count;) {
            const size_t count = std::min(w.capacity, w.slice.count - offset);
            const size_t begin = w.slice.begin + offset;
            const unsigned blocks = static_cast<unsigned>(std::min((count + 255) / 256, size_t(65535)));
            priceOptions<<<blocks, 256>>>(w.buffer, begin, count);
            if (!cudaCheck(cudaGetLastError(), "launching pricing kernel", w.device)) { failed.store(1); break; }
            if (begin < collectCount) {
                const size_t toCopy = std::min(count, collectCount - begin);
                if (!cudaCheck(cudaMemcpy(localResults.data() + (begin - rankSlice.begin), w.buffer,
                                          toCopy * sizeof(double), cudaMemcpyDeviceToHost),
                               "copying results", w.device)) { failed.store(1); break; }
            }
            offset += count;
        }
        if (!cudaCheck(cudaDeviceSynchronize(), "synchronizing pricing kernel", w.device)) failed.store(1);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    localFailed = failed.load();
    MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyFailed) MPI_Abort(MPI_COMM_WORLD, 1);
    #pragma omp parallel for num_threads(static_cast<int>(devices.size())) schedule(static)
    for (int k = 0; k < static_cast<int>(devices.size()); ++k) {
        cudaSetDevice(work[k].device);
        if (work[k].buffer) cudaFree(work[k].buffer);
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        printf("Options per second: %.0f\n", numOptions / maxElapsed);
    }

    // Point-to-point chunks avoid MPI's int count limit for large result sets.
    std::vector<double> results;
    if (rank == 0) {
        results.resize(collectCount);
        std::copy(localResults.begin(), localResults.end(), results.begin());
        for (int source = 1; source < ranks; ++source) {
            const Slice s = partition(numOptions, ranks, source);
            const size_t count = s.begin < collectCount ? std::min(s.count, collectCount - s.begin) : 0;
            for (size_t off = 0; off < count;) {
                const int chunk = static_cast<int>(std::min(count - off, static_cast<size_t>(INT_MAX)));
                MPI_Recv(results.data() + s.begin + off, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                off += static_cast<size_t>(chunk);
            }
        }
    } else {
        for (size_t off = 0; off < localCount;) {
            const int chunk = static_cast<int>(std::min(localCount - off, static_cast<size_t>(INT_MAX)));
            MPI_Send(localResults.data() + off, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            off += static_cast<size_t>(chunk);
        }
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            results.resize(std::min(numOptions, size_t(10)));
            const bool valid = validateResults(results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
