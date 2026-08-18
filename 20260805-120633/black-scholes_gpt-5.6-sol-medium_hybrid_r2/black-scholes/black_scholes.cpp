#include <algorithm>
#include <array>
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

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
             {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
             {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
             {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
             {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}}};
}

static OptionInput optionAt(const size_t i) noexcept {
    constexpr auto tests = getTestOptions();
    OptionInput option = tests[i % tests.size()];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(tests.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

struct HostData {
    int* type = nullptr;
    double *strike = nullptr, *spot = nullptr, *q = nullptr, *r = nullptr;
    double *t = nullptr, *vol = nullptr, *result = nullptr;
};

struct DeviceData {
    int* type = nullptr;
    double *strike = nullptr, *spot = nullptr, *q = nullptr, *r = nullptr;
    double *t = nullptr, *vol = nullptr, *result = nullptr;
};

static void cudaCheck(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void allocateHost(HostData& h, size_t n, int rank) {
    if (n == 0) return;
#define HOST_ALLOC(member, type_) \
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&h.member), n * sizeof(type_)), \
              "cudaMallocHost(" #member ")", rank)
    HOST_ALLOC(type, int);
    HOST_ALLOC(strike, double); HOST_ALLOC(spot, double); HOST_ALLOC(q, double);
    HOST_ALLOC(r, double); HOST_ALLOC(t, double); HOST_ALLOC(vol, double);
    HOST_ALLOC(result, double);
#undef HOST_ALLOC
}

static void allocateDevice(DeviceData& d, size_t n, int rank) {
    if (n == 0) return;
#define DEVICE_ALLOC(member, type_) \
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&d.member), n * sizeof(type_)), \
              "cudaMalloc(" #member ")", rank)
    DEVICE_ALLOC(type, int);
    DEVICE_ALLOC(strike, double); DEVICE_ALLOC(spot, double); DEVICE_ALLOC(q, double);
    DEVICE_ALLOC(r, double); DEVICE_ALLOC(t, double); DEVICE_ALLOC(vol, double);
    DEVICE_ALLOC(result, double);
#undef DEVICE_ALLOC
}

static void freeHost(HostData& h) {
    cudaFreeHost(h.type); cudaFreeHost(h.strike); cudaFreeHost(h.spot); cudaFreeHost(h.q);
    cudaFreeHost(h.r); cudaFreeHost(h.t); cudaFreeHost(h.vol); cudaFreeHost(h.result);
}

static void freeDevice(DeviceData& d) {
    cudaFree(d.type); cudaFree(d.strike); cudaFree(d.spot); cudaFree(d.q);
    cudaFree(d.r); cudaFree(d.t); cudaFree(d.vol); cudaFree(d.result);
}

__device__ __forceinline__ double cumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.707106781186547524400844362104849039));
}

__global__ void blackScholesKernel(const int* __restrict__ type,
                                   const double* __restrict__ strike,
                                   const double* __restrict__ spot,
                                   const double* __restrict__ q,
                                   const double* __restrict__ r,
                                   const double* __restrict__ t,
                                   const double* __restrict__ vol,
                                   double* __restrict__ result, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double T = t[i];
    const double sigma = vol[i];
    if (T <= 0.0 || sigma <= 0.0) {
        result[i] = 0.0;
        return;
    }
    const double S = spot[i];
    const double K = strike[i];
    const double rootT = sqrt(T);
    const double d1 = (log(S / K) + (r[i] - q[i] + 0.5 * sigma * sigma) * T) /
                      (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double discountedK = K * exp(-r[i] * T);
    const double discountedS = S * exp(-q[i] * T);
    result[i] = type[i] == CALL
                    ? discountedS * cumulativeNormal(d1) - discountedK * cumulativeNormal(d2)
                    : discountedK * cumulativeNormal(-d2) - discountedS * cumulativeNormal(-d1);
}

static bool validateResults(const std::vector<double>& results) {
    const size_t checks = std::min<size_t>(10, results.size());
    int invalid = 0;
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < checks; ++i) {
        const OptionInput option = optionAt(i);
        const double computed = results[i];
        const double error = std::fabs(computed - option.value);
        const double relative = error / (std::fabs(option.value) + 1.0e-10);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, option.value, relative);
        invalid += computed < 0.0 || computed > 1000.0 || !std::isfinite(computed);
    }
    return invalid == 0;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t numOptions = 10000;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '\0' || *end != '\0' || parsed > std::numeric_limits<size_t>::max())
                parseStatus = 1;
            else
                numOptions = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else parseStatus = 1;
    }
    if (parseStatus != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "-n exceeds this MPI implementation's gather limit\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices are available\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    const size_t first = (numOptions * static_cast<size_t>(rank)) / ranks;
    const size_t last = (numOptions * static_cast<size_t>(rank + 1)) / ranks;
    const size_t localCount = last - first;
    HostData host;
    DeviceData device;
    allocateHost(host, localCount, rank);
    allocateDevice(device, localCount, rank);

#pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(localCount); ++j) {
        const OptionInput option = optionAt(first + static_cast<size_t>(j));
        host.type[j] = option.type; host.strike[j] = option.strike; host.spot[j] = option.spot;
        host.q[j] = option.q; host.r[j] = option.r; host.t[j] = option.t; host.vol[j] = option.vol;
    }

    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate", rank);
#define COPY_INPUT(member, type_) \
    if (localCount) cudaCheck(cudaMemcpyAsync(device.member, host.member, localCount * sizeof(type_), \
                                               cudaMemcpyHostToDevice, stream), \
                              "copy " #member " to device", rank)
    COPY_INPUT(type, int); COPY_INPUT(strike, double); COPY_INPUT(spot, double); COPY_INPUT(q, double);
    COPY_INPUT(r, double); COPY_INPUT(t, double); COPY_INPUT(vol, double);
#undef COPY_INPUT
    cudaCheck(cudaStreamSynchronize(stream), "input transfer", rank);

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n", numOptions);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
                    ranks, omp_get_max_threads(), deviceCount);
        std::printf("Validation: %s\nPricing options...\n", validate ? "enabled" : "disabled");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (localCount) {
        constexpr unsigned threads = 256;
        const unsigned long long blocks64 = (localCount + threads - 1) / threads;
        if (blocks64 > std::numeric_limits<unsigned>::max()) {
            std::fprintf(stderr, "Rank %d: local problem is too large for one kernel launch\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        blackScholesKernel<<<static_cast<unsigned>(blocks64), threads, 0, stream>>>(
            device.type, device.strike, device.spot, device.q, device.r, device.t, device.vol,
            device.result, localCount);
        cudaCheck(cudaGetLastError(), "blackScholesKernel launch", rank);
    }
    cudaCheck(cudaStreamSynchronize(stream), "CUDA computation", rank);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // The original benchmark stops timing before observing results.  Gather only
    // values that are actually consumed, avoiding a rank-0 bottleneck by default.
    const size_t neededResults = printResults ? numOptions
                                 : (validate ? std::min<size_t>(10, numOptions) : 0);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int p = 0; p < ranks; ++p) {
        const size_t begin = (numOptions * static_cast<size_t>(p)) / ranks;
        const size_t end = (numOptions * static_cast<size_t>(p + 1)) / ranks;
        counts[p] = static_cast<int>(begin < neededResults
                                         ? std::min(end, neededResults) - begin
                                         : 0);
        displacements[p] = static_cast<int>(begin);
    }
    const int sendCount = counts[rank];
    if (sendCount) {
        cudaCheck(cudaMemcpyAsync(host.result, device.result,
                                  static_cast<size_t>(sendCount) * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream), "copy results to host", rank);
        cudaCheck(cudaStreamSynchronize(stream), "result transfer", rank);
    }
    std::vector<double> results;
    if (rank == 0) results.resize(neededResults);
    MPI_Gatherv(host.result, sendCount, MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Options per second: %.0f\n", elapsed > 0.0 ? numOptions / elapsed : 0.0);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            std::printf("Validating results...\n");
            const bool valid = validateResults(results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    cudaStreamDestroy(stream);
    freeDevice(device);
    freeHost(host);
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
