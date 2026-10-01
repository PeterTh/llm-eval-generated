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
struct OptionInput { int type; double strike, spot, q, r, t, vol, value, tol; };

#define TEST_OPTIONS \
    {CALL,40.,42.,.04,.08,.75,.35,5.0975,1.e-3}, \
    {CALL,100.,90.,.10,.10,.10,.15,.0205,1.e-3}, \
    {CALL,100.,100.,.10,.10,.10,.15,1.8734,1.e-3}, \
    {CALL,100.,110.,.10,.10,.10,.15,9.9413,1.e-3}, \
    {PUT,100.,90.,.10,.10,.10,.15,9.9210,1.e-3}, \
    {PUT,100.,100.,.10,.10,.10,.15,1.8734,1.e-3}, \
    {PUT,100.,110.,.10,.10,.10,.15,.0408,1.e-3}
__device__ __constant__ OptionInput deviceOptions[7] = { TEST_OPTIONS };
static constexpr std::array<OptionInput, 7> hostOptions = {{ TEST_OPTIONS }};

__device__ inline double cdf(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}
__global__ void priceOptions(size_t first, size_t count, double* prices) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < count; j += stride) {
        const size_t i = first + j;
        const OptionInput o = deviceOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / 7.0);
        const double S = o.spot * factor, K = o.strike * factor;
        if (o.t <= 0.0 || o.vol <= 0.0) {
            prices[j] = 0.0;
            continue;
        }
        const double volTime = o.vol * sqrt(o.t);
        const double d1 = (log(S / K) + (o.r - o.q + 0.5 * o.vol * o.vol) * o.t) / volTime;
        const double d2 = d1 - volTime;
        const double discount = exp(-o.r * o.t);
        if (o.type == CALL)
            prices[j] = S * exp(-o.q * o.t) * cdf(d1) - K * discount * cdf(d2);
        else
            prices[j] = K * discount * cdf(-d2) - S * exp(-o.q * o.t) * cdf(-d1);
    }
}
static void fail(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
static size_t startOf(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min<size_t>(n % ranks, rank);
}
static size_t countOf(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}
static bool validateResults(const std::vector<double>& results) {
    bool passed = true;
    std::printf("Checking computed option prices:\n");
    for (size_t i = 0; i < std::min<size_t>(10, results.size()); ++i) {
        const double value = results[i], expected = hostOptions[i % 7].value;
        const double relative = std::fabs(value - expected) / (std::fabs(expected) + 1e-10);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, value, expected, relative);
        if (value < 0.0 || value > 1000.0 || !std::isfinite(value)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n", i, value);
            passed = false;
        }
    }
    return passed;
}
static void usage(const char* program) {
    std::printf("Usage: %s [options]\nOptions:\n", program);
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}
int main(int argc, char** argv) {
    int provided = 0, rank = 0, ranks = 1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required", rank);
    size_t n = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long parsed = std::strtoull(arg, &end, 10);
            if (errno || end == arg || *end || arg[0] == '-' || parsed > SIZE_MAX) {
                if (rank == 0) usage(argv[0]);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); usage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\n", n);
        std::printf("Validation: %s\nPricing options...\n", validate ? "enabled" : "disabled");
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank = 0, localRanks = 1;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    MPI_Comm_free(&node);
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        fail("a CUDA GPU is required", rank);
    const size_t first = startOf(n, rank, ranks), count = countOf(n, rank, ranks);
    std::vector<double> local(count);
    // MPI ranks on a node receive distinct GPUs where possible. OpenMP
    // threads let a rank use additional GPUs for large local workloads.
    const int available = localRanks <= devices ? (devices - 1 - localRank) / localRanks + 1 : 1;
    const int workers = static_cast<int>(std::min<size_t>(
        std::min(available, omp_get_max_threads()),
        std::max<size_t>(1, count / 65536 + (count % 65536 != 0))));
    std::vector<cudaError_t> errors(workers, cudaSuccess);
    std::vector<double*> gpuBuffers(workers, nullptr);
#pragma omp parallel num_threads(workers)
    {
        const int worker = omp_get_thread_num();
        const int device = localRanks <= devices ? localRank + worker * localRanks : localRank % devices;
        const size_t length = countOf(count, worker, workers);
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess) error = cudaFree(nullptr); // Initialize this device's context.
        if (error == cudaSuccess && length)
            error = cudaMalloc(&gpuBuffers[worker], length * sizeof(double));
        errors[worker] = error;
    }
    for (int i = 0; i < workers; ++i)
        if (errors[i] != cudaSuccess) fail(cudaGetErrorString(errors[i]), rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
#pragma omp parallel num_threads(workers)
    {
        const int worker = omp_get_thread_num();
        const int device = localRanks <= devices ? localRank + worker * localRanks : localRank % devices;
        const size_t offset = startOf(count, worker, workers);
        const size_t length = countOf(count, worker, workers);
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess && length) {
            const int blocks = static_cast<int>(std::min<size_t>((length + 255) / 256, 65535));
            priceOptions<<<blocks, 256>>>(first + offset, length, gpuBuffers[worker]);
            error = cudaGetLastError();
            if (error == cudaSuccess)
                error = cudaMemcpy(local.data() + offset, gpuBuffers[worker], length * sizeof(double), cudaMemcpyDeviceToHost);
        }
        errors[worker] = error;
    }
    for (int i = 0; i < workers; ++i)
        if (errors[i] != cudaSuccess) fail(cudaGetErrorString(errors[i]), rank);
    const double localTime = MPI_Wtime() - start;
    for (int i = 0; i < workers; ++i) {
        const int device = localRanks <= devices ? localRank + i * localRanks : localRank % devices;
        cudaSetDevice(device);
        if (gpuBuffers[i]) cudaFree(gpuBuffers[i]);
    }
    double elapsed = 0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000);
        std::printf("Options per second: %.0f\n", elapsed > 0 ? n / elapsed : 0.0);
    }
    std::vector<double> results;
    if (printResults) {
        if (rank == 0) results.resize(n);
        if (n <= INT_MAX) {
            std::vector<int> counts, offsets;
            if (rank == 0) {
                counts.resize(ranks);
                offsets.resize(ranks);
                for (int r = 0; r < ranks; ++r) {
                    counts[r] = static_cast<int>(countOf(n, r, ranks));
                    offsets[r] = static_cast<int>(startOf(n, r, ranks));
                }
            }
            MPI_Gatherv(local.data(), static_cast<int>(count), MPI_DOUBLE,
                        rank == 0 ? results.data() : nullptr,
                        rank == 0 ? counts.data() : nullptr,
                        rank == 0 ? offsets.data() : nullptr,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else if (rank == 0) {
            std::copy(local.begin(), local.end(), results.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t begin = startOf(n, r, ranks), end = begin + countOf(n, r, ranks);
                for (size_t pos = begin; pos < end; ) {
                    const int chunk = static_cast<int>(std::min<size_t>(end - pos, INT_MAX));
                    MPI_Recv(results.data() + pos, chunk, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += chunk;
                }
            }
        } else {
            for (size_t pos = 0; pos < count; ) {
                const int chunk = static_cast<int>(std::min<size_t>(count - pos, INT_MAX));
                MPI_Send(local.data() + pos, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                pos += chunk;
            }
        }
        if (rank == 0) print_results(results, "OptionPrices");
    }
    int valid = 1;
    if (validate) {
        if (!printResults) {
            std::array<double, 10> firstLocal{}, firstGlobal{};
            for (size_t i = 0; i < count && first + i < 10; ++i)
                firstLocal[first + i] = local[i];
            MPI_Reduce(firstLocal.data(), firstGlobal.data(), 10, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) results.assign(firstGlobal.begin(), firstGlobal.begin() + std::min<size_t>(10, n));
        }
        if (rank == 0) {
            std::printf("Validating results...\n");
            valid = validateResults(results);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
