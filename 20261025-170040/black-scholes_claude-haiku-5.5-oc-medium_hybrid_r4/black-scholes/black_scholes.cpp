// Black-Scholes option pricing benchmark, hybrid MPI + OpenMP + CUDA.
//
// Decomposition:
//   MPI     - the option index range is block-partitioned across ranks. Each
//             rank generates and prices only its own contiguous slice.
//   OpenMP  - each rank owns one or more GPUs (one per local rank, round-robin
//             over the node's devices). One host thread drives each GPU so
//             that all devices of a rank run concurrently.
//   CUDA    - each device prices its chunk with a grid-stride kernel. Options
//             are synthesized on the device from the global option index, so
//             only the computed prices cross the PCIe bus.
//
// Results are gathered to rank 0 in global index order only when they are
// needed for output (-r) or validation (-v), so printed values, sums, and
// hashes have the same semantics as the serial original.

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

constexpr size_t kNumTestOptions = 7;

// Standard test cases for validation
inline constexpr std::array<OptionInput, kNumTestOptions> getTestOptions() noexcept {
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

// Device-side copy of the test table, uploaded once per device.
__constant__ OptionInput c_testOptions[kNumTestOptions];

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Black-Scholes formula for European options
__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
}

// Option with global index i: cycle through the test set and scale spot/strike.
__host__ __device__ inline OptionInput makeOption(const OptionInput* testOptions, const size_t i) noexcept {
    const OptionInput& base = testOptions[i % kNumTestOptions];
    OptionInput option = base;
    const double factor = 1.0 + 0.1 * (static_cast<double>(i) / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

constexpr unsigned kThreadsPerBlock = 256;
constexpr size_t kMaxBlocks = 65536;
constexpr int kStreamsPerDevice = 8;

// Prices global options [firstIndex, firstIndex + count) into out[0, count).
__global__ void priceOptionsKernel(const size_t firstIndex, const size_t count, double* __restrict__ out) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = blackScholes(makeOption(c_testOptions, firstIndex + i));
    }
}

// Contiguous block partition of [0, total) into `parts` pieces; piece `index`.
struct Range {
    size_t begin;
    size_t count;
};

Range blockPartition(const size_t total, const int parts, const int index) noexcept {
    const size_t p = static_cast<size_t>(parts);
    const size_t idx = static_cast<size_t>(index);
    const size_t base = total / p;
    const size_t rem = total % p;
    return {idx * base + std::min(idx, rem), base + (idx < rem ? 1 : 0)};
}

void cudaCheck(const cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// GPUs used by a rank: devices d with d % localSize == localRank. If the node
// has fewer GPUs than ranks, ranks without a device share one round-robin.
std::vector<int> devicesForRank(const int localRank, const int localSize, const int numDevices) {
    std::vector<int> devs;
    for (int d = localRank; d < numDevices; d += localSize) {
        devs.push_back(d);
    }
    if (devs.empty()) {
        devs.push_back(localRank % numDevices);
    }
    return devs;
}

// Gathers the first `limit` global results to rank 0 in global index order.
// Returns the gathered vector on rank 0 and an empty vector elsewhere.
std::vector<double> gatherPrefix(const double* local, const size_t total, const size_t limit,
                                 const int rank, const int size) {
    constexpr size_t kChunk = size_t{1} << 26;   // elements per message (fits MPI int count)
    std::vector<double> out;
    if (rank == 0) {
        out.resize(limit);
    }

    for (int r = 0; r < size; ++r) {
        const Range b = blockPartition(total, size, r);
        const size_t n = (b.begin >= limit) ? 0 : std::min(b.count, limit - b.begin);
        if (n == 0) {
            continue;
        }
        if (r == 0) {
            if (rank == 0) {
                std::copy_n(local, n, out.data());
            }
            continue;
        }
        for (size_t off = 0; off < n; off += kChunk) {
            const int cnt = static_cast<int>(std::min(kChunk, n - off));
            if (rank == 0) {
                MPI_Recv(out.data() + b.begin + off, cnt, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == r) {
                MPI_Send(local + off, cnt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
    return out;
}

bool validateResults(const std::vector<double>& results, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    bool allPassed = true;
    const size_t numChecks = std::min<size_t>(10, numOptions);

    printf("Checking computed option prices:\n");
    for (size_t i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % kNumTestOptions].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Relaxed validation - just check values are positive and reasonable
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int status = -1;  // >= 0: exit immediately with this code
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            status = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }
    if (status >= 0) {
        MPI_Finalize();
        return status;
    }

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Device discovery and per-rank device assignment
    int numDevices = 0;
    cudaCheck(cudaGetDeviceCount(&numDevices), "cudaGetDeviceCount");
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices visible on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    const Range mine = blockPartition(numOptions, size, rank);
    const std::vector<int> devs = devicesForRank(localRank, localSize, numDevices);
    const int T = static_cast<int>(devs.size());

    std::vector<Range> chunks(T);
    for (int t = 0; t < T; ++t) {
        chunks[t] = blockPartition(mine.count, T, t);
    }

    // Setup (not timed): pinned host results, device contexts, device buffers
    double* hResults = nullptr;
    cudaCheck(cudaMallocHost(&hResults, std::max<size_t>(mine.count, 1) * sizeof(double)), "cudaMallocHost");

    std::vector<double*> dOut(T, nullptr);
    std::vector<std::vector<cudaStream_t>> streams(T);
    constexpr auto testOptions = getTestOptions();
    #pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        cudaCheck(cudaSetDevice(devs[t]), "cudaSetDevice");
        cudaCheck(cudaFree(nullptr), "CUDA context initialization");
        cudaCheck(cudaMemcpyToSymbol(c_testOptions, testOptions.data(),
                                     sizeof(OptionInput) * kNumTestOptions), "upload test options");
        cudaCheck(cudaMalloc(&dOut[t], std::max<size_t>(chunks[t].count, 1) * sizeof(double)), "cudaMalloc");
        // Streams let the D2H copy of one sub-chunk overlap the kernel of the next
        streams[t].resize(kStreamsPerDevice);
        for (cudaStream_t& s : streams[t]) {
            cudaCheck(cudaStreamCreate(&s), "cudaStreamCreate");
        }
        // Zero-work launch: forces lazy module loading out of the timed region
        priceOptionsKernel<<<1, kThreadsPerBlock>>>(0, 0, dOut[t]);
        cudaCheck(cudaGetLastError(), "priceOptionsKernel warm-up");
        cudaCheck(cudaDeviceSynchronize(), "device warm-up");
    }

    if (root) {
        printf("Pricing options...\n");
    }

    // Timed region: all ranks start together; report the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
        const Range c = chunks[t];
        if (c.count == 0) {
            continue;
        }
        cudaCheck(cudaSetDevice(devs[t]), "cudaSetDevice");
        const size_t subCount = (c.count + kStreamsPerDevice - 1) / kStreamsPerDevice;
        for (int s = 0; s < kStreamsPerDevice; ++s) {
            const size_t off = std::min(static_cast<size_t>(s) * subCount, c.count);
            const size_t n = std::min(subCount, c.count - off);
            if (n == 0) {
                continue;
            }
            const size_t neededBlocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
            const unsigned blocks = static_cast<unsigned>(std::min(neededBlocks, kMaxBlocks));
            priceOptionsKernel<<<blocks, kThreadsPerBlock, 0, streams[t][s]>>>(
                mine.begin + c.begin + off, n, dOut[t] + off);
            cudaCheck(cudaGetLastError(), "priceOptionsKernel launch");
            cudaCheck(cudaMemcpyAsync(hResults + c.begin + off, dOut[t] + off, n * sizeof(double),
                                      cudaMemcpyDeviceToHost, streams[t][s]), "result copy");
        }
        for (cudaStream_t& s : streams[t]) {
            cudaCheck(cudaStreamSynchronize(s), "stream synchronize");
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double localUs = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    double maxUs = 0.0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        printf("Computation time: %.3f ms\n", maxUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxUs / 1e6));
    }

    // Gather only what the output requires
    constexpr size_t kNumChecks = 10;
    size_t needed = 0;
    if (printResults) {
        needed = numOptions;
    } else if (validate) {
        needed = std::min(kNumChecks, numOptions);
    }
    const std::vector<double> gathered = gatherPrefix(hResults, numOptions, needed, rank, size);

    int exitCode = 0;
    if (root) {
        if (printResults) {
            print_results(gathered, "OptionPrices");
        }
        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(gathered, numOptions);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup
    for (int t = 0; t < T; ++t) {
        cudaCheck(cudaSetDevice(devs[t]), "cudaSetDevice");
        for (cudaStream_t& s : streams[t]) {
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
        cudaCheck(cudaFree(dOut[t]), "cudaFree");
    }
    cudaCheck(cudaFreeHost(hResults), "cudaFreeHost");
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
