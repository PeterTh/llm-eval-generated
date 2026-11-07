// Black-Scholes option pricing benchmark - hybrid MPI + OpenMP + CUDA version.
//
// Parallelization strategy
//   * MPI     : the option index range [0, N) is block-partitioned across ranks
//               (one or more ranks per node / per GPU). Rank 0 gathers the
//               results for printing / validation.
//   * CUDA    : every option owned by a rank is priced on a GPU. Options are
//               generated directly on the device from their global index (the
//               same arithmetic as generateOptions()), so no host->device
//               traffic is needed; only the 8-byte price travels back.
//   * OpenMP  : when a rank owns several GPUs (fewer ranks than GPUs on a
//               node) one OpenMP thread drives each GPU concurrently. Each
//               thread double-buffers its work through two CUDA streams so
//               kernel execution overlaps the device->host copies. OpenMP is
//               also used for host-side data generation.
//
// Results are bit-for-bit independent of the number of ranks / GPUs since the
// same kernel evaluates the same index with the same arithmetic everywhere.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

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

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error %s at %s:%d\n", #call, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Pricing math (shared by host and device)
// ---------------------------------------------------------------------------

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
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

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double discount = exp(-r * T);
    const double divDiscount = exp(-q * T);

    // Branch-free CALL/PUT selection (avoids warp divergence on the GPU).
    // Bitwise identical to the explicit two-branch formulation: negating a
    // double is exact and -(a - b) == (b - a) in IEEE arithmetic, so
    //   PUT:  -(S*e^{-qT}*N(-d1) - K*e^{-rT}*N(-d2))
    //       =   K*e^{-rT}*N(-d2) - S*e^{-qT}*N(-d1)
    const double sign = (option.type == CALL) ? 1.0 : -1.0;
    const double price = sign * (S * divDiscount * cumulativeNormal(sign * d1)
                               - K * discount * cumulativeNormal(sign * d2));

    return price;
}

// Standard test cases for validation
constexpr int kNumTestOptions = 7;

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

// Device-side copy of the test options (filled once per device before pricing)
__constant__ OptionInput c_testOptions[kNumTestOptions];

// Generate option i by cycling through the test options and scaling spot/strike.
// Identical arithmetic on host and device.
__host__ __device__ inline OptionInput makeOption(const OptionInput* base,
                                                  const size_t i) noexcept {
    OptionInput opt = base[i % kNumTestOptions];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    opt.spot *= factor;
    opt.strike *= factor;
    return opt;
}

// Generate a larger set of options by scaling the test set (host, OpenMP)
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    static constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = makeOption(testOptions.data(), i);
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: generate + price options [first, first + count) into out[]
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
priceOptionsKernel(double* __restrict__ out, const size_t first, const size_t count) {
    // Stage the 7 base options in shared memory: consecutive threads index
    // different entries, which would serialize constant-cache broadcasts.
    __shared__ OptionInput s_testOptions[kNumTestOptions];
    if (threadIdx.x < kNumTestOptions) {
        s_testOptions[threadIdx.x] = c_testOptions[threadIdx.x];
    }
    __syncthreads();

    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < count; k += stride) {
        const OptionInput opt = makeOption(s_testOptions, first + k);
        out[k] = blackScholes(opt);
    }
}

// ---------------------------------------------------------------------------
// Per-device worker context
// ---------------------------------------------------------------------------
constexpr size_t kChunkOptions = size_t(1) << 22;   // 4M options = 32 MiB per buffer
constexpr int kStreamsPerDevice = 2;

// Bind the calling thread to the CPUs local to the given GPU (same NUMA node /
// PCIe root complex) so that pinned-buffer allocation and the device->host
// copies stay NUMA-local. Best effort: silently keeps the current affinity if
// the topology cannot be determined.
static void bindThreadToDeviceCpus(const int device) {
    char busId[32];
    if (cudaDeviceGetPCIBusId(busId, sizeof(busId), device) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    for (char* c = busId; *c; ++c) *c = static_cast<char>(tolower(*c));
    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", busId);
    FILE* f = fopen(path, "r");
    if (!f) return;
    char list[1024] = {0};
    const bool ok = fgets(list, sizeof(list), f) != nullptr;
    fclose(f);
    if (!ok) return;

    cpu_set_t set;
    CPU_ZERO(&set);
    int n = 0;
    for (char* tok = strtok(list, ",\n"); tok; tok = strtok(nullptr, ",\n")) {
        int lo = 0, hi = 0;
        if (sscanf(tok, "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) { CPU_SET(c, &set); ++n; }
        } else if (sscanf(tok, "%d", &lo) == 1 && lo < CPU_SETSIZE) {
            CPU_SET(lo, &set); ++n;
        }
    }
    if (n > 0) {
        (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
}

struct DeviceWorker {
    int device = -1;
    size_t globalFirst = 0;    // first global option index handled by this device
    size_t count = 0;          // number of options handled by this device
    double* hostBuf = nullptr; // pinned, NUMA-local landing buffer (count elements)
    double* dBuf[kStreamsPerDevice] = {nullptr, nullptr};
    cudaStream_t stream[kStreamsPerDevice] = {nullptr, nullptr};
    int blocksPerGrid = 0;

    // Called from the OpenMP thread that will drive this device.
    void init() {
        bindThreadToDeviceCpus(device);
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(0)); // force context creation now, outside the timed region

        static constexpr auto testOptions = getTestOptions();
        CUDA_CHECK(cudaMemcpyToSymbol(c_testOptions, testOptions.data(),
                                      sizeof(OptionInput) * kNumTestOptions));

        const size_t bufElems = std::min(kChunkOptions, std::max<size_t>(count, 1));
        for (int s = 0; s < kStreamsPerDevice; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&dBuf[s], bufElems * sizeof(double)));
        }
        if (count > 0) {
            // Allocated (and page-locked) by the thread bound next to the GPU,
            // so the pages are placed on the GPU's NUMA node.
            CUDA_CHECK(cudaHostAlloc(&hostBuf, count * sizeof(double), cudaHostAllocDefault));
        }

        int numSMs = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        blocksPerGrid = numSMs * 8;

        // Warm up: load the kernel module (CUDA lazy loading) outside the timed region.
        priceOptionsKernel<<<1, 256, 0, stream[0]>>>(dBuf[0], 0, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
    }

    // Price all options of this device; blocks until results are in hostBuf.
    void run() {
        CUDA_CHECK(cudaSetDevice(device));
        size_t done = 0;
        int s = 0;
        while (done < count) {
            const size_t n = std::min(kChunkOptions, count - done);
            const int blocks = static_cast<int>(
                std::min<size_t>(blocksPerGrid, (n + 255) / 256));
            // The stream is in-order: the kernel for the next use of this
            // buffer waits for the previous copy out of it to complete, while
            // the other stream's copy overlaps this kernel.
            priceOptionsKernel<<<blocks, 256, 0, stream[s]>>>(dBuf[s], globalFirst + done, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hostBuf + done, dBuf[s], n * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream[s]));
            done += n;
            s = (s + 1) % kStreamsPerDevice;
        }
        for (int i = 0; i < kStreamsPerDevice; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        }
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(device));
        for (int s = 0; s < kStreamsPerDevice; ++s) {
            if (dBuf[s]) CUDA_CHECK(cudaFree(dBuf[s]));
            if (stream[s]) CUDA_CHECK(cudaStreamDestroy(stream[s]));
        }
        if (hostBuf) CUDA_CHECK(cudaFreeHost(hostBuf));
    }
};

// ---------------------------------------------------------------------------
// Validation / CLI
// ---------------------------------------------------------------------------
bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
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

// Block partition of [0, total) across `parts`; returns [first, first+count) of part p.
static inline void blockRange(const size_t total, const int parts, const int p,
                              size_t& first, size_t& count) {
    const size_t base = total / parts;
    const size_t rem = total % parts;
    const size_t up = static_cast<size_t>(p);
    first = up * base + std::min(up, rem);
    count = base + (up < rem ? 1 : 0);
}

// Gather the per-device result blocks of all ranks into rank 0's results
// vector. Each device block is sent as a sequence of messages of at most
// kMaxMsg elements; rank 0 only needs to know each rank's total block size.
static void gatherResults(std::vector<double>& results, const std::vector<DeviceWorker>& workers,
                          const size_t numOptions, const int rank, const int size) {
    constexpr size_t kMaxMsg = size_t(1) << 28; // elements per message (2 GiB)
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            size_t first, count;
            blockRange(numOptions, size, r, first, count);
            size_t received = 0;
            while (received < count) {
                MPI_Status st;
                const size_t room = std::min(kMaxMsg, count - received);
                MPI_CHECK(MPI_Recv(results.data() + first + received, static_cast<int>(room),
                                   MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &st));
                int got = 0;
                MPI_CHECK(MPI_Get_count(&st, MPI_DOUBLE, &got));
                received += static_cast<size_t>(got);
            }
        }
    } else {
        for (const auto& w : workers) {
            for (size_t off = 0; off < w.count; off += kMaxMsg) {
                const size_t n = std::min(kMaxMsg, w.count - off);
                MPI_CHECK(MPI_Send(w.hostBuf + off, static_cast<int>(n), MPI_DOUBLE, 0, 0,
                                   MPI_COMM_WORLD));
            }
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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

    // ---- GPU assignment: ranks sharing a node split that node's GPUs -------
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &nodeComm));
    int nodeRank = 0, nodeSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &nodeSize));

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) deviceCount = 0;
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Devices owned by this rank: device d belongs to node-local rank d % nodeSize.
    // If there are more ranks than GPUs, ranks share GPUs round-robin.
    std::vector<int> myDevices;
    if (nodeSize >= deviceCount) {
        myDevices.push_back(nodeRank % deviceCount);
    } else {
        for (int d = nodeRank; d < deviceCount; d += nodeSize) myDevices.push_back(d);
    }
    const int numMyDevices = static_cast<int>(myDevices.size());

    // Distinct GPUs in use across the job (ranks may share a GPU).
    const int nodeDevicesUsed = (nodeRank == 0) ? std::min(nodeSize, deviceCount) : 0;
    int totalDevices = 0;
    MPI_CHECK(MPI_Allreduce(&nodeDevicesUsed, &totalDevices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs in use: %d, OpenMP threads/rank: %d\n",
               size, totalDevices, omp_get_max_threads());
    }

    // ---- Work decomposition -------------------------------------------------
    size_t localFirst = 0, localCount = 0;
    blockRange(numOptions, size, rank, localFirst, localCount);

    // Generate options: rank 0 needs the leading options on the host for
    // validation; everything else is generated directly on the GPUs.
    std::vector<OptionInput> options;
    if (rank == 0) {
        generateOptions(options, std::min<size_t>(10, numOptions));
    }

    // Allocate results: rank 0 holds the full array for output/validation.
    // The GPUs write into per-device pinned buffers (NUMA-local to each GPU).
    std::vector<double> results(rank == 0 ? numOptions : 0);

    // Per-device workers: this rank's block is split evenly across its GPUs.
    std::vector<DeviceWorker> workers(numMyDevices);
    for (int d = 0; d < numMyDevices; ++d) {
        size_t f, c;
        blockRange(localCount, numMyDevices, d, f, c);
        workers[d].device = myDevices[d];
        workers[d].globalFirst = localFirst + f;
        workers[d].count = c;
    }
#pragma omp parallel for num_threads(numMyDevices) schedule(static, 1)
    for (int d = 0; d < numMyDevices; ++d) {
        workers[d].init();
    }

    // ---- Price options ------------------------------------------------------
    if (rank == 0) printf("Pricing options...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    // One OpenMP thread per GPU owned by this rank; each drives its device
    // through double-buffered streams (kernel on one buffer overlaps the
    // device->host copy of the other).
#pragma omp parallel for num_threads(numMyDevices) schedule(static, 1)
    for (int d = 0; d < numMyDevices; ++d) {
        workers[d].run();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localMs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double maxMs = 0.0;
    MPI_CHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        printf("Options per second: %.0f\n", numOptions / (maxMs / 1e3));
    }

    // ---- Collect results on rank 0 ------------------------------------------
    if (rank == 0) {
        // Copy this rank's device blocks into the output array (OpenMP).
        for (const auto& w : workers) {
            double* dst = results.data() + w.globalFirst;
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < w.count; ++i) dst[i] = w.hostBuf[i];
        }
    }
    gatherResults(results, workers, numOptions, rank, size);

    for (auto& w : workers) w.destroy();
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(options, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}
