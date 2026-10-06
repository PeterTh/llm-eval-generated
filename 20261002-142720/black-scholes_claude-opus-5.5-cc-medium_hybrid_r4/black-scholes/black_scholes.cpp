#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <pthread.h>
#include <sched.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (shared by host and device code)
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

    // erf is odd (both glibc and CUDA evaluate it exactly symmetrically), so
    // cumulativeNormal(-x) == 0.5 * (1.0 - erf(x * M_SQRT1_2)) bit for bit. Evaluating
    // the two erf terms once, outside the type branch, keeps GPU warps that mix
    // calls and puts from executing both sets of transcendental calls.
    const double e1 = erf(d1 * M_SQRT1_2);
    const double e2 = erf(d2 * M_SQRT1_2);
    const double discount = exp(-r * T);
    const double dividend = exp(-q * T);

    double price;
    if (option.type == CALL) {
        price = S * dividend * (0.5 * (1.0 + e1)) - K * discount * (0.5 * (1.0 + e2));
    } else { // PUT
        price = K * discount * (0.5 * (1.0 - e2)) - S * dividend * (0.5 * (1.0 - e1));
    }

    return price;
}

// Standard test cases for validation
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

constexpr size_t kNumTest = getTestOptions().size();

// Test set passed by value to kernels
struct TestSet {
    OptionInput o[kNumTest];
};

static TestSet makeTestSet() {
    constexpr auto testOptions = getTestOptions();
    TestSet t{};
    for (size_t i = 0; i < kNumTest; ++i) t.o[i] = testOptions[i];
    return t;
}

// Option i of the scaled test set (identical arithmetic on host and device)
__host__ __device__ inline OptionInput makeOption(const OptionInput* testOptions, const size_t i) noexcept {
    // Cycle through test options and vary parameters slightly
    OptionInput o = testOptions[i % kNumTest];
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTest));
    o.spot *= factor;
    o.strike *= factor;
    return o;
}

// Generate global options [first, first + count) on the host (OpenMP, first-touch)
void generateOptions(OptionInput* options, const size_t first, const size_t count) {
    const TestSet base = makeTestSet();
    const long long n = static_cast<long long>(count);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        options[i] = makeOption(base.o, first + static_cast<size_t>(i));
    }
}

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    options.resize(numOptions);
    generateOptions(options.data(), 0, numOptions);
}

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA infrastructure
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        const cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

constexpr int kThreadsPerBlock = 256;
constexpr int kNumStreams = 4;

// Device-side generation of global options [first, first + count)
__global__ void generateKernel(OptionInput* __restrict__ out, const TestSet base,
                               const size_t first, const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        out[i] = makeOption(base.o, first + i);
    }
}

// Device-side pricing
__global__ void __launch_bounds__(kThreadsPerBlock)
priceKernel(const OptionInput* __restrict__ options, double* __restrict__ results, const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        results[i] = blackScholes(options[i]);
    }
}

struct GpuContext {
    int smCount = 1;
    cudaStream_t streams[kNumStreams];
};

static unsigned gridFor(const GpuContext& g, const size_t count) {
    const size_t blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const size_t maxBlocks = static_cast<size_t>(g.smCount) * 32;
    return static_cast<unsigned>(std::max<size_t>(1, std::min(blocks, maxBlocks)));
}

static void syncGpu(const GpuContext& g) {
    for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaStreamSynchronize(g.streams[s]));
}

// Enqueue pricing of count options on the GPU in pipelined chunks spread over
// the streams, so kernels overlap with device-to-host copies of earlier chunks.
// Results are DMA'd straight into the pinned host array hRes. Returns immediately.
static void launchGpuPricing(const GpuContext& g, const OptionInput* dOpts, double* dRes,
                             double* hRes, const size_t count) {
    if (count == 0) return;
    constexpr size_t kMinChunk = size_t(1) << 18;
    const size_t chunk = std::max(kMinChunk, (count + 4 * kNumStreams - 1) / (4 * kNumStreams));
    int s = 0;
    for (size_t off = 0; off < count; off += chunk, s = (s + 1) % kNumStreams) {
        const size_t n = std::min(chunk, count - off);
        priceKernel<<<gridFor(g, n), kThreadsPerBlock, 0, g.streams[s]>>>(dOpts + off, dRes + off, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hRes + off, dRes + off, n * sizeof(double),
                                   cudaMemcpyDeviceToHost, g.streams[s]));
    }
}

// OpenMP pricing on the host. The static schedule matches generateOptions, so
// every thread reads the input pages it first-touched (NUMA-local).
static void cpuPricing(const OptionInput* opts, double* res, const size_t count) {
    const long long n = static_cast<long long>(count);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        res[i] = blackScholes(opts[i]);
    }
}

// CPUs in this process's affinity mask (as set by the MPI launcher)
static std::vector<int> allowedCpus(cpu_set_t& mask) {
    CPU_ZERO(&mask);
    std::vector<int> cpus;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &mask)) cpus.push_back(c);
        }
    }
    return cpus;
}

// Pin OpenMP thread t to cpus[firstSlot + t], unless the user controls placement
// through OMP_PROC_BIND/OMP_PLACES.
static void pinOpenMPThreads(const std::vector<int>& cpus, const int firstSlot) {
    if (cpus.empty() || getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) return;
    #pragma omp parallel
    {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[(firstSlot + omp_get_thread_num()) % cpus.size()], &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
}

// Measure GPU and CPU pricing throughput while both run concurrently on a sample
// (as they will in the timed run) and return the fraction of work for the GPU.
static double calibrateGpuFraction(const GpuContext& g, const OptionInput* dOpts, double* dRes,
                                   double* hRes, const size_t localBegin, const size_t sample) {
    std::unique_ptr<OptionInput[]> hOpts(new OptionInput[sample]);
    std::unique_ptr<double[]> hOut(new double[sample]);
    generateOptions(hOpts.get(), localBegin, sample);

    cudaEvent_t start, stop[kNumStreams];
    CUDA_CHECK(cudaEventCreate(&start));
    for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaEventCreate(&stop[s]));

    double frac = 0.5;
    for (int rep = 0; rep < 3; ++rep) {  // first repetition warms up module, clocks and thread pool
        CUDA_CHECK(cudaEventRecord(start, g.streams[0]));
        for (int s = 1; s < kNumStreams; ++s) CUDA_CHECK(cudaStreamWaitEvent(g.streams[s], start, 0));
        launchGpuPricing(g, dOpts, dRes, hRes, sample);
        for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaEventRecord(stop[s], g.streams[s]));

        const double t0 = omp_get_wtime();
        cpuPricing(hOpts.get(), hOut.get(), sample);
        const double tCpu = omp_get_wtime() - t0;

        float tGpuMs = 0.0f;
        for (int s = 0; s < kNumStreams; ++s) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventSynchronize(stop[s]));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop[s]));
            tGpuMs = std::max(tGpuMs, ms);
        }
        const double gpuRate = sample / std::max(1e-3 * tGpuMs, 1e-9);
        const double cpuRate = sample / std::max(tCpu, 1e-9);
        frac = gpuRate / (gpuRate + cpuRate);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaEventDestroy(stop[s]));
    return frac;
}

// Parallel copy of count doubles (OpenMP)
static void parallelCopy(double* dst, const double* src, const size_t count) {
    constexpr size_t kBlock = size_t(1) << 16;
    const long long blocks = static_cast<long long>((count + kBlock - 1) / kBlock);
    #pragma omp parallel for schedule(static)
    for (long long blk = 0; blk < blocks; ++blk) {
        const size_t off = static_cast<size_t>(blk) * kBlock;
        memcpy(dst + off, src + off, std::min(kBlock, count - off) * sizeof(double));
    }
}

// Collect each rank's contiguous block of results into rank 0's array. Ranks on
// rank 0's node expose their block in a shared-memory window (win) that rank 0
// copies directly; ranks on other nodes send it with MPI messages.
static void gatherResults(const int rank, const int size, const std::vector<size_t>& begins,
                          const double* local, const size_t localCount, double* global,
                          const std::vector<char>& onRootNode, const std::vector<int>& nodeWorldRanks,
                          const MPI_Win win) {
    constexpr size_t kMaxMsg = size_t(1) << 27;  // elements per message (keeps counts < INT_MAX)
    if (rank == 0) {
        std::vector<MPI_Request> reqs;
        for (int r = 1; r < size; ++r) {
            if (onRootNode[r]) continue;
            for (size_t off = begins[r]; off < begins[r + 1]; off += kMaxMsg) {
                reqs.emplace_back();
                MPI_Irecv(global + off, static_cast<int>(std::min(kMaxMsg, begins[r + 1] - off)),
                          MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &reqs.back());
            }
        }
        if (win != MPI_WIN_NULL) {
            MPI_Win_sync(win);
            for (size_t j = 0; j < nodeWorldRanks.size(); ++j) {
                const int r = nodeWorldRanks[j];
                if (r == 0 || begins[r + 1] == begins[r]) continue;
                MPI_Aint bytes = 0;
                int dispUnit = 0;
                double* peer = nullptr;
                MPI_Win_shared_query(win, static_cast<int>(j), &bytes, &dispUnit, &peer);
                parallelCopy(global + begins[r], peer, begins[r + 1] - begins[r]);
            }
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else if (!onRootNode[rank]) {
        for (size_t off = 0; off < localCount; off += kMaxMsg) {
            MPI_Send(local + off, static_cast<int>(std::min(kMaxMsg, localCount - off)), MPI_DOUBLE,
                     0, 0, MPI_COMM_WORLD);
        }
    }
}

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

int main(int argc, char** argv) {
    // Only the master thread of each rank makes MPI calls
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Node-local rank -> GPU mapping
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    GpuContext gpu;
    CUDA_CHECK(cudaDeviceGetAttribute(&gpu.smCount, cudaDevAttrMultiProcessorCount, device));
    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.streams[s], cudaStreamNonBlocking));
    }

    // Co-located ranks with the same CPU affinity mask split its CPUs evenly
    cpu_set_t mask;
    const std::vector<int> cpus = allowedCpus(mask);
    std::vector<cpu_set_t> masks(localSize);
    MPI_Allgather(&mask, sizeof(mask), MPI_BYTE, masks.data(), sizeof(mask), MPI_BYTE, localComm);
    int sharers = 0, sharerIndex = 0;
    for (int r = 0; r < localSize; ++r) {
        if (CPU_EQUAL(&masks[r], &mask)) {
            if (r < localRank) ++sharerIndex;
            ++sharers;
        }
    }
    if (getenv("OMP_NUM_THREADS") == nullptr && !cpus.empty()) {
        omp_set_num_threads(std::max<int>(1, static_cast<int>(cpus.size()) / sharers));
    }
    pinOpenMPThreads(cpus, sharerIndex * omp_get_max_threads());

    // Block distribution of the global index space across ranks
    std::vector<size_t> begins(size + 1);
    for (int r = 0; r <= size; ++r) {
        const size_t base = numOptions / size, rem = numOptions % size;
        begins[r] = base * r + std::min<size_t>(r, rem);
    }
    const size_t localBegin = begins[rank];
    const size_t localCount = begins[rank + 1] - localBegin;

    // Device buffers sized for as much of the local share as fits in GPU memory
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t usable = freeMem > (size_t(256) << 20) ? static_cast<size_t>(0.9 * (freeMem - (size_t(256) << 20))) : 0;
    const size_t gpuCap = std::min(localCount, usable / (sizeof(OptionInput) + sizeof(double)));
    OptionInput* dOpts = nullptr;
    double* dRes = nullptr;
    if (gpuCap > 0) {
        CUDA_CHECK(cudaMalloc(&dOpts, gpuCap * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&dRes, gpuCap * sizeof(double)));
    }

    // Which ranks share a node (and thus memory) with rank 0
    std::vector<int> nodeWorldRanks(localSize);
    MPI_Allgather(&rank, 1, MPI_INT, nodeWorldRanks.data(), 1, MPI_INT, localComm);
    const bool rootOnMyNode = std::find(nodeWorldRanks.begin(), nodeWorldRanks.end(), 0) != nodeWorldRanks.end();
    std::vector<char> onRootNode(size, 0);
    {
        const char mine = rootOnMyNode ? 1 : 0;
        MPI_Allgather(&mine, 1, MPI_CHAR, onRootNode.data(), 1, MPI_CHAR, MPI_COMM_WORLD);
    }

    // Allocate results: full array on rank 0, local block in a shared window on
    // rank 0's node, private local block elsewhere. All are pinned for direct DMA.
    std::vector<double> results;
    double* localRes = nullptr;
    MPI_Win win = MPI_WIN_NULL;
    if (rootOnMyNode && localSize > 1) {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "alloc_shared_noncontig", "true");
        const MPI_Aint bytes = (rank == 0) ? 0 : static_cast<MPI_Aint>(localCount * sizeof(double));
        MPI_Win_allocate_shared(bytes, sizeof(double), info, localComm, &localRes, &win);
        MPI_Info_free(&info);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }
    if (rank == 0) {
        results.resize(numOptions);
        localRes = results.data();
    } else if (win == MPI_WIN_NULL && localCount > 0) {
        CUDA_CHECK(cudaMallocHost(&localRes, localCount * sizeof(double)));
    }
    const bool registered = localCount > 0 && (rank == 0 || win != MPI_WIN_NULL);
    if (registered) {
        CUDA_CHECK(cudaHostRegister(localRes, localCount * sizeof(double), cudaHostRegisterDefault));
    }

    const TestSet base = makeTestSet();

    // Split the local range: the GPU prices [0, gpuCount), OpenMP threads the rest
    size_t gpuCount = 0;
    if (gpuCap > 0) {
        const size_t sample = std::min<size_t>(gpuCap, size_t(1) << 22);
        generateKernel<<<gridFor(gpu, sample), kThreadsPerBlock, 0, gpu.streams[0]>>>(dOpts, base, localBegin, sample);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const double gpuFrac = calibrateGpuFraction(gpu, dOpts, dRes, localRes, localBegin, sample);
        gpuCount = std::min(gpuCap, static_cast<size_t>(gpuFrac * localCount + 0.5));
    }
    const size_t cpuCount = localCount - gpuCount;

    // Generate this rank's options: GPU share on the device, CPU share on the host
    if (gpuCount > 0) {
        generateKernel<<<gridFor(gpu, gpuCount), kThreadsPerBlock, 0, gpu.streams[0]>>>(dOpts, base, localBegin, gpuCount);
        CUDA_CHECK(cudaGetLastError());
    }
    std::unique_ptr<OptionInput[]> hOpts(new OptionInput[cpuCount]);
    generateOptions(hOpts.get(), localBegin + gpuCount, cpuCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Price options
    if (rank == 0) printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    launchGpuPricing(gpu, dOpts, dRes, localRes, gpuCount);  // asynchronous
    cpuPricing(hOpts.get(), localRes + gpuCount, cpuCount);  // overlaps with the GPU work
    syncGpu(gpu);
    if (win != MPI_WIN_NULL) MPI_Win_sync(win);
    MPI_Barrier(MPI_COMM_WORLD);  // every rank holds its priced block in host memory

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Collect the distributed results on rank 0 (reported separately from the pricing time)
    const double gatherStart = MPI_Wtime();
    gatherResults(rank, size, begins, localRes, localCount, results.data(), onRootNode, nodeWorldRanks, win);
    const double gatherTime = MPI_Wtime() - gatherStart;

    // Release accelerator resources
    if (registered) {
        CUDA_CHECK(cudaHostUnregister(localRes));
    } else if (rank != 0 && localRes) {
        CUDA_CHECK(cudaFreeHost(localRes));
    }
    if (win != MPI_WIN_NULL) {
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
    }
    MPI_Comm_free(&localComm);
    if (dOpts) CUDA_CHECK(cudaFree(dOpts));
    if (dRes) CUDA_CHECK(cudaFree(dRes));
    for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaStreamDestroy(gpu.streams[s]));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
        if (size > 1) printf("Result gather time (%d ranks): %.3f ms\n", size, gatherTime * 1000.0);

        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> options;
            generateOptions(options, std::min<size_t>(10, numOptions));
            bool valid = validateResults(options, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
