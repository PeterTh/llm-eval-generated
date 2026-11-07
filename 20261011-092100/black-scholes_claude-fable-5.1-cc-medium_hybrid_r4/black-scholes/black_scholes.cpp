// Black-Scholes option pricing benchmark -- hybrid MPI + OpenMP + CUDA version.
//
// Decomposition:
//   * MPI     : the global option range [0, N) is split into contiguous, equally
//               sized slices, one per rank.  Rank 0 gathers the priced slices
//               (outside the timed region) for result printing / validation.
//   * CUDA    : each rank owns one or more GPUs (chosen from the node-local rank
//               id).  The bulk of a rank's slice is priced on the GPU(s).  Inputs
//               are generated directly into device memory (SoA layout) before the
//               timed region, exactly like the original generates them into host
//               memory before timing.  Pricing kernels run in chunks on several
//               streams so kernel execution overlaps the PCIe copy of results
//               back into (pinned) host memory.
//   * OpenMP  : one thread per GPU drives that GPU asynchronously; all threads
//               then price the CPU share of the slice (tail of the slice) with a
//               dynamically scheduled loop.  OpenMP is also used to generate the
//               CPU share of the inputs.
//
// The CPU share per rank is BS_CPU_FRACTION (default below).  The split is
// static, so results are deterministic for a given rank count and fraction.
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                           \
            MPI_Abort(MPI_COMM_WORLD, 2);                                           \
        }                                                                           \
    } while (0)

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

// Black-Scholes formula for European options
__host__ __device__ inline double blackScholesCore(const int type, const double S,
                                                   const double K, const double q,
                                                   const double r, const double T,
                                                   const double sigma) noexcept {
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    // erf is exactly odd, so N(-d) = 0.5 * (1 - erf(d / sqrt 2)) reproduces
    // cumulativeNormal(-d) bit for bit while sharing the two erf evaluations
    // between the call and put branches.
    const double e1 = erf(d1 * M_SQRT1_2);
    const double e2 = erf(d2 * M_SQRT1_2);
    const double discount = exp(-r * T);
    const double divDiscount = exp(-q * T);

    double price;
    if (type == CALL) {
        price = S * divDiscount * (0.5 * (1.0 + e1)) - K * discount * (0.5 * (1.0 + e2));
    } else { // PUT
        price = K * discount * (0.5 * (1.0 - e2)) - S * divDiscount * (0.5 * (1.0 - e1));
    }

    return price;
}

inline double blackScholes(const OptionInput& option) noexcept {
    return blackScholesCore(option.type, option.spot, option.strike, option.q,
                            option.r, option.t, option.vol);
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

constexpr int kNumTestOptions = 7;

// Option i of the generated set (identical to the original generateOptions()).
__host__ __device__ inline OptionInput makeOption(const OptionInput* table,
                                                  const size_t i) noexcept {
    OptionInput o = table[i % kNumTestOptions];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    o.spot *= factor;
    o.strike *= factor;
    return o;
}

// Generate a set of options [first, first+count) into host memory (CPU share).
void generateOptions(std::vector<OptionInput>& options, const size_t first,
                     const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < count; ++k) {
        options[k] = makeOption(testOptions.data(), first + k);
    }
}

// ----------------------------------------------------------------------------
// CUDA side
// ----------------------------------------------------------------------------
__constant__ OptionInput c_testOptions[kNumTestOptions];

// Structure-of-arrays view of the options resident on one GPU.
struct DeviceOptions {
    int* type;
    double* strike;
    double* spot;
    double* q;
    double* r;
    double* t;
    double* vol;
    double* result;
    size_t count;
};

__global__ void __launch_bounds__(256)
generateOptionsKernel(DeviceOptions d, const size_t firstGlobalIndex) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < d.count;
         k += stride) {
        const OptionInput o = makeOption(c_testOptions, firstGlobalIndex + k);
        d.type[k] = o.type;
        d.strike[k] = o.strike;
        d.spot[k] = o.spot;
        d.q[k] = o.q;
        d.r[k] = o.r;
        d.t[k] = o.t;
        d.vol[k] = o.vol;
    }
}

__global__ void __launch_bounds__(256)
blackScholesKernel(const DeviceOptions d, const size_t begin, const size_t end) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < end;
         k += stride) {
        d.result[k] = blackScholesCore(d.type[k], d.spot[k], d.strike[k], d.q[k], d.r[k],
                                       d.t[k], d.vol[k]);
    }
}

constexpr int kNumStreams = 4;
constexpr size_t kGpuChunk = size_t(1) << 22;  // options per kernel/copy chunk

struct GpuContext {
    int device = -1;
    size_t first = 0;   // offset into the rank-local slice
    size_t count = 0;
    DeviceOptions d{};
    cudaStream_t streams[kNumStreams]{};
    int numSMs = 0;
};

static void gpuSetup(GpuContext& g, const size_t globalFirst) {
    CUDA_CHECK(cudaSetDevice(g.device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, g.device));
    g.numSMs = prop.multiProcessorCount;

    constexpr auto testOptions = getTestOptions();
    CUDA_CHECK(cudaMemcpyToSymbol(c_testOptions, testOptions.data(),
                                  sizeof(OptionInput) * kNumTestOptions));

    g.d.count = g.count;
    if (g.count == 0) return;
    CUDA_CHECK(cudaMalloc(&g.d.type, g.count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d.strike, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.spot, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.q, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.r, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.t, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.vol, g.count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d.result, g.count * sizeof(double)));
    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g.streams[s], cudaStreamNonBlocking));
    }

    const int blocks = std::min<size_t>(g.numSMs * 16, (g.count + 255) / 256);
    generateOptionsKernel<<<blocks, 256>>>(g.d, globalFirst + g.first);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Enqueue all pricing work for this GPU (asynchronous; returns immediately).
static void gpuLaunch(GpuContext& g, double* hostResults) {
    if (g.count == 0) return;
    CUDA_CHECK(cudaSetDevice(g.device));
    int s = 0;
    for (size_t begin = 0; begin < g.count; begin += kGpuChunk, s = (s + 1) % kNumStreams) {
        const size_t end = std::min(begin + kGpuChunk, g.count);
        const size_t n = end - begin;
        const int blocks = static_cast<int>(std::min<size_t>(g.numSMs * 8, (n + 255) / 256));
        blackScholesKernel<<<blocks, 256, 0, g.streams[s]>>>(g.d, begin, end);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hostResults + g.first + begin, g.d.result + begin,
                                   n * sizeof(double), cudaMemcpyDeviceToHost, g.streams[s]));
    }
}

static void gpuWait(GpuContext& g) {
    if (g.count == 0) return;
    CUDA_CHECK(cudaSetDevice(g.device));
    for (int s = 0; s < kNumStreams; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(g.streams[s]));
    }
}

static void gpuTeardown(GpuContext& g) {
    if (g.count == 0) return;
    CUDA_CHECK(cudaSetDevice(g.device));
    for (int s = 0; s < kNumStreams; ++s) CUDA_CHECK(cudaStreamDestroy(g.streams[s]));
    CUDA_CHECK(cudaFree(g.d.type));
    CUDA_CHECK(cudaFree(g.d.strike));
    CUDA_CHECK(cudaFree(g.d.spot));
    CUDA_CHECK(cudaFree(g.d.q));
    CUDA_CHECK(cudaFree(g.d.r));
    CUDA_CHECK(cudaFree(g.d.t));
    CUDA_CHECK(cudaFree(g.d.vol));
    CUDA_CHECK(cudaFree(g.d.result));
}

// ----------------------------------------------------------------------------

bool validateResults(const size_t numOptions, const std::vector<double>& results) {
    bool allPassed = true;
    constexpr auto testOptions = getTestOptions();
    const int numChecks = std::min(static_cast<size_t>(10), numOptions);

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = makeOption(testOptions.data(), i).value;
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
    printf("Environment:\n");
    printf("  BS_CPU_FRACTION  Share of each rank's options priced on the CPU (OpenMP)\n");
    printf("                   instead of the GPU(s) (default: 0.20)\n");
    printf("  OMP_NUM_THREADS  Threads per rank (default: CPUs in the affinity mask\n");
    printf("                   divided among node-local ranks). Binding the ranks\n");
    printf("                   (e.g. mpirun --bind-to numa, OMP_PROC_BIND=close\n");
    printf("                   OMP_PLACES=cores) gives the best CPU-side throughput.\n");
}

// Gather rank-local slices into results on rank 0 (chunked, handles > INT_MAX).
static void gatherResults(const std::vector<double>& local, std::vector<double>& results,
                          const size_t numOptions, const int rank, const int size) {
    constexpr size_t kMaxChunk = size_t(1) << 28;
    if (rank == 0) {
        results.resize(numOptions);
        std::memcpy(results.data(), local.data(), local.size() * sizeof(double));
        for (int src = 1; src < size; ++src) {
            const size_t lo = numOptions * static_cast<size_t>(src) / size;
            const size_t hi = numOptions * static_cast<size_t>(src + 1) / size;
            for (size_t off = lo; off < hi; off += kMaxChunk) {
                const int n = static_cast<int>(std::min(kMaxChunk, hi - off));
                MPI_Recv(results.data() + off, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t off = 0; off < local.size(); off += kMaxChunk) {
            const int n = static_cast<int>(std::min(kMaxChunk, local.size() - off));
            MPI_Send(local.data() + off, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

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

    double cpuFraction = 0.20;
    if (const char* env = getenv("BS_CPU_FRACTION")) {
        cpuFraction = std::clamp(atof(env), 0.0, 1.0);
    }

    // --- GPU assignment: node-local ranks share the node's GPUs -------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::vector<int> myDevices;
    if (localSize >= numDevices) {
        myDevices.push_back(localRank % numDevices);
    } else {
        for (int d = localRank; d < numDevices; d += localSize) myDevices.push_back(d);
    }
    const int numGpus = static_cast<int>(myDevices.size());
    // Unless the user pinned the thread count, share the node's cores between
    // the node-local ranks instead of oversubscribing them.
    int numThreads = omp_get_max_threads();
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        // omp_get_max_threads() defaults to the CPU count of the process's
        // initial affinity mask (queried by the runtime before it pins the
        // initial thread).  Estimate how many node-local ranks share that mask
        // and divide the CPUs among them.
        const long online = std::max(1L, sysconf(_SC_NPROCESSORS_ONLN));
        const int maskCpus = std::max(1, numThreads);
        const int sharing = std::max(1, static_cast<int>(
                                            (static_cast<long>(localSize) * maskCpus + online / 2) / online));
        numThreads = std::max(1, maskCpus / sharing);
        // Unbound ranks: leave a few CPUs for the CUDA/MPI helper threads so a
        // GPU-driving thread is never descheduled behind spinning workers.
        if (maskCpus >= online) numThreads = std::max(1, numThreads - (numGpus + 2));
    }
    numThreads = std::max(numThreads, numGpus);
    omp_set_num_threads(numThreads);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/rank: %d, CPU fraction: %.3f\n",
               size, numThreads, numGpus, cpuFraction);
    }

    // --- Decomposition --------------------------------------------------------
    const size_t lo = numOptions * static_cast<size_t>(rank) / size;
    const size_t hi = numOptions * static_cast<size_t>(rank + 1) / size;
    const size_t localN = hi - lo;
    const size_t gpuN = static_cast<size_t>(std::llround((1.0 - cpuFraction) * localN));
    const size_t cpuN = localN - gpuN;  // tail of the slice is priced on the CPU

    // Generate options: GPU shares directly on their devices, then the CPU
    // share on the host.  CUDA context creation leaves the process idle for a
    // long time, during which the kernel's automatic NUMA balancing starts
    // sampling (unmapping) old pages; allocating the host-side working set
    // after GPU setup keeps those hinting faults out of the timed region.
    std::vector<GpuContext> gpus(numGpus);
    for (int g = 0; g < numGpus; ++g) {
        gpus[g].device = myDevices[g];
        gpus[g].first = gpuN * static_cast<size_t>(g) / numGpus;
        gpus[g].count = gpuN * static_cast<size_t>(g + 1) / numGpus - gpus[g].first;
    }
    for (auto& g : gpus) gpuSetup(g, lo);

    std::vector<OptionInput> cpuOptions;
    generateOptions(cpuOptions, lo + gpuN, cpuN);

    // Allocate (pinned) local results
    std::vector<double> localResults(localN);
    if (gpuN > 0) {
        CUDA_CHECK(cudaHostRegister(localResults.data(), gpuN * sizeof(double),
                                    cudaHostRegisterDefault));
    }

    // --- Price options --------------------------------------------------------
    if (rank == 0) printf("Pricing options...\n");
    // GPU setup above leaves the CPU idle for a while; make sure the OpenMP team
    // exists and every core is out of its idle state before timing starts.
#pragma omp parallel
    {
        const double t0 = omp_get_wtime();
        volatile double sink = 0.0;
        while (omp_get_wtime() - t0 < 5e-3) sink = sink + 1.0;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    double* const resPtr = localResults.data();
    double* const cpuResPtr = resPtr + gpuN;
    const OptionInput* const cpuOptPtr = cpuOptions.data();
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (tid < numGpus) gpuLaunch(gpus[tid], resPtr);

#pragma omp for schedule(dynamic, 2048) nowait
        for (size_t k = 0; k < cpuN; ++k) {
            cpuResPtr[k] = blackScholes(cpuOptPtr[k]);
        }

        if (tid < numGpus) gpuWait(gpus[tid]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        printf("Options per second: %.0f\n", numOptions / (maxMs / 1e3));
    }

    if (gpuN > 0) CUDA_CHECK(cudaHostUnregister(localResults.data()));
    for (auto& g : gpus) gpuTeardown(g);

    // --- Gather results on rank 0 (outside the timed region) ----------------
    std::vector<double> results;
    if (printResults || validate) {
        gatherResults(localResults, results, numOptions, rank, size);
    }

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(numOptions, results);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
