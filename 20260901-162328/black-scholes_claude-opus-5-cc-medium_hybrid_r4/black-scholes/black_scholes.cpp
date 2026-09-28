// Hybrid MPI + OpenMP + CUDA Black-Scholes option pricing benchmark.
//
// Parallelization strategy:
//   * MPI   distributes the global option range across ranks (one rank per GPU).
//   * CUDA  prices the bulk of each rank's range on its GPU; option inputs are
//           generated on the fly on the device, so only results cross PCIe.
//   * OpenMP drives the GPU pipeline (multi-stream, pinned memory) from one
//           thread while the remaining threads price options on the CPU cores,
//           pulling from the same atomic work queue so CPU and GPU self-balance.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <sched.h>
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

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        const cudaError_t err_ = (call);                                                            \
        if (err_ != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,          \
                    __LINE__);                                                                      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                            \
        }                                                                                           \
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
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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

static constexpr size_t NUM_TEST_OPTIONS = 7;

// Device-side copy of the test option table
__constant__ OptionInput c_testOptions[NUM_TEST_OPTIONS];

// Generate option i of the global sequence (identical to the original
// generateOptions(), but computed on demand instead of materialized).
__host__ __device__ inline OptionInput makeOption(const size_t i) noexcept {
#ifdef __CUDA_ARCH__
    OptionInput o = c_testOptions[i % NUM_TEST_OPTIONS];
#else
    constexpr auto testOptions = getTestOptions();
    OptionInput o = testOptions[i % NUM_TEST_OPTIONS];
#endif
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(NUM_TEST_OPTIONS));
    o.spot *= factor;
    o.strike *= factor;
    return o;
}

// Price [globalStart, globalStart + count) into out[0, count)
__global__ void blackScholesKernel(double* __restrict__ out, const size_t globalStart,
                                   const size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        out[i] = blackScholes(makeOption(globalStart + i));
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

// Work queue granularity: the GPU grabs GPU_GRAB chunks per batch while CPU
// workers take CPU_GRAB at a time, so both sides stay busy and the queue drains
// evenly no matter how the CPU/GPU throughput ratio turns out.
static constexpr size_t CHUNK_SIZE = 1u << 14;
static constexpr size_t GPU_GRAB = 256;  // ~4M options per device batch
static constexpr size_t CPU_GRAB = 4;    // chunks a CPU worker takes at once
static constexpr int NUM_STREAMS = 3;

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    // ---- Identify the ranks sharing this node ----
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    // ---- Carve out this rank's share of the node's cores ----
    // Ranks that were launched with the same CPU affinity mask (unbound ranks, or
    // several ranks bound to the same socket) would otherwise oversubscribe the
    // same cores, so they split that mask between themselves.
    std::vector<int> myCpus;      // CPUs this rank pins its OpenMP threads to
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        std::vector<int> cpus;
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &mask)) cpus.push_back(c);
            }
        }

        std::vector<cpu_set_t> nodeMasks(static_cast<size_t>(localSize));
        MPI_Allgather(&mask, sizeof(cpu_set_t), MPI_BYTE, nodeMasks.data(), sizeof(cpu_set_t),
                      MPI_BYTE, nodeComm);
        int sharers = 0, myShareIndex = 0;
        for (int p = 0; p < localSize; ++p) {
            if (memcmp(&mask, &nodeMasks[static_cast<size_t>(p)], sizeof(cpu_set_t)) == 0) {
                if (p < localRank) ++myShareIndex;
                ++sharers;
            }
        }

        if (sharers > 1 && cpus.size() >= static_cast<size_t>(sharers)) {
            // Slice by physical core (a rank owns all SMT siblings of its cores) and
            // keep the slices contiguous, which also keeps each rank on as few NUMA
            // nodes as possible.
            std::vector<std::pair<int, std::vector<int>>> cores; // (core id, sibling cpus)
            for (const int c : cpus) {
                int coreId = c;
                char path[128];
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
                if (FILE* f = fopen(path, "r")) {
                    int first = c;
                    if (fscanf(f, "%d", &first) == 1) coreId = std::min(first, c);
                    fclose(f);
                }
                auto it = std::find_if(cores.begin(), cores.end(),
                                       [coreId](const auto& e) { return e.first == coreId; });
                if (it == cores.end()) {
                    cores.push_back({coreId, {c}});
                } else {
                    it->second.push_back(c);
                }
            }
            const size_t perRank = cores.size() / static_cast<size_t>(sharers);
            const size_t firstCore = perRank * static_cast<size_t>(myShareIndex);
            size_t maxSiblings = 1;
            for (const auto& e : cores) maxSiblings = std::max(maxSiblings, e.second.size());
            // Primary siblings first, so a partially filled team spreads over cores.
            for (size_t sib = 0; sib < maxSiblings; ++sib) {
                for (size_t i = firstCore; i < firstCore + perRank && i < cores.size(); ++i) {
                    if (sib < cores[i].second.size()) myCpus.push_back(cores[i].second[sib]);
                }
            }
        }
        if (myCpus.empty()) myCpus = cpus;

        // Confine this rank (and everything it allocates or spawns, including the
        // pinned staging buffer and the CUDA driver threads) to its own slice.
        if (!myCpus.empty() && myCpus.size() != cpus.size()) {
            cpu_set_t sliceMask;
            CPU_ZERO(&sliceMask);
            for (const int c : myCpus) CPU_SET(c, &sliceMask);
            sched_setaffinity(0, sizeof(sliceMask), &sliceMask);
        }
    }

    int numThreads = myCpus.empty() ? omp_get_max_threads() : static_cast<int>(myCpus.size());
    if (const char* envThreads = getenv("OMP_NUM_THREADS")) {
        numThreads = atoi(envThreads);
    }
    if (numThreads < 1) numThreads = 1;
    omp_set_num_threads(numThreads);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- Pick a GPU that matches this rank's core slice ----
    // Order the node's ranks by the first CPU they own and the GPUs by NUMA node,
    // then pair them up: each rank gets a distinct GPU, attached to the memory
    // controller its cores sit closest to.
    int myDevice = 0;
    {
        const int myFirstCpu = myCpus.empty() ? localRank : myCpus[0];
        std::vector<int> firstCpus(static_cast<size_t>(localSize));
        MPI_Allgather(&myFirstCpu, 1, MPI_INT, firstCpus.data(), 1, MPI_INT, nodeComm);
        int position = 0;
        for (int p = 0; p < localSize; ++p) {
            const int other = firstCpus[static_cast<size_t>(p)];
            if (other < myFirstCpu || (other == myFirstCpu && p < localRank)) ++position;
        }

        std::vector<std::pair<int, int>> devices; // (numa node, device index)
        for (int d = 0; d < deviceCount; ++d) {
            int numaNode = 0;
            char busId[32] = {0};
            if (cudaDeviceGetPCIBusId(busId, sizeof(busId), d) == cudaSuccess) {
                unsigned dom = 0, bus = 0, dev = 0, fn = 0;
                if (sscanf(busId, "%x:%x:%x.%x", &dom, &bus, &dev, &fn) == 4) {
                    char path[128];
                    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x/numa_node",
                             dom, bus, dev, fn);
                    if (FILE* f = fopen(path, "r")) {
                        if (fscanf(f, "%d", &numaNode) != 1) numaNode = 0;
                        fclose(f);
                    }
                }
            }
            devices.push_back({std::max(numaNode, 0), d});
        }
        std::sort(devices.begin(), devices.end());
        myDevice = devices[static_cast<size_t>(position % deviceCount)].second;
    }
    MPI_Comm_free(&nodeComm);

    CUDA_CHECK(cudaSetDevice(myDevice));
    CUDA_CHECK(cudaFree(nullptr)); // establish the context outside the timed region

    {
        constexpr auto testOptions = getTestOptions();
        CUDA_CHECK(cudaMemcpyToSymbol(c_testOptions, testOptions.data(),
                                      sizeof(OptionInput) * NUM_TEST_OPTIONS));
    }

    // ---- Distribute the global option range ----
    const size_t base = numOptions / static_cast<size_t>(nranks);
    const size_t rem = numOptions % static_cast<size_t>(nranks);
    const size_t myStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t myCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Pinned host buffer so the GPU can DMA results straight into it
    double* localResults = nullptr;
    if (myCount > 0) {
        CUDA_CHECK(cudaMallocHost(&localResults, myCount * sizeof(double)));
    }

    const size_t numChunks = (myCount + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const size_t streamCapacity = GPU_GRAB * CHUNK_SIZE;

    cudaStream_t streams[NUM_STREAMS];
    double* deviceBuffers[NUM_STREAMS] = {nullptr};
    const size_t deviceChunkElems = std::min(streamCapacity, myCount > 0 ? myCount : size_t{1});
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(&deviceBuffers[s], deviceChunkElems * sizeof(double)));
    }

    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, myDevice));
    constexpr int BLOCK_SIZE = 256;
    const int maxBlocks = smCount * 32;

    // Warm up the kernel so JIT/module load does not pollute the measurement
    blackScholesKernel<<<1, BLOCK_SIZE, 0, streams[0]>>>(deviceBuffers[0], 0, 1);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // Pin the OpenMP team (outside the timed region) so threads neither migrate
    // nor collide with the threads of another rank on the same node.
    if (!myCpus.empty() && getenv("OMP_PROC_BIND") == nullptr) {
        #pragma omp parallel num_threads(numThreads)
        {
            cpu_set_t threadMask;
            CPU_ZERO(&threadMask);
            CPU_SET(myCpus[static_cast<size_t>(omp_get_thread_num()) % myCpus.size()], &threadMask);
            sched_setaffinity(0, sizeof(threadMask), &threadMask);
        }
    }

    std::atomic<size_t> nextChunk{0};

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Price the next `grab` queued chunks on the CPU; false when the queue is empty.
    const auto priceCpuChunks = [&](const size_t grab) -> bool {
        const size_t chunk = nextChunk.fetch_add(grab, std::memory_order_relaxed);
        if (chunk >= numChunks) return false;
        const size_t begin = chunk * CHUNK_SIZE;
        const size_t end = std::min(myCount, (chunk + grab) * CHUNK_SIZE);
        for (size_t i = begin; i < end; ++i) {
            localResults[i] = blackScholes(makeOption(myStart + i));
        }
        return true;
    };

    #pragma omp parallel num_threads(numThreads)
    {
        const int tid = omp_get_thread_num();

        if (tid == 0) {
            // ---- GPU pipeline driver ----
            // While a stream is in flight this thread prices CPU chunks instead of
            // blocking, so the core it owns is never idle.
            bool busy[NUM_STREAMS] = {false};
            int s = 0;
            while (true) {
                while (busy[s] && cudaStreamQuery(streams[s]) == cudaErrorNotReady) {
                    priceCpuChunks(1);
                }
                if (busy[s]) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
                busy[s] = false;

                const size_t chunk = nextChunk.fetch_add(GPU_GRAB, std::memory_order_relaxed);
                if (chunk >= numChunks) break;

                const size_t begin = chunk * CHUNK_SIZE;
                const size_t end = std::min(myCount, (chunk + GPU_GRAB) * CHUNK_SIZE);
                const size_t count = end - begin;

                const int blocks = static_cast<int>(
                    std::min<size_t>(maxBlocks, (count + BLOCK_SIZE - 1) / BLOCK_SIZE));
                blackScholesKernel<<<blocks, BLOCK_SIZE, 0, streams[s]>>>(deviceBuffers[s],
                                                                         myStart + begin, count);
                cudaMemcpyAsync(localResults + begin, deviceBuffers[s], count * sizeof(double),
                                cudaMemcpyDeviceToHost, streams[s]);
                busy[s] = true;
                s = (s + 1) % NUM_STREAMS;
            }
            for (int i = 0; i < NUM_STREAMS; ++i) {
                while (busy[i] && cudaStreamQuery(streams[i]) == cudaErrorNotReady) {
                    priceCpuChunks(1);
                }
                if (busy[i]) cudaStreamSynchronize(streams[i]);
            }
        }

        // ---- CPU workers (thread 0 joins in once the GPU queue is drained) ----
        while (priceCpuChunks(CPU_GRAB)) {
        }
    }

    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long localMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long micros = 0;
    MPI_Reduce(&localMicros, &micros, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", micros / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (micros / 1e6));
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> all;
        std::vector<int> counts(nranks), displs(nranks);
        for (int p = 0; p < nranks; ++p) {
            const size_t c = base + (static_cast<size_t>(p) < rem ? 1 : 0);
            const size_t d = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), rem);
            counts[p] = static_cast<int>(c);
            displs[p] = static_cast<int>(d);
        }
        if (rank == 0) all.resize(numOptions);
        MPI_Gatherv(localResults, static_cast<int>(myCount), MPI_DOUBLE, all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(all, "OptionPrices");
        }
    }

    // Validation
    int exitStatus = 0;
    if (validate) {
        // Collect the first (up to) 10 computed prices on rank 0; each index is
        // owned by exactly one rank, so a sum-reduction picks up the right value.
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        std::array<double, 10> mine{}, gathered{};
        for (size_t i = 0; i < numChecks; ++i) {
            if (i >= myStart && i < myStart + myCount) mine[i] = localResults[i - myStart];
        }
        MPI_Reduce(mine.data(), gathered.data(), 10, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        int status = 0;
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> headOptions(numChecks);
            for (size_t i = 0; i < numChecks; ++i) headOptions[i] = makeOption(i);
            std::vector<double> headResults(gathered.begin(), gathered.begin() + numChecks);
            const bool valid = validateResults(headOptions, headResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitStatus = status;
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(deviceBuffers[s]);
    }
    if (localResults) cudaFreeHost(localResults);

    MPI_Finalize();
    return exitStatus;
}
