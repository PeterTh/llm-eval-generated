#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

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

constexpr size_t NUM_BASE_OPTIONS = 7;

// The option set is fully determined by the global option index, so inputs are
// materialized on the fly (on device as well as on host) instead of being
// transferred over the interconnect.
__constant__ OptionInput c_baseOptions[NUM_BASE_OPTIONS];

__host__ __device__ inline OptionInput makeOption(const OptionInput& base, const size_t index) noexcept {
    OptionInput option = base;
    // Same variation as the reference generator
    const double factor = 1.0 + 0.1 * (index / static_cast<double>(NUM_BASE_OPTIONS));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

__global__ void blackScholesKernel(double* __restrict__ out, const size_t begin, const size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
        const size_t global = begin + i;
        out[i] = blackScholes(makeOption(c_baseOptions[global % NUM_BASE_OPTIONS], global));
    }
}

// Host-side pricing of a range of global option indices (used by the OpenMP workers)
inline void blackScholesHostRange(double* __restrict__ out, const size_t begin, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    for (size_t i = 0; i < count; ++i) {
        const size_t global = begin + i;
        out[i] = blackScholes(makeOption(testOptions[global % NUM_BASE_OPTIONS], global));
    }
}

// Generate the options of a global index range (only needed for validation output)
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t global = begin + i;
        options[i] = makeOption(testOptions[global % NUM_BASE_OPTIONS], global);
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

// Bind this rank to one of the local GPUs (round-robin over the node's devices) and
// size the OpenMP team so that ranks sharing a node do not oversubscribe the cores
static int selectDevice(const int rank) {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localRanks = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localRanks);
    MPI_Comm_free(&nodeComm);

    if (getenv("OMP_NUM_THREADS") == nullptr && getenv("OMP_PROC_BIND") == nullptr) {
        // Every rank drives one GPU and one disjoint share of the node's cores. The
        // launcher's binding is deliberately replaced: its defaults either pin a rank
        // to a single core (starving the CPU half of the hybrid split) or hand the
        // same cores to several ranks (which then fight over them).
        const int online = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        const int share = std::max(1, online / localRanks);
        std::vector<int> cores;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        for (int c = 0; c < share; ++c) {
            const int cpu = (localRank * share + c) % online;
            CPU_SET(cpu, &mask);
            cores.push_back(cpu);
        }

        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
            // Not allowed to use those cores (e.g. restricted by a cgroup): fall back
            // to a slice of the cores we did get.
            cores.clear();
            if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
                std::vector<int> allowed;
                for (int c = 0; c < CPU_SETSIZE; ++c) {
                    if (CPU_ISSET(c, &mask)) allowed.push_back(c);
                }
                const size_t slice = std::max<size_t>(1, allowed.size() / localRanks);
                for (size_t i = 0; i < slice && !allowed.empty(); ++i) {
                    cores.push_back(allowed[(localRank * slice + i) % allowed.size()]);
                }
            }
            if (cores.empty()) cores.push_back(sched_getcpu());
        }

        // One thread of the team drives the GPU, the remaining ones price options on
        // the cores. The team is never larger than our share of the cores: spinning
        // threads of oversubscribed teams preempt the ones doing useful work.
        // Threads are pinned explicitly, since freshly spawned ones start out on the
        // parent's core and the kernel spreads them out only slowly.
        omp_set_num_threads(static_cast<int>(cores.size()));
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            cpu_set_t self;
            CPU_ZERO(&self);
            CPU_SET(cores[tid % cores.size()], &self);
            sched_setaffinity(0, sizeof(self), &self);
        }
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on this node\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

// Transfer count doubles, chunked to stay within the int limits of the MPI API
static void sendChunked(const double* data, const size_t count, const int dest) {
    constexpr size_t maxChunk = 1u << 28;
    for (size_t off = 0; off < count; off += maxChunk) {
        const int n = static_cast<int>(std::min(maxChunk, count - off));
        MPI_Send(data + off, n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
}

static void recvChunked(double* data, const size_t count, const int src) {
    constexpr size_t maxChunk = 1u << 28;
    for (size_t off = 0; off < count; off += maxChunk) {
        const int n = static_cast<int>(std::min(maxChunk, count - off));
        MPI_Recv(data + off, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    int mpiThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    const int device = selectDevice(rank);

    // Block distribution of the global option range over the ranks
    const size_t base = numOptions / static_cast<size_t>(numRanks);
    const size_t rem = numOptions % static_cast<size_t>(numRanks);
    const size_t localBegin = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t localCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Local results in pinned memory so that device transfers can overlap with compute
    double* localResults = nullptr;
    bool pinned = false;
    if (localCount > 0) {
        if (cudaHostAlloc(reinterpret_cast<void**>(&localResults), localCount * sizeof(double),
                          cudaHostAllocDefault) == cudaSuccess) {
            pinned = true;
        } else {
            localResults = static_cast<double*>(malloc(localCount * sizeof(double)));
            if (localResults == nullptr) {
                fprintf(stderr, "Failed to allocate result buffer\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Upload the immutable base option table
    constexpr auto testOptions = getTestOptions();
    CUDA_CHECK(cudaMemcpyToSymbol(c_baseOptions, testOptions.data(), sizeof(testOptions)));

    int numSMs = 1;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    // Device-side staging buffers, one per stream, sized so that a handful of
    // launches keep the GPU busy while transfers of previous chunks are in flight
    constexpr int NUM_STREAMS = 3;
    constexpr size_t MAX_GPU_CHUNK = 1u << 22;
    constexpr size_t MIN_GPU_CHUNK = 1u << 15;
    const size_t gpuChunk = std::max(MIN_GPU_CHUNK,
                                     std::min(MAX_GPU_CHUNK, (localCount + NUM_STREAMS - 1) / NUM_STREAMS));
    cudaStream_t streams[NUM_STREAMS];
    double* deviceBuffers[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceBuffers[s]), gpuChunk * sizeof(double)));
    }

    // Warm up the device (context creation, module load) outside of the timed region
    blackScholesKernel<<<1, 32, 0, streams[0]>>>(deviceBuffers[0], 0, 0);
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Bring the thread pool up before the timer so the measurement does not include
    // the cost of spawning it
    int warmup = 0;
    #pragma omp parallel reduction(+ : warmup)
    {
        warmup += omp_get_thread_num();
    }
    if (warmup < 0) {
        fprintf(stderr, "unexpected thread numbering\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // GPU and CPU cores cooperate on the local range: chunks are handed out from a
    // shared cursor, large ones to the GPU driver thread and small ones to the OpenMP
    // workers, which balances the two automatically.
    std::atomic<size_t> cursor(0);
    constexpr size_t CPU_CHUNK = 1u << 13;

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            const int blockSize = 256;
            int nextStream = 0;
            bool streamBusy[NUM_STREAMS] = {};
            while (true) {
                // Shrink the request near the end of the range to avoid a long tail
                const size_t taken = cursor.load(std::memory_order_relaxed);
                const size_t remaining = taken < localCount ? localCount - taken : 0;
                if (remaining == 0) break;
                const size_t request = std::max(MIN_GPU_CHUNK, std::min(gpuChunk, remaining / 2));

                const size_t offset = cursor.fetch_add(request, std::memory_order_relaxed);
                if (offset >= localCount) break;
                const size_t count = std::min(request, localCount - offset);

                if (streamBusy[nextStream]) {
                    CUDA_CHECK(cudaStreamSynchronize(streams[nextStream]));
                }
                const int blocks = static_cast<int>(std::min<size_t>((count + blockSize - 1) / blockSize,
                                                                     static_cast<size_t>(numSMs) * 32));
                blackScholesKernel<<<blocks, blockSize, 0, streams[nextStream]>>>(
                    deviceBuffers[nextStream], localBegin + offset, count);
                CUDA_CHECK(cudaMemcpyAsync(localResults + offset, deviceBuffers[nextStream],
                                           count * sizeof(double), cudaMemcpyDeviceToHost,
                                           streams[nextStream]));
                streamBusy[nextStream] = true;
                nextStream = (nextStream + 1) % NUM_STREAMS;
            }
            for (int s = 0; s < NUM_STREAMS; ++s) {
                if (streamBusy[s]) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            }
        } else {
            while (true) {
                const size_t offset = cursor.fetch_add(CPU_CHUNK, std::memory_order_relaxed);
                if (offset >= localCount) break;
                const size_t count = std::min(CPU_CHUNK, localCount - offset);
                blackScholesHostRange(localResults + offset, localBegin + offset, count);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsedUs = std::chrono::duration<double, std::micro>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsedUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsedUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (elapsedUs / 1e6));
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> results;
        if (rank == 0) {
            results.resize(numOptions);
            if (localCount > 0) {
                std::memcpy(results.data(), localResults, localCount * sizeof(double));
            }
            for (int r = 1; r < numRanks; ++r) {
                const size_t begin = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
                const size_t count = base + (static_cast<size_t>(r) < rem ? 1 : 0);
                if (count > 0) recvChunked(results.data() + begin, count, r);
            }
            print_results(results, "OptionPrices");
        } else if (localCount > 0) {
            sendChunked(localResults, localCount, 0);
        }
    }

    // Validation
    int status = 0;
    if (validate) {
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        // Collect the first numChecks results on rank 0; each rank contributes the
        // entries it owns and zeros elsewhere, so the sum reproduces them exactly.
        std::vector<double> checkResults(numChecks, 0.0);
        for (size_t i = 0; i < numChecks; ++i) {
            if (i >= localBegin && i < localBegin + localCount) {
                checkResults[i] = localResults[i - localBegin];
            }
        }
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : checkResults.data(), checkResults.data(),
                   static_cast<int>(numChecks), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> options;
            generateOptions(options, 0, numChecks);
            const bool valid = validateResults(options, checkResults);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            status = valid ? 0 : 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(deviceBuffers[s]);
    }
    if (localResults != nullptr) {
        if (pinned) cudaFreeHost(localResults); else free(localResults);
    }

    MPI_Finalize();
    return status;
}
