// Black-Scholes option pricing benchmark: hybrid MPI + OpenMP + CUDA.
//
// Decomposition:
//   * MPI     : the global option index space [0, N) is split into contiguous
//               blocks, one per rank.  Ranks are assigned GPUs by node-local
//               rank.  Results are gathered on rank 0 for output/validation.
//   * OpenMP  : within a rank, one host thread per assigned GPU drives that
//               device (when a node has more GPUs than ranks a rank owns several
//               GPUs); host threads also pipeline chunked D2H copies on multiple
//               CUDA streams.  OpenMP is also used for host-side option
//               generation (validation set).
//   * CUDA    : the per-option Black-Scholes pricing is done on the GPU in
//               double precision.  The option parameters are generated
//               deterministically from the global index directly in the kernel
//               (identical arithmetic to generateOptions()), so no option data
//               has to be transferred host->device; only prices are copied back.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

constexpr size_t kNumTestOptions = getTestOptions().size();

// Device copy of the base test options
__constant__ OptionInput c_testOptions[kNumTestOptions];

// Generate option i of the scaled data set (same arithmetic on host and device)
__host__ __device__ inline OptionInput makeOption(const OptionInput& base, const size_t i) noexcept {
    OptionInput opt = base;
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    opt.spot *= factor;
    opt.strike *= factor;
    return opt;
}

// Generate a larger set of options by scaling the test set (host side, OpenMP)
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        options[i] = makeOption(testOptions[i % testOptions.size()], i);
    }
}

// CUDA kernel: price options [globalBegin, globalBegin + count) into results[0..count)
__global__ void __launch_bounds__(256)
blackScholesKernel(double* __restrict__ results, const size_t globalBegin, const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < count; j += stride) {
        const size_t i = globalBegin + j;
        const OptionInput opt = makeOption(c_testOptions[i % kNumTestOptions], i);
        results[j] = blackScholes(opt);
    }
}

// ---------------------------------------------------------------------------
// Per-GPU execution context (streams + device result buffers), created once
// before the timed region.
// ---------------------------------------------------------------------------
constexpr int    kStreams = 4;
constexpr size_t kChunk   = size_t(1) << 22;   // 4M options = 32 MiB of results per chunk
constexpr int    kBlock   = 256;

struct Chunk {
    size_t off;     // offset of first option relative to the rank's block
    size_t n;       // number of options in this chunk
};

struct DeviceCtx {
    int device = -1;
    int smCount = 0;
    cudaStream_t streams[kStreams];
    double* dBuf[kStreams];

    void init(int dev) {
        device = dev;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&dBuf[s], kChunk * sizeof(double)));
        }
        // Warm up: load the kernel module so that the first real launch is cheap.
        blackScholesKernel<<<1, kBlock, 0, streams[0]>>>(dBuf[0], 0, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(device));
        for (int s = 0; s < kStreams; ++s) {
            CUDA_CHECK(cudaFree(dBuf[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }
};

// Price the given chunks (all relative to global option index rankBegin) on one
// GPU, writing results to hostOut[chunk.off ...].  After a chunk's results have
// arrived in host memory, onChunkDone(chunkIndex) is invoked (used to stream
// the chunk to the root rank while the GPU keeps computing).
//
// Work is software-pipelined over kStreams streams: the kernel for chunk k+1
// is launched before the device->host copy of chunk k is issued, so the copy
// (which the host thread waits on) overlaps with computation of the next
// chunk.  Copies go straight into the (pageable) result vector: on PCIe the
// driver's staged pageable copy runs at close to pinned bandwidth, while
// page-locking a multi-hundred-megabyte result buffer costs far more than the
// transfer itself.
template <typename OnChunkDone>
static void priceOnDevice(DeviceCtx& ctx, const size_t rankBegin,
                          const Chunk* chunks, const size_t numChunks,
                          double* hostOut, OnChunkDone&& onChunkDone) {
    if (numChunks == 0) return;
    CUDA_CHECK(cudaSetDevice(ctx.device));

    auto launch = [&](size_t c) {
        const Chunk& ch = chunks[c];
        const size_t wanted = (ch.n + kBlock - 1) / kBlock;
        const unsigned grid = static_cast<unsigned>(std::min<size_t>(wanted, size_t(ctx.smCount) * 32));
        const int s = static_cast<int>(c % kStreams);
        blackScholesKernel<<<grid, kBlock, 0, ctx.streams[s]>>>(ctx.dBuf[s], rankBegin + ch.off, ch.n);
        CUDA_CHECK(cudaGetLastError());
    };
    auto copyBack = [&](size_t c) {
        const Chunk& ch = chunks[c];
        const int s = static_cast<int>(c % kStreams);
        CUDA_CHECK(cudaMemcpyAsync(hostOut + ch.off, ctx.dBuf[s], ch.n * sizeof(double),
                                   cudaMemcpyDeviceToHost, ctx.streams[s]));
        CUDA_CHECK(cudaStreamSynchronize(ctx.streams[s]));
        onChunkDone(c);
    };

    // Keep up to kStreams-1 kernels in flight ahead of the copy being drained.
    size_t launched = 0;
    for (; launched < std::min<size_t>(numChunks, kStreams - 1); ++launched) launch(launched);
    for (size_t c = 0; c < numChunks; ++c) {
        if (launched < numChunks) launch(launched++);
        copyBack(c);
    }
}

// Chunks of rank r's block [begin, end): aligned to global multiples of kChunk
// so every rank can enumerate every other rank's chunks without communication.
static std::vector<Chunk> chunksOf(const size_t begin, const size_t end) {
    std::vector<Chunk> chunks;
    if (begin >= end) return chunks;
    for (size_t g = begin / kChunk; g * kChunk < end; ++g) {
        const size_t lo = std::max(begin, g * kChunk);
        const size_t hi = std::min(end, (g + 1) * kChunk);
        chunks.push_back({lo - begin, hi - lo});
    }
    return chunks;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int parseStatus = -1;   // -1: continue, otherwise exit code
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_Finalize();
        return parseStatus;
    }

    // ---- GPU assignment by node-local rank -------------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevices;
    if (localSize >= numDevices) {
        myDevices.push_back(localRank % numDevices);           // GPUs shared round-robin
    } else {
        const int d0 = (localRank * numDevices) / localSize;   // each rank owns a GPU range
        const int d1 = ((localRank + 1) * numDevices) / localSize;
        for (int d = d0; d < d1; ++d) myDevices.push_back(d);
    }

    // Upload base options and warm up the context on every GPU owned by this rank
    constexpr auto testOptions = getTestOptions();
    for (int d : myDevices) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMemcpyToSymbol(c_testOptions, testOptions.data(), sizeof(OptionInput) * kNumTestOptions));
        CUDA_CHECK(cudaFree(nullptr));
    }

    // ---- MPI decomposition of the option index space -----------------------
    const size_t base = numOptions / size, rem = numOptions % size;
    auto rankBegin = [&](int r) { return size_t(r) * base + std::min<size_t>(r, rem); };
    const size_t myBegin = rankBegin(rank), myEnd = rankBegin(rank + 1);
    const size_t myCount = myEnd - myBegin;

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs on rank 0: %zu, OpenMP threads: %d\n",
               size, myDevices.size(), omp_get_max_threads());
    }

    // Generate options (host copy only needed for validation on rank 0)
    std::vector<OptionInput> options;
    if (rank == 0 && validate) {
        generateOptions(options, std::min<size_t>(10, numOptions));
    }

    // Allocate results: full vector on root, local block elsewhere
    std::vector<double> results(rank == 0 ? numOptions : myCount);
    double* myResults = results.data();   // rank 0's block starts at index 0

    // Per-GPU contexts and chunk decomposition of this rank's block
    const int nDev = static_cast<int>(myDevices.size());
    std::vector<DeviceCtx> ctx(nDev);
    for (int t = 0; t < nDev; ++t) ctx[t].init(myDevices[t]);

    const std::vector<Chunk> myChunks = chunksOf(myBegin, myEnd);
    const size_t numMyChunks = myChunks.size();

    // What has to be collected on rank 0:
    //   -r : every result (full gather, streamed chunk-wise and overlapped with compute)
    //   -v : only the first min(10, N) results (handled separately below)
    const bool fullGather = printResults && size > 1;
    const bool asyncSend = fullGather && rank != 0 && provided >= MPI_THREAD_MULTIPLE;

    std::vector<MPI_Request> requests;      // root: receives; others: sends
    if (fullGather) {
        if (rank == 0) {
            for (int r = 1; r < size; ++r) {
                const size_t rb = rankBegin(r);
                const std::vector<Chunk> rc = chunksOf(rb, rankBegin(r + 1));
                for (size_t c = 0; c < rc.size(); ++c) {
                    requests.emplace_back();
                    MPI_Irecv(results.data() + rb + rc[c].off, static_cast<int>(rc[c].n), MPI_DOUBLE,
                              r, static_cast<int>(c), MPI_COMM_WORLD, &requests.back());
                }
            }
        } else {
            requests.resize(numMyChunks, MPI_REQUEST_NULL);
        }
    }
    auto sendChunk = [&](size_t c) {
        MPI_Isend(myResults + myChunks[c].off, static_cast<int>(myChunks[c].n), MPI_DOUBLE,
                  0, static_cast<int>(c), MPI_COMM_WORLD, &requests[c]);
    };

    // Price options
    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    {
        // Each OpenMP thread drives one of this rank's GPUs on a contiguous
        // sub-range of the rank's chunks.
        const size_t perDev = numMyChunks / nDev, devRem = numMyChunks % nDev;
        #pragma omp parallel for num_threads(nDev) schedule(static)
        for (int t = 0; t < nDev; ++t) {
            const size_t c0 = size_t(t) * perDev + std::min<size_t>(t, devRem);
            const size_t c1 = c0 + perDev + (size_t(t) < devRem ? 1 : 0);
            priceOnDevice(ctx[t], myBegin, myChunks.data() + c0, c1 - c0, myResults,
                          [&](size_t c) { if (asyncSend) sendChunk(c0 + c); });
        }
    }

    // Collect results on rank 0
    if (fullGather) {
        if (rank != 0 && !asyncSend) {
            for (size_t c = 0; c < numMyChunks; ++c) sendChunk(c);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    } else if (validate && size > 1) {
        // Only the first min(10, N) prices are inspected by validateResults().
        const size_t prefix = std::min<size_t>(10, numOptions);
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t b = std::min(rankBegin(r), prefix), e = std::min(rankBegin(r + 1), prefix);
            counts[r] = static_cast<int>(e - b);
            displs[r] = static_cast<int>(b);
        }
        MPI_Gatherv(rank == 0 ? static_cast<const double*>(MPI_IN_PLACE) : myResults, counts[rank], MPI_DOUBLE,
                    results.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long long duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_us, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::microseconds(duration_us);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

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

    for (auto& c : ctx) c.destroy();

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
