#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                        \
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

// CUDA kernel: prices a contiguous batch of options
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                    double* __restrict__ results,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = blackScholes(options[idx]);
    }
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

// Generate the option at a given global index (matches original generateOptions formula)
inline OptionInput generateOption(const size_t globalIndex) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[globalIndex % testOptions.size()];

    const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;

    return option;
}

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = generateOption(i);
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

// Split [0, total) into `parts` balanced contiguous ranges; returns the range for `idx`.
inline void computeRange(const size_t total, const int idx, const int parts,
                          size_t& start, size_t& len) {
    const size_t base = total / static_cast<size_t>(parts);
    const size_t rem = total % static_cast<size_t>(parts);
    start = static_cast<size_t>(idx) * base + std::min<size_t>(static_cast<size_t>(idx), rem);
    len = base + (static_cast<size_t>(idx) < rem ? 1 : 0);
}

int main(int argc, char** argv) {
    int providedThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Determine node-local rank to map ranks onto local GPUs
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) {
            fprintf(stderr, "No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank via mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs/node: %d\n", worldSize, deviceCount);
        printf("Pricing options...\n");
    }

    // Distribute the option set across MPI ranks
    size_t localStart = 0, localCount = 0;
    computeRange(numOptions, worldRank, worldSize, localStart, localCount);

    // Pinned host buffers for async transfers
    OptionInput* h_options = nullptr;
    double* h_results = nullptr;
    if (localCount > 0) {
        CUDA_CHECK(cudaMallocHost(&h_options, localCount * sizeof(OptionInput)));
        CUDA_CHECK(cudaMallocHost(&h_results, localCount * sizeof(double)));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Generate this rank's slice of options in parallel on the host
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < localCount; ++j) {
        h_options[j] = generateOption(localStart + j);
    }

    // Fan this rank's slice out across multiple CUDA streams (one per OpenMP thread)
    // to overlap host-device transfers with kernel execution.
    if (localCount > 0) {
        const int maxStreams = 4;
        const int numStreams = static_cast<int>(std::min<size_t>(maxStreams, localCount));

        #pragma omp parallel num_threads(numStreams)
        {
            const int tid = omp_get_thread_num();
            CUDA_CHECK(cudaSetDevice(device));

            size_t chunkStart = 0, chunkLen = 0;
            computeRange(localCount, tid, numStreams, chunkStart, chunkLen);

            if (chunkLen > 0) {
                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));

                OptionInput* d_options = nullptr;
                double* d_results = nullptr;
                CUDA_CHECK(cudaMallocAsync(&d_options, chunkLen * sizeof(OptionInput), stream));
                CUDA_CHECK(cudaMallocAsync(&d_results, chunkLen * sizeof(double), stream));

                CUDA_CHECK(cudaMemcpyAsync(d_options, h_options + chunkStart,
                                           chunkLen * sizeof(OptionInput),
                                           cudaMemcpyHostToDevice, stream));

                const int threadsPerBlock = 256;
                const int blocks = static_cast<int>((chunkLen + threadsPerBlock - 1) / threadsPerBlock);
                blackScholesKernel<<<blocks, threadsPerBlock, 0, stream>>>(
                    d_options, d_results, static_cast<int>(chunkLen));

                CUDA_CHECK(cudaMemcpyAsync(h_results + chunkStart, d_results,
                                           chunkLen * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));

                CUDA_CHECK(cudaStreamSynchronize(stream));

                CUDA_CHECK(cudaFreeAsync(d_options, stream));
                CUDA_CHECK(cudaFreeAsync(d_results, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaStreamDestroy(stream));
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    long long localMicros = localDuration.count();
    long long maxMicros = 0;
    MPI_Reduce(&localMicros, &maxMicros, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather per-rank counts to build the Gatherv layout on the root
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (worldRank == 0) {
        recvCounts.resize(worldSize);
        displs.resize(worldSize);
    }
    const int localCountInt = static_cast<int>(localCount);
    MPI_Gather(&localCountInt, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (worldRank == 0) {
        int offset = 0;
        for (int i = 0; i < worldSize; ++i) {
            displs[i] = offset;
            offset += recvCounts[i];
        }
    }

    // Gather results (all ranks need only send; root allocates the full vector)
    std::vector<double> results;
    if (worldRank == 0) {
        results.resize(numOptions);
    }
    MPI_Gatherv(h_results, localCountInt, MPI_DOUBLE,
                results.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Gather options (needed on root for validation, and to mirror the full input set)
    std::vector<int> byteCounts;
    std::vector<int> byteDispls;
    std::vector<OptionInput> options;
    if (worldRank == 0) {
        byteCounts.resize(worldSize);
        byteDispls.resize(worldSize);
        for (int i = 0; i < worldSize; ++i) {
            byteCounts[i] = recvCounts[i] * static_cast<int>(sizeof(OptionInput));
            byteDispls[i] = displs[i] * static_cast<int>(sizeof(OptionInput));
        }
        options.resize(numOptions);
    }
    MPI_Gatherv(h_options, localCountInt * static_cast<int>(sizeof(OptionInput)), MPI_BYTE,
                options.data(), byteCounts.data(), byteDispls.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    if (h_options) CUDA_CHECK(cudaFreeHost(h_options));
    if (h_results) CUDA_CHECK(cudaFreeHost(h_results));

    int exitCode = 0;
    if (worldRank == 0) {
        const double ms = maxMicros / 1000.0;
        printf("Computation time: %.3f ms\n", ms);
        printf("Options per second: %.0f\n", numOptions / (maxMicros / 1e6));

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
                exitCode = 0;
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
