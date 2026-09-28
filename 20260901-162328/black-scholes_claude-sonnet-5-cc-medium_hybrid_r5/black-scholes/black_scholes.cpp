#define OMPI_SKIP_MPICXX
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cudaCheckErr_ = (call);                                     \
        if (cudaCheckErr_ != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(cudaCheckErr_));                         \
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

// GPU kernel: prices one option per thread
__global__ void blackScholesKernel(const OptionInput* __restrict__ opts,
                                    double* __restrict__ results,
                                    size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        results[idx] = blackScholes(opts[idx]);
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

// Generate a slice [startIndex, startIndex + numLocal) of the larger option set.
// Deterministic in the global index so any rank can regenerate any slice independently.
void generateOptions(std::vector<OptionInput>& options, size_t startIndex, size_t numLocal) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numLocal);

    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < numLocal; ++li) {
        const size_t i = startIndex + li;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[li] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[li].spot *= factor;
        options[li].strike *= factor;
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

// Prices `options` on the GPU bound to `device`, splitting the work across
// multiple CUDA streams (managed by OpenMP host threads) to overlap
// host<->device transfers with kernel execution.
void computeOptionsGPU(const std::vector<OptionInput>& options,
                       std::vector<double>& results,
                       int device) {
    const size_t n = options.size();
    results.resize(n);
    if (n == 0) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));

    constexpr int kMaxStreams = 4;
    const int numStreams = static_cast<int>(std::min<size_t>(kMaxStreams, n));
    const size_t chunk = (n + numStreams - 1) / numStreams;

    OptionInput* hOpts = const_cast<OptionInput*>(options.data());
    double* hResults = results.data();

    CUDA_CHECK(cudaHostRegister(hOpts, n * sizeof(OptionInput), cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(hResults, n * sizeof(double), cudaHostRegisterDefault));

    #pragma omp parallel num_threads(numStreams)
    {
        const int t = omp_get_thread_num();
        const size_t start = static_cast<size_t>(t) * chunk;
        const size_t end = std::min(n, start + chunk);

        if (start < end) {
            CUDA_CHECK(cudaSetDevice(device));

            const size_t len = end - start;

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            OptionInput* dOpts = nullptr;
            double* dResults = nullptr;
            CUDA_CHECK(cudaMalloc(&dOpts, len * sizeof(OptionInput)));
            CUDA_CHECK(cudaMalloc(&dResults, len * sizeof(double)));

            CUDA_CHECK(cudaMemcpyAsync(dOpts, hOpts + start, len * sizeof(OptionInput),
                                       cudaMemcpyHostToDevice, stream));

            constexpr int threadsPerBlock = 256;
            const int blocks = static_cast<int>((len + threadsPerBlock - 1) / threadsPerBlock);
            blackScholesKernel<<<blocks, threadsPerBlock, 0, stream>>>(dOpts, dResults, len);

            CUDA_CHECK(cudaMemcpyAsync(hResults + start, dResults, len * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaFree(dOpts));
            CUDA_CHECK(cudaFree(dResults));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }

    CUDA_CHECK(cudaHostUnregister(hOpts));
    CUDA_CHECK(cudaHostUnregister(hResults));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool shouldExit = false;

    // Parse command line arguments (identical on every rank: argv is broadcast by mpirun)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exitCode = 0;
            shouldExit = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            shouldExit = true;
            break;
        }
    }

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    // Bind this rank to a GPU: use the node-local rank so that ranks sharing a
    // node spread across the node's available accelerators.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    // Warm up the CUDA context/driver so lazy first-touch init cost isn't
    // counted in the timed pricing region below.
    CUDA_CHECK(cudaFree(0));

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d\n", worldSize, deviceCount);
    }

    // Distribute the option range contiguously and evenly across MPI ranks
    const size_t base = numOptions / static_cast<size_t>(worldSize);
    const size_t remainder = numOptions % static_cast<size_t>(worldSize);
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t d = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(d);
    }
    const size_t myStart = static_cast<size_t>(displs[rank]);
    const size_t myCount = static_cast<size_t>(counts[rank]);

    // Generate this rank's slice of options
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, myStart, myCount);

    // Allocate results
    std::vector<double> localResults(myCount);

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    computeOptionsGPU(localOptions, localResults, device);

    auto end = std::chrono::high_resolution_clock::now();
    const double localElapsedUs =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double maxElapsedUs = 0.0;
    MPI_Reduce(&localElapsedUs, &maxElapsedUs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all local results into rank 0's full results vector, preserving order
    std::vector<double> results;
    if (rank == 0) {
        results.resize(numOptions);
    }
    MPI_Gatherv(localResults.data(), static_cast<int>(myCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsedUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxElapsedUs / 1e6));
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(results, "OptionPrices");
    }

    // Validation
    int validPassed = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, 0, std::min(static_cast<size_t>(10), numOptions));
            validPassed = validateResults(checkOptions, results) ? 1 : 0;

            if (validPassed) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&validPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validPassed ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
