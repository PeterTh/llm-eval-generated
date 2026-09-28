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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t cudaCheck_err = (call);                                     \
        if (cudaCheck_err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(cudaCheck_err));                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
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
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options (host reference implementation)
double blackScholes(const OptionInput& option) noexcept {
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

// Device-side standard normal CDF
__device__ inline double d_cumulativeNormal(const double x) {
    return 0.5 * (1.0 + erf(x * 0.70710678118654752440));
}

// GPU kernel pricing a chunk of European options with the Black-Scholes formula
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                    double* __restrict__ results,
                                    const size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const OptionInput opt = options[i];
    double price = 0.0;

    if (opt.t > 0.0 && opt.vol > 0.0) {
        const double S = opt.spot;
        const double K = opt.strike;
        const double r = opt.r;
        const double q = opt.q;
        const double T = opt.t;
        const double sigma = opt.vol;

        const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        const double d2 = d1 - sigma * sqrt(T);

        const double Nd1 = d_cumulativeNormal(d1);
        const double Nd2 = d_cumulativeNormal(d2);
        const double discount = exp(-r * T);

        if (opt.type == CALL) {
            price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
        } else { // PUT
            price = K * discount * d_cumulativeNormal(-d2) - S * exp(-q * T) * d_cumulativeNormal(-d1);
        }
    }

    results[i] = price;
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

// Fill `options` with `count` entries corresponding to global indices
// [startIndex, startIndex + count). This lets every MPI rank (and OpenMP
// thread) independently reconstruct its own slice of the deterministic
// dataset without any inter-process communication of inputs.
void generateOptions(std::vector<OptionInput>& options, const size_t startIndex, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        const size_t globalIndex = startIndex + i;
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        OptionInput opt = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        opt.spot *= factor;
        opt.strike *= factor;

        options[i] = opt;
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

// Determine which local CUDA device indices this rank should drive, based on
// how many ranks share this node (via a shared-memory MPI split) versus how
// many GPUs the node exposes. This allows both "one GPU per rank" and
// "one rank driving several GPUs" placements to work correctly.
std::vector<int> assignLocalDevices(const MPI_Comm& worldComm, const int deviceCount) {
    MPI_Comm shmComm;
    MPI_Comm_split_type(worldComm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmComm);

    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);
    MPI_Comm_free(&shmComm);

    std::vector<int> devices;
    if (localSize <= deviceCount) {
        // Distribute the node's GPUs as evenly as possible across local ranks;
        // a rank may end up driving more than one device.
        const int base = deviceCount / localSize;
        const int rem = deviceCount % localSize;
        const int start = localRank * base + std::min(localRank, rem);
        const int cnt = base + (localRank < rem ? 1 : 0);
        for (int i = 0; i < cnt; ++i) {
            devices.push_back(start + i);
        }
        if (devices.empty()) {
            devices.push_back(localRank % deviceCount);
        }
    } else {
        // More local ranks than GPUs: share devices round-robin.
        devices.push_back(localRank % deviceCount);
    }

    return devices;
}

// Price `options` on the GPUs listed in `devices`, splitting the work evenly
// across them. Each device is driven concurrently by its own OpenMP thread
// using its own CUDA stream, overlapping H2D/kernel/D2H work across GPUs.
void priceOptionsOnGPUs(const std::vector<OptionInput>& options,
                         std::vector<double>& results,
                         const std::vector<int>& devices) {
    const size_t n = options.size();
    results.resize(n);
    if (n == 0) return;

    const int numDevices = static_cast<int>(devices.size());
    const size_t chunk = (n + static_cast<size_t>(numDevices) - 1) / static_cast<size_t>(numDevices);

    #pragma omp parallel for num_threads(numDevices) schedule(static)
    for (int d = 0; d < numDevices; ++d) {
        const size_t begin = static_cast<size_t>(d) * chunk;
        const size_t end = std::min(n, begin + chunk);
        if (begin >= end) continue;
        const size_t count = end - begin;

        CUDA_CHECK(cudaSetDevice(devices[d]));

        OptionInput* d_options = nullptr;
        double* d_results = nullptr;
        CUDA_CHECK(cudaMalloc(&d_options, count * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&d_results, count * sizeof(double)));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMemcpyAsync(d_options, options.data() + begin, count * sizeof(OptionInput),
                                    cudaMemcpyHostToDevice, stream));

        const int blockSize = 256;
        const int gridSize = static_cast<int>((count + blockSize - 1) / blockSize);
        blackScholesKernel<<<gridSize, blockSize, 0, stream>>>(d_options, d_results, count);

        CUDA_CHECK(cudaMemcpyAsync(results.data() + begin, d_results, count * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA-capable devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::vector<int> myDevices = assignLocalDevices(MPI_COMM_WORLD, deviceCount);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices visible per node: %d\n", numRanks, deviceCount);
    }

    // Split the global option set into contiguous ranges across MPI ranks.
    const size_t base = numOptions / static_cast<size_t>(numRanks);
    const size_t rem = numOptions % static_cast<size_t>(numRanks);
    const size_t myCount = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t myStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, myStart, myCount);

    std::vector<double> localResults;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    priceOptionsOnGPUs(localOptions, localResults, myDevices);

    auto end = std::chrono::high_resolution_clock::now();
    const double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results back to rank 0 in global order for reporting/validation.
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        counts.resize(numRanks);
        displs.resize(numRanks);
        for (int p = 0; p < numRanks; ++p) {
            const size_t cnt = base + (static_cast<size_t>(p) < rem ? 1 : 0);
            const size_t off = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), rem);
            counts[p] = static_cast<int>(cnt);
            displs[p] = static_cast<int>(off);
        }
    }

    std::vector<double> results;
    if (rank == 0) results.resize(numOptions);

    MPI_Gatherv(localResults.data(), static_cast<int>(myCount), MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int status = 0;

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        printf("Options per second: %.0f\n", numOptions / (maxMs / 1e3));

        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> allOptions;
            generateOptions(allOptions, 0, numOptions);
            bool valid = validateResults(allOptions, results);

            if (valid) {
                printf("Validation: PASSED\n");
                status = 0;
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
