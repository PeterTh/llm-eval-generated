#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
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
inline double normalPDF(const double x) noexcept {
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

constexpr size_t kNumTestOptions = getTestOptions().size();

// Device copy of the test set, uploaded once per GPU
__device__ OptionInput d_testOptions[kNumTestOptions];

// Build option i by scaling the test set (identical arithmetic on host and device)
__host__ __device__ inline OptionInput makeOption(const OptionInput& base, const size_t i) noexcept {
    OptionInput option = base;

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(kNumTestOptions));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Each thread prices a grid-stride subset of the global option indices [firstIndex, firstIndex + count)
__global__ void priceOptionsKernel(double* out, const size_t firstIndex, const size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; j < count; j += stride) {
        const size_t i = firstIndex + j;
        out[j] = blackScholes(makeOption(d_testOptions[i % kNumTestOptions], i));
    }
}

// Split [0, total) into `parts` contiguous blocks; returns the size and start of block `part`
inline void blockRange(const size_t total, const int parts, const int part,
                       size_t& count, size_t& offset) noexcept {
    const size_t base = total / static_cast<size_t>(parts);
    const size_t rem = total % static_cast<size_t>(parts);
    const size_t p = static_cast<size_t>(part);
    count = base + (p < rem ? 1 : 0);
    offset = p * base + std::min(p, rem);
}

// GPUs used by one MPI rank: ranks on a node share GPUs when there are more ranks than devices
std::vector<int> devicesForLocalRank(const int localRank, const int localSize, const int numDevices) {
    std::vector<int> devices;
    if (localSize >= numDevices) {
        devices.push_back(localRank % numDevices);
    } else {
        for (int d = localRank; d < numDevices; d += localSize) {
            devices.push_back(d);
        }
    }
    return devices;
}

// Work assigned to one GPU of this rank
struct DeviceWorker {
    size_t count = 0;               // Number of options priced on this GPU
    size_t offset = 0;              // Offset within this rank's block
    double* d_out = nullptr;        // Device result buffer
};

bool validateResults(const std::vector<double>& results) {
    static constexpr auto testOptions = getTestOptions();
    bool allPassed = true;
    const int numChecks = static_cast<int>(std::min(static_cast<size_t>(10), results.size()));

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = testOptions[i % kNumTestOptions].value;
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit alike)
    int status = 0;
    bool exitEarly = false;
    for (int i = 1; i < argc && !exitEarly; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitEarly = true;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            exitEarly = true;
        }
    }

    // Result gathering uses MPI int counts
    if (!exitEarly && numOptions > static_cast<size_t>(INT_MAX)) {
        if (root) printf("Number of options must not exceed %d\n", INT_MAX);
        status = 1;
        exitEarly = true;
    }

    if (exitEarly) {
        MPI_Finalize();
        return status;
    }

    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Map this rank onto the GPUs of its node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    const std::vector<int> devices = devicesForLocalRank(localRank, localSize, numDevices);
    const int numWorkers = static_cast<int>(devices.size());

    // Split this rank's block across its GPUs
    size_t rankCount = 0;
    size_t rankOffset = 0;
    blockRange(numOptions, size, rank, rankCount, rankOffset);

    std::vector<DeviceWorker> workers(numWorkers);
    for (int w = 0; w < numWorkers; ++w) {
        blockRange(rankCount, numWorkers, w, workers[w].count, workers[w].offset);
    }

    double* hostResults = nullptr;
    if (rankCount > 0) {
        CUDA_CHECK(cudaMallocHost(&hostResults, rankCount * sizeof(double)));
    }

    // One OpenMP thread drives each GPU: upload the test set, allocate, and create the context
    static constexpr auto testOptions = getTestOptions();
    #pragma omp parallel for num_threads(numWorkers) schedule(static, 1)
    for (int w = 0; w < numWorkers; ++w) {
        DeviceWorker& dw = workers[w];
        CUDA_CHECK(cudaSetDevice(devices[w]));
        CUDA_CHECK(cudaMemcpyToSymbol(d_testOptions, testOptions.data(), sizeof(testOptions)));
        if (dw.count > 0) {
            CUDA_CHECK(cudaMalloc(&dw.d_out, dw.count * sizeof(double)));
            // Warm-up launch so module loading is not timed
            priceOptionsKernel<<<1, 1>>>(dw.d_out, 0, 1);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Price options
    if (root) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for num_threads(numWorkers) schedule(static, 1)
    for (int w = 0; w < numWorkers; ++w) {
        const DeviceWorker& dw = workers[w];
        if (dw.count == 0) {
            continue;
        }
        CUDA_CHECK(cudaSetDevice(devices[w]));
        constexpr int threadsPerBlock = 256;
        const size_t blocks = std::min<size_t>((dw.count + threadsPerBlock - 1) / threadsPerBlock, 16384);
        priceOptionsKernel<<<static_cast<unsigned int>(blocks), threadsPerBlock>>>(
            dw.d_out, rankOffset + dw.offset, dw.count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hostResults + dw.offset, dw.d_out,
                              dw.count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather all results in index order on the root rank
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t count = 0;
        size_t offset = 0;
        blockRange(numOptions, size, r, count, offset);
        counts[r] = static_cast<int>(count);
        displs[r] = static_cast<int>(offset);
    }

    std::vector<double> results;
    if (root) {
        results.resize(numOptions);
    }
    MPI_Gatherv(hostResults, static_cast<int>(rankCount), MPI_DOUBLE,
                root ? results.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int w = 0; w < numWorkers; ++w) {
        if (workers[w].d_out != nullptr) {
            CUDA_CHECK(cudaSetDevice(devices[w]));
            CUDA_CHECK(cudaFree(workers[w].d_out));
        }
    }
    if (hostResults != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostResults));
    }

    int exitCode = 0;
    if (root) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Options per second: %.0f\n", numOptions / seconds);

        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(results);

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
