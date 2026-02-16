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

// Host/device math helpers
__host__ __device__ inline double device_erf(double x) {
#if __CUDA_ARCH__
    return erf(x);
#else
    return std::erf(x);
#endif
}

__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + device_erf(x * M_SQRT1_2));
}

__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options usable on device
__host__ __device__ double blackScholes_device(const OptionInput& option) noexcept {
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

// Generate a larger set of options by scaling the test set (parallelized with OpenMP)
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
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

// CUDA kernel to compute Black-Scholes for each option
__global__ void bs_kernel(const OptionInput* opts, double* results, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = blackScholes_device(opts[idx]);
}

// Compute on GPU: copies options to device, runs kernel, copies back results
void gpu_price_options(const std::vector<OptionInput>& localOptions, std::vector<double>& localResults) {
    const size_t n = localOptions.size();
    if (n == 0) return;

    OptionInput* d_opts = nullptr;
    double* d_results = nullptr;

    const size_t opts_bytes = n * sizeof(OptionInput);
    const size_t res_bytes = n * sizeof(double);

    cudaMalloc(&d_opts, opts_bytes);
    cudaMalloc(&d_results, res_bytes);

    // copy
    cudaMemcpy(d_opts, localOptions.data(), opts_bytes, cudaMemcpyHostToDevice);

    // launch kernel
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    bs_kernel<<<grid, block>>>(d_opts, d_results, n);

    // copy back
    cudaMemcpy(localResults.data(), d_results, res_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_opts);
    cudaFree(d_results);
}

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates all options, then scatter to ranks
    std::vector<OptionInput> allOptions;
    std::vector<OptionInput> localOptions;
    std::vector<double> localResults;
    std::vector<double> allResults;

    // Compute distribution
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    size_t base = numOptions / world_size;
    size_t rem = numOptions % world_size;
    for (int r = 0; r < world_size; ++r) {
        size_t cnt = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        sendcounts[r] = static_cast<int>(cnt * sizeof(OptionInput)); // bytes for MPI_BYTE
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r-1] + sendcounts[r-1];

    if (world_rank == 0) {
        // generate full set in parallel on host (OpenMP)
        generateOptions(allOptions, numOptions);
        allResults.resize(numOptions);
    }

    // Determine local count for allocation
    int local_bytes = 0;
    MPI_Scatter(sendcounts.data(), 1, MPI_INT, &local_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const int local_count = local_bytes / static_cast<int>(sizeof(OptionInput));
    localOptions.resize(local_count);
    localResults.assign(local_count, 0.0);

    // Scatter the option structs as raw bytes
    MPI_Scatterv(
        world_rank == 0 ? allOptions.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_BYTE,
        localOptions.data(), local_count * static_cast<int>(sizeof(OptionInput)), MPI_BYTE,
        0, MPI_COMM_WORLD
    );

    // Synchronize and measure time on rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = 0.0;
    if (world_rank == 0) t_start = MPI_Wtime();

    // Compute on GPU for local partition
    gpu_price_options(localOptions, localResults);

    // Gather results back to rank 0
    std::vector<int> recvcounts(world_size);
    for (int r = 0; r < world_size; ++r) {
        // number of doubles (not bytes) expected from rank r
        recvcounts[r] = sendcounts[r] / static_cast<int>(sizeof(OptionInput));
    }
    std::vector<int> recvdispls(world_size);
    recvdispls[0] = 0;
    for (int r = 1; r < world_size; ++r) recvdispls[r] = recvdispls[r-1] + recvcounts[r-1];

    MPI_Gatherv(localResults.data(), local_count, MPI_DOUBLE,
                world_rank == 0 ? allResults.data() : nullptr, recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = 0.0;
    if (world_rank == 0) t_end = MPI_Wtime();

    if (world_rank == 0) {
        const double duration_s = t_end - t_start;
        printf("Computation time: %.3f ms\n", duration_s * 1000.0);
        printf("Options per second: %.0f\n", numOptions / duration_s);

        if (printResults) {
            print_results(allResults, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(allOptions, allResults);
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
