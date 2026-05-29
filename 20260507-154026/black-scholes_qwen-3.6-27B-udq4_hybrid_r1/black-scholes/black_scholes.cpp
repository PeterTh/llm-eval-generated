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

// Packed struct for device transfer (POD)
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

// ---------------------------------------------------------------------------
// CUDA device functions
// ---------------------------------------------------------------------------

__device__ __forceinline__ double device_erf(double x) {
    // Abramowitz & Stegun 7.1.26 approximation (|error| < 1.5e-7)
    const double sign = (x < 0.0) ? -1.0 : 1.0;
    const double ax   = fabs(x);
    const double t    = 1.0 / (1.0 + 0.3275911 * ax);
    const double poly = (((((1.061405429 * t - 1.453132008) * t)
                           + 1.421413741) * t
                           - 0.284496736) * t
                           + 0.254829592) * t;
    return sign * (1.0 - poly * exp(-ax * ax));
}

__device__ __forceinline__ double device_cumNormal(double x) {
    return 0.5 * (1.0 + device_erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double device_blackScholes(const OptionInput& opt) {
    const double S     = opt.spot;
    const double K     = opt.strike;
    const double r     = opt.r;
    const double q     = opt.q;
    const double T     = opt.t;
    const double sigma = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sigmaSqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T)
                     / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;

    const double Nd1      = device_cumNormal(d1);
    const double Nd2      = device_cumNormal(d2);
    const double discount = exp(-r * T);
    const double fwd      = S * exp(-q * T);

    if (opt.type == CALL) {
        return fwd * Nd1 - K * discount * Nd2;
    } else {
        return K * discount * device_cumNormal(-d2)
             - fwd * device_cumNormal(-d1);
    }
}

// One thread per option
__global__ void pricingKernel(const OptionInput* d_options,
                               double* __restrict__ d_results,
                               size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_results[idx] = device_blackScholes(d_options[idx]);
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

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

// OpenMP-parallelised option generation (global index aware)
void generateOptions(std::vector<OptionInput>& options,
                     const size_t numOptions,
                     const size_t globalOffset) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t g = globalOffset + i;
        const OptionInput& base = testOptions[g % testOptions.size()];
        options[i] = base;
        const double factor = 1.0
                           + 0.1 * (g / static_cast<double>(testOptions.size()));
        options[i].spot   *= factor;
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
        const double error    = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);

        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, computed, expected, relError);

        if (computed < 0.0 || computed > 1000.0
         || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n",
                   i, computed);
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

// ---------------------------------------------------------------------------
// Main – hybrid MPI + CUDA + OpenMP
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Initialise MPI
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // 2. Parse CLI
    size_t numOptions     = 10000;
    bool   validate       = false;
    bool   printResults   = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // 3. Select GPU for this MPI rank
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int my_device = std::min(mpi_rank, std::max(0, device_count - 1));
    cudaSetDevice(my_device);

    int gpu_device = -1;
    cudaGetDevice(&gpu_device);

    if (mpi_rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Hybrid: MPI ranks=%d | OpenMP threads=%d | CUDA GPUs=%d\n",
               mpi_size, omp_get_max_threads(), device_count);
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // 4. MPI distribution – each rank gets a contiguous chunk
    const size_t local_n     = (numOptions + mpi_size - 1) / mpi_size; // ceil
    const size_t local_start = mpi_rank * local_n;
    const size_t work_n      = (local_start >= numOptions) ? 0
                                 : std::min(local_n, numOptions - local_start);

    // 5. Generate local options (OpenMP parallelised)
    std::vector<OptionInput> local_options;
    generateOptions(local_options, work_n, local_start);

    // 6. CUDA pricing
    std::vector<double> local_results(work_n);

    auto start = std::chrono::high_resolution_clock::now();

    if (work_n > 0) {
        OptionInput* d_opts = nullptr;
        double*      d_res  = nullptr;

        cudaMalloc(&d_opts, work_n * sizeof(OptionInput));
        cudaMalloc(&d_res,  work_n * sizeof(double));

        cudaMemcpy(d_opts, local_options.data(),
                   work_n * sizeof(OptionInput), cudaMemcpyHostToDevice);

        const int threads = 256;
        const int blocks  = (static_cast<int>(work_n) + threads - 1) / threads;
        pricingKernel<<<blocks, threads>>>(d_opts, d_res, work_n);
        cudaDeviceSynchronize();

        cudaMemcpy(local_results.data(), d_res,
                   work_n * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_opts);
        cudaFree(d_res);
    }

    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 7. Gather all results to rank 0
    std::vector<double>       all_results;
    std::vector<OptionInput>  all_options;

    if (mpi_rank == 0) {
        all_results.resize(numOptions);
        all_options.resize(numOptions);
    }

    std::vector<int> recvcounts(mpi_size);
    MPI_Gather(&work_n, 1, MPI_INT,
               recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> displs(mpi_size, 0);
    for (int r = 1; r < mpi_size; ++r) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    // Gather results
    MPI_Gatherv(local_results.data(), static_cast<int>(work_n), MPI_DOUBLE,
                all_results.data(),
                recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Gather options (for validation) – use byte-scaled counts/displs
    {
        const int opt_bytes = static_cast<int>(sizeof(OptionInput));
        std::vector<int> opt_recvcounts(mpi_size);
        std::vector<int> opt_displs(mpi_size, 0);
        for (int r = 0; r < mpi_size; ++r) {
            opt_recvcounts[r] = recvcounts[r] * opt_bytes;
            if (r > 0) opt_displs[r] = opt_displs[r - 1] + opt_recvcounts[r - 1];
        }
        MPI_Gatherv(local_options.data(),
                    static_cast<int>(work_n) * opt_bytes, MPI_BYTE,
                    reinterpret_cast<void*>(all_options.data()),
                    opt_recvcounts.data(), opt_displs.data(),
                    MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // 8. Timing & results (rank 0 only)
    if (mpi_rank == 0) {
        const double total_ms = duration.count() / 1000.0;
        printf("Computation time: %.3f ms\n", total_ms);
        printf("Options per second: %.0f\n",
               numOptions / (total_ms / 1000.0));

        if (printResults) {
            print_results(all_results, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(all_options, all_results);
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
