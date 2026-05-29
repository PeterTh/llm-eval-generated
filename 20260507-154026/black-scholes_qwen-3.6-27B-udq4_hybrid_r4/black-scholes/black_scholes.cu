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

// CUDA-compatible struct (POD, no padding issues)
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

// ─── CUDA device functions ───────────────────────────────────────────

// CUDA device CDF — uses device erf for bit-equivalence with host code
__device__ double device_cdf(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const OptionInput* d_options,
                                    double* d_results,
                                    const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const OptionInput opt = d_options[idx];
    const double S       = opt.spot;
    const double K       = opt.strike;
    const double r       = opt.r;
    const double q       = opt.q;
    const double T       = opt.t;
    const double sigma   = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        d_results[idx] = 0.0;
        return;
    }

    const double sigma_sqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrtT;
    const double d2 = d1 - sigma_sqrtT;

    const double Nd1 = device_cdf(d1);
    const double Nd2 = device_cdf(d2);
    const double discount = exp(-r * T);

    double price;
    if (opt.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * device_cdf(-d2) - S * exp(-q * T) * device_cdf(-d1);
    }

    d_results[idx] = price;
}

// ─── Host-side helpers ───────────────────────────────────────────────

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

// Generate options – parallelized with OpenMP
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
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

// ─── Main: MPI (across nodes) → OpenMP (thread gen) → CUDA (GPU kernel) ─

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Parse args on every rank (simple, deterministic)
    size_t numOptions = 10000;
    bool validate     = false;
    bool printResults = false;

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

    // ── Print banner (rank 0 only) ────────────────────────────────────
    if (rank == 0) {
        int omp_threads = omp_get_max_threads();
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI ranks:         %d\n", numRanks);
        printf("OpenMP threads:    %d\n", omp_threads);
        printf("Validation:        %s\n", validate ? "enabled" : "disabled");
    }

    // ── Distribute work across MPI ranks ──────────────────────────────
    // Each rank gets a contiguous chunk: [local_start, local_start + local_n)
    const size_t base_n   = numOptions / numRanks;
    const size_t remainder = numOptions % numRanks;
    const size_t local_n  = base_n + (rank < static_cast<int>(remainder) ? 1 : 0);
    size_t local_start = base_n * rank + std::min(static_cast<size_t>(rank), remainder);

    // Prefix sums so every rank knows where other ranks' data starts
    std::vector<size_t> prefix_n(numRanks);
    prefix_n[0] = 0;
    for (int r = 1; r < numRanks; ++r) {
        const size_t rn = base_n + (r < static_cast<int>(remainder) ? 1 : 0);
        prefix_n[r] = prefix_n[r - 1] + rn;
    }

    // ── Generate local options (OpenMP) ───────────────────────────────
    std::vector<OptionInput> local_options;
    generateOptions(local_options, local_n);

    // Shift indices: generateOptions uses global index i; we need local_start+i
    {
        constexpr auto testOptions = getTestOptions();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local_n; ++i) {
            const size_t global_i = local_start + i;
            const OptionInput& base = testOptions[global_i % testOptions.size()];
            local_options[i] = base;
            const double factor = 1.0 + 0.1 * (global_i / static_cast<double>(testOptions.size()));
            local_options[i].spot   *= factor;
            local_options[i].strike *= factor;
        }
    }

    // ── GPU pricing via CUDA kernel ───────────────────────────────────
    if (rank == 0) printf("Pricing options...\n");
    std::vector<double> local_results(local_n);

    OptionInput* d_options = nullptr;
    double*      d_results = nullptr;

    cudaMalloc(&d_options, local_n * sizeof(OptionInput));
    cudaMalloc(&d_results, local_n * sizeof(double));

    cudaMemcpy(d_options, local_options.data(), local_n * sizeof(OptionInput), cudaMemcpyHostToDevice);

    const int block_size = 256;
    const int grid_size  = (static_cast<int>(local_n) + block_size - 1) / block_size;

    auto start = std::chrono::high_resolution_clock::now();

    blackScholesKernel<<<grid_size, block_size>>>(d_options, d_results, static_cast<int>(local_n));
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(local_results.data(), d_results, local_n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_options);
    cudaFree(d_results);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // ── Gather results to rank 0 ──────────────────────────────────────
    std::vector<double> all_results;
    std::vector<OptionInput> all_options;

    if (rank == 0) {
        all_results.resize(numOptions);
        all_options.resize(numOptions);
    }

    {
        // Gather results (doubles)
        std::vector<int> recvcounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvcounts[r] = static_cast<int>(base_n + (r < static_cast<int>(remainder) ? 1 : 0));
            displs[r]     = static_cast<int>(prefix_n[r]);
        }
        MPI_Gatherv(local_results.data(), static_cast<int>(local_n), MPI_DOUBLE,
                    rank == 0 ? all_results.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Gather option inputs as raw bytes
        {
            const int local_bytes = static_cast<int>(local_n * sizeof(OptionInput));
            std::vector<int> recv_bytes(numRanks);
            std::vector<int> disp_bytes(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rn = base_n + (r < static_cast<int>(remainder) ? 1 : 0);
                recv_bytes[r] = static_cast<int>(rn * sizeof(OptionInput));
                disp_bytes[r] = static_cast<int>(prefix_n[r] * sizeof(OptionInput));
            }
            MPI_Gatherv(local_options.data(), local_bytes, MPI_BYTE,
                        rank == 0 ? all_options.data() : nullptr,
                        recv_bytes.data(), disp_bytes.data(), MPI_BYTE,
                        0, MPI_COMM_WORLD);
        }
    }

    // ── Timing (reduce across ranks) ─────────────────────────────────
    double local_time_ms = duration.count() / 1000.0;
    double local_ops_sec = local_n / (duration.count() / 1e6);
    double total_time_ms = 0.0;
    double total_ops_sec = 0.0;
    MPI_Reduce(&local_time_ms, &total_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_ops_sec, &total_ops_sec, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", total_time_ms);
        printf("Options per second: %.0f\n", total_ops_sec);

        // Print results for external validation
        if (printResults) {
            print_results(all_results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(all_options, all_results);

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
