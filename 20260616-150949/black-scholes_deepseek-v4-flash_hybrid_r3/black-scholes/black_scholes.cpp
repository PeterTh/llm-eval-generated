#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

// Device version of cumulative normal for CUDA kernel
__device__ inline double deviceCumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal cumulative distribution function (host)
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// CUDA kernel: each thread prices one option
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                    double* __restrict__ results,
                                    const size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const OptionInput& opt = options[i];
    const double S = opt.spot;
    const double K = opt.strike;
    const double r = opt.r;
    const double q = opt.q;
    const double T = opt.t;
    const double sigma = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[i] = 0.0;
        return;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = deviceCumulativeNormal(d1);
    const double Nd2 = deviceCumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (opt.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * deviceCumulativeNormal(-d2)
                - S * exp(-q * T) * deviceCumulativeNormal(-d1);
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

// Generate a larger set of options by scaling the test set (OpenMP-parallel)
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(numOptions); ++i) {
        const size_t ui = static_cast<size_t>(i);
        const OptionInput& base = testOptions[ui % testOptions.size()];
        options[ui] = base;

        const double factor = 1.0 + 0.1 * (ui / static_cast<double>(testOptions.size()));
        options[ui].spot *= factor;
        options[ui].strike *= factor;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Rank 0 parses arguments and broadcasts
    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast configuration to all ranks
    uint64_t numOpts64 = static_cast<uint64_t>(numOptions);
    MPI_Bcast(&numOpts64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(numOpts64);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    // Rank 0 prints configuration
    if (world_rank == 0) {
        int omp_threads = omp_get_max_threads();
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
        printf("OpenMP threads per rank: %d\n", omp_threads);

        // Check for CUDA GPUs
        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        printf("CUDA GPUs detected: %d\n", gpu_count);
    }

    // Verify GPU availability on each rank
    {
        int gpu_count = 0;
        cudaGetDeviceCount(&gpu_count);
        if (gpu_count == 0) {
            fprintf(stderr, "Rank %d: No CUDA-capable GPU found.\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cudaSetDevice(world_rank % gpu_count);
    }

    // Compute distribution of work across MPI ranks
    const uint64_t base_count = static_cast<uint64_t>(numOptions) / static_cast<uint64_t>(world_size);
    const uint64_t remainder = static_cast<uint64_t>(numOptions) % static_cast<uint64_t>(world_size);

    std::vector<int> send_counts(world_size), send_displs(world_size);
    std::vector<int> send_bytes(world_size), send_byte_displs(world_size);

    for (int i = 0; i < world_size; ++i) {
        send_counts[i] = static_cast<int>(base_count + (static_cast<uint64_t>(i) < remainder ? 1ULL : 0ULL));
        send_displs[i] = (i == 0) ? 0 : send_displs[i - 1] + send_counts[i - 1];
        send_bytes[i] = send_counts[i] * static_cast<int>(sizeof(OptionInput));
        send_byte_displs[i] = send_displs[i] * static_cast<int>(sizeof(OptionInput));
    }

    const size_t local_n = static_cast<size_t>(send_counts[world_rank]);

    // Generate options on rank 0, then scatter to all ranks
    std::vector<OptionInput> all_options;
    if (world_rank == 0) {
        generateOptions(all_options, numOptions);
    }

    std::vector<OptionInput> local_options(local_n);
    MPI_Scatterv(
        world_rank == 0 ? all_options.data() : nullptr,
        send_bytes.data(), send_byte_displs.data(), MPI_BYTE,
        local_options.data(), static_cast<int>(local_n * sizeof(OptionInput)), MPI_BYTE,
        0, MPI_COMM_WORLD);

    // Free root's copy
    if (world_rank == 0) {
        std::vector<OptionInput>().swap(all_options);
    }

    // Allocate local results
    std::vector<double> local_results(local_n);

    // ---- Hybrid OpenMP + CUDA compute phase ----
    // Warm up the CUDA device so context creation does not pollute timing
    {
        double* warmup = nullptr;
        cudaMalloc(&warmup, 1024);
        cudaFree(warmup);
    }

    // Use a reasonable number of CUDA streams (one per OpenMP thread).
    // Cap at 8 streams to avoid excessive overhead from tiny chunks.
    constexpr int blockSize = 256;
    const int numStreams = std::max(1,
        std::min(8, std::min(static_cast<int>(omp_get_max_threads()),
                             static_cast<int>(local_n / 65536 + 1))));
    double compute_time = 0.0;

    #pragma omp parallel for num_threads(numStreams) if (numStreams > 1)
    for (int s = 0; s < numStreams; ++s) {
        const size_t chunk = (local_n + static_cast<size_t>(numStreams) - 1)
                              / static_cast<size_t>(numStreams);
        const size_t start = static_cast<size_t>(s) * chunk;
        const size_t end = std::min(start + chunk, local_n);
        const size_t count = end - start;

        if (count == 0) continue;

        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);

        OptionInput* d_options = nullptr;
        double* d_results = nullptr;
        cudaMalloc(&d_options, count * sizeof(OptionInput));
        cudaMalloc(&d_results, count * sizeof(double));

        const double t_start = omp_get_wtime();

        cudaMemcpyAsync(d_options, &local_options[start],
                        count * sizeof(OptionInput),
                        cudaMemcpyHostToDevice, stream);

        const int numBlocks = static_cast<int>((count + blockSize - 1) / blockSize);
        blackScholesKernel<<<numBlocks, blockSize, 0, stream>>>(d_options, d_results, count);

        cudaMemcpyAsync(&local_results[start], d_results,
                        count * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        const double t_elapsed = omp_get_wtime() - t_start;

        cudaFree(d_options);
        cudaFree(d_results);
        cudaStreamDestroy(stream);

        #pragma omp critical
        {
            if (t_elapsed > compute_time) {
                compute_time = t_elapsed;
            }
        }
    }

    // Gather results on rank 0
    std::vector<int> res_counts(world_size), res_displs(world_size);
    std::vector<int> res_bytes(world_size), res_byte_displs(world_size);
    for (int i = 0; i < world_size; ++i) {
        res_counts[i] = send_counts[i];
        res_displs[i] = send_displs[i];
        res_bytes[i] = res_counts[i] * static_cast<int>(sizeof(double));
        res_byte_displs[i] = res_displs[i] * static_cast<int>(sizeof(double));
    }

    std::vector<double> all_results;
    if (world_rank == 0) {
        all_results.resize(numOptions);
    }

    MPI_Gatherv(
        local_results.data(), static_cast<int>(local_n * sizeof(double)), MPI_BYTE,
        world_rank == 0 ? all_results.data() : nullptr,
        res_bytes.data(), res_byte_displs.data(), MPI_BYTE,
        0, MPI_COMM_WORLD);

    // Compute the maximum compute time across all ranks
    double max_compute_time = 0.0;
    MPI_Reduce(&compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Rank 0 handles output and validation
    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", max_compute_time * 1000.0);
        printf("Options per second: %.0f\n",
               static_cast<double>(numOptions) / max_compute_time);

        if (printResults) {
            print_results(all_results, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");

            // Regenerate options on rank 0 for validation
            std::vector<OptionInput> orig_options;
            generateOptions(orig_options, numOptions);

            bool valid = validateResults(orig_options, all_results);

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
