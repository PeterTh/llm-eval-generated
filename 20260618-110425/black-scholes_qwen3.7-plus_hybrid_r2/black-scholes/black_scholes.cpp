#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
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

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;
    double strike;
    double spot;
    double q;
    double r;
    double t;
    double vol;
    double value;
    double tol;
};

__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

__global__ void __launch_bounds__(256)
blackScholesKernel(const OptionInput* __restrict__ options,
                   double* __restrict__ results,
                   const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const OptionInput opt = options[idx];
    const double S = opt.spot;
    const double K = opt.strike;
    const double r = opt.r;
    const double q = opt.q;
    const double T = opt.t;
    const double sigma = opt.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        results[idx] = 0.0;
        return;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (opt.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormal(-d2)
                - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    results[idx] = price;
}

// Host-side Black-Scholes (kept for API compatibility)
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
    } else {
        price = K * discount * cumulativeNormal(-d2)
                - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
}

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT,  100.00, 90.00,  0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(numOptions); ++i) {
        const size_t idx = static_cast<size_t>(i);
        const OptionInput& base = testOptions[idx % testOptions.size()];
        options[idx] = base;
        const double factor = 1.0 + 0.1 * (idx / static_cast<double>(testOptions.size()));
        options[idx].spot *= factor;
        options[idx].strike *= factor;
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Initialize CUDA and select GPU device (round-robin assignment)
    int numGPUs = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&numGPUs);
    if (cudaErr != cudaSuccess || numGPUs == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int gpuId = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuId));

    // Parse command line arguments (all ranks receive same args from mpirun)
    size_t numOptions = 10000;
    bool validate = false;
    bool printResultsFlag = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResultsFlag = true;
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
        printf("MPI processes: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), numGPUs);
    }

    // Compute work distribution across MPI ranks
    const size_t base_count = numOptions / static_cast<size_t>(nprocs);
    const size_t rem = numOptions % static_cast<size_t>(nprocs);
    const size_t my_count = base_count + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Build byte-level counts/displacements for MPI_Scatterv (struct transfer)
    // and option-level counts/displacements for MPI_Gatherv (double transfer)
    std::vector<int> byte_counts(nprocs), byte_displs(nprocs);
    std::vector<int> opt_counts(nprocs), opt_displs(nprocs);
    for (int i = 0; i < nprocs; i++) {
        size_t cnt = base_count + (static_cast<size_t>(i) < rem ? 1 : 0);
        opt_counts[i] = static_cast<int>(cnt);
        byte_counts[i] = static_cast<int>(cnt * sizeof(OptionInput));
        if (i == 0) {
            opt_displs[i] = 0;
            byte_displs[i] = 0;
        } else {
            opt_displs[i] = opt_displs[i - 1] + opt_counts[i - 1];
            byte_displs[i] = byte_displs[i - 1] + byte_counts[i - 1];
        }
    }

    // Generate options on rank 0 (OpenMP-parallelized)
    std::vector<OptionInput> all_options;
    if (rank == 0) {
        generateOptions(all_options, numOptions);
    }

    // Scatter options to all ranks via MPI
    std::vector<OptionInput> my_options(my_count);
    MPI_Scatterv(
        rank == 0 ? all_options.data() : nullptr,
        byte_counts.data(),
        byte_displs.data(),
        MPI_BYTE,
        my_options.data(),
        static_cast<int>(my_count * sizeof(OptionInput)),
        MPI_BYTE,
        0, MPI_COMM_WORLD);

    // Allocate device memory
    OptionInput* d_options = nullptr;
    double* d_results = nullptr;
    if (my_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_options, my_count * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&d_results, my_count * sizeof(double)));
    }

    std::vector<double> my_results(my_count);

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    // Timed section: H2D transfer + GPU kernel + D2H transfer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (my_count > 0) {
        // Host-to-device transfer
        CUDA_CHECK(cudaMemcpy(d_options, my_options.data(),
                              my_count * sizeof(OptionInput), cudaMemcpyHostToDevice));

        // Launch Black-Scholes CUDA kernel
        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(my_count) + blockSize - 1) / blockSize;
        blackScholesKernel<<<numBlocks, blockSize>>>(
            d_options, d_results, static_cast<int>(my_count));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Device-to-host transfer
        CUDA_CHECK(cudaMemcpy(my_results.data(), d_results,
                              my_count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Free device memory
    if (my_count > 0) {
        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
    }

    // Gather results to rank 0 via MPI
    std::vector<double> all_results;
    if (rank == 0) {
        all_results.resize(numOptions);
    }

    MPI_Gatherv(
        my_results.data(),
        static_cast<int>(my_count),
        MPI_DOUBLE,
        rank == 0 ? all_results.data() : nullptr,
        rank == 0 ? opt_counts.data() : nullptr,
        rank == 0 ? opt_displs.data() : nullptr,
        MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Rank 0: output timing, results, and validation
    int exit_code = 0;
    if (rank == 0) {
        double elapsed = end_time - start_time;
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        if (elapsed > 0.0) {
            printf("Options per second: %.0f\n", numOptions / elapsed);
        }

        if (printResultsFlag) {
            print_results(all_results, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(all_options, all_results);
            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    // Broadcast exit code so all ranks return consistently
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
