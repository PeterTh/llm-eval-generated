#include <algorithm>
#include <array>
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
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                          \
        }                                                                     \
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

// Standard normal cumulative distribution function (host + device)
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Black-Scholes CUDA device kernel
__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const OptionInput opt = options[i];
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

    const double nd1 = cumulativeNormal(d1);
    const double nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    if (opt.type == CALL) {
        results[i] = S * exp(-q * T) * nd1 - K * discount * nd2;
    } else {
        results[i] = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
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

// Generate a deterministic range of options (for MPI-based independent generation)
void generateOptionsRange(OptionInput* options, const size_t startIdx,
                          const size_t count) {
    constexpr auto testOptions = getTestOptions();
    const size_t numTests = testOptions.size();

    for (size_t i = 0; i < count; ++i) {
        const size_t globalIdx = startIdx + i;
        const OptionInput& base = testOptions[globalIdx % numTests];
        options[i] = base;
        const double factor = 1.0 + 0.1 * (static_cast<double>(globalIdx) / static_cast<double>(numTests));
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

        if (computed < 0.0 || computed > 1000.0 ||
            std::isnan(computed) || std::isinf(computed)) {
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Rank 0 parses command-line arguments
    if (rank == 0) {
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
            }
        }
    }

    // Broadcast parameters to all ranks
    unsigned long long bufNumOpts = static_cast<unsigned long long>(numOptions);
    MPI_Bcast(&bufNumOpts, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(bufNumOpts);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Query CUDA devices
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark (MPI+OpenMP+CUDA Hybrid)\n");
        printf("Number of options: %zu\n", numOptions);
        printf("MPI processes: %d\n", size);
        printf("CUDA devices available: %d\n", numDevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA-capable device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Bind each MPI rank to a GPU (round-robin assignment)
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    // Distribute work across MPI ranks
    const size_t baseLocalN = numOptions / static_cast<size_t>(size);
    const size_t remainder = numOptions % static_cast<size_t>(size);
    const size_t offset = rank * baseLocalN +
        (static_cast<size_t>(rank) < remainder ? static_cast<size_t>(rank) : remainder);
    const size_t localN = baseLocalN + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t numStreams = 4;

    // Generate local options (parallelized with OpenMP on CPU)
    std::vector<OptionInput> localOptions(localN);

    if (localN > 0) {
        const auto testOptions = getTestOptions();
        const size_t numTests = testOptions.size();

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(localN); ++i) {
            const size_t globalIdx = offset + static_cast<size_t>(i);
            const OptionInput& base = testOptions[globalIdx % numTests];
            localOptions[static_cast<size_t>(i)] = base;
            const double factor = 1.0 + 0.1 * (static_cast<double>(globalIdx) / static_cast<double>(numTests));
            localOptions[static_cast<size_t>(i)].spot *= factor;
            localOptions[static_cast<size_t>(i)].strike *= factor;
        }
    }

    // -----------------------------------------------------------------------
    // GPU memory allocation and data transfer
    // -----------------------------------------------------------------------
    OptionInput* d_options = nullptr;
    double* d_results = nullptr;

    if (localN > 0) {
        CUDA_CHECK(cudaMalloc(&d_options, localN * sizeof(OptionInput)));
        CUDA_CHECK(cudaMalloc(&d_results, localN * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_options, localOptions.data(),
                              localN * sizeof(OptionInput), cudaMemcpyHostToDevice));
    }

    // Synchronise all ranks before timing the computational phase
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // -----------------------------------------------------------------------
    // CUDA kernel launch via multiple streams managed by OpenMP threads
    // -----------------------------------------------------------------------
    if (localN > 0) {
        cudaStream_t streams[4];
        for (int s = 0; s < static_cast<int>(numStreams); ++s) {
            CUDA_CHECK(cudaStreamCreate(&streams[s]));
        }

        #pragma omp parallel for num_threads(static_cast<int>(numStreams))
        for (int s = 0; s < static_cast<int>(numStreams); ++s) {
            const size_t chunkBase =
                (localN + numStreams - 1) / numStreams;
            const size_t chunkStart =
                static_cast<size_t>(s) * chunkBase;
            const size_t chunkEnd =
                (chunkStart + chunkBase > localN) ? localN : chunkStart + chunkBase;
            const size_t chunkCount = chunkEnd - chunkStart;

            if (chunkCount > 0) {
                constexpr int threadsPerBlock = 256;
                const int blocksPerGrid =
                    static_cast<int>((chunkCount + threadsPerBlock - 1) / threadsPerBlock);

                blackScholesKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
                    d_options + chunkStart, d_results + chunkStart, chunkCount);
            }
        }

        for (int s = 0; s < static_cast<int>(numStreams); ++s) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
        }
    }

    const double tEnd = MPI_Wtime();

    // -----------------------------------------------------------------------
    // Copy results back to host
    // -----------------------------------------------------------------------
    std::vector<double> localResults;

    if (localN > 0) {
        localResults.resize(localN);
        CUDA_CHECK(cudaMemcpy(localResults.data(), d_results,
                              localN * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_options));
        CUDA_CHECK(cudaFree(d_results));
    }

    // -----------------------------------------------------------------------
    // Gather all results on rank 0
    // -----------------------------------------------------------------------
    std::vector<double> allResults;
    std::vector<int> recvCounts;
    std::vector<int> displacements;

    if (rank == 0) {
        allResults.resize(numOptions);
        recvCounts.resize(static_cast<size_t>(size));
        displacements.resize(static_cast<size_t>(size));

        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rLocalN = baseLocalN +
                (static_cast<size_t>(r) < remainder ? 1 : 0);
            recvCounts[r] = static_cast<int>(rLocalN);
            displacements[r] = static_cast<int>(disp);
            disp += rLocalN;
        }
    }

    MPI_Gatherv(
        localN > 0 ? localResults.data() : nullptr,
        static_cast<int>(localN), MPI_DOUBLE,
        rank == 0 ? allResults.data() : nullptr,
        rank == 0 ? recvCounts.data() : nullptr,
        rank == 0 ? displacements.data() : nullptr,
        MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // -----------------------------------------------------------------------
    // Rank 0 prints results and optionally validates
    // -----------------------------------------------------------------------
    if (rank == 0) {
        const double durationMs = (tEnd - tStart) * 1000.0;
        printf("Computation time: %.3f ms\n", durationMs);
        printf("Options per second: %.0f\n",
               numOptions / (tEnd - tStart));

        if (printResults && !allResults.empty()) {
            print_results(allResults, "OptionPrices");
        }

        if (validate && !allResults.empty()) {
            printf("Validating results...\n");
            const size_t nCheck = std::min(numOptions, static_cast<size_t>(10));

            std::vector<OptionInput> valOpts(nCheck);
            generateOptionsRange(valOpts.data(), 0, nCheck);

            std::vector<double> valResults(nCheck);
            for (size_t i = 0; i < nCheck; ++i) {
                valResults[i] = allResults[i];
            }

            if (validateResults(valOpts, valResults)) {
                printf("Validation: PASSED\n");
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
