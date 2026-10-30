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

__device__ inline double cumulativeNormal(const double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const OptionInput* opts, double* res, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const OptionInput& o = opts[idx];
    const double S = o.spot;
    const double K = o.strike;
    const double r = o.r;
    const double q = o.q;
    const double T = o.t;
    const double sigma = o.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        res[idx] = 0.0;
        return;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);

    double price;
    if (o.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    res[idx] = price;
}

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

void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    #pragma omp parallel for
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numOptions = 10000;
    bool validate = false;
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

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<OptionInput> options;
    generateOptions(options, numOptions);

    // partition
    size_t localN = 0;
    for (size_t i = (size_t)rank; i < numOptions; i += (size_t)size) ++localN;

    std::vector<OptionInput> localOpts(localN);
    std::vector<double> localRes(localN);
    size_t j = 0;
    for (size_t i = (size_t)rank; i < numOptions; i += (size_t)size) {
        localOpts[j++] = options[i];
    }

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // CUDA
    OptionInput* d_opts = nullptr;
    double* d_res = nullptr;
    if (localN > 0) {
        cudaMalloc(&d_opts, localN * sizeof(OptionInput));
        cudaMalloc(&d_res, localN * sizeof(double));
        cudaMemcpy(d_opts, localOpts.data(), localN * sizeof(OptionInput), cudaMemcpyHostToDevice);
        int block = 256;
        int grid = (int)((localN + block - 1) / block);
        blackScholesKernel<<<grid, block>>>(d_opts, d_res, localN);
        cudaDeviceSynchronize();
        cudaMemcpy(localRes.data(), d_res, localN * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_opts);
        cudaFree(d_res);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // gather counts
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t c = 0;
        for (size_t i = (size_t)r; i < numOptions; i += (size_t)size) ++c;
        counts[r] = (int)c;
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];

    std::vector<double> results;
    if (rank == 0) results.resize(numOptions);
    MPI_Gatherv(localRes.data(), (int)localN, MPI_DOUBLE,
                rank == 0 ? results.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));

        if (printResults) {
            print_results(results, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(options, results);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
