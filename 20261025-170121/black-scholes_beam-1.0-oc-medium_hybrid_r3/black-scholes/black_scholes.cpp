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

__device__ double cumulativeNormal_dev(const double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const OptionInput* opts, double* res, const size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const OptionInput& o = opts[i];
    const double S = o.spot;
    const double K = o.strike;
    const double r = o.r;
    const double q = o.q;
    const double T = o.t;
    const double sigma = o.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        res[i] = 0.0;
        return;
    }

    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);

    const double Nd1 = cumulativeNormal_dev(d1);
    const double Nd2 = cumulativeNormal_dev(d2);
    const double discount = exp(-r * T);

    double price;
    if (o.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * cumulativeNormal_dev(-d2) - S * exp(-q * T) * cumulativeNormal_dev(-d1);
    }
    res[i] = price;
}

inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

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
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }

    return price;
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

    MPI_Bcast(&numOptions, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<OptionInput> allOpts;
    if (rank == 0) generateOptions(allOpts, numOptions);
    else allOpts.resize(numOptions);

    std::vector<int> counts(size), displs(size);
    size_t base = numOptions / size;
    size_t rem = numOptions % size;
    size_t off = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>((base + (r < (int)rem)) * sizeof(OptionInput));
        displs[r] = static_cast<int>(off * sizeof(OptionInput));
        off += base + (r < (int)rem);
    }

    std::vector<OptionInput> localOpts((numOptions / size) + (rank < (int)rem ? 1 : 0));
    MPI_Scatterv(allOpts.data(), counts.data(), displs.data(), MPI_BYTE,
                 localOpts.data(), static_cast<int>(localOpts.size() * sizeof(OptionInput)), MPI_BYTE,
                 0, MPI_COMM_WORLD);

    std::vector<double> localResults(localOpts.size());

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t n = localOpts.size();
    OptionInput* d_opts = nullptr;
    double* d_res = nullptr;
    cudaMalloc(&d_opts, n * sizeof(OptionInput));
    cudaMalloc(&d_res, n * sizeof(double));
    cudaMemcpy(d_opts, localOpts.data(), n * sizeof(OptionInput), cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    blackScholesKernel<<<grid, block>>>(d_opts, d_res, n);
    cudaDeviceSynchronize();
    cudaMemcpy(localResults.data(), d_res, n * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_opts);
    cudaFree(d_res);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double localMs = duration.count() / 1000.0;
    double maxMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> rcounts(size), rdispls(size);
    off = 0;
    for (int r = 0; r < size; ++r) {
        size_t c = base + (r < (int)rem);
        rcounts[r] = static_cast<int>(c);
        rdispls[r] = static_cast<int>(off);
        off += c;
    }
    std::vector<double> allResults(numOptions);
    MPI_Gatherv(localResults.data(), static_cast<int>(localResults.size()), MPI_DOUBLE,
                allResults.data(), rcounts.data(), rdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        printf("Options per second: %.0f\n", numOptions / (maxMs / 1000.0));

        if (printResults) print_results(allResults, "OptionPrices");

        if (validate) {
            printf("Validating results...\n");
            bool valid = validateResults(allOpts, allResults);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}