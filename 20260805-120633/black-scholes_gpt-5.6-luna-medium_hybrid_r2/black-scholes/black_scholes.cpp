#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

enum OptionType { CALL = 0, PUT = 1 };

struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

__device__ __forceinline__ double cumulativeNormalDevice(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const int* type, const double* strike,
                                   const double* spot, const double* q,
                                   const double* r, const double* maturity,
                                   const double* vol, double* result,
                                   size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const double S = spot[i], K = strike[i], sigma = vol[i], T = maturity[i];
    if (T <= 0.0 || sigma <= 0.0) {
        result[i] = 0.0;
        return;
    }
    const double rate = r[i], dividend = q[i];
    const double rootT = sqrt(T);
    const double d1 = (log(S / K) + (rate - dividend + 0.5 * sigma * sigma) * T) /
                      (sigma * rootT);
    const double d2 = d1 - sigma * rootT;
    const double nd1 = cumulativeNormalDevice(d1);
    const double nd2 = cumulativeNormalDevice(d2);
    const double discount = exp(-rate * T);
    if (type[i] == CALL)
        result[i] = S * exp(-dividend * T) * nd1 - K * discount * nd2;
    else
        result[i] = K * discount * cumulativeNormalDevice(-d2) -
                    S * exp(-dividend * T) * cumulativeNormalDevice(-d1);
}

inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{{CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
             {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
             {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
             {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
             {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
             {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3}}};
}

void generateOptions(std::vector<OptionInput>& options, size_t first, size_t count) {
    constexpr auto tests = getTestOptions();
    options.resize(count);
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(count); ++j) {
        const size_t i = first + static_cast<size_t>(j);
        options[static_cast<size_t>(j)] = tests[i % tests.size()];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(tests.size()));
        options[static_cast<size_t>(j)].spot *= factor;
        options[static_cast<size_t>(j)].strike *= factor;
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n"
           "  -n <num>     Number of options to price (default: 10000)\n"
           "  -v           Enable validation against known values\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n", name);
}

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

bool validateResults(const std::vector<OptionInput>& options, const std::vector<double>& results) {
    bool passed = true;
    const int checks = static_cast<int>(std::min<size_t>(10, options.size()));
    printf("Checking computed option prices:\n");
    #pragma omp parallel for reduction(&:passed)
    for (int i = 0; i < checks; ++i) {
        const double error = fabs(results[i] - options[i].value);
        const double relative = error / (fabs(options[i].value) + 1e-10);
        #pragma omp critical
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
               i, results[i], options[i].value, relative);
        if (results[i] < 0.0 || results[i] > 1000.0 || std::isnan(results[i]) ||
            std::isinf(results[i])) passed = false;
    }
    return passed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t total = 10000;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) total = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { fprintf(stderr, "No CUDA accelerator available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice");

    const size_t begin = (total * static_cast<size_t>(rank)) / static_cast<size_t>(world);
    const size_t end = (total * static_cast<size_t>(rank + 1)) / static_cast<size_t>(world);
    const size_t localCount = end - begin;
    std::vector<OptionInput> local;
    generateOptions(local, begin, localCount);
    std::vector<int> type(localCount);
    std::vector<double> strike(localCount), spot(localCount), q(localCount), rate(localCount), maturity(localCount), vol(localCount), localResults(localCount);
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(localCount); ++j) {
        const size_t i = static_cast<size_t>(j);
        type[i] = local[i].type; strike[i] = local[i].strike; spot[i] = local[i].spot;
        q[i] = local[i].q; rate[i] = local[i].r; maturity[i] = local[i].t; vol[i] = local[i].vol;
    }

    // cudaMalloc(0) is invalid on ranks receiving no work (e.g. more ranks
    // than options), so retain a one-element allocation while launching no
    // threads for that rank.
    const size_t allocationCount = std::max<size_t>(1, localCount);
    int* dType; double *dStrike, *dSpot, *dQ, *dRate, *dT, *dVol, *dResult;
    cudaCheck(cudaMalloc(&dType, allocationCount * sizeof(int)), "cudaMalloc(type)");
    cudaCheck(cudaMalloc(&dStrike, allocationCount * sizeof(double)), "cudaMalloc(strike)");
    cudaCheck(cudaMalloc(&dSpot, allocationCount * sizeof(double)), "cudaMalloc(spot)");
    cudaCheck(cudaMalloc(&dQ, allocationCount * sizeof(double)), "cudaMalloc(q)");
    cudaCheck(cudaMalloc(&dRate, allocationCount * sizeof(double)), "cudaMalloc(rate)");
    cudaCheck(cudaMalloc(&dT, allocationCount * sizeof(double)), "cudaMalloc(maturity)");
    cudaCheck(cudaMalloc(&dVol, allocationCount * sizeof(double)), "cudaMalloc(vol)");
    cudaCheck(cudaMalloc(&dResult, allocationCount * sizeof(double)), "cudaMalloc(result)");
    cudaCheck(cudaMemcpy(dType, type.data(), localCount * sizeof(int), cudaMemcpyHostToDevice), "copy type");
    cudaCheck(cudaMemcpy(dStrike, strike.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy strike");
    cudaCheck(cudaMemcpy(dSpot, spot.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy spot");
    cudaCheck(cudaMemcpy(dQ, q.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy q");
    cudaCheck(cudaMemcpy(dRate, rate.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy rate");
    cudaCheck(cudaMemcpy(dT, maturity.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy maturity");
    cudaCheck(cudaMemcpy(dVol, vol.data(), localCount * sizeof(double), cudaMemcpyHostToDevice), "copy vol");

    if (rank == 0) { printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\n", total, validate ? "enabled" : "disabled"); printf("Pricing options...\n"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const int blocks = static_cast<int>((localCount + 255) / 256);
    if (localCount) {
        blackScholesKernel<<<blocks, 256>>>(dType, dStrike, dSpot, dQ, dRate, dT, dVol, dResult, localCount);
        cudaCheck(cudaGetLastError(), "blackScholesKernel launch");
    }
    cudaCheck(cudaMemcpy(localResults.data(), dResult, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy results");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endTime = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(endTime - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<int> counts(world), displacements(world);
    const int localInt = static_cast<int>(localCount);
    MPI_Gather(&localInt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> results;
    if (rank == 0) { results.resize(total); int offset = 0; for (int i = 0; i < world; ++i) { displacements[i] = offset; offset += counts[i]; } }
    MPI_Gatherv(localResults.data(), localInt, MPI_DOUBLE, results.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cudaFree(dType); cudaFree(dStrike); cudaFree(dSpot); cudaFree(dQ); cudaFree(dRate); cudaFree(dT); cudaFree(dVol); cudaFree(dResult);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nOptions per second: %.0f\n", seconds * 1000.0, total / seconds);
        if (printResults) print_results(results, "OptionPrices");
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, 0, total);
            const bool valid = validateResults(validationOptions, results);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
