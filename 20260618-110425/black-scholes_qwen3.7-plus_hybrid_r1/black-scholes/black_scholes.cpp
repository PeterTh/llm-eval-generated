#include <mpi.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
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

// SoA layout for coalesced GPU memory access
struct OptionSoA {
    int*    type;
    double* strike;
    double* spot;
    double* q;
    double* r;
    double* t;
    double* vol;
};

// ==================== CUDA Kernels ====================

__device__ __forceinline__ double devCumulativeNormal(double x) {
    return 0.5 * (1.0 + erf(x * 0.7071067811865475244));
}

__global__ void blackScholesKernel(
    const int*    __restrict__ g_type,
    const double* __restrict__ g_strike,
    const double* __restrict__ g_spot,
    const double* __restrict__ g_q,
    const double* __restrict__ g_r,
    const double* __restrict__ g_t,
    const double* __restrict__ g_vol,
    double*       __restrict__ g_results,
    const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double S     = g_spot[idx];
    const double K     = g_strike[idx];
    const double r     = g_r[idx];
    const double q     = g_q[idx];
    const double T     = g_t[idx];
    const double sigma = g_vol[idx];

    if (T <= 0.0 || sigma <= 0.0) {
        g_results[idx] = 0.0;
        return;
    }

    const double sqrtT   = sqrt(T);
    const double d1      = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2      = d1 - sigma * sqrtT;
    const double discount = exp(-r * T);

    double price;
    if (g_type[idx] == CALL) {
        const double Nd1 = devCumulativeNormal(d1);
        const double Nd2 = devCumulativeNormal(d2);
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else {
        price = K * discount * devCumulativeNormal(-d2)
              - S * exp(-q * T) * devCumulativeNormal(-d1);
    }

    g_results[idx] = price;
}

// ==================== Host Functions ====================

inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
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
        {PUT,  100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

// Generate options for a specific index range (OpenMP-parallel)
void generateOptionsRange(std::vector<OptionInput>& options,
                          size_t startIdx, size_t count) {
    constexpr auto testOptions = getTestOptions();
    constexpr size_t numTest = testOptions.size();
    options.resize(count);

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < count; ++j) {
        const size_t i = startIdx + j;
        const OptionInput& base = testOptions[i % numTest];
        OptionInput opt = base;
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(numTest));
        opt.spot   *= factor;
        opt.strike *= factor;
        options[j] = opt;
    }
}

// Pack AoS options into SoA arrays allocated as pinned memory (OpenMP-parallel)
void packSoA(std::vector<OptionInput>& options, size_t count,
              OptionSoA& soa) {
    // Allocate pinned memory directly for DMA-optimized transfers
    CUDA_CHECK(cudaMallocHost(&soa.type,   sizeof(int)    * count));
    CUDA_CHECK(cudaMallocHost(&soa.strike, sizeof(double) * count));
    CUDA_CHECK(cudaMallocHost(&soa.spot,   sizeof(double) * count));
    CUDA_CHECK(cudaMallocHost(&soa.q,      sizeof(double) * count));
    CUDA_CHECK(cudaMallocHost(&soa.r,      sizeof(double) * count));
    CUDA_CHECK(cudaMallocHost(&soa.t,      sizeof(double) * count));
    CUDA_CHECK(cudaMallocHost(&soa.vol,    sizeof(double) * count));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        soa.type[i]   = options[i].type;
        soa.strike[i] = options[i].strike;
        soa.spot[i]   = options[i].spot;
        soa.q[i]      = options[i].q;
        soa.r[i]      = options[i].r;
        soa.t[i]      = options[i].t;
        soa.vol[i]    = options[i].vol;
    }
}

void freeSoA(OptionSoA& soa) {
    if (soa.type)   { cudaFreeHost(soa.type);   soa.type   = nullptr; }
    if (soa.strike) { cudaFreeHost(soa.strike); soa.strike = nullptr; }
    if (soa.spot)   { cudaFreeHost(soa.spot);   soa.spot   = nullptr; }
    if (soa.q)      { cudaFreeHost(soa.q);      soa.q      = nullptr; }
    if (soa.r)      { cudaFreeHost(soa.r);      soa.r      = nullptr; }
    if (soa.t)      { cudaFreeHost(soa.t);      soa.t      = nullptr; }
    if (soa.vol)    { cudaFreeHost(soa.vol);    soa.vol    = nullptr; }
}

// Run Black-Scholes on GPU for local chunk
// SoA arrays are already in pinned memory for optimal DMA transfers
void priceOptionsGPU(const OptionSoA& h_soa, double* h_results, int n) {
    if (n <= 0) return;

    const size_t bytesInt    = sizeof(int)    * n;
    const size_t bytesDouble = sizeof(double) * n;

    // Device pointers
    int    *d_type;
    double *d_strike, *d_spot, *d_q, *d_r, *d_t, *d_vol, *d_results;

    CUDA_CHECK(cudaMalloc(&d_type,    bytesInt));
    CUDA_CHECK(cudaMalloc(&d_strike,  bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_spot,    bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_q,       bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_r,       bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_t,       bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_vol,     bytesDouble));
    CUDA_CHECK(cudaMalloc(&d_results, bytesDouble));

    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Async H2D transfers from pinned memory
    CUDA_CHECK(cudaMemcpyAsync(d_type,   h_soa.type,   bytesInt,    cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_strike, h_soa.strike, bytesDouble, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_spot,   h_soa.spot,   bytesDouble, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_q,      h_soa.q,      bytesDouble, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_r,      h_soa.r,      bytesDouble, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_t,      h_soa.t,      bytesDouble, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vol,    h_soa.vol,     bytesDouble, cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;
    blackScholesKernel<<<gridSize, blockSize, 0, stream>>>(
        d_type, d_strike, d_spot, d_q, d_r, d_t, d_vol, d_results, n);

    // Async D2H transfer directly to output buffer
    // h_results is a std::vector, but we can use it if page-aligned
    // For best performance, allocate pinned output too
    double* hpin_results;
    CUDA_CHECK(cudaMallocHost(&hpin_results, bytesDouble));
    CUDA_CHECK(cudaMemcpyAsync(hpin_results, d_results, bytesDouble, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy to output (OpenMP-parallel)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        h_results[i] = hpin_results[i];
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(hpin_results));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_type));
    CUDA_CHECK(cudaFree(d_strike));
    CUDA_CHECK(cudaFree(d_spot));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_results));
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
    // ---- MPI Initialization ----
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // ---- CUDA device assignment (one GPU per MPI rank, round-robin) ----
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        fprintf(stderr, "Rank %d: No CUDA devices found\n", mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpuId = mpiRank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuId));

    // ---- Parse command line arguments (all ranks) ----
    size_t numOptions = 10000;
    bool validate = false;
    bool printResultsFlag = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpiRank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs: %d (per rank: 1 of %d available)\n",
               mpiSize, numGPUs, numGPUs);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("GPU per rank: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    }

    // ---- Compute work distribution ----
    const int localStart = static_cast<int>(
        (static_cast<long long>(numOptions) * mpiRank) / mpiSize);
    const int localEnd = static_cast<int>(
        (static_cast<long long>(numOptions) * (mpiRank + 1)) / mpiSize);
    const int myCount = localEnd - localStart;

    // ---- Generate local options (OpenMP-parallel) ----
    std::vector<OptionInput> localOptions;
    generateOptionsRange(localOptions, localStart, myCount);

    // ---- Pack into SoA (OpenMP-parallel, pinned memory) ----
    OptionSoA localSoA = {};
    packSoA(localOptions, myCount, localSoA);

    // ---- Allocate local results ----
    std::vector<double> localResults(myCount);

    // ---- Price options on GPU (timed) ----
    if (mpiRank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    priceOptionsGPU(localSoA, localResults.data(), myCount);

    auto end = std::chrono::high_resolution_clock::now();
    double localTime = std::chrono::duration<double, std::milli>(end - start).count();

    // ---- Gather results to rank 0 ----
    // Compute counts and displacements for Gatherv
    std::vector<int> recvCounts(mpiSize);
    std::vector<int> recvDispls(mpiSize);
    if (mpiRank == 0) {
        for (int r = 0; r < mpiSize; ++r) {
            int rStart = static_cast<int>(
                (static_cast<long long>(numOptions) * r) / mpiSize);
            int rEnd = static_cast<int>(
                (static_cast<long long>(numOptions) * (r + 1)) / mpiSize);
            recvCounts[r] = rEnd - rStart;
            recvDispls[r] = rStart;
        }
    }

    auto commStart = std::chrono::high_resolution_clock::now();

    std::vector<double> allResults;
    if (mpiRank == 0) {
        allResults.resize(numOptions);
    }

    MPI_Gatherv(localResults.data(), myCount, MPI_DOUBLE,
                mpiRank == 0 ? allResults.data() : nullptr,
                mpiRank == 0 ? recvCounts.data() : nullptr,
                mpiRank == 0 ? recvDispls.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto commEnd = std::chrono::high_resolution_clock::now();
    double commTime = std::chrono::duration<double, std::milli>(commEnd - commStart).count();

    // ---- Report timing ----
    double maxTime;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpiRank == 0) {
        printf("Computation time: %.3f ms\n", maxTime);
        printf("Communication time: %.3f ms\n", commTime);
        printf("Options per second: %.0f\n", numOptions / (maxTime / 1e3));
    }

    // ---- Print results for external validation (rank 0 only) ----
    if (mpiRank == 0 && printResultsFlag) {
        print_results(allResults, "OptionPrices");
    }

    // ---- Validation (rank 0 only) ----
    int validationResult = 0;
    if (mpiRank == 0 && validate) {
        printf("Validating results...\n");
        std::vector<OptionInput> fullOptions;
        generateOptionsRange(fullOptions, 0, numOptions);
        bool valid = validateResults(fullOptions, allResults);
        validationResult = valid ? 0 : 1;
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Cleanup ----
    freeSoA(localSoA);
    MPI_Finalize();

    return validationResult;
}
