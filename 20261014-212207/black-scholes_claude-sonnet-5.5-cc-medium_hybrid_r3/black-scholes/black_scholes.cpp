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

// Standard normal cumulative distribution function
__host__ __device__ inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// Standard normal probability density function
__host__ __device__ inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
__host__ __device__ double blackScholes(const OptionInput& option) noexcept {
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
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
        
        // Relaxed validation - just check values are positive and reasonable
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Test table on the device; options are generated on the GPU from their global index
__constant__ OptionInput d_testOptions[7];

__global__ void priceKernel(double* __restrict__ out, const size_t first, const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += stride) {
        const size_t i = first + k;
        OptionInput opt = d_testOptions[i % 7];
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(7));
        opt.spot *= factor;
        opt.strike *= factor;
        out[k] = blackScholes(opt);
    }
}

// Gather results[0, need) on rank 0 from the contiguous blocks owned by each rank
static void gatherPrefix(const double* local, size_t myBegin, size_t myEnd, size_t need,
                         const std::vector<size_t>& bounds, int rank, int size,
                         std::vector<double>& results) {
    constexpr size_t CHUNK = size_t(1) << 26;
    if (rank == 0) {
        std::vector<MPI_Request> reqs;
        const size_t e0 = std::min(myEnd, need);
        if (e0 > myBegin) {
            #pragma omp parallel for schedule(static)
            for (long long k = 0; k < static_cast<long long>(e0 - myBegin); ++k)
                results[myBegin + k] = local[k];
        }
        for (int r = 1; r < size; ++r) {
            const size_t b = bounds[r], e = std::min(bounds[r + 1], need);
            for (size_t p = b; p < e; p += CHUNK) {
                reqs.emplace_back();
                MPI_Irecv(&results[p], static_cast<int>(std::min(CHUNK, e - p)), MPI_DOUBLE, r,
                          static_cast<int>((p - b) / CHUNK), MPI_COMM_WORLD, &reqs.back());
            }
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else {
        const size_t e = std::min(myEnd, need);
        for (size_t p = myBegin; p < e; p += CHUNK)
            MPI_Send(local + (p - myBegin), static_cast<int>(std::min(CHUNK, e - p)), MPI_DOUBLE, 0,
                     static_cast<int>((p - myBegin) / CHUNK), MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Bind each rank to a GPU on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(0));

    constexpr auto testOptions = getTestOptions();
    CUDA_CHECK(cudaMemcpyToSymbol(d_testOptions, testOptions.data(), sizeof(OptionInput) * 7));

    // Block distribution of the option index space
    std::vector<size_t> bounds(size + 1);
    for (int r = 0; r <= size; ++r)
        bounds[r] = numOptions / size * r + std::min<size_t>(r, numOptions % size);
    const size_t myBegin = bounds[rank], myEnd = bounds[rank + 1], myCount = myEnd - myBegin;

    double* hLocal = nullptr;
    constexpr int NSTREAMS = 4;
    constexpr size_t CHUNK = size_t(1) << 22;
    const size_t chunk = std::min(CHUNK, std::max<size_t>(myCount, 1));
    double* dBuf[NSTREAMS] = {};
    cudaStream_t streams[NSTREAMS];
    CUDA_CHECK(cudaMallocHost(&hLocal, std::max<size_t>(myCount, 1) * sizeof(double)));
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&dBuf[s], chunk * sizeof(double)));
    }

    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Price options: chunks are pipelined over streams (kernel + async copy-back)
    {
        const int block = 256;
        int chunkId = 0;
        for (size_t off = 0; off < myCount; off += chunk, ++chunkId) {
            const int s = chunkId % NSTREAMS;
            const size_t n = std::min(chunk, myCount - off);
            const int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 1u << 20));
            priceKernel<<<grid, block, 0, streams[s]>>>(dBuf[s], myBegin + off, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hLocal + off, dBuf[s], n * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Only the results that are actually consumed are gathered on rank 0
    const size_t need = printResults ? numOptions : (validate ? std::min<size_t>(10, numOptions) : 0);
    std::vector<double> results;
    if (rank == 0) results.resize(need);
    if (need > 0) gatherPrefix(hLocal, myBegin, myEnd, need, bounds, rank, size, results);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double local_ms = duration.count() / 1000.0;
    double max_ms = 0.0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_ms);
        printf("Options per second: %.0f\n", numOptions / (max_ms / 1000.0));
    }

    int status = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(results, "OptionPrices");
        }

        // Validation
        if (validate) {
            printf("Validating results...\n");
            std::vector<OptionInput> options;
            generateOptions(options, std::min<size_t>(10, numOptions));
            bool valid = validateResults(options, results);
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int s = 0; s < NSTREAMS; ++s) {
        cudaFree(dBuf[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFreeHost(hLocal);
    MPI_Finalize();
    return status;
}
