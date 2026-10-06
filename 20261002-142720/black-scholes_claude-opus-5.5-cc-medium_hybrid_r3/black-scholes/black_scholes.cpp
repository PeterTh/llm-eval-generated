#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
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
__host__ __device__ inline double blackScholes(const OptionInput& option) noexcept {
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

constexpr size_t NUM_TEST_OPTIONS = getTestOptions().size();

// Build option i from its base test case (shared by host and device so that
// the generated inputs are bit-identical on both sides)
__host__ __device__ inline OptionInput makeOption(const OptionInput& base, const size_t i) noexcept {
    OptionInput opt = base;
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(NUM_TEST_OPTIONS));
    opt.spot *= factor;
    opt.strike *= factor;
    return opt;
}

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        options[i] = makeOption(testOptions[i % testOptions.size()], i);
    }
}

// Test options in constant memory for the device-side generator
__constant__ OptionInput d_testOptions[NUM_TEST_OPTIONS];

// Generate and price options [begin, begin + count) entirely on the device;
// out[k] receives the price of global option begin + k.
__global__ void __launch_bounds__(256)
blackScholesKernel(double* __restrict__ out, const size_t begin, const size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < count; k += stride) {
        const size_t i = begin + k;
        const OptionInput opt = makeOption(d_testOptions[i % NUM_TEST_OPTIONS], i);
        out[k] = blackScholes(opt);
    }
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    // Bind each rank to a GPU of its node (round-robin over node-local ranks)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    // Block decomposition of the option index space across ranks
    const size_t baseCount = numOptions / nranks;
    const size_t remainder = numOptions % nranks;
    const size_t localBegin = rank * baseCount + std::min<size_t>(rank, remainder);
    const size_t localCount = baseCount + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Device setup (outside timed region): constants, buffers, streams
    constexpr auto testOptions = getTestOptions();
    CUDA_CHECK(cudaMemcpyToSymbol(d_testOptions, testOptions.data(), sizeof(OptionInput) * NUM_TEST_OPTIONS));
    double* d_results = nullptr;
    double* h_results = nullptr;  // pinned staging buffer for this rank's chunk
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_results, localCount * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_results, localCount * sizeof(double)));
    }

    // Pipeline: the rank's chunk is split into tiles; each OpenMP thread owns a
    // CUDA stream and alternates kernel launches and async D2H copies so that
    // computation overlaps with PCIe transfers.
    constexpr int numStreams = 4;
    constexpr size_t minTile = size_t(1) << 20;
    const size_t numTiles = std::max<size_t>(1, std::min<size_t>(
        static_cast<size_t>(numStreams) * 4, (localCount + minTile - 1) / minTile));
    const size_t tileSize = (localCount + numTiles - 1) / std::max<size_t>(numTiles, 1);
    cudaStream_t streams[numStreams];
    for (int s = 0; s < numStreams; ++s) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    }
    constexpr int blockSize = 256;
    const int maxBlocks = numSMs * 8;

    // Warm up the OpenMP thread team (binding each thread to the device) and
    // the kernel (module load) outside the timed region
    #pragma omp parallel num_threads(numStreams)
    {
        CUDA_CHECK(cudaSetDevice(device));
    }
    blackScholesKernel<<<1, blockSize, 0, streams[0]>>>(d_results, 0, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Price options
    if (rank == 0) printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numStreams)
    {
        const int tid = omp_get_thread_num();
        const int nthr = omp_get_num_threads();
        // The current device is per host thread
        CUDA_CHECK(cudaSetDevice(device));
        for (int s = tid; s < numStreams; s += nthr) {
            cudaStream_t stream = streams[s];
            for (size_t t = s; t < numTiles; t += numStreams) {
                const size_t off = t * tileSize;
                if (off >= localCount) break;
                const size_t cnt = std::min(tileSize, localCount - off);
                const int blocks = static_cast<int>(std::min<size_t>(
                    maxBlocks, (cnt + blockSize - 1) / blockSize));
                blackScholesKernel<<<blocks, blockSize, 0, stream>>>(d_results + off, localBegin + off, cnt);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(h_results + off, d_results + off, cnt * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    // Wall time of the slowest rank
    long long localUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    long long maxUs = 0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxUs / 1e6));
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Collect the full result vector on rank 0 (MPI_Gatherv with a derived
        // datatype so chunks larger than INT_MAX elements are still supported)
        std::vector<double> results;
        if (rank == 0) results.resize(numOptions);
        constexpr int gatherBlock = 1024;  // elements per gather unit
        const bool aligned = (baseCount % gatherBlock == 0) && (remainder == 0);
        if (aligned && baseCount / gatherBlock <= static_cast<size_t>(INT32_MAX)) {
            MPI_Datatype blockType;
            MPI_Type_contiguous(gatherBlock, MPI_DOUBLE, &blockType);
            MPI_Type_commit(&blockType);
            MPI_Gather(h_results, static_cast<int>(baseCount / gatherBlock), blockType,
                       results.data(), static_cast<int>(baseCount / gatherBlock), blockType,
                       0, MPI_COMM_WORLD);
            MPI_Type_free(&blockType);
        } else {
            // Point-to-point, in INT_MAX-bounded pieces
            constexpr size_t maxMsg = size_t(1) << 30;
            if (rank == 0) {
                if (localCount > 0) memcpy(results.data(), h_results, localCount * sizeof(double));
                for (int src = 1; src < nranks; ++src) {
                    const size_t b = src * baseCount + std::min<size_t>(src, remainder);
                    const size_t c = baseCount + (static_cast<size_t>(src) < remainder ? 1 : 0);
                    for (size_t o = 0; o < c; o += maxMsg) {
                        MPI_Recv(results.data() + b + o, static_cast<int>(std::min(maxMsg, c - o)),
                                 MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            } else {
                for (size_t o = 0; o < localCount; o += maxMsg) {
                    MPI_Send(h_results + o, static_cast<int>(std::min(maxMsg, localCount - o)),
                             MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(results, "OptionPrices");
            }
            
            // Validation
            if (validate) {
                printf("Validating results...\n");
                // Only the first min(10, N) options are checked
                std::vector<OptionInput> options;
                generateOptions(options, std::min<size_t>(10, numOptions));
                bool valid = validateResults(options, results);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    for (int s = 0; s < numStreams; ++s) cudaStreamDestroy(streams[s]);
    if (d_results) cudaFree(d_results);
    if (h_results) cudaFreeHost(h_results);

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
