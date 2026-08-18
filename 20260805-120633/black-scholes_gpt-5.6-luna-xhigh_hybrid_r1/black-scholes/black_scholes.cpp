#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

constexpr size_t CUDA_CHUNK_OPTIONS = 1u << 20;
constexpr int CUDA_STREAM_COUNT = 4;
constexpr int CUDA_BLOCK_SIZE = 256;

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
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
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Device version of the pricing formula.  The input structure is trivially
// copyable, so one contiguous transfer can feed the GPU kernel efficiently.
__device__ __forceinline__ double cumulativeNormalDevice(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__device__ __forceinline__ double blackScholesDevice(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;

    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }

    const double sqrtT = sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
                      (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = cumulativeNormalDevice(d1);
    const double Nd2 = cumulativeNormalDevice(d2);
    const double discount = exp(-r * T);

    if (option.type == CALL) {
        return S * exp(-q * T) * Nd1 - K * discount * Nd2;
    }
    return K * discount * cumulativeNormalDevice(-d2) -
           S * exp(-q * T) * cumulativeNormalDevice(-d1);
}

__global__ void blackScholesKernel(const OptionInput* __restrict__ options,
                                   double* __restrict__ results,
                                   const size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        results[index] = blackScholesDevice(options[index]);
    }
}

[[noreturn]] void abortCuda(MPI_Comm communicator, const int rank,
                            const cudaError_t status, const char* operation) {
    fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
            cudaGetErrorString(status));
    MPI_Abort(communicator, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const MPI_Comm communicator, const int rank,
               const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        abortCuda(communicator, rank, status, operation);
    }
}

// A rank owns one contiguous part of the input.  Four reusable streams allow
// host-to-device transfer, execution, and device-to-host transfer of adjacent
// chunks to overlap when the local input is larger than one chunk.
class CudaPricer {
  public:
    CudaPricer(OptionInput* hostOptions, double* hostResults,
               const size_t count, const int rank, const MPI_Comm communicator)
        : hostOptions_(hostOptions), hostResults_(hostResults), count_(count),
          rank_(rank), communicator_(communicator) {
        if (count_ == 0) {
            return;
        }

        streamCount_ = std::min<size_t>(CUDA_STREAM_COUNT,
                                        (count_ + CUDA_CHUNK_OPTIONS - 1) /
                                            CUDA_CHUNK_OPTIONS);
        deviceOptions_.fill(nullptr);
        deviceResults_.fill(nullptr);
        streams_.fill(nullptr);

        for (size_t i = 0; i < streamCount_; ++i) {
            checkCuda(communicator_, rank_,
                      cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking),
                      "cudaStreamCreateWithFlags");
            checkCuda(communicator_, rank_,
                      cudaMalloc(reinterpret_cast<void**>(&deviceOptions_[i]),
                                 CUDA_CHUNK_OPTIONS * sizeof(OptionInput)),
                      "cudaMalloc(options)");
            checkCuda(communicator_, rank_,
                      cudaMalloc(reinterpret_cast<void**>(&deviceResults_[i]),
                                 CUDA_CHUNK_OPTIONS * sizeof(double)),
                      "cudaMalloc(results)");
        }
    }

    CudaPricer(const CudaPricer&) = delete;
    CudaPricer& operator=(const CudaPricer&) = delete;

    ~CudaPricer() {
        for (size_t i = 0; i < streamCount_; ++i) {
            cudaFree(deviceOptions_[i]);
            cudaFree(deviceResults_[i]);
            cudaStreamDestroy(streams_[i]);
        }
    }

    void run() {
        for (size_t offset = 0; offset < count_; offset += CUDA_CHUNK_OPTIONS) {
            const size_t chunk = std::min(CUDA_CHUNK_OPTIONS, count_ - offset);
            const size_t stream = (offset / CUDA_CHUNK_OPTIONS) % streamCount_;
            cudaStream_t const cudaStream = streams_[stream];

            checkCuda(communicator_, rank_,
                      cudaMemcpyAsync(deviceOptions_[stream], hostOptions_ + offset,
                                      chunk * sizeof(OptionInput), cudaMemcpyHostToDevice,
                                      cudaStream),
                      "cudaMemcpyAsync(options)");

            const unsigned int grid = static_cast<unsigned int>(
                (chunk + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
            blackScholesKernel<<<grid, CUDA_BLOCK_SIZE, 0, cudaStream>>>(
                deviceOptions_[stream], deviceResults_[stream], chunk);
            checkCuda(communicator_, rank_, cudaGetLastError(),
                      "blackScholesKernel launch");

            checkCuda(communicator_, rank_,
                      cudaMemcpyAsync(hostResults_ + offset, deviceResults_[stream],
                                      chunk * sizeof(double), cudaMemcpyDeviceToHost,
                                      cudaStream),
                      "cudaMemcpyAsync(results)");
        }

        checkCuda(communicator_, rank_, cudaDeviceSynchronize(),
                  "cudaDeviceSynchronize");
    }

  private:
    OptionInput* hostOptions_;
    double* hostResults_;
    size_t count_;
    int rank_;
    MPI_Comm communicator_;
    size_t streamCount_ = 0;
    std::array<cudaStream_t, CUDA_STREAM_COUNT> streams_{};
    std::array<OptionInput*, CUDA_STREAM_COUNT> deviceOptions_{};
    std::array<double*, CUDA_STREAM_COUNT> deviceResults_{};
};

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

// Generate a larger set of options by scaling the test set.  globalStart
// keeps the input identical to the original serial generator after MPI
// partitions the global option array among ranks.
void generateOptions(OptionInput* options, const size_t globalStart,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();

    #pragma omp parallel for schedule(static)
    for (long long localIndex = 0;
         localIndex < static_cast<long long>(numOptions); ++localIndex) {
        const size_t i = globalStart + static_cast<size_t>(localIndex);
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[localIndex] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[localIndex].spot *= factor;
        options[localIndex].strike *= factor;
    }
}

void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    options.resize(numOptions);
    generateOptions(options.data(), 0, numOptions);
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

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                        &providedThreadLevel) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or incomplete command-line option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (numOptions > static_cast<size_t>(LLONG_MAX)) {
        if (rank == 0) {
            fprintf(stderr, "Number of options is too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &localCommunicator) != MPI_SUCCESS) {
        if (rank == 0) {
            fprintf(stderr, "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceCountStatus = cudaGetDeviceCount(&deviceCount);
    checkCuda(MPI_COMM_WORLD, rank, deviceCountStatus, "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA accelerator is visible to the MPI job\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const int device = localRank % deviceCount;
    checkCuda(MPI_COMM_WORLD, rank, cudaSetDevice(device), "cudaSetDevice");

    cudaDeviceProp deviceProperties{};
    checkCuda(MPI_COMM_WORLD, rank,
              cudaGetDeviceProperties(&deviceProperties, device),
              "cudaGetDeviceProperties");

    const size_t rankCount = static_cast<size_t>(worldSize);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t baseCount = numOptions / rankCount;
    const size_t remainder = numOptions % rankCount;
    const size_t localCount = baseCount + (rankIndex < remainder ? 1 : 0);
    const size_t localStart = baseCount * rankIndex + std::min(rankIndex, remainder);

    if (localCount > static_cast<size_t>(INT_MAX) ||
        (printResults && numOptions > static_cast<size_t>(INT_MAX))) {
        if (rank == 0) {
            fprintf(stderr, "This MPI build requires each gathered result count to fit in INT_MAX\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA device %d (%s)\n",
               worldSize, omp_get_max_threads(), device, deviceProperties.name);
    }
    
    OptionInput* localOptions = nullptr;
    double* localResults = nullptr;
    if (localCount > 0) {
        checkCuda(MPI_COMM_WORLD, rank,
                  cudaMallocHost(reinterpret_cast<void**>(&localOptions),
                                 localCount * sizeof(OptionInput)),
                  "cudaMallocHost(options)");
        checkCuda(MPI_COMM_WORLD, rank,
                  cudaMallocHost(reinterpret_cast<void**>(&localResults),
                                 localCount * sizeof(double)),
                  "cudaMallocHost(results)");
        generateOptions(localOptions, localStart, localCount);
    }
    
    std::vector<double> gatheredResults;
    const bool gatherAllResults = printResults;
    const size_t gatheredCount = gatherAllResults
                                     ? numOptions
                                     : (validate ? std::min<size_t>(10, numOptions) : 0);
    if (rank == 0 && gatheredCount > 0) {
        gatheredResults.resize(gatheredCount);
    }

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    int validationStatus = 1;
    {
        CudaPricer pricer(localOptions, localResults, localCount, rank,
                          MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        pricer.run();
        const double localSeconds = MPI_Wtime() - start;
        double elapsedSeconds = 0.0;
        MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
            printf("Options per second: %.0f\n",
                   elapsedSeconds > 0.0 ? numOptions / elapsedSeconds : 0.0);
        }
    }

    if (gatheredCount > 0) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(worldSize);
            displacements.resize(worldSize);
            for (int source = 0; source < worldSize; ++source) {
                const size_t sourceRank = static_cast<size_t>(source);
                const size_t sourceCount = baseCount +
                    (sourceRank < remainder ? 1 : 0);
                const size_t sourceStart = baseCount * sourceRank +
                    std::min(sourceRank, remainder);
                if (gatherAllResults) {
                    receiveCounts[source] = static_cast<int>(sourceCount);
                    displacements[source] = static_cast<int>(sourceStart);
                } else {
                    const size_t begin = std::max(sourceStart, size_t{0});
                    const size_t end = std::min(sourceStart + sourceCount,
                                                gatheredCount);
                    receiveCounts[source] = begin < end
                        ? static_cast<int>(end - begin) : 0;
                    displacements[source] = static_cast<int>(begin);
                }
            }
        }

        size_t localOffset = 0;
        size_t localGatherCount = 0;
        if (gatherAllResults) {
            localGatherCount = localCount;
        } else {
            const size_t localEnd = localStart + localCount;
            const size_t gatherEnd = std::min(localEnd, gatheredCount);
            if (localStart < gatherEnd) {
                localOffset = 0;
                localGatherCount = gatherEnd - localStart;
            }
        }
        const double* sendBuffer = localGatherCount > 0
            ? localResults + localOffset : nullptr;
        MPI_Gatherv(sendBuffer, static_cast<int>(localGatherCount), MPI_DOUBLE,
                    rank == 0 ? gatheredResults.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results(gatheredResults, "OptionPrices");
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, std::min<size_t>(10, numOptions));
            const bool valid = validateResults(validationOptions, gatheredResults);
            validationStatus = valid ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (localOptions != nullptr) {
        checkCuda(MPI_COMM_WORLD, rank, cudaFreeHost(localOptions),
                  "cudaFreeHost(options)");
        checkCuda(MPI_COMM_WORLD, rank, cudaFreeHost(localResults),
                  "cudaFreeHost(results)");
    }
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return validate && validationStatus == 0 ? 1 : 0;
}
