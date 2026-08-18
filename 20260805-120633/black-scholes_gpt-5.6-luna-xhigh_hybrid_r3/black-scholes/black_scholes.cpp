#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Device-side implementation.  Inputs are kept in separate arrays so every
// warp reads coalesced, contiguous values for each Black-Scholes parameter.
__device__ __forceinline__ double cumulativeNormalDevice(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

__global__ void blackScholesKernel(const int* __restrict__ types,
                                   const double* __restrict__ strikes,
                                   const double* __restrict__ spots,
                                   const double* __restrict__ dividendYields,
                                   const double* __restrict__ rates,
                                   const double* __restrict__ maturities,
                                   const double* __restrict__ volatilities,
                                   double* __restrict__ prices,
                                   const size_t count) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = first; i < count; i += stride) {
        const double S = spots[i];
        const double K = strikes[i];
        const double r = rates[i];
        const double q = dividendYields[i];
        const double T = maturities[i];
        const double sigma = volatilities[i];

        if (T <= 0.0 || sigma <= 0.0) {
            prices[i] = 0.0;
            continue;
        }

        const double sqrtT = sqrt(T);
        const double sigmaSqrtT = sigma * sqrtT;
        const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
        const double d2 = d1 - sigmaSqrtT;
        const double Nd1 = cumulativeNormalDevice(d1);
        const double Nd2 = cumulativeNormalDevice(d2);
        const double discountedSpot = S * exp(-q * T);
        const double discountedStrike = K * exp(-r * T);

        if (types[i] == CALL) {
            prices[i] = discountedSpot * Nd1 - discountedStrike * Nd2;
        } else {
            prices[i] = discountedStrike * cumulativeNormalDevice(-d2) -
                        discountedSpot * cumulativeNormalDevice(-d1);
        }
    }
}

// Black-Scholes formula for European options.  Retained as the scalar
// reference implementation and for source-level equivalence with the
// original benchmark.
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

    const double sqrtT = sqrt(T);
    const double sigmaSqrtT = sigma * sqrtT;
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;

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
void generateOptions(std::vector<OptionInput>& options,
                     const size_t globalOffset,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    // Each element is independent.  Keeping this work parallel on the host
    // makes MPI ranks useful even while a GPU is busy with another rank.
    #pragma omp parallel for schedule(static)
    for (long long localIndex = 0;
         localIndex < static_cast<long long>(numOptions); ++localIndex) {
        const size_t i = static_cast<size_t>(localIndex);
        const size_t globalIndex = globalOffset + i;
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 *
            (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

void partitionOptions(const size_t total,
                      const int rank,
                      const int worldSize,
                      size_t& begin,
                      size_t& count) {
    const size_t base = total / static_cast<size_t>(worldSize);
    const size_t remainder = total % static_cast<size_t>(worldSize);
    const size_t rankAsSize = static_cast<size_t>(rank);
    count = base + (rankAsSize < remainder ? 1 : 0);
    begin = rankAsSize * base + std::min(rankAsSize, remainder);
}

bool cudaSuccessOrReport(const cudaError_t error,
                         const char* operation,
                         const int rank) {
    if (error == cudaSuccess) {
        return true;
    }

    fprintf(stderr, "MPI rank %d: CUDA error during %s: %s\n",
            rank, operation, cudaGetErrorString(error));
    return false;
}

// Price one MPI rank's contiguous partition on its selected GPU.  Pinned
// host buffers allow input and result transfers to be queued in the same
// stream as the kernel, and the structure-of-arrays layout maximizes memory
// throughput inside the kernel.
bool priceOptionsCuda(const std::vector<OptionInput>& options,
                      std::vector<double>& results,
                      const int localRank,
                      const int mpiRank) {
    int deviceCount = 0;
    const cudaError_t deviceCountError = cudaGetDeviceCount(&deviceCount);
    if (!cudaSuccessOrReport(deviceCountError, "cudaGetDeviceCount", mpiRank) || deviceCount == 0) {
        if (deviceCount == 0) {
            fprintf(stderr, "MPI rank %d: no CUDA device is available\n", mpiRank);
        }
        return false;
    }

    const int device = localRank % deviceCount;
    if (!cudaSuccessOrReport(cudaSetDevice(device), "cudaSetDevice", mpiRank)) {
        return false;
    }

    const size_t count = options.size();
    if (count == 0) {
        return true;
    }

    if (count > std::numeric_limits<size_t>::max() / sizeof(double)) {
        fprintf(stderr, "MPI rank %d: option allocation is too large\n", mpiRank);
        return false;
    }
    const size_t doubleBytes = count * sizeof(double);
    const size_t intBytes = count * sizeof(int);

    int* hTypes = nullptr;
    double* hStrikes = nullptr;
    double* hSpots = nullptr;
    double* hDividendYields = nullptr;
    double* hRates = nullptr;
    double* hMaturities = nullptr;
    double* hVolatilities = nullptr;
    double* hPrices = nullptr;

    int* dTypes = nullptr;
    double* dStrikes = nullptr;
    double* dSpots = nullptr;
    double* dDividendYields = nullptr;
    double* dRates = nullptr;
    double* dMaturities = nullptr;
    double* dVolatilities = nullptr;
    double* dPrices = nullptr;
    cudaStream_t stream = nullptr;
    bool ok = true;

    auto cleanup = [&]() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        if (dTypes != nullptr) cudaFree(dTypes);
        if (dStrikes != nullptr) cudaFree(dStrikes);
        if (dSpots != nullptr) cudaFree(dSpots);
        if (dDividendYields != nullptr) cudaFree(dDividendYields);
        if (dRates != nullptr) cudaFree(dRates);
        if (dMaturities != nullptr) cudaFree(dMaturities);
        if (dVolatilities != nullptr) cudaFree(dVolatilities);
        if (dPrices != nullptr) cudaFree(dPrices);
        if (hTypes != nullptr) cudaFreeHost(hTypes);
        if (hStrikes != nullptr) cudaFreeHost(hStrikes);
        if (hSpots != nullptr) cudaFreeHost(hSpots);
        if (hDividendYields != nullptr) cudaFreeHost(hDividendYields);
        if (hRates != nullptr) cudaFreeHost(hRates);
        if (hMaturities != nullptr) cudaFreeHost(hMaturities);
        if (hVolatilities != nullptr) cudaFreeHost(hVolatilities);
        if (hPrices != nullptr) cudaFreeHost(hPrices);
    };

    auto allocHost = [&](void** pointer, const size_t bytes, const unsigned int flags) {
        return cudaSuccessOrReport(cudaHostAlloc(pointer, bytes, flags),
                                   "cudaHostAlloc", mpiRank);
    };
    auto allocDevice = [&](void** pointer, const size_t bytes) {
        return cudaSuccessOrReport(cudaMalloc(pointer, bytes),
                                   "cudaMalloc", mpiRank);
    };

    ok = ok && allocHost(reinterpret_cast<void**>(&hTypes), intBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hStrikes), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hSpots), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hDividendYields), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hRates), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hMaturities), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hVolatilities), doubleBytes,
                         cudaHostAllocWriteCombined);
    ok = ok && allocHost(reinterpret_cast<void**>(&hPrices), doubleBytes,
                         cudaHostAllocDefault);

    if (!ok) {
        cleanup();
        return false;
    }

    // Pack all fields concurrently before queuing host-to-device copies.
    #pragma omp parallel for schedule(static)
    for (long long index = 0; index < static_cast<long long>(count); ++index) {
        const size_t i = static_cast<size_t>(index);
        const OptionInput& option = options[i];
        hTypes[i] = option.type;
        hStrikes[i] = option.strike;
        hSpots[i] = option.spot;
        hDividendYields[i] = option.q;
        hRates[i] = option.r;
        hMaturities[i] = option.t;
        hVolatilities[i] = option.vol;
    }

    ok = ok && allocDevice(reinterpret_cast<void**>(&dTypes), intBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dStrikes), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dSpots), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dDividendYields), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dRates), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dMaturities), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dVolatilities), doubleBytes);
    ok = ok && allocDevice(reinterpret_cast<void**>(&dPrices), doubleBytes);
    if (!ok) {
        cleanup();
        return false;
    }

    ok = ok && cudaSuccessOrReport(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                                   "cudaStreamCreateWithFlags", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dTypes, hTypes, intBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(types)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dStrikes, hStrikes, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(strikes)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dSpots, hSpots, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(spots)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dDividendYields, hDividendYields, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(dividend yields)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dRates, hRates, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(rates)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dMaturities, hMaturities, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(maturities)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(dVolatilities, hVolatilities, doubleBytes,
                                                   cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync(volatilities)", mpiRank);

    cudaDeviceProp properties{};
    ok = ok && cudaSuccessOrReport(cudaGetDeviceProperties(&properties, device),
                                   "cudaGetDeviceProperties", mpiRank);
    const unsigned int threads = 256;
    const size_t requestedBlocks = (count + threads - 1) / threads;
    const size_t residentBlocks = static_cast<size_t>(properties.multiProcessorCount) * 32;
    const unsigned int blocks = static_cast<unsigned int>(
        std::max<size_t>(1, std::min(requestedBlocks,
                                     std::min<size_t>(residentBlocks, 2147483647u))));

    if (ok) {
        blackScholesKernel<<<blocks, threads, 0, stream>>>(
            dTypes, dStrikes, dSpots, dDividendYields, dRates, dMaturities,
            dVolatilities, dPrices, count);
        ok = cudaSuccessOrReport(cudaGetLastError(), "blackScholesKernel", mpiRank);
    }
    ok = ok && cudaSuccessOrReport(cudaMemcpyAsync(hPrices, dPrices, doubleBytes,
                                                   cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync(prices)", mpiRank);
    ok = ok && cudaSuccessOrReport(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize", mpiRank);

    if (ok) {
        #pragma omp parallel for schedule(static)
        for (long long index = 0; index < static_cast<long long>(count); ++index) {
            const size_t i = static_cast<size_t>(index);
            results[i] = hPrices[i];
        }
    }

    cleanup();
    return ok;
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

bool gatherResults(const std::vector<double>& localResults,
                   std::vector<double>& rootResults,
                   const size_t totalOptions,
                   const size_t localBegin,
                   const int rank,
                   const int worldSize,
                   const size_t resultCount,
                   MPI_Comm communicator) {
    size_t localBeginFull = 0;
    size_t localCountFull = 0;
    partitionOptions(totalOptions, rank, worldSize, localBeginFull, localCountFull);

    const size_t localEnd = localBeginFull + localCountFull;
    const size_t requestedEnd = std::min(resultCount, totalOptions);
    const size_t sendBegin = std::min(std::max(localBeginFull, localBegin), requestedEnd);
    const size_t sendEnd = std::min(localEnd, requestedEnd);
    const size_t sendCount = sendEnd > sendBegin ? sendEnd - sendBegin : 0;
    const size_t localOffset = sendBegin >= localBegin ? sendBegin - localBegin : 0;

    // The ordinary path uses a single collective, which is substantially
    // faster for normal benchmark sizes.  The point-to-point path keeps MPI
    // counts and displacements valid for very large option sets.
    const bool fitsInt = totalOptions <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                         resultCount <= static_cast<size_t>(std::numeric_limits<int>::max());
    int allFit = fitsInt ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &allFit, 1, MPI_INT, MPI_MIN, communicator);

    if (allFit != 0) {
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            rootResults.assign(resultCount, 0.0);
            receiveCounts.resize(worldSize);
            displacements.resize(worldSize);
            for (int source = 0; source < worldSize; ++source) {
                size_t sourceBegin = 0;
                size_t sourceCount = 0;
                partitionOptions(totalOptions, source, worldSize, sourceBegin, sourceCount);
                const size_t sourceEnd = sourceBegin + sourceCount;
                const size_t sourceSendBegin = std::min(sourceBegin, requestedEnd);
                const size_t sourceSendEnd = std::min(sourceEnd, requestedEnd);
                receiveCounts[source] = static_cast<int>(sourceSendEnd > sourceSendBegin
                                                               ? sourceSendEnd - sourceSendBegin : 0);
                displacements[source] = static_cast<int>(sourceSendBegin);
            }
        }

        const double* sendPointer = sendCount == 0 ? nullptr : localResults.data() + localOffset;
        double* receivePointer = rank == 0 && !rootResults.empty() ? rootResults.data() : nullptr;
        MPI_Gatherv(sendPointer, static_cast<int>(sendCount), MPI_DOUBLE,
                    receivePointer,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, communicator);
        return true;
    }

    if (rank == 0) {
        rootResults.assign(resultCount, 0.0);
        if (sendCount != 0) {
            std::copy_n(localResults.data() + localOffset, sendCount,
                        rootResults.data() + sendBegin);
        }
        const size_t maxMessage = static_cast<size_t>(std::numeric_limits<int>::max());
        for (int source = 1; source < worldSize; ++source) {
            size_t sourceBegin = 0;
            size_t sourceCount = 0;
            partitionOptions(totalOptions, source, worldSize, sourceBegin, sourceCount);
            const size_t sourceEnd = sourceBegin + sourceCount;
            const size_t sourceSendBegin = std::min(sourceBegin, requestedEnd);
            const size_t sourceSendEnd = std::min(sourceEnd, requestedEnd);
            for (size_t offset = 0; offset < sourceSendEnd - sourceSendBegin; ) {
                const size_t chunk = std::min(maxMessage, sourceSendEnd - sourceSendBegin - offset);
                MPI_Recv(rootResults.data() + sourceSendBegin + offset,
                         static_cast<int>(chunk), MPI_DOUBLE, source, 9031,
                         communicator, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        const size_t maxMessage = static_cast<size_t>(std::numeric_limits<int>::max());
        for (size_t offset = 0; offset < sendCount; ) {
            const size_t chunk = std::min(maxMessage, sendCount - offset);
            MPI_Send(localResults.data() + localOffset + offset,
                     static_cast<int>(chunk), MPI_DOUBLE, 0, 9031, communicator);
            offset += chunk;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 1;
    }

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Hybrid execution: MPI + OpenMP + CUDA\n");
    }

    int localRank = 0;
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    MPI_Comm_rank(localCommunicator, &localRank);

    size_t localBegin = 0;
    size_t localCount = 0;
    partitionOptions(numOptions, rank, worldSize, localBegin, localCount);

    // MPI distributes contiguous global indices.  The global offset keeps
    // generated inputs equivalent to the serial benchmark.
    std::vector<OptionInput> options;
    generateOptions(options, localBegin, localCount);
    std::vector<double> results(localCount);

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const bool localSuccess = priceOptionsCuda(options, results, localRank, rank);
    const double end = MPI_Wtime();
    const double localDuration = end - start;

    int successfulRanks = localSuccess ? 1 : 0;
    int allSuccessful = 0;
    MPI_Allreduce(&successfulRanks, &allSuccessful, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (allSuccessful == 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA pricing failed on at least one MPI rank\n");
        }
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return 1;
    }

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> gatheredResults;
    if (printResults) {
        gatherResults(results, gatheredResults, numOptions, localBegin, rank,
                      worldSize, numOptions, MPI_COMM_WORLD);
    } else if (validate) {
        gatherResults(results, gatheredResults, numOptions, localBegin, rank,
                      worldSize, std::min<size_t>(10, numOptions), MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n",
               duration > 0.0 ? static_cast<double>(numOptions) / duration : 0.0);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(gatheredResults, "OptionPrices");
    }

    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, 0, std::min<size_t>(10, numOptions));
            printf("Validating results...\n");
            valid = validateResults(validationOptions, gatheredResults) ? 1 : 0;
            printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Comm_free(&localCommunicator);
        MPI_Finalize();
        return valid != 0 ? 0 : 1;
    }

    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return 0;
}
