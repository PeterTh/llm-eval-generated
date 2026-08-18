#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr int kThreadsPerBlock = 256;

enum OptionType : int {
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

OptionInput makeOption(const std::size_t index) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[index % testOptions.size()];
    const double factor = 1.0 + 0.1 *
        (index / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

void mpiCheck(const int status, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, message, &length);
    std::fprintf(stderr, "MPI error in %s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
}

void cudaCheck(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "Rank %d CUDA error in %s: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
}

std::size_t rankCount(const std::size_t total, const int rank,
                      const int ranks) noexcept {
    const std::size_t quotient = total / static_cast<std::size_t>(ranks);
    const std::size_t remainder = total % static_cast<std::size_t>(ranks);
    return quotient + (static_cast<std::size_t>(rank) < remainder ? 1U : 0U);
}

std::size_t rankOffset(const std::size_t total, const int rank,
                       const int ranks) noexcept {
    const std::size_t quotient = total / static_cast<std::size_t>(ranks);
    const std::size_t remainder = total % static_cast<std::size_t>(ranks);
    return quotient * static_cast<std::size_t>(rank) +
           std::min(static_cast<std::size_t>(rank), remainder);
}

std::size_t alignUp(const std::size_t value, const std::size_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

struct OptionArrays {
    int* type = nullptr;
    double* strike = nullptr;
    double* spot = nullptr;
    double* q = nullptr;
    double* r = nullptr;
    double* t = nullptr;
    double* vol = nullptr;
};

OptionArrays mapOptionArrays(void* storage, const std::size_t count) noexcept {
    auto* bytes = static_cast<unsigned char*>(storage);
    OptionArrays arrays;
    arrays.type = reinterpret_cast<int*>(bytes);
    bytes += alignUp(count * sizeof(int), 256U);
    arrays.strike = reinterpret_cast<double*>(bytes);
    arrays.spot = arrays.strike + count;
    arrays.q = arrays.spot + count;
    arrays.r = arrays.q + count;
    arrays.t = arrays.r + count;
    arrays.vol = arrays.t + count;
    return arrays;
}

std::size_t optionStorageBytes(const std::size_t count) {
    constexpr std::size_t arraysBytesPerOption = 6U * sizeof(double);
    if (count > (std::numeric_limits<std::size_t>::max() - 255U) /
                    (arraysBytesPerOption + sizeof(int))) {
        std::fprintf(stderr, "Requested option count is too large\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return alignUp(count * sizeof(int), 256U) +
           count * arraysBytesPerOption;
}

void fillOptions(const OptionArrays& arrays, const std::size_t globalOffset,
                 const std::size_t count) {
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < count; ++i) {
        const OptionInput option = makeOption(globalOffset + i);
        arrays.type[i] = option.type;
        arrays.strike[i] = option.strike;
        arrays.spot[i] = option.spot;
        arrays.q[i] = option.q;
        arrays.r[i] = option.r;
        arrays.t[i] = option.t;
        arrays.vol[i] = option.vol;
    }
}

__device__ __forceinline__ double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + erf(x * kInvSqrt2));
}

__global__ void blackScholesKernel(
    const int* __restrict__ type, const double* __restrict__ strike,
    const double* __restrict__ spot, const double* __restrict__ q,
    const double* __restrict__ r, const double* __restrict__ t,
    const double* __restrict__ vol, double* __restrict__ result,
    const std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         i < count; i += stride) {
        const double maturity = t[i];
        const double sigma = vol[i];
        if (maturity <= 0.0 || sigma <= 0.0) {
            result[i] = 0.0;
            continue;
        }

        const double s = spot[i];
        const double k = strike[i];
        const double sqrtT = sqrt(maturity);
        const double sigmaSqrtT = sigma * sqrtT;
        const double d1 = (log(s / k) +
                           (r[i] - q[i] + 0.5 * sigma * sigma) * maturity) /
                          sigmaSqrtT;
        const double d2 = d1 - sigmaSqrtT;
        const double discountedS = s * exp(-q[i] * maturity);
        const double discountedK = k * exp(-r[i] * maturity);

        if (type[i] == CALL) {
            result[i] = discountedS * cumulativeNormal(d1) -
                        discountedK * cumulativeNormal(d2);
        } else {
            result[i] = discountedK * cumulativeNormal(-d2) -
                        discountedS * cumulativeNormal(-d1);
        }
    }
}

void gatherResults(const double* localResults, const std::size_t localCount,
                   const std::size_t total, const int rank, const int ranks,
                   std::vector<double>& globalResults) {
    constexpr int resultTag = 721;
    constexpr std::size_t maxChunk =
        static_cast<std::size_t>(std::numeric_limits<int>::max());

    if (rank == 0) {
        globalResults.resize(total);
        if (localCount != 0U) {
            std::copy_n(localResults, localCount, globalResults.data());
        }
        for (int source = 1; source < ranks; ++source) {
            const std::size_t sourceCount = rankCount(total, source, ranks);
            std::size_t received = 0;
            const std::size_t offset = rankOffset(total, source, ranks);
            while (received < sourceCount) {
                const std::size_t chunk = std::min(maxChunk, sourceCount - received);
                mpiCheck(MPI_Recv(globalResults.data() + offset + received,
                                  static_cast<int>(chunk), MPI_DOUBLE, source,
                                  resultTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                         "MPI_Recv(results)");
                received += chunk;
            }
        }
    } else {
        std::size_t sent = 0;
        while (sent < localCount) {
            const std::size_t chunk = std::min(maxChunk, localCount - sent);
            mpiCheck(MPI_Send(localResults + sent, static_cast<int>(chunk),
                              MPI_DOUBLE, 0, resultTag, MPI_COMM_WORLD),
                     "MPI_Send(results)");
            sent += chunk;
        }
    }
}

bool validateResults(const std::vector<double>& results) {
    const std::size_t numChecks = std::min<std::size_t>(10U, results.size());
    bool allPassed = true;

    std::printf("Checking computed option prices:\n");
    for (std::size_t i = 0; i < numChecks; ++i) {
        const OptionInput option = makeOption(i);
        const double computed = results[i];
        const double error = std::fabs(computed - option.value);
        const double relError = error / (std::fabs(option.value) + 1.0e-10);
        std::printf("  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
                    i, computed, option.value, relError);
        if (computed < 0.0 || computed > 1000.0 || !std::isfinite(computed)) {
            std::printf("Validation failed at option %zu: invalid value %.4f\n",
                        i, computed);
            allPassed = false;
        }
    }
    return allPassed;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseCount(const char* text, std::size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-') {
        return false;
    }
    if (parsed > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided),
             "MPI_Init_thread");

    int rank = 0;
    int ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseCount(argv[++i], numOptions)) {
                argumentsValid = false;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localComm),
             "MPI_Comm_split_type");
    int localRank = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "MPI_Comm_rank(local)");

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA-capable GPU is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    cudaCheck(cudaFree(nullptr), "CUDA context initialization");

    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties");

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const std::size_t localCount = rankCount(numOptions, rank, ranks);
    const std::size_t globalOffset = rankOffset(numOptions, rank, ranks);

    void* hostInputStorage = nullptr;
    void* deviceInputStorage = nullptr;
    double* hostResults = nullptr;
    double* deviceResults = nullptr;
    std::size_t storageBytes = 0;
    OptionArrays hostOptions{};
    OptionArrays deviceOptions{};

    if (localCount != 0U) {
        storageBytes = optionStorageBytes(localCount);
        cudaCheck(cudaMallocHost(&hostInputStorage, storageBytes),
                  "cudaMallocHost(inputs)");
        cudaCheck(cudaMalloc(&deviceInputStorage, storageBytes),
                  "cudaMalloc(inputs)");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostResults),
                                 localCount * sizeof(double)),
                  "cudaMallocHost(results)");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceResults),
                             localCount * sizeof(double)),
                  "cudaMalloc(results)");

        hostOptions = mapOptionArrays(hostInputStorage, localCount);
        deviceOptions = mapOptionArrays(deviceInputStorage, localCount);
        fillOptions(hostOptions, globalOffset, localCount);
        cudaCheck(cudaMemcpy(deviceInputStorage, hostInputStorage, storageBytes,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(inputs)");

        // Force lazy module loading and kernel setup out of the measured region.
        blackScholesKernel<<<1, kThreadsPerBlock>>>(
            deviceOptions.type, deviceOptions.strike, deviceOptions.spot,
            deviceOptions.q, deviceOptions.r, deviceOptions.t, deviceOptions.vol,
            deviceResults, 1U);
        cudaCheck(cudaGetLastError(), "blackScholesKernel warm-up launch");
        cudaCheck(cudaDeviceSynchronize(), "blackScholesKernel warm-up");
    }

    if (rank == 0) {
        std::printf("Pricing options...\n");
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier(start)");
    const double start = MPI_Wtime();

    if (localCount != 0U) {
        const std::size_t requiredBlocks =
            (localCount + kThreadsPerBlock - 1U) / kThreadsPerBlock;
        const std::size_t residentGrid =
            static_cast<std::size_t>(properties.multiProcessorCount) * 8U;
        const int blocks = static_cast<int>(std::min(requiredBlocks, residentGrid));
        blackScholesKernel<<<blocks, kThreadsPerBlock>>>(
            deviceOptions.type, deviceOptions.strike, deviceOptions.spot,
            deviceOptions.q, deviceOptions.r, deviceOptions.t, deviceOptions.vol,
            deviceResults, localCount);
        cudaCheck(cudaGetLastError(), "blackScholesKernel launch");
        cudaCheck(cudaMemcpy(hostResults, deviceResults,
                             localCount * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(results)");
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    mpiCheck(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce(time)");

    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double rate = elapsed > 0.0 ? numOptions / elapsed : 0.0;
        std::printf("Options per second: %.0f\n", rate);
    }

    std::vector<double> results;
    if (printResults || validate) {
        gatherResults(hostResults, localCount, numOptions, rank, ranks, results);
    }

    int returnCode = EXIT_SUCCESS;
    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }
    if (rank == 0 && validate) {
        std::printf("Validating results...\n");
        if (validateResults(results)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            returnCode = EXIT_FAILURE;
        }
    }
    mpiCheck(MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "MPI_Bcast(return code)");

    if (deviceResults != nullptr) {
        cudaCheck(cudaFree(deviceResults), "cudaFree(results)");
        cudaCheck(cudaFree(deviceInputStorage), "cudaFree(inputs)");
        cudaCheck(cudaFreeHost(hostResults), "cudaFreeHost(results)");
        cudaCheck(cudaFreeHost(hostInputStorage), "cudaFreeHost(inputs)");
    }
    mpiCheck(MPI_Comm_free(&localComm), "MPI_Comm_free");
    mpiCheck(MPI_Finalize(), "MPI_Finalize");
    return returnCode;
}
