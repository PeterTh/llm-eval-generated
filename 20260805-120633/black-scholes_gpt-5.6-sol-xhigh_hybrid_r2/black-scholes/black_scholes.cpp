#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr double kInvSqrtTwo = 0.707106781186547524400844362104849039;
constexpr std::size_t kNumericFields = 6;
constexpr std::size_t kChunkOptions = 1U << 20;
constexpr int kMaxStreams = 4;
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

void mpiCheck(const int status, const char* expression, const char* file,
              const int line) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(status, error, &length);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "MPI error on rank %d at %s:%d (%s): %.*s\n",
                 rank, file, line, expression, length, error);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

#define MPI_CHECK(expression) \
    mpiCheck((expression), #expression, __FILE__, __LINE__)

void cudaCheck(const cudaError_t status, const char* expression,
               const char* file, const int line) {
    if (status == cudaSuccess) {
        return;
    }

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::fprintf(stderr, "CUDA error on rank %d at %s:%d (%s): %s\n",
                 rank, file, line, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)

struct Range {
    std::size_t begin;
    std::size_t count;
};

Range rankRange(const std::size_t total, const int rank,
                const int rankCount) noexcept {
    const std::size_t ranks = static_cast<std::size_t>(rankCount);
    const std::size_t rankIndex = static_cast<std::size_t>(rank);
    const std::size_t base = total / ranks;
    const std::size_t remainder = total % ranks;
    const std::size_t count = base + (rankIndex < remainder ? 1U : 0U);
    const std::size_t begin = rankIndex * base + std::min(rankIndex, remainder);
    return {begin, count};
}

OptionInput makeOption(const std::size_t globalIndex) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[globalIndex % testOptions.size()];
    const double factor =
        1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    PinnedBuffer(const std::size_t count, const unsigned int flags)
        : count_(count) {
        if (count_ == 0) {
            return;
        }
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            std::fprintf(stderr, "Pinned allocation size overflow\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            std::abort();
        }
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data_),
                                 count_ * sizeof(T), flags));
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct HostOptions {
    explicit HostOptions(const std::size_t count)
        : count(count),
          numeric(count * kNumericFields,
                  cudaHostAllocPortable | cudaHostAllocWriteCombined),
          types(count,
                cudaHostAllocPortable | cudaHostAllocWriteCombined) {}

    double* strike() noexcept { return numeric.data(); }
    double* spot() noexcept { return numeric.data() + count; }
    double* q() noexcept { return numeric.data() + 2 * count; }
    double* r() noexcept { return numeric.data() + 3 * count; }
    double* t() noexcept { return numeric.data() + 4 * count; }
    double* vol() noexcept { return numeric.data() + 5 * count; }

    const double* field(const std::size_t fieldIndex) const noexcept {
        return numeric.data() + fieldIndex * count;
    }

    std::size_t count;
    PinnedBuffer<double> numeric;
    PinnedBuffer<int> types;
};

void generateOptions(HostOptions& options, const std::size_t globalBegin) {
    // Each MPI rank first-touches its pinned input arrays in parallel.  The
    // structure-of-arrays layout gives the CUDA kernel coalesced loads.
#pragma omp parallel for schedule(static)
    for (std::int64_t localIndex = 0;
         localIndex < static_cast<std::int64_t>(options.count); ++localIndex) {
        const std::size_t i = static_cast<std::size_t>(localIndex);
        const OptionInput option = makeOption(globalBegin + i);
        options.types.data()[i] = option.type;
        options.strike()[i] = option.strike;
        options.spot()[i] = option.spot;
        options.q()[i] = option.q;
        options.r()[i] = option.r;
        options.t()[i] = option.t;
        options.vol()[i] = option.vol;
    }
}

__device__ __forceinline__ double cumulativeNormalDevice(const double x) {
    return 0.5 * (1.0 + erf(x * kInvSqrtTwo));
}

__global__ __launch_bounds__(kThreadsPerBlock)
void blackScholesKernel(const int* __restrict__ types,
                        const double* __restrict__ strike,
                        const double* __restrict__ spot,
                        const double* __restrict__ q,
                        const double* __restrict__ r,
                        const double* __restrict__ t,
                        const double* __restrict__ vol,
                        double* __restrict__ results,
                        const std::size_t count) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) +
                          threadIdx.x;
    if (i >= count) {
        return;
    }

    const double time = t[i];
    const double sigma = vol[i];
    if (time <= 0.0 || sigma <= 0.0) {
        results[i] = 0.0;
        return;
    }

    const double stock = spot[i];
    const double exercise = strike[i];
    const double sqrtTime = sqrt(time);
    const double sigmaSqrtTime = sigma * sqrtTime;
    const double d1 =
        (log(stock / exercise) +
         (r[i] - q[i] + 0.5 * sigma * sigma) * time) /
        sigmaSqrtTime;
    const double d2 = d1 - sigmaSqrtTime;
    const double discountedStock = stock * exp(-q[i] * time);
    const double discountedStrike = exercise * exp(-r[i] * time);
    const double call =
        discountedStock * cumulativeNormalDevice(d1) -
        discountedStrike * cumulativeNormalDevice(d2);

    // Put-call parity avoids a divergent second formula while retaining the
    // same Black-Scholes semantics for both option types.
    results[i] = types[i] == CALL
                     ? call
                     : call - discountedStock + discountedStrike;
}

class GpuExecutor {
public:
    explicit GpuExecutor(const std::size_t optionCount)
        : optionCount_(optionCount),
          capacity_(std::min(optionCount, kChunkOptions)) {
        if (optionCount_ == 0) {
            return;
        }

        const std::size_t chunkCount =
            (optionCount_ + kChunkOptions - 1) / kChunkOptions;
        streamCount_ = static_cast<int>(
            std::min<std::size_t>(kMaxStreams, chunkCount));

        for (int i = 0; i < streamCount_; ++i) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams_[i],
                                                  cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&numeric_[i]),
                                  capacity_ * kNumericFields * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&types_[i]),
                                  capacity_ * sizeof(int)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&results_[i]),
                                  capacity_ * sizeof(double)));
        }
    }

    GpuExecutor(const GpuExecutor&) = delete;
    GpuExecutor& operator=(const GpuExecutor&) = delete;

    ~GpuExecutor() {
        for (int i = 0; i < streamCount_; ++i) {
            cudaFree(numeric_[i]);
            cudaFree(types_[i]);
            cudaFree(results_[i]);
            cudaStreamDestroy(streams_[i]);
        }
    }

    void execute(const HostOptions& options, double* hostResults) {
        for (std::size_t chunkIndex = 0, begin = 0; begin < optionCount_;
             ++chunkIndex, begin += kChunkOptions) {
            const int streamIndex =
                static_cast<int>(chunkIndex % streamCount_);
            const cudaStream_t stream = streams_[streamIndex];
            const std::size_t count =
                std::min(kChunkOptions, optionCount_ - begin);

            // A stream's fixed-size staging buffers can be reused once its
            // previous D2H copy has completed. Other streams continue running.
            if (chunkIndex >= static_cast<std::size_t>(streamCount_)) {
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            for (std::size_t field = 0; field < kNumericFields; ++field) {
                CUDA_CHECK(cudaMemcpyAsync(
                    numeric_[streamIndex] + field * capacity_,
                    options.field(field) + begin, count * sizeof(double),
                    cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(types_[streamIndex],
                                       options.types.data() + begin,
                                       count * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));

            const unsigned int blocks = static_cast<unsigned int>(
                (count + kThreadsPerBlock - 1) / kThreadsPerBlock);
            blackScholesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
                types_[streamIndex], numeric_[streamIndex],
                numeric_[streamIndex] + capacity_,
                numeric_[streamIndex] + 2 * capacity_,
                numeric_[streamIndex] + 3 * capacity_,
                numeric_[streamIndex] + 4 * capacity_,
                numeric_[streamIndex] + 5 * capacity_, results_[streamIndex],
                count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hostResults + begin,
                                       results_[streamIndex],
                                       count * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        }

        for (int i = 0; i < streamCount_; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
        }
    }

private:
    std::size_t optionCount_ = 0;
    std::size_t capacity_ = 0;
    int streamCount_ = 0;
    std::array<cudaStream_t, kMaxStreams> streams_{};
    std::array<double*, kMaxStreams> numeric_{};
    std::array<int*, kMaxStreams> types_{};
    std::array<double*, kMaxStreams> results_{};
};

double blackScholesHost(const OptionInput& option) noexcept {
    if (option.t <= 0.0 || option.vol <= 0.0) {
        return 0.0;
    }

    const double sqrtTime = std::sqrt(option.t);
    const double sigmaSqrtTime = option.vol * sqrtTime;
    const double d1 =
        (std::log(option.spot / option.strike) +
         (option.r - option.q + 0.5 * option.vol * option.vol) * option.t) /
        sigmaSqrtTime;
    const double d2 = d1 - sigmaSqrtTime;
    const auto cdf = [](const double x) {
        return 0.5 * (1.0 + std::erf(x * kInvSqrtTwo));
    };
    const double discountedStock = option.spot * std::exp(-option.q * option.t);
    const double discountedStrike =
        option.strike * std::exp(-option.r * option.t);
    if (option.type == CALL) {
        return discountedStock * cdf(d1) - discountedStrike * cdf(d2);
    }
    return discountedStrike * cdf(-d2) - discountedStock * cdf(-d1);
}

bool validateResults(const std::size_t numOptions,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const std::size_t numChecks = std::min<std::size_t>(10, numOptions);

    std::printf("Checking computed option prices:\n");
    for (std::size_t i = 0; i < numChecks; ++i) {
        const OptionInput option = makeOption(i);
        const double computed = results[i];
        const double expected = option.value;
        const double relativeKnownError =
            std::fabs(computed - expected) / (std::fabs(expected) + 1.0e-10);
        const double reference = blackScholesHost(option);
        const double referenceError = std::fabs(computed - reference);
        const double referenceTolerance =
            5.0e-12 * std::max(1.0, std::fabs(reference));

        std::printf(
            "  Option %zu: computed=%.4f, expected=%.4f, rel_error=%.4e\n",
            i, computed, expected, relativeKnownError);

        if (!std::isfinite(computed) || computed < 0.0 ||
            referenceError > referenceTolerance) {
            std::printf(
                "Validation failed at option %zu: computed=%.17g, "
                "CPU_reference=%.17g\n",
                i, computed, reference);
            allPassed = false;
        }
    }
    return allPassed;
}

void gatherResults(const double* localResults, const Range localRange,
                   const std::size_t total, const int rank,
                   const int rankCount, std::vector<double>& globalResults) {
    if (rank == 0) {
        globalResults.resize(total);
    }

    // MPI_Gatherv uses int counts and displacements. Windowing keeps both in
    // range even when the global problem contains more than INT_MAX options.
    constexpr std::size_t maxWindow = 1U << 30;
    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        counts.resize(static_cast<std::size_t>(rankCount));
        displacements.resize(static_cast<std::size_t>(rankCount));
    }

    for (std::size_t windowBegin = 0; windowBegin < total;
         windowBegin += maxWindow) {
        const std::size_t windowEnd =
            std::min(total, windowBegin + maxWindow);
        const std::size_t localEnd = localRange.begin + localRange.count;
        const std::size_t sendBegin = std::max(windowBegin, localRange.begin);
        const std::size_t sendEnd = std::min(windowEnd, localEnd);
        const int sendCount = sendEnd > sendBegin
                                  ? static_cast<int>(sendEnd - sendBegin)
                                  : 0;
        const double* sendBuffer =
            sendCount == 0
                ? localResults
                : localResults + (sendBegin - localRange.begin);

        if (rank == 0) {
            for (int source = 0; source < rankCount; ++source) {
                const Range sourceRange = rankRange(total, source, rankCount);
                const std::size_t sourceEnd =
                    sourceRange.begin + sourceRange.count;
                const std::size_t receiveBegin =
                    std::max(windowBegin, sourceRange.begin);
                const std::size_t receiveEnd =
                    std::min(windowEnd, sourceEnd);
                counts[static_cast<std::size_t>(source)] =
                    receiveEnd > receiveBegin
                        ? static_cast<int>(receiveEnd - receiveBegin)
                        : 0;
                displacements[static_cast<std::size_t>(source)] =
                    receiveEnd > receiveBegin
                        ? static_cast<int>(receiveBegin - windowBegin)
                        : 0;
            }
        }

        MPI_CHECK(MPI_Gatherv(
            sendBuffer, sendCount, MPI_DOUBLE,
            rank == 0 ? globalResults.data() + windowBegin : nullptr,
            rank == 0 ? counts.data() : nullptr,
            rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
            MPI_COMM_WORLD));
    }
}

void gatherValidationPrefix(const double* localResults, const Range localRange,
                            const std::size_t total, const int rank,
                            std::vector<double>& validationResults) {
    constexpr std::size_t maxChecks = 10;
    const std::size_t checkCount = std::min(maxChecks, total);
    std::array<double, maxChecks> localValues{};
    std::array<double, maxChecks> globalValues{};
    const std::size_t localEnd = localRange.begin + localRange.count;
    const std::size_t copyEnd = std::min(checkCount, localEnd);
    for (std::size_t globalIndex = localRange.begin;
         globalIndex < copyEnd; ++globalIndex) {
        localValues[globalIndex] =
            localResults[globalIndex - localRange.begin];
    }

    MPI_CHECK(MPI_Reduce(localValues.data(), globalValues.data(),
                         static_cast<int>(checkCount), MPI_DOUBLE, MPI_SUM, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) {
        validationResults.assign(globalValues.begin(),
                                 globalValues.begin() + checkCount);
    }
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of options to price (default: 10000)\n");
    std::printf("  -v           Enable validation against known values\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

enum ParseStatus : int {
    PARSE_OK = 0,
    PARSE_HELP = 1,
    PARSE_ERROR = 2
};

ParseStatus parseArguments(const int argc, char** argv,
                           std::uint64_t& numOptions, int& validate,
                           int& printResults) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            if (text[0] == '-') {
                std::printf("Invalid option count: %s\n", text);
                printUsage(argv[0]);
                return PARSE_ERROR;
            }
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(text, &end, 10);
            if (errno == ERANGE || end == text || *end != '\0') {
                std::printf("Invalid option count: %s\n", text);
                printUsage(argv[0]);
                return PARSE_ERROR;
            }
            numOptions = static_cast<std::uint64_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return PARSE_HELP;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return PARSE_ERROR;
        }
    }
    return PARSE_OK;
}

}  // namespace

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &providedThreadLevel));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    int rank = 0;
    int rankCount = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &rankCount));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    std::uint64_t optionCount64 = 10000;
    int validate = 0;
    int printResults = 0;
    int parseStatus = PARSE_OK;
    if (rank == 0) {
        parseStatus =
            parseArguments(argc, argv, optionCount64, validate, printResults);
    }
    MPI_CHECK(MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (parseStatus != PARSE_OK) {
        MPI_CHECK(MPI_Finalize());
        return parseStatus == PARSE_HELP ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    MPI_CHECK(MPI_Bcast(&optionCount64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (optionCount64 > static_cast<std::uint64_t>(
                            std::numeric_limits<std::int64_t>::max()) ||
        optionCount64 > std::numeric_limits<std::size_t>::max() ||
        optionCount64 >
            std::numeric_limits<std::size_t>::max() /
                (kNumericFields * sizeof(double))) {
        if (rank == 0) {
            std::fprintf(stderr, "Requested problem size is too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const std::size_t numOptions = static_cast<std::size_t>(optionCount64);
    const Range localRange = rankRange(numOptions, rank, rankCount);

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localRankCount = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localRankCount));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices are visible\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Create the CUDA context before timing.

    // Unless explicitly configured by the user, divide the CPUs visible to
    // this node evenly among its GPU-driving MPI ranks.
    omp_set_dynamic(0);
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(
            std::max(1, omp_get_num_procs() / localRankCount));
    }
    const int openMpThreads = omp_get_max_threads();

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    const int localOversubscribed = localRankCount > deviceCount ? 1 : 0;
    int anyOversubscribed = 0;
    MPI_CHECK(MPI_Reduce(&localOversubscribed, &anyOversubscribed, 1, MPI_INT,
                         MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        std::printf("Black-Scholes Option Pricing Benchmark\n");
        std::printf("Number of options: %zu\n", numOptions);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Parallel configuration: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA device '%s'\n",
                    rankCount, openMpThreads, deviceProperties.name);
        if (anyOversubscribed) {
            std::printf("Warning: at least one node has more MPI ranks than "
                        "visible CUDA devices; GPUs will be shared.\n");
        }
    }

    int exitCode = EXIT_SUCCESS;
    {
        HostOptions localOptions(localRange.count);
        PinnedBuffer<double> localResults(localRange.count,
                                          cudaHostAllocPortable);
        generateOptions(localOptions, localRange.begin);
        GpuExecutor executor(localRange.count);

        if (rank == 0) {
            std::printf("Pricing options...\n");
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        executor.execute(localOptions, localResults.data());
        const double localDuration = MPI_Wtime() - start;

        double duration = 0.0;
        MPI_CHECK(MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX,
                             0, MPI_COMM_WORLD));
        if (rank == 0) {
            const double optionsPerSecond =
                duration > 0.0 ? static_cast<double>(numOptions) / duration
                               : 0.0;
            std::printf("Computation time: %.3f ms\n", duration * 1000.0);
            std::printf("Options per second: %.0f\n", optionsPerSecond);
        }

        std::vector<double> globalResults;
        if (printResults) {
            gatherResults(localResults.data(), localRange, numOptions, rank,
                          rankCount, globalResults);
        } else if (validate) {
            // Validation consumes only the same ten leading values as the
            // original benchmark, avoiding a non-scalable all-results gather.
            gatherValidationPrefix(localResults.data(), localRange, numOptions,
                                   rank, globalResults);
        }

        if (rank == 0) {
            if (printResults) {
                print_results(globalResults, "OptionPrices");
            }
            if (validate) {
                std::printf("Validating results...\n");
                if (validateResults(numOptions, globalResults)) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = EXIT_FAILURE;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    MPI_CHECK(MPI_Comm_free(&localCommunicator));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
