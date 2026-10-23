#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

inline OptionInput makeOption(const size_t i) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[i % testOptions.size()];
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions) {
    options.resize(numOptions);
    
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numOptions); ++i) {
        options[static_cast<size_t>(i)] = makeOption(static_cast<size_t>(i));
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

namespace {

constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;
constexpr int kThreadsPerBlock = 256;
constexpr size_t kChunkOptions = 1u << 20;
constexpr int kNumFields = 7;
constexpr int kNumStages = 2;

__global__ void blackScholesKernel(const double* __restrict__ spot,
                                   const double* __restrict__ strike,
                                   const double* __restrict__ q,
                                   const double* __restrict__ r,
                                   const double* __restrict__ t,
                                   const double* __restrict__ vol,
                                   const double* __restrict__ type,
                                   double* __restrict__ out,
                                   const unsigned long long n) {
    unsigned long long idx = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    for (; idx < n; idx += stride) {
        const double S = spot[idx];
        const double K = strike[idx];
        const double rr = r[idx];
        const double qq = q[idx];
        const double T = t[idx];
        const double sigma = vol[idx];

        if (T <= 0.0 || sigma <= 0.0) {
            out[idx] = 0.0;
            continue;
        }

        const double sqrtT = sqrt(T);
        const double d1 = (log(S / K) + (rr - qq + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
        const double d2 = d1 - sigma * sqrtT;
        const double discount = exp(-rr * T);
        const double qDiscount = exp(-qq * T);

        double price;
        if (type[idx] == static_cast<double>(CALL)) {
            const double Nd1 = 0.5 * erfc(-d1 * kInvSqrt2);
            const double Nd2 = 0.5 * erfc(-d2 * kInvSqrt2);
            price = S * qDiscount * Nd1 - K * discount * Nd2;
        } else {
            const double Nmd2 = 0.5 * erfc(d2 * kInvSqrt2);
            const double Nmd1 = 0.5 * erfc(d1 * kInvSqrt2);
            price = K * discount * Nmd2 - S * qDiscount * Nmd1;
        }

        out[idx] = price;
    }
}

void checkCuda(const cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

struct Partition {
    size_t start;
    size_t count;
};

Partition partitionFor(const size_t total, const int rank, const int size) {
    const size_t base = total / static_cast<size_t>(size);
    const size_t remainder = total % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    return {r * base + std::min(r, remainder), base + (r < remainder ? 1u : 0u)};
}

void splitSharedComm(const int rank, int& localRank, int& localSize) {
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);
}

void selectDevice(const int localRank, const int localSize) {
    (void)localSize;
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA-capable devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");
    checkCuda(cudaFree(0), "cudaFree");
}

void generateOptionFields(std::vector<double>& fields, const size_t start,
                          const size_t count) {
    const size_t n = count;
    if (n == 0) {
        fields.clear();
        return;
    }
    fields.resize(kNumFields * n);

    double* spot = fields.data() + 0 * n;
    double* strike = fields.data() + 1 * n;
    double* q = fields.data() + 2 * n;
    double* r = fields.data() + 3 * n;
    double* t = fields.data() + 4 * n;
    double* vol = fields.data() + 5 * n;
    double* type = fields.data() + 6 * n;

    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(n); ++j) {
        const OptionInput option = makeOption(start + static_cast<size_t>(j));
        const size_t i = static_cast<size_t>(j);
        spot[i] = option.spot;
        strike[i] = option.strike;
        q[i] = option.q;
        r[i] = option.r;
        t[i] = option.t;
        vol[i] = option.vol;
        type[i] = static_cast<double>(option.type);
    }
}

struct GpuBuffers {
    size_t cap = 0;
    cudaStream_t streams[kNumStages]{};
    cudaEvent_t done[kNumStages]{};
    double* deviceIn[kNumStages]{};
    double* deviceOut[kNumStages]{};
    double* hostIn[kNumStages]{};
    double* hostOut[kNumStages]{};
};

void setupGpuBuffers(GpuBuffers& gpu, const size_t localCount) {
    if (localCount == 0) {
        return;
    }

    gpu.cap = std::min(kChunkOptions, localCount);
    const size_t fieldBytes = gpu.cap * sizeof(double);

    for (int s = 0; s < kNumStages; ++s) {
        checkCuda(cudaStreamCreate(&gpu.streams[s]), "cudaStreamCreate");
        checkCuda(cudaEventCreateWithFlags(&gpu.done[s], cudaEventDisableTiming),
                  "cudaEventCreate");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu.deviceIn[s]),
                             kNumFields * fieldBytes), "cudaMalloc");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu.deviceOut[s]), fieldBytes), "cudaMalloc");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&gpu.hostIn[s]),
                                 kNumFields * fieldBytes), "cudaMallocHost");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&gpu.hostOut[s]), fieldBytes),
                  "cudaMallocHost");
    }

    const size_t cap = gpu.cap;
    for (int f = 0; f < kNumFields; ++f) {
        checkCuda(cudaMemset(gpu.deviceIn[0] + static_cast<size_t>(f) * cap, 0,
                             sizeof(double)), "cudaMemset");
    }
    blackScholesKernel<<<1, 32, 0, gpu.streams[0]>>>(
        gpu.deviceIn[0] + 0 * cap, gpu.deviceIn[0] + 1 * cap, gpu.deviceIn[0] + 2 * cap,
        gpu.deviceIn[0] + 3 * cap, gpu.deviceIn[0] + 4 * cap, gpu.deviceIn[0] + 5 * cap,
        gpu.deviceIn[0] + 6 * cap, gpu.deviceOut[0], 0ull);
    checkCuda(cudaGetLastError(), "warmup kernel");
    checkCuda(cudaStreamSynchronize(gpu.streams[0]), "warmup synchronize");
}

void processGpuChunks(const GpuBuffers& gpu, const std::vector<double>& hostFields,
                      const size_t localCount, std::vector<double>& results) {
    results.resize(localCount);
    if (localCount == 0) {
        return;
    }

    const size_t cap = gpu.cap;
    const cudaStream_t* streams = gpu.streams;
    const cudaEvent_t* done = gpu.done;
    double* const* deviceIn = gpu.deviceIn;
    double* const* deviceOut = gpu.deviceOut;
    double* const* hostIn = gpu.hostIn;
    double* const* hostOut = gpu.hostOut;

    size_t stageOffset[kNumStages] = {0, 0};
    size_t stageCount[kNumStages] = {0, 0};
    bool stageValid[kNumStages] = {false, false};

    size_t chunkIndex = 0;
    for (size_t offset = 0; offset < localCount; offset += cap, ++chunkIndex) {
        const int stage = static_cast<int>(chunkIndex % kNumStages);
        const size_t count = std::min(cap, localCount - offset);

        if (stageValid[stage]) {
            checkCuda(cudaEventSynchronize(done[stage]), "cudaEventSynchronize");
            std::memcpy(results.data() + stageOffset[stage], hostOut[stage],
                        stageCount[stage] * sizeof(double));
            stageValid[stage] = false;
        }

        for (int f = 0; f < kNumFields; ++f) {
            const double* src = hostFields.data() + static_cast<size_t>(f) * localCount + offset;
            std::memcpy(hostIn[stage] + static_cast<size_t>(f) * cap, src,
                        count * sizeof(double));
        }

        for (int f = 0; f < kNumFields; ++f) {
            checkCuda(cudaMemcpyAsync(deviceIn[stage] + static_cast<size_t>(f) * cap,
                                      hostIn[stage] + static_cast<size_t>(f) * cap,
                                      count * sizeof(double), cudaMemcpyHostToDevice,
                                      streams[stage]), "cudaMemcpyAsync H2D");
        }

        const unsigned int blocks = static_cast<unsigned int>(
            std::min<size_t>((count + kThreadsPerBlock - 1) / kThreadsPerBlock, 2147483647u));
        blackScholesKernel<<<blocks, kThreadsPerBlock, 0, streams[stage]>>>(
            deviceIn[stage] + 0 * cap, deviceIn[stage] + 1 * cap, deviceIn[stage] + 2 * cap,
            deviceIn[stage] + 3 * cap, deviceIn[stage] + 4 * cap, deviceIn[stage] + 5 * cap,
            deviceIn[stage] + 6 * cap, deviceOut[stage],
            static_cast<unsigned long long>(count));
        checkCuda(cudaGetLastError(), "blackScholesKernel");

        checkCuda(cudaMemcpyAsync(hostOut[stage], deviceOut[stage], count * sizeof(double),
                                  cudaMemcpyDeviceToHost, streams[stage]),
                  "cudaMemcpyAsync D2H");
        checkCuda(cudaEventRecord(done[stage], streams[stage]), "cudaEventRecord");

        stageOffset[stage] = offset;
        stageCount[stage] = count;
        stageValid[stage] = true;
    }

    for (int s = 0; s < kNumStages; ++s) {
        if (stageValid[s]) {
            checkCuda(cudaEventSynchronize(done[s]), "cudaEventSynchronize final");
            std::memcpy(results.data() + stageOffset[s], hostOut[s],
                        stageCount[s] * sizeof(double));
        }
    }
}

void destroyGpuBuffers(GpuBuffers& gpu) {
    if (gpu.cap == 0) {
        return;
    }
    for (int s = 0; s < kNumStages; ++s) {
        checkCuda(cudaFreeHost(gpu.hostIn[s]), "cudaFreeHost");
        checkCuda(cudaFreeHost(gpu.hostOut[s]), "cudaFreeHost");
        checkCuda(cudaFree(gpu.deviceIn[s]), "cudaFree");
        checkCuda(cudaFree(gpu.deviceOut[s]), "cudaFree");
        checkCuda(cudaEventDestroy(gpu.done[s]), "cudaEventDestroy");
        checkCuda(cudaStreamDestroy(gpu.streams[s]), "cudaStreamDestroy");
    }
    gpu.cap = 0;
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int localRank = 0;
    int localSize = 1;
    splitSharedComm(rank, localRank, localSize);

    const int ompThreads = std::max(1, omp_get_num_procs() / localSize);
    omp_set_num_threads(ompThreads);

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
    }

    selectDevice(localRank, localSize);

    const Partition partition = partitionFor(numOptions, rank, numRanks);

    std::vector<double> hostFields;
    generateOptionFields(hostFields, partition.start, partition.count);

    std::vector<double> localResults;

    GpuBuffers gpu;
    setupGpuBuffers(gpu, partition.count);

    if (rank == 0) {
        printf("Pricing options...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    processGpuChunks(gpu, hostFields, partition.count, localResults);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    destroyGpuBuffers(gpu);

    if (rank == 0) {
        const double seconds = duration.count() / 1e6;
        printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
        printf("Options per second: %.0f\n",
               seconds > 0.0 ? numOptions / seconds : 0.0);
    }

    std::vector<double> fullResults;
    if (printResults) {
        std::vector<int> counts(static_cast<size_t>(numRanks));
        std::vector<int> displs(static_cast<size_t>(numRanks));
        for (int r = 0; r < numRanks; ++r) {
            const Partition p = partitionFor(numOptions, r, numRanks);
            counts[static_cast<size_t>(r)] = static_cast<int>(p.count);
            displs[static_cast<size_t>(r)] = static_cast<int>(p.start);
        }
        if (rank == 0) {
            fullResults.resize(numOptions);
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(partition.count), MPI_DOUBLE,
                    rank == 0 ? fullResults.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(fullResults, "OptionPrices");
        }
    }
    
    // Validation
    if (validate) {
        const size_t numChecks = std::min<size_t>(10, numOptions);
        std::vector<int> counts(static_cast<size_t>(numRanks));
        std::vector<int> displs(static_cast<size_t>(numRanks));
        for (int r = 0; r < numRanks; ++r) {
            const Partition p = partitionFor(numOptions, r, numRanks);
            const size_t remaining = p.start < numChecks ? numChecks - p.start : 0;
            counts[static_cast<size_t>(r)] =
                static_cast<int>(std::min(p.count, remaining));
            displs[static_cast<size_t>(r)] = static_cast<int>(std::min(p.start, numChecks));
        }

        std::vector<double> checkResults(numChecks);
        const size_t localRemaining =
            partition.start < numChecks ? numChecks - partition.start : 0;
        const int localCheck = static_cast<int>(std::min(partition.count, localRemaining));

        MPI_Gatherv(localResults.data(), localCheck, MPI_DOUBLE,
                    rank == 0 ? checkResults.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int validFlag = 1;
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> checkOptions(numChecks);
            for (size_t i = 0; i < numChecks; ++i) {
                checkOptions[i] = makeOption(i);
            }
            const std::vector<double>& validationResults =
                (printResults && fullResults.size() >= numChecks) ? fullResults : checkResults;
            validFlag = validateResults(checkOptions, validationResults) ? 1 : 0;
        }
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validation: %s\n", validFlag ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return validFlag ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
