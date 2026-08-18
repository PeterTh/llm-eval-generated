#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_THREADS_PER_BLOCK = 256;
constexpr int CUDA_WARPS_PER_BLOCK = CUDA_THREADS_PER_BLOCK / 32;

static_assert(CUDA_THREADS_PER_BLOCK % 32 == 0, "A block must contain whole warps");

[[noreturn]] void abortRun(const int rank, const char* message) {
    std::fprintf(stderr, "[rank %d] %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortCuda(const int rank, const char* operation, const cudaError_t status) {
    std::fprintf(stderr, "[rank %d] CUDA error during %s: %s\n", rank, operation,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const int rank, const char* operation) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING] = {};
    int errorLength = 0;
    MPI_Error_string(status, error, &errorLength);
    std::fprintf(stderr, "[rank %d] MPI error during %s: %.*s\n", rank, operation, errorLength,
                 error);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

// CUDA setup and execution are performed by OpenMP workers.  Reporting the first
// asynchronous host-side failure after the parallel region keeps MPI calls on the
// thread that requested MPI_THREAD_FUNNELED.
struct CudaFailure {
    std::atomic<bool> failed{false};

    bool check(const cudaError_t status, const int rank, const char* operation) {
        if (status == cudaSuccess) {
            return true;
        }

        bool expected = false;
        if (failed.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "[rank %d] CUDA error during %s: %s\n", rank, operation,
                         cudaGetErrorString(status));
        }
        return false;
    }
};

void fill(double* values, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        values[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Keep the original benchmark's deterministic CSR layout exactly.
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned++] = j;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// This is deliberately retained only for validation.  The benchmark computation
// itself always runs on CUDA devices.
void spmvCpuReference(const double* val, const index_t* cols, const index_t* rowDelimiters,
                      const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(dim); ++row) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row]; element < rowDelimiters[row + 1]; ++element) {
            sum += val[element] * vec[cols[element]];
        }
        out[row] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    int64_t firstMismatch = std::numeric_limits<int64_t>::max();
#pragma omp parallel for schedule(static) reduction(min : firstMismatch)
    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const bool valid = std::abs(ref) < 1e-10 ? std::abs(res) <= MAX_RELATIVE_ERROR
                                                  : std::abs((res - ref) / ref) <= MAX_RELATIVE_ERROR;
        if (!valid) {
            firstMismatch = std::min(firstMismatch, i);
        }
    }
    if (firstMismatch == std::numeric_limits<int64_t>::max()) {
        return true;
    }

    const index_t i = static_cast<index_t>(firstMismatch);
    const double ref = reference[i];
    const double res = result[i];
    if (std::abs(ref) < 1e-10) {
        std::printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
    } else {
        const double relativeError = std::abs((res - ref) / ref);
        std::printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n", i, ref,
                    res, relativeError);
    }
    return false;
}

__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t rows,
                           double* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const index_t row = static_cast<index_t>(blockIdx.x * CUDA_WARPS_PER_BLOCK + threadIdx.x / 32);
    if (row >= rows) {
        return;
    }

    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t element = begin + lane; element < end; element += 32) {
        sum += val[element] * vec[cols[element]];
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

struct GpuWorker {
    int device = -1;
    index_t rowBegin = 0;
    index_t rows = 0;
    size_t valueOffset = 0;
    size_t nnz = 0;
    std::vector<index_t> hostRowDelimiters;

    double* dVal = nullptr;
    index_t* dCols = nullptr;
    index_t* dRowDelimiters = nullptr;
    double* dVec = nullptr;
    double* dOut = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    float elapsedMs = 0.0F;
};

void releaseGpuWorker(GpuWorker& worker) {
    cudaSetDevice(worker.device);
    if (worker.graphExec != nullptr) cudaGraphExecDestroy(worker.graphExec);
    if (worker.graph != nullptr) cudaGraphDestroy(worker.graph);
    if (worker.startEvent != nullptr) cudaEventDestroy(worker.startEvent);
    if (worker.stopEvent != nullptr) cudaEventDestroy(worker.stopEvent);
    if (worker.dVal != nullptr) cudaFree(worker.dVal);
    if (worker.dCols != nullptr) cudaFree(worker.dCols);
    if (worker.dRowDelimiters != nullptr) cudaFree(worker.dRowDelimiters);
    if (worker.dVec != nullptr) cudaFree(worker.dVec);
    if (worker.dOut != nullptr) cudaFree(worker.dOut);
    if (worker.stream != nullptr) cudaStreamDestroy(worker.stream);
}

int checkedMpiCount(const size_t count, const int rank, const char* description) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        char message[256] = {};
        std::snprintf(message, sizeof(message), "%s exceeds the MPI count limit", description);
        abortRun(rank, message);
    }
    return static_cast<int>(count);
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank, "MPI_Comm_rank");
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), rank, "MPI_Comm_size");
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        abortRun(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    int command = 1;  // 0: invalid arguments, 1: run, 2: help

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numRows = static_cast<index_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                sparsity = static_cast<index_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = static_cast<index_t>(std::atoi(argv[++i]));
            } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
                maxVal = std::atof(argv[++i]);
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                command = 2;
                break;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                command = 0;
                break;
            }
        }
        if (command == 1 && (numRows == 0 || sparsity == 0)) {
            std::printf("Matrix dimension and sparsity must both be greater than zero\n");
            command = 0;
        }
    }

    checkMpi(MPI_Bcast(&command, 1, MPI_INT, 0, MPI_COMM_WORLD), rank, "broadcasting command");
    if (command != 1) {
        MPI_Finalize();
        return command == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    checkMpi(MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD), rank, "broadcasting rows");
    checkMpi(MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD), rank,
             "broadcasting sparsity");
    checkMpi(MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD), rank,
             "broadcasting iterations");
    checkMpi(MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank,
             "broadcasting maximum value");
    checkMpi(MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD), rank,
             "broadcasting validation flag");
    checkMpi(MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD), rank,
             "broadcasting results flag");
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    // This retains the original unsigned arithmetic and therefore its input semantics.
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
                    100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("OpenMP host threads available per rank: %d\n", omp_get_max_threads());
    }

    // Root generates the benchmark's deterministic input once; CSR row ownership
    // and all nonzeros belonging to those rows are then distributed to MPI ranks.
    std::vector<double> hVec(numRows);
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> hReference;
    std::vector<double> hOut;

    if (rank == 0) {
        std::printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        hOut.resize(numRows);
        fill(hVec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);

        if (validate) {
            std::printf("Computing reference solution...\n");
            hReference.resize(numRows);
            spmvCpuReference(globalVal.data(), globalCols.data(), globalRowDelimiters.data(),
                             hVec.data(), numRows, hReference.data());
        }
    }

    const uint64_t rowBegin64 = static_cast<uint64_t>(numRows) * rank / worldSize;
    const uint64_t rowEnd64 = static_cast<uint64_t>(numRows) * (rank + 1) / worldSize;
    const index_t localRows = static_cast<index_t>(rowEnd64 - rowBegin64);

    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    std::vector<int> rowStartCounts;
    std::vector<int> rowStartDisplacements;
    std::vector<int> outputCounts;
    std::vector<int> outputDisplacements;
    std::vector<index_t> rowEndDelimiters;
    if (rank == 0) {
        nnzCounts.resize(worldSize);
        nnzDisplacements.resize(worldSize);
        rowStartCounts.resize(worldSize);
        rowStartDisplacements.resize(worldSize);
        outputCounts.resize(worldSize);
        outputDisplacements.resize(worldSize);
        rowEndDelimiters.resize(worldSize);
        for (int process = 0; process < worldSize; ++process) {
            const index_t first = static_cast<index_t>(static_cast<uint64_t>(numRows) * process / worldSize);
            const index_t last = static_cast<index_t>(static_cast<uint64_t>(numRows) * (process + 1) / worldSize);
            const size_t processNnz = static_cast<size_t>(globalRowDelimiters[last]) -
                                      static_cast<size_t>(globalRowDelimiters[first]);
            nnzCounts[process] = checkedMpiCount(processNnz, rank, "CSR nonzero partition");
            nnzDisplacements[process] = checkedMpiCount(globalRowDelimiters[first], rank,
                                                         "CSR nonzero displacement");
            rowStartCounts[process] = checkedMpiCount(static_cast<size_t>(last - first), rank,
                                                       "CSR row-start partition");
            rowStartDisplacements[process] = checkedMpiCount(first, rank,
                                                               "CSR row-start displacement");
            rowEndDelimiters[process] = globalRowDelimiters[last];
            outputCounts[process] = checkedMpiCount(static_cast<size_t>(last - first), rank,
                                                     "output partition");
            outputDisplacements[process] = checkedMpiCount(first, rank, "output displacement");
        }
    }

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> localVal;
    std::vector<index_t> localCols;

    checkMpi(MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                          rank == 0 ? rowStartCounts.data() : nullptr,
                          rank == 0 ? rowStartDisplacements.data() : nullptr, MPI_UINT32_T,
                          localRowDelimiters.data(), checkedMpiCount(localRows, rank, "local row starts"),
                          MPI_UINT32_T, 0, MPI_COMM_WORLD),
             rank, "scattering CSR row starts");
    checkMpi(MPI_Scatter(rank == 0 ? rowEndDelimiters.data() : nullptr, 1, MPI_UINT32_T,
                         localRowDelimiters.data() + localRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD),
             rank, "scattering CSR row ends");

    const size_t rankNnzBegin = localRowDelimiters.front();
    const size_t localNnz = static_cast<size_t>(localRowDelimiters.back()) - rankNnzBegin;
    localVal.resize(localNnz);
    localCols.resize(localNnz);

    checkMpi(MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr,
                          rank == 0 ? nnzCounts.data() : nullptr,
                          rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                          localVal.data(), checkedMpiCount(localNnz, rank, "local values"), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD),
             rank, "scattering CSR values");
    checkMpi(MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                          rank == 0 ? nnzCounts.data() : nullptr,
                          rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                          localCols.data(), checkedMpiCount(localNnz, rank, "local columns"), MPI_UINT32_T,
                          0, MPI_COMM_WORLD),
             rank, "scattering CSR columns");
    checkMpi(MPI_Bcast(hVec.data(), checkedMpiCount(hVec.size(), rank, "dense vector"), MPI_DOUBLE, 0,
                       MPI_COMM_WORLD),
             rank, "broadcasting dense vector");

    MPI_Comm localComm = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm), rank,
             "creating node-local communicator");
    int localRank = 0;
    int localSize = 1;
    checkMpi(MPI_Comm_rank(localComm, &localRank), rank, "getting local rank");
    checkMpi(MPI_Comm_size(localComm, &localSize), rank, "getting local size");

    int visibleDeviceCount = 0;
    const cudaError_t deviceCountStatus = cudaGetDeviceCount(&visibleDeviceCount);
    if (deviceCountStatus != cudaSuccess) {
        abortCuda(rank, "discovering CUDA devices", deviceCountStatus);
    }
    if (visibleDeviceCount == 0) {
        abortRun(rank, "no CUDA device is visible to this MPI rank");
    }

    // When there are fewer ranks than devices on a node, each rank owns a disjoint
    // range and OpenMP launches one independent CUDA stream per assigned device.
    // With one rank per GPU this naturally reduces to one GPU worker per rank.
    std::vector<int> assignedDevices;
    if (localSize <= visibleDeviceCount) {
        const int deviceBegin = static_cast<int>(static_cast<int64_t>(localRank) * visibleDeviceCount / localSize);
        const int deviceEnd = static_cast<int>(static_cast<int64_t>(localRank + 1) * visibleDeviceCount / localSize);
        for (int device = deviceBegin; device < deviceEnd; ++device) {
            assignedDevices.push_back(device);
        }
    } else {
        assignedDevices.push_back(localRank % visibleDeviceCount);
    }
    checkMpi(MPI_Comm_free(&localComm), rank, "releasing node-local communicator");

    const int rankGpuWorkers = static_cast<int>(assignedDevices.size());
    int totalGpuWorkers = 0;
    checkMpi(MPI_Reduce(&rankGpuWorkers, &totalGpuWorkers, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD), rank,
             "counting GPU workers");
    if (rank == 0) {
        std::printf("CUDA workers: %d (disjoint GPUs per node when rank placement permits)\n", totalGpuWorkers);
    }

    std::vector<GpuWorker> workers;
    workers.reserve(assignedDevices.size());
    std::vector<index_t> workerBoundaries(assignedDevices.size() + 1);
    workerBoundaries.front() = 0;
    workerBoundaries.back() = localRows;
    for (size_t worker = 1; worker + 1 < workerBoundaries.size(); ++worker) {
        const size_t targetNnz = rankNnzBegin + localNnz * worker / assignedDevices.size();
        const auto boundary = std::lower_bound(localRowDelimiters.begin(), localRowDelimiters.end(),
                                               static_cast<index_t>(targetNnz));
        workerBoundaries[worker] = static_cast<index_t>(
            std::clamp<std::ptrdiff_t>(boundary - localRowDelimiters.begin(), workerBoundaries[worker - 1],
                                       localRows));
    }
    for (size_t worker = 0; worker < assignedDevices.size(); ++worker) {
        GpuWorker gpuWorker;
        gpuWorker.device = assignedDevices[worker];
        gpuWorker.rowBegin = workerBoundaries[worker];
        gpuWorker.rows = workerBoundaries[worker + 1] - workerBoundaries[worker];
        const size_t globalValueBegin = localRowDelimiters[gpuWorker.rowBegin];
        const size_t globalValueEnd = localRowDelimiters[gpuWorker.rowBegin + gpuWorker.rows];
        gpuWorker.valueOffset = globalValueBegin - rankNnzBegin;
        gpuWorker.nnz = globalValueEnd - globalValueBegin;
        workers.push_back(std::move(gpuWorker));
    }

    // This first-touch/preparation pass and all CUDA control work below are OpenMP
    // parallel regions.  Device-resident SpMV owns the timed numerical work.
#pragma omp parallel for schedule(static)
    for (int64_t workerIndex = 0; workerIndex < static_cast<int64_t>(workers.size()); ++workerIndex) {
        GpuWorker& worker = workers[workerIndex];
        worker.hostRowDelimiters.resize(static_cast<size_t>(worker.rows) + 1);
        const index_t base = localRowDelimiters[worker.rowBegin];
        for (size_t row = 0; row <= worker.rows; ++row) {
            worker.hostRowDelimiters[row] = localRowDelimiters[worker.rowBegin + row] - base;
        }
    }

    std::vector<double> localOut(localRows, 0.0);
    omp_set_dynamic(0);
    CudaFailure setupFailure;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(workers.size()))
    for (int64_t workerIndex = 0; workerIndex < static_cast<int64_t>(workers.size()); ++workerIndex) {
        GpuWorker& worker = workers[workerIndex];
        if (!setupFailure.check(cudaSetDevice(worker.device), rank, "selecting CUDA device")) continue;
        if (!setupFailure.check(cudaStreamCreateWithFlags(&worker.stream, cudaStreamNonBlocking), rank,
                                "creating CUDA stream"))
            continue;
        if (!setupFailure.check(cudaMalloc(reinterpret_cast<void**>(&worker.dVal),
                                           std::max<size_t>(worker.nnz, 1) * sizeof(double)),
                                rank, "allocating CUDA values"))
            continue;
        if (!setupFailure.check(cudaMalloc(reinterpret_cast<void**>(&worker.dCols),
                                           std::max<size_t>(worker.nnz, 1) * sizeof(index_t)),
                                rank, "allocating CUDA columns"))
            continue;
        if (!setupFailure.check(cudaMalloc(reinterpret_cast<void**>(&worker.dRowDelimiters),
                                           worker.hostRowDelimiters.size() * sizeof(index_t)),
                                rank, "allocating CUDA row delimiters"))
            continue;
        if (!setupFailure.check(cudaMalloc(reinterpret_cast<void**>(&worker.dVec),
                                           std::max<size_t>(hVec.size(), 1) * sizeof(double)),
                                rank, "allocating CUDA vector"))
            continue;
        if (!setupFailure.check(cudaMalloc(reinterpret_cast<void**>(&worker.dOut),
                                           std::max<size_t>(worker.rows, 1) * sizeof(double)),
                                rank, "allocating CUDA output"))
            continue;
        if (!setupFailure.check(cudaEventCreate(&worker.startEvent), rank, "creating CUDA start event")) continue;
        if (!setupFailure.check(cudaEventCreate(&worker.stopEvent), rank, "creating CUDA stop event")) continue;

        if (worker.nnz != 0) {
            if (!setupFailure.check(cudaMemcpyAsync(worker.dVal, localVal.data() + worker.valueOffset,
                                                    worker.nnz * sizeof(double), cudaMemcpyHostToDevice,
                                                    worker.stream),
                                    rank, "copying values to CUDA device"))
                continue;
            if (!setupFailure.check(cudaMemcpyAsync(worker.dCols, localCols.data() + worker.valueOffset,
                                                    worker.nnz * sizeof(index_t), cudaMemcpyHostToDevice,
                                                    worker.stream),
                                    rank, "copying columns to CUDA device"))
                continue;
        }
        if (!setupFailure.check(cudaMemcpyAsync(worker.dRowDelimiters, worker.hostRowDelimiters.data(),
                                                worker.hostRowDelimiters.size() * sizeof(index_t),
                                                cudaMemcpyHostToDevice, worker.stream),
                                rank, "copying row delimiters to CUDA device"))
            continue;
        if (!setupFailure.check(cudaMemcpyAsync(worker.dVec, hVec.data(), hVec.size() * sizeof(double),
                                                cudaMemcpyHostToDevice, worker.stream),
                                rank, "copying vector to CUDA device"))
            continue;
        if (!setupFailure.check(cudaStreamSynchronize(worker.stream), rank, "synchronizing CUDA input copies"))
            continue;

        if (iterations != 0 && worker.rows != 0) {
            if (!setupFailure.check(cudaStreamBeginCapture(worker.stream, cudaStreamCaptureModeThreadLocal), rank,
                                    "beginning CUDA graph capture"))
                continue;
            const dim3 block(CUDA_THREADS_PER_BLOCK);
            const dim3 grid((worker.rows + CUDA_WARPS_PER_BLOCK - 1) / CUDA_WARPS_PER_BLOCK);
            for (index_t iteration = 0; iteration < iterations; ++iteration) {
                spmvKernel<<<grid, block, 0, worker.stream>>>(worker.dVal, worker.dCols,
                                                               worker.dRowDelimiters, worker.dVec, worker.rows,
                                                               worker.dOut);
            }
            if (!setupFailure.check(cudaStreamEndCapture(worker.stream, &worker.graph), rank,
                                    "ending CUDA graph capture"))
                continue;
            if (!setupFailure.check(cudaGetLastError(), rank, "capturing CUDA SpMV kernel")) continue;
            if (!setupFailure.check(cudaGraphInstantiate(&worker.graphExec, worker.graph, nullptr, nullptr, 0), rank,
                                    "instantiating CUDA graph"))
                continue;
        }
    }
    if (setupFailure.failed.load()) {
        abortRun(rank, "CUDA setup failed");
    }

    if (rank == 0) {
        std::printf("Computing SpMV...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), rank, "synchronizing benchmark start");

    CudaFailure executionFailure;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(workers.size()))
    for (int64_t workerIndex = 0; workerIndex < static_cast<int64_t>(workers.size()); ++workerIndex) {
        GpuWorker& worker = workers[workerIndex];
        if (iterations == 0 || worker.rows == 0) continue;
        if (!executionFailure.check(cudaSetDevice(worker.device), rank, "selecting CUDA device for timing")) continue;
        if (!executionFailure.check(cudaEventRecord(worker.startEvent, worker.stream), rank,
                                    "recording CUDA start event"))
            continue;
        if (!executionFailure.check(cudaGraphLaunch(worker.graphExec, worker.stream), rank,
                                    "launching CUDA SpMV graph"))
            continue;
        if (!executionFailure.check(cudaEventRecord(worker.stopEvent, worker.stream), rank,
                                    "recording CUDA stop event"))
            continue;
        if (!executionFailure.check(cudaEventSynchronize(worker.stopEvent), rank,
                                    "synchronizing CUDA stop event"))
            continue;
        executionFailure.check(cudaEventElapsedTime(&worker.elapsedMs, worker.startEvent, worker.stopEvent), rank,
                               "reading CUDA elapsed time");
    }
    if (executionFailure.failed.load()) {
        abortRun(rank, "CUDA execution failed");
    }

    double localElapsedMs = 0.0;
    for (const GpuWorker& worker : workers) {
        localElapsedMs = std::max(localElapsedMs, static_cast<double>(worker.elapsedMs));
    }
    double elapsedMs = 0.0;
    checkMpi(MPI_Reduce(&localElapsedMs, &elapsedMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), rank,
             "reducing benchmark time");

    CudaFailure outputFailure;
#pragma omp parallel for schedule(static) num_threads(static_cast<int>(workers.size()))
    for (int64_t workerIndex = 0; workerIndex < static_cast<int64_t>(workers.size()); ++workerIndex) {
        GpuWorker& worker = workers[workerIndex];
        if (iterations == 0 || worker.rows == 0) continue;
        if (!outputFailure.check(cudaSetDevice(worker.device), rank, "selecting CUDA device for output")) continue;
        if (!outputFailure.check(cudaMemcpyAsync(localOut.data() + worker.rowBegin, worker.dOut,
                                                 static_cast<size_t>(worker.rows) * sizeof(double),
                                                 cudaMemcpyDeviceToHost, worker.stream),
                                 rank, "copying CUDA output"))
            continue;
        outputFailure.check(cudaStreamSynchronize(worker.stream), rank, "synchronizing CUDA output copy");
    }
    if (outputFailure.failed.load()) {
        abortRun(rank, "CUDA output transfer failed");
    }

    checkMpi(MPI_Gatherv(localOut.data(), checkedMpiCount(localOut.size(), rank, "local output"), MPI_DOUBLE,
                         rank == 0 ? hOut.data() : nullptr, rank == 0 ? outputCounts.data() : nullptr,
                         rank == 0 ? outputDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD),
             rank, "gathering SpMV output");

#pragma omp parallel for schedule(static) num_threads(static_cast<int>(workers.size()))
    for (int64_t workerIndex = 0; workerIndex < static_cast<int64_t>(workers.size()); ++workerIndex) {
        releaseGpuWorker(workers[workerIndex]);
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", elapsedMs);
        const double averageTime = iterations == 0 ? 0.0 : elapsedMs / static_cast<double>(iterations);
        const double gflops = elapsedMs == 0.0 ? 0.0 :
            (2.0 * static_cast<double>(nItems) * iterations) / (elapsedMs / 1000.0) / 1e9;
        std::printf("Average time per iteration: %.3f ms\n", averageTime);
        std::printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(hOut, "OutputVector");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (verifyResults(hReference.data(), hOut.data(), numRows)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    checkMpi(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD), rank, "broadcasting exit status");
    MPI_Finalize();
    return exitCode;
}
