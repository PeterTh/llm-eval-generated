#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_SIZE = 256;

// All CUDA failures are fatal: this benchmark intentionally has no CPU fallback.
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t error_ = (call);                                                 \
        if (error_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(error_));                                      \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                       \
        }                                                                                 \
    } while (false)

void fill(double* values, const index_t count, const double maxValue) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Keep the original deterministic CSR construction so the generated problem and
// externally printed results retain the baseline benchmark's semantics.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t nnz,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const double probability = static_cast<double>(nnz) /
                               (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);
    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = nnzAssigned;
        for (index_t col = 0; col < dim; ++col) {
            const uint64_t position = static_cast<uint64_t>(row) * dim + col;
            const uint64_t entriesLeft = static_cast<uint64_t>(dim) * dim - position;
            const index_t needToAssign = nnz - nnzAssigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < nnz && randomValue <= probability) || fillRemaining) {
                cols[nnzAssigned++] = col;
            }
        }
    }
    rowDelimiters[dim] = nnz;
}

void spmvCpu(const double* values, const index_t* cols, const index_t* rowDelimiters,
             const double* vector, const index_t rows, double* output) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(rows); ++row) {
        double sum = 0.0;
        for (index_t item = rowDelimiters[row]; item < rowDelimiters[row + 1]; ++item) {
            sum += values[item] * vector[cols[item]];
        }
        output[row] = sum;
    }
}

// One thread per row is more efficient for very short rows because it avoids
// spending an entire warp on only a handful of nonzeros.
__global__ void spmvScalarKernel(const double* __restrict__ values,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vector,
                                 const index_t rows, double* __restrict__ output) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (index_t item = rowDelimiters[row]; item < rowDelimiters[row + 1]; ++item) {
        sum = fma(values[item], vector[cols[item]], sum);
    }
    output[row] = sum;
}

// A warp cooperatively processes one row. CSR values and column indices are read
// coalescently, while read-only vector entries benefit from the GPU cache.
__global__ void spmvWarpKernel(const double* __restrict__ values,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vector,
                               const index_t rows, double* __restrict__ output) {
    const index_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t row = globalThread >> 5;
    const unsigned lane = threadIdx.x & 31U;
    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    for (index_t item = rowDelimiters[row] + lane; item < rowDelimiters[row + 1];
         item += 32) {
        sum = fma(values[item], vector[cols[item]], sum);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, offset);
    }
    if (lane == 0) {
        output[row] = sum;
    }
}

void launchSpmv(cudaStream_t stream, const double* values, const index_t* cols,
                const index_t* rowDelimiters, const double* vector, const index_t rows,
                const index_t nnz, double* output) {
    if (rows == 0) {
        return;
    }

    const double averageNnz = static_cast<double>(nnz) / rows;
    if (averageNnz < 12.0) {
        const int blocks = static_cast<int>((rows + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        spmvScalarKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
            values, cols, rowDelimiters, vector, rows, output);
    } else {
        constexpr int warpsPerBlock = CUDA_BLOCK_SIZE / 32;
        const int blocks = static_cast<int>((rows + warpsPerBlock - 1) / warpsPerBlock);
        spmvWarpKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
            values, cols, rowDelimiters, vector, rows, output);
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t globalRowOffset, const int rank) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool invalid = !std::isfinite(ref) || !std::isfinite(res);
        if (!invalid && std::abs(ref) < 1e-10) {
            invalid = std::abs(res) > MAX_RELATIVE_ERROR;
        } else if (!invalid) {
            invalid = std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        }
        if (invalid) {
            std::fprintf(stderr,
                         "Rank %d validation failed at index %u: reference %.10e, "
                         "got %.10e\n",
                         rank, globalRowOffset + i, ref, res);
            return false;
        }
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
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
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required FUNNELED thread support.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxValue = 1.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxValue = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nnz64 = sparsity == 0 ? 0 : matrixEntries / sparsity;
    if (numRows == 0 || sparsity == 0 || iterations == 0 ||
        numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nnz64 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        worldSize > static_cast<int>(numRows)) {
        argumentsValid = false;
        if (rank == 0 && !showHelp) {
            std::fprintf(stderr,
                         "Arguments require positive dimensions/iterations, at least one row "
                         "per MPI rank, and MPI-compatible 32-bit element counts.\n");
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const index_t totalNnz = static_cast<index_t>(nnz64);

    // Assign one accelerator to each local MPI rank. Launching one rank per GPU is
    // preferred; modulo mapping remains correct when a launcher oversubscribes GPUs.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA accelerators are available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Create the CUDA context before timing.

    const index_t globalRowBegin = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * rank) / worldSize);
    const index_t globalRowEnd = static_cast<index_t>(
        (static_cast<uint64_t>(numRows) * (rank + 1)) / worldSize);
    const index_t localRows = globalRowEnd - globalRowBegin;

    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    std::vector<int> delimiterCounts(worldSize);
    std::vector<int> nnzCounts(worldSize);
    std::vector<int> nnzDisplacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const uint64_t begin = static_cast<uint64_t>(numRows) * process / worldSize;
        const uint64_t end = static_cast<uint64_t>(numRows) * (process + 1) / worldSize;
        rowCounts[process] = static_cast<int>(end - begin);
        rowDisplacements[process] = static_cast<int>(begin);
        // Send each global row delimiter exactly once. The receiving rank adds
        // its final delimiter from its already-known local nonzero count.
        delimiterCounts[process] = rowCounts[process];
    }

    std::vector<double> globalValues;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> hostVector(numRows);

    if (rank == 0) {
        globalValues.resize(totalNnz);
        globalCols.resize(totalNnz);
        globalRowDelimiters.resize(numRows + 1);

        // The C standard specifies the initial rand state as if srand(1) had run.
        // Make that baseline behavior explicit in the MPI program.
        srand(1);
        fill(hostVector.data(), numRows, maxValue);
        fill(globalValues.data(), totalNnz, maxValue);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), totalNnz, numRows);

        for (int process = 0; process < worldSize; ++process) {
            const int firstRow = rowDisplacements[process];
            const int lastRow = firstRow + rowCounts[process];
            nnzDisplacements[process] = static_cast<int>(globalRowDelimiters[firstRow]);
            nnzCounts[process] = static_cast<int>(globalRowDelimiters[lastRow] -
                                                  globalRowDelimiters[firstRow]);
        }
    }

    MPI_Bcast(nnzCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDisplacements.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(hostVector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);
    std::vector<double> hostValues(localNnz);
    std::vector<index_t> hostCols(localNnz);
    std::vector<index_t> hostRowDelimiters(localRows + 1);

    MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr, nnzCounts.data(),
                 nnzDisplacements.data(), MPI_DOUBLE, hostValues.data(), nnzCounts[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(),
                 nnzDisplacements.data(), MPI_UINT32_T, hostCols.data(), nnzCounts[rank],
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 delimiterCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                 hostRowDelimiters.data(), static_cast<int>(localRows), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    const index_t localBase = hostRowDelimiters.front();
    hostRowDelimiters[localRows] = localBase + localNnz;
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row <= static_cast<int64_t>(localRows); ++row) {
        hostRowDelimiters[row] -= localBase;
    }

    // Release root-only global CSR storage before allocating device storage.
    globalValues.clear();
    globalValues.shrink_to_fit();
    globalCols.clear();
    globalCols.shrink_to_fit();
    globalRowDelimiters.clear();
    globalRowDelimiters.shrink_to_fit();

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", totalNnz,
                    100.0 * (1.0 - static_cast<double>(totalNnz) / matrixEntries));
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxValue);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), up to %d OpenMP thread(s) "
                    "per rank, CUDA device %d on rank 0\n",
                    worldSize, omp_get_max_threads(), device);
        std::printf("Initializing data structures...\n");
    }

    double* deviceValues = nullptr;
    index_t* deviceCols = nullptr;
    index_t* deviceRowDelimiters = nullptr;
    double* deviceVector = nullptr;
    double* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceValues, std::max<size_t>(1, localNnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceCols, std::max<size_t>(1, localNnz) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&deviceRowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&deviceVector, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceOutput, localRows * sizeof(double)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (localNnz != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceValues, hostValues.data(), localNnz * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceCols, hostCols.data(), localNnz * sizeof(index_t),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(deviceRowDelimiters, hostRowDelimiters.data(),
                               (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(deviceVector, hostVector.data(), numRows * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            std::printf("Computing reference solution...\n");
        }
        reference.resize(localRows);
        spmvCpu(hostValues.data(), hostCols.data(), hostRowDelimiters.data(), hostVector.data(),
                localRows, reference.data());
    }

    // Warm up the selected kernel, then capture all iterations into one CUDA graph.
    // This removes per-iteration host launch latency without changing the work done.
    launchSpmv(stream, deviceValues, deviceCols, deviceRowDelimiters, deviceVector, localRows,
               localNnz, deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        launchSpmv(stream, deviceValues, deviceCols, deviceRowDelimiters, deviceVector,
                   localRows, localNnz, deviceOutput);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    if (rank == 0) {
        std::printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> localOutput;
    if (validate || printResults) {
        localOutput.resize(localRows);
        CUDA_CHECK(cudaMemcpy(localOutput.data(), deviceOutput, localRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    int globallyValid = 1;
    if (validate) {
        const int locallyValid = verifyResults(reference.data(), localOutput.data(), localRows,
                                               globalRowBegin, rank)
                                     ? 1
                                     : 0;
        MPI_Allreduce(&locallyValid, &globallyValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    std::vector<double> globalOutput;
    if (printResults) {
        if (rank == 0) {
            globalOutput.resize(numRows);
        }
        MPI_Gatherv(localOutput.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr, rowCounts.data(),
                    rowDisplacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const double elapsedMs = elapsed * 1000.0;
        const double averageMs = elapsedMs / iterations;
        const double gflops = (2.0 * totalNnz * iterations) / elapsed / 1.0e9;
        std::printf("Computation time: %.3f ms\n", elapsedMs);
        std::printf("Average time per iteration: %.3f ms\n", averageMs);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(globalOutput, "OutputVector");
        }
        if (validate) {
            std::printf("Validating result...\n");
            std::printf("Validation: %s\n", globallyValid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceVector));
    CUDA_CHECK(cudaFree(deviceRowDelimiters));
    CUDA_CHECK(cudaFree(deviceCols));
    CUDA_CHECK(cudaFree(deviceValues));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return globallyValid ? EXIT_SUCCESS : EXIT_FAILURE;
}
