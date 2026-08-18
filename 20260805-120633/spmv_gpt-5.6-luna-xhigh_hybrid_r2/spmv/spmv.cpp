#include <algorithm>
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

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with the same rand() sequence as the original benchmark.
//
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assign random positions to a given number of elements in a square matrix
//   and encode them in compressed sparse row (CSR) format.
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;

    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));

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
                cols[nnzAssigned] = j;
                ++nnzAssigned;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   OpenMP reference implementation of CSR sparse matrix-vector multiplication.
//   The inner reduction remains serial per row, preserving the original
//   floating-point accumulation order.
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const index_t col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result,
                   const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       i, ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// One warp owns one row. This avoids atomics and keeps the irregular CSR
// reduction local to a warp. Rows are independent, so all ranks can launch
// the same kernel on their local CSR partition without communication in the
// iteration loop.
__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec,
                              double* __restrict__ out,
                              const index_t rows) {
    const int lane = threadIdx.x & 31;
    const index_t row = static_cast<index_t>(blockIdx.x * WARPS_PER_BLOCK +
                                             threadIdx.x / 32);
    if (row >= rows) {
        return;
    }

    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t j = begin + static_cast<index_t>(lane); j < end; j += 32) {
        sum += __ldg(val + j) * __ldg(vec + __ldg(cols + j));
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

struct RowPartition {
    index_t begin;
    index_t end;
};

RowPartition partitionRows(const index_t rows, const int rank, const int ranks) {
    const uint64_t base = static_cast<uint64_t>(rows) / ranks;
    const uint64_t remainder = static_cast<uint64_t>(rows) % ranks;
    const uint64_t begin = static_cast<uint64_t>(rank) * base +
                           std::min<uint64_t>(rank, remainder);
    const uint64_t count = base + (rank < remainder ? 1 : 0);
    return {static_cast<index_t>(begin), static_cast<index_t>(begin + count)};
}

[[noreturn]] void abortMpi(const int rank, const char* message) {
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

[[noreturn]] void abortCuda(const int rank, const cudaError_t error,
                            const char* expression, const char* file,
                            const int line) {
    fprintf(stderr, "Rank %d: CUDA error at %s:%d in %s: %s\n", rank, file,
            line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define CUDA_CHECK_RANK(rank, call)                                             \
    do {                                                                        \
        const cudaError_t cuda_status = (call);                                \
        if (cuda_status != cudaSuccess) {                                      \
            abortCuda((rank), cuda_status, #call, __FILE__, __LINE__);         \
        }                                                                       \
    } while (false)

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortMpi(rank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = atof(argv[++i]);
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

    if (sparsity == 0) {
        if (rank == 0) {
            fprintf(stderr, "Sparsity must be greater than zero.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int localRank = 0;
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    MPI_Comm_rank(localCommunicator, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceCountStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceCountStatus != cudaSuccess || deviceCount == 0) {
        abortCuda(rank, deviceCountStatus == cudaSuccess
                            ? cudaErrorNoDevice
                            : deviceCountStatus,
                  "cudaGetDeviceCount", __FILE__, __LINE__);
    }
    CUDA_CHECK_RANK(rank, cudaSetDevice(localRank % deviceCount));

    const uint64_t totalMatrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nItems64 = totalMatrixEntries / sparsity;
    if (nItems64 > std::numeric_limits<index_t>::max()) {
        abortMpi(rank, "The requested matrix has more than 2^32 - 1 non-zero entries");
    }
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nItems64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        abortMpi(rank, "The requested dimensions exceed MPI count limits");
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (Hybrid MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / totalMatrixEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank zero materializes the global CSR matrix. MPI then distributes
    // contiguous row blocks, avoiding a full matrix copy on every rank.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_reference;
    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
    }
    std::vector<double> h_vec(numRows);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution with OpenMP...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    const RowPartition localRows = partitionRows(numRows, rank, ranks);
    const index_t localRowCount = localRows.end - localRows.begin;
    const index_t localNnz = rank == 0
                                 ? h_rowDelimiters[localRows.end] -
                                       h_rowDelimiters[localRows.begin]
                                 : 0;

    // Each rank needs its own local non-zero count. The root sends all counts
    // through a small metadata broadcast before the large CSR transfers.
    std::vector<index_t> allLocalNnz;
    if (rank == 0) {
        allLocalNnz.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            const RowPartition pRows = partitionRows(numRows, p, ranks);
            allLocalNnz[p] = h_rowDelimiters[pRows.end] -
                             h_rowDelimiters[pRows.begin];
        }
    } else {
        allLocalNnz.resize(ranks);
    }
    MPI_Bcast(allLocalNnz.data(), ranks, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localNnzActual = rank == 0 ? localNnz : allLocalNnz[rank];

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    if (rank == 0) {
        rowCounts.resize(ranks);
        rowDisplacements.resize(ranks);
        nnzCounts.resize(ranks);
        nnzDisplacements.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            const RowPartition pRows = partitionRows(numRows, p, ranks);
            const uint64_t pRowsCount = pRows.end - pRows.begin;
            const uint64_t pNnz = allLocalNnz[p];
            const uint64_t pNnzOffset = h_rowDelimiters[pRows.begin];
            if (pRowsCount + 1 > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                pRows.begin > static_cast<index_t>(std::numeric_limits<int>::max()) ||
                pNnz > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                pNnzOffset > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                abortMpi(rank, "CSR partition exceeds MPI_Scatterv count limits");
            }
            rowCounts[p] = static_cast<int>(pRowsCount + 1);
            rowDisplacements[p] = static_cast<int>(pRows.begin);
            nnzCounts[p] = static_cast<int>(pNnz);
            nnzDisplacements[p] = static_cast<int>(pNnzOffset);
        }
    }

    std::vector<double> localVal(localNnzActual);
    std::vector<index_t> localCols(localNnzActual);
    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRowCount) + 1);
    std::vector<double> localOut(localRowCount);

    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnzActual), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? nnzCounts.data() : nullptr,
                 rank == 0 ? nnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnzActual),
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_rowDelimiters.data() : nullptr,
                 rank == 0 ? rowCounts.data() : nullptr,
                 rank == 0 ? rowDisplacements.data() : nullptr, MPI_UINT32_T,
                 localRowDelimiters.data(), static_cast<int>(localRowCount) + 1,
                 MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Normalize row offsets to the local CSR segment. This is an unconditional
    // OpenMP operation on every rank, including non-root ranks.
    const index_t localNnzOffset = localRowDelimiters[0];
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i <= static_cast<int64_t>(localRowCount); ++i) {
        localRowDelimiters[i] -= localNnzOffset;
    }

    double* dVal = nullptr;
    index_t* dCols = nullptr;
    index_t* dRowDelimiters = nullptr;
    double* dVec = nullptr;
    double* dOut = nullptr;
    cudaStream_t stream = nullptr;

    const size_t deviceNnz = std::max<size_t>(1, localNnzActual);
    const size_t deviceRows = std::max<size_t>(1, localRowCount);
    const size_t deviceVector = std::max<size_t>(1, numRows);
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&dVal),
                                     deviceNnz * sizeof(double)));
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&dCols),
                                     deviceNnz * sizeof(index_t)));
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&dRowDelimiters),
                                     (static_cast<size_t>(localRowCount) + 1) *
                                         sizeof(index_t)));
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&dVec),
                                     deviceVector * sizeof(double)));
    CUDA_CHECK_RANK(rank, cudaMalloc(reinterpret_cast<void**>(&dOut),
                                     deviceRows * sizeof(double)));
    CUDA_CHECK_RANK(rank, cudaStreamCreateWithFlags(&stream,
                                                    cudaStreamNonBlocking));

    if (localNnzActual != 0) {
        CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                                  dVal, localVal.data(),
                                  static_cast<size_t>(localNnzActual) * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                                  dCols, localCols.data(),
                                  static_cast<size_t>(localNnzActual) * sizeof(index_t),
                                  cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                              dRowDelimiters, localRowDelimiters.data(),
                              (static_cast<size_t>(localRowCount) + 1) *
                                  sizeof(index_t),
                              cudaMemcpyHostToDevice, stream));
    if (localRowCount != 0) {
        // std::vector output is value-initialized in the original benchmark;
        // this also defines the result for the valid edge case iterations=0.
        CUDA_CHECK_RANK(rank, cudaMemsetAsync(
                                  dOut, 0,
                                  static_cast<size_t>(localRowCount) *
                                      sizeof(double),
                                  stream));
    }
    if (numRows != 0) {
        CUDA_CHECK_RANK(rank, cudaMemcpyAsync(
                                  dVec, h_vec.data(),
                                  static_cast<size_t>(numRows) * sizeof(double),
                                  cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK_RANK(rank, cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK_RANK(rank, cudaEventCreate(&startEvent));
    CUDA_CHECK_RANK(rank, cudaEventCreate(&stopEvent));
    CUDA_CHECK_RANK(rank, cudaEventRecord(startEvent, stream));

    const int blocks = static_cast<int>((localRowCount + WARPS_PER_BLOCK - 1) /
                                         WARPS_PER_BLOCK);
    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRowCount != 0) {
            spmvCsrKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
                dVal, dCols, dRowDelimiters, dVec, dOut, localRowCount);
        }
    }
    CUDA_CHECK_RANK(rank, cudaGetLastError());
    CUDA_CHECK_RANK(rank, cudaEventRecord(stopEvent, stream));
    CUDA_CHECK_RANK(rank, cudaEventSynchronize(stopEvent));

    float localMillisecondsFloat = 0.0f;
    CUDA_CHECK_RANK(rank, cudaEventElapsedTime(&localMillisecondsFloat,
                                               startEvent, stopEvent));
    const double localMilliseconds = static_cast<double>(localMillisecondsFloat);
    double maximumMilliseconds = 0.0;
    MPI_Allreduce(&localMilliseconds, &maximumMilliseconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);

    if (localRowCount != 0) {
        CUDA_CHECK_RANK(rank, cudaMemcpy(localOut.data(), dOut,
                                         static_cast<size_t>(localRowCount) *
                                             sizeof(double),
                                         cudaMemcpyDeviceToHost));
    }

    std::vector<int> outputCounts;
    std::vector<int> outputDisplacements;
    std::vector<double> h_out;
    if (rank == 0) {
        outputCounts.resize(ranks);
        outputDisplacements.resize(ranks);
        h_out.resize(numRows);
        for (int p = 0; p < ranks; ++p) {
            const RowPartition pRows = partitionRows(numRows, p, ranks);
            outputCounts[p] = static_cast<int>(pRows.end - pRows.begin);
            outputDisplacements[p] = static_cast<int>(pRows.begin);
        }
    }
    MPI_Gatherv(localOut.data(), static_cast<int>(localRowCount), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? outputCounts.data() : nullptr,
                rank == 0 ? outputDisplacements.data() : nullptr, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing SpMV with CUDA (%d MPI rank(s), %d OpenMP thread(s) per rank)...\n",
               ranks, omp_get_max_threads());
        printf("Computation time: %.3f ms\n", maximumMilliseconds);

        const double seconds = maximumMilliseconds / 1000.0;
        const double gflops = seconds > 0.0
                                  ? (2.0 * nItems * iterations) / seconds / 1e9
                                  : 0.0;
        const double avgTime = iterations != 0
                                   ? maximumMilliseconds / iterations
                                   : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int validationPassed = 1;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        validationPassed = verifyResults(h_reference.data(), h_out.data(), numRows)
                               ? 1
                               : 0;
        printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
    }
    if (validate) {
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK_RANK(rank, cudaEventDestroy(startEvent));
    CUDA_CHECK_RANK(rank, cudaEventDestroy(stopEvent));
    CUDA_CHECK_RANK(rank, cudaStreamDestroy(stream));
    CUDA_CHECK_RANK(rank, cudaFree(dVal));
    CUDA_CHECK_RANK(rank, cudaFree(dCols));
    CUDA_CHECK_RANK(rank, cudaFree(dRowDelimiters));
    CUDA_CHECK_RANK(rank, cudaFree(dVec));
    CUDA_CHECK_RANK(rank, cudaFree(dOut));
    MPI_Comm_free(&localCommunicator);
    MPI_Finalize();

    return validate ? (validationPassed ? 0 : 1) : 0;
}
