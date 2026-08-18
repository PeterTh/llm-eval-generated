#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int CUDA_BLOCK_THREADS = 256;
constexpr int CUDA_WARPS_PER_BLOCK = CUDA_BLOCK_THREADS / 32;

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize an array with the same deterministic random sequence as the
//   original benchmark.  Random number generation is intentionally kept
//   serial so that results remain reproducible; the conversion and stores are
//   parallelized with OpenMP.
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    std::vector<std::uint32_t> randomValues(n);
    for (index_t i = 0; i < n; ++i) {
        randomValues[i] = static_cast<std::uint32_t>(rand());
    }

#pragma omp parallel for simd schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        A[i] = maxVal * (static_cast<double>(randomValues[i]) /
                         (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
//
// Purpose:
//   Assign random positions to a given number of elements in a square matrix
//   and encode them in compressed sparse row (CSR) format.  The random stream
//   and traversal order match the original implementation.
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;

    const double matrixSize = static_cast<double>(dim) * static_cast<double>(dim);
    const double prob = static_cast<double>(n) / matrixSize;

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are
    // assigned.  Use 64-bit arithmetic for the bookkeeping so large matrix
    // dimensions do not wrap before the requested CSR size is reached.
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const std::uint64_t position =
                static_cast<std::uint64_t>(i) * dim + j;
            const std::uint64_t numEntriesLeft =
                static_cast<std::uint64_t>(dim) * dim - position;
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

    // Convention: put the number of non-zeroes at the end of the row
    // delimiters array.
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format.  This is
//   used only for the validation reference, but is also OpenMP-parallelized so
//   validation does not become a serial bottleneck on rank 0.
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(dim); ++row) {
        const index_t i = static_cast<index_t>(row);
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const index_t col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// CUDA kernels
//
// The warp kernel assigns one warp to each row.  This avoids atomics and
// gives long CSR rows enough independent memory requests.  For very sparse
// matrices, the scalar kernel avoids wasting 31 lanes on each short row.
// ****************************************************************************
__device__ double warpReduceSum(double value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void spmvWarpKernel(const double* __restrict__ val,
                               const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters,
                               const double* __restrict__ vec,
                               const index_t numRows,
                               double* __restrict__ out) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp =
        (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    if (warp >= numRows) {
        return;
    }

    const index_t begin = rowDelimiters[warp];
    const index_t end = rowDelimiters[warp + 1];
    double sum = 0.0;
    for (std::uint64_t j = static_cast<std::uint64_t>(begin) + lane;
         j < end; j += 32) {
        const index_t entry = static_cast<index_t>(j);
        sum += val[entry] * vec[cols[entry]];
    }

    sum = warpReduceSum(sum);
    if (lane == 0) {
        out[warp] = sum;
    }
}

__global__ void spmvScalarKernel(const double* __restrict__ val,
                                 const index_t* __restrict__ cols,
                                 const index_t* __restrict__ rowDelimiters,
                                 const double* __restrict__ vec,
                                 const index_t numRows,
                                 double* __restrict__ out) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) {
        return;
    }

    double sum = 0.0;
    for (index_t j = rowDelimiters[row]; j < rowDelimiters[row + 1]; ++j) {
        sum += val[j] * vec[cols[j]];
    }
    out[row] = sum;
}

void launchSpmv(const double* d_val, const index_t* d_cols,
                const index_t* d_rowDelimiters, const double* d_vec,
                const index_t localRows, const index_t localNnz,
                double* d_out) {
    if (localRows == 0) {
        return;
    }

    // A warp is beneficial once rows have enough work to amortize the
    // reduction.  The scalar path remains a CUDA path and is faster for the
    // short rows produced by very sparse inputs.
    const bool useWarpKernel =
        static_cast<std::uint64_t>(localNnz) >=
        static_cast<std::uint64_t>(localRows) * 8u;
    if (useWarpKernel) {
        const unsigned int blocks =
            (localRows + CUDA_WARPS_PER_BLOCK - 1) / CUDA_WARPS_PER_BLOCK;
        spmvWarpKernel<<<blocks, CUDA_BLOCK_THREADS>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    } else {
        const unsigned int blocks =
            (localRows + CUDA_BLOCK_THREADS - 1) / CUDA_BLOCK_THREADS;
        spmvScalarKernel<<<blocks, CUDA_BLOCK_THREADS>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out);
    }
}

// ****************************************************************************
// Function: verifyResults
// ****************************************************************************
bool verifyResults(const double* reference, const double* result,
                   const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                       i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e "
                       "(rel error: %.10e)\n",
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

[[noreturn]] void abortMpi(int rank, const char* message) {
    fprintf(stderr, "MPI rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* expression, const int line,
               const int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "CUDA error at line %d in %s: %s",
                      line, expression, cudaGetErrorString(status));
        abortMpi(rank, message);
    }
}

#define CUDA_CHECK(call, rank) \
    checkCuda((call), #call, __LINE__, (rank))

#define MPI_CHECK(call, rank)                                      \
    do {                                                           \
        const int mpiError = (call);                              \
        if (mpiError != MPI_SUCCESS) {                             \
            char mpiMessage[MPI_MAX_ERROR_STRING + 1] = {};       \
            int mpiMessageLength = 0;                             \
            MPI_Error_string(mpiError, mpiMessage, &mpiMessageLength); \
            mpiMessage[mpiMessageLength] = '\0';                  \
            char message[512];                                     \
            std::snprintf(message, sizeof(message),               \
                          "MPI error in MPI call: %.240s", mpiMessage); \
            abortMpi((rank), message);                             \
        }                                                          \
    } while (false)

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortMpi(worldRank, "MPI implementation does not provide MPI_THREAD_FUNNELED");
    }

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badArguments = false;

    // Parse command line arguments identically on every MPI rank.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            badArguments = true;
        }
    }

    if (worldRank == 0 && showHelp) {
        printUsage(argv[0]);
    }
    if (showHelp) {
        MPI_Finalize();
        return 0;
    }
    if (badArguments || numRows == 0 || sparsity == 0 || iterations == 0) {
        if (worldRank == 0) {
            if (badArguments) {
                printf("Unknown or incomplete option.\n");
            }
            if (numRows == 0 || sparsity == 0 || iterations == 0) {
                printf("Matrix size, sparsity, and iterations must be greater than zero.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const std::uint64_t matrixEntries =
        static_cast<std::uint64_t>(numRows) * numRows;
    const std::uint64_t nItems64 = matrixEntries / sparsity;
    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max()) ||
        nItems64 > static_cast<std::uint64_t>(std::numeric_limits<index_t>::max()) ||
        nItems64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        abortMpi(worldRank,
                 "Matrix dimensions or non-zero count exceed the supported MPI/CSR range");
    }
    const index_t nItems = static_cast<index_t>(nItems64);

    // Select a GPU by local MPI rank, so rank numbering remains correct on a
    // multi-node cluster.  Every rank initializes CUDA and executes its own
    // local SpMV; there is no CPU fallback path.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localComm),
              worldRank);
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank), worldRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount), worldRank);
    if (deviceCount == 0) {
        abortMpi(worldRank, "No CUDA accelerator is visible to this MPI rank");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device), worldRank);
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device), worldRank);

    const std::uint64_t rowBegin64 =
        static_cast<std::uint64_t>(numRows) * worldRank / worldSize;
    const std::uint64_t rowEnd64 =
        static_cast<std::uint64_t>(numRows) * (worldRank + 1) / worldSize;
    const index_t rowBegin = static_cast<index_t>(rowBegin64);
    const index_t rowEnd = static_cast<index_t>(rowEnd64);
    const index_t localRows = rowEnd - rowBegin;

    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize,
               omp_get_max_threads());
        printf("CUDA device: %s\n", deviceProperties.name);
    }

    // Rank 0 owns the deterministic global input.  CSR rows and non-zeros are
    // then distributed once; only the dense vector is replicated.
    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<index_t> globalRowDelimiters;
    std::vector<index_t> globalRowLengths;
    std::vector<double> h_reference;
    std::vector<double> h_out;
    std::vector<double> h_vec(numRows);

    if (worldRank == 0) {
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);
        globalRowLengths.resize(numRows);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems,
                         numRows);

#pragma omp parallel for schedule(static)
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(numRows);
             ++row) {
            globalRowLengths[row] =
                globalRowDelimiters[row + 1] - globalRowDelimiters[row];
        }

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(globalVal.data(), globalCols.data(),
                    globalRowDelimiters.data(), h_vec.data(), numRows,
                    h_reference.data());
        }
    }

    MPI_CHECK(MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0,
                        MPI_COMM_WORLD),
              worldRank);

    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> nnzCounts;
    std::vector<int> nnzDisplacements;
    if (worldRank == 0) {
        rowCounts.resize(worldSize);
        rowDisplacements.resize(worldSize);
        nnzCounts.resize(worldSize);
        nnzDisplacements.resize(worldSize);
        for (int rank = 0; rank < worldSize; ++rank) {
            const index_t begin = static_cast<index_t>(
                static_cast<std::uint64_t>(numRows) * rank / worldSize);
            const index_t end = static_cast<index_t>(
                static_cast<std::uint64_t>(numRows) * (rank + 1) / worldSize);
            rowCounts[rank] = static_cast<int>(end - begin);
            rowDisplacements[rank] = static_cast<int>(begin);
            nnzCounts[rank] = static_cast<int>(
                globalRowDelimiters[end] - globalRowDelimiters[begin]);
            nnzDisplacements[rank] = static_cast<int>(globalRowDelimiters[begin]);
        }
    }

    std::vector<index_t> localRowLengths(localRows);
    MPI_CHECK(MPI_Scatterv(
                  worldRank == 0 ? globalRowLengths.data() : nullptr,
                  worldRank == 0 ? rowCounts.data() : nullptr,
                  worldRank == 0 ? rowDisplacements.data() : nullptr,
                  MPI_UINT32_T, localRowLengths.data(), static_cast<int>(localRows),
                  MPI_UINT32_T, 0, MPI_COMM_WORLD),
              worldRank);

    std::vector<index_t> localRowDelimiters(static_cast<std::size_t>(localRows) + 1,
                                            0);
    for (index_t row = 0; row < localRows; ++row) {
        localRowDelimiters[row + 1] =
            localRowDelimiters[row] + localRowLengths[row];
    }
    const index_t localNnz = localRowDelimiters[localRows];

    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_CHECK(MPI_Scatterv(worldRank == 0 ? globalVal.data() : nullptr,
                           worldRank == 0 ? nnzCounts.data() : nullptr,
                           worldRank == 0 ? nnzDisplacements.data() : nullptr,
                           MPI_DOUBLE, localVal.data(), static_cast<int>(localNnz),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD),
              worldRank);
    MPI_CHECK(MPI_Scatterv(worldRank == 0 ? globalCols.data() : nullptr,
                           worldRank == 0 ? nnzCounts.data() : nullptr,
                           worldRank == 0 ? nnzDisplacements.data() : nullptr,
                           MPI_UINT32_T, localCols.data(), static_cast<int>(localNnz),
                           MPI_UINT32_T, 0, MPI_COMM_WORLD),
              worldRank);

    // The global CSR is no longer needed after distribution; clearing every
    // rank's root copies reduces peak job memory before the timed phase.
    globalVal.clear();
    globalVal.shrink_to_fit();
    globalCols.clear();
    globalCols.shrink_to_fit();
    globalRowDelimiters.clear();
    globalRowDelimiters.shrink_to_fit();
    globalRowLengths.clear();
    globalRowLengths.shrink_to_fit();

    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (localNnz > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_val),
                              static_cast<std::size_t>(localNnz) * sizeof(double)),
                   worldRank);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cols),
                              static_cast<std::size_t>(localNnz) * sizeof(index_t)),
                   worldRank);
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rowDelimiters),
                          (static_cast<std::size_t>(localRows) + 1) *
                              sizeof(index_t)),
               worldRank);
    if (numRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vec),
                              static_cast<std::size_t>(numRows) * sizeof(double)),
                   worldRank);
    }
    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out),
                              static_cast<std::size_t>(localRows) * sizeof(double)),
                   worldRank);
    }

    if (localNnz > 0) {
        CUDA_CHECK(cudaMemcpy(d_val, localVal.data(),
                              static_cast<std::size_t>(localNnz) * sizeof(double),
                              cudaMemcpyHostToDevice),
                   worldRank);
        CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(),
                              static_cast<std::size_t>(localNnz) * sizeof(index_t),
                              cudaMemcpyHostToDevice),
                   worldRank);
    }
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(),
                          (static_cast<std::size_t>(localRows) + 1) *
                              sizeof(index_t),
                          cudaMemcpyHostToDevice),
               worldRank);
    if (numRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(),
                              static_cast<std::size_t>(numRows) * sizeof(double),
                              cudaMemcpyHostToDevice),
                   worldRank);
    }

    // Synchronize ranks before timing.  The timed region contains all CUDA
    // kernel launches and device execution, but excludes one-time MPI/CUDA
    // data movement and the final result copy.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD), worldRank);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, localRows, localNnz,
                   d_out);
    }
    CUDA_CHECK(cudaGetLastError(), worldRank);
    CUDA_CHECK(cudaDeviceSynchronize(), worldRank);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX,
                         0, MPI_COMM_WORLD),
              worldRank);

    std::vector<double> localOut(localRows);
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(localOut.data(), d_out,
                              static_cast<std::size_t>(localRows) * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   worldRank);
    }

    MPI_CHECK(MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                          worldRank == 0 ? h_out.data() : nullptr,
                          worldRank == 0 ? rowCounts.data() : nullptr,
                          worldRank == 0 ? rowDisplacements.data() : nullptr,
                          MPI_DOUBLE, 0, MPI_COMM_WORLD),
              worldRank);

    if (worldRank == 0) {
        const double totalMilliseconds = elapsedSeconds * 1000.0;
        const double avgTime = totalMilliseconds / iterations;
        const double gflops =
            (2.0 * static_cast<double>(nItems) * iterations) /
            elapsedSeconds / 1e9;

        printf("Computation time: %.3f ms\n", totalMilliseconds);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(h_out, "OutputVector");
        }
    }

    int exitCode = 0;
    if (validate && worldRank == 0) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD), worldRank);

    if (d_val != nullptr) {
        CUDA_CHECK(cudaFree(d_val), worldRank);
    }
    if (d_cols != nullptr) {
        CUDA_CHECK(cudaFree(d_cols), worldRank);
    }
    CUDA_CHECK(cudaFree(d_rowDelimiters), worldRank);
    if (d_vec != nullptr) {
        CUDA_CHECK(cudaFree(d_vec), worldRank);
    }
    if (d_out != nullptr) {
        CUDA_CHECK(cudaFree(d_out), worldRank);
    }
    MPI_CHECK(MPI_Comm_free(&localComm), worldRank);
    MPI_Finalize();
    return exitCode;
}
