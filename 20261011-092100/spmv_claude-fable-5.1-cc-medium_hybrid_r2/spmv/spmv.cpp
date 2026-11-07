// Hybrid MPI + OpenMP + CUDA sparse matrix-vector multiplication benchmark.
//
// Parallelization strategy:
//   * MPI:    the CSR matrix is generated on rank 0 (the sequential rand()
//             stream must be preserved for equivalent semantics), then rows are
//             partitioned across ranks in contiguous blocks balanced by the
//             number of non-zeros.  Each rank owns a row slice of the matrix and
//             a full copy of the dense input vector.  Results are gathered back
//             to rank 0 for output and validation.
//   * CUDA:   every rank binds to one GPU (selected by node-local rank) and runs
//             a CSR kernel in which several threads (1..32, chosen from the
//             average row length) cooperate on one row with shuffle reductions.
//   * OpenMP: host-side work that is not bound by the sequential RNG (CSR row
//             pointer rebasing, reference solution, validation) is threaded.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                                    \
    do {                                                                                   \
        int err__ = (call);                                                                \
        if (err__ != MPI_SUCCESS) {                                                        \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,       \
                    __LINE__);                                                             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
//
// Arguments:
//   A: pointer to the array to initialize
//   n: number of elements in the array
//   maxVal: specifies range of random values [0, maxVal]
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
//   Assigns random positions to a given number of elements in a square
//   matrix. The function encodes these positions in compressed sparse
//   row (CSR) format.
//
// Arguments:
//   cols:          array for column indexes of elements (size should be = n)
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   n:             number of nonzero elements
//   dim:           number of rows/columns in the matrix
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    // Convention: put the number of non-zeroes at the end of the row delimiters array
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on the host
//   (OpenMP-threaded).  Used for the reference solution during validation.
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector of size dim to be used for multiplication
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Kernel: spmvCsrKernel
//
// Purpose:
//   CSR SpMV where THREADS_PER_ROW consecutive lanes of a warp cooperate on a
//   single row; partial sums are combined with warp shuffles.  THREADS_PER_ROW
//   is selected on the host from the average row length so that short rows do
//   not waste lanes and long rows get coalesced access to val/cols.
//
// ****************************************************************************
constexpr int SPMV_BLOCK_SIZE = 256;

template <int THREADS_PER_ROW>
__global__ void __launch_bounds__(SPMV_BLOCK_SIZE)
spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
              const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
              const index_t numRows, double* __restrict__ out) {
    constexpr int ROWS_PER_BLOCK = SPMV_BLOCK_SIZE / THREADS_PER_ROW;
    const int lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const index_t row = static_cast<index_t>(blockIdx.x) * ROWS_PER_BLOCK +
                        (threadIdx.x / THREADS_PER_ROW);
    if (row >= numRows) return;

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double t = 0.0;
    if constexpr (THREADS_PER_ROW == 1) {
        for (index_t j = rowStart; j < rowEnd; ++j) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
    } else {
        // Lanes stride through the row so that neighbouring lanes read
        // neighbouring val/cols entries (coalesced).
        for (index_t j = rowStart + lane; j < rowEnd; j += THREADS_PER_ROW) {
            t += val[j] * __ldg(&vec[cols[j]]);
        }
        // Reduce within the THREADS_PER_ROW sub-group of the warp.
        constexpr unsigned FULL_MASK = 0xffffffffu;
#pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(FULL_MASK, t, offset, THREADS_PER_ROW);
        }
    }
    if (lane == 0) {
        out[row] = t;
    }
}

// Launches the kernel instantiation matching the chosen threads-per-row.
void launchSpmv(const int threadsPerRow, const double* d_val, const index_t* d_cols,
                const index_t* d_rowDelimiters, const double* d_vec, const index_t numRows,
                double* d_out, cudaStream_t stream) {
    if (numRows == 0) return;
    const int rowsPerBlock = SPMV_BLOCK_SIZE / threadsPerRow;
    const unsigned grid = static_cast<unsigned>((numRows + rowsPerBlock - 1) / rowsPerBlock);
    switch (threadsPerRow) {
        case 32:
            spmvCsrKernel<32><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                    d_vec, numRows, d_out);
            break;
        case 16:
            spmvCsrKernel<16><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                    d_vec, numRows, d_out);
            break;
        case 8:
            spmvCsrKernel<8><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                   d_vec, numRows, d_out);
            break;
        case 4:
            spmvCsrKernel<4><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                   d_vec, numRows, d_out);
            break;
        case 2:
            spmvCsrKernel<2><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                   d_vec, numRows, d_out);
            break;
        default:
            spmvCsrKernel<1><<<grid, SPMV_BLOCK_SIZE, 0, stream>>>(d_val, d_cols, d_rowDelimiters,
                                                                   d_vec, numRows, d_out);
            break;
    }
}

// Picks the number of cooperating lanes per row from the average row length.
int chooseThreadsPerRow(const index_t localRows, const index_t localNnz) {
    if (localRows == 0) return 1;
    const double avg = static_cast<double>(localNnz) / static_cast<double>(localRows);
    if (avg >= 32.0) return 32;
    if (avg >= 16.0) return 16;
    if (avg >= 8.0) return 8;
    if (avg >= 4.0) return 4;
    if (avg >= 2.0) return 2;
    return 1;
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//   (OpenMP-threaded; reports the lowest failing index like the serial scan).
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    index_t firstBad = size;
#pragma omp parallel for schedule(static) reduction(min : firstBad)
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        bool bad;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            bad = std::abs(res) > MAX_RELATIVE_ERROR;
        } else {
            // Check relative error
            bad = std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
        }
        if (bad && i < firstBad) firstBad = i;
    }
    if (firstBad == size) return true;

    const index_t i = firstBad;
    const double ref = reference[i];
    const double res = result[i];
    if (std::abs(ref) < 1e-10) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
    } else {
        const double relError = std::abs((res - ref) / ref);
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n", i,
               ref, res, relError);
    }
    return false;
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

// Binds the calling rank to a GPU based on its node-local rank.
static void selectDevice(const int worldRank) {
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available (%s)\n", worldRank,
                cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context now, outside the timed region
}

static int toMpiCount(const size_t n) {
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Per-rank element count %zu exceeds MPI int range\n", n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(n);
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int worldRank = 0, worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    const bool isRoot = (worldRank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank; only root prints)
    int parseStatus = -1;  // -1: continue, otherwise exit code
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
            if (isRoot) printUsage(argv[0]);
            parseStatus = 0;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus >= 0) {
        MPI_Finalize();
        return parseStatus;
    }

    selectDevice(worldRank);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", worldSize, omp_get_max_threads());
    }

    // ------------------------------------------------------------------
    // Global data: generated on root only (sequential RNG stream).
    // ------------------------------------------------------------------
    std::vector<double> h_val;            // Non-zero values
    std::vector<index_t> h_cols;          // Column indices
    std::vector<index_t> h_rowDelimiters; // Row delimiters
    std::vector<double> h_vec(numRows);   // Dense vector (replicated on every rank)
    std::vector<double> h_out;            // Output vector (root only)

    // Row partition boundaries: rank r owns rows [rowStart[r], rowStart[r+1]).
    std::vector<index_t> rowStart(worldSize + 1, 0);

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Partition rows so that every rank receives roughly nItems/worldSize
        // non-zeros (contiguous row blocks keep the CSR slices trivially local).
        rowStart[0] = 0;
        for (int r = 1; r < worldSize; ++r) {
            const index_t target = static_cast<index_t>(
                (static_cast<uint64_t>(nItems) * static_cast<uint64_t>(r)) /
                static_cast<uint64_t>(worldSize));
            const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
            index_t row = static_cast<index_t>(it - h_rowDelimiters.begin());
            row = std::min(row, numRows);
            rowStart[r] = std::max(row, rowStart[r - 1]);
        }
        rowStart[worldSize] = numRows;
    }

    // ------------------------------------------------------------------
    // Distribute: partition table, dense vector, CSR row slices.
    // ------------------------------------------------------------------
    MPI_CHECK(MPI_Bcast(rowStart.data(), worldSize + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(h_vec.data(), toMpiCount(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD));

    const index_t myRow0 = rowStart[worldRank];
    const index_t localRows = rowStart[worldRank + 1] - myRow0;

    std::vector<int> rowCounts(worldSize), rowDispls(worldSize);
    std::vector<int> nnzCounts(worldSize), nnzDispls(worldSize);
    if (isRoot) {
        for (int r = 0; r < worldSize; ++r) {
            rowCounts[r] = toMpiCount(rowStart[r + 1] - rowStart[r] + 1);
            rowDispls[r] = toMpiCount(rowStart[r]);
            const index_t nnz0 = h_rowDelimiters[rowStart[r]];
            const index_t nnz1 = h_rowDelimiters[rowStart[r + 1]];
            nnzCounts[r] = toMpiCount(nnz1 - nnz0);
            nnzDispls[r] = toMpiCount(nnz0);
        }
    }

    // Local CSR slice (row delimiters still carry the global nnz offset).
    std::vector<index_t> l_rowDelimiters(localRows + 1);
    MPI_CHECK(MPI_Scatterv(isRoot ? h_rowDelimiters.data() : nullptr, rowCounts.data(),
                           rowDispls.data(), MPI_UINT32_T, l_rowDelimiters.data(),
                           toMpiCount(localRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD));
    const index_t nnzOffset = l_rowDelimiters[0];
    const index_t localNnz = l_rowDelimiters[localRows] - nnzOffset;

    // Rebase row delimiters to the local slice.
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        l_rowDelimiters[i] -= nnzOffset;
    }

    std::vector<double> l_val(localNnz);
    std::vector<index_t> l_cols(localNnz);
    MPI_CHECK(MPI_Scatterv(isRoot ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                           MPI_DOUBLE, l_val.data(), toMpiCount(localNnz), MPI_DOUBLE, 0,
                           MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(isRoot ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                           MPI_UINT32_T, l_cols.data(), toMpiCount(localNnz), MPI_UINT32_T, 0,
                           MPI_COMM_WORLD));

    // For validation, compute reference solution (root, OpenMP-threaded)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows,
                h_reference.data());
    }

    // ------------------------------------------------------------------
    // Upload the local slice to the GPU.
    // ------------------------------------------------------------------
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    // cudaMalloc(0) is legal but returns nullptr; keep at least one element.
    const size_t nnzAlloc = std::max<size_t>(localNnz, 1);
    const size_t rowAlloc = std::max<size_t>(localRows, 1);
    const size_t vecAlloc = std::max<size_t>(numRows, 1);
    CUDA_CHECK(cudaMalloc(&d_val, nnzAlloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnzAlloc * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (rowAlloc + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, vecAlloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, rowAlloc * sizeof(double)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(d_val, l_val.data(), localNnz * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cols, l_cols.data(), localNnz * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rowDelimiters, l_rowDelimiters.data(),
                               (localRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), numRows * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    const int threadsPerRow = chooseThreadsPerRow(localRows, localNnz);

    // Warm-up launch (JIT/module load, cache fill) outside the timed region.
    launchSpmv(threadsPerRow, d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------
    // Perform SpMV computation
    // ------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv(threadsPerRow, d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ------------------------------------------------------------------
    // Gather the result slices to root.
    // ------------------------------------------------------------------
    std::vector<double> l_out(localRows);
    CUDA_CHECK(cudaMemcpyAsync(l_out.data(), d_out, localRows * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> outCounts(worldSize), outDispls(worldSize);
    if (isRoot) {
        for (int r = 0; r < worldSize; ++r) {
            outCounts[r] = toMpiCount(rowStart[r + 1] - rowStart[r]);
            outDispls[r] = toMpiCount(rowStart[r]);
        }
    }
    MPI_CHECK(MPI_Gatherv(l_out.data(), toMpiCount(localRows), MPI_DOUBLE,
                          isRoot ? h_out.data() : nullptr, outCounts.data(), outDispls.data(),
                          MPI_DOUBLE, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(h_out, "OutputVector");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}
