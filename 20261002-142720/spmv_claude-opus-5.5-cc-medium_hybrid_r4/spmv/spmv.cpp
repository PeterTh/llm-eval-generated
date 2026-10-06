#include <algorithm>
#include <chrono>
#include <cmath>
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
constexpr int BLOCK_SIZE = 256;
// Max bytes per MPI message (keeps counts well inside int range)
constexpr size_t MPI_CHUNK_BYTES = size_t(1) << 30;

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
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
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP-parallel over rows; used for the reference solution)
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
#pragma omp parallel for schedule(dynamic, 256)
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
// Function: spmvCsrVectorKernel
//
// Purpose:
//   CSR-vector SpMV on the GPU. A group of TPR consecutive threads (within
//   a warp) cooperates on one row: coalesced loads of val/cols, then a
//   shuffle reduction. rowDelimiters are local to this rank's row block.
//   All threads stay active (no early exit) so full-warp shuffles are safe.
//
// ****************************************************************************
template <int TPR>
__global__ void __launch_bounds__(BLOCK_SIZE)
spmvCsrVectorKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                    const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                    const index_t numRows, double* __restrict__ out) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = tid / TPR;
    const unsigned lane = threadIdx.x & (TPR - 1);
    const bool active = row < numRows;

    index_t rowStart = 0, rowEnd = 0;
    if (active) {
        rowStart = __ldg(&rowDelimiters[row]);
        rowEnd = __ldg(&rowDelimiters[row + 1]);
    }

    double t = 0.0;
#pragma unroll 4
    for (index_t j = rowStart + lane; j < rowEnd; j += TPR) {
        t += __ldg(&val[j]) * __ldg(&vec[__ldg(&cols[j])]);
    }

#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, offset, TPR);
    }
    if (active && lane == 0) out[row] = t;
}

template <int TPR>
void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, const index_t numRows, double* out, cudaStream_t stream) {
    const size_t threads = static_cast<size_t>(numRows) * TPR;
    const unsigned blocks = static_cast<unsigned>((threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    spmvCsrVectorKernel<TPR><<<blocks, BLOCK_SIZE, 0, stream>>>(val, cols, rowDelimiters, vec,
                                                                numRows, out);
}

// Pick threads-per-row from the mean row length of the local block
using SpmvLauncher = void (*)(const double*, const index_t*, const index_t*, const double*,
                              index_t, double*, cudaStream_t);
SpmvLauncher selectLauncher(const double avgRowLen) {
    if (avgRowLen <= 2.0) return launchSpmv<2>;
    if (avgRowLen <= 4.0) return launchSpmv<4>;
    if (avgRowLen <= 8.0) return launchSpmv<8>;
    if (avgRowLen <= 16.0) return launchSpmv<16>;
    return launchSpmv<32>;
}

// ****************************************************************************
// MPI helpers for messages that may exceed INT_MAX elements
// ****************************************************************************
void sendLarge(const void* buf, size_t bytes, int dest, int tag) {
    const char* p = static_cast<const char*>(buf);
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, MPI_CHUNK_BYTES);
        MPI_Send(p, static_cast<int>(chunk), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
        p += chunk;
        bytes -= chunk;
    }
}

void recvLarge(void* buf, size_t bytes, int src, int tag) {
    char* p = static_cast<char*>(buf);
    while (bytes > 0) {
        const size_t chunk = std::min(bytes, MPI_CHUNK_BYTES);
        MPI_Recv(p, static_cast<int>(chunk), MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        p += chunk;
        bytes -= chunk;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//   (OpenMP-parallel; reports the first failing index like the serial scan)
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
static bool entryFails(const double ref, const double res) {
    if (std::abs(ref) < 1e-10) {
        // For very small values, check absolute error
        return std::abs(res) > MAX_RELATIVE_ERROR;
    }
    // Check relative error
    return std::abs((res - ref) / ref) > MAX_RELATIVE_ERROR;
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    index_t firstFail = size;
#pragma omp parallel for reduction(min : firstFail)
    for (index_t i = 0; i < size; ++i) {
        if (i < firstFail && entryFails(reference[i], result[i])) firstFail = i;
    }
    if (firstFail == size) return true;

    const index_t i = firstFail;
    const double ref = reference[i];
    const double res = result[i];
    if (std::abs(ref) < 1e-10) {
        printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
    } else {
        const double relError = std::abs((res - ref) / ref);
        printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
               i, ref, res, relError);
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

int run(int argc, char** argv, const int rank, const int nranks) {
    const bool root = (rank == 0);
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank)
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Bind each rank to a GPU (round-robin over node-local ranks)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
        CUDA_CHECK(cudaFree(nullptr));  // establish context early
    }

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Full data structures live on the root only (the rand() sequence is serial)
    std::vector<double> h_val;            // Non-zero values
    std::vector<index_t> h_cols;          // Column indices
    std::vector<index_t> h_rowDelimiters; // Row delimiters
    std::vector<double> h_vec(numRows);   // Dense vector (replicated on all ranks)
    std::vector<double> h_out;            // Output vector (root)
    std::vector<double> h_reference;

    // Row partition: rowBegin[r]..rowBegin[r+1] belongs to rank r
    std::vector<index_t> rowBegin(nranks + 1, 0);

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Balance work (nnz + per-row overhead) across ranks
        const double totalCost = static_cast<double>(nItems) + numRows;
        index_t row = 0;
        for (int r = 1; r < nranks; ++r) {
            const double target = totalCost * r / nranks;
            while (row < numRows &&
                   static_cast<double>(h_rowDelimiters[row]) + row < target) {
                ++row;
            }
            rowBegin[r] = row;
        }
        rowBegin[nranks] = numRows;

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Distribute partition, dense vector and local CSR blocks
    MPI_Bcast(rowBegin.data(), nranks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t localFirstRow = rowBegin[rank];
    const index_t localRows = rowBegin[rank + 1] - localFirstRow;

    std::vector<index_t> localRowDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<index_t> recvCols;
    std::vector<double> recvVal;
    const index_t* localCols = nullptr;
    const double* localVal = nullptr;
    index_t localNnz = 0;
    index_t localNnzBegin = 0;

    if (root) {
        for (int r = 1; r < nranks; ++r) {
            const index_t rb = rowBegin[r], re = rowBegin[r + 1];
            const index_t nb = h_rowDelimiters[rb], ne = h_rowDelimiters[re];
            sendLarge(&h_rowDelimiters[rb], (static_cast<size_t>(re - rb) + 1) * sizeof(index_t), r, 0);
            sendLarge(h_cols.data() + nb, static_cast<size_t>(ne - nb) * sizeof(index_t), r, 1);
            sendLarge(h_val.data() + nb, static_cast<size_t>(ne - nb) * sizeof(double), r, 2);
        }
        std::copy(h_rowDelimiters.begin() + localFirstRow,
                  h_rowDelimiters.begin() + localFirstRow + localRows + 1,
                  localRowDelimiters.begin());
        localNnzBegin = localRowDelimiters[0];
        localNnz = localRowDelimiters[localRows] - localNnzBegin;
        localCols = h_cols.data() + localNnzBegin;
        localVal = h_val.data() + localNnzBegin;
    } else {
        recvLarge(localRowDelimiters.data(), localRowDelimiters.size() * sizeof(index_t), 0, 0);
        localNnzBegin = localRowDelimiters[0];
        localNnz = localRowDelimiters[localRows] - localNnzBegin;
        recvCols.resize(localNnz);
        recvVal.resize(localNnz);
        recvLarge(recvCols.data(), static_cast<size_t>(localNnz) * sizeof(index_t), 0, 1);
        recvLarge(recvVal.data(), static_cast<size_t>(localNnz) * sizeof(double), 0, 2);
        localCols = recvCols.data();
        localVal = recvVal.data();
    }

    // Rebase local row delimiters to start at zero
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i <= static_cast<size_t>(localRows); ++i) {
        localRowDelimiters[i] -= localNnzBegin;
    }

    // Device buffers
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double *d_val = nullptr, *d_vec = nullptr, *d_out = nullptr;
    index_t *d_cols = nullptr, *d_rowDelimiters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(localNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(localNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (static_cast<size_t>(localRows) + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, std::max<size_t>(numRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(localRows, 1) * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(d_val, localVal, static_cast<size_t>(localNnz) * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cols, localCols, static_cast<size_t>(localNnz) * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rowDelimiters, localRowDelimiters.data(),
                               (static_cast<size_t>(localRows) + 1) * sizeof(index_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vec, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, std::max<size_t>(localRows, 1) * sizeof(double), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Local slice of the output in pinned memory for fast D2H
    double* h_localOut = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_localOut, std::max<size_t>(localRows, 1) * sizeof(double)));

    const SpmvLauncher launch =
        selectLauncher(localRows ? static_cast<double>(localNnz) / localRows : 0.0);

    std::vector<int> recvCounts, recvDispls;
    if (root) {
        recvCounts.resize(nranks);
        recvDispls.resize(nranks);
        for (int r = 0; r < nranks; ++r) {
            recvCounts[r] = static_cast<int>(rowBegin[r + 1] - rowBegin[r]);
            recvDispls[r] = static_cast<int>(rowBegin[r]);
        }
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localRows > 0) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            launch(d_val, d_cols, d_rowDelimiters, d_vec, localRows, d_out, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_localOut, d_out, static_cast<size_t>(localRows) * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (iterations == 0 && localRows > 0) {
        std::fill(h_localOut, h_localOut + localRows, 0.0);
    }
    MPI_Gatherv(h_localOut, static_cast<int>(localRows), MPI_DOUBLE,
                root ? h_out.data() : nullptr, recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFreeHost(h_localOut));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (!root) return 0;

    printf("Computation time: %ld ms\n", duration.count());

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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int rc = run(argc, argv, rank, nranks);

    // Propagate the root's exit status (e.g. validation result) to all ranks
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    fflush(stdout);
    MPI_Finalize();
    return rc;
}
