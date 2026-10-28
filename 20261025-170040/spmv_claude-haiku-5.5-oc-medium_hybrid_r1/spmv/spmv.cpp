#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
// Average non-zeros per row at which a warp-per-row kernel beats a thread-per-row kernel
constexpr double VECTOR_KERNEL_MIN_AVG_NNZ = 8.0;

// Message tags for MPI point-to-point communication
constexpr int TAG_VAL = 1;
constexpr int TAG_COLS = 2;
constexpr int TAG_OUT = 3;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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
//   Computes sparse matrix-vector multiplication using CSR format.
//   Rows are processed in parallel with OpenMP; each row's accumulation
//   order is unchanged from the serial algorithm.
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
// Kernel: spmvCsrScalarKernel
//
// Purpose:
//   CSR SpMV with one thread per row. Preferred when rows are short.
//
// ****************************************************************************
__global__ void spmvCsrScalarKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                    const index_t* __restrict__ rowDelimiters,
                                    const double* __restrict__ vec, const index_t numRows,
                                    double* __restrict__ out) {
    const index_t row = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= numRows) {
        return;
    }
    double t = 0.0;
    const index_t end = rowDelimiters[row + 1];
    for (index_t j = rowDelimiters[row]; j < end; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[row] = t;
}

// ****************************************************************************
// Kernel: spmvCsrVectorKernel
//
// Purpose:
//   CSR SpMV with one warp per row; lanes stride over the row's non-zeros
//   and reduce with warp shuffles. Preferred when rows are long.
//
// ****************************************************************************
__global__ void spmvCsrVectorKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                                    const index_t* __restrict__ rowDelimiters,
                                    const double* __restrict__ vec, const index_t numRows,
                                    double* __restrict__ out) {
    const index_t row =
        static_cast<index_t>(blockIdx.x) * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    // Whole warp exits together (row is warp-uniform), so shuffles below are safe
    if (row >= numRows) {
        return;
    }
    double t = 0.0;
    const index_t end = rowDelimiters[row + 1];
    for (index_t j = rowDelimiters[row] + lane; j < end; j += WARP_SIZE) {
        t += val[j] * vec[cols[j]];
    }
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffffu, t, offset);
    }
    if (lane == 0) {
        out[row] = t;
    }
}

// ****************************************************************************
// Function: verifyResults
//
// Purpose:
//   Verifies correctness of results by comparing to reference solution
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            // Check relative error
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

// ****************************************************************************
// Function: partitionRowsByNnz
//
// Purpose:
//   Splits rows into contiguous blocks, one per process, with approximately
//   equal non-zero counts.
//
// Returns:
//   bounds of size parts+1; process p owns rows [bounds[p], bounds[p+1])
//
// ****************************************************************************
std::vector<index_t> partitionRowsByNnz(const std::vector<index_t>& rowDelimiters, const index_t dim,
                                        const index_t nnz, const int parts) {
    std::vector<index_t> bounds(parts + 1);
    bounds[0] = 0;
    bounds[parts] = dim;
    for (int p = 1; p < parts; ++p) {
        const auto target = static_cast<index_t>(static_cast<uint64_t>(nnz) * p / parts);
        const auto it = std::lower_bound(rowDelimiters.begin(), rowDelimiters.begin() + dim + 1, target);
        const auto b = static_cast<index_t>(it - rowDelimiters.begin());
        bounds[p] = std::max(b, bounds[p - 1]);
    }
    return bounds;
}

// Send/receive large buffers in chunks that fit the MPI int count argument
template <typename T>
void sendChunked(const T* buf, const size_t count, MPI_Datatype type, const int dest, const int tag) {
    constexpr size_t kMaxChunk = static_cast<size_t>(1) << 30;
    for (size_t off = 0; off < count; off += kMaxChunk) {
        const int c = static_cast<int>(std::min(kMaxChunk, count - off));
        MPI_Send(const_cast<T*>(buf + off), c, type, dest, tag, MPI_COMM_WORLD);
    }
}

template <typename T>
void recvChunked(T* buf, const size_t count, MPI_Datatype type, const int src, const int tag) {
    constexpr size_t kMaxChunk = static_cast<size_t>(1) << 30;
    for (size_t off = 0; off < count; off += kMaxChunk) {
        const int c = static_cast<int>(std::min(kMaxChunk, count - off));
        MPI_Recv(buf + off, c, type, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

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

    // Bind each rank to a GPU on its node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Root process generates the full problem exactly as the serial benchmark does
    std::vector<double> h_val;                          // Non-zero values (root only)
    std::vector<index_t> h_cols;                        // Column indices (root only)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)
    std::vector<double> h_out(root ? numRows : 0);      // Output vector (root only)
    std::vector<double> h_reference;

    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Share the row structure and dense vector with all processes
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Nnz-balanced row partition; process p owns rows [rowBegin[p], rowBegin[p+1])
    const std::vector<index_t> rowBegin = partitionRowsByNnz(h_rowDelimiters, numRows, nItems, numProcs);
    const index_t rowLo = rowBegin[rank];
    const index_t localRows = rowBegin[rank + 1] - rowLo;
    const index_t nnzBase = h_rowDelimiters[rowLo];
    const index_t localNnz = h_rowDelimiters[rowBegin[rank + 1]] - nnzBase;

    // Distribute the local non-zero slices
    std::vector<double> localVal;
    std::vector<index_t> localCols;
    const double* valPtr = nullptr;
    const index_t* colsPtr = nullptr;
    if (root) {
        for (int p = 1; p < numProcs; ++p) {
            const index_t base = h_rowDelimiters[rowBegin[p]];
            const size_t count = h_rowDelimiters[rowBegin[p + 1]] - base;
            sendChunked(h_val.data() + base, count, MPI_DOUBLE, p, TAG_VAL);
            sendChunked(h_cols.data() + base, count, MPI_UINT32_T, p, TAG_COLS);
        }
        valPtr = h_val.data() + nnzBase;
        colsPtr = h_cols.data() + nnzBase;
    } else {
        localVal.resize(localNnz);
        localCols.resize(localNnz);
        recvChunked(localVal.data(), localNnz, MPI_DOUBLE, 0, TAG_VAL);
        recvChunked(localCols.data(), localNnz, MPI_UINT32_T, 0, TAG_COLS);
        valPtr = localVal.data();
        colsPtr = localCols.data();
    }

    // Local row delimiters rebased to the local non-zero array
    std::vector<index_t> localRowPtr(localRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        localRowPtr[i] = h_rowDelimiters[rowLo + i] - nnzBase;
    }

    // Upload the problem to the device once; it is reused by every iteration
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowPtr = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * std::max<size_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * std::max<size_t>(localNnz, 1)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, sizeof(index_t) * (localRows + 1)));
    CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * std::max<size_t>(numRows, 1)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * std::max<size_t>(localRows, 1)));
    CUDA_CHECK(cudaMemcpy(d_val, valPtr, sizeof(double) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, colsPtr, sizeof(index_t) * localNnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, localRowPtr.data(), sizeof(index_t) * (localRows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows, cudaMemcpyHostToDevice));

    const bool useVectorKernel =
        (static_cast<double>(nItems) / static_cast<double>(numRows)) >= VECTOR_KERNEL_MIN_AVG_NNZ;
    const unsigned int scalarBlocks = static_cast<unsigned int>((localRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const unsigned int vectorBlocks = static_cast<unsigned int>((localRows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    // Launches one SpMV pass over the local rows (no-op for ranks without rows)
    auto launchSpmv = [&]() {
        if (localRows == 0) {
            return;
        }
        if (useVectorKernel) {
            spmvCsrVectorKernel<<<vectorBlocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowPtr, d_vec,
                                                                     localRows, d_out);
        } else {
            spmvCsrScalarKernel<<<scalarBlocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowPtr, d_vec,
                                                                     localRows, d_out);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
    }
    // Untimed warm-up pass to absorb one-time CUDA module/context initialization
    launchSpmv();
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launchSpmv();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Slowest process determines the reported time
    const double localSec = std::chrono::duration<double>(end - start).count();
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(maxSec));

    // Retrieve local output and gather the full result on the root
    std::vector<double> localOut(localRows);
    CUDA_CHECK(cudaMemcpy(localOut.data(), d_out, sizeof(double) * localRows, cudaMemcpyDeviceToHost));
    if (root) {
        std::copy(localOut.begin(), localOut.end(), h_out.begin());
        for (int p = 1; p < numProcs; ++p) {
            recvChunked(h_out.data() + rowBegin[p], rowBegin[p + 1] - rowBegin[p], MPI_DOUBLE, p, TAG_OUT);
        }
    } else {
        sendChunked(localOut.data(), localRows, MPI_DOUBLE, 0, TAG_OUT);
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int validInt = 1;
    if (root) {
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
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            validInt = valid ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
