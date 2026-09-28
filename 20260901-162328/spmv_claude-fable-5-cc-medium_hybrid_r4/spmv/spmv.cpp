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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
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
//   Computes sparse matrix-vector multiplication using CSR format
//   (OpenMP-parallel; used for the reference/validation solution)
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
#pragma omp parallel for schedule(guided)
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
// Function: spmvKernel
//
// Purpose:
//   CSR-vector SpMV CUDA kernel: one warp per matrix row, warp-level
//   reduction of partial sums.
//
// ****************************************************************************
constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;

__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t nRows,
                           double* __restrict__ out) {
    const index_t warpsPerBlock = blockDim.x / WARP_SIZE;
    const index_t row = blockIdx.x * warpsPerBlock + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    if (row >= nRows) {
        return;
    }

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
        sum += val[j] * __ldg(&vec[cols[j]]);
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        out[row] = sum;
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
    int nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on all ranks)
    int parseStatus = 0;  // 0 = ok, 1 = show help, 2 = error
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
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nRanks, omp_get_max_threads());
    }

    // Select one GPU per rank based on the node-local rank
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Allocate and initialize data structures on rank 0 (sequential rand()
    // preserves the exact input data of the original benchmark)
    std::vector<double> h_val;                          // Non-zero values (rank 0)
    std::vector<index_t> h_cols;                        // Column indices (rank 0)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);                 // Dense vector (all ranks)
    std::vector<double> h_out;                          // Output vector (rank 0)

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Broadcast the dense vector and row delimiters so every rank can
    // compute its nnz-balanced row partition locally
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Partition rows across ranks, balancing the number of nonzeros
    std::vector<index_t> rowStarts(nRanks + 1);
    rowStarts[0] = 0;
    for (int r = 1; r < nRanks; ++r) {
        const index_t target = static_cast<index_t>(
            (static_cast<uint64_t>(nItems) * r) / nRanks);
        const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
        rowStarts[r] = std::max(rowStarts[r - 1],
                                static_cast<index_t>(it - h_rowDelimiters.begin()));
        rowStarts[r] = std::min(rowStarts[r], numRows);
    }
    rowStarts[nRanks] = numRows;

    const index_t myRowStart = rowStarts[rank];
    const index_t myRowEnd = rowStarts[rank + 1];
    const index_t myRows = myRowEnd - myRowStart;
    const index_t myNnzStart = h_rowDelimiters[myRowStart];
    const index_t myNnz = h_rowDelimiters[myRowEnd] - myNnzStart;

    // Scatter the CSR value/column slices
    std::vector<int> nnzCounts(nRanks), nnzDispls(nRanks);
    std::vector<int> rowCounts(nRanks), rowDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStarts[r]]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowStarts[r + 1]] - h_rowDelimiters[rowStarts[r]]);
        rowDispls[r] = static_cast<int>(rowStarts[r]);
        rowCounts[r] = static_cast<int>(rowStarts[r + 1] - rowStarts[r]);
    }

    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_DOUBLE, l_val.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, nnzCounts.data(), nnzDispls.data(),
                 MPI_UINT32_T, l_cols.data(), static_cast<int>(myNnz), MPI_UINT32_T, 0,
                 MPI_COMM_WORLD);

    // Rebase the local row delimiters to the local nnz slice
    std::vector<index_t> l_rowDelimiters(myRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= myRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[myRowStart + i] - myNnzStart;
    }

    // For validation, compute reference solution (OpenMP-parallel on rank 0)
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Upload the local matrix slice and full vector to the GPU
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, std::max<size_t>(myNnz, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, std::max<size_t>(myNnz, 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (myRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, std::max<size_t>(myRows, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(),
                          (myRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    const index_t warpsPerBlock = BLOCK_SIZE / WARP_SIZE;
    const index_t numBlocks = (myRows + warpsPerBlock - 1) / warpsPerBlock;

    // Warm up: make sure the kernel is loaded and the GPU is ready
    if (myRows > 0) {
        spmvKernel<<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (myRows > 0) {
            spmvKernel<<<numBlocks, BLOCK_SIZE>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                  myRows, d_out);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    const long localDuration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Copy the local result back and gather the full output vector on rank 0
    std::vector<double> l_out(std::max<index_t>(myRows, 1));
    if (myRows > 0) {
        CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, myRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rowCounts.data(), rowDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDuration / 1000.0) / 1e9;
        const double avgTime = maxDuration / static_cast<double>(iterations);

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

    // All ranks return the same exit code
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
