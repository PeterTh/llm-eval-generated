#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
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
constexpr unsigned THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
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
// Function: partitionRows
//
// Purpose:
//   Splits the rows into contiguous blocks holding roughly equal numbers of
//   nonzeros. Block p consists of rows [bounds[p], bounds[p+1]).
//
// Arguments:
//   rowDelimiters: CSR row delimiters, size dim+1
//   dim:           number of rows/columns in the matrix
//   numParts:      number of blocks
//   bounds:        output, size numParts+1
//
// ****************************************************************************
void partitionRows(const index_t* rowDelimiters, const index_t dim, const int numParts,
                   std::vector<index_t>& bounds) {
    const index_t nnz = rowDelimiters[dim];
    bounds.assign(numParts + 1, dim);
    bounds[0] = 0;
    for (int p = 1; p < numParts; ++p) {
        const auto target = static_cast<index_t>(static_cast<uint64_t>(nnz) * p / numParts);
        bounds[p] = static_cast<index_t>(std::lower_bound(rowDelimiters, rowDelimiters + dim, target) - rowDelimiters);
    }
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   Computes sparse matrix-vector multiplication using CSR format on the host
//   (OpenMP over rows). Each row is accumulated sequentially, so the result
//   is independent of the thread count.
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
//   Computes sparse matrix-vector multiplication using CSR format on the GPU.
//   One thread handles one row and accumulates it in the same order as
//   spmvCpu, so the GPU result is bit-identical to the host result (the GPU is
//   built with --fmad=false to match the host rounding).
//
// Arguments: see spmvCpu (dim is the number of local rows)
//
// ****************************************************************************
__global__ void spmvCsrKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec, const index_t dim,
                              double* __restrict__ out) {
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) {
        return;
    }
    double t = 0.0;
    for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
        t += val[j] * vec[cols[j]];
    }
    out[i] = t;
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
//   rowOffset: global index of the first element (used in messages)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t rowOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + rowOffset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + rowOffset, ref, res, relError);
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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

    // Bind each rank to one GPU on its node (round-robin if ranks outnumber GPUs)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Share this node's CPU cores among its ranks for the OpenMP sections
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;
    if (nItems > static_cast<index_t>(INT_MAX)) {
        if (root) {
            fprintf(stderr, "Too many non-zero elements (%u) for MPI scatter counts\n", nItems);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        printf("Initializing data structures...\n");
    }

    // The matrix is generated on the root rank with exactly the same random
    // sequence as the serial benchmark, then distributed.
    std::vector<double> h_val;                          // Non-zero values (root only)
    std::vector<index_t> h_cols;                        // Column indices (root only)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    if (root) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }
    MPI_Bcast(h_rowDelimiters.data(), numRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each rank owns a contiguous block of rows with a balanced share of nonzeros
    std::vector<index_t> bounds;
    partitionRows(h_rowDelimiters.data(), numRows, size, bounds);

    std::vector<int> nnzCounts(size);
    std::vector<int> nnzDispls(size);
    std::vector<int> rowCounts(size);
    std::vector<int> rowDispls(size);
    for (int r = 0; r < size; ++r) {
        rowDispls[r] = static_cast<int>(bounds[r]);
        rowCounts[r] = static_cast<int>(bounds[r + 1] - bounds[r]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[bounds[r]]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[bounds[r + 1]]) - nnzDispls[r];
    }

    const index_t localRows = static_cast<index_t>(rowCounts[rank]);
    const index_t localNnz = static_cast<index_t>(nnzCounts[rank]);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz);
    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Row delimiters rebased to this rank's slice of the nonzeros
    std::vector<index_t> localRowDelimiters(localRows + 1);
    const index_t rowStart = bounds[rank];
    const index_t nnzStart = static_cast<index_t>(nnzDispls[rank]);
    #pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= localRows; ++i) {
        localRowDelimiters[i] = h_rowDelimiters[rowStart + i] - nnzStart;
    }

    // For validation, compute the local part of the reference solution
    std::vector<double> h_reference;
    if (validate) {
        if (root) {
            printf("Computing reference solution...\n");
        }
        h_reference.resize(localRows);
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Move the local problem onto this rank's GPU
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_val, localNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, localNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (localRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, localRows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_val, localVal.data(), localNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, localCols.data(), localNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, localRowDelimiters.data(), (localRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, localRows * sizeof(double)));

    // Perform SpMV computation
    if (root) {
        printf("Computing SpMV...\n");
    }
    const unsigned blocks = (localRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (localRows > 0) {
            spmvCsrKernel<<<blocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                         localRows, d_out);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSec = std::chrono::duration<double>(end - start).count();

    // Bring the result of the last iteration back to the host
    std::vector<double> h_out(localRows);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, localRows * sizeof(double), cudaMemcpyDeviceToHost));

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / elapsedSec / 1e9;
        const double avgTime = (elapsedSec * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> h_full(root ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    h_full.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            print_results(h_full, "OutputVector");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        int localValid = verifyResults(h_reference.data(), h_out.data(), localRows, rowStart) ? 1 : 0;
        int allValid = 0;
        MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (allValid) {
            if (root) {
                printf("Validation: PASSED\n");
            }
        } else {
            if (root) {
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));
    MPI_Finalize();
    return exitCode;
}
