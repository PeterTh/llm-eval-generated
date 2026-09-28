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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
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
//   (OpenMP-parallel; used for the validation reference solution)
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
// Function: spmvKernel
//
// Purpose:
//   CSR-vector SpMV kernel: TPR threads cooperate on each row, partial sums
//   are combined with warp shuffles. TPR is chosen from the average number
//   of nonzeros per row.
//
// ****************************************************************************
template <int TPR>
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t nRows,
                           double* __restrict__ out) {
    const index_t row = (blockIdx.x * blockDim.x + threadIdx.x) / TPR;
    const int lane = threadIdx.x % TPR;
    if (row >= nRows) {
        return;
    }

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += TPR) {
        sum += val[j] * __ldg(&vec[cols[j]]);
    }
#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

template <int TPR>
void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, const index_t nRows, double* out, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    const unsigned int nThreads = static_cast<unsigned int>(nRows) * TPR;
    const unsigned int nBlocks = (nThreads + BLOCK - 1) / BLOCK;
    spmvKernel<TPR><<<nBlocks, BLOCK, 0, stream>>>(val, cols, rowDelimiters, vec, nRows, out);
}

using spmvLauncher = void (*)(const double*, const index_t*, const index_t*, const double*,
                              const index_t, double*, cudaStream_t);

// Pick the cooperating-thread count per row based on the mean row length
spmvLauncher selectLauncher(const index_t nnz, const index_t nRows) {
    const double avg = nRows > 0 ? static_cast<double>(nnz) / nRows : 0.0;
    if (avg <= 4.0) return launchSpmv<2>;
    if (avg <= 8.0) return launchSpmv<4>;
    if (avg <= 16.0) return launchSpmv<8>;
    if (avg <= 32.0) return launchSpmv<16>;
    return launchSpmv<32>;
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
    bool valid = true;
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < size; ++i) {
        if (!valid) continue;
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                valid = false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relError);
                valid = false;
            }
        }
    }
    return valid;
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

    // Parse command line arguments (identical argv on every rank)
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
            if (rank == 0) printUsage(argv[0]);
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

    // Bind each rank to a GPU on its node (round-robin over local ranks)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

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
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               nRanks, deviceCount, omp_get_max_threads());
    }

    // Rank 0 generates the full problem (serial, to preserve the original
    // rand() sequence); everyone gets the dense vector and row delimiters.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
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

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0,
              MPI_COMM_WORLD);

    // Partition contiguous row blocks so each rank gets ~equal nonzeros.
    // Every rank computes the same partition from the row delimiters.
    std::vector<index_t> rowStart(nRanks + 1);
    rowStart[0] = 0;
    rowStart[nRanks] = numRows;
    for (int r = 1; r < nRanks; ++r) {
        const index_t target = static_cast<index_t>(
            (static_cast<uint64_t>(nItems) * r) / static_cast<uint64_t>(nRanks));
        const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
        index_t row = static_cast<index_t>(it - h_rowDelimiters.begin());
        row = std::min(row, numRows);
        rowStart[r] = std::max(row, rowStart[r - 1]);
    }

    const index_t myRowBegin = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRowBegin;
    const index_t myNnzBegin = h_rowDelimiters[myRowBegin];
    const index_t myNnz = h_rowDelimiters[rowStart[rank + 1]] - myNnzBegin;

    std::vector<int> nnzCounts(nRanks), nnzDispls(nRanks);
    std::vector<int> rowCounts(nRanks), rowDispls(nRanks);
    for (int r = 0; r < nRanks; ++r) {
        rowCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
        rowDispls[r] = static_cast<int>(rowStart[r]);
        nnzCounts[r] = static_cast<int>(h_rowDelimiters[rowStart[r + 1]] -
                                        h_rowDelimiters[rowStart[r]]);
        nnzDispls[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
    }

    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 l_val.data(), static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 l_cols.data(), static_cast<int>(myNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Rebase this rank's row delimiters to its local nonzero range
    std::vector<index_t> l_rowDelimiters(myRows + 1);
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i <= myRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[myRowBegin + i] - myNnzBegin;
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Upload this rank's matrix block and the full dense vector to its GPU
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

    const spmvLauncher launch = selectLauncher(myNnz, myRows);
    std::vector<double> l_out(std::max<index_t>(myRows, 1));

    // Warm-up launch so timing excludes one-time CUDA initialization cost
    if (myRows > 0) {
        launch(d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (myRows > 0) {
            launch(d_val, d_cols, d_rowDelimiters, d_vec, myRows, d_out, nullptr);
        }
    }
    if (myRows > 0) {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, myRows * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Assemble the distributed result on rank 0
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                h_out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDuration = duration.count();
    long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", globalDuration);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (globalDuration / 1000.0) / 1e9;
        const double avgTime = globalDuration / static_cast<double>(iterations);

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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
