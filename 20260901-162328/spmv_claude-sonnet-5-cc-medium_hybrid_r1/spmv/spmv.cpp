#include <mpi.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
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
    // Each row's dot product is fully independent, so parallelizing across
    // rows does not change the computed values.
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
// Function: spmvKernel
//
// Purpose:
//   CUDA kernel that computes sparse matrix-vector multiplication using CSR
//   format. One warp is assigned per matrix row so that rows with differing
//   numbers of non-zeros are handled with balanced memory throughput.
//
// ****************************************************************************
__global__ void spmvKernel(const double* __restrict__ val, const index_t* __restrict__ cols,
                            const index_t* __restrict__ rowDelimiters,
                            const double* __restrict__ vec, const index_t numRows,
                            double* __restrict__ out) {
    const int warpsPerBlock = blockDim.x / 32;
    const int warpIdInBlock = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const index_t row = static_cast<index_t>(blockIdx.x) * warpsPerBlock + warpIdInBlock;

    if (row >= numRows) return;

    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd = rowDelimiters[row + 1];

    double sum = 0.0;
    for (index_t j = rowStart + lane; j < rowEnd; j += 32) {
        sum += val[j] * vec[cols[j]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
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

// ****************************************************************************
// Function: computeNnzBalancedPartition
//
// Purpose:
//   Splits the [0, dim) rows across `numRanks` contiguous blocks such that
//   each block holds roughly the same number of non-zero elements. Every
//   rank computes the same result independently from the (broadcast) global
//   rowDelimiters array, so no extra communication is required.
//
// ****************************************************************************
std::vector<index_t> computeNnzBalancedPartition(const std::vector<index_t>& rowDelimiters,
                                                  const index_t dim, const int numRanks) {
    std::vector<index_t> rowStart(numRanks + 1);
    const index_t nItems = rowDelimiters[dim];
    rowStart[0] = 0;
    for (int r = 1; r < numRanks; ++r) {
        const index_t targetNnz =
            static_cast<index_t>((static_cast<uint64_t>(nItems) * r) / static_cast<uint64_t>(numRanks));
        index_t lo = rowStart[r - 1];
        index_t hi = dim;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (rowDelimiters[mid] < targetNnz) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        rowStart[r] = lo;
    }
    rowStart[numRanks] = dim;
    return rowStart;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Pick one GPU per rank, using the node-local rank so that ranks sharing a
    // node spread across that node's GPUs instead of colliding on device 0.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA-capable devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank: mpirun replicates argv)
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA\n", numRanks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Only rank 0 generates the full matrix, bit-for-bit identical to the
    // original sequential algorithm (the RNG sequence must not be split
    // across ranks). The data is then distributed via MPI.
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<double> h_vec(numRows);
    std::vector<index_t> h_rowDelimiters(numRows + 1);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Every rank derives the same nnz-balanced row partition from the
    // (now shared) rowDelimiters array, avoiding an extra broadcast.
    const std::vector<index_t> rowStart = computeNnzBalancedPartition(h_rowDelimiters, numRows, numRanks);

    const index_t myRowStart = rowStart[rank];
    const index_t myRowEnd = rowStart[rank + 1];
    const index_t myNumRows = myRowEnd - myRowStart;
    const index_t myNnzStart = h_rowDelimiters[myRowStart];
    const index_t myNnzEnd = h_rowDelimiters[myRowEnd];
    const index_t myNnz = myNnzEnd - myNnzStart;

    std::vector<index_t> h_localRowDelims(myNumRows + 1);
    #pragma omp parallel for
    for (index_t i = 0; i <= myNumRows; ++i) {
        h_localRowDelims[i] = h_rowDelimiters[myRowStart + i] - myNnzStart;
    }

    std::vector<int> sendCounts, sendDispls;
    if (rank == 0) {
        sendCounts.resize(numRanks);
        sendDispls.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            sendCounts[r] = static_cast<int>(h_rowDelimiters[rowStart[r + 1]] - h_rowDelimiters[rowStart[r]]);
            sendDispls[r] = static_cast<int>(h_rowDelimiters[rowStart[r]]);
        }
    }

    std::vector<double> h_localVal(myNnz);
    std::vector<index_t> h_localCols(myNnz);
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr, rank == 0 ? sendCounts.data() : nullptr,
                 rank == 0 ? sendDispls.data() : nullptr, MPI_DOUBLE, h_localVal.data(),
                 static_cast<int>(myNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr, rank == 0 ? sendCounts.data() : nullptr,
                 rank == 0 ? sendDispls.data() : nullptr, MPI_UINT32_T, h_localCols.data(),
                 static_cast<int>(myNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // For validation, compute reference solution on the fully assembled
    // matrix (rank 0 only), parallelized across CPU cores with OpenMP.
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate device buffers for this rank's shard and upload once; the
    // input data does not change across iterations.
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelims = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val, myNnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols, myNnz * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelims, (myNumRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec, numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out, myNumRows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_val, h_localVal.data(), myNnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_localCols.data(), myNnz * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelims, h_localRowDelims.data(), (myNumRows + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int warpsPerBlock = 8;
    constexpr int threadsPerBlock = warpsPerBlock * 32;
    const int blocks = myNumRows == 0 ? 0
                                       : static_cast<int>((myNumRows + warpsPerBlock - 1) / warpsPerBlock);

    if (rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (blocks > 0) {
            spmvKernel<<<blocks, threadsPerBlock>>>(d_val, d_cols, d_rowDelims, d_vec, myNumRows, d_out);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> h_localOut(myNumRows);
    CUDA_CHECK(cudaMemcpy(h_localOut.data(), d_out, myNumRows * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> h_out;
    std::vector<int> recvCounts, recvDispls;
    if (rank == 0) {
        h_out.resize(numRows);
        recvCounts.resize(numRanks);
        recvDispls.resize(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
            recvDispls[r] = static_cast<int>(rowStart[r]);
        }
    }
    MPI_Gatherv(h_localOut.data(), static_cast<int>(myNumRows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr, rank == 0 ? recvCounts.data() : nullptr,
                rank == 0 ? recvDispls.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelims));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    int exitCode = 0;
    if (rank == 0) {
        const long durationMs = static_cast<long>(maxMs);
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxMs / 1000.0) / 1e9;
        const double avgTime = maxMs / static_cast<double>(iterations);

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
                exitCode = 0;
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
