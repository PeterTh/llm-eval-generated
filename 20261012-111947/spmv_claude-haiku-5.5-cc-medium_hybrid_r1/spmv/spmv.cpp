#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

// Abort the whole job on any CUDA failure
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
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
//   Rows are distributed across OpenMP threads.
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
    #pragma omp parallel for
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
// Function: spmvCsrVector
//
// Purpose:
//   GPU kernel computing out = A*vec for a block of rows in CSR format.
//   One warp is assigned to each row. Lanes stride over the row's nonzeros
//   and the partial sums are combined with a warp shuffle reduction.
//
// Arguments:
//   val: array holding the non-zero values of the block
//   cols: array of column indices for each element
//   rowDelimiters: array of size rows+1 holding indices to rows of the block
//   vec: full dense vector (indexed by global column)
//   rows: number of rows in the block
//   out: output - result for the rows of the block
//
// ****************************************************************************
__global__ void spmvCsrVector(const double* __restrict__ val, const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                              const index_t rows, double* __restrict__ out) {
    const index_t row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const index_t end = rowDelimiters[row + 1];

    double t = 0.0;
    for (index_t j = rowDelimiters[row] + lane; j < end; j += WARP_SIZE) {
        t += val[j] * vec[cols[j]];
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
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
//   rowOffset: global index of the first element (for reporting)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t rowOffset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const index_t idx = rowOffset + i;
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", idx, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       idx, ref, res, relError);
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

template <typename T>
T* deviceAlloc(const size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, std::max<size_t>(count, 1) * sizeof(T)));
    return ptr;
}

int runBenchmark(int argc, char** argv, const int rank, const int numRanks) {
    const bool isRoot = rank == 0;

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
            if (isRoot) printUsage(argv[0]);
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    // Node-local rank selects one GPU and shares the node's CPU cores among its ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    int nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % numDevices));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / nodeSize));

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (isRoot) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates the problem exactly as the serial code does
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated on all ranks)
    std::vector<double> h_val;                          // Non-zero values (rank 0 only)
    std::vector<index_t> h_cols;                        // Column indices (rank 0 only)
    std::vector<index_t> h_rowDelimiters;               // Row delimiters (rank 0 only)
    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Partition rows into contiguous blocks balanced by nonzero count
    std::vector<index_t> rowBounds(numRanks + 1);
    std::vector<index_t> nnzBounds(numRanks + 1);
    if (isRoot) {
        rowBounds[0] = 0;
        rowBounds[numRanks] = numRows;
        for (int r = 1; r < numRanks; ++r) {
            const uint64_t target = static_cast<uint64_t>(nItems) * r / numRanks;
            const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.end(), target);
            rowBounds[r] = std::max(static_cast<index_t>(it - h_rowDelimiters.begin()), rowBounds[r - 1]);
        }
        for (int r = 0; r <= numRanks; ++r) {
            nnzBounds[r] = h_rowDelimiters[rowBounds[r]];
        }
    }
    MPI_Bcast(rowBounds.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzBounds.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    for (int r = 0; r < numRanks; ++r) {
        if (nnzBounds[r + 1] - nnzBounds[r] > static_cast<index_t>(INT_MAX)) {
            if (isRoot) printf("Error: per-rank nonzero count exceeds MPI limit; use more ranks\n");
            return 1;
        }
    }

    const index_t rowsLocal = rowBounds[rank + 1] - rowBounds[rank];
    const index_t nnzLocal = nnzBounds[rank + 1] - nnzBounds[rank];

    // Per-rank counts and displacements for scatter/gather collectives
    std::vector<int> rowCounts(numRanks);
    std::vector<int> rowDispls(numRanks);
    std::vector<int> nnzCounts(numRanks);
    std::vector<int> nnzDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        rowCounts[r] = static_cast<int>(rowBounds[r + 1] - rowBounds[r]);
        rowDispls[r] = static_cast<int>(rowBounds[r]);
        nnzCounts[r] = static_cast<int>(nnzBounds[r + 1] - nnzBounds[r]);
        nnzDispls[r] = static_cast<int>(nnzBounds[r]);
    }

    // Distribute the matrix block and the vector
    std::vector<double> l_val(nnzLocal);
    std::vector<index_t> l_cols(nnzLocal);
    std::vector<index_t> l_rowDelimiters(rowsLocal + 1);
    MPI_Scatterv(h_rowDelimiters.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                 l_rowDelimiters.data(), static_cast<int>(rowsLocal), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 l_cols.data(), static_cast<int>(nnzLocal), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 l_val.data(), static_cast<int>(nnzLocal), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rebase local row delimiters to this block's nonzero range
    for (index_t i = 0; i < rowsLocal; ++i) {
        l_rowDelimiters[i] -= nnzBounds[rank];
    }
    l_rowDelimiters[rowsLocal] = nnzLocal;

    // Upload block to this rank's GPU
    double* d_val = deviceAlloc<double>(nnzLocal);
    index_t* d_cols = deviceAlloc<index_t>(nnzLocal);
    index_t* d_rowDelimiters = deviceAlloc<index_t>(rowsLocal + 1);
    double* d_vec = deviceAlloc<double>(numRows);
    double* d_out = deviceAlloc<double>(rowsLocal);
    CUDA_CHECK(cudaMemcpy(d_val, l_val.data(), nnzLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, l_cols.data(), nnzLocal * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, l_rowDelimiters.data(), (rowsLocal + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), numRows * sizeof(double), cudaMemcpyHostToDevice));

    // For validation, each rank computes the reference for its own rows on the CPU
    std::vector<double> l_reference;
    if (validate) {
        if (isRoot) printf("Computing reference solution...\n");
        l_reference.resize(rowsLocal);
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(),
                h_vec.data(), rowsLocal, l_reference.data());
    }

    const unsigned int numBlocks = static_cast<unsigned int>((rowsLocal + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    auto launch = [&]() {
        if (numBlocks > 0) {
            spmvCsrVector<<<numBlocks, THREADS_PER_BLOCK>>>(d_val, d_cols, d_rowDelimiters, d_vec,
                                                             rowsLocal, d_out);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    // Perform SpMV computation
    if (isRoot) printf("Computing SpMV...\n");

    // Untimed warm-up launch to absorb one-time GPU module loading
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();

    // Wall time of the whole job is the slowest rank
    double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double, std::milli>(maxMs));

    // Gather the output vector on rank 0
    std::vector<double> l_out(rowsLocal);
    CUDA_CHECK(cudaMemcpy(l_out.data(), d_out, rowsLocal * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<double> h_out;
    if (isRoot) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(l_out.data(), static_cast<int>(rowsLocal), MPI_DOUBLE,
                h_out.data(), rowCounts.data(), rowDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults && isRoot) {
        print_results(h_out, "OutputVector");
    }

    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    // Validation
    if (validate) {
        if (isRoot) printf("Validating result...\n");
        const int localValid = verifyResults(l_reference.data(), l_out.data(), rowsLocal, rowBounds[rank]) ? 1 : 0;
        int valid = 0;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (isRoot) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int rc = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return rc;
}
