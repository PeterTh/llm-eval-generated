#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

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
// Function: sendChunked / recvChunked
//
// Purpose:
//   Point-to-point transfer of arbitrarily large arrays (beyond INT_MAX
//   elements) by splitting into chunks.
//
// ****************************************************************************
constexpr size_t MPI_CHUNK = size_t(1) << 26;

template <typename T>
void sendChunked(const T* buf, size_t count, int dest, int tag) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const size_t c = std::min(MPI_CHUNK, count - off);
        MPI_Send(buf + off, static_cast<int>(c * sizeof(T)), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    }
}

template <typename T>
void recvChunked(T* buf, size_t count, int src, int tag) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const size_t c = std::min(MPI_CHUNK, count - off);
        MPI_Recv(buf + off, static_cast<int>(c * sizeof(T)), MPI_BYTE, src, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
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

    // Global data (only populated on the root rank, which generates the
    // matrix with the same sequential RNG stream as the serial code)
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);   // Dense vector (replicated on all ranks)
    std::vector<double> h_out;            // Gathered output vector (root only)
    std::vector<double> h_reference;

    // Row partition: rank r owns rows [rowStart[r], rowStart[r+1])
    std::vector<index_t> rowStart(nprocs + 1, 0);

    if (root) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Balance work: contiguous row blocks with ~equal (nnz + rows) cost
        const double totalCost = static_cast<double>(nItems) + numRows;
        for (int r = 1; r < nprocs; ++r) {
            const double target = totalCost * r / nprocs;
            // first row boundary i where cost of rows [0,i) >= target
            index_t lo = rowStart[r - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                const double cost = static_cast<double>(h_rowDelimiters[mid]) + mid;
                if (cost < target) lo = mid + 1; else hi = mid;
            }
            rowStart[r] = lo;
        }
        rowStart[nprocs] = numRows;

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Distribute partition and dense vector
    MPI_Bcast(rowStart.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    for (size_t off = 0; off < numRows; off += MPI_CHUNK) {
        const size_t c = std::min(MPI_CHUNK, static_cast<size_t>(numRows) - off);
        MPI_Bcast(h_vec.data() + off, static_cast<int>(c), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    const index_t myRow0 = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRow0;

    // Local CSR block (row delimiters rebased to 0)
    std::vector<index_t> l_rowDelimiters(myRows + 1);
    std::vector<double> l_val;
    std::vector<index_t> l_cols;
    std::vector<double> l_out(myRows);

    if (root) {
        // Send row blocks to the other ranks
        for (int r = 1; r < nprocs; ++r) {
            const index_t r0 = rowStart[r], r1 = rowStart[r + 1];
            const index_t nzBeg = h_rowDelimiters[r0];
            const size_t nnz = static_cast<size_t>(h_rowDelimiters[r1]) - nzBeg;
            sendChunked(h_rowDelimiters.data() + r0, static_cast<size_t>(r1 - r0) + 1, r, 1);
            sendChunked(h_cols.data() + nzBeg, nnz, r, 2);
            sendChunked(h_val.data() + nzBeg, nnz, r, 3);
        }
        // Own block
        const index_t nzBeg = h_rowDelimiters[myRow0];
        const index_t nzEnd = h_rowDelimiters[myRow0 + myRows];
        for (index_t i = 0; i <= myRows; ++i) {
            l_rowDelimiters[i] = h_rowDelimiters[myRow0 + i] - nzBeg;
        }
        l_cols.assign(h_cols.begin() + nzBeg, h_cols.begin() + nzEnd);
        l_val.assign(h_val.begin() + nzBeg, h_val.begin() + nzEnd);
        // Global matrix no longer needed
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    } else {
        recvChunked(l_rowDelimiters.data(), static_cast<size_t>(myRows) + 1, 0, 1);
        const index_t nzBeg = l_rowDelimiters[0];
        for (index_t i = 0; i <= myRows; ++i) l_rowDelimiters[i] -= nzBeg;
        const size_t nnz = l_rowDelimiters[myRows];
        l_cols.resize(nnz);
        l_val.resize(nnz);
        recvChunked(l_cols.data(), nnz, 0, 2);
        recvChunked(l_val.data(), nnz, 0, 3);
    }

    // Gather layout for the output vector
    std::vector<int> recvCounts, displs;
    if (root) {
        recvCounts.resize(nprocs);
        displs.resize(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            recvCounts[r] = static_cast<int>(rowStart[r + 1] - rowStart[r]);
            displs[r] = static_cast<int>(rowStart[r]);
        }
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(),
                h_vec.data(), myRows, l_out.data());
    }
    // Assemble the distributed result vector on the root rank
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                h_out.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
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
        fflush(stdout);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
