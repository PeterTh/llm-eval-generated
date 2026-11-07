#include <mpi.h>

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
//   Operates on a (possibly local) block of rows; column indices are
//   global indices into the fully replicated dense vector.
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   cols: array of column indices for each element
//   rowDelimiters: array of size dim+1 holding indices to rows;
//                  last element is the index one past the last element
//   vec: dense vector to be used for multiplication
//   dim: number of rows in this block
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < rowEnd; ++j) {
            t += val[j] * vec[cols[j]];
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
// MPI helpers: send/recv arbitrary-length buffers in INT_MAX-sized pieces so
// that element counts never overflow the int count argument.
// ****************************************************************************
template <typename T>
void sendLarge(const T* buf, size_t count, MPI_Datatype type, int dest, int tag) {
    size_t off = 0;
    do {
        const size_t chunk = std::min<size_t>(count - off, static_cast<size_t>(INT_MAX));
        MPI_Send(buf + off, static_cast<int>(chunk), type, dest, tag, MPI_COMM_WORLD);
        off += chunk;
    } while (off < count);
}

template <typename T>
void recvLarge(T* buf, size_t count, MPI_Datatype type, int src, int tag) {
    size_t off = 0;
    do {
        const size_t chunk = std::min<size_t>(count - off, static_cast<size_t>(INT_MAX));
        MPI_Recv(buf + off, static_cast<int>(chunk), type, src, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        off += chunk;
    } while (off < count);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

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
        printf("MPI processes: %d\n", nprocs);
    }

    // ------------------------------------------------------------------------
    // Data generation on the root rank (identical random stream to the
    // original sequential program), then distribution.
    // ------------------------------------------------------------------------
    std::vector<double> h_val;                          // Non-zero values (root only)
    std::vector<index_t> h_cols;                        // Column indices (root only)
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters (all ranks)
    std::vector<double> h_vec(numRows);                 // Dense vector (replicated)
    std::vector<double> h_out;                          // Output vector (root only)

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Row delimiters and the dense vector are small (O(numRows)); replicate them.
    MPI_Bcast(h_rowDelimiters.data(), static_cast<int>(numRows + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition rows across ranks so each rank gets roughly nItems/nprocs
    // non-zeros (balances work rather than row count). Every rank computes
    // the same partition from the replicated row delimiters.
    std::vector<index_t> rowStart(nprocs + 1);
    {
        rowStart[0] = 0;
        index_t r = 0;
        for (int p = 1; p < nprocs; ++p) {
            const double target = static_cast<double>(nItems) * p / nprocs;
            while (r < numRows && static_cast<double>(h_rowDelimiters[r]) < target) ++r;
            rowStart[p] = r;
        }
        rowStart[nprocs] = numRows;
    }

    const index_t myRow0 = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRow0;
    const index_t myNnz0 = h_rowDelimiters[myRow0];
    const index_t myNnz = h_rowDelimiters[myRow0 + myRows] - myNnz0;

    // Local CSR block (row delimiters rebased to start at 0)
    std::vector<index_t> l_rowDelimiters(myRows + 1);
    for (index_t i = 0; i <= myRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[myRow0 + i] - myNnz0;
    }
    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    std::vector<double> l_out(myRows);

    if (isRoot) {
        // Own block: copy locally
        std::memcpy(l_val.data(), h_val.data() + myNnz0, myNnz * sizeof(double));
        std::memcpy(l_cols.data(), h_cols.data() + myNnz0, myNnz * sizeof(index_t));
        // Remote blocks: send
        for (int p = 1; p < nprocs; ++p) {
            const index_t n0 = h_rowDelimiters[rowStart[p]];
            const index_t cnt = h_rowDelimiters[rowStart[p + 1]] - n0;
            sendLarge(h_val.data() + n0, cnt, MPI_DOUBLE, p, 100);
            sendLarge(h_cols.data() + n0, cnt, MPI_UINT32_T, p, 101);
        }
    } else {
        recvLarge(l_val.data(), myNnz, MPI_DOUBLE, 0, 100);
        recvLarge(l_cols.data(), myNnz, MPI_UINT32_T, 0, 101);
    }

    // For validation, compute reference solution on the root with the full matrix
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ------------------------------------------------------------------------
    // Perform SpMV computation (each rank on its row block)
    // ------------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(),
                h_vec.data(), myRows, l_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (isRoot) duration = std::chrono::milliseconds(maxDuration);

    // Gather the distributed result vector onto the root
    {
        std::vector<int> recvCounts(nprocs), displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            recvCounts[p] = static_cast<int>(rowStart[p + 1] - rowStart[p]);
            displs[p] = static_cast<int>(rowStart[p]);
        }
        MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                    isRoot ? h_out.data() : nullptr, recvCounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

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

    // All ranks return the same exit code
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
