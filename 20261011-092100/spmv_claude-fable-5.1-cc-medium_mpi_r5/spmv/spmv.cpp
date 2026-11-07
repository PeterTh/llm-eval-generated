#include <mpi.h>

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
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        const index_t rowEnd = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < rowEnd; ++j) {
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
// MPI helpers
//
// Point-to-point transfers of possibly very large buffers (counts are split
// into chunks that fit into an int, so the full uint32_t index range works).
// ****************************************************************************
template <typename T>
void sendLarge(const T* buf, size_t count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
    constexpr size_t CHUNK = static_cast<size_t>(INT_MAX);
    size_t off = 0;
    do {
        const size_t n = std::min(CHUNK, count - off);
        MPI_Send(buf + off, static_cast<int>(n), type, dest, tag, comm);
        off += n;
    } while (off < count);
}

template <typename T>
void recvLarge(T* buf, size_t count, MPI_Datatype type, int src, int tag, MPI_Comm comm) {
    constexpr size_t CHUNK = static_cast<size_t>(INT_MAX);
    size_t off = 0;
    do {
        const size_t n = std::min(CHUNK, count - off);
        MPI_Recv(buf + off, static_cast<int>(n), type, src, tag, comm, MPI_STATUS_IGNORE);
        off += n;
    } while (off < count);
}

template <typename T>
void bcastLarge(T* buf, size_t count, MPI_Datatype type, int root, MPI_Comm comm) {
    constexpr size_t CHUNK = static_cast<size_t>(INT_MAX);
    size_t off = 0;
    do {
        const size_t n = std::min(CHUNK, count - off);
        MPI_Bcast(buf + off, static_cast<int>(n), type, root, comm);
        off += n;
    } while (off < count);
}

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Computes a 1D row partition of the CSR matrix such that every rank
//   receives (approximately) the same number of non-zero elements.
//
// Arguments:
//   rowDelimiters: CSR row pointer array (size numRows+1)
//   numRows:       number of rows
//   numRanks:      number of MPI ranks
//   rowStart:      output - rowStart[r] is the first row owned by rank r,
//                  rowStart[numRanks] == numRows
//
// ****************************************************************************
void partitionRows(const index_t* rowDelimiters, const index_t numRows, const int numRanks,
                   std::vector<index_t>& rowStart) {
    rowStart.assign(static_cast<size_t>(numRanks) + 1, numRows);
    rowStart[0] = 0;
    const double totalNnz = static_cast<double>(rowDelimiters[numRows]);
    index_t row = 0;
    for (int r = 1; r < numRanks; ++r) {
        // Target: cumulative nnz boundary for the first r ranks.
        const double target = totalNnz * static_cast<double>(r) / static_cast<double>(numRanks);
        // Also guarantee that each rank keeps at least a proportional share of
        // the rows when the matrix has no non-zeros at all.
        const index_t minRow = static_cast<index_t>(
            (static_cast<unsigned long long>(numRows) * static_cast<unsigned long long>(r)) /
            static_cast<unsigned long long>(numRanks));
        while (row < numRows && (static_cast<double>(rowDelimiters[row]) < target || row < minRow)) {
            ++row;
        }
        rowStart[r] = row;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank sees the same argv)
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
        printf("MPI ranks: %d\n", numRanks);
    }

    // ------------------------------------------------------------------------
    // Data generation (root only, identical random sequence to the serial code)
    // ------------------------------------------------------------------------
    std::vector<double> h_val;             // Non-zero values (root: full matrix)
    std::vector<index_t> h_cols;           // Column indices (root: full matrix)
    std::vector<index_t> h_rowDelimiters;  // Row delimiters (root: full matrix)
    std::vector<double> h_vec(numRows);    // Dense vector (replicated on all ranks)
    std::vector<double> h_out;             // Output vector (root: full result)
    std::vector<index_t> rowStart(static_cast<size_t>(numRanks) + 1);

    if (isRoot) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        partitionRows(h_rowDelimiters.data(), numRows, numRanks, rowStart);
    }

    // ------------------------------------------------------------------------
    // Distribute the matrix: 1D row partition balanced by non-zero count.
    // The dense input vector is replicated, so no communication is needed
    // inside the timed SpMV loop.
    // ------------------------------------------------------------------------
    MPI_Bcast(rowStart.data(), numRanks + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    bcastLarge(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const index_t myRowStart = rowStart[rank];
    const index_t myRowEnd = rowStart[rank + 1];
    const index_t myRows = myRowEnd - myRowStart;

    std::vector<index_t> l_rowDelimiters(static_cast<size_t>(myRows) + 1);
    index_t myNnzStart = 0;
    index_t myNnz = 0;

    if (isRoot) {
        // Send every other rank its slice of the CSR structure.
        for (int r = 1; r < numRanks; ++r) {
            const index_t rs = rowStart[r];
            const index_t re = rowStart[r + 1];
            const index_t nnzStart = h_rowDelimiters[rs];
            const index_t nnzEnd = h_rowDelimiters[re];
            sendLarge(h_rowDelimiters.data() + rs, static_cast<size_t>(re - rs) + 1, MPI_UINT32_T,
                      r, 0, MPI_COMM_WORLD);
            sendLarge(h_val.data() + nnzStart, nnzEnd - nnzStart, MPI_DOUBLE, r, 1, MPI_COMM_WORLD);
            sendLarge(h_cols.data() + nnzStart, nnzEnd - nnzStart, MPI_UINT32_T, r, 2,
                      MPI_COMM_WORLD);
        }
        // Root's own slice is a prefix of the full arrays (no copy needed).
        myNnzStart = h_rowDelimiters[myRowStart];
        myNnz = h_rowDelimiters[myRowEnd] - myNnzStart;
        std::memcpy(l_rowDelimiters.data(), h_rowDelimiters.data() + myRowStart,
                    (static_cast<size_t>(myRows) + 1) * sizeof(index_t));
    } else {
        recvLarge(l_rowDelimiters.data(), static_cast<size_t>(myRows) + 1, MPI_UINT32_T, 0, 0,
                  MPI_COMM_WORLD);
        myNnzStart = l_rowDelimiters[0];
        myNnz = l_rowDelimiters[myRows] - myNnzStart;
        h_val.resize(myNnz);
        h_cols.resize(myNnz);
        recvLarge(h_val.data(), myNnz, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        recvLarge(h_cols.data(), myNnz, MPI_UINT32_T, 0, 2, MPI_COMM_WORLD);
    }

    // Make the local row delimiters relative to the local value/column arrays.
    for (index_t i = 0; i <= myRows; ++i) {
        l_rowDelimiters[i] -= myNnzStart;
    }
    const double* l_val = isRoot ? h_val.data() + myNnzStart : h_val.data();
    const index_t* l_cols = isRoot ? h_cols.data() + myNnzStart : h_cols.data();
    std::vector<double> l_out(myRows);

    // For validation, compute reference solution (serial, on the full matrix)
    std::vector<double> h_reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // ------------------------------------------------------------------------
    // Perform SpMV computation (purely local on every rank)
    // ------------------------------------------------------------------------
    if (isRoot) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val, l_cols, l_rowDelimiters.data(), h_vec.data(), myRows, l_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ------------------------------------------------------------------------
    // Gather the distributed result on the root
    // ------------------------------------------------------------------------
    if (isRoot) {
        std::memcpy(h_out.data(), l_out.data(), static_cast<size_t>(myRows) * sizeof(double));
        for (int r = 1; r < numRanks; ++r) {
            recvLarge(h_out.data() + rowStart[r], rowStart[r + 1] - rowStart[r], MPI_DOUBLE, r, 3,
                      MPI_COMM_WORLD);
        }
    } else {
        sendLarge(l_out.data(), myRows, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
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
        fflush(stdout);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
