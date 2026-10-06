#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
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


// ****************************************************************************
// MPI helpers
// ****************************************************************************
template <typename T> MPI_Datatype mpiType();
template <> MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template <> MPI_Datatype mpiType<index_t>() { return MPI_UINT32_T; }

// Messages are split into chunks so that element counts always fit in an int.
constexpr size_t MPI_CHUNK = size_t(1) << 28;

template <typename T>
void sendLarge(const T* buf, size_t count, int dest, int tag, MPI_Comm comm,
               std::vector<MPI_Request>& reqs) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        reqs.emplace_back();
        MPI_Isend(buf + off, c, mpiType<T>(), dest, tag, comm, &reqs.back());
    }
}

template <typename T>
void recvLarge(T* buf, size_t count, int src, int tag, MPI_Comm comm,
               std::vector<MPI_Request>& reqs) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        reqs.emplace_back();
        MPI_Irecv(buf + off, c, mpiType<T>(), src, tag, comm, &reqs.back());
    }
}

template <typename T>
void bcastLarge(T* buf, size_t count, int root, MPI_Comm comm) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        MPI_Bcast(buf + off, c, mpiType<T>(), root, comm);
    }
}

// Split rows into contiguous blocks balancing (nnz + rows) per rank.
std::vector<index_t> partitionRows(const index_t* rowDelimiters, const index_t dim, const int nprocs) {
    std::vector<index_t> rowStart(nprocs + 1);
    const double total = static_cast<double>(rowDelimiters[dim]) + dim;
    rowStart[0] = 0;
    for (int p = 1; p < nprocs; ++p) {
        const double target = total * p / nprocs;
        index_t lo = rowStart[p - 1], hi = dim;
        while (lo < hi) {
            const index_t mid = lo + (hi - lo) / 2;
            if (static_cast<double>(rowDelimiters[mid]) + mid < target) lo = mid + 1;
            else hi = mid;
        }
        rowStart[p] = lo;
    }
    rowStart[nprocs] = dim;
    return rowStart;
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
// Function: spmvLocal
//
// Purpose:
//   Computes the rows [0, nRows) of this rank's block. rowDelimiters holds
//   global offsets; base is the global offset of the first local nonzero.
//
// ****************************************************************************
void spmvLocal(const double* __restrict__ val, const index_t* __restrict__ cols,
               const index_t* __restrict__ rowDelimiters, const index_t base,
               const double* __restrict__ vec, const index_t nRows, double* __restrict__ out) {
    for (index_t i = 0; i < nRows; ++i) {
        double t = 0.0;
        const index_t end = rowDelimiters[i + 1] - base;
        for (index_t j = rowDelimiters[i] - base; j < end; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
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
        printf("MPI processes: %d\n", nprocs);
    }

    // Root holds the full matrix (generation is inherently sequential because
    // of the rand() stream); other ranks only hold their row block.
    std::vector<double> h_val;                          // Non-zero values
    std::vector<index_t> h_cols;                        // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;                          // Output vector

    if (root) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Distribute row structure and dense vector to all ranks
    bcastLarge(h_rowDelimiters.data(), static_cast<size_t>(numRows) + 1, 0, MPI_COMM_WORLD);
    bcastLarge(h_vec.data(), numRows, 0, MPI_COMM_WORLD);

    const std::vector<index_t> rowStart = partitionRows(h_rowDelimiters.data(), numRows, nprocs);
    const index_t myFirstRow = rowStart[rank];
    const index_t myNumRows = rowStart[rank + 1] - myFirstRow;
    const index_t myBase = h_rowDelimiters[myFirstRow];
    const index_t myNnz = h_rowDelimiters[rowStart[rank + 1]] - myBase;

    // Distribute nonzeros (root keeps using its own full arrays)
    std::vector<double> l_val;
    std::vector<index_t> l_cols;
    const double* myVal;
    const index_t* myCols;
    {
        std::vector<MPI_Request> reqs;
        if (root) {
            for (int p = 1; p < nprocs; ++p) {
                const index_t b = h_rowDelimiters[rowStart[p]];
                const size_t cnt = h_rowDelimiters[rowStart[p + 1]] - b;
                sendLarge(h_val.data() + b, cnt, p, 0, MPI_COMM_WORLD, reqs);
                sendLarge(h_cols.data() + b, cnt, p, 1, MPI_COMM_WORLD, reqs);
            }
            myVal = h_val.data() + myBase;
            myCols = h_cols.data() + myBase;
        } else {
            l_val.resize(myNnz);
            l_cols.resize(myNnz);
            recvLarge(l_val.data(), myNnz, 0, 0, MPI_COMM_WORLD, reqs);
            recvLarge(l_cols.data(), myNnz, 0, 1, MPI_COMM_WORLD, reqs);
            myVal = l_val.data();
            myCols = l_cols.data();
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }
    const index_t* myRowDelimiters = h_rowDelimiters.data() + myFirstRow;
    std::vector<double> l_out(root ? 0 : myNumRows);
    double* myOut = root ? h_out.data() : l_out.data();

    // Gather layout for the output vector
    std::vector<int> recvCounts(nprocs), displs(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        recvCounts[p] = static_cast<int>(rowStart[p + 1] - rowStart[p]);
        displs[p] = static_cast<int>(rowStart[p]);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && root) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvLocal(myVal, myCols, myRowDelimiters, myBase, h_vec.data(), myNumRows, myOut);
    }

    // Assemble the full output vector on root
    if (root) {
        MPI_Gatherv(MPI_IN_PLACE, recvCounts[0], MPI_DOUBLE, h_out.data(), recvCounts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(myOut, static_cast<int>(myNumRows), MPI_DOUBLE, nullptr, nullptr, nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", maxMs);

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
