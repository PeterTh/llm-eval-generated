#include <mpi.h>

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
// Struct: LocalMatrix
//
// Purpose:
//   The contiguous block of CSR rows owned by one MPI rank. Column indices
//   are global; row delimiters are relative to the first local nonzero.
//
// ****************************************************************************
struct LocalMatrix {
    index_t rowBegin = 0;   // first global row owned by this rank
    index_t rowEnd = 0;     // one past the last global row owned by this rank
    index_t nnzBegin = 0;   // global index of the first local nonzero
    std::vector<double> val;                // local non-zero values
    std::vector<index_t> cols;              // local column indices
    std::vector<index_t> rowDelimiters;     // size rows()+1, relative to nnzBegin

    index_t rows() const { return rowEnd - rowBegin; }
};

// ****************************************************************************
// Function: partitionRows
//
// Purpose:
//   Block-distributes dim rows over size ranks. Blocks differ in size by at
//   most one row.
//
// ****************************************************************************
void partitionRows(const index_t dim, const int rank, const int size,
                   index_t& begin, index_t& end) {
    const uint64_t d = dim;
    begin = static_cast<index_t>(d * static_cast<uint64_t>(rank) / static_cast<uint64_t>(size));
    end = static_cast<index_t>(d * static_cast<uint64_t>(rank + 1) / static_cast<uint64_t>(size));
}

// ****************************************************************************
// Function: initLocalMatrix
//
// Purpose:
//   Regenerates the same random sparsity pattern as the serial generator
//   (identical RNG call sequence), but materializes only the rows in
//   [m.rowBegin, m.rowEnd). The generator is inherently sequential, so every
//   rank replays it and keeps its own block; no communication is required and
//   per-rank memory scales with the local share of the matrix.
//
// Arguments:
//   m:    receives the local rows (rowBegin/rowEnd must be set by the caller)
//   n:    total number of nonzero elements
//   dim:  number of rows/columns in the matrix
//
// ****************************************************************************
void initLocalMatrix(LocalMatrix& m, const index_t n, const index_t dim) {
    const index_t localRows = m.rows();
    m.nnzBegin = 0;
    m.cols.clear();
    m.rowDelimiters.assign(localRows + 1, 0);
    if (localRows == 0) {
        return;
    }

    // Figure out the probability that a nonzero should be assigned
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator (same seed as the serial version)
    srand(8675309);

    index_t nnzAssigned = 0;
    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < m.rowEnd; ++i) {
        const bool local = (i >= m.rowBegin);
        if (i == m.rowBegin) {
            m.nnzBegin = nnzAssigned;
        }
        if (local) {
            m.rowDelimiters[i - m.rowBegin] = nnzAssigned - m.nnzBegin;
        }
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                // Assign (i,j) a value
                if (local) {
                    m.cols.push_back(j);
                }
                nnzAssigned++;
            }
        }
    }
    // Convention: the number of local non-zeroes ends the local row delimiters
    m.rowDelimiters[localRows] = nnzAssigned - m.nnzBegin;
}

// ****************************************************************************
// Function: initLocalData
//
// Purpose:
//   Builds this rank's matrix block and the full dense vector. Random values
//   are drawn in the same order as the serial program (vector first, then the
//   non-zero values, with the matrix pattern generated from its own seed), so
//   the data is bit-identical to the serial version. Only the local slice of
//   the non-zero values is stored.
//
// ****************************************************************************
void initLocalData(LocalMatrix& m, std::vector<double>& vec,
                   const index_t n, const index_t dim, const double maxVal) {
    initLocalMatrix(m, n, dim);

    // The serial program draws the vector and values from the default RNG
    // state (equivalent to seed 1) before the matrix generator reseeds.
    srand(1);
    vec.resize(dim);
    fill(vec.data(), dim, maxVal);

    const index_t localNnz = static_cast<index_t>(m.cols.size());
    m.val.resize(localNnz);
    // Consume the full value stream, storing only the local slice
    for (index_t k = 0; k < n; ++k) {
        const double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (k >= m.nnzBegin && k - m.nnzBegin < localNnz) {
            m.val[k - m.nnzBegin] = v;
        }
    }
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
//   dim: number of rows in the (local) matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
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
//   offset: global index of element 0 (for diagnostics)
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size,
                   const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", offset + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       offset + i, ref, res, relError);
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

    // Parse command line arguments (identical on every rank)
    int exitCode = -1;  // -1: continue running the benchmark
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
            exitCode = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
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
        printf("MPI ranks: %d\n", size);
        printf("Initializing data structures...\n");
    }

    // Each rank owns a contiguous block of rows
    LocalMatrix mat;
    partitionRows(numRows, rank, size, mat.rowBegin, mat.rowEnd);
    const index_t localRows = mat.rows();

    std::vector<double> h_vec;  // Dense vector, replicated on every rank
    initLocalData(mat, h_vec, nItems, numRows, maxVal);

    std::vector<double> h_out(localRows);  // Local slice of the output vector

    // For validation, compute the reference solution for the local rows
    std::vector<double> h_reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(mat.val.data(), mat.cols.data(), mat.rowDelimiters.data(),
                h_vec.data(), localRows, h_reference.data());
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(mat.val.data(), mat.cols.data(), mat.rowDelimiters.data(),
                h_vec.data(), localRows, h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();

    // The slowest rank determines the wall time of the parallel computation
    const double localSec = std::chrono::duration<double>(end - start).count();
    double slowestSec = 0.0;
    MPI_Reduce(&localSec, &slowestSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(slowestSec));
        printf("Computation time: %ld ms\n", static_cast<long>(durationMs.count()));

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / slowestSec / 1e9;
        const double avgTime = (slowestSec * 1000.0) / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather the full output vector)
    if (printResults) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            index_t b = 0;
            index_t e = 0;
            partitionRows(numRows, r, size, b, e);
            counts[r] = static_cast<int>(e - b);
            displs[r] = static_cast<int>(b);
        }
        std::vector<double> fullOut(root ? numRows : 0);
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    fullOut.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) {
            print_results(fullOut, "OutputVector");
        }
    }

    // Validation
    int exitStatus = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        const bool localValid = verifyResults(h_reference.data(), h_out.data(), localRows, mat.rowBegin);
        int localOk = localValid ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        const bool valid = (allOk != 0);

        if (root) {
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        exitStatus = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitStatus;
}
