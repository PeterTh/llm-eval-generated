#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
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
// Generates the same random stream as the serial version, but only stores
// elements with global index in [lo, hi) (into A[0 .. hi-lo)).
void fillRange(double* A, const index_t n, const double maxVal, const index_t lo, const index_t hi) {
    for (index_t i = 0; i < hi && i < n; ++i) {
        double v = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
        if (i >= lo) A[i - lo] = v;
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
void initRandomMatrix(std::vector<index_t>& cols, std::vector<index_t>& rowDelimiters, const index_t n,
                      const index_t dim, const index_t r0, const index_t r1, index_t& nnzLo) {
    if (r0 >= r1) {
        nnzLo = 0;
        rowDelimiters[0] = 0;
        return;
    }
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);

    // Every rank replays the (sequential) random stream but only keeps its own rows
    bool fillRemaining = false;
    nnzLo = 0;
    for (index_t i = 0; i < r1; ++i) {
        if (i == r0) nnzLo = nnzAssigned;
        const bool mine = i >= r0;
        if (mine) rowDelimiters[i - r0] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                if (mine) cols.push_back(j);
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[r1 - r0] = (r1 >= dim) ? n : nnzAssigned;
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
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + offset, ref, res, relError);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = rank == 0;
    auto finish = [](int code) {
        MPI_Finalize();
        return code;
    };

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
            return finish(0);
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
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
        printf("MPI ranks: %d\n", nprocs);
        printf("Initializing data structures...\n");
    }

    // 1D row-block distribution
    auto rowStart = [&](int r) {
        return static_cast<index_t>(static_cast<uint64_t>(numRows) * static_cast<uint64_t>(r) / nprocs);
    };
    const index_t r0 = rowStart(rank), r1 = rowStart(rank + 1);
    const index_t localRows = r1 - r0;

    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(localRows + 1);
    index_t nnzLo = 0;
    initRandomMatrix(h_cols, h_rowDelimiters, nItems, numRows, r0, r1, nnzLo);
    const index_t nnzLocal = static_cast<index_t>(h_cols.size());
    for (auto& d : h_rowDelimiters) d -= nnzLo;

    // Same stream as serial code (default seed): vector first, then values
    srand(1);
    std::vector<double> h_vec(numRows);
    fillRange(h_vec.data(), numRows, maxVal, 0, numRows);
    std::vector<double> h_val(nnzLocal);
    fillRange(h_val.data(), nItems, maxVal, nnzLo, nnzLo + nnzLocal);
    std::vector<double> h_out(localRows);

    std::vector<double> h_reference;
    if (validate) {
        if (root) printf("Computing reference solution...\n");
        h_reference.resize(localRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), localRows,
                h_reference.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (root) printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), localRows,
                h_out.data());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double localMs = std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Allreduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long ms = static_cast<long>(maxMs);

    if (root) {
        printf("Computation time: %ld ms\n", ms);
        const double gflops = (2.0 * nItems * iterations) / (ms / 1000.0) / 1e9;
        const double avgTime = ms / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> full;
        std::vector<int> counts(nprocs), displs(nprocs);
        if (root) full.resize(numRows);
        for (int r = 0; r < nprocs; ++r) {
            displs[r] = static_cast<int>(rowStart(r));
            counts[r] = static_cast<int>(rowStart(r + 1)) - displs[r];
        }
        MPI_Gatherv(h_out.data(), static_cast<int>(localRows), MPI_DOUBLE, full.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) print_results(full, "OutputVector");
    }

    if (validate) {
        if (root) printf("Validating result...\n");
        int ok = verifyResults(h_reference.data(), h_out.data(), localRows, r0) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&ok, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (root) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        return finish(allOk ? 0 : 1);
    }

    return finish(0);
}
