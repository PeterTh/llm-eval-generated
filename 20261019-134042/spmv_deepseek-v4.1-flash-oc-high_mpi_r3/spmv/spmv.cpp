#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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
            if (isRoot) {
                printUsage(argv[0]);
            }
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
    }

    // Partition the rows of the matrix into contiguous blocks, one per rank.
    const index_t rowStart = static_cast<index_t>(
        (static_cast<uint64_t>(rank) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
    const index_t rowEnd = static_cast<index_t>(
        (static_cast<uint64_t>(rank + 1) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
    const index_t localRows = rowEnd - rowStart;

    // The dense input vector is replicated on every rank: each rank needs
    // entries from the whole vector because column indices are arbitrary.
    std::vector<double> vec(numRows);

    // The root builds the full problem, then scatters row blocks. Doing the
    // generation on one rank preserves the exact random sequence of the
    // original serial code, so results remain bit-for-bit identical.
    std::vector<double> f_val;
    std::vector<index_t> f_cols;
    std::vector<index_t> f_rd;

    // Per-rank block descriptors (values/columns are described by the number
    // of non-zeros in each row block, row delimiters by the row range).
    std::vector<int> valCounts(nprocs, 0);
    std::vector<int> valDispls(nprocs, 0);
    std::vector<int> rdCounts(nprocs, 0);
    std::vector<int> rdDispls(nprocs, 0);

    if (isRoot) {
        printf("Initializing data structures...\n");

        f_val.resize(nItems);                 // Non-zero values
        f_cols.resize(nItems);                // Column indices
        f_rd.resize(numRows + 1);             // Row delimiters

        fill(vec.data(), numRows, maxVal);
        fill(f_val.data(), nItems, maxVal);
        initRandomMatrix(f_cols.data(), f_rd.data(), nItems, numRows);

        for (int r = 0; r < nprocs; ++r) {
            const index_t rs = static_cast<index_t>(
                (static_cast<uint64_t>(r) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
            const index_t re = static_cast<index_t>(
                (static_cast<uint64_t>(r + 1) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
            rdDispls[r] = static_cast<int>(rs);
            rdCounts[r] = static_cast<int>(re - rs + 1);
            valDispls[r] = static_cast<int>(f_rd[rs]);
            valCounts[r] = static_cast<int>(f_rd[re] - f_rd[rs]);
        }
    }

    MPI_Bcast(valCounts.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(valDispls.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rdCounts.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rdDispls.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);

    // Local, row-block-owned portions of the CSR matrix.
    std::vector<double> l_val(static_cast<size_t>(valCounts[rank]));
    std::vector<index_t> l_cols(static_cast<size_t>(valCounts[rank]));
    std::vector<index_t> l_rd(static_cast<size_t>(rdCounts[rank]));
    std::vector<double> l_out(localRows);

    MPI_Scatterv(isRoot ? f_val.data() : nullptr, valCounts.data(), valDispls.data(), MPI_DOUBLE,
                 l_val.data(), valCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? f_cols.data() : nullptr, valCounts.data(), valDispls.data(), MPI_UINT32_T,
                 l_cols.data(), valCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(isRoot ? f_rd.data() : nullptr, rdCounts.data(), rdDispls.data(), MPI_UINT32_T,
                 l_rd.data(), rdCounts[rank], MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Rebase the local row delimiters so they index into the local CSR arrays.
    if (!l_rd.empty()) {
        const index_t base = l_rd[0];
        for (auto& d : l_rd) {
            d -= base;
        }
    }

    // The vector is constant across all iterations, so replicate it once.
    if (numRows > 0) {
        MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // For validation, compute reference solution (root keeps the full problem).
    std::vector<double> reference;
    if (validate && isRoot) {
        printf("Computing reference solution...\n");
        reference.resize(numRows);
        spmvCpu(f_val.data(), f_cols.data(), f_rd.data(),
                vec.data(), numRows, reference.data());
    }

    // The full matrix is no longer needed on the root once the row blocks have
    // been distributed, unless it is used for the reference solution above.
    if (isRoot && !validate) {
        std::vector<double>().swap(f_val);
        std::vector<index_t>().swap(f_cols);
        std::vector<index_t>().swap(f_rd);
    }

    // Perform SpMV computation
    if (isRoot) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val.data(), l_cols.data(), l_rd.data(),
                vec.data(), localRows, l_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long localDurationMs = duration.count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (isRoot) {
        duration = std::chrono::milliseconds(maxDurationMs);
    }

    // Collect the distributed output vector on the root for reporting and
    // validation.
    std::vector<int> outCounts(nprocs, 0);
    std::vector<int> outDispls(nprocs, 0);
    for (int r = 0; r < nprocs; ++r) {
        const index_t rs = static_cast<index_t>(
            (static_cast<uint64_t>(r) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
        const index_t re = static_cast<index_t>(
            (static_cast<uint64_t>(r + 1) * static_cast<uint64_t>(numRows)) / static_cast<uint64_t>(nprocs));
        outDispls[r] = static_cast<int>(rs);
        outCounts[r] = static_cast<int>(re - rs);
    }

    std::vector<double> h_out;
    if (isRoot) {
        h_out.resize(numRows);
    }
    MPI_Gatherv(l_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                isRoot ? h_out.data() : nullptr, outCounts.data(), outDispls.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (isRoot) {
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
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), h_out.data(), numRows);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
