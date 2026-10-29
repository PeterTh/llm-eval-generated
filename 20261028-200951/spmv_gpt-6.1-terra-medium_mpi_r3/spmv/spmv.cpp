#include <mpi.h>
#include <algorithm>
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

// Transfer large arrays without relying on MPI's int-sized counts.
template <typename T>
void transfer(T* data, size_t count, MPI_Datatype type, int peer, int tag, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, type, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (sparsity == 0 || static_cast<uint64_t>(numRows) * numRows > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = (numRows * numRows) / sparsity;
    auto rowStart = [numRows, ranks](int r) -> index_t {
        return static_cast<index_t>((static_cast<uint64_t>(numRows) * r) / ranks);
    };
    const index_t first = rowStart(rank), last = rowStart(rank + 1);
    const index_t localRows = last - first;
    std::vector<double> vec(numRows), values, output(localRows), reference;
    std::vector<index_t> cols, rows(localRows + 1);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               numRows ? 100.0 * (1.0 - static_cast<double>(nItems) / (static_cast<double>(numRows) * numRows)) : 0.0);
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal, validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        std::vector<double> allValues(nItems);
        std::vector<index_t> allCols(nItems), allRows(static_cast<size_t>(numRows) + 1);
        fill(vec.data(), numRows, maxVal);
        fill(allValues.data(), nItems, maxVal);
        initRandomMatrix(allCols.data(), allRows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(allValues.data(), allCols.data(), allRows.data(), vec.data(), numRows, reference.data());
        }
        for (int r = 1; r < ranks; ++r) {
            index_t begin = rowStart(r), end = rowStart(r + 1);
            index_t offset = allRows[begin], count = allRows[end] - offset;
            MPI_Send(&count, 1, MPI_UINT32_T, r, 0, MPI_COMM_WORLD);
            transfer(allRows.data() + begin, static_cast<size_t>(end - begin) + 1, MPI_UINT32_T, r, 1, true);
            if (count) {
                transfer(allCols.data() + offset, count, MPI_UINT32_T, r, 2, true);
                transfer(allValues.data() + offset, count, MPI_DOUBLE, r, 3, true);
            }
        }
        index_t count = allRows[last];
        rows.assign(allRows.begin(), allRows.begin() + last + 1);
        cols.assign(allCols.begin(), allCols.begin() + count);
        values.assign(allValues.begin(), allValues.begin() + count);
    } else {
        index_t count;
        MPI_Recv(&count, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        transfer(rows.data(), rows.size(), MPI_UINT32_T, 0, 1, false);
        index_t offset = rows[0];
        for (auto& x : rows) x -= offset;
        cols.resize(count);
        values.resize(count);
        if (count) {
            transfer(cols.data(), count, MPI_UINT32_T, 0, 2, false);
            transfer(values.data(), count, MPI_DOUBLE, 0, 3, false);
        }
    }
    for (size_t offset = 0; offset < vec.size();) {
        int count = static_cast<int>(std::min(vec.size() - offset, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += count;
    }

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter)
        spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), localRows, output.data());
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> result;
    if (printResults || validate) {
        if (rank == 0) {
            result.resize(numRows);
            std::copy(output.begin(), output.end(), result.begin());
            for (int r = 1; r < ranks; ++r) {
                index_t begin = rowStart(r), count = rowStart(r + 1) - begin;
                if (count) transfer(result.data() + begin, count, MPI_DOUBLE, r, 4, false);
            }
        } else if (localRows) {
            transfer(output.data(), localRows, MPI_DOUBLE, 0, 4, true);
        }
    }
    int valid = 1;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        printf("Average time per iteration: %.3f ms\n", iterations ? duration * 1000.0 / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n", duration > 0 ? 2.0 * nItems * iterations / duration / 1e9 : 0.0);
        if (printResults) print_results(result, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            valid = verifyResults(reference.data(), result.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
