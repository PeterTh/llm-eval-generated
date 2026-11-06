#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
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

// MPI counts are int, while a CSR block can contain more than INT_MAX entries.
// Transfer large arrays in chunks without imposing that limit on the matrix.
template <typename T>
void sendArray(const T* data, size_t count, MPI_Datatype type, int dest, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void receiveArray(T* data, size_t count, MPI_Datatype type, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    // The original CSR generator uses 32-bit indices and 32-bit cell offsets.
    const uint64_t cells = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || cells > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(cells / sparsity);

    std::vector<double> vec(numRows);
    std::vector<double> values, reference;
    std::vector<index_t> columns, rowDelimiters;
    std::vector<index_t> boundaries;
    std::vector<int> rowCounts, rowDisplacements;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / cells));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");

        values.resize(nItems);
        columns.resize(nItems);
        rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // Match the C library's initial rand() state even if MPI_Init uses rand().
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(columns.data(), rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), columns.data(), rowDelimiters.data(),
                    vec.data(), numRows, reference.data());
        }

        // Choose contiguous row boundaries near equal shares of the nonzeros.
        // Whole rows stay together, retaining the original accumulation order.
        boundaries.resize(static_cast<size_t>(ranks) + 1);
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        for (int p = 1; p < ranks; ++p) {
            if (nItems == 0) {
                boundaries[p] = static_cast<index_t>(static_cast<uint64_t>(numRows) * p / ranks);
            } else {
                const index_t target = static_cast<index_t>(static_cast<uint64_t>(nItems) * p / ranks);
                auto it = std::lower_bound(rowDelimiters.begin() + boundaries[p - 1],
                                           rowDelimiters.end(), target);
                index_t pos = static_cast<index_t>(it - rowDelimiters.begin());
                if (pos > boundaries[p - 1] &&
                    target - rowDelimiters[pos - 1] <= rowDelimiters[pos] - target) --pos;
                boundaries[p] = pos;
            }
        }
        rowCounts.resize(ranks);
        rowDisplacements.resize(ranks);
        for (int p = 0; p < ranks; ++p) {
            rowCounts[p] = static_cast<int>(boundaries[p + 1] - boundaries[p]);
            rowDisplacements[p] = static_cast<int>(boundaries[p]);
        }
    }

    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    index_t localRange[2];
    // Adjacent boundary pairs overlap, so send them explicitly along with CSR data.
    std::vector<index_t> localRows, localColumns;
    std::vector<double> localValues;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            index_t range[2] = {boundaries[p], boundaries[p + 1]};
            const index_t first = rowDelimiters[range[0]];
            const index_t count = rowDelimiters[range[1]] - first;
            MPI_Send(range, 2, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            sendArray(rowDelimiters.data() + range[0],
                      static_cast<size_t>(range[1] - range[0]) + 1, MPI_UINT32_T, p, 1);
            if (count) {
                sendArray(columns.data() + first, count, MPI_UINT32_T, p, 2);
                sendArray(values.data() + first, count, MPI_DOUBLE, p, 3);
            }
        }
        localRange[0] = 0;
        localRange[1] = boundaries[1];
        const index_t count = rowDelimiters[localRange[1]];
        localRows.assign(rowDelimiters.begin(), rowDelimiters.begin() + localRange[1] + 1);
        std::vector<index_t>().swap(rowDelimiters);
        localColumns.assign(columns.begin(), columns.begin() + count);
        std::vector<index_t>().swap(columns);
        localValues.assign(values.begin(), values.begin() + count);
        std::vector<double>().swap(values);
    } else {
        MPI_Recv(localRange, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localRows.resize(static_cast<size_t>(localRange[1] - localRange[0]) + 1);
        receiveArray(localRows.data(), localRows.size(), MPI_UINT32_T, 0, 1);
        const index_t first = localRows.front();
        const index_t count = localRows.back() - first;
        for (index_t& offset : localRows) offset -= first;
        localColumns.resize(count);
        localValues.resize(count);
        receiveArray(localColumns.data(), count, MPI_UINT32_T, 0, 2);
        receiveArray(localValues.data(), count, MPI_DOUBLE, 0, 3);
    }

    const index_t localCount = localRange[1] - localRange[0];
    std::vector<double> localOut(localCount);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localColumns.data(), localRows.data(),
                vec.data(), localCount, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> output;
    if (printResults || validate) {
        if (rank == 0) output.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    output.data(), rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double gflops = (2.0 * nItems * iterations) / maxElapsed / 1e9;
        const double avgTime = maxElapsed * 1000.0 / iterations;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), output.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
