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

// MPI's classic interfaces use int counts. Chunk transfers to support the
// full range of the benchmark's 32-bit CSR indices.
template <typename T>
void sendChunks(const T* data, size_t count, MPI_Datatype type, int dest, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void receiveChunks(T* data, size_t count, MPI_Datatype type, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void broadcastChunks(T* data, size_t count, MPI_Datatype type) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
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

    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || matrixSize > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity (32-bit CSR indices required).\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    std::vector<double> vec(numRows);
    std::vector<double> values, reference;
    std::vector<index_t> cols, rowDelimiters;
    std::vector<index_t> boundaries(static_cast<size_t>(ranks) + 1);
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               matrixSize ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize) : 0.0);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        values.resize(nItems);
        cols.resize(nItems);
        rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        // MPI implementations may use rand() during initialization.
        // Restore the C library's initial random state used by the original.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), cols.data(), rowDelimiters.data(),
                    vec.data(), numRows, reference.data());
        }

        // Balance both nonzeros and the cost of visiting empty rows. Boundaries
        // always fall on whole rows, so each sum has the original order.
        boundaries[0] = 0;
        boundaries[ranks] = numRows;
        const uint64_t totalWork = static_cast<uint64_t>(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = (totalWork * static_cast<uint64_t>(p)) / ranks;
            index_t lo = boundaries[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (static_cast<uint64_t>(rowDelimiters[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
    }

    broadcastChunks(vec.data(), vec.size(), MPI_DOUBLE);
    broadcastChunks(boundaries.data(), boundaries.size(), MPI_UINT32_T);
    const index_t firstRow = boundaries[rank];
    const index_t localRows = boundaries[rank + 1] - firstRow;
    std::vector<index_t> localRowsPtr(static_cast<size_t>(localRows) + 1);
    std::vector<index_t> localCols;
    std::vector<double> localValues;
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t begin = boundaries[p], end = boundaries[p + 1];
            const index_t first = rowDelimiters[begin];
            const index_t count = rowDelimiters[end] - first;
            sendChunks(rowDelimiters.data() + begin, static_cast<size_t>(end - begin) + 1,
                       MPI_UINT32_T, p, 0);
            if (count) {
                sendChunks(cols.data() + first, count, MPI_UINT32_T, p, 1);
                sendChunks(values.data() + first, count, MPI_DOUBLE, p, 2);
            }
        }
        std::copy_n(rowDelimiters.data(), localRowsPtr.size(), localRowsPtr.data());
        const index_t localNnz = rowDelimiters[localRows];
        localCols.assign(cols.begin(), cols.begin() + localNnz);
        localValues.assign(values.begin(), values.begin() + localNnz);
        std::vector<index_t>().swap(cols);
        std::vector<index_t>().swap(rowDelimiters);
        std::vector<double>().swap(values);
    } else {
        receiveChunks(localRowsPtr.data(), localRowsPtr.size(), MPI_UINT32_T, 0, 0);
        const index_t first = localRowsPtr.front();
        const index_t localNnz = localRowsPtr.back() - first;
        localCols.resize(localNnz);
        localValues.resize(localNnz);
        receiveChunks(localCols.data(), localNnz, MPI_UINT32_T, 0, 1);
        receiveChunks(localValues.data(), localNnz, MPI_DOUBLE, 0, 2);
        for (auto& offset : localRowsPtr) offset -= first;
    }
    std::vector<double> localOut(localRows);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localRowsPtr.data(),
                vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> out;
    if (printResults || validate) {
        if (rank == 0) {
            out.resize(numRows);
            std::copy(localOut.begin(), localOut.end(), out.begin());
            for (int p = 1; p < ranks; ++p) {
                const size_t count = boundaries[p + 1] - boundaries[p];
                if (count) receiveChunks(out.data() + boundaries[p], count, MPI_DOUBLE, p, 3);
            }
        } else {
            sendChunks(localOut.data(), localOut.size(), MPI_DOUBLE, 0, 3);
        }
    }

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double gflops = maxElapsed > 0 ? (2.0 * nItems * iterations) / maxElapsed / 1e9 : 0.0;
        const double avgTime = iterations ? maxElapsed * 1000.0 / iterations : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
