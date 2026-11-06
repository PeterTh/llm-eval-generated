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
            uint64_t numEntriesLeft = static_cast<uint64_t>(dim) * dim -
                                      (static_cast<uint64_t>(i) * dim + j);
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

// MPI counts are ints, so transfer large arrays in chunks.
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

void broadcastVector(std::vector<double>& vec) {
    size_t offset = 0;
    while (offset < vec.size()) {
        const int chunk = static_cast<int>(std::min(vec.size() - offset,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_Bcast(vec.data() + offset, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += chunk;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

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
    if (sparsity == 0 || numRows == std::numeric_limits<index_t>::max() ||
        matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid matrix size or sparsity for 32-bit CSR indices\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    std::vector<double> vec(numRows), localVal, reference;
    std::vector<index_t> localCols, localRows;
    std::vector<index_t> rowStarts;
    index_t myRows = 0;

    if (rank == 0) {
        std::vector<double> values(nItems);
        std::vector<index_t> cols(nItems), rows(static_cast<size_t>(numRows) + 1);
        // MPI implementations may use rand(); restore the standard initial state.
        srand(1);
        fill(vec.data(), numRows, maxVal);
        fill(values.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(values.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
        }
        broadcastVector(vec);

        // Balance both nonzeros and per-row work while keeping rows contiguous.
        rowStarts.resize(static_cast<size_t>(processes) + 1);
        rowStarts[0] = 0;
        rowStarts[processes] = numRows;
        const uint64_t totalWork = static_cast<uint64_t>(numRows) + nItems;
        for (int p = 1; p < processes; ++p) {
            const uint64_t target = totalWork * static_cast<uint64_t>(p) / processes;
            index_t lo = 0, hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (static_cast<uint64_t>(mid) + rows[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            rowStarts[p] = lo;
        }
        for (int p = 1; p < processes; ++p) {
            const index_t first = rowStarts[p], end = rowStarts[p + 1];
            const index_t meta[2] = {end - first, rows[end] - rows[first]};
            MPI_Send(meta, 2, MPI_UINT32_T, p, 0, MPI_COMM_WORLD);
            sendArray(rows.data() + first, static_cast<size_t>(meta[0]) + 1, MPI_UINT32_T, p, 1);
            if (meta[1] != 0) {
                sendArray(cols.data() + rows[first], meta[1], MPI_UINT32_T, p, 2);
                sendArray(values.data() + rows[first], meta[1], MPI_DOUBLE, p, 3);
            }
        }
        myRows = rowStarts[1];
        const index_t myNnz = rows[myRows];
        localRows.assign(rows.begin(), rows.begin() + static_cast<size_t>(myRows) + 1);
        localCols.assign(cols.begin(), cols.begin() + myNnz);
        localVal.assign(values.begin(), values.begin() + myNnz);
    } else {
        broadcastVector(vec);
        index_t meta[2];
        MPI_Recv(meta, 2, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        myRows = meta[0];
        localRows.resize(static_cast<size_t>(myRows) + 1);
        localCols.resize(meta[1]);
        localVal.resize(meta[1]);
        receiveArray(localRows.data(), localRows.size(), MPI_UINT32_T, 0, 1);
        receiveArray(localCols.data(), localCols.size(), MPI_UINT32_T, 0, 2);
        receiveArray(localVal.data(), localVal.size(), MPI_DOUBLE, 0, 3);
        const index_t firstNnz = localRows[0];
        for (auto& offset : localRows) offset -= firstNnz;
    }

    std::vector<double> localOut(myRows);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRows.data(),
                vec.data(), myRows, localOut.data());
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double gflops = (2.0 * nItems * iterations) / elapsed / 1e9;
        const double avgTime = elapsed * 1000.0 / iterations;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int status = 0;
    if (printResults || validate) {
        if (rank == 0) {
            std::vector<double> out(numRows);
            std::copy(localOut.begin(), localOut.end(), out.begin());
            for (int p = 1; p < processes; ++p) {
                const size_t count = rowStarts[p + 1] - rowStarts[p];
                if (count != 0) receiveArray(out.data() + rowStarts[p], count, MPI_DOUBLE, p, 4);
            }
            if (printResults) print_results(out, "OutputVector");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = verifyResults(reference.data(), out.data(), numRows);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        } else {
            sendArray(localOut.data(), localOut.size(), MPI_DOUBLE, 0, 4);
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
