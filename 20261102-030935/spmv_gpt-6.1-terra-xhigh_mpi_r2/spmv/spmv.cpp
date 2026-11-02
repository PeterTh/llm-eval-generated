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

// MPI's traditional count arguments are int, while a CSR array may be larger.
template <typename T>
void sendArray(const T* data, size_t count, MPI_Datatype type, int destination, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, type, destination, tag, MPI_COMM_WORLD);
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

    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || numRows > static_cast<index_t>(INT_MAX) ||
        matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Matrix dimensions or sparsity exceed supported limits.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    std::vector<double> vec(numRows), val, reference;
    std::vector<index_t> cols, rows;
    // Partition metadata contains the first and one-past-last row and nonzero.
    std::vector<index_t> partitions;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / static_cast<double>(matrixSize)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        val.resize(nItems);
        cols.resize(nItems);
        rows.resize(static_cast<size_t>(numRows) + 1);
        // Keep the original random number sequence and matrix exactly intact.
        fill(vec.data(), numRows, maxVal);
        fill(val.data(), nItems, maxVal);
        initRandomMatrix(cols.data(), rows.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), numRows, reference.data());
        }

        // Charge each row one unit in addition to its nonzeros. This balances
        // empty rows too, and handles the dense tail produced by the generator.
        std::vector<index_t> boundaries(static_cast<size_t>(ranks) + 1);
        boundaries[ranks] = numRows;
        const uint64_t totalWork = static_cast<uint64_t>(nItems) + numRows;
        for (int p = 1; p < ranks; ++p) {
            const uint64_t target = totalWork * static_cast<uint64_t>(p) / ranks;
            index_t lo = boundaries[p - 1], hi = numRows;
            while (lo < hi) {
                const index_t mid = lo + (hi - lo) / 2;
                if (static_cast<uint64_t>(rows[mid]) + mid < target) lo = mid + 1;
                else hi = mid;
            }
            boundaries[p] = lo;
        }
        partitions.resize(static_cast<size_t>(ranks) * 4);
        for (int p = 0; p < ranks; ++p) {
            const index_t first = boundaries[p], last = boundaries[p + 1];
            partitions[4 * static_cast<size_t>(p)] = first;
            partitions[4 * static_cast<size_t>(p) + 1] = last;
            partitions[4 * static_cast<size_t>(p) + 2] = rows[first];
            partitions[4 * static_cast<size_t>(p) + 3] = rows[last];
        }
    }

    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    index_t part[4];
    MPI_Scatter(partitions.data(), 4, MPI_UINT32_T, part, 4, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = part[1] - part[0];
    const index_t localNnz = part[3] - part[2];
    if (rank == 0) {
        for (int p = 1; p < ranks; ++p) {
            const index_t* block = partitions.data() + 4 * static_cast<size_t>(p);
            sendArray(rows.data() + block[0], static_cast<size_t>(block[1] - block[0]) + 1,
                      MPI_UINT32_T, p, 0);
            if (block[3] != block[2]) {
                sendArray(cols.data() + block[2], block[3] - block[2], MPI_UINT32_T, p, 1);
                sendArray(val.data() + block[2], block[3] - block[2], MPI_DOUBLE, p, 2);
            }
        }
        rows.resize(static_cast<size_t>(localRows) + 1);
        cols.resize(localNnz);
        val.resize(localNnz);
        // Return the unused root storage before entering the timed region.
        if (localNnz != nItems) {
            std::vector<index_t>(cols).swap(cols);
            std::vector<double>(val).swap(val);
        }
    } else {
        rows.resize(static_cast<size_t>(localRows) + 1);
        cols.resize(localNnz);
        val.resize(localNnz);
        receiveArray(rows.data(), rows.size(), MPI_UINT32_T, 0, 0);
        receiveArray(cols.data(), localNnz, MPI_UINT32_T, 0, 1);
        receiveArray(val.data(), localNnz, MPI_DOUBLE, 0, 2);
        for (index_t& row : rows) row -= part[2];
    }

    std::vector<double> localOut(localRows);
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(val.data(), cols.data(), rows.data(), vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Average time per iteration: %.3f ms\n", duration * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / duration / 1e9);
    }

    std::vector<double> out;
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        if (rank == 0) {
            out.resize(numRows);
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int p = 0; p < ranks; ++p) {
                const index_t* block = partitions.data() + 4 * static_cast<size_t>(p);
                counts[p] = static_cast<int>(block[1] - block[0]);
                displacements[p] = static_cast<int>(block[0]);
            }
        }
        MPI_Gatherv(localOut.data(), static_cast<int>(localRows), MPI_DOUBLE,
                    out.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int result = 0;
    if (rank == 0) {
        if (printResults) print_results(out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
