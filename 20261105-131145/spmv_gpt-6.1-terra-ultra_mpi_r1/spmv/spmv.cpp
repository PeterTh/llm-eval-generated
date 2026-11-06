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

// MPI's traditional count argument is an int, while CSR arrays can be larger.
// Transfer large arrays in chunks without imposing a smaller matrix size limit.
template <typename T>
void sendArray(const T* data, uint64_t count, int dest, int tag, MPI_Datatype type) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void recvArray(T* data, uint64_t count, int source, int tag, MPI_Datatype type) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void broadcastArray(T* data, uint64_t count, MPI_Datatype type) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min<uint64_t>(count, INT_MAX));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

uint64_t rowBoundary(index_t rows, int rank, int ranks) {
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(rank) /
           static_cast<uint64_t>(ranks);
}

void gatherRows(const std::vector<double>& local, std::vector<double>& global,
                index_t rows, int rank, int ranks, int tag) {
    if (rank == 0) {
        global.resize(rows);
        std::copy(local.begin(), local.end(), global.begin());
        for (int r = 1; r < ranks; ++r) {
            const uint64_t begin = rowBoundary(rows, r, ranks);
            const uint64_t count = rowBoundary(rows, r + 1, ranks) - begin;
            if (count) recvArray(global.data() + begin, count, r, tag, MPI_DOUBLE);
        }
    } else {
        sendArray(local.data(), local.size(), 0, tag, MPI_DOUBLE);
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
    int parseStatus = 0;

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
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or too many non-zero elements for CSR indices\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);
    const uint64_t firstRow = rowBoundary(numRows, rank, ranks);
    const index_t localRows = static_cast<index_t>(rowBoundary(numRows, rank + 1, ranks) - firstRow);

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
    }

    std::vector<index_t> localCols, localOffsets;
    std::vector<index_t> counts;
    if (rank == 0) {
        counts.resize(ranks);
        index_t assigned = 0;
        bool fillRemaining = false;
        const double prob = matrixSize ? static_cast<double>(nItems) / static_cast<double>(matrixSize) : 0.0;
        srand(8675309);
        // The original generator visits matrix entries in row-major order. Generate
        // one rank's rows at a time, so rank 0 never holds the entire matrix.
        for (int r = 0; r < ranks; ++r) {
            const uint64_t begin = rowBoundary(numRows, r, ranks);
            const uint64_t end = rowBoundary(numRows, r + 1, ranks);
            std::vector<index_t> offsets(static_cast<size_t>(end - begin) + 1);
            std::vector<index_t> cols;
            for (uint64_t row = begin; row < end; ++row) {
                offsets[row - begin] = static_cast<index_t>(cols.size());
                for (uint64_t col = 0; col < numRows; ++col) {
                    const uint64_t entriesLeft = matrixSize - (row * numRows + col);
                    const index_t need = nItems - assigned;
                    if (entriesLeft <= need) fillRemaining = true;
                    const double randVal = static_cast<double>(rand()) / RAND_MAX;
                    if ((assigned < nItems && randVal <= prob) || fillRemaining) {
                        cols.push_back(static_cast<index_t>(col));
                        ++assigned;
                    }
                }
            }
            offsets[end - begin] = static_cast<index_t>(cols.size());
            counts[r] = static_cast<index_t>(cols.size());
            if (r == 0) {
                localOffsets = std::move(offsets);
                localCols = std::move(cols);
            } else {
                MPI_Send(&counts[r], 1, MPI_UINT32_T, r, 10, MPI_COMM_WORLD);
                sendArray(offsets.data(), offsets.size(), r, 11, MPI_UINT32_T);
                sendArray(cols.data(), cols.size(), r, 12, MPI_UINT32_T);
            }
        }
    } else {
        index_t count;
        MPI_Recv(&count, 1, MPI_UINT32_T, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localOffsets.resize(static_cast<size_t>(localRows) + 1);
        localCols.resize(count);
        recvArray(localOffsets.data(), localOffsets.size(), 0, 11, MPI_UINT32_T);
        recvArray(localCols.data(), localCols.size(), 0, 12, MPI_UINT32_T);
    }

    std::vector<double> vec(numRows);
    std::vector<double> localValues(localCols.size());
    if (rank == 0) {
        // C starts rand() as though seeded with 1. MPI initialization may have
        // used rand(), so explicitly restore the original input sequence.
        srand(1);
        fill(vec.data(), numRows, maxVal);
    }
    broadcastArray(vec.data(), vec.size(), MPI_DOUBLE);
    if (rank == 0) {
        fill(localValues.data(), counts[0], maxVal);
        std::vector<double> values;
        for (int r = 1; r < ranks; ++r) {
            values.resize(counts[r]);
            fill(values.data(), counts[r], maxVal);
            sendArray(values.data(), values.size(), r, 13, MPI_DOUBLE);
        }
    } else {
        recvArray(localValues.data(), localValues.size(), 0, 13, MPI_DOUBLE);
    }

    std::vector<double> localOut(localRows);
    std::vector<double> localReference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        localReference.resize(localRows);
        spmvCpu(localValues.data(), localCols.data(), localOffsets.data(), vec.data(),
                localRows, localReference.data());
    }

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localCols.data(), localOffsets.data(), vec.data(),
                localRows, localOut.data());
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        printf("Average time per iteration: %.3f ms\n", elapsed * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / elapsed / 1e9);
    }

    std::vector<double> output;
    if (printResults || validate) gatherRows(localOut, output, numRows, rank, ranks, 14);
    if (rank == 0 && printResults) print_results(output, "OutputVector");

    int valid = 1;
    if (validate) {
        std::vector<double> reference;
        gatherRows(localReference, reference, numRows, rank, ranks, 15);
        if (rank == 0) {
            printf("Validating result...\n");
            valid = verifyResults(reference.data(), output.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
