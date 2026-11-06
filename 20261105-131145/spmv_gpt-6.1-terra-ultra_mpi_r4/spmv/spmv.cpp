#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <utility>
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

// MPI counts are limited to int even though the CSR index type is uint32_t.
// Split transfers so large local partitions are not truncated.
void sendChunks(const void* data, size_t count, MPI_Datatype type, size_t elementSize,
                int destination, int tag) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(bytes, chunk, type, destination, tag, MPI_COMM_WORLD);
        bytes += static_cast<size_t>(chunk) * elementSize;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveChunks(void* data, size_t count, MPI_Datatype type, size_t elementSize,
                   int source, int tag) {
    auto* bytes = static_cast<unsigned char*>(data);
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(bytes, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bytes += static_cast<size_t>(chunk) * elementSize;
        count -= static_cast<size_t>(chunk);
    }
}

void broadcastVector(std::vector<double>& vec) {
    size_t offset = 0;
    while (offset < vec.size()) {
        const int count = static_cast<int>(std::min(vec.size() - offset, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(vec.data() + offset, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        offset += static_cast<size_t>(count);
    }
}

// This partition also works when there are more processes than rows.
index_t rowBoundary(index_t rows, int rank, int ranks) {
    return static_cast<index_t>((static_cast<uint64_t>(rows) * static_cast<unsigned>(rank)) /
                                static_cast<unsigned>(ranks));
}

void gatherRows(const std::vector<double>& local, std::vector<double>& result,
                index_t rows, int rank, int ranks, int tag) {
    if (rank == 0) {
        std::copy(local.begin(), local.end(), result.begin());
        for (int source = 1; source < ranks; ++source) {
            const index_t begin = rowBoundary(rows, source, ranks);
            const index_t end = rowBoundary(rows, source + 1, ranks);
            if (end != begin) {
                receiveChunks(result.data() + begin, end - begin, MPI_DOUBLE, sizeof(double), source, tag);
            }
        }
    } else {
        sendChunks(local.data(), local.size(), MPI_DOUBLE, sizeof(double), 0, tag);
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
    if (sparsity == 0 || matrixSize / sparsity > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix has too many non-zero elements\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);
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

    const index_t rowBegin = rowBoundary(numRows, rank, ranks);
    const index_t rowEnd = rowBoundary(numRows, rank + 1, ranks);
    const index_t localRows = rowEnd - rowBegin;
    std::vector<index_t> localCols;
    std::vector<index_t> localDelimiters(static_cast<size_t>(localRows) + 1);
    std::vector<double> vec(numRows);

    // Generate the original CSR sequence on rank 0, one row partition at a
    // time. Only one remote partition is buffered, avoiding a full matrix on
    // the root. initRandomMatrix always starts its own RNG sequence.
    if (rank == 0) {
        srand(8675309);
        uint64_t assigned = 0;
        bool fillRemaining = false;
        const double prob = static_cast<double>(nItems) / static_cast<double>(matrixSize);
        for (int dest = 0; dest < ranks; ++dest) {
            const index_t begin = rowBoundary(numRows, dest, ranks);
            const index_t end = rowBoundary(numRows, dest + 1, ranks);
            std::vector<index_t> cols;
            std::vector<index_t> delimiters(static_cast<size_t>(end - begin) + 1);
            for (index_t i = begin; i < end; ++i) {
                delimiters[i - begin] = static_cast<index_t>(cols.size());
                for (index_t j = 0; j < numRows; ++j) {
                    const uint64_t entriesLeft = matrixSize - (static_cast<uint64_t>(i) * numRows + j);
                    if (entriesLeft <= static_cast<uint64_t>(nItems) - assigned) fillRemaining = true;
                    const double randVal = static_cast<double>(rand()) / RAND_MAX;
                    if ((assigned < nItems && randVal <= prob) || fillRemaining) {
                        cols.push_back(j);
                        ++assigned;
                    }
                }
            }
            delimiters[end - begin] = static_cast<index_t>(cols.size());
            if (dest == 0) {
                localCols = std::move(cols);
                localDelimiters = std::move(delimiters);
            } else {
                const uint64_t count = cols.size();
                MPI_Send(&count, 1, MPI_UINT64_T, dest, 1, MPI_COMM_WORLD);
                sendChunks(delimiters.data(), delimiters.size(), MPI_UINT32_T, sizeof(index_t), dest, 2);
                sendChunks(cols.data(), cols.size(), MPI_UINT32_T, sizeof(index_t), dest, 3);
            }
        }
    } else {
        uint64_t count = 0;
        MPI_Recv(&count, 1, MPI_UINT64_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localCols.resize(static_cast<size_t>(count));
        receiveChunks(localDelimiters.data(), localDelimiters.size(), MPI_UINT32_T, sizeof(index_t), 0, 2);
        receiveChunks(localCols.data(), localCols.size(), MPI_UINT32_T, sizeof(index_t), 0, 3);
    }

    // The C library starts as if seeded with 1. Restore that state explicitly:
    // MPI_Init may itself use rand(). Values follow the vector in this stream.
    std::vector<double> localVal(localCols.size());
    if (rank == 0) {
        srand(1);
        fill(vec.data(), numRows, maxVal);
    }
    broadcastVector(vec);
    if (rank == 0) {
        fill(localVal.data(), static_cast<index_t>(localVal.size()), maxVal);
        for (int dest = 1; dest < ranks; ++dest) {
            uint64_t count = 0;
            MPI_Recv(&count, 1, MPI_UINT64_T, dest, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Bound the temporary buffer independently of the matrix size.
            std::vector<double> buffer(static_cast<size_t>(std::min<uint64_t>(count, 1048576)));
            while (count != 0) {
                const index_t chunk = static_cast<index_t>(std::min<uint64_t>(count, buffer.size()));
                fill(buffer.data(), chunk, maxVal);
                sendChunks(buffer.data(), chunk, MPI_DOUBLE, sizeof(double), dest, 5);
                count -= chunk;
            }
        }
    } else {
        const uint64_t count = localVal.size();
        MPI_Send(&count, 1, MPI_UINT64_T, 0, 4, MPI_COMM_WORLD);
        // Match the root's bounded messages, including their boundaries.
        size_t offset = 0;
        while (offset < localVal.size()) {
            const size_t chunk = std::min(localVal.size() - offset, static_cast<size_t>(1048576));
            receiveChunks(localVal.data() + offset, chunk, MPI_DOUBLE, sizeof(double), 0, 5);
            offset += chunk;
        }
    }

    std::vector<double> localOut(localRows);
    std::vector<double> reference;
    if (validate) {
        if (rank == 0) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
        }
        std::vector<double> localReference(localRows);
        spmvCpu(localVal.data(), localCols.data(), localDelimiters.data(),
                vec.data(), localRows, localReference.data());
        gatherRows(localReference, reference, numRows, rank, ranks, 6);
    }

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localDelimiters.data(),
                vec.data(), localRows, localOut.data());
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> output;
    if (printResults || validate) {
        if (rank == 0) output.resize(numRows);
        gatherRows(localOut, output, numRows, rank, ranks, 7);
    }

    int valid = 1;
    if (rank == 0) {
        const double milliseconds = elapsed * 1000.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", milliseconds / static_cast<double>(iterations));
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / elapsed / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            valid = verifyResults(reference.data(), output.data(), numRows) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
