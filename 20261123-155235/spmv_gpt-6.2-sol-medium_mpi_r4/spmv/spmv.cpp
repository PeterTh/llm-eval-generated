#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using offset_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr int ROOT = 0;
constexpr offset_t TRANSFER_CHUNK = 256 * 1024;

enum MessageTag { RowOffsets, Columns, Values, Validation };

struct CsrMatrix {
    std::vector<offset_t> rowDelimiters;
    std::vector<index_t> cols;
    std::vector<double> val;
};

// Contiguous row blocks keep CSR traversal local and preserve global output order.
index_t rowStart(index_t dim, int rank, int ranks) {
    return rank * (dim / ranks) + std::min(static_cast<index_t>(rank), dim % ranks);
}

void fill(double* data, offset_t count, double maxVal) {
    for (offset_t i = 0; i < count; ++i) {
        data[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Generate each block in the original row-major random stream. Once every
// remaining position has a fixed outcome, the remaining draws are unused.
void initRandomMatrix(CsrMatrix& matrix, index_t dim, index_t firstRow, index_t endRow,
                      double prob, offset_t& remaining) {
    matrix.rowDelimiters.resize(static_cast<size_t>(endRow - firstRow) + 1);
    matrix.cols.clear();
    const offset_t blockEntries = static_cast<offset_t>(dim) * (endRow - firstRow);
    if (remaining != 0) {
        const auto expected = static_cast<offset_t>(blockEntries * prob);
        matrix.cols.reserve(std::min({blockEntries, remaining, expected + expected / 16 + 64}));
    }
    offset_t entriesLeft = static_cast<offset_t>(dim) * (dim - firstRow);
    for (index_t i = firstRow; i < endRow; ++i) {
        matrix.rowDelimiters[i - firstRow] = matrix.cols.size();
        for (index_t j = 0; j < dim && remaining != 0; ++j, --entriesLeft) {
            if (entriesLeft == remaining || static_cast<double>(rand()) / RAND_MAX <= prob) {
                matrix.cols.push_back(j);
                --remaining;
            }
        }
    }
    matrix.rowDelimiters.back() = matrix.cols.size();
}

void distributeMatrix(CsrMatrix& local, std::vector<double>& vec, index_t dim,
                      offset_t nItems, double maxVal, int rank, int ranks) {
    const int activeRanks = std::min(ranks, static_cast<int>(dim));
    if (rank == ROOT) {
        const offset_t entries = static_cast<offset_t>(dim) * dim;
        const double prob = entries == 0 ? 0.0 : static_cast<double>(nItems) / entries;
        offset_t remaining = nItems;
        std::vector<offset_t> counts(activeRanks);

        srand(8675309);
        initRandomMatrix(local, dim, 0, rowStart(dim, 1, ranks), prob, remaining);
        {
            CsrMatrix block;
            for (int peer = 1; peer < activeRanks; ++peer) {
                initRandomMatrix(block, dim, rowStart(dim, peer, ranks),
                                 rowStart(dim, peer + 1, ranks), prob, remaining);
                counts[peer] = block.cols.size();
                MPI_Send(block.rowDelimiters.data(), block.rowDelimiters.size(),
                         MPI_UINT64_T, peer, RowOffsets, MPI_COMM_WORLD);
                for (offset_t start = 0; start < counts[peer]; start += TRANSFER_CHUNK) {
                    const int count = std::min(TRANSFER_CHUNK, counts[peer] - start);
                    MPI_Send(block.cols.data() + start, count, MPI_UINT32_T,
                             peer, Columns, MPI_COMM_WORLD);
                }
            }
        }

        // The original vector and matrix values use the default rand() stream,
        // independent of the seed used to assign column positions.
        srand(1);
        fill(vec.data(), dim, maxVal);
        MPI_Bcast(vec.data(), dim, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        local.val.resize(local.cols.size());
        fill(local.val.data(), local.val.size(), maxVal);
        std::vector<double> buffer(activeRanks > 1 ? std::min(TRANSFER_CHUNK, nItems) : 0);
        for (int peer = 1; peer < activeRanks; ++peer) {
            for (offset_t start = 0; start < counts[peer]; start += TRANSFER_CHUNK) {
                const int count = std::min(TRANSFER_CHUNK, counts[peer] - start);
                fill(buffer.data(), count, maxVal);
                MPI_Send(buffer.data(), count, MPI_DOUBLE, peer, Values, MPI_COMM_WORLD);
            }
        }
    } else {
        const index_t rows = rowStart(dim, rank + 1, ranks) - rowStart(dim, rank, ranks);
        local.rowDelimiters.resize(static_cast<size_t>(rows) + 1);
        if (rows != 0) {
            MPI_Recv(local.rowDelimiters.data(), local.rowDelimiters.size(), MPI_UINT64_T,
                     ROOT, RowOffsets, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        const offset_t count = local.rowDelimiters.back();
        local.cols.resize(count);
        for (offset_t start = 0; start < count; start += TRANSFER_CHUNK) {
            const int chunk = std::min(TRANSFER_CHUNK, count - start);
            MPI_Recv(local.cols.data() + start, chunk, MPI_UINT32_T,
                     ROOT, Columns, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Bcast(vec.data(), dim, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        local.val.resize(count);
        for (offset_t start = 0; start < count; start += TRANSFER_CHUNK) {
            const int chunk = std::min(TRANSFER_CHUNK, count - start);
            MPI_Recv(local.val.data() + start, chunk, MPI_DOUBLE,
                     ROOT, Values, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
//   dim: number of rows/columns in the matrix
//   out: output - result from the spmv calculation
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const offset_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (offset_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

// ****************************************************************************
// Function: firstMismatch
//
// Purpose:
//   Returns the first local row outside the tolerance, or size on success.
//
// Arguments:
//   reference: array holding the reference result vector
//   result: array holding the result vector to verify
//   size: number of elements per vector
//
// ****************************************************************************
index_t firstMismatch(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        const double error = std::abs(ref) < 1e-10 ? std::abs(res) : std::abs((res - ref) / ref);
        if (error > MAX_RELATIVE_ERROR) {
            return i;
        }
    }
    return size;
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
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
            if (rank == ROOT) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == ROOT) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max()) || sparsity == 0) {
        if (rank == ROOT) {
            fprintf(stderr, "The matrix dimension must fit in an MPI count and sparsity must be positive.\n");
        }
        return 1;
    }
    const offset_t entries = static_cast<offset_t>(numRows) * numRows;
    const offset_t nItems = entries / sparsity;
    const index_t firstRow = rowStart(numRows, rank, ranks);
    const index_t localRows = rowStart(numRows, rank + 1, ranks) - firstRow;

    if (rank == ROOT) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %" PRIu64 " (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    CsrMatrix local;
    std::vector<double> vec(numRows);
    std::vector<double> output(localRows);
    distributeMatrix(local, vec, numRows, nItems, maxVal, rank, ranks);

    std::vector<double> reference;
    if (validate) {
        if (rank == ROOT) {
            printf("Computing reference solution...\n");
        }
        reference.resize(localRows);
        spmvCpu(local.val.data(), local.cols.data(), local.rowDelimiters.data(),
                vec.data(), localRows, reference.data());
    }

    if (rank == ROOT) {
        printf("Computing SpMV...\n");
    }
    // The matrix and input vector are immutable. Every iteration is rank-local.
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local.val.data(), local.cols.data(), local.rowDelimiters.data(),
                vec.data(), localRows, output.data());
    }

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT) {
        const double gflops = seconds > 0.0 ? (2.0 * nItems * iterations) / seconds / 1e9 : 0.0;
        const double avgTime = iterations > 0 ? seconds * 1000.0 / iterations : 0.0;
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<int> rowCounts, displacements;
        std::vector<double> gathered;
        if (rank == ROOT) {
            rowCounts.resize(ranks);
            displacements.resize(ranks);
            gathered.resize(numRows);
            for (int peer = 0; peer < ranks; ++peer) {
                displacements[peer] = rowStart(numRows, peer, ranks);
                rowCounts[peer] = rowStart(numRows, peer + 1, ranks) - displacements[peer];
            }
        }
        MPI_Gatherv(output.data(), localRows, MPI_DOUBLE, gathered.data(), rowCounts.data(),
                    displacements.data(), MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        if (rank == ROOT) {
            print_results(gathered, "OutputVector");
        }
    }

    if (validate) {
        if (rank == ROOT) {
            printf("Validating result...\n");
        }
        const index_t mismatch = firstMismatch(reference.data(), output.data(), localRows);
        const index_t failedRow = mismatch == localRows ? numRows : firstRow + mismatch;
        index_t firstFailure = numRows;
        MPI_Allreduce(&failedRow, &firstFailure, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
        if (firstFailure != numRows) {
            double values[2];
            if (failedRow == firstFailure) {
                values[0] = reference[mismatch];
                values[1] = output[mismatch];
                if (rank != ROOT) {
                    MPI_Send(values, 2, MPI_DOUBLE, ROOT, Validation, MPI_COMM_WORLD);
                }
            }
            if (rank == ROOT) {
                if (failedRow != firstFailure) {
                    MPI_Recv(values, 2, MPI_DOUBLE, MPI_ANY_SOURCE, Validation,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                printf("Validation failed at index %u: reference %.10e, got %.10e",
                       firstFailure, values[0], values[1]);
                if (std::abs(values[0]) >= 1e-10) {
                    printf(" (rel error: %.10e)", std::abs((values[1] - values[0]) / values[0]));
                }
                printf("\nValidation: FAILED\n");
            }
            return 1;
        }
        if (rank == ROOT) {
            printf("Validation: PASSED\n");
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    try {
        const int status = runBenchmark(argc, argv, rank, ranks);
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
