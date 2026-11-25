#include <mpi.h>

#include <algorithm>
#include <climits>
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

// Generate a contiguous row block, continuing the original random stream.
void generateRows(index_t first, index_t rows, index_t dim, index_t total,
                  index_t& assigned, std::vector<index_t>& offsets,
                  std::vector<index_t>& columns) {
    offsets.resize(static_cast<size_t>(rows) + 1);
    columns.clear();
    const uint64_t cells = static_cast<uint64_t>(dim) * dim;
    const double probability = static_cast<double>(total) / static_cast<double>(cells);
    for (index_t i = 0; i < rows; ++i) {
        offsets[i] = static_cast<index_t>(columns.size());
        uint64_t remaining = cells - static_cast<uint64_t>(first + i) * dim;
        for (index_t j = 0; j < dim && assigned < total; ++j, --remaining) {
            if (remaining == total - assigned ||
                static_cast<double>(rand()) / RAND_MAX <= probability) {
                columns.push_back(j);
                ++assigned;
            }
        }
    }
    offsets[rows] = static_cast<index_t>(columns.size());
}

// A block can exceed the count range of the MPI 3 point-to-point interface.
template <class T>
void transferBlock(T* data, size_t size, MPI_Datatype type, int owner, int rank) {
    while (size > 0) {
        const int count = static_cast<int>(std::min<size_t>(size, INT_MAX));
        if (rank == 0) {
            MPI_Send(data, count, type, owner, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(data, count, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        data += count;
        size -= count;
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
bool verifyResults(const double* reference, const double* result, const index_t size, index_t first) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", first + i, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       first + i, ref, res, relError);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const uint64_t cells = static_cast<uint64_t>(numRows) * numRows;
    if (sparsity == 0 || numRows > INT_MAX || cells / sparsity > UINT32_MAX) {
        if (rank == 0) fprintf(stderr, "Unsupported matrix dimensions or sparsity\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(cells / sparsity);
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
    }

    const auto rowStart = [=](int owner) -> index_t {
        return static_cast<uint64_t>(owner) * (numRows / ranks) +
               std::min<index_t>(owner, numRows % ranks);
    };
    const index_t first = rowStart(rank);
    const index_t rows = rowStart(rank + 1) - first;
    std::vector<index_t> offsets, columns;
    std::vector<index_t> counts;
    if (rank == 0) {
        counts.resize(ranks);
        index_t assigned = 0;
        srand(8675309);
        generateRows(0, rows, numRows, nItems, assigned, offsets, columns);
        counts[0] = offsets.back();
        std::vector<index_t> blockOffsets, blockColumns;
        for (int owner = 1; owner < ranks; ++owner) {
            const index_t begin = rowStart(owner);
            generateRows(begin, rowStart(owner + 1) - begin, numRows, nItems,
                         assigned, blockOffsets, blockColumns);
            counts[owner] = blockOffsets.back();
            transferBlock(blockOffsets.data(), blockOffsets.size(), MPI_UINT32_T, owner, rank);
            transferBlock(blockColumns.data(), blockColumns.size(), MPI_UINT32_T, owner, rank);
        }
    } else {
        offsets.resize(static_cast<size_t>(rows) + 1);
        transferBlock(offsets.data(), offsets.size(), MPI_UINT32_T, rank, rank);
        columns.resize(offsets.back());
        transferBlock(columns.data(), columns.size(), MPI_UINT32_T, rank, rank);
    }

    std::vector<double> vec(numRows), values(columns.size()), out(rows);
    if (rank == 0) {
        // The original fills the vector and values from rand's default seed.
        srand(1);
        fill(vec.data(), numRows, maxVal);
    }
    MPI_Bcast(vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        fill(values.data(), counts[0], maxVal);
        std::vector<double> block;
        for (int owner = 1; owner < ranks; ++owner) {
            block.resize(counts[owner]);
            fill(block.data(), counts[owner], maxVal);
            transferBlock(block.data(), block.size(), MPI_DOUBLE, owner, rank);
        }
    } else {
        transferBlock(values.data(), values.size(), MPI_DOUBLE, rank, rank);
    }

    std::vector<double> reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        reference.resize(rows);
        spmvCpu(values.data(), columns.data(), offsets.data(), vec.data(), rows, reference.data());
    }

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(values.data(), columns.data(), offsets.data(), vec.data(), rows, out.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double ms = seconds * 1000.0;
        printf("Computation time: %.0f ms\n", ms);
        printf("Average time per iteration: %.3f ms\n", iterations ? ms / iterations : 0.0);
        printf("Performance: %.3f GFLOPS\n",
               seconds > 0 ? 2.0 * nItems * iterations / seconds / 1e9 : 0.0);
    }

    if (printResults) {
        std::vector<double> result;
        std::vector<int> sizes, starts;
        if (rank == 0) {
            result.resize(numRows);
            sizes.resize(ranks);
            starts.resize(ranks);
            for (int owner = 0; owner < ranks; ++owner) {
                starts[owner] = rowStart(owner);
                sizes[owner] = rowStart(owner + 1) - starts[owner];
            }
        }
        MPI_Gatherv(out.data(), rows, MPI_DOUBLE, result.data(), sizes.data(), starts.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(result, "OutputVector");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localValid = verifyResults(reference.data(), out.data(), rows, first);
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
