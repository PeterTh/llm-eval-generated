#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = uint32_t;
using nnz_t = uint64_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

// The root rank performs initialization so the generated matrix, vector, and
// externally reported result are bit-for-bit identical to the serial version.
void fill(double* array, const size_t count, const double maxVal) {
    for (size_t i = 0; i < count; ++i) {
        array[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Generates a CSR pattern with exactly n nonzero elements.  The 64-bit row
// offsets avoid overflow for matrices larger than the original 32-bit CSR
// offset representation could describe.
void initRandomMatrix(index_t* cols, nnz_t* rowDelimiters, const nnz_t n, const index_t dim) {
    nnz_t nnzAssigned = 0;
    const nnz_t matrixEntries = static_cast<nnz_t>(dim) * dim;
    const double probability = static_cast<double>(n) / static_cast<double>(matrixEntries);

    srand(8675309);

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const nnz_t entriesLeft = matrixEntries - (static_cast<nnz_t>(i) * dim + j);
            const nnz_t needToAssign = n - nnzAssigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randomValue <= probability) || fillRemaining) {
                cols[nnzAssigned++] = j;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// The matrix is partitioned by contiguous rows.  Every rank holds the full,
// immutable input vector, so the timed SpMV loop is communication-free.
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const nnz_t* __restrict rowDelimiters, const double* __restrict vec,
             const size_t rows, double* __restrict out) {
    for (size_t i = 0; i < rows; ++i) {
        double total = 0.0;
        for (nnz_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            total += val[j] * vec[cols[j]];
        }
        out[i] = total;
    }
}

bool verifyResults(const double* reference, const double* result, const size_t size) {
    for (size_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %zu: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
            const double relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %zu: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i, ref, res, relativeError);
                return false;
            }
        }
    }
    return true;
}

// Assign row ranges by CSR work rather than by row count.  This keeps the
// nonzero work per rank balanced even when a random matrix is uneven by row.
std::vector<index_t> partitionRows(const std::vector<nnz_t>& rowDelimiters, const int ranks) {
    const index_t numRows = static_cast<index_t>(rowDelimiters.size() - 1);
    const nnz_t totalNnz = rowDelimiters.back();
    std::vector<index_t> starts(static_cast<size_t>(ranks) + 1);
    starts[0] = 0;

    for (int rank = 1; rank < ranks; ++rank) {
        const nnz_t target = (totalNnz * static_cast<nnz_t>(rank)) / ranks;
        const auto begin = rowDelimiters.begin() + starts[rank - 1];
        const auto candidate = std::lower_bound(begin, rowDelimiters.end(), target);

        auto selected = candidate;
        if (candidate != begin) {
            const auto previous = candidate - 1;
            const nnz_t above = candidate == rowDelimiters.end()
                                      ? std::numeric_limits<nnz_t>::max()
                                      : *candidate - target;
            const nnz_t below = target - *previous;
            if (below <= above) {
                selected = previous;
            }
        }
        starts[rank] = static_cast<index_t>(selected - rowDelimiters.begin());
    }
    starts[ranks] = numRows;
    return starts;
}

// A compiler memory barrier keeps the default benchmark path from being
// optimized away when neither result printing nor validation is requested.
inline void preserveBenchmarkWork(const void* output) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(output) : "memory");
#else
    (void)output;
#endif
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            showHelp = true;
        } else {
            parseError = true;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    const nnz_t matrixEntries = static_cast<nnz_t>(numRows) * numRows;
    if (sparsity == 0 || numRows >= static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Matrix dimensions and sparsity must be representable by MPI.\n");
        }
        MPI_Finalize();
        return 1;
    }
    const nnz_t nItems = matrixEntries / sparsity;
    if (nItems > static_cast<nnz_t>(std::numeric_limits<size_t>::max()) ||
        nItems > static_cast<nnz_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("The number of non-zero elements exceeds this MPI implementation's collective count limit.\n");
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<double> globalVal;
    std::vector<index_t> globalCols;
    std::vector<nnz_t> globalRowDelimiters;
    std::vector<double> inputVector(static_cast<size_t>(numRows));
    std::vector<double> reference;
    std::vector<index_t> rowStarts;
    std::vector<int> mpiRowCounts;
    std::vector<int> mpiRowDisplacements;
    std::vector<int> mpiRowPointerCounts;
    std::vector<int> mpiRowPointerDisplacements;
    std::vector<int> mpiNnzCounts;
    std::vector<int> mpiNnzDisplacements;

    if (rank == 0) {
        const double sparsePercent = matrixEntries == 0
                                         ? 100.0
                                         : 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries);
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %llu (%.2f%% sparse)\n",
               static_cast<unsigned long long>(nItems), sparsePercent);
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");

        globalVal.resize(static_cast<size_t>(nItems));
        globalCols.resize(static_cast<size_t>(nItems));
        globalRowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(inputVector.data(), inputVector.size(), maxVal);
        fill(globalVal.data(), globalVal.size(), maxVal);
        initRandomMatrix(globalCols.data(), globalRowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(static_cast<size_t>(numRows));
            spmvCpu(globalVal.data(), globalCols.data(), globalRowDelimiters.data(), inputVector.data(),
                    reference.size(), reference.data());
        }

        rowStarts = partitionRows(globalRowDelimiters, ranks);
        mpiRowCounts.resize(ranks);
        mpiRowDisplacements.resize(ranks);
        mpiRowPointerCounts.resize(ranks);
        mpiRowPointerDisplacements.resize(ranks);
        mpiNnzCounts.resize(ranks);
        mpiNnzDisplacements.resize(ranks);
        for (int process = 0; process < ranks; ++process) {
            const index_t firstRow = rowStarts[process];
            const index_t lastRow = rowStarts[process + 1];
            const nnz_t localNnz = globalRowDelimiters[lastRow] - globalRowDelimiters[firstRow];
            const nnz_t localRows = static_cast<nnz_t>(lastRow) - firstRow;

            mpiRowCounts[process] = static_cast<int>(localRows);
            mpiRowDisplacements[process] = static_cast<int>(firstRow);
            mpiRowPointerCounts[process] = static_cast<int>(localRows + 1);
            mpiRowPointerDisplacements[process] = static_cast<int>(firstRow);
            mpiNnzCounts[process] = static_cast<int>(localNnz);
            mpiNnzDisplacements[process] = static_cast<int>(globalRowDelimiters[firstRow]);
        }
    }

    int localRowsInt = 0;
    MPI_Scatter(rank == 0 ? mpiRowCounts.data() : nullptr, 1, MPI_INT,
                &localRowsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const size_t localRows = static_cast<size_t>(localRowsInt);

    std::vector<nnz_t> localRowDelimiters(localRows + 1);
    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rank == 0 ? mpiRowPointerCounts.data() : nullptr,
                 rank == 0 ? mpiRowPointerDisplacements.data() : nullptr, MPI_UINT64_T,
                 localRowDelimiters.data(), localRowsInt + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    const nnz_t localNnz = localRowDelimiters.back() - localRowDelimiters.front();
    const nnz_t localNnzStart = localRowDelimiters.front();
    for (nnz_t& offset : localRowDelimiters) {
        offset -= localNnzStart;
    }

    std::vector<double> localVal(static_cast<size_t>(localNnz));
    std::vector<index_t> localCols(static_cast<size_t>(localNnz));
    std::vector<double> localOut(localRows);

    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr,
                 rank == 0 ? mpiNnzCounts.data() : nullptr,
                 rank == 0 ? mpiNnzDisplacements.data() : nullptr, MPI_DOUBLE,
                 localVal.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr,
                 rank == 0 ? mpiNnzCounts.data() : nullptr,
                 rank == 0 ? mpiNnzDisplacements.data() : nullptr, MPI_UINT32_T,
                 localCols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // The vector is tiny relative to the sparse matrix for typical SpMV cases;
    // replicating it removes all communication from the performance-critical loop.
    MPI_Bcast(inputVector.data(), static_cast<int>(inputVector.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<double>().swap(globalVal);
        std::vector<index_t>().swap(globalCols);
        std::vector<nnz_t>().swap(globalRowDelimiters);
    }

    if (rank == 0) {
        printf("Computing SpMV...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(), inputVector.data(), localRows,
                localOut.data());
        preserveBenchmarkWork(localOut.data());
    }
    const double localElapsedSeconds = MPI_Wtime() - startTime;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = elapsedSeconds * 1000.0;
        const double operations = 2.0 * static_cast<double>(nItems) * iterations;
        const double gflops = elapsedSeconds > 0.0 ? operations / elapsedSeconds / 1.0e9 : 0.0;
        const double avgTime = iterations > 0 ? durationMs / iterations : 0.0;
        printf("Computation time: %.3f ms\n", durationMs);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOut;
    if (printResults || validate) {
        if (rank == 0) {
            globalOut.resize(static_cast<size_t>(numRows));
        }
        MPI_Gatherv(localOut.data(), localRowsInt, MPI_DOUBLE,
                    rank == 0 ? globalOut.data() : nullptr,
                    rank == 0 ? mpiRowCounts.data() : nullptr,
                    rank == 0 ? mpiRowDisplacements.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0 && printResults) {
        print_results(globalOut, "OutputVector");
    }
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        if (verifyResults(reference.data(), globalOut.data(), globalOut.size())) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
