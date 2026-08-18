#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) / (static_cast<double>(dim) * dim);
    srand(8675309);

    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned++] = j;
            }
        }
    }
    rowDelimiters[dim] = n;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t i = 0; i < dim; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            t += val[j] * vec[cols[j]];
        }
        out[i] = t;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
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
    int exitCode = 0;
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

    if (sparsity == 0 || static_cast<unsigned long long>(numRows) * numRows >
                             std::numeric_limits<index_t>::max()) {
        if (rank == 0) printf("Matrix dimensions and sparsity must be valid 32-bit values.\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = (numRows * numRows) / sparsity;
    if (nItems > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix has too many non-zero entries for this MPI implementation.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("MPI processes: %d\n", ranks);
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
    }

    const index_t localFirstRow = static_cast<index_t>((static_cast<unsigned long long>(rank) * numRows) / ranks);
    const index_t localLastRow = static_cast<index_t>((static_cast<unsigned long long>(rank + 1) * numRows) / ranks);
    const index_t localRowCount = localLastRow - localFirstRow;
    std::vector<double> vec(numRows);
    std::vector<double> globalVal, reference, out;
    std::vector<index_t> globalCols, globalRowOffsets(numRows + 1);
    std::vector<int> nnzCounts(ranks), nnzDisplacements(ranks), rowCounts(ranks), rowDisplacements(ranks);

    if (rank == 0) {
        printf("Initializing data structures...\n");
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        fill(vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRowOffsets.data(), nItems, numRows);
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(globalVal.data(), globalCols.data(), globalRowOffsets.data(), vec.data(), numRows, reference.data());
        }
        for (int p = 0; p < ranks; ++p) {
            const index_t first = static_cast<index_t>((static_cast<unsigned long long>(p) * numRows) / ranks);
            const index_t last = static_cast<index_t>((static_cast<unsigned long long>(p + 1) * numRows) / ranks);
            nnzDisplacements[p] = static_cast<int>(globalRowOffsets[first]);
            nnzCounts[p] = static_cast<int>(globalRowOffsets[last] - globalRowOffsets[first]);
            rowDisplacements[p] = static_cast<int>(first);
            rowCounts[p] = static_cast<int>(last - first);
        }
    }

    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int localNnzInt = 0;
    MPI_Scatter(nnzCounts.data(), 1, MPI_INT, &localNnzInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const index_t localNnz = static_cast<index_t>(localNnzInt);
    std::vector<double> localVal(localNnz);
    std::vector<index_t> localCols(localNnz), localRowOffsets(localRowCount + 1);
    std::vector<double> localOut(localRowCount);
    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 localVal.data(), localNnzInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(), MPI_UNSIGNED,
                 localCols.data(), localNnzInt, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // The global CSR value/index arrays are only needed for setup on rank 0.
    globalVal.clear();
    globalVal.shrink_to_fit();
    globalCols.clear();
    globalCols.shrink_to_fit();
    MPI_Bcast(globalRowOffsets.data(), static_cast<int>(numRows + 1), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    const index_t localOffset = globalRowOffsets[localFirstRow];
    for (index_t row = 0; row <= localRowCount; ++row) {
        localRowOffsets[row] = globalRowOffsets[localFirstRow + row] - localOffset;
    }
    globalRowOffsets.clear();
    globalRowOffsets.shrink_to_fit();

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowOffsets.data(), vec.data(), localRowCount, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        if (rank == 0) out.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRowCount), MPI_DOUBLE,
                    rank == 0 ? out.data() : nullptr, rowCounts.data(), rowDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double avgTime = iterations == 0 ? 0.0 : duration * 1000.0 / iterations;
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) print_results(out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            exitCode = verifyResults(reference.data(), out.data(), numRows) ? 0 : 1;
            printf("Validation: %s\n", exitCode == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
