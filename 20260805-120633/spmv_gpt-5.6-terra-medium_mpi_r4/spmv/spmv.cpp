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

void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = static_cast<double>(n) /
                        (static_cast<double>(dim) * static_cast<double>(dim));

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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    index_t numRows = 1024, sparsity = 10, iterations = 10;
    double maxVal = 1.0;
    bool validate = false, printResults = false;
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
            MPI_Finalize();
            return 0;
        } else {
            parseStatus = 1;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }
    if (sparsity == 0) {
        parseStatus = 1;
        if (rank == 0) printf("Sparsity must be greater than zero\n");
    }
    int globalParseStatus = 0;
    MPI_Allreduce(&parseStatus, &globalParseStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (globalParseStatus) {
        MPI_Finalize();
        return 1;
    }

    if (numRows > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix dimension is too large for MPI collectives\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = (numRows * numRows) / sparsity;
    if (nItems > static_cast<index_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Matrix has too many non-zero elements for MPI collectives\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> rowCounts(worldSize), rowDisplacements(worldSize);
    std::vector<int> nnzCounts(worldSize), nnzDisplacements(worldSize);
    for (int p = 0; p < worldSize; ++p) {
        const index_t firstRow = static_cast<index_t>((static_cast<uint64_t>(p) * numRows) / worldSize);
        const index_t lastRow = static_cast<index_t>((static_cast<uint64_t>(p + 1) * numRows) / worldSize);
        rowCounts[p] = static_cast<int>(lastRow - firstRow);
        rowDisplacements[p] = static_cast<int>(firstRow);
    }
    const index_t localRowCount = static_cast<index_t>(rowCounts[rank]);

    std::vector<double> globalVal;
    std::vector<index_t> globalCols, globalRows;
    std::vector<double> vec(numRows);
    std::vector<double> reference;
    if (rank == 0) {
        globalVal.resize(nItems);
        globalCols.resize(nItems);
        globalRows.resize(numRows + 1);
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
               100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\nMax value: %.2f\nValidation: %s\n", iterations, maxVal,
               validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
        fill(vec.data(), numRows, maxVal);
        fill(globalVal.data(), nItems, maxVal);
        initRandomMatrix(globalCols.data(), globalRows.data(), nItems, numRows);
        for (int p = 0; p < worldSize; ++p) {
            const int start = rowDisplacements[p];
            const int end = start + rowCounts[p];
            nnzDisplacements[p] = static_cast<int>(globalRows[start]);
            nnzCounts[p] = static_cast<int>(globalRows[end] - globalRows[start]);
        }
        if (validate) {
            printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(globalVal.data(), globalCols.data(), globalRows.data(), vec.data(), numRows,
                    reference.data());
        }
    }

    MPI_Bcast(vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDisplacements.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    const int localNnz = nnzCounts[rank];
    std::vector<double> localVal(localNnz), localOut(localRowCount);
    std::vector<index_t> localCols(localNnz), localRowOffsets(localRowCount + 1);

    // Row delimiters overlap by one entry between ranks; Scatterv permits this read-only overlap.
    std::vector<int> delimiterCounts(worldSize), delimiterDisplacements(worldSize);
    for (int p = 0; p < worldSize; ++p) {
        delimiterCounts[p] = rowCounts[p] + 1;
        delimiterDisplacements[p] = rowDisplacements[p];
    }
    MPI_Scatterv(rank == 0 ? globalRows.data() : nullptr, delimiterCounts.data(),
                 delimiterDisplacements.data(), MPI_UINT32_T, localRowOffsets.data(),
                 static_cast<int>(localRowCount + 1), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalVal.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_DOUBLE, localVal.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalCols.data() : nullptr, nnzCounts.data(), nnzDisplacements.data(),
                 MPI_UINT32_T, localCols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localOffset = localRowOffsets[0];
    for (index_t& offset : localRowOffsets) offset -= localOffset;

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowOffsets.data(), vec.data(), localRowCount,
                localOut.data());
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> output;
    if (printResults || validate) {
        if (rank == 0) output.resize(numRows);
        MPI_Gatherv(localOut.data(), static_cast<int>(localRowCount), MPI_DOUBLE,
                    rank == 0 ? output.data() : nullptr, rowCounts.data(), rowDisplacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int result = 0;
    if (rank == 0) {
        const double milliseconds = seconds * 1000.0;
        printf("Computation time: %.3f ms\n", milliseconds);
        printf("Average time per iteration: %.3f ms\n", milliseconds / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / seconds / 1e9);
        if (printResults) print_results(output, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            if (verifyResults(reference.data(), output.data(), numRows)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
