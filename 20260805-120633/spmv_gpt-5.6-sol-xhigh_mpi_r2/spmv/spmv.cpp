#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using index_t = std::uint32_t;

constexpr double MAX_RELATIVE_ERROR = 0.02;

#if defined(__GNUC__) || defined(__clang__)
#define SPMV_RESTRICT __restrict__
#else
#define SPMV_RESTRICT
#endif

// Initialize an array with the same random-number sequence as the original
// serial benchmark.
void fill(double* array, const index_t size, const double maxVal) {
    for (index_t i = 0; i < size; ++i) {
        array[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Build the original global CSR sparsity pattern on rank zero.  Initialization
// is outside the timed region; the resulting rows are distributed afterwards.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n,
                      const index_t dim) {
    index_t nnzAssigned = 0;
    const double prob = dim == 0
        ? 0.0
        : static_cast<double>(n) /
              (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);
    bool fillRemaining = false;
    const std::uint64_t entries = static_cast<std::uint64_t>(dim) * dim;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const std::uint64_t position = static_cast<std::uint64_t>(i) * dim + j;
            const std::uint64_t numEntriesLeft = entries - position;
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

// Compute the rank-local rows.  The local row delimiters start at zero, so the
// hot loop is identical to the original serial row kernel.
void spmvCpu(const double* SPMV_RESTRICT val,
             const index_t* SPMV_RESTRICT cols,
             const index_t* SPMV_RESTRICT rowDelimiters,
             const double* SPMV_RESTRICT vec, const index_t rows,
             double* SPMV_RESTRICT out) noexcept {
    for (index_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        const index_t end = rowDelimiters[i + 1];
        for (index_t j = rowDelimiters[i]; j < end; ++j) {
            sum += val[j] * vec[cols[j]];
        }
        out[i] = sum;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                            i, ref, res);
                return false;
            }
        } else {
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseIndex(const char* text, index_t& value) {
    if (text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<index_t>::max()) {
        return false;
    }
    value = static_cast<index_t>(parsed);
    return true;
}

bool parseDouble(const char* text, double& value) {
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    value = parsed;
    return true;
}

// Choose contiguous row boundaries whose cumulative nonzero counts are as
// close as possible to equal shares.  Contiguity permits a zero-copy gather of
// the final output while balancing the actual work rather than just row count.
std::vector<index_t> partitionRows(const std::vector<index_t>& rowDelimiters,
                                   const index_t rows, const index_t nonzeros,
                                   const int processCount) {
    std::vector<index_t> firstRow(static_cast<std::size_t>(processCount) + 1);
    firstRow.front() = 0;
    firstRow.back() = rows;

    if (nonzeros == 0) {
        for (int rank = 1; rank < processCount; ++rank) {
            firstRow[rank] = static_cast<index_t>(
                (static_cast<std::uint64_t>(rows) * rank) / processCount);
        }
        return firstRow;
    }

    for (int rank = 1; rank < processCount; ++rank) {
        const std::uint64_t target =
            (static_cast<std::uint64_t>(nonzeros) * rank) / processCount;
        const auto searchBegin = rowDelimiters.begin() + firstRow[rank - 1];
        const auto highIt = std::lower_bound(searchBegin, rowDelimiters.end(), target);
        index_t high = static_cast<index_t>(highIt - rowDelimiters.begin());
        if (high > rows) {
            high = rows;
        }
        index_t chosen = high;
        if (high > firstRow[rank - 1]) {
            const index_t low = high - 1;
            const std::uint64_t lowDistance = target - rowDelimiters[low];
            const std::uint64_t highDistance = rowDelimiters[high] - target;
            if (lowDistance <= highDistance) {
                chosen = low;
            }
        }
        firstRow[rank] = chosen;
    }
    return firstRow;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    int validate = 0;
    int printResults = 0;
    // -1 means continue, 0 means help/success, and 1 means an argument error.
    int startupStatus = -1;

    if (rank == 0) {
        for (int i = 1; i < argc && startupStatus == -1; ++i) {
            const char* option = argv[i];
            if (std::strcmp(option, "-n") == 0 || std::strcmp(option, "-s") == 0 ||
                std::strcmp(option, "-i") == 0) {
                index_t* destination = std::strcmp(option, "-n") == 0 ? &numRows
                    : (std::strcmp(option, "-s") == 0 ? &sparsity : &iterations);
                if (i + 1 >= argc || !parseIndex(argv[++i], *destination)) {
                    std::fprintf(stderr, "Invalid value for %s\n", option);
                    startupStatus = 1;
                }
            } else if (std::strcmp(option, "-m") == 0) {
                if (i + 1 >= argc || !parseDouble(argv[++i], maxVal)) {
                    std::fprintf(stderr, "Invalid value for -m\n");
                    startupStatus = 1;
                }
            } else if (std::strcmp(option, "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(option, "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(option, "-h") == 0) {
                printUsage(argv[0]);
                startupStatus = 0;
            } else {
                std::fprintf(stderr, "Unknown option: %s\n", option);
                startupStatus = 1;
            }
        }

        if (startupStatus == -1 && sparsity == 0) {
            std::fprintf(stderr, "Sparsity must be greater than zero\n");
            startupStatus = 1;
        }

        if (startupStatus == 1) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&startupStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (startupStatus != -1) {
        MPI_Finalize();
        return startupStatus;
    }

    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const std::uint64_t matrixEntries = static_cast<std::uint64_t>(numRows) * numRows;
    const std::uint64_t nItemsWide = matrixEntries / sparsity;
    // This implementation uses the benchmark's 32-bit CSR indices and the
    // widely supported MPI-3 collective interfaces, whose counts/displacements
    // are signed ints.
    if (nItemsWide > std::numeric_limits<index_t>::max() ||
        nItemsWide > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        numRows > static_cast<index_t>(std::numeric_limits<int>::max() - 1)) {
        if (rank == 0) {
            std::fprintf(stderr, "Problem size exceeds the supported 32-bit CSR/MPI range\n");
        }
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nItemsWide);

    std::vector<double> vector(numRows);
    std::vector<double> globalValues;
    std::vector<index_t> globalColumns;
    std::vector<index_t> globalRowDelimiters;
    std::vector<double> reference;

    if (rank == 0) {
        std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        std::printf("Matrix size: %u x %u\n", numRows, numRows);
        std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        const double sparsePercent = matrixEntries == 0
            ? 100.0
            : 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries);
        std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, sparsePercent);
        std::printf("Iterations: %u\n", iterations);
        std::printf("Max value: %.2f\n", maxVal);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d\n", processCount);
        std::printf("Initializing and distributing data structures...\n");

        globalValues.resize(nItems);
        globalColumns.resize(nItems);
        globalRowDelimiters.resize(static_cast<std::size_t>(numRows) + 1);

        // C specifies the initial rand() state as if srand(1) had been called.
        // Reset it explicitly because MPI_Init is allowed to initialize library
        // internals before the benchmark reaches this point.
        srand(1);
        fill(vector.data(), numRows, maxVal);
        fill(globalValues.data(), nItems, maxVal);
        initRandomMatrix(globalColumns.data(), globalRowDelimiters.data(), nItems, numRows);

        if (validate) {
            std::printf("Computing reference solution...\n");
            reference.resize(numRows);
            spmvCpu(globalValues.data(), globalColumns.data(), globalRowDelimiters.data(),
                    vector.data(), numRows, reference.data());
        }
    }

    MPI_Bcast(vector.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<index_t> firstRow(static_cast<std::size_t>(processCount) + 1);
    if (rank == 0) {
        firstRow = partitionRows(globalRowDelimiters, numRows, nItems, processCount);
    }
    MPI_Bcast(firstRow.data(), processCount + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<index_t> firstNnz(static_cast<std::size_t>(processCount) + 1);
    if (rank == 0) {
        for (int process = 0; process <= processCount; ++process) {
            firstNnz[process] = globalRowDelimiters[firstRow[process]];
        }
    }
    MPI_Bcast(firstNnz.data(), processCount + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    std::vector<int> rowCounts(processCount);
    std::vector<int> rowDisplacements(processCount);
    std::vector<int> nnzCounts(processCount);
    std::vector<int> nnzDisplacements(processCount);
    std::vector<int> outputCounts(processCount);
    std::vector<int> outputDisplacements(processCount);
    for (int process = 0; process < processCount; ++process) {
        const index_t processRows = firstRow[process + 1] - firstRow[process];
        rowCounts[process] = static_cast<int>(processRows + 1);
        rowDisplacements[process] = static_cast<int>(firstRow[process]);
        nnzCounts[process] = static_cast<int>(firstNnz[process + 1] - firstNnz[process]);
        nnzDisplacements[process] = static_cast<int>(firstNnz[process]);
        outputCounts[process] = static_cast<int>(processRows);
        outputDisplacements[process] = static_cast<int>(firstRow[process]);
    }

    const index_t localRows = firstRow[rank + 1] - firstRow[rank];
    const index_t localNnz = firstNnz[rank + 1] - firstNnz[rank];
    std::vector<index_t> localRowDelimiters(static_cast<std::size_t>(localRows) + 1);
    std::vector<index_t> localColumns(localNnz);
    std::vector<double> localValues(localNnz);
    std::vector<double> localOutput(localRows);

    MPI_Scatterv(rank == 0 ? globalRowDelimiters.data() : nullptr,
                 rowCounts.data(), rowDisplacements.data(), MPI_UINT32_T,
                 localRowDelimiters.data(), rowCounts[rank], MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalColumns.data() : nullptr,
                 nnzCounts.data(), nnzDisplacements.data(), MPI_UINT32_T,
                 localColumns.data(), nnzCounts[rank], MPI_UINT32_T,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalValues.data() : nullptr,
                 nnzCounts.data(), nnzDisplacements.data(), MPI_DOUBLE,
                 localValues.data(), nnzCounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    const index_t localBase = firstNnz[rank];
    for (index_t& delimiter : localRowDelimiters) {
        delimiter -= localBase;
    }

    // Release the global matrix before benchmarking; rank zero participates
    // with the same local-only representation as every other process.
    if (rank == 0) {
        std::vector<double>().swap(globalValues);
        std::vector<index_t>().swap(globalColumns);
        std::vector<index_t>().swap(globalRowDelimiters);
        std::printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localValues.data(), localColumns.data(), localRowDelimiters.data(),
                vector.data(), localRows, localOutput.data());
    }
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const double averageMs = iterations == 0
            ? 0.0 : elapsedSeconds * 1000.0 / iterations;
        const double gflops = elapsedSeconds == 0.0
            ? 0.0
            : (2.0 * static_cast<double>(nItems) * iterations) /
                  elapsedSeconds / 1.0e9;
        std::printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
        std::printf("Average time per iteration: %.3f ms\n", averageMs);
        std::printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> globalOutput;
    if (validate || printResults) {
        if (rank == 0) {
            globalOutput.resize(numRows);
        }
        MPI_Gatherv(localOutput.data(), outputCounts[rank], MPI_DOUBLE,
                    rank == 0 ? globalOutput.data() : nullptr,
                    outputCounts.data(), outputDisplacements.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int exitStatus = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(globalOutput, "OutputVector");
        }
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = verifyResults(reference.data(), globalOutput.data(), numRows);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitStatus = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
