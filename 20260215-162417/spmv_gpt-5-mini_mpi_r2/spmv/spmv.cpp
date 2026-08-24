#include <mpi.h>
#include <chrono>
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
// ****************************************************************************
void fill(double* A, const index_t n, const double maxVal) {
    for (index_t i = 0; i < n; ++i) {
        A[i] = maxVal * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// ****************************************************************************
// Function: initRandomMatrix
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));
    srand(8675309);
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
            index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            double randVal = static_cast<double>(rand()) / RAND_MAX;
            if ((nnzAssigned < n && randVal <= prob) || fillRemaining) {
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
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
// ****************************************************************************
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
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse, but root will perform init)
    if (world_rank == 0) {
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
                if (world_rank == 0) printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            }
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&numRows, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sparsity, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int validate_i = validate ? 1 : 0;
    int print_i = printResults ? 1 : 0;
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_i != 0);
    printResults = (print_i != 0);

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (world_rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");
    }

    // Root initializes full data and then scatters
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    if (world_rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_out.resize(numRows);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // Prepare per-rank row distribution (contiguous block of rows)
    std::vector<int> rows_per_rank(world_size);
    int base = numRows / world_size;
    int rem = numRows % world_size;
    for (int r = 0; r < world_size; ++r) rows_per_rank[r] = base + (r < rem ? 1 : 0);

    int localRows = rows_per_rank[world_rank];
    int rowStart = 0;
    for (int r = 0; r < world_rank; ++r) rowStart += rows_per_rank[r];

    // Root computes nnz counts and displacements for scattering vals and cols
    std::vector<int> nnzCounts(world_size, 0);
    std::vector<int> nnzDispls(world_size, 0);
    std::vector<int> rowCounts(world_size, 0);
    std::vector<int> rowDispls(world_size, 0);
    std::vector<index_t> rowDelimitersPacked; // packed adjusted row delimiters per rank
    if (world_rank == 0) {
        for (int r = 0, disp = 0, rowDisp = 0; r < world_size; ++r) {
            int rStart = 0;
            for (int j = 0; j < r; ++j) rStart += rows_per_rank[j];
            int rRows = rows_per_rank[r];
            int rEnd = rStart + rRows;
            index_t startNnz = h_rowDelimiters[rStart];
            index_t endNnz = h_rowDelimiters[rEnd];
            int localNnz = static_cast<int>(endNnz - startNnz);
            nnzCounts[r] = localNnz;
            nnzDispls[r] = disp;
            disp += localNnz;

            // pack adjusted row delimiters
            rowCounts[r] = rRows + 1;
            rowDispls[r] = rowDisp;
            for (int i = rStart; i < rEnd; ++i) {
                rowDelimitersPacked.push_back(static_cast<index_t>(h_rowDelimiters[i] - startNnz));
            }
            rowDelimitersPacked.push_back(static_cast<index_t>(endNnz - startNnz));
            rowDisp += rowCounts[r];
        }
    }

    // Each rank now knows localRows and rowStart. Compute local nnz by broadcasting nnzCounts or using scatterv's recvcount
    // Broadcast nnzCounts so each rank can allocate receive buffers
    MPI_Bcast(nnzCounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);

    int localNnz = nnzCounts[world_rank];
    std::vector<double> local_val(localNnz);
    std::vector<index_t> local_cols(localNnz);
    std::vector<index_t> local_rowDelimiters(localRows + 1);
    std::vector<double> local_out(localRows);

    // Scatter vals and cols
    MPI_Scatterv(h_val.empty() ? nullptr : h_val.data(), nnzCounts.data(), nnzDispls.data(), MPI_DOUBLE,
                 local_val.data(), localNnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(h_cols.empty() ? nullptr : h_cols.data(), nnzCounts.data(), nnzDispls.data(), MPI_UINT32_T,
                 local_cols.data(), localNnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // Scatter packed row delimiters
    if (world_rank == 0) {
        // rowCounts and rowDispls already set
    }
    MPI_Bcast(rowCounts.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rowDispls.data(), world_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        MPI_Scatterv(rowDelimitersPacked.data(), rowCounts.data(), rowDispls.data(), MPI_UINT32_T,
                     MPI_IN_PLACE, rowCounts[0], MPI_UINT32_T, 0, MPI_COMM_WORLD);
        // copy root portion from packed into local_rowDelimiters
        std::copy(rowDelimitersPacked.begin(), rowDelimitersPacked.begin() + rowCounts[0], local_rowDelimiters.begin());
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UINT32_T,
                     local_rowDelimiters.data(), localRows + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    // Broadcast the dense vector to all ranks
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // For validation, compute reference on root only (using full data)
    std::vector<double> h_reference;
    if (validate && world_rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), h_vec.data(), numRows, h_reference.data());
    }

    if (world_rank == 0) printf("Computing SpMV...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelimiters.data(), h_vec.data(), localRows, local_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double duration_sec = t1 - t0;
    double max_duration_sec = 0.0;
    MPI_Reduce(&duration_sec, &max_duration_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather local_out to root
    std::vector<int> rowsCountsInt(world_size);
    for (int r = 0; r < world_size; ++r) rowsCountsInt[r] = rows_per_rank[r];
    std::vector<int> rowsDisplsInt(world_size);
    for (int r = 1; r < world_size; ++r) rowsDisplsInt[r] = rowsDisplsInt[r-1] + rowsCountsInt[r-1];

    if (world_rank == 0) {
        // h_out already sized earlier
    }
    MPI_Gatherv(local_out.data(), localRows, MPI_DOUBLE,
                h_out.empty() ? nullptr : h_out.data(), rowsCountsInt.data(), rowsDisplsInt.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %.3f ms\n", max_duration_sec * 1000.0);
        const double gflops = (2.0 * static_cast<double>(nItems) * iterations) / (max_duration_sec) / 1e9;
        const double avgTime = (max_duration_sec * 1000.0) / static_cast<double>(iterations);
        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(h_out, "OutputVector");

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
