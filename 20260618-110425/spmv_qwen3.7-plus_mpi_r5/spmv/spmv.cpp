#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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
            index_t numEntriesLeft = (dim * dim) - ((i * dim) + j);
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
//   rowDelimiters: array of size localNRows+1 holding indices to local rows
//   vec: dense vector of size globalDim to be used for multiplication
//   localNRows: number of rows owned by this process
//   out: output - result from the spmv calculation (local portion)
//
// ****************************************************************************
void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t localNRows, double* out) {
    for (index_t i = 0; i < localNRows; ++i) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResultsFlag = false;

    // Parse command line arguments on all ranks
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
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
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

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    // Compute row distribution: 1D block decomposition
    // Each rank gets a contiguous block of rows
    const index_t baseRowsPerProc = numRows / nprocs;
    const index_t extraRows = numRows % nprocs;

    // Compute local row count for this rank
    index_t localNRows;
    if (static_cast<index_t>(rank) < extraRows) {
        localNRows = baseRowsPerProc + 1;
    } else {
        localNRows = baseRowsPerProc;
    }
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Rank 0 generates all data, then distributes via Scatterv
    // We need to know nnz per rank for scattering val/cols arrays
    std::vector<int> nnzPerProc(nprocs, 0);
    std::vector<int> nnzDispls(nprocs, 0);

    // On rank 0, generate the full matrix structure to determine nnz per row
    std::vector<index_t> fullRowDelimiters;
    std::vector<double> fullVal;
    std::vector<index_t> fullCols;
    std::vector<double> fullVec;

    if (rank == 0) {
        fullRowDelimiters.resize(numRows + 1);
        fullCols.resize(nItems);
        fullVec.resize(numRows);
        fullVal.resize(nItems);

        fill(fullVec.data(), numRows, maxVal);
        fill(fullVal.data(), nItems, maxVal);
        initRandomMatrix(fullCols.data(), fullRowDelimiters.data(), nItems, numRows);

        // Compute nnz per process
        for (int p = 0; p < nprocs; ++p) {
            index_t pRowStart, pNRows;
            if (static_cast<index_t>(p) < extraRows) {
                pNRows = baseRowsPerProc + 1;
                pRowStart = static_cast<index_t>(p) * (baseRowsPerProc + 1);
            } else {
                pNRows = baseRowsPerProc;
                pRowStart = extraRows * (baseRowsPerProc + 1) +
                            (static_cast<index_t>(p) - extraRows) * baseRowsPerProc;
            }
            nnzPerProc[p] = static_cast<int>(fullRowDelimiters[pRowStart + pNRows] - fullRowDelimiters[pRowStart]);
        }
        nnzDispls[0] = 0;
        for (int p = 1; p < nprocs; ++p) {
            nnzDispls[p] = nnzDispls[p - 1] + nnzPerProc[p - 1];
        }
    }

    // Broadcast nnzPerProc and nnzDispls to all processes
    MPI_Bcast(nnzPerProc.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnzDispls.data(), nprocs, MPI_INT, 0, MPI_COMM_WORLD);

    // For validation, compute reference on rank 0 using the full data before scattering
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(fullVal.data(), fullCols.data(), fullRowDelimiters.data(),
                fullVec.data(), numRows, h_reference.data());
    }

    // Broadcast the dense vector to all processes
    std::vector<double> h_vec(numRows);
    if (rank == 0) {
        h_vec = fullVec;
    }
    MPI_Bcast(h_vec.data(), numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter val and cols arrays
    std::vector<double> localVal(localNRows > 0 ? nnzPerProc[rank] : 0);
    std::vector<index_t> localCols(localNRows > 0 ? nnzPerProc[rank] : 0);

    // We need rowDelimiters counts for Scatterv (each row has a delimiter entry)
    // But rowDelimiters is numRows+1 entries; we scatter numRows entries (one per row)
    // and handle the last delimiter locally.
    std::vector<int> rowsPerProc(nprocs, 0);
    std::vector<int> rowsDispls(nprocs, 0);
    for (int p = 0; p < nprocs; ++p) {
        if (static_cast<index_t>(p) < extraRows) {
            rowsPerProc[p] = static_cast<int>(baseRowsPerProc + 1);
        } else {
            rowsPerProc[p] = static_cast<int>(baseRowsPerProc);
        }
    }
    rowsDispls[0] = 0;
    for (int p = 1; p < nprocs; ++p) {
        rowsDispls[p] = rowsDispls[p - 1] + rowsPerProc[p - 1];
    }

    // Scatter val and cols using nnz-based counts/displacements
    // Use MPI_Scatterv with appropriate sendcounts
    // For Scatterv, sendcounts and displs are in terms of the datatype count
    // We need to convert index_t counts to byte-compatible MPI types
    // Use MPI_UINT32_T for index_t (uint32_t)
    MPI_Scatterv(fullVal.data(), nnzPerProc.data(), nnzDispls.data(), MPI_DOUBLE,
                 localVal.data(), nnzPerProc[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(fullCols.data(), nnzPerProc.data(), nnzDispls.data(), MPI_UINT32_T,
                 localCols.data(), nnzPerProc[rank], MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Scatter row delimiters: each rank gets localNRows entries from fullRowDelimiters
    // Then adjust to local indexing
    std::vector<index_t> localRowDelimiters(localNRows + 1);
    {
        // Scatter the first numRows entries (one per row)
        // Each rank gets localNRows entries starting at rowsDispls[rank]
        std::vector<index_t> tempDelimiters(localNRows);
        MPI_Scatterv(fullRowDelimiters.data(), rowsPerProc.data(), rowsDispls.data(), MPI_UINT32_T,
                     tempDelimiters.data(), rowsPerProc[rank], MPI_UINT32_T,
                     0, MPI_COMM_WORLD);

        // Adjust to local indexing: subtract the nnz offset for this rank
        const index_t nnzOffset = static_cast<index_t>(nnzDispls[rank]);
        for (index_t i = 0; i < localNRows; ++i) {
            localRowDelimiters[i] = tempDelimiters[i] - nnzOffset;
        }
        // Last delimiter = total local nnz
        localRowDelimiters[localNRows] = static_cast<index_t>(nnzPerProc[rank]);
    }

    // Free full data on rank 0
    if (rank == 0) {
        fullRowDelimiters.clear(); fullRowDelimiters.shrink_to_fit();
        fullCols.clear(); fullCols.shrink_to_fit();
        fullVal.clear(); fullVal.shrink_to_fit();
        fullVec.clear(); fullVec.shrink_to_fit();
    }

    // Local output vector
    std::vector<double> localOut(localNRows, 0.0);

    // Perform parallel SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRowDelimiters.data(),
                h_vec.data(), localNRows, localOut.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Gather timing: use the max time across all ranks
    double localDuration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    std::vector<double> h_out;
    if (rank == 0) {
        h_out.resize(numRows);
    }

    // Compute displacements for gathering output (in doubles)
    std::vector<int> outRowsPerProc(nprocs, 0);
    std::vector<int> outRowsDispls(nprocs, 0);
    for (int p = 0; p < nprocs; ++p) {
        if (static_cast<index_t>(p) < extraRows) {
            outRowsPerProc[p] = static_cast<int>(baseRowsPerProc + 1);
        } else {
            outRowsPerProc[p] = static_cast<int>(baseRowsPerProc);
        }
    }
    outRowsDispls[0] = 0;
    for (int p = 1; p < nprocs; ++p) {
        outRowsDispls[p] = outRowsDispls[p - 1] + outRowsPerProc[p - 1];
    }

    MPI_Gatherv(localOut.data(), static_cast<int>(localNRows), MPI_DOUBLE,
                h_out.data(), outRowsPerProc.data(), outRowsDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double durationMs = maxDuration * 1000.0;
        printf("Computation time: %.0f ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / maxDuration / 1e9;
        const double avgTime = durationMs / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResultsFlag) {
            print_results(h_out, "OutputVector");
        }

        // Validation
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
