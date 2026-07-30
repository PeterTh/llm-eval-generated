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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Parse command line arguments on rank 0
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    int exitCode = 0;
    if (rank == 0) {
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
                printUsage(argv[0]);
                exitCode = -1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
    }

    // Broadcast exit code so all ranks exit together on -h or error
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitCode != 0) {
        MPI_Finalize();
        return (exitCode == -1) ? 0 : exitCode;
    }

    // Broadcast parameters to all ranks
    {
        uint32_t buf[3] = {numRows, sparsity, iterations};
        MPI_Bcast(buf, 3, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        numRows = buf[0]; sparsity = buf[1]; iterations = buf[2];
    }
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    {
        int buf[2] = {validate ? 1 : 0, printResults ? 1 : 0};
        MPI_Bcast(buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
        validate = buf[0] != 0;
        printResults = buf[1] != 0;
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", nprocs);
    }

    // Compute row partitioning (deterministic block distribution, all ranks compute same)
    const index_t np = static_cast<index_t>(nprocs);
    const index_t baseRows = numRows / np;
    const index_t remRows = numRows % np;
    const index_t myRank = static_cast<index_t>(rank);

    index_t local_nrows = baseRows + (myRank < remRows ? 1 : 0);

    // Rank 0 initializes all data structures
    std::vector<double> h_val, h_vec;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_reference;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);
        h_vec.resize(numRows);

        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // Compute reference solution on rank 0 before distributing data
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // All ranks allocate the dense vector buffer for the broadcast
    h_vec.resize(numRows);
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 prepares Scatterv metadata and adjusted row delimiters
    std::vector<int> sendcounts_val, displs_val;
    std::vector<int> sendcounts_rd, displs_rd;
    std::vector<index_t> allAdjRowDelims;

    if (rank == 0) {
        sendcounts_val.resize(nprocs);
        displs_val.resize(nprocs);
        sendcounts_rd.resize(nprocs);
        displs_rd.resize(nprocs);
        allAdjRowDelims.resize(numRows + nprocs);

        index_t rs = 0;
        int rdOff = 0;
        for (int r = 0; r < nprocs; ++r) {
            index_t lr = baseRows + (static_cast<index_t>(r) < remRows ? 1 : 0);
            index_t re = rs + lr;

            // Non-zero values and column indices are contiguous per row block
            sendcounts_val[r] = static_cast<int>(h_rowDelimiters[re] - h_rowDelimiters[rs]);
            displs_val[r] = static_cast<int>(h_rowDelimiters[rs]);

            // Adjusted row delimiters: local_nrows + 1 entries per rank
            sendcounts_rd[r] = static_cast<int>(lr) + 1;
            displs_rd[r] = rdOff;

            index_t baseD = h_rowDelimiters[rs];
            for (index_t i = 0; i <= lr; ++i) {
                allAdjRowDelims[rdOff + i] = h_rowDelimiters[rs + i] - baseD;
            }
            rdOff += static_cast<int>(lr) + 1;
            rs = re;
        }
    }

    // Scatter local_nnz to each rank so they can allocate receive buffers
    int local_nnz = 0;
    MPI_Scatter(rank == 0 ? sendcounts_val.data() : nullptr, 1, MPI_INT,
                &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local data structures on each rank
    std::vector<double> local_val(local_nnz);
    std::vector<index_t> local_cols(local_nnz);
    std::vector<index_t> local_rowDelims(local_nrows + 1);
    std::vector<double> local_out(local_nrows, 0.0);

    // Scatter CSR data to all ranks
    MPI_Scatterv(rank == 0 ? h_val.data() : nullptr,
                 rank == 0 ? sendcounts_val.data() : nullptr,
                 rank == 0 ? displs_val.data() : nullptr,
                 MPI_DOUBLE,
                 local_val.data(), local_nnz, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? h_cols.data() : nullptr,
                 rank == 0 ? sendcounts_val.data() : nullptr,
                 rank == 0 ? displs_val.data() : nullptr,
                 MPI_UINT32_T,
                 local_cols.data(), local_nnz, MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? allAdjRowDelims.data() : nullptr,
                 rank == 0 ? sendcounts_rd.data() : nullptr,
                 rank == 0 ? displs_rd.data() : nullptr,
                 MPI_UINT32_T,
                 local_rowDelims.data(), static_cast<int>(local_nrows + 1), MPI_UINT32_T,
                 0, MPI_COMM_WORLD);

    // Free rank 0's full data structures (no longer needed)
    if (rank == 0) {
        h_val.clear(); h_val.shrink_to_fit();
        h_cols.clear(); h_cols.shrink_to_fit();
        h_rowDelimiters.clear(); h_rowDelimiters.shrink_to_fit();
        allAdjRowDelims.clear(); allAdjRowDelims.shrink_to_fit();
        sendcounts_val.clear(); displs_val.clear();
        sendcounts_rd.clear(); displs_rd.clear();
    }

    // Perform parallel SpMV computation
    if (rank == 0) {
        printf("Computing SpMV...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(local_val.data(), local_cols.data(), local_rowDelims.data(),
                h_vec.data(), local_nrows, local_out.data());
    }

    double tEnd = MPI_Wtime();
    double localTime = tEnd - tStart;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results on rank 0
    std::vector<int> recvcounts_out, displs_out;
    std::vector<double> h_out;

    if (rank == 0) {
        h_out.resize(numRows);
        recvcounts_out.resize(nprocs);
        displs_out.resize(nprocs);

        index_t rs = 0;
        for (int r = 0; r < nprocs; ++r) {
            index_t lr = baseRows + (static_cast<index_t>(r) < remRows ? 1 : 0);
            recvcounts_out[r] = static_cast<int>(lr);
            displs_out[r] = static_cast<int>(rs);
            rs += lr;
        }
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(local_nrows), MPI_DOUBLE,
                rank == 0 ? h_out.data() : nullptr,
                rank == 0 ? recvcounts_out.data() : nullptr,
                rank == 0 ? displs_out.data() : nullptr,
                MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 reports timing, performance, and validation
    if (rank == 0) {
        long long durationMs = static_cast<long long>(maxTime * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / maxTime / 1e9;
        const double avgTime = maxTime * 1000.0 / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
