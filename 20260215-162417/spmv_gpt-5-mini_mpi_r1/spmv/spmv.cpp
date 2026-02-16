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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints usage/errors)
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (worldRank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n", 
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Root initializes full data structures and distributes to workers
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out;
    std::vector<double> h_reference;

    if (worldRank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(numRows + 1);

        if (worldRank == 0) printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        // For validation, compute reference solution
        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                    h_vec.data(), numRows, h_reference.data());
        }
    }

    // Broadcast vector to all ranks
    MPI_Bcast(h_vec.data(), static_cast<int>(numRows), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local row partition
    int r = worldRank;
    int p = worldSize;
    index_t rowStart = (static_cast<index_t>(r) * numRows) / static_cast<index_t>(p);
    index_t rowEnd = (static_cast<index_t>(r + 1) * numRows) / static_cast<index_t>(p);
    index_t localRows = rowEnd - rowStart;

    // Determine nnz range for this rank (root computes and sends pieces)
    index_t localNnz = 0;
    std::vector<double> local_vals;
    std::vector<index_t> local_cols;
    std::vector<index_t> local_rowDelimiters(localRows + 1);

    if (worldRank == 0) {
        for (int peer = 0; peer < p; ++peer) {
            index_t peerRowStart = (static_cast<index_t>(peer) * numRows) / static_cast<index_t>(p);
            index_t peerRowEnd = (static_cast<index_t>(peer + 1) * numRows) / static_cast<index_t>(p);
            index_t peerRows = peerRowEnd - peerRowStart;
            index_t peerStartNnz = h_rowDelimiters[peerRowStart];
            index_t peerEndNnz = h_rowDelimiters[peerRowEnd];
            index_t peerNnz = (peerEndNnz > peerStartNnz) ? (peerEndNnz - peerStartNnz) : 0;

            if (peer == 0) {
                localNnz = peerNnz;
                local_vals.resize(localNnz);
                local_cols.resize(localNnz);
                // copy values
                if (localNnz > 0) {
                    std::memcpy(local_vals.data(), h_val.data() + peerStartNnz, sizeof(double) * localNnz);
                    std::memcpy(local_cols.data(), h_cols.data() + peerStartNnz, sizeof(index_t) * localNnz);
                }
                for (index_t i = 0; i <= localRows; ++i) {
                    local_rowDelimiters[i] = h_rowDelimiters[peerRowStart + i] - peerStartNnz;
                }
            } else {
                // send sizes
                uint64_t sendNnz = static_cast<uint64_t>(peerNnz);
                MPI_Send(&sendNnz, 1, MPI_UNSIGNED_LONG_LONG, peer, 0, MPI_COMM_WORLD);
                // send row delimiters
                if (peerRows > 0) {
                    // create temporary adjusted row delimiters
                    std::vector<index_t> tmpRow(peerRows + 1);
                    for (index_t i = 0; i <= peerRows; ++i) tmpRow[i] = h_rowDelimiters[peerRowStart + i] - peerStartNnz;
                    MPI_Send(tmpRow.data(), static_cast<int>(peerRows + 1), MPI_UINT32_T, peer, 1, MPI_COMM_WORLD);
                } else {
                    // send empty row delimiters
                    MPI_Send(nullptr, 0, MPI_UINT32_T, peer, 1, MPI_COMM_WORLD);
                }

                // send cols and vals if any
                if (peerNnz > 0) {
                    MPI_Send(h_cols.data() + peerStartNnz, static_cast<int>(peerNnz), MPI_UINT32_T, peer, 2, MPI_COMM_WORLD);
                    MPI_Send(h_val.data() + peerStartNnz, static_cast<int>(peerNnz), MPI_DOUBLE, peer, 3, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        // receive nnz
        uint64_t recvNnz = 0;
        MPI_Recv(&recvNnz, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        localNnz = static_cast<index_t>(recvNnz);
        local_vals.resize(localNnz);
        local_cols.resize(localNnz);
        // receive row delimiters
        if (localRows > 0) {
            MPI_Recv(local_rowDelimiters.data(), static_cast<int>(localRows + 1), MPI_UINT32_T, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // receive cols and vals
        if (localNnz > 0) {
            MPI_Recv(local_cols.data(), static_cast<int>(localNnz), MPI_UINT32_T, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_vals.data(), static_cast<int>(localNnz), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Local output buffer
    std::vector<double> local_out(localRows);

    // Warm-up barrier
    MPI_Barrier(MPI_COMM_WORLD);
    double tStart = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        // perform local SpMV
        spmvCpu(local_vals.data(), local_cols.data(), local_rowDelimiters.data(), h_vec.data(), localRows, local_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tEnd = MPI_Wtime();
    double localDuration = tEnd - tStart;

    // Reduce to get max duration
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to root
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (worldRank == 0) {
        h_out.resize(numRows);
        recvcounts.resize(p);
        displs.resize(p);
        for (int peer = 0; peer < p; ++peer) {
            index_t peerRowStart = (static_cast<index_t>(peer) * numRows) / static_cast<index_t>(p);
            index_t peerRowEnd = (static_cast<index_t>(peer + 1) * numRows) / static_cast<index_t>(p);
            recvcounts[peer] = static_cast<int>(peerRowEnd - peerRowStart);
            displs[peer] = static_cast<int>(peerRowStart);
        }
    }

    MPI_Gatherv(local_out.data(), static_cast<int>(localRows), MPI_DOUBLE,
                (worldRank == 0) ? h_out.data() : nullptr, (worldRank == 0) ? recvcounts.data() : nullptr,
                (worldRank == 0) ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        long duration_ms = static_cast<long>(maxDuration * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (maxDuration) / 1e9;
        const double avgTime = (maxDuration * 1000.0) / static_cast<double>(iterations);

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
