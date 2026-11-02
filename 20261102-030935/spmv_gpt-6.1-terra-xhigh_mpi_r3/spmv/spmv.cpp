#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
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
            uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

// MPI's traditional count argument is an int. Chunk large transfers so the
// CSR index range is not artificially limited by that interface.
template <typename T>
void sendArray(const T* data, size_t count, MPI_Datatype type, int dest, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, type, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void receiveArray(T* data, size_t count, MPI_Datatype type, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

template <typename T>
void broadcastArray(T* data, size_t count, MPI_Datatype type) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, type, 0, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
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

    const uint64_t matrixSize = uint64_t(numRows) * numRows;
    if (sparsity == 0 || (matrixSize / sparsity) > std::numeric_limits<index_t>::max()) {
        if (rank == 0) fprintf(stderr, "Invalid sparsity or matrix exceeds the CSR index range\n");
        MPI_Finalize();
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixSize / sparsity);

    std::vector<double> h_val;
    std::vector<index_t> h_cols, h_rowDelimiters;
    std::vector<double> h_vec(numRows), h_reference;
    // Each entry stores the first and last row and nonzero of a rank.
    std::vector<index_t> partitions;
    if (rank == 0) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing data structures...\n");

        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

        if (validate) {
            printf("Computing reference solution...\n");
            h_reference.resize(numRows);
            spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                    h_vec.data(), numRows, h_reference.data());
        }

        // Split at row boundaries near equal nonzero counts. Empty matrices
        // are split by rows instead, and ranks may receive zero rows.
        std::vector<index_t> boundaries(static_cast<size_t>(ranks) + 1);
        boundaries[ranks] = numRows;
        for (int r = 1; r < ranks; ++r) {
            if (nItems == 0) {
                boundaries[r] = static_cast<index_t>(uint64_t(numRows) * r / ranks);
            } else {
                const index_t target = static_cast<index_t>(uint64_t(nItems) * r / ranks);
                boundaries[r] = static_cast<index_t>(std::lower_bound(
                    h_rowDelimiters.begin(), h_rowDelimiters.end(), target) - h_rowDelimiters.begin());
            }
        }
        partitions.resize(static_cast<size_t>(ranks) * 4);
        for (int r = 0; r < ranks; ++r) {
            const size_t k = static_cast<size_t>(r) * 4;
            partitions[k] = boundaries[r];
            partitions[k + 1] = boundaries[r + 1];
            partitions[k + 2] = h_rowDelimiters[boundaries[r]];
            partitions[k + 3] = h_rowDelimiters[boundaries[r + 1]];
        }
    }

    index_t local[4];
    MPI_Scatter(partitions.data(), 4, MPI_UINT32_T, local, 4, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    const index_t localRows = local[1] - local[0];
    const index_t localNnz = local[3] - local[2];
    broadcastArray(h_vec.data(), numRows, MPI_DOUBLE);
    if (rank == 0) {
        for (int r = 1; r < ranks; ++r) {
            const size_t k = static_cast<size_t>(r) * 4;
            const index_t firstRow = partitions[k], lastRow = partitions[k + 1];
            const index_t firstNnz = partitions[k + 2], lastNnz = partitions[k + 3];
            sendArray(h_rowDelimiters.data() + firstRow, size_t(lastRow - firstRow) + 1,
                      MPI_UINT32_T, r, 0);
            if (lastNnz != firstNnz) {
                sendArray(h_cols.data() + firstNnz, lastNnz - firstNnz, MPI_UINT32_T, r, 1);
                sendArray(h_val.data() + firstNnz, lastNnz - firstNnz, MPI_DOUBLE, r, 2);
            }
        }
        // Release the portions sent to other ranks before the timed section.
        if (ranks > 1) {
            std::vector<index_t>(h_rowDelimiters.begin(),
                                 h_rowDelimiters.begin() + static_cast<size_t>(localRows) + 1).swap(h_rowDelimiters);
            std::vector<index_t>(h_cols.begin(), h_cols.begin() + localNnz).swap(h_cols);
            std::vector<double>(h_val.begin(), h_val.begin() + localNnz).swap(h_val);
        }
    } else {
        h_rowDelimiters.resize(static_cast<size_t>(localRows) + 1);
        h_cols.resize(localNnz);
        h_val.resize(localNnz);
        receiveArray(h_rowDelimiters.data(), h_rowDelimiters.size(), MPI_UINT32_T, 0, 0);
        receiveArray(h_cols.data(), localNnz, MPI_UINT32_T, 0, 1);
        receiveArray(h_val.data(), localNnz, MPI_DOUBLE, 0, 2);
    }
    for (index_t& offset : h_rowDelimiters) offset -= local[2];
    std::vector<double> localOut(localRows);

    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), localRows, localOut.data());
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Average time per iteration: %.3f ms\n", duration * 1000.0 / iterations);
        printf("Performance: %.3f GFLOPS\n", (2.0 * nItems * iterations) / duration / 1e9);
    }

    std::vector<double> h_out;
    if (printResults || validate) {
        if (rank == 0) {
            h_out.resize(numRows);
            std::copy(localOut.begin(), localOut.end(), h_out.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t k = static_cast<size_t>(r) * 4;
                const index_t count = partitions[k + 1] - partitions[k];
                if (count != 0)
                    receiveArray(h_out.data() + partitions[k], count, MPI_DOUBLE, r, 3);
            }
        } else {
            sendArray(localOut.data(), localRows, MPI_DOUBLE, 0, 3);
        }
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results(h_out, "OutputVector");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
