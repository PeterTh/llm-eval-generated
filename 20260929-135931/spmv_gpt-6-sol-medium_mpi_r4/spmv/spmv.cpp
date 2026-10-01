#include <algorithm>
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

// MPI counts are ints, so transfer large arrays in bounded pieces.
template <typename T>
void sendArray(const T* data, size_t count, MPI_Datatype type, int peer, int tag) {
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += std::min(chunk, count - offset)) {
        const int length = static_cast<int>(std::min(chunk, count - offset));
        MPI_Send(data + offset, length, type, peer, tag, MPI_COMM_WORLD);
    }
}

template <typename T>
void recvArray(T* data, size_t count, MPI_Datatype type, int peer, int tag) {
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += std::min(chunk, count - offset)) {
        const int length = static_cast<int>(std::min(chunk, count - offset));
        MPI_Recv(data + offset, length, type, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

template <typename T>
void broadcastArray(T* data, size_t count, MPI_Datatype type) {
    constexpr size_t chunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count; offset += std::min(chunk, count - offset)) {
        const int length = static_cast<int>(std::min(chunk, count - offset));
        MPI_Bcast(data + offset, length, type, 0, MPI_COMM_WORLD);
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
    }

    // Allocate and initialize data structures
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters;
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out;

    if (rank == 0) {
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        printf("Initializing data structures...\n");
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate && rank == 0) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(), 
                h_vec.data(), numRows, h_reference.data());
    }

    // Divide at row boundaries near equal cumulative nonzero counts. This
    // balances the work while preserving each row's original summation order.
    std::vector<index_t> boundaries;
    if (rank == 0) {
        boundaries.resize(static_cast<size_t>(ranks) + 1);
        boundaries.front() = 0;
        boundaries.back() = numRows;
        for (int p = 1; p < ranks; ++p) {
            if (nItems == 0) {
                boundaries[p] = static_cast<index_t>(static_cast<uint64_t>(numRows) * p / ranks);
            } else {
                const index_t target = static_cast<index_t>(static_cast<uint64_t>(nItems) * p / ranks);
                boundaries[p] = static_cast<index_t>(std::lower_bound(
                    h_rowDelimiters.begin() + boundaries[p - 1], h_rowDelimiters.end(), target)
                    - h_rowDelimiters.begin());
            }
        }
    }

    uint64_t metadata[4] = {};
    std::vector<double> localVal;
    std::vector<index_t> localCols, localRows;
    for (int p = 0; p < ranks; ++p) {
        if (rank == 0) {
            const index_t first = boundaries[p], last = boundaries[p + 1];
            const index_t firstNnz = h_rowDelimiters[first];
            const index_t lastNnz = h_rowDelimiters[last];
            metadata[0] = first;
            metadata[1] = last - first;
            metadata[2] = firstNnz;
            metadata[3] = lastNnz - firstNnz;
            if (p != 0) {
                MPI_Send(metadata, 4, MPI_UINT64_T, p, 0, MPI_COMM_WORLD);
                sendArray(h_rowDelimiters.data() + first, static_cast<size_t>(metadata[1]) + 1, MPI_UINT32_T, p, 1);
                if (metadata[3] != 0) {
                    sendArray(h_cols.data() + firstNnz, metadata[3], MPI_UINT32_T, p, 2);
                    sendArray(h_val.data() + firstNnz, metadata[3], MPI_DOUBLE, p, 3);
                }
            }
        }
        if (rank == p) {
            if (p != 0) MPI_Recv(metadata, 4, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localRows.resize(static_cast<size_t>(metadata[1]) + 1);
            localCols.resize(metadata[3]);
            localVal.resize(metadata[3]);
            if (p == 0) {
                std::copy_n(h_rowDelimiters.data(), localRows.size(), localRows.data());
                if (!localCols.empty()) {
                    std::copy_n(h_cols.data(), localCols.size(), localCols.data());
                    std::copy_n(h_val.data(), localVal.size(), localVal.data());
                }
            } else {
                recvArray(localRows.data(), localRows.size(), MPI_UINT32_T, 0, 1);
                recvArray(localCols.data(), localCols.size(), MPI_UINT32_T, 0, 2);
                recvArray(localVal.data(), localVal.size(), MPI_DOUBLE, 0, 3);
            }
            for (index_t& offset : localRows) offset -= static_cast<index_t>(metadata[2]);
        }
    }
    if (rank == 0) {
        std::vector<double>().swap(h_val);
        std::vector<index_t>().swap(h_cols);
        std::vector<index_t>().swap(h_rowDelimiters);
    }
    broadcastArray(h_vec.data(), h_vec.size(), MPI_DOUBLE);
    std::vector<double> localOut(localRows.size() - 1);

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(localVal.data(), localCols.data(), localRows.data(),
                h_vec.data(), static_cast<index_t>(localOut.size()), localOut.data());
    }

    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) h_out.resize(numRows);
    if (rank == 0) {
        std::copy(localOut.begin(), localOut.end(), h_out.begin());
        for (int p = 1; p < ranks; ++p) {
            recvArray(h_out.data() + boundaries[p], boundaries[p + 1] - boundaries[p], MPI_DOUBLE, p, 4);
        }
    } else {
        sendArray(localOut.data(), localOut.size(), MPI_DOUBLE, 0, 4);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));

        // Calculate performance metrics
        const double gflops = duration > 0.0 ? (2.0 * nItems * iterations) / duration / 1e9 : 0.0;
        const double avgTime = iterations ? duration * 1000.0 / iterations : 0.0;

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(h_out, "OutputVector");

        if (validate) {
            printf("Validating result...\n");
            const bool valid = verifyResults(h_reference.data(), h_out.data(), numRows);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
