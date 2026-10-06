#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cstdint>
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
void spmvCpu(const double* __restrict val, const index_t* __restrict cols,
             const index_t* __restrict rowDelimiters, const double* __restrict vec,
             const index_t dim, double* __restrict out) {
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

// ****************************************************************************
// MPI helpers: point-to-point transfers of large arrays in int-sized chunks
// ****************************************************************************
constexpr size_t MPI_CHUNK = size_t(1) << 28;

template <typename T>
MPI_Datatype mpiType();
template <>
MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template <>
MPI_Datatype mpiType<index_t>() { return MPI_UINT32_T; }

template <typename T>
void bcastLarge(T* data, size_t count, MPI_Comm comm) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        MPI_Bcast(data + off, c, mpiType<T>(), 0, comm);
    }
}

template <typename T>
void isendLarge(const T* data, size_t count, int dest, int tag, MPI_Comm comm,
                std::vector<MPI_Request>& reqs) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        reqs.emplace_back();
        MPI_Isend(data + off, c, mpiType<T>(), dest, tag, comm, &reqs.back());
    }
}

template <typename T>
void irecvLarge(T* data, size_t count, int src, int tag, MPI_Comm comm,
                std::vector<MPI_Request>& reqs) {
    for (size_t off = 0; off < count; off += MPI_CHUNK) {
        const int c = static_cast<int>(std::min(MPI_CHUNK, count - off));
        reqs.emplace_back();
        MPI_Irecv(data + off, c, mpiType<T>(), src, tag, comm, &reqs.back());
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identically on every rank)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    if (root) {
        printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
        printf("Matrix size: %u x %u\n", numRows, numRows);
        printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
        printf("Non-zero elements: %u (%.2f%% sparse)\n",
               nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
        printf("Iterations: %u\n", iterations);
        printf("Max value: %.2f\n", maxVal);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Global data: the dense vector and row delimiters are replicated on all
    // ranks; the full matrix (values/columns) only exists on the root.
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_val;                          // Non-zero values (root)
    std::vector<index_t> h_cols;                        // Column indices (root)
    std::vector<double> h_out;                          // Output vector (root)

    if (root) {
        printf("Initializing data structures...\n");
        h_val.resize(nItems);
        h_cols.resize(nItems);
        h_out.resize(numRows);
        // Same sequence of rand() calls as the sequential version
        fill(h_vec.data(), numRows, maxVal);
        fill(h_val.data(), nItems, maxVal);
        initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);
    }

    bcastLarge(h_vec.data(), h_vec.size(), MPI_COMM_WORLD);
    bcastLarge(h_rowDelimiters.data(), h_rowDelimiters.size(), MPI_COMM_WORLD);

    // Row-block partition balanced by number of non-zeros
    std::vector<index_t> rowStart(nprocs + 1);
    rowStart[0] = 0;
    rowStart[nprocs] = numRows;
    for (int p = 1; p < nprocs; ++p) {
        const uint64_t target = (static_cast<uint64_t>(nItems) * p) / nprocs;
        const auto it = std::lower_bound(h_rowDelimiters.begin(), h_rowDelimiters.begin() + numRows,
                                         target, [](index_t a, uint64_t b) { return a < b; });
        rowStart[p] = std::max(rowStart[p - 1], static_cast<index_t>(it - h_rowDelimiters.begin()));
    }

    const index_t myFirstRow = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myFirstRow;
    const index_t myNzBegin = h_rowDelimiters[myFirstRow];
    const size_t myNnz = static_cast<size_t>(h_rowDelimiters[rowStart[rank + 1]]) - myNzBegin;

    // Local CSR block with zero-based row delimiters
    std::vector<index_t> l_rowDelimiters(myRows + 1);
    for (index_t i = 0; i <= myRows; ++i) {
        l_rowDelimiters[i] = h_rowDelimiters[myFirstRow + i] - myNzBegin;
    }
    std::vector<double> l_val(myNnz);
    std::vector<index_t> l_cols(myNnz);
    std::vector<double> l_out(myRows);

    // Distribute matrix blocks from the root
    {
        std::vector<MPI_Request> reqs;
        if (root) {
            for (int p = 1; p < nprocs; ++p) {
                const index_t b = h_rowDelimiters[rowStart[p]];
                const size_t cnt = static_cast<size_t>(h_rowDelimiters[rowStart[p + 1]]) - b;
                isendLarge(h_val.data() + b, cnt, p, 0, MPI_COMM_WORLD, reqs);
                isendLarge(h_cols.data() + b, cnt, p, 1, MPI_COMM_WORLD, reqs);
            }
            std::copy_n(h_val.data() + myNzBegin, myNnz, l_val.data());
            std::copy_n(h_cols.data() + myNzBegin, myNnz, l_cols.data());
        } else {
            irecvLarge(l_val.data(), myNnz, 0, 0, MPI_COMM_WORLD, reqs);
            irecvLarge(l_cols.data(), myNnz, 0, 1, MPI_COMM_WORLD, reqs);
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // For validation, compute reference solution (sequentially on the root)
    std::vector<double> h_reference;
    if (validate && root) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }
    // Full matrix copies are no longer needed
    std::vector<double>().swap(h_val);
    std::vector<index_t>().swap(h_cols);

    // Gather layout for the output vector
    std::vector<int> recvCounts(nprocs), displs(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        recvCounts[p] = static_cast<int>(rowStart[p + 1] - rowStart[p]);
        displs[p] = static_cast<int>(rowStart[p]);
    }

    // Perform SpMV computation
    if (root) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(l_val.data(), l_cols.data(), l_rowDelimiters.data(),
                h_vec.data(), myRows, l_out.data());
    }
    // Assemble the full output vector on the root
    MPI_Gatherv(l_out.data(), static_cast<int>(myRows), MPI_DOUBLE,
                root ? h_out.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

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
                exitCode = 1;
            }
        }
        fflush(stdout);
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
