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
bool verifyResults(const double* reference, const double* result, const index_t size, const index_t offset = 0) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            // For very small values, check absolute error
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i + offset, ref, res);
                return false;
            }
        } else {
            // Check relative error
            const double relError = std::abs((res - ref) / ref);
            if (relError > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e (rel error: %.10e)\n",
                       i + offset, ref, res, relError);
                return false;
            }
        }
    }
    return true;
}

// Send/recv large arrays in chunks so counts always fit in an int
template <typename T>
void sendLarge(const T* data, size_t count, int dest, int tag) {
    const size_t chunk = size_t(1) << 26;
    for (size_t off = 0; off < count; off += chunk) {
        const int c = static_cast<int>(std::min(chunk, count - off));
        MPI_Send(data + off, c * static_cast<int>(sizeof(T)), MPI_BYTE, dest, tag, MPI_COMM_WORLD);
    }
}

template <typename T>
void recvLarge(T* data, size_t count, int src, int tag) {
    const size_t chunk = size_t(1) << 26;
    for (size_t off = 0; off < count; off += chunk) {
        const int c = static_cast<int>(std::min(chunk, count - off));
        MPI_Recv(data + off, c * static_cast<int>(sizeof(T)), MPI_BYTE, src, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically)
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
        printf("MPI ranks: %d\n", nprocs);
    }

    // Rank 0 generates the global problem (the random sequence is inherently serial)
    std::vector<double> h_vec(numRows);  // Dense vector (replicated on all ranks)
    std::vector<double> g_val;
    std::vector<index_t> g_cols, g_rowDelimiters;
    std::vector<index_t> rowStart(nprocs + 1, 0);  // row partition boundaries

    if (rank == 0) {
        printf("Initializing data structures...\n");
        g_val.resize(nItems);
        g_cols.resize(nItems);
        g_rowDelimiters.resize(static_cast<size_t>(numRows) + 1);
        fill(h_vec.data(), numRows, maxVal);
        fill(g_val.data(), nItems, maxVal);
        initRandomMatrix(g_cols.data(), g_rowDelimiters.data(), nItems, numRows);

        // Partition rows so each rank gets a balanced number of nonzeros
        index_t r = 0;
        for (int p = 1; p < nprocs; ++p) {
            const uint64_t target = static_cast<uint64_t>(nItems) * p / nprocs;
            while (r < numRows && g_rowDelimiters[r] < target) ++r;
            rowStart[p] = r;
        }
        rowStart[nprocs] = numRows;
    }
    MPI_Bcast(rowStart.data(), nprocs + 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    {
        const size_t chunk = size_t(1) << 26;
        for (size_t off = 0; off < numRows; off += chunk) {
            const int c = static_cast<int>(std::min<size_t>(chunk, numRows - off));
            MPI_Bcast(h_vec.data() + off, c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    const index_t myRowBeg = rowStart[rank];
    const index_t myRows = rowStart[rank + 1] - myRowBeg;

    // Distribute local matrix slices
    std::vector<double> h_val;
    std::vector<index_t> h_cols;
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(myRows) + 1);
    std::vector<double> h_out(myRows);

    if (rank == 0) {
        for (int p = 1; p < nprocs; ++p) {
            const index_t rb = rowStart[p], re = rowStart[p + 1];
            const index_t nb = g_rowDelimiters[rb], ne = g_rowDelimiters[re];
            std::vector<index_t> rd(g_rowDelimiters.begin() + rb, g_rowDelimiters.begin() + re + 1);
            for (auto& x : rd) x -= nb;
            sendLarge(rd.data(), rd.size(), p, 0);
            sendLarge(g_val.data() + nb, ne - nb, p, 1);
            sendLarge(g_cols.data() + nb, ne - nb, p, 2);
        }
        const index_t ne = g_rowDelimiters[rowStart[1]];
        h_val.assign(g_val.begin(), g_val.begin() + ne);
        h_cols.assign(g_cols.begin(), g_cols.begin() + ne);
        std::copy(g_rowDelimiters.begin(), g_rowDelimiters.begin() + myRows + 1, h_rowDelimiters.begin());
        std::vector<double>().swap(g_val);
        std::vector<index_t>().swap(g_cols);
        std::vector<index_t>().swap(g_rowDelimiters);
    } else {
        recvLarge(h_rowDelimiters.data(), h_rowDelimiters.size(), 0, 0);
        const size_t localNnz = h_rowDelimiters[myRows];
        h_val.resize(localNnz);
        h_cols.resize(localNnz);
        recvLarge(h_val.data(), localNnz, 0, 1);
        recvLarge(h_cols.data(), localNnz, 0, 2);
    }

    // For validation, compute reference solution (for the local rows)
    std::vector<double> h_reference;
    if (validate) {
        if (rank == 0) printf("Computing reference solution...\n");
        h_reference.resize(myRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), myRows, h_reference.data());
    }

    // Perform SpMV computation
    if (rank == 0) printf("Computing SpMV...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), myRows, h_out.data());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(max_ms);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
        const double avgTime = duration.count() / static_cast<double>(iterations);

        printf("Average time per iteration: %.3f ms\n", avgTime);
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (gather the full output on rank 0)
    if (printResults) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            counts[p] = static_cast<int>(rowStart[p + 1] - rowStart[p]);
            displs[p] = static_cast<int>(rowStart[p]);
        }
        std::vector<double> full;
        if (rank == 0) full.resize(numRows);
        MPI_Gatherv(h_out.data(), static_cast<int>(myRows), MPI_DOUBLE, full.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "OutputVector");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const int localOk = verifyResults(h_reference.data(), h_out.data(), myRows, myRowBeg) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (rank == 0) printf(allOk ? "Validation: PASSED\n" : "Validation: FAILED\n");
        MPI_Finalize();
        return allOk ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
