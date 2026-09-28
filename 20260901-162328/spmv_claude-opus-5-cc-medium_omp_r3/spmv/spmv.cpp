#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Function: allocFirstTouch
//
// Purpose:
//   Allocates an uninitialized array and zero-initializes it from all OpenMP
//   threads using the same static distribution that the compute kernel uses.
//   On NUMA systems this places each page on the node of the thread that will
//   later read/write it (first-touch policy), which is essential for scaling a
//   bandwidth-bound kernel such as SpMV across sockets.
//
// Arguments:
//   n: number of elements
//
// ****************************************************************************
template <typename T>
std::unique_ptr<T[]> allocFirstTouch(const size_t n) {
    std::unique_ptr<T[]> mem(new T[n]);
    T* __restrict__ p = mem.get();
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        p[i] = T{};
    }
    return mem;
}

// ****************************************************************************
// Function: partitionBoundary
//
// Purpose:
//   Computes the first row of partition `part` of a split of the matrix into
//   `nparts` chunks holding (as close as possible) an equal number of non-zero
//   elements each. Boundaries are found by binary search over the row
//   delimiters, so the partitioning is consistent between neighbouring threads
//   (thread t ends exactly where thread t+1 begins) and covers all rows.
//
// Arguments:
//   rowDelimiters: array of size dim+1 holding indices to rows
//   dim:           number of rows/columns in the matrix
//   part:          partition index in [0, nparts]
//   nparts:        total number of partitions
//
// ****************************************************************************
static index_t partitionBoundary(const index_t* rowDelimiters, const index_t dim, const int part,
                                 const int nparts) {
    if (part <= 0) {
        return 0;
    }
    if (part >= nparts) {
        return dim;
    }
    const uint64_t nnz = rowDelimiters[dim];
    const index_t target = static_cast<index_t>((nnz * static_cast<uint64_t>(part)) / static_cast<uint64_t>(nparts));
    return static_cast<index_t>(std::lower_bound(rowDelimiters, rowDelimiters + dim + 1, target) - rowDelimiters);
}

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
//   rowBegin/rowEnd: half-open range of rows handled by the calling thread
//
// Note:
//   The accumulation order within a row is unchanged with respect to the
//   original serial code, so results are bit-identical; rows are independent
//   and can be distributed freely across threads.
//
// ****************************************************************************
static inline void spmvCpuRows(const double* __restrict__ val, const index_t* __restrict__ cols,
                               const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                               const index_t rowBegin, const index_t rowEnd, double* __restrict__ out) {
    for (index_t i = rowBegin; i < rowEnd; ++i) {
        double t = 0.0;
        for (index_t j = rowDelimiters[i]; j < rowDelimiters[i + 1]; ++j) {
            const auto col = cols[j];
            t += val[j] * vec[col];
        }
        out[i] = t;
    }
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const index_t rowBegin = partitionBoundary(rowDelimiters, dim, tid, nthreads);
        const index_t rowEnd = partitionBoundary(rowDelimiters, dim, tid + 1, nthreads);
        spmvCpuRows(val, cols, rowDelimiters, vec, rowBegin, rowEnd, out);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Calculate number of non-zero elements
    const index_t nItems = (numRows * numRows) / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Allocate and initialize data structures. The buffers are touched in
    // parallel first so that their pages are distributed over the NUMA nodes
    // according to the access pattern of the kernel.
    auto h_val = allocFirstTouch<double>(nItems);              // Non-zero values
    auto h_cols = allocFirstTouch<index_t>(nItems);            // Column indices
    auto h_rowDelimiters = allocFirstTouch<index_t>(numRows + 1);  // Row delimiters
    auto h_vec = allocFirstTouch<double>(numRows);             // Dense vector
    auto h_out = allocFirstTouch<double>(numRows);             // Output vector

    printf("Initializing data structures...\n");
    fill(h_vec.get(), numRows, maxVal);
    fill(h_val.get(), nItems, maxVal);
    initRandomMatrix(h_cols.get(), h_rowDelimiters.get(), nItems, numRows);

    // For validation, compute reference solution
    std::unique_ptr<double[]> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference = allocFirstTouch<double>(numRows);
        spmvCpu(h_val.get(), h_cols.get(), h_rowDelimiters.get(),
                h_vec.get(), numRows, h_reference.get());
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // A single parallel region spans all iterations: the (nnz-balanced) row
    // partitioning is computed once and each thread keeps working on the same
    // rows, so no fork/join or repartitioning cost is paid per iteration.
    // Iterations are independent (vec is read-only, every out element is
    // written by the same thread every time), hence no barrier is required.
#pragma omp parallel
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const index_t rowBegin = partitionBoundary(h_rowDelimiters.get(), numRows, tid, nthreads);
        const index_t rowEnd = partitionBoundary(h_rowDelimiters.get(), numRows, tid + 1, nthreads);
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvCpuRows(h_val.get(), h_cols.get(), h_rowDelimiters.get(), h_vec.get(), rowBegin, rowEnd,
                        h_out.get());
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / (duration.count() / 1000.0) / 1e9;
    const double avgTime = duration.count() / static_cast<double>(iterations);
    
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> out(h_out.get(), h_out.get() + numRows);
        print_results(out, "OutputVector");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        const bool valid = verifyResults(h_reference.get(), h_out.get(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
