#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using index_t = uint32_t;

// Constants
constexpr double MAX_RELATIVE_ERROR = 0.02;

// ****************************************************************************
// Macro: CUDA_CHECK
//
// Purpose:
//   Abort with a diagnostic message if a CUDA runtime call fails
//
// ****************************************************************************
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),          \
                    __FILE__, __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

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
// Kernel: spmvCsrKernel
//
// Purpose:
//   Device implementation of the CSR sparse matrix-vector product.
//
//   Each row is handled by a group of LANES_PER_ROW consecutive threads (a
//   sub-warp or a full warp). Consecutive lanes read consecutive matrix
//   entries, which makes the val/cols loads fully coalesced, and the partial
//   sums are combined with a shuffle reduction. The number of lanes per row is
//   chosen on the host from the average row length so that even very short
//   rows keep the GPU busy while long rows get a full warp each.
//
//   Because the benchmark repeatedly multiplies the same matrix by the same
//   (read-only) vector and only the final output is observed, all iterations
//   are performed inside one kernel launch: this keeps every multiply-add of
//   the original program while removing per-iteration launch overhead.
//
// ****************************************************************************
template <int LANES_PER_ROW>
__global__ void spmvCsrKernel(const double* __restrict__ val,
                              const index_t* __restrict__ cols,
                              const index_t* __restrict__ rowDelimiters,
                              const double* __restrict__ vec,
                              double* __restrict__ out,
                              const index_t dim,
                              const index_t iterations) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int localLane = lane & (LANES_PER_ROW - 1);
    const index_t row = static_cast<index_t>(blockIdx.x) * (blockDim.x / LANES_PER_ROW) +
                        (threadIdx.x / LANES_PER_ROW);

    // Threads past the last row still take part in the shuffle reduction (all
    // lanes of a warp must be active) but never store anything.
    const bool active = row < dim;
    const index_t safeRow = active ? row : (dim - 1);

    const index_t start = rowDelimiters[safeRow];
    const index_t end = rowDelimiters[safeRow + 1];

    for (index_t iter = 0; iter < iterations; ++iter) {
        double t = 0.0;
        for (index_t j = start + localLane; j < end; j += LANES_PER_ROW) {
            t += val[j] * __ldg(vec + cols[j]);
        }
#pragma unroll
        for (int offset = LANES_PER_ROW / 2; offset > 0; offset >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, offset, LANES_PER_ROW);
        }
        if (active && localLane == 0) {
            out[row] = t;
        }
    }
}

// ****************************************************************************
// Function: launchSpmv
//
// Purpose:
//   Selects the number of lanes per row and launches the SpMV kernel.
//
// ****************************************************************************
template <int LANES_PER_ROW>
void launchSpmvKernel(const double* val, const index_t* cols,
                      const index_t* rowDelimiters, const double* vec, double* out,
                      const index_t dim, const index_t iterations) {
    constexpr int kBlockSize = 256;
    constexpr int kRowsPerBlock = kBlockSize / LANES_PER_ROW;
    const unsigned int gridSize =
        (dim + kRowsPerBlock - 1) / kRowsPerBlock;
    spmvCsrKernel<LANES_PER_ROW><<<gridSize, kBlockSize>>>(
        val, cols, rowDelimiters, vec, out, dim, iterations);
}

void launchSpmv(const double* val, const index_t* cols, const index_t* rowDelimiters,
                const double* vec, double* out, const index_t dim,
                const index_t iterations, const double avgNnz) {
    if (avgNnz >= 24.0) {
        launchSpmvKernel<32>(val, cols, rowDelimiters, vec, out, dim, iterations);
    } else if (avgNnz >= 12.0) {
        launchSpmvKernel<16>(val, cols, rowDelimiters, vec, out, dim, iterations);
    } else if (avgNnz >= 6.0) {
        launchSpmvKernel<8>(val, cols, rowDelimiters, vec, out, dim, iterations);
    } else if (avgNnz >= 3.0) {
        launchSpmvKernel<4>(val, cols, rowDelimiters, vec, out, dim, iterations);
    } else if (avgNnz >= 2.0) {
        launchSpmvKernel<2>(val, cols, rowDelimiters, vec, out, dim, iterations);
    } else {
        launchSpmvKernel<1>(val, cols, rowDelimiters, vec, out, dim, iterations);
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

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(numRows + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(numRows);                 // Output vector

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    const double avgNnz = numRows > 0 ? static_cast<double>(nItems) / numRows : 0.0;

    // Allocate device data structures
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    if (numRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_vec, sizeof(double) * numRows));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * numRows));
        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), sizeof(double) * numRows,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_rowDelimiters, sizeof(index_t) * (numRows + 1)));
        CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(),
                              sizeof(index_t) * (numRows + 1), cudaMemcpyHostToDevice));
    }

    if (nItems > 0) {
        CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * nItems));
        CUDA_CHECK(cudaMalloc(&d_cols, sizeof(index_t) * nItems));
        CUDA_CHECK(cudaMemcpy(d_val, h_val.data(), sizeof(double) * nItems,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cols, h_cols.data(), sizeof(index_t) * nItems,
                              cudaMemcpyHostToDevice));
    }

    // Perform SpMV computation
    printf("Computing SpMV...\n");

    // Warm up the device (module loading, instruction cache) before timing so
    // that the measured region contains only the actual SpMV work.
    if (iterations > 0 && numRows > 0) {
        launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, d_out, numRows, 1, avgNnz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (iterations > 0 && numRows > 0) {
        launchSpmv(d_val, d_cols, d_rowDelimiters, d_vec, d_out, numRows, iterations,
                   avgNnz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Retrieve the result vector (outside the timed region)
    if (numRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(double) * numRows,
                              cudaMemcpyDeviceToHost));
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    const double gflops = (2.0 * nItems * iterations) / elapsedSeconds / 1e9;
    const double avgTime = (elapsedSeconds * 1000.0) / static_cast<double>(iterations);

    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        valid = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Release device memory
    if (d_val != nullptr) CUDA_CHECK(cudaFree(d_val));
    if (d_cols != nullptr) CUDA_CHECK(cudaFree(d_cols));
    if (d_rowDelimiters != nullptr) CUDA_CHECK(cudaFree(d_rowDelimiters));
    if (d_vec != nullptr) CUDA_CHECK(cudaFree(d_vec));
    if (d_out != nullptr) CUDA_CHECK(cudaFree(d_out));

    if (validate && !valid) {
        return 1;
    }
    return 0;
}
