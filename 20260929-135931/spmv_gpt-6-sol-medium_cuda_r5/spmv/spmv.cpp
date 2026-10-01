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

// A group of LANES threads handles one CSR row. Adjacent lanes read adjacent
// nonzeros, and the partial sums are reduced within the group.
template <int LANES>
__global__ void spmvCuda(const double* __restrict__ val,
                         const index_t* __restrict__ cols,
                         const index_t* __restrict__ rowDelimiters,
                         const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    constexpr int THREADS = 256;
    const index_t row = blockIdx.x * (THREADS / LANES) + threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    index_t begin = 0;
    index_t end = 0;
    if (row < dim) {
        begin = rowDelimiters[row];
        end = rowDelimiters[row + 1];
    }

    double sum = 0.0;
    for (index_t j = begin + lane; j < end; j += LANES) {
        sum += val[j] * vec[cols[j]];
    }
    for (int offset = LANES / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, LANES);
    }
    if (lane == 0 && row < dim) out[row] = sum;
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    cudaError_t allocate(size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&data),
                          (count ? count : 1) * sizeof(T));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

bool cudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

void launchSpmv(int lanes, unsigned int blocks, const double* val,
                const index_t* cols, const index_t* rowDelimiters,
                const double* vec, index_t dim, double* out) {
    switch (lanes) {
        case 4: spmvCuda<4><<<blocks, 256>>>(val, cols, rowDelimiters, vec, dim, out); break;
        case 8: spmvCuda<8><<<blocks, 256>>>(val, cols, rowDelimiters, vec, dim, out); break;
        case 16: spmvCuda<16><<<blocks, 256>>>(val, cols, rowDelimiters, vec, dim, out); break;
        default: spmvCuda<32><<<blocks, 256>>>(val, cols, rowDelimiters, vec, dim, out); break;
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

    // Keep the fixed matrix and vector on the GPU for all iterations.
    DeviceBuffer<double> d_val, d_vec, d_out;
    DeviceBuffer<index_t> d_cols, d_rowDelimiters;
    if (!cudaOk(d_val.allocate(nItems), "Allocating matrix values") ||
        !cudaOk(d_cols.allocate(nItems), "Allocating column indices") ||
        !cudaOk(d_rowDelimiters.allocate(static_cast<size_t>(numRows) + 1), "Allocating row delimiters") ||
        !cudaOk(d_vec.allocate(numRows), "Allocating vector") ||
        !cudaOk(d_out.allocate(numRows), "Allocating output")) return 1;

    if (!cudaOk(cudaMemcpy(d_val.data, h_val.data(), static_cast<size_t>(nItems) * sizeof(double), cudaMemcpyHostToDevice), "Copying matrix values") ||
        !cudaOk(cudaMemcpy(d_cols.data, h_cols.data(), static_cast<size_t>(nItems) * sizeof(index_t), cudaMemcpyHostToDevice), "Copying column indices") ||
        !cudaOk(cudaMemcpy(d_rowDelimiters.data, h_rowDelimiters.data(), (static_cast<size_t>(numRows) + 1) * sizeof(index_t), cudaMemcpyHostToDevice), "Copying row delimiters") ||
        !cudaOk(cudaMemcpy(d_vec.data, h_vec.data(), static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyHostToDevice), "Copying vector")) return 1;

    const double nonzerosPerRow = numRows ? static_cast<double>(nItems) / numRows : 0.0;
    const int lanes = nonzerosPerRow <= 4 ? 4 :
                      nonzerosPerRow <= 8 ? 8 :
                      nonzerosPerRow <= 16 ? 16 : 32;
    const unsigned int blocks = (static_cast<uint64_t>(numRows) + 256 / lanes - 1) / (256 / lanes);

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    cudaEvent_t start, end;
    if (!cudaOk(cudaEventCreate(&start), "Creating start event")) return 1;
    if (!cudaOk(cudaEventCreate(&end), "Creating end event")) {
        cudaEventDestroy(start);
        return 1;
    }
    if (!cudaOk(cudaEventRecord(start), "Recording start event")) return 1;

    for (index_t iter = 0; iter < iterations; ++iter) {
        if (blocks) launchSpmv(lanes, blocks, d_val.data, d_cols.data,
                              d_rowDelimiters.data, d_vec.data, numRows, d_out.data);
    }

    if (!cudaOk(cudaGetLastError(), "Launching SpMV kernel") ||
        !cudaOk(cudaEventRecord(end), "Recording end event") ||
        !cudaOk(cudaEventSynchronize(end), "Executing SpMV kernel")) return 1;
    float durationMs = 0.0f;
    if (!cudaOk(cudaEventElapsedTime(&durationMs, start, end), "Timing SpMV kernel")) return 1;
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    if (iterations && numRows &&
        !cudaOk(cudaMemcpy(h_out.data(), d_out.data, static_cast<size_t>(numRows) * sizeof(double), cudaMemcpyDeviceToHost), "Copying output")) return 1;

    printf("Computation time: %.3f ms\n", static_cast<double>(durationMs));
    
    // Calculate performance metrics
    const double gflops = durationMs > 0 ? (2.0 * nItems * iterations) / (durationMs * 1e6) : 0.0;
    const double avgTime = iterations ? durationMs / iterations : 0.0;
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
