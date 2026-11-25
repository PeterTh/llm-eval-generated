#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    const uint64_t matrixSize = static_cast<uint64_t>(dim) * dim;
    const double prob = matrixSize == 0 ? 0.0 : static_cast<double>(n) / matrixSize;

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = matrixSize - (static_cast<uint64_t>(i) * dim + j);
            const index_t needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            const double randVal = static_cast<double>(rand()) / RAND_MAX;
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

namespace {

constexpr unsigned BLOCK_SIZE = 128;

// A row uses a power-of-two group of lanes within a warp. Adjacent lanes
// load adjacent CSR entries and reduce the partial sums in registers.
template <unsigned LanesPerRow>
__global__ void spmvKernel(const double* __restrict__ val,
                          const index_t* __restrict__ cols,
                          const index_t* __restrict__ rowDelimiters,
                          const double* __restrict__ vec, index_t rows,
                          double* __restrict__ out) {
    const unsigned lane = threadIdx.x % LanesPerRow;
    const index_t row = blockIdx.x * (BLOCK_SIZE / LanesPerRow) +
                        threadIdx.x / LanesPerRow;
    double sum = 0.0;
    if (row < rows) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        // Use a wide loop index so a full 32-bit CSR offset cannot wrap.
        for (size_t entry = static_cast<size_t>(begin) + lane; entry < end;
             entry += LanesPerRow) {
            sum += val[entry] * vec[cols[entry]];
        }
    }

    // Keep all warp lanes participating, including those past the final row.
    for (unsigned delta = LanesPerRow / 2; delta > 0; delta /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, delta, LanesPerRow);
    }
    if (lane == 0 && row < rows) {
        out[row] = sum;
    }
}

void checkCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)), "Allocating device memory");
        }
    }

    explicit DeviceBuffer(const std::vector<T>& source) : DeviceBuffer(source.size()) {
        if (!source.empty()) {
            checkCuda(cudaMemcpy(data, source.data(), source.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "Copying inputs to the GPU");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }

    T* data = nullptr;
};

float spmvCuda(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& rowDelimiters, const std::vector<double>& vec,
               index_t iterations, bool readOutput, std::vector<double>& out) {
    const index_t rows = static_cast<index_t>(vec.size());
    if (rows == 0 || iterations == 0) {
        return 0.0f;
    }

    const DeviceBuffer<double> d_val(val);
    const DeviceBuffer<index_t> d_cols(cols);
    const DeviceBuffer<index_t> d_rowDelimiters(rowDelimiters);
    const DeviceBuffer<double> d_vec(vec);
    const DeviceBuffer<double> d_out(rows);

    // Pack short rows together and use full warps for longer rows or smaller grids.
    const size_t entriesPerRow = val.size() / rows;
    unsigned lanes = 1;
    auto kernel = spmvKernel<1>;
    if (entriesPerRow >= 32 || (rows <= 2048 && entriesPerRow >= 16)) {
        lanes = 32;
        kernel = spmvKernel<32>;
    } else if (entriesPerRow >= 16) {
        lanes = 16;
        kernel = spmvKernel<16>;
    } else if (entriesPerRow >= 8) {
        lanes = 8;
        kernel = spmvKernel<8>;
    } else if (entriesPerRow >= 2) {
        lanes = 4;
        kernel = spmvKernel<4>;
    }
    const unsigned rowsPerBlock = BLOCK_SIZE / lanes;
    const unsigned blocks = (rows - 1) / rowsPerBlock + 1;
    const auto launch = [&]() {
        kernel<<<blocks, BLOCK_SIZE>>>(d_val.data, d_cols.data, d_rowDelimiters.data,
                                      d_vec.data, rows, d_out.data);
    };

    cudaEvent_t start;
    cudaEvent_t end;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&end), "Creating CUDA end event");

    launch();
    checkCuda(cudaGetLastError(), "Launching SpMV warm-up");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");
    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();
    }
    checkCuda(cudaGetLastError(), "Launching CUDA SpMV");
    checkCuda(cudaEventRecord(end), "Recording CUDA end event");
    checkCuda(cudaEventSynchronize(end), "Running CUDA SpMV");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "Timing CUDA SpMV");
    checkCuda(cudaEventDestroy(start), "Destroying CUDA start event");
    checkCuda(cudaEventDestroy(end), "Destroying CUDA end event");
    if (readOutput) {
        checkCuda(cudaMemcpy(out.data(), d_out.data, out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "Copying results from the GPU");
    }
    return milliseconds;
}

} // namespace

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
    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero\n");
        return 1;
    }
    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros = matrixSize / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Too many non-zero elements for 32-bit CSR offsets\n");
        return 1;
    }
    const auto nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n",
           nItems, matrixSize == 0 ? 100.0 :
               100.0 * (1.0 - static_cast<double>(nItems) / matrixSize));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(numRows) + 1);
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

    // Perform SpMV computation
    printf("Computing SpMV...\n");
    const double milliseconds = spmvCuda(h_val, h_cols, h_rowDelimiters, h_vec,
                                         iterations, validate || printResults, h_out);
    printf("Computation time: %.3f ms\n", milliseconds);

    // Calculate performance metrics
    const double gflops = milliseconds > 0.0 ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
    const double avgTime = iterations > 0 ? milliseconds / iterations : 0.0;

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
