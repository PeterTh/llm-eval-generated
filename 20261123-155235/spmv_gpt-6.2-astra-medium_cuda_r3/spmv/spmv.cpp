#include <cuda_runtime.h>

#include <algorithm>
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
    double prob = dim ? static_cast<double>(n) / (static_cast<double>(dim) * dim) : 0.0;

    // glibc rand() uses the 128-byte random() generator. A private state
    // produces the same sequence without locking for every matrix entry.
#ifdef __GLIBC__
    random_data rng{};
    alignas(int32_t) char state[128];
    initstate_r(8675309, state, sizeof(state), &rng);
    auto draw = [&rng]() {
        int32_t value;
        random_r(&rng, &value);
        return value;
    };
#else
    srand(8675309);
    auto draw = []() { return rand(); };
#endif

    // Convert the original floating-point comparison into an inclusive integer
    // cutoff, checking the boundary with the original division and rounding.
    int cutoff = static_cast<int>(prob * RAND_MAX);
    if (cutoff < RAND_MAX && static_cast<double>(cutoff + 1) / RAND_MAX <= prob) ++cutoff;
    if (static_cast<double>(cutoff) / RAND_MAX > prob) --cutoff;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        const uint64_t remaining = static_cast<uint64_t>(dim - i) * dim;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j) {
            if (remaining - j == n - nnzAssigned || draw() <= cutoff) {
                cols[nnzAssigned++] = j;
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : bytes(count * sizeof(T)) {
        if (bytes) checkCuda(cudaMalloc(&data, bytes), "cudaMalloc");
    }
    explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
        if (bytes) checkCuda(cudaMemcpy(data, host.data(), bytes, cudaMemcpyHostToDevice),
                             "copy input to GPU");
    }
    ~DeviceArray() { if (data) cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data = nullptr;
private:
    size_t bytes;
};

constexpr unsigned sliceWidth = 32;

// Store each group of 32 rows column-wise. Adjacent lanes then read adjacent
// entries, while each lane visits its own row in the original CSR order.
__global__ void spmvKernel(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const size_t* __restrict__ slices,
                           const index_t* __restrict__ lengths,
                           const double* __restrict__ vector, index_t rows,
                           double* __restrict__ output) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    size_t offset = slices[row / sliceWidth] + row % sliceWidth;
    const index_t length = lengths[row];
    double sum = 0.0;
    #pragma unroll 4
    for (index_t j = 0; j < length; ++j, offset += sliceWidth) {
        sum += __dmul_rn(values[offset], vector[columns[offset]]);
    }
    output[row] = sum;
}

// With few rows, use a warp per row to expose more parallelism in the
// gathers and products. One lane sums the staged products in CSR order.
constexpr unsigned warpsPerBlock = 4;
constexpr unsigned tileEntries = 128;
__global__ void spmvWarpKernel(const double* __restrict__ values,
                               const index_t* __restrict__ columns,
                               const index_t* __restrict__ offsets,
                               const double* __restrict__ vector, index_t rows,
                               double* __restrict__ output) {
    __shared__ double products[warpsPerBlock][tileEntries];
    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    const index_t row = blockIdx.x * warpsPerBlock + warp;
    if (row >= rows) return;
    const size_t end = offsets[row + 1];
    double sum = 0.0;
    for (size_t base = offsets[row]; base < end; base += tileEntries) {
        const unsigned count = end - base < tileEntries ? end - base : tileEntries;
        for (unsigned j = lane; j < count; j += 32) {
            products[warp][j] = __dmul_rn(values[base + j], vector[columns[base + j]]);
        }
        __syncwarp();
        if (lane == 0) {
            for (unsigned j = 0; j < count; ++j) sum += products[warp][j];
        }
        __syncwarp();
    }
    if (lane == 0) output[row] = sum;
}

double runSpmv(const std::vector<double>& values, const std::vector<index_t>& columns,
               const std::vector<index_t>& offsets, const std::vector<double>& vector,
               std::vector<double>& output, index_t iterations) {
    const size_t rows = vector.size();
    const bool useWarp = rows && rows <= 4096 && values.size() / rows >= 8;
    std::vector<index_t> lengths;
    std::vector<size_t> slices;
    std::vector<double> packedValues;
    std::vector<index_t> packedColumns;
    if (!useWarp) {
        lengths.resize(rows);
        slices.resize((rows + sliceWidth - 1) / sliceWidth);
        size_t slots = 0;
        for (size_t slice = 0; slice < slices.size(); ++slice) {
            slices[slice] = slots;
            const size_t first = slice * sliceWidth;
            index_t longest = 0;
            for (size_t row = first; row < std::min(first + sliceWidth, rows); ++row) {
                lengths[row] = offsets[row + 1] - offsets[row];
                longest = std::max(longest, lengths[row]);
            }
            slots += static_cast<size_t>(longest) * sliceWidth;
        }

        packedValues.resize(slots);
        packedColumns.resize(slots);
        for (size_t row = 0; row < rows; ++row) {
            size_t dst = slices[row / sliceWidth] + row % sliceWidth;
            for (index_t j = offsets[row]; j < offsets[row + 1]; ++j, dst += sliceWidth) {
                packedValues[dst] = values[j];
                packedColumns[dst] = columns[j];
            }
        }
    }

    DeviceArray<double> dValues(useWarp ? values : packedValues), dVector(vector), dOutput(rows);
    DeviceArray<index_t> dColumns(useWarp ? columns : packedColumns), dRows(useWarp ? offsets : lengths);
    DeviceArray<size_t> dSlices(slices);
    // Resolve the CUDA module outside the timing interval, including any JIT.
    cudaFuncAttributes attributes;
    if (useWarp) checkCuda(cudaFuncGetAttributes(&attributes, spmvWarpKernel), "load CSR kernel");
    else checkCuda(cudaFuncGetAttributes(&attributes, spmvKernel), "load sliced kernel");

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&end), "create end event");
    checkCuda(cudaEventRecord(start), "record start event");
    constexpr unsigned threads = 64;
    const unsigned blocks = (rows + threads - 1) / threads;
    if (useWarp) {
        const unsigned warpBlocks = (rows + warpsPerBlock - 1) / warpsPerBlock;
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvWarpKernel<<<warpBlocks, 32 * warpsPerBlock>>>(dValues.data, dColumns.data,
                dRows.data, dVector.data, rows, dOutput.data);
        }
    } else if (blocks && iterations) {
        for (index_t iter = 0; iter < iterations; ++iter) {
            spmvKernel<<<blocks, threads>>>(dValues.data, dColumns.data, dSlices.data,
                                            dRows.data, dVector.data, rows, dOutput.data);
        }
    }
    checkCuda(cudaGetLastError(), "launch SpMV");
    checkCuda(cudaEventRecord(end), "record end event");
    checkCuda(cudaEventSynchronize(end), "wait for SpMV");
    float elapsed;
    checkCuda(cudaEventElapsedTime(&elapsed, start, end), "measure SpMV");
    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(end), "destroy end event");
    if (rows && iterations) {
        checkCuda(cudaMemcpy(output.data(), dOutput.data, rows * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy result from GPU");
    }
    return elapsed;
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
    const uint64_t entries = static_cast<uint64_t>(numRows) * numRows;
    if (!sparsity || entries / sparsity > std::numeric_limits<index_t>::max()) {
        std::fprintf(stderr, "Sparsity must be positive and the nonzero count must fit in 32 bits.\n");
        return 1;
    }
    const index_t nItems = entries / sparsity;

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / entries));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(numRows) + 1);  // Row delimiters
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
    const double elapsedMs = runSpmv(h_val, h_cols, h_rowDelimiters, h_vec, h_out, iterations);
    printf("Computation time: %.3f ms\n", elapsedMs);
    const double gflops = elapsedMs > 0 ? (2.0 * nItems * iterations) / (elapsedMs * 1e6) : 0;
    const double avgTime = iterations ? elapsedMs / iterations : 0;

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
