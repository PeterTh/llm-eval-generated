#include <cuda_runtime.h>

#include <cmath>
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

    const uint64_t entries = static_cast<uint64_t>(dim) * dim;
    const double prob = entries ? static_cast<double>(n) / static_cast<double>(entries) : 0.0;

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    uint64_t numEntriesLeft = entries;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j, --numEntriesLeft) {
            if (numEntriesLeft == n - nnzAssigned || static_cast<double>(rand()) / RAND_MAX <= prob) {
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

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed (%s)\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : bytes(count * sizeof(T)) {
        if (bytes != 0) {
            checkCuda(cudaMalloc(&data, bytes), "cudaMalloc");
        }
    }

    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data; }

    void upload(const T* source) const {
        if (bytes != 0) {
            checkCuda(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice), "Uploading SpMV input");
        }
    }

    void download(T* destination) const {
        checkCuda(cudaMemcpy(destination, data, bytes, cudaMemcpyDeviceToHost), "Downloading SpMV output");
    }

private:
    T* data = nullptr;
    size_t bytes;
};

constexpr index_t blockSize = 128;
constexpr index_t warpWidth = 32;

// Each group of adjacent lanes handles one row, reading consecutive CSR entries.
template <index_t lanesPerRow>
__global__ void spmvGpu(const double* __restrict__ values, const index_t* __restrict__ columns,
                        const index_t* __restrict__ offsets, const double* __restrict__ vector,
                        index_t numRows, double* __restrict__ output) {
    constexpr index_t rowsPerBlock = blockSize / lanesPerRow;
    const index_t row = blockIdx.x * rowsPerBlock + threadIdx.x / lanesPerRow;
    const index_t lane = threadIdx.x % lanesPerRow;
    const bool validRow = row < numRows;
    index_t first = 0;
    index_t last = 0;
    if (validRow) {
        first = offsets[row];
        last = offsets[row + 1];
    }

    double sum = 0.0;
    for (index_t offset = lane; offset < last - first; offset += lanesPerRow) {
        const index_t entry = first + offset;
        sum += values[entry] * vector[columns[entry]];
    }

    // Out-of-range groups contribute zeros and still participate in the shuffle.
    for (index_t step = lanesPerRow / 2; step > 0; step /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, step, lanesPerRow);
    }
    if (validRow && lane == 0) {
        output[row] = sum;
    }
}

template <index_t lanesPerRow>
void launchSpmv(const double* values, const index_t* columns, const index_t* offsets,
                const double* vector, index_t numRows, index_t iterations, double* output) {
    constexpr index_t rowsPerBlock = blockSize / lanesPerRow;
    const index_t blocks = (numRows - 1) / rowsPerBlock + 1;
    for (index_t iter = 0; iter < iterations; ++iter) {
        spmvGpu<lanesPerRow><<<blocks, blockSize>>>(values, columns, offsets, vector, numRows, output);
    }
    checkCuda(cudaGetLastError(), "Launching SpMV kernel");
}

double runSpmv(const std::vector<double>& values, const std::vector<index_t>& columns,
               const std::vector<index_t>& offsets, const std::vector<double>& vector,
               index_t iterations, std::vector<double>& output, bool readOutput) {
    if (vector.empty() || iterations == 0) {
        return 0.0;
    }

    const auto numRows = static_cast<index_t>(vector.size());
    DeviceBuffer<double> dValues(values.size());
    DeviceBuffer<index_t> dColumns(columns.size());
    DeviceBuffer<index_t> dOffsets(offsets.size());
    DeviceBuffer<double> dVector(vector.size());
    DeviceBuffer<double> dOutput(output.size());
    dValues.upload(values.data());
    dColumns.upload(columns.data());
    dOffsets.upload(offsets.data());
    dVector.upload(vector.data());

    // Narrow groups keep more rows in flight when the average row is short.
    using Launch = decltype(&launchSpmv<1>);
    const Launch launches[] = {launchSpmv<1>, launchSpmv<2>, launchSpmv<4>,
                               launchSpmv<8>, launchSpmv<16>, launchSpmv<warpWidth>};
    const auto entriesPerRow = values.size() / numRows;
    size_t choice = 0;
    for (size_t lanes = 1; lanes < warpWidth && entriesPerRow >= 4 * lanes; lanes *= 2) {
        ++choice;
    }

    // Load and warm up the selected kernel before timing the requested iterations.
    launches[choice](dValues.get(), dColumns.get(), dOffsets.get(), dVector.get(),
                     numRows, 1, dOutput.get());

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&stop), "Creating CUDA stop event");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");
    launches[choice](dValues.get(), dColumns.get(), dOffsets.get(), dVector.get(),
                     numRows, iterations, dOutput.get());
    checkCuda(cudaEventRecord(stop), "Recording CUDA stop event");
    checkCuda(cudaEventSynchronize(stop), "Waiting for SpMV");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measuring SpMV time");
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    if (readOutput) {
        dOutput.download(output.data());
    }
    return milliseconds;
}

}  // namespace

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

    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero\n");
        return 1;
    }
    const uint64_t matrixEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros = matrixEntries / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Non-zero count exceeds the range of 32-bit CSR offsets\n");
        return 1;
    }
    const auto nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, matrixEntries ? 100.0 * (1.0 - static_cast<double>(nItems) / matrixEntries) : 0.0);
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
    const double milliseconds = runSpmv(h_val, h_cols, h_rowDelimiters, h_vec,
                                        iterations, h_out, validate || printResults);
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    const double gflops = milliseconds > 0.0 ? (2.0 * nItems * iterations) / (milliseconds * 1e6) : 0.0;
    const double avgTime = iterations ? milliseconds / iterations : 0.0;
    
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
