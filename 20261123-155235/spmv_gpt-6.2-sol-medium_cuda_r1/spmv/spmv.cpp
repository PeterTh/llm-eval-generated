#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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
template <typename ColumnIndex>
void initRandomMatrix(ColumnIndex* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    // Figure out the probability that a nonzero should be assigned
    const uint64_t totalEntries = uint64_t{dim} * dim;
    const double prob = totalEntries == 0 ? 0.0 : static_cast<double>(n) / static_cast<double>(totalEntries);

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    uint64_t entriesLeft = totalEntries;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim && nnzAssigned < n; ++j, --entriesLeft) {
            index_t needToAssign = n - nnzAssigned;
            if (entriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if (fillRemaining || static_cast<double>(rand()) / RAND_MAX <= prob) {
                // Assign (i,j) a value
                cols[nnzAssigned] = static_cast<ColumnIndex>(j);
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
template <typename ColumnIndex>
void spmvCpu(const double* val, const ColumnIndex* cols, const index_t* rowDelimiters,
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

constexpr unsigned BLOCK_THREADS = 128;
constexpr unsigned WARP_THREADS = 32;

template <typename ColumnIndex, unsigned ThreadsPerRow, bool Unroll>
__global__ void spmvCuda(const double* __restrict__ val, const ColumnIndex* __restrict__ cols,
                         const index_t* __restrict__ rowDelimiters,
                         const double* __restrict__ vec, index_t dim, double* __restrict__ out) {
    const index_t row = blockIdx.x * (BLOCK_THREADS / ThreadsPerRow) + threadIdx.x / ThreadsPerRow;
    if (row >= dim) {
        return;
    }

    const unsigned lane = threadIdx.x % ThreadsPerRow;
    const index_t begin = rowDelimiters[row];
    const index_t length = rowDelimiters[row + 1] - begin;
    double sum = 0.0;
    index_t offset = lane;
    if constexpr (Unroll) {
        // Independent sums let each lane overlap the loads from four consecutive groups.
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        for (; offset + 3 * ThreadsPerRow < length; offset += 4 * ThreadsPerRow) {
            const index_t entry = begin + offset;
            sum += val[entry] * vec[cols[entry]];
            sum1 += val[entry + ThreadsPerRow] * vec[cols[entry + ThreadsPerRow]];
            sum2 += val[entry + 2 * ThreadsPerRow] * vec[cols[entry + 2 * ThreadsPerRow]];
            sum3 += val[entry + 3 * ThreadsPerRow] * vec[cols[entry + 3 * ThreadsPerRow]];
        }
        sum = (sum + sum1) + (sum2 + sum3);
    }
    for (; offset < length; offset += ThreadsPerRow) {
        const index_t entry = begin + offset;
        sum += val[entry] * vec[cols[entry]];
    }

    if constexpr (ThreadsPerRow > 1) {
        // The last warp can have fewer rows, but each participating group is complete.
        const unsigned firstLane = threadIdx.x % WARP_THREADS - lane;
        const unsigned mask = (0xffffffffu >> (WARP_THREADS - ThreadsPerRow)) << firstLane;
        for (unsigned delta = ThreadsPerRow / 2; delta > 0; delta /= 2) {
            sum += __shfl_down_sync(mask, sum, delta, ThreadsPerRow);
        }
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size) {
        if (size != 0) {
            checkCuda(cudaMalloc(&data_, size * sizeof(T)), "Allocating device memory");
        }
    }

    explicit DeviceBuffer(const std::vector<T>& source) : DeviceBuffer(source.size()) {
        if (!source.empty()) {
            checkCuda(cudaMemcpy(data_, source.data(), source.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "Copying input to device");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    T* data() const { return data_; }

private:
    T* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { checkCuda(cudaEventCreate(&event_), "Creating CUDA timing event"); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(event_); }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

template <typename ColumnIndex, unsigned ThreadsPerRow, bool Unroll = false>
void launchSpmv(const double* val, const ColumnIndex* cols, const index_t* rowDelimiters,
                const double* vec, index_t dim, double* out) {
    constexpr unsigned rowsPerBlock = BLOCK_THREADS / ThreadsPerRow;
    const unsigned blocks = 1 + (dim - 1) / rowsPerBlock;
    spmvCuda<ColumnIndex, ThreadsPerRow, Unroll><<<blocks, BLOCK_THREADS>>>(
        val, cols, rowDelimiters, vec, dim, out);
}

template <typename ColumnIndex>
double benchmarkCuda(const std::vector<double>& val, const std::vector<ColumnIndex>& cols,
                     const std::vector<index_t>& rowDelimiters, const std::vector<double>& vec,
                     index_t iterations, bool readOutput, std::vector<double>& out) {
    const CudaEvent start;
    const CudaEvent stop;
    const auto dim = static_cast<index_t>(vec.size());
    if (dim == 0 || iterations == 0) {
        return 0.0;
    }

    const DeviceBuffer<double> dVal(val);
    const DeviceBuffer<ColumnIndex> dCols(cols);
    const DeviceBuffer<index_t> dRowDelimiters(rowDelimiters);
    const DeviceBuffer<double> dVec(vec);
    const DeviceBuffer<double> dOut(dim);

    // Use subwarps for short rows and a full warp for longer rows.
    const size_t entriesPerRow = val.size() / dim;
    auto launch = launchSpmv<ColumnIndex, WARP_THREADS>;
    if (entriesPerRow <= 2) {
        launch = launchSpmv<ColumnIndex, 1>;
    } else if (entriesPerRow <= 4) {
        launch = launchSpmv<ColumnIndex, 2>;
    } else if (entriesPerRow <= 8) {
        launch = launchSpmv<ColumnIndex, 4>;
    } else if (entriesPerRow <= 16) {
        launch = launchSpmv<ColumnIndex, 8>;
    } else if (entriesPerRow <= 32) {
        launch = launchSpmv<ColumnIndex, 16>;
    } else if (entriesPerRow >= 512) {
        launch = launchSpmv<ColumnIndex, WARP_THREADS, true>;
    }

    const auto runSpmv = [&] {
        launch(dVal.data(), dCols.data(), dRowDelimiters.data(), dVec.data(), dim, dOut.data());
    };
    // Warm up the kernel before timing the stream of benchmark iterations.
    runSpmv();
    checkCuda(cudaGetLastError(), "Launching CUDA SpMV warmup");
    checkCuda(cudaEventRecord(start.get()), "Recording SpMV start time");
    for (index_t iter = 0; iter < iterations; ++iter) {
        runSpmv();
    }
    checkCuda(cudaGetLastError(), "Launching CUDA SpMV");
    checkCuda(cudaEventRecord(stop.get()), "Recording SpMV end time");
    checkCuda(cudaEventSynchronize(stop.get()), "Waiting for CUDA SpMV");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()), "Measuring CUDA SpMV");
    if (readOutput) {
        checkCuda(cudaMemcpy(out.data(), dOut.data(), out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "Copying SpMV output to host");
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

template <typename ColumnIndex>
int runBenchmark(index_t numRows, index_t nItems, index_t iterations, double maxVal,
                 bool validate, bool printResults) {
    // Allocate and initialize data structures
    const bool readOutput = validate || printResults;
    std::vector<double> h_val(nItems);
    std::vector<ColumnIndex> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(size_t{numRows} + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(readOutput ? numRows : 0);

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
    double milliseconds;
    try {
        milliseconds = benchmarkCuda(h_val, h_cols, h_rowDelimiters, h_vec, iterations,
                                     readOutput, h_out);
    } catch (const std::runtime_error& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }

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
    const uint64_t totalEntries = uint64_t{numRows} * numRows;
    if (sparsity == 0 || totalEntries / sparsity > UINT32_MAX) {
        fprintf(stderr, "Sparsity must be positive and the number of nonzeros must fit in CSR indices.\n");
        return 1;
    }
    const auto nItems = static_cast<index_t>(totalEntries / sparsity);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems,
           totalEntries == 0 ? 100.0 :
           100.0 * (1.0 - static_cast<double>(nItems) / static_cast<double>(totalEntries)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Narrow column indices reduce memory traffic in the gather kernel.
    if (numRows <= uint32_t{UINT16_MAX} + 1) {
        return runBenchmark<uint16_t>(numRows, nItems, iterations, maxVal, validate, printResults);
    }
    return runBenchmark<index_t>(numRows, nItems, iterations, maxVal, validate, printResults);
}
