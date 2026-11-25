#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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
            const uint64_t numEntriesLeft =
                static_cast<uint64_t>(dim) * (dim - i) - j;
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

namespace {

constexpr unsigned warpWidth = 32;

struct alignas(8) SlicedRow {
    index_t index;
    index_t length;
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(T)), "cudaMalloc");
        }
    }

    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const { return data_; }

    void upload(const T* source, size_t count, size_t offset = 0) {
        if (count != 0) {
            checkCuda(cudaMemcpy(data_ + offset, source, count * sizeof(T),
                                 cudaMemcpyHostToDevice), "uploading SpMV data");
        }
    }

private:
    T* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { checkCuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

// Store one slice per warp, interleaving the entries of its rows. Each lane
// traverses its row in CSR order, so it produces the same rounded sum.
__global__ void spmvSliced(const double* __restrict__ values,
                           const index_t* __restrict__ columns,
                           const size_t* __restrict__ sliceOffsets,
                           const SlicedRow* __restrict__ rows, const double* __restrict__ vec,
                           index_t dim, double* __restrict__ out) {
    const size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= dim) {
        return;
    }

    const SlicedRow row = rows[slot];
    size_t entry = sliceOffsets[slot / warpWidth] + slot % warpWidth;
    double sum = 0.0;
    constexpr unsigned unroll = 8;
    index_t j = 0;
    for (const index_t end = row.length / unroll * unroll; j < end;
         j += unroll, entry += warpWidth * unroll) {
        double products[unroll];
        #pragma unroll
        for (unsigned k = 0; k < unroll; ++k) {
            const size_t pos = entry + k * warpWidth;
            products[k] = values[pos] * vec[columns[pos]];
        }
        #pragma unroll
        for (unsigned k = 0; k < unroll; ++k) {
            sum += products[k];
        }
    }
    for (; j < row.length; ++j, entry += warpWidth) {
        sum += values[entry] * vec[columns[entry]];
    }
    out[row.index] = sum;
}

constexpr unsigned csrWarps = 4;
constexpr unsigned csrTile = 128;

// A warp loads and multiplies each tile cooperatively, then its first lane
// accumulates the products in CSR order. This exposes more parallel work
// when there are too few rows to fill the GPU with one lane per row.
__global__ void spmvWarp(const double* __restrict__ values, const index_t* __restrict__ columns,
                         const index_t* __restrict__ rowDelimiters, const double* __restrict__ vec,
                         index_t dim, double* __restrict__ out) {
    __shared__ double products[csrWarps][csrTile];
    const unsigned warp = threadIdx.x / warpWidth;
    const unsigned lane = threadIdx.x % warpWidth;
    const index_t row = blockIdx.x * csrWarps + warp;
    if (row >= dim) {
        return;
    }
    const index_t start = rowDelimiters[row];
    const index_t length = rowDelimiters[row + 1] - start;
    double sum = 0.0;
    for (index_t base = 0; base < length;) {
        const unsigned count = min(csrTile, length - base);
        for (unsigned k = lane; k < count; k += warpWidth) {
            const index_t pos = start + base + k;
            products[warp][k] = values[pos] * vec[columns[pos]];
        }
        __syncwarp();
        if (lane == 0) {
            #pragma unroll 8
            for (unsigned k = 0; k < count; ++k) {
                sum += products[warp][k];
            }
        }
        __syncwarp();
        base += count;
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

template <typename Launch>
double timeSpmv(const Launch& launch, index_t iterations) {
    CudaEvent start;
    CudaEvent stop;
    launch();
    checkCuda(cudaGetLastError(), "launching SpMV warm-up");
    checkCuda(cudaEventRecord(start.get()), "recording SpMV start");
    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();
    }
    checkCuda(cudaGetLastError(), "launching SpMV");
    checkCuda(cudaEventRecord(stop.get()), "recording SpMV end");
    checkCuda(cudaEventSynchronize(stop.get()), "executing SpMV");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start.get(), stop.get()), "timing SpMV");
    return elapsedMs;
}

double runSliced(const std::vector<double>& values, const std::vector<index_t>& columns,
                 const std::vector<index_t>& rowDelimiters, const DeviceBuffer<double>& dVec,
                 index_t dim, index_t iterations, DeviceBuffer<double>& dOut) {
    std::vector<SlicedRow> rows(dim);
    for (index_t row = 0; row < dim; ++row) {
        rows[row] = {row, rowDelimiters[row + 1] - rowDelimiters[row]};
    }
    // Group rows of similar length to reduce both padding and warp divergence.
    std::sort(rows.begin(), rows.end(), [](const SlicedRow& a, const SlicedRow& b) {
        return a.length > b.length;
    });

    const size_t sliceCount = (rows.size() + warpWidth - 1) / warpWidth;
    std::vector<size_t> sliceOffsets(sliceCount + 1);
    for (size_t slice = 0; slice < sliceCount; ++slice) {
        const size_t length = rows[slice * warpWidth].length;
        sliceOffsets[slice + 1] = sliceOffsets[slice] + length * warpWidth;
    }

    DeviceBuffer<double> dValues(sliceOffsets.back());
    DeviceBuffer<index_t> dColumns(sliceOffsets.back());
    DeviceBuffer<size_t> dSliceOffsets(sliceCount);
    DeviceBuffer<SlicedRow> dRows(dim);
    dSliceOffsets.upload(sliceOffsets.data(), sliceCount);
    dRows.upload(rows.data(), rows.size());

    // Pack and upload batches of slices to bound the extra host memory.
    constexpr size_t batchEntries = 1 << 20;
    std::vector<double> valueBatch;
    std::vector<index_t> columnBatch;
    for (size_t first = 0; first < sliceCount;) {
        size_t last = first + 1;
        while (last < sliceCount && sliceOffsets[last + 1] - sliceOffsets[first] <= batchEntries) {
            ++last;
        }
        const size_t base = sliceOffsets[first];
        const size_t count = sliceOffsets[last] - base;
        valueBatch.resize(count);
        columnBatch.resize(count);
        for (size_t slice = first; slice < last; ++slice) {
            const size_t slot = slice * warpWidth;
            const size_t offset = sliceOffsets[slice] - base;
            const unsigned lanes = std::min<size_t>(warpWidth, rows.size() - slot);
            // Transpose a few entries per row at a time for cache locality.
            constexpr index_t tile = 32;
            for (size_t start = 0; start < rows[slot].length; start += tile) {
                for (unsigned lane = 0; lane < lanes; ++lane) {
                    const SlicedRow row = rows[slot + lane];
                    const size_t rowStart = rowDelimiters[row.index];
                    const size_t stop = std::min<size_t>(start + tile, row.length);
                    for (size_t j = start; j < stop; ++j) {
                        const size_t packed = offset + j * warpWidth + lane;
                        valueBatch[packed] = values[rowStart + j];
                        columnBatch[packed] = columns[rowStart + j];
                    }
                }
            }
        }
        dValues.upload(valueBatch.data(), count, base);
        dColumns.upload(columnBatch.data(), count, base);
        first = last;
    }

    const unsigned threads = dim <= 8192 ? 32 : 128;
    const unsigned blocks = (rows.size() + threads - 1) / threads;
    return timeSpmv([&] {
        spmvSliced<<<blocks, threads>>>(dValues.data(), dColumns.data(), dSliceOffsets.data(),
                                       dRows.data(), dVec.data(), dim, dOut.data());
    }, iterations);
}

double runWarp(const std::vector<double>& values, const std::vector<index_t>& columns,
               const std::vector<index_t>& rowDelimiters, const DeviceBuffer<double>& dVec,
               index_t dim, index_t iterations, DeviceBuffer<double>& dOut) {
    DeviceBuffer<double> dValues(values.size());
    DeviceBuffer<index_t> dColumns(columns.size());
    DeviceBuffer<index_t> dRowDelimiters(rowDelimiters.size());
    dValues.upload(values.data(), values.size());
    dColumns.upload(columns.data(), columns.size());
    dRowDelimiters.upload(rowDelimiters.data(), rowDelimiters.size());
    const unsigned blocks = (dim + csrWarps - 1) / csrWarps;
    return timeSpmv([&] {
        spmvWarp<<<blocks, csrWarps * warpWidth>>>(dValues.data(), dColumns.data(),
                                                  dRowDelimiters.data(), dVec.data(), dim, dOut.data());
    }, iterations);
}

double spmvCuda(const std::vector<double>& values, const std::vector<index_t>& columns,
                const std::vector<index_t>& rowDelimiters, const std::vector<double>& vec,
                index_t iterations, bool readOutput, std::vector<double>& out) {
    const index_t dim = static_cast<index_t>(vec.size());
    if (dim == 0 || iterations == 0) {
        return 0.0;
    }
    DeviceBuffer<double> dVec(dim);
    DeviceBuffer<double> dOut(dim);
    dVec.upload(vec.data(), vec.size());
    // For small problems, spread the products across more warps. Longer rows
    // benefit from sliced storage and avoid the shared-memory staging cost.
    const bool useWarpKernel = dim <= 1024 && values.size() <= 256u * dim;
    const double elapsedMs = useWarpKernel
        ? runWarp(values, columns, rowDelimiters, dVec, dim, iterations, dOut)
        : runSliced(values, columns, rowDelimiters, dVec, dim, iterations, dOut);
    if (readOutput) {
        checkCuda(cudaMemcpy(out.data(), dOut.data(), out.size() * sizeof(double),
                             cudaMemcpyDeviceToHost), "downloading SpMV result");
    }
    return elapsedMs;
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
        fprintf(stderr, "Sparsity must be greater than zero.\n");
        return 1;
    }
    const uint64_t matrixSize = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros = matrixSize / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The number of non-zero entries exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixSize));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(static_cast<size_t>(numRows) + 1); // Row delimiters
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
    double durationMs;
    try {
        durationMs = spmvCuda(h_val, h_cols, h_rowDelimiters, h_vec, iterations,
                              validate || printResults, h_out);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA SpMV failed: %s\n", error.what());
        return 1;
    }
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const double gflops = durationMs > 0.0 ? (2.0 * nItems * iterations) / (durationMs * 1e6) : 0.0;
    const double avgTime = iterations != 0 ? durationMs / iterations : 0.0;
    
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
