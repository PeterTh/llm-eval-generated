#include <cuda_runtime.h>

#include <algorithm>
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
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = uint64_t{dim} * dim - (uint64_t{i} * dim + j);
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

constexpr index_t warpSizeCsr = 32;
constexpr index_t rowsPerSort = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data) {
            cudaFree(data);
        }
    }

    void allocate(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(T)), "device allocation");
        }
    }

    void upload(const std::vector<T>& source) {
        allocate(source.size());
        if (!source.empty()) {
            checkCuda(cudaMemcpy(data, source.data(), source.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "input transfer");
        }
    }

    T* data = nullptr;
};

struct PackedRow {
    index_t row;
    index_t length;
};

struct LongRow {
    index_t row;
    index_t length;
    size_t start;
};

struct DeviceMatrix {
    DeviceBuffer<double> val;
    DeviceBuffer<index_t> cols;
    DeviceBuffer<index_t> csrOffsets;
    DeviceBuffer<PackedRow> rows;
    DeviceBuffer<size_t> sliceOffsets;
    DeviceBuffer<LongRow> longRows;
    index_t packedRowCount = 0;
    index_t longRowCount = 0;
};

template <index_t stride>
__device__ double sumRow(const double* val, const index_t* cols, const double* vec,
                         size_t offset, index_t length) {
    double sum = 0.0;
    // Keep separate rounding and CSR accumulation order for the printed result hash.
    #pragma unroll 4
    for (index_t i = 0; i < length; ++i, offset += stride) {
        sum = __dadd_rn(sum, __dmul_rn(val[offset], vec[cols[offset]]));
    }
    return sum;
}

__global__ void spmvCsr(const double* __restrict__ val, const index_t* __restrict__ cols,
                        const index_t* __restrict__ offsets, const double* __restrict__ vec,
                        index_t numRows, double* __restrict__ out) {
    const index_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        const index_t start = offsets[row];
        out[row] = sumRow<1>(val, cols, vec, start, offsets[row + 1] - start);
    }
}

__global__ void spmvPacked(const double* __restrict__ val, const index_t* __restrict__ cols,
                           const PackedRow* __restrict__ rows,
                           const size_t* __restrict__ sliceOffsets,
                           const LongRow* __restrict__ longRows,
                           const double* __restrict__ vec, index_t packedRows,
                           index_t packedBlocks, double* __restrict__ out) {
    if (blockIdx.x >= packedBlocks) {
        // The generator can leave a few nearly full rows at the end. Give
        // each outlier a block to load and multiply its entries in parallel.
        constexpr index_t tileSize = 256;
        __shared__ double products[tileSize];
        const LongRow row = longRows[blockIdx.x - packedBlocks];
        double sum = 0.0;
        for (size_t first = 0; first < row.length; first += tileSize) {
            const index_t count = min(size_t{tileSize}, row.length - first);
            for (index_t i = threadIdx.x; i < count; i += blockDim.x) {
                const size_t offset = row.start + first + i;
                products[i] = __dmul_rn(val[offset], vec[cols[offset]]);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                #pragma unroll 8
                for (index_t i = 0; i < count; ++i) {
                    sum = __dadd_rn(sum, products[i]);
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out[row.row] = sum;
        }
        return;
    }

    const index_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread < packedRows) {
        const PackedRow row = rows[thread];
        const size_t start = sliceOffsets[thread / warpSizeCsr] + thread % warpSizeCsr;
        out[row.row] = sumRow<warpSizeCsr>(val, cols, vec, start, row.length);
    }
}

// Store one warp of rows per slice, interleaving entries at the same position
// in each row. Sort locally by length to limit padding and preserve locality.
bool uploadPacked(const std::vector<double>& val, const std::vector<index_t>& cols,
                  const std::vector<index_t>& offsets, DeviceMatrix& device) {
    const size_t numRows = offsets.size() - 1;
    const size_t longRowThreshold = std::max(size_t{64}, 2 * (val.size() / numRows));
    std::vector<PackedRow> rows;
    rows.reserve(numRows);
    std::vector<LongRow> longRows;
    for (size_t row = 0; row < numRows; ++row) {
        const index_t length = offsets[row + 1] - offsets[row];
        if (length > longRowThreshold) {
            longRows.push_back({static_cast<index_t>(row), length, 0});
        } else {
            rows.push_back({static_cast<index_t>(row), length});
        }
    }
    for (size_t first = 0; first < rows.size(); first += rowsPerSort) {
        std::sort(rows.begin() + first, rows.begin() + std::min(first + rowsPerSort, rows.size()),
                  [](const PackedRow& a, const PackedRow& b) { return a.length > b.length; });
    }

    std::vector<size_t> sliceOffsets;
    sliceOffsets.reserve((rows.size() + warpSizeCsr - 1) / warpSizeCsr);
    size_t packedSize = 0;
    for (size_t first = 0; first < rows.size(); first += warpSizeCsr) {
        sliceOffsets.push_back(packedSize);
        packedSize += size_t{rows[first].length} * warpSizeCsr;
    }
    for (auto& row : longRows) {
        row.start = packedSize;
        packedSize += row.length;
    }
    // CSR keeps buffer usage bounded when a few long rows would create large slices.
    if (packedSize > val.size() + val.size() / 4 + 1024) {
        return false;
    }

    std::vector<double> packedVal(packedSize);
    std::vector<index_t> packedCols(packedSize);
    for (size_t first = 0; first < rows.size(); first += warpSizeCsr) {
        const size_t slice = sliceOffsets[first / warpSizeCsr];
        const size_t end = std::min(first + warpSizeCsr, rows.size());
        for (size_t thread = first; thread < end; ++thread) {
            const PackedRow row = rows[thread];
            size_t target = slice + thread - first;
            size_t source = offsets[row.row];
            for (index_t i = 0; i < row.length; ++i, ++source, target += warpSizeCsr) {
                packedVal[target] = val[source];
                packedCols[target] = cols[source];
            }
        }
    }
    for (const LongRow& row : longRows) {
        std::copy_n(val.data() + offsets[row.row], row.length, packedVal.data() + row.start);
        std::copy_n(cols.data() + offsets[row.row], row.length, packedCols.data() + row.start);
    }
    device.val.upload(packedVal);
    device.cols.upload(packedCols);
    device.rows.upload(rows);
    device.sliceOffsets.upload(sliceOffsets);
    device.longRows.upload(longRows);
    device.packedRowCount = static_cast<index_t>(rows.size());
    device.longRowCount = static_cast<index_t>(longRows.size());
    return true;
}

double spmvGpu(const std::vector<double>& val, const std::vector<index_t>& cols,
               const std::vector<index_t>& offsets, const std::vector<double>& vec,
               index_t iterations, double* out) {
    const index_t numRows = static_cast<index_t>(vec.size());
    if (numRows == 0 || iterations == 0) {
        return 0.0;
    }

    DeviceMatrix matrix;
    DeviceBuffer<double> deviceVec;
    DeviceBuffer<double> deviceOut;
    const bool packed = !val.empty() && uploadPacked(val, cols, offsets, matrix);
    if (!packed) {
        matrix.val.upload(val);
        matrix.cols.upload(cols);
        matrix.csrOffsets.upload(offsets);
    }
    deviceVec.upload(vec);
    deviceOut.allocate(numRows);

    int device, smCount;
    checkCuda(cudaGetDevice(&device), "device query");
    checkCuda(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device),
              "multiprocessor count query");
    const index_t regularRows = packed ? matrix.packedRowCount : numRows;
    // Use smaller blocks until there are enough rows to keep all SMs busy.
    const index_t blockSize = regularRows >= 256u * smCount ? 128 :
                              regularRows >= 128u * smCount ? 64 : 32;
    const index_t blocks = (size_t{regularRows} + blockSize - 1) / blockSize;
    auto launch = [&] {
        if (packed) {
            spmvPacked<<<blocks + matrix.longRowCount, blockSize>>>(
                matrix.val.data, matrix.cols.data, matrix.rows.data, matrix.sliceOffsets.data,
                matrix.longRows.data, deviceVec.data, regularRows, blocks, deviceOut.data);
        } else {
            spmvCsr<<<blocks, blockSize>>>(matrix.val.data, matrix.cols.data, matrix.csrOffsets.data,
                                          deviceVec.data, numRows, deviceOut.data);
        }
    };

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "start event creation");
    checkCuda(cudaEventCreate(&stop), "stop event creation");
    launch(); // Warm up the kernel before starting the timed iterations.
    checkCuda(cudaGetLastError(), "warm-up kernel launch");
    checkCuda(cudaEventRecord(start), "start event recording");
    for (index_t iter = 0; iter < iterations; ++iter) {
        launch();
    }
    checkCuda(cudaGetLastError(), "SpMV kernel launch");
    checkCuda(cudaEventRecord(stop), "stop event recording");
    checkCuda(cudaEventSynchronize(stop), "SpMV kernel execution");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "elapsed time measurement");
    checkCuda(cudaEventDestroy(start), "start event destruction");
    checkCuda(cudaEventDestroy(stop), "stop event destruction");

    if (out != nullptr) {
        checkCuda(cudaMemcpy(out, deviceOut.data, size_t{numRows} * sizeof(double),
                             cudaMemcpyDeviceToHost), "result transfer");
    }
    return elapsedMs;
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

    // Calculate number of non-zero elements
    if (sparsity == 0) {
        fprintf(stderr, "Sparsity must be greater than zero.\n");
        return 1;
    }
    const uint64_t matrixEntries = uint64_t{numRows} * numRows;
    if (matrixEntries / sparsity > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "The non-zero element count exceeds the CSR index range.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(matrixEntries / sparsity);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    const double sparsePercent = matrixEntries == 0 ? 0.0 :
        100.0 * (1.0 - static_cast<double>(nItems) / static_cast<double>(matrixEntries));
    printf("Non-zero elements: %u (%.2f%% sparse)\n", nItems, sparsePercent);
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(size_t{numRows} + 1);  // Row delimiters
    std::vector<double> h_vec(numRows);                 // Dense vector
    std::vector<double> h_out(printResults || validate ? numRows : 0);

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
    const double durationMs = spmvGpu(h_val, h_cols, h_rowDelimiters, h_vec, iterations,
                                      (printResults || validate) ? h_out.data() : nullptr);
    printf("Computation time: %.3f ms\n", durationMs);

    // Calculate performance metrics
    const double gflops = durationMs > 0.0 ? (2.0 * nItems * iterations) / (durationMs * 1e6) : 0.0;
    const double avgTime = iterations > 0 ? durationMs / iterations : 0.0;
    
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
