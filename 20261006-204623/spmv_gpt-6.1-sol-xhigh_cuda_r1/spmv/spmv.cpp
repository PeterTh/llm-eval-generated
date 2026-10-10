#include <cuda_runtime.h>

#include <algorithm>
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

    // Figure out the probability that a nonzero should be assigned
    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    // Seed random number generator
    srand(8675309);

    // Randomly decide whether entry i,j gets a value, but ensure n values are assigned
    bool fillRemaining = false;
    for (index_t i = 0; i < dim; ++i) {
        rowDelimiters[i] = nnzAssigned;
        for (index_t j = 0; j < dim; ++j) {
            const uint64_t numEntriesLeft = uint64_t(dim) * dim - (uint64_t(i) * dim + j);
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Neighboring lanes cooperate on a CSR row. Small groups pack sparse rows into
// a warp; full warps coalesce loads from longer rows. Each row has one writer.
template <unsigned ThreadsPerRow>
__global__ void spmvKernel(const double* __restrict__ val,
                           const index_t* __restrict__ cols,
                           const index_t* __restrict__ rowDelimiters,
                           const double* __restrict__ vec, const index_t dim,
                           double* __restrict__ out) {
    constexpr unsigned rowsPerBlock = 256 / ThreadsPerRow;
    const unsigned lane = threadIdx.x % ThreadsPerRow;
    const uint64_t stride = uint64_t(gridDim.x) * rowsPerBlock;
    for (uint64_t row = uint64_t(blockIdx.x) * rowsPerBlock + threadIdx.x / ThreadsPerRow;
         row < dim; row += stride) {
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        double sum = 0.0;
        // Widen the traversal index so adding a lane cannot overflow CSR indices.
        for (uint64_t j = uint64_t(begin) + lane; j < end; j += ThreadsPerRow) {
            sum += val[j] * __ldg(vec + cols[j]);
        }
        if constexpr (ThreadsPerRow > 1) {
            // Name exactly this row's lanes, including on a partially filled warp.
            const unsigned group = (threadIdx.x & 31) / ThreadsPerRow;
            const unsigned mask = (0xffffffffu >> (32 - ThreadsPerRow))
                                  << (group * ThreadsPerRow);
            #pragma unroll
            for (unsigned offset = ThreadsPerRow / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset, ThreadsPerRow);
            }
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        if (count_ != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                      "allocating device memory");
        }
    }
    ~DeviceArray() { if (data_) cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* data() const { return data_; }
    void upload(const T* source, cudaStream_t stream) {
        if (count_ != 0) {
            checkCuda(cudaMemcpyAsync(data_, source, count_ * sizeof(T),
                                      cudaMemcpyHostToDevice, stream), "uploading input");
        }
    }

private:
    size_t count_;
    T* data_ = nullptr;
};

class CudaSpmv {
public:
    CudaSpmv(const std::vector<double>& val, const std::vector<index_t>& cols,
             const std::vector<index_t>& rowDelimiters, const std::vector<double>& vec)
        : val_(val.size()), cols_(cols.size()), rows_(rowDelimiters.size()),
          vec_(vec.size()), out_(vec.size()), dim_(static_cast<index_t>(vec.size())) {
        int device = 0;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDevice(&device), "getting CUDA device");
        checkCuda(cudaGetDeviceProperties(&properties, device), "querying CUDA device");
        checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "creating stream");
        checkCuda(cudaEventCreate(&start_), "creating start event");
        checkCuda(cudaEventCreate(&stop_), "creating stop event");
        val_.upload(val.data(), stream_);
        cols_.upload(cols.data(), stream_);
        rows_.upload(rowDelimiters.data(), stream_);
        vec_.upload(vec.data(), stream_);

        const size_t average = dim_ == 0 ? 0 : (val.size() + dim_ - 1) / dim_;
        while (threadsPerRow_ < 32 && threadsPerRow_ < average) {
            threadsPerRow_ *= 2;
        }
        const unsigned rowsPerBlock = 256 / threadsPerRow_;
        blocks_ = static_cast<unsigned>(std::min<size_t>(
            (size_t(dim_) + rowsPerBlock - 1) / rowsPerBlock,
            size_t(properties.multiProcessorCount) * 32));
    }

    ~CudaSpmv() {
        if (tailExec_) cudaGraphExecDestroy(tailExec_);
        if (tailGraph_) cudaGraphDestroy(tailGraph_);
        if (graphExec_) cudaGraphExecDestroy(graphExec_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaStreamDestroy(stream_);
    }
    CudaSpmv(const CudaSpmv&) = delete;
    CudaSpmv& operator=(const CudaSpmv&) = delete;

    double run(index_t iterations, std::vector<double>& output) {
        if (dim_ == 0 || iterations == 0) {
            checkCuda(cudaStreamSynchronize(stream_), "finishing input transfers");
            return 0.0; // Preserve the initially zero output when there are no iterations.
        }

        // Uploads, warm-up and graph construction are outside the timed computation.
        launch();
        const index_t batch = std::min<index_t>(iterations, 128);
        const index_t remainder = iterations % batch;
        if (batch > 1) {
            prepareGraph(batch, graph_, graphExec_);
            if (remainder > 1) prepareGraph(remainder, tailGraph_, tailExec_);
        }
        checkCuda(cudaStreamSynchronize(stream_), "warming up CUDA computation");

        checkCuda(cudaEventRecord(start_, stream_), "recording start event");
        if (graphExec_) {
            for (index_t i = 0; i < iterations / batch; ++i) {
                checkCuda(cudaGraphLaunch(graphExec_, stream_), "launching computation graph");
            }
            if (tailExec_) {
                checkCuda(cudaGraphLaunch(tailExec_, stream_), "launching remainder graph");
            } else if (remainder != 0) {
                launch();
            }
        } else {
            launch();
        }
        checkCuda(cudaEventRecord(stop_, stream_), "recording stop event");
        checkCuda(cudaEventSynchronize(stop_), "waiting for CUDA computation");
        float milliseconds = 0.0f;
        checkCuda(cudaEventElapsedTime(&milliseconds, start_, stop_), "measuring CUDA computation");
        checkCuda(cudaMemcpyAsync(output.data(), out_.data(), size_t(dim_) * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream_), "downloading result");
        checkCuda(cudaStreamSynchronize(stream_), "finishing result transfer");
        return milliseconds;
    }

private:
    void prepareGraph(index_t count, cudaGraph_t& graph, cudaGraphExec_t& executable) {
        checkCuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                  "starting graph capture");
        for (index_t i = 0; i < count; ++i) launch();
        checkCuda(cudaStreamEndCapture(stream_, &graph), "finishing graph capture");
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                  "instantiating computation graph");
        checkCuda(cudaGraphUpload(executable, stream_), "uploading computation graph");
    }

    template <unsigned ThreadsPerRow>
    void launchRows() {
        spmvKernel<ThreadsPerRow><<<blocks_, 256, 0, stream_>>>(
            val_.data(), cols_.data(), rows_.data(), vec_.data(), dim_, out_.data());
    }

    void launch() {
        switch (threadsPerRow_) {
            case 1: launchRows<1>(); break;
            case 2: launchRows<2>(); break;
            case 4: launchRows<4>(); break;
            case 8: launchRows<8>(); break;
            case 16: launchRows<16>(); break;
            default: launchRows<32>(); break;
        }
        checkCuda(cudaGetLastError(), "launching SpMV kernel");
    }

    DeviceArray<double> val_;
    DeviceArray<index_t> cols_;
    DeviceArray<index_t> rows_;
    DeviceArray<double> vec_;
    DeviceArray<double> out_;
    index_t dim_;
    unsigned threadsPerRow_ = 1;
    unsigned blocks_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    cudaGraph_t tailGraph_ = nullptr;
    cudaGraphExec_t tailExec_ = nullptr;
};

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
    if (sparsity == 0 || numRows == std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Invalid matrix dimensions or sparsity.\n");
        return 1;
    }
    const uint64_t matrixElements = uint64_t(numRows) * numRows;
    const uint64_t nonzeros = matrixElements / sparsity;
    if (nonzeros > std::numeric_limits<index_t>::max()) {
        fprintf(stderr, "Matrix exceeds the 32-bit CSR index capacity.\n");
        return 1;
    }
    const index_t nItems = static_cast<index_t>(nonzeros);

    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n", 
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / matrixElements));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize data structures
    std::vector<double> h_val(nItems);                  // Non-zero values
    std::vector<index_t> h_cols(nItems);                // Column indices
    std::vector<index_t> h_rowDelimiters(size_t(numRows) + 1);  // Row delimiters
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
    CudaSpmv computation(h_val, h_cols, h_rowDelimiters, h_vec);
    const double milliseconds = computation.run(iterations, h_out);

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
