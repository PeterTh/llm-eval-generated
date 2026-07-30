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
constexpr int WARP_SIZE = 32;

// ****************************************************************************
// CUDA error checking macro
// ****************************************************************************
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ****************************************************************************
// Kernel: spmv_csr_warp_kernel
//
// Purpose:
//   Warp-per-row CSR SpMV kernel. Each warp processes one row of the matrix.
//   Threads within a warp cooperatively compute the dot product for a row,
//   using warp shuffle reduction for efficient partial sum aggregation.
//
// Arguments:
//   val:           non-zero values (device)
//   cols:          column indices (device)
//   rowDelimiters: CSR row pointers (device)
//   vec:           dense input vector (device)
//   dim:           number of rows
//   out:           output result vector (device)
//
// ****************************************************************************
__global__ void spmv_csr_warp_kernel(const double* __restrict__ val,
                                     const index_t* __restrict__ cols,
                                     const index_t* __restrict__ rowDelimiters,
                                     const double* __restrict__ vec,
                                     const index_t dim,
                                     double* __restrict__ out) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    if (warpId >= static_cast<int>(dim)) return;

    const index_t row = static_cast<index_t>(warpId);
    const index_t rowStart = rowDelimiters[row];
    const index_t rowEnd   = rowDelimiters[row + 1];

    double sum = 0.0;
    for (index_t j = rowStart + static_cast<index_t>(laneId); j < rowEnd; j += WARP_SIZE) {
        sum += val[j] * vec[cols[j]];
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (laneId == 0) {
        out[row] = sum;
    }
}

// ****************************************************************************
// Function: fill
//
// Purpose:
//   Initialize array with random values
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
//   matrix in CSR format.
//
// ****************************************************************************
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t n, const index_t dim) {
    index_t nnzAssigned = 0;

    double prob = static_cast<double>(n) / (static_cast<double>(dim) * static_cast<double>(dim));

    srand(8675309);

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
                cols[nnzAssigned] = j;
                nnzAssigned++;
            }
        }
    }
    rowDelimiters[dim] = n;
}

// ****************************************************************************
// Function: spmvCpu
//
// Purpose:
//   CPU reference SpMV for validation
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
// Function: verifyResults
//
// Purpose:
//   Verifies correctness by comparing to reference solution
//
// ****************************************************************************
bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                printf("Validation failed at index %u: reference %.10e, got %.10e\n", i, ref, res);
                return false;
            }
        } else {
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

    // Print GPU info
    int deviceId = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark [CUDA]\n");
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Matrix size: %u x %u\n", numRows, numRows);
    printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    printf("Non-zero elements: %u (%.2f%% sparse)\n",
           nItems, 100.0 * (1.0 - static_cast<double>(nItems) / (numRows * numRows)));
    printf("Iterations: %u\n", iterations);
    printf("Max value: %.2f\n", maxVal);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate and initialize host data
    std::vector<double> h_val(nItems);
    std::vector<index_t> h_cols(nItems);
    std::vector<index_t> h_rowDelimiters(numRows + 1);
    std::vector<double> h_vec(numRows);
    std::vector<double> h_out(numRows);

    printf("Initializing data structures...\n");
    fill(h_vec.data(), numRows, maxVal);
    fill(h_val.data(), nItems, maxVal);
    initRandomMatrix(h_cols.data(), h_rowDelimiters.data(), nItems, numRows);

    // For validation, compute reference solution on CPU
    std::vector<double> h_reference;
    if (validate) {
        printf("Computing reference solution...\n");
        h_reference.resize(numRows);
        spmvCpu(h_val.data(), h_cols.data(), h_rowDelimiters.data(),
                h_vec.data(), numRows, h_reference.data());
    }

    // Allocate device memory
    double* d_val = nullptr;
    index_t* d_cols = nullptr;
    index_t* d_rowDelimiters = nullptr;
    double* d_vec = nullptr;
    double* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_val,           nItems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols,          nItems * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_rowDelimiters, (numRows + 1) * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&d_vec,           numRows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,           numRows * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_val,           h_val.data(),           nItems * sizeof(double),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols,          h_cols.data(),          nItems * sizeof(index_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowDelimiters, h_rowDelimiters.data(), (numRows + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vec,           h_vec.data(),           numRows * sizeof(double),      cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    // Warp-per-row: each warp handles one row
    const int threadsPerBlock = 256;
    const int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    const int numBlocks = (numRows + warpsPerBlock - 1) / warpsPerBlock;

    // Warmup + synchronization before timing
    spmv_csr_warp_kernel<<<numBlocks, threadsPerBlock>>>(
        d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create CUDA events for accurate GPU timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    printf("Computing SpMV...\n");

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (index_t iter = 0; iter < iterations; ++iter) {
        spmv_csr_warp_kernel<<<numBlocks, threadsPerBlock>>>(
            d_val, d_cols, d_rowDelimiters, d_vec, numRows, d_out);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));

    long long durationMs = static_cast<long long>(elapsedMs);

    printf("Computation time: %lld ms\n", durationMs);

    // Calculate performance metrics
    const double elapsedSec = elapsedMs / 1000.0;
    const double gflops = (2.0 * nItems * iterations) / elapsedSec / 1e9;
    const double avgTime = elapsedMs / static_cast<double>(iterations);

    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, numRows * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(h_out, "OutputVector");
    }

    // Validation
    bool validationPassed = true;
    if (validate) {
        printf("Validating result...\n");
        validationPassed = verifyResults(h_reference.data(), h_out.data(), numRows);

        if (validationPassed) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_rowDelimiters));
    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_out));

    return validate ? (validationPassed ? 0 : 1) : 0;
}
