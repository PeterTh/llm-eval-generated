#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS error " + std::to_string(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        check(cudaMalloc(&data, count * sizeof(T)), "Allocating GPU memory");
    }

    ~DeviceBuffer() { cudaFree(data); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

struct CudaHandles {
    cublasHandle_t blas = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~CudaHandles() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (blas) cublasDestroy(blas);
    }
};

constexpr int blockSize = 256;

__global__ void addDiagonal(double* matrix, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        matrix[static_cast<size_t>(row) * n + row] += n;
    }
}

constexpr int panelWidth = 32;
constexpr int diagonalRowsPerBlock = 8;
constexpr int solveRowsPerBlock = 8;
constexpr int matrixTile = 16;
constexpr int smallMatrix = 512;

template <bool subtract>
__global__ void symmetricProduct(const double* input, double* output, int n, int k, int pitch) {
    if (blockIdx.x > blockIdx.y) {
        return;
    }
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = blockIdx.y * matrixTile + y;
    const int col = blockIdx.x * matrixTile + x;
    const int tileCol = blockIdx.x * matrixTile + y;
    __shared__ double left[matrixTile][matrixTile + 1];
    __shared__ double right[matrixTile][matrixTile + 1];
    double sum = 0.0;

    for (int first = 0; first < k; first += matrixTile) {
        const int inner = first + x;
        left[y][x] = row < n && inner < k ? input[static_cast<size_t>(row) * pitch + inner] : 0.0;
        right[y][x] = tileCol < n && inner < k ? input[static_cast<size_t>(tileCol) * pitch + inner] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < matrixTile; ++i) {
            sum = fma(left[y][i], right[x][i], sum);
        }
        __syncthreads();
    }

    if (row < n && col <= row) {
        const size_t index = static_cast<size_t>(row) * pitch + col;
        if constexpr (subtract) {
            output[index] -= sum;
        } else {
            output[index] = sum;
        }
    }
}

template <bool subtract>
void updateWithProduct(cublasHandle_t blas, const double* input, double* output,
                       int n, int k, int pitch) {
    if (n <= smallMatrix) {
        const int tiles = (n + matrixTile - 1) / matrixTile;
        symmetricProduct<subtract><<<dim3(tiles, tiles), dim3(matrixTile, matrixTile)>>>(
            input, output, n, k, pitch);
        check(cudaGetLastError(), "Launching the tiled symmetric product");
        return;
    }

    // The column-major input is the transpose of the row-major input. Its
    // upper-triangle output occupies the row-major lower triangle.
    const double alpha = subtract ? -1.0 : 1.0;
    const double beta = subtract ? 1.0 : 0.0;
    check(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k,
                      &alpha, input, pitch, &beta, output, pitch),
          "Computing the GPU symmetric product");
}

__global__ void factorDiagonal(double* matrix, int n, int first, int width, int* info) {
    if (*info != 0) {
        return;
    }
    __shared__ double tile[panelWidth][panelWidth + 1];
    __shared__ double inverse;
    __shared__ bool validPivot;
    const int col = threadIdx.x;

    for (int row = threadIdx.y; row < width; row += diagonalRowsPerBlock) {
        if (col <= row) {
            tile[row][col] = matrix[static_cast<size_t>(first + row) * n + first + col];
        }
    }
    __syncthreads();

    for (int pivot = 0; pivot < width; ++pivot) {
        if (threadIdx.y == 0 && col == 0) {
            const double value = tile[pivot][pivot];
            validPivot = !(value <= 0.0);
            if (validPivot) {
                inverse = rsqrt(value);
                tile[pivot][pivot] = sqrt(value);
            } else {
                *info = first + pivot + 1;
            }
        }
        __syncthreads();
        if (!validPivot) {
            return;
        }

        if (col == pivot) {
            for (int row = threadIdx.y; row < width; row += diagonalRowsPerBlock) {
                if (row > pivot) {
                    tile[row][pivot] *= inverse;
                }
            }
        }
        __syncthreads();

        if (col > pivot) {
            for (int row = threadIdx.y; row < width; row += diagonalRowsPerBlock) {
                if (col <= row) {
                    tile[row][col] = fma(-tile[row][pivot], tile[col][pivot], tile[row][col]);
                }
            }
        }
        __syncthreads();
    }

    for (int row = threadIdx.y; row < width; row += diagonalRowsPerBlock) {
        if (col <= row) {
            matrix[static_cast<size_t>(first + row) * n + first + col] = tile[row][col];
        }
    }
}

__global__ void solvePanel(double* matrix, int n, int first, const int* info) {
    if (*info != 0) {
        return;
    }
    __shared__ double diagonal[panelWidth][panelWidth + 1];
    __shared__ double inverse[panelWidth];
    const int col = threadIdx.x;
    for (int row = threadIdx.y; row < panelWidth; row += solveRowsPerBlock) {
        if (col <= row) {
            diagonal[col][row] = matrix[static_cast<size_t>(first + row) * n + first + col];
        }
    }
    if (threadIdx.y == 0) {
        inverse[col] = 1.0 / matrix[static_cast<size_t>(first + col) * n + first + col];
    }
    __syncthreads();

    const int row = first + panelWidth + blockIdx.x * solveRowsPerBlock + threadIdx.y;
    if (row < n) {
        const size_t index = static_cast<size_t>(row) * n + first + col;
        double value = matrix[index];
        // Each warp solves one row, broadcasting each completed entry to update the rest.
        for (int pivot = 0; pivot < panelWidth; ++pivot) {
            if (col == pivot) {
                value *= inverse[pivot];
            }
            const double solved = __shfl_sync(0xffffffff, value, pivot);
            if (col > pivot) {
                value = fma(-solved, diagonal[pivot][col], value);
            }
        }
        matrix[index] = value;
    }
}

void factorCholesky(cublasHandle_t blas, double* matrix, int n, int* info) {
    for (int first = 0; first < n; first += panelWidth) {
        const int width = std::min(panelWidth, n - first);
        factorDiagonal<<<1, dim3(panelWidth, diagonalRowsPerBlock)>>>(matrix, n, first, width, info);
        check(cudaGetLastError(), "Launching the diagonal factorization");

        const int next = first + width;
        const int remaining = n - next;
        if (remaining > 0) {
            const int blocks = (remaining + solveRowsPerBlock - 1) / solveRowsPerBlock;
            solvePanel<<<blocks, dim3(panelWidth, solveRowsPerBlock)>>>(matrix, n, first, info);
            check(cudaGetLastError(), "Launching the panel solve");

            const double* panel = matrix + static_cast<size_t>(next) * n + first;
            double* trailing = matrix + static_cast<size_t>(next) * n + next;
            updateWithProduct<true>(blas, panel, trailing, remaining, width, n);
        }
    }
}

__global__ void clearUpperTriangle(double* matrix, int n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y;
    if (row < col && col < n) {
        matrix[row * n + col] = 0.0;
    }
}

struct ErrorMaxima {
    unsigned long long absolute;
    unsigned long long relative;
};

__global__ void reconstructionError(const double* reconstructed, const double* original,
                                    int n, ErrorMaxima* maxima) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y;
    if (static_cast<size_t>(blockIdx.x) * blockDim.x > row) {
        return;
    }
    double absolute = 0.0;
    double relative = 0.0;
    if (col <= row) {
        const size_t offset = row * n + col;
        absolute = fabs(reconstructed[offset] - original[offset]);
        relative = absolute / (fabs(original[offset]) + 1e-10);
    }

    // Ignore NaNs, matching the original maximum initialized to zero.
    absolute = fmax(absolute, 0.0);
    relative = fmax(relative, 0.0);
    for (int stride = warpSize / 2; stride > 0; stride /= 2) {
        absolute = fmax(absolute, __shfl_down_sync(0xffffffff, absolute, stride));
        relative = fmax(relative, __shfl_down_sync(0xffffffff, relative, stride));
    }

    __shared__ double warpAbsolute[blockSize / 32];
    __shared__ double warpRelative[blockSize / 32];
    if (threadIdx.x % warpSize == 0) {
        warpAbsolute[threadIdx.x / warpSize] = absolute;
        warpRelative[threadIdx.x / warpSize] = relative;
    }
    __syncthreads();

    if (threadIdx.x < warpSize) {
        absolute = threadIdx.x < blockSize / warpSize ? warpAbsolute[threadIdx.x] : 0.0;
        relative = threadIdx.x < blockSize / warpSize ? warpRelative[threadIdx.x] : 0.0;
        for (int stride = warpSize / 2; stride > 0; stride /= 2) {
            absolute = fmax(absolute, __shfl_down_sync(0xffffffff, absolute, stride));
            relative = fmax(relative, __shfl_down_sync(0xffffffff, relative, stride));
        }
        if (threadIdx.x == 0) {
            // Unsigned bit ordering agrees with floating-point ordering for nonnegative doubles.
            atomicMax(&maxima->absolute, static_cast<unsigned long long>(__double_as_longlong(absolute)));
            atomicMax(&maxima->relative, static_cast<unsigned long long>(__double_as_longlong(relative)));
        }
    }
}

bool validateCholesky(cublasHandle_t blas, const double* factor, const double* original, int n) {
    double maxError = 0.0;
    double relError = 0.0;
    if (n > 0) {
        DeviceBuffer<double> reconstructed(static_cast<size_t>(n) * n);
        DeviceBuffer<ErrorMaxima> maxima(1);
        check(cudaMemset(maxima.data, 0, sizeof(ErrorMaxima)), "Initializing validation maxima");
        updateWithProduct<false>(blas, factor, reconstructed.data, n, n, n);
        const dim3 grid((n + blockSize - 1) / blockSize, n);
        reconstructionError<<<grid, blockSize>>>(reconstructed.data, original, n, maxima.data);
        check(cudaGetLastError(), "Launching validation reduction");

        ErrorMaxima result;
        check(cudaMemcpy(&result, maxima.data, sizeof(result), cudaMemcpyDeviceToHost),
              "Reading validation maxima");
        std::memcpy(&maxError, &result.absolute, sizeof(maxError));
        std::memcpy(&relError, &result.relative, sizeof(relError));
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printPerformance(size_t n, float milliseconds) {
    printf("Computation time: %.3f ms\n", milliseconds);
    const double operations = static_cast<double>(n) * n * n / 3.0;
    const double gflops = milliseconds > 0.0f ? operations / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
}

int runBenchmark(int n, bool validate, bool printResults) {
    printf("Generating positive definite matrix...\n");
    if (n == 0) {
        printf("Computing Cholesky decomposition...\n");
        printPerformance(0, 0.0f);
        if (printResults) {
            print_results(std::vector<double>{}, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            validateCholesky(nullptr, nullptr, nullptr, 0);
            printf("Validation: PASSED\n");
        }
        return 0;
    }

    const size_t count = static_cast<size_t>(n) * n;
    const size_t bytes = count * sizeof(double);
    const int blocks = (n + blockSize - 1) / blockSize;
    CudaHandles cuda;
    if (n > smallMatrix) {
        check(cublasCreate(&cuda.blas), "Creating the cuBLAS handle");
    }
    check(cudaEventCreate(&cuda.start), "Creating the start event");
    check(cudaEventCreate(&cuda.stop), "Creating the stop event");

    DeviceBuffer<double> matrix(count);
    DeviceBuffer<double> input(count);
    {
        std::vector<double> random(count);
        unsigned int seed = 42;
        for (double& value : random) {
            value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
        }
        check(cudaMemcpy(input.data, random.data(), bytes, cudaMemcpyHostToDevice),
              "Uploading the seeded matrix");
    }
    updateWithProduct<false>(cuda.blas, input.data, matrix.data, n, n, n);
    addDiagonal<<<blocks, blockSize>>>(matrix.data, n);
    check(cudaGetLastError(), "Launching the diagonal shift");
    if (validate) {
        check(cudaMemcpyAsync(input.data, matrix.data, bytes, cudaMemcpyDeviceToDevice),
              "Saving the original matrix on the GPU");
    }

    DeviceBuffer<int> info(1);
    check(cudaMemset(info.data, 0, sizeof(int)), "Initializing the Cholesky status");

    printf("Computing Cholesky decomposition...\n");
    check(cudaEventRecord(cuda.start), "Recording the start event");
    factorCholesky(cuda.blas, matrix.data, n, info.data);
    check(cudaEventRecord(cuda.stop), "Recording the stop event");

    int factorStatus;
    check(cudaMemcpy(&factorStatus, info.data, sizeof(factorStatus), cudaMemcpyDeviceToHost),
          "Reading the Cholesky status");
    if (factorStatus > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", factorStatus - 1);
        printf("Cholesky decomposition failed\n");
        return 1;
    }

    float milliseconds;
    check(cudaEventElapsedTime(&milliseconds, cuda.start, cuda.stop), "Measuring GPU Cholesky time");
    printPerformance(n, milliseconds);

    if (printResults || validate) {
        clearUpperTriangle<<<dim3(blocks, n), blockSize>>>(matrix.data, n);
        check(cudaGetLastError(), "Launching upper-triangle clearing");
    }
    if (printResults) {
        std::vector<double> factor(count);
        check(cudaMemcpy(factor.data(), matrix.data, bytes, cudaMemcpyDeviceToHost), "Downloading the factor");
        print_results(factor, "CholeskyL");
    }
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateCholesky(cuda.blas, matrix.data, input.data, n);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

} // namespace

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // The CUDA libraries accept int dimensions and the row-wise kernels use grid.y.
    if (n > 65535 || (n != 0 && n > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
        fprintf(stderr, "Error: Matrix size is too large\n");
        return 1;
    }
    try {
        return runBenchmark(static_cast<int>(n), validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
