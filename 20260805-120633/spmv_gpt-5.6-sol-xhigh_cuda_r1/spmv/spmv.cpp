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

constexpr double MAX_RELATIVE_ERROR = 0.02;
constexpr unsigned int CUDA_BLOCK_SIZE = 256;

// Fail immediately on a CUDA error.  Continuing after a failed allocation or
// launch would otherwise produce misleading benchmark results.
void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
                     expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }
    const T* get() const { return data_; }

  private:
    T* data_ = nullptr;
};

void fill(double* values, const index_t count, const double maxValue) {
    for (index_t i = 0; i < count; ++i) {
        values[i] = maxValue * (rand() / (static_cast<double>(RAND_MAX) + 1.0));
    }
}

// Construct exactly the same row-major random matrix as the original code.
// The position arithmetic is widened so valid large dimensions do not wrap at
// 2^32 while the CSR indices themselves remain compact 32-bit values.
void initRandomMatrix(index_t* cols, index_t* rowDelimiters, const index_t nonzeros,
                      const index_t dim) {
    index_t assigned = 0;
    const uint64_t totalEntries = static_cast<uint64_t>(dim) * dim;
    const double probability = static_cast<double>(nonzeros) /
                               static_cast<double>(totalEntries);

    srand(8675309);

    bool fillRemaining = false;
    for (index_t row = 0; row < dim; ++row) {
        rowDelimiters[row] = assigned;
        for (index_t col = 0; col < dim; ++col) {
            const uint64_t position = static_cast<uint64_t>(row) * dim + col;
            const uint64_t entriesLeft = totalEntries - position;
            const index_t needed = nonzeros - assigned;
            if (entriesLeft <= needed) {
                fillRemaining = true;
            }
            const double randomValue = static_cast<double>(rand()) / RAND_MAX;
            if ((assigned < nonzeros && randomValue <= probability) || fillRemaining) {
                cols[assigned++] = col;
            }
        }
    }
    rowDelimiters[dim] = nonzeros;
}

void spmvCpu(const double* val, const index_t* cols, const index_t* rowDelimiters,
             const double* vec, const index_t dim, double* out) {
    for (index_t row = 0; row < dim; ++row) {
        double sum = 0.0;
        for (index_t element = rowDelimiters[row];
             element < rowDelimiters[row + 1]; ++element) {
            sum += val[element] * vec[cols[element]];
        }
        out[row] = sum;
    }
}

// For short and medium rows, multiple independent rows share a warp.  The
// compile-time subwarp width avoids idle lanes without sacrificing coalesced
// CSR reads.  Every lane participates in the shuffle, including lanes beyond
// the final row, so the shuffle mask is always valid.
template <unsigned int THREADS_PER_ROW>
__global__ void spmvSubwarpKernel(const double* __restrict__ val,
                                  const index_t* __restrict__ cols,
                                  const index_t* __restrict__ rowDelimiters,
                                  const double* __restrict__ vec,
                                  index_t dim,
                                  double* __restrict__ out) {
    static_assert(THREADS_PER_ROW >= 1 && THREADS_PER_ROW <= 32,
                  "A row group must fit in a warp");
    static_assert((THREADS_PER_ROW & (THREADS_PER_ROW - 1)) == 0,
                  "The row group width must be a power of two");

    const uint64_t globalThread =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t row64 = globalThread / THREADS_PER_ROW;
    const unsigned int lane = threadIdx.x & (THREADS_PER_ROW - 1);
    const bool validRow = row64 < dim;

    double sum = 0.0;
    if (validRow) {
        const index_t row = static_cast<index_t>(row64);
        const index_t begin = rowDelimiters[row];
        const index_t end = rowDelimiters[row + 1];
        for (index_t element = begin + lane; element < end;
             element += THREADS_PER_ROW) {
            sum += val[element] * vec[cols[element]];
        }
    }

    if constexpr (THREADS_PER_ROW > 1) {
#pragma unroll
        for (unsigned int offset = THREADS_PER_ROW / 2; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
        }
    }

    if (validRow && lane == 0) {
        out[static_cast<index_t>(row64)] = sum;
    }
}

// Very long rows benefit from the additional memory-level parallelism of a
// full block.  The reduction uses shuffles within each warp and only one
// synchronization to combine the warp sums.
__global__ void spmvBlockKernel(const double* __restrict__ val,
                                const index_t* __restrict__ cols,
                                const index_t* __restrict__ rowDelimiters,
                                const double* __restrict__ vec,
                                index_t dim,
                                double* __restrict__ out) {
    __shared__ double warpSums[CUDA_BLOCK_SIZE / 32];

    const index_t row = blockIdx.x;
    if (row >= dim) {
        return;
    }

    const index_t begin = rowDelimiters[row];
    const index_t end = rowDelimiters[row + 1];
    double sum = 0.0;
    for (index_t element = begin + threadIdx.x; element < end;
         element += blockDim.x) {
        sum += val[element] * vec[cols[element]];
    }

#pragma unroll
    for (unsigned int offset = 16; offset != 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5u;
    if (lane == 0) {
        warpSums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < (CUDA_BLOCK_SIZE / 32) ? warpSums[lane] : 0.0;
#pragma unroll
        for (unsigned int offset = 16; offset != 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            out[row] = sum;
        }
    }
}

enum class KernelKind {
    Scalar,
    Subwarp2,
    Subwarp4,
    Subwarp8,
    Subwarp16,
    Warp,
    Block
};

KernelKind selectKernel(index_t nonzeros, index_t rows) {
    const double average = static_cast<double>(nonzeros) / rows;
    if (average <= 2.0) return KernelKind::Scalar;
    if (average <= 4.0) return KernelKind::Subwarp2;
    if (average <= 8.0) return KernelKind::Subwarp4;
    if (average <= 16.0) return KernelKind::Subwarp8;
    if (average <= 32.0) return KernelKind::Subwarp16;
    if (average <= 512.0) return KernelKind::Warp;
    return KernelKind::Block;
}

template <unsigned int THREADS_PER_ROW>
void launchSubwarp(const double* val, const index_t* cols,
                   const index_t* rowDelimiters, const double* vec,
                   index_t rows, double* out, cudaStream_t stream) {
    const uint64_t threadCount = static_cast<uint64_t>(rows) * THREADS_PER_ROW;
    const unsigned int blocks = static_cast<unsigned int>(
        (threadCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    spmvSubwarpKernel<THREADS_PER_ROW><<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
        val, cols, rowDelimiters, vec, rows, out);
}

void launchSpmv(KernelKind kind, const double* val, const index_t* cols,
                const index_t* rowDelimiters, const double* vec,
                index_t rows, double* out, cudaStream_t stream = nullptr) {
    switch (kind) {
        case KernelKind::Scalar:
            launchSubwarp<1>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Subwarp2:
            launchSubwarp<2>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Subwarp4:
            launchSubwarp<4>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Subwarp8:
            launchSubwarp<8>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Subwarp16:
            launchSubwarp<16>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Warp:
            launchSubwarp<32>(val, cols, rowDelimiters, vec, rows, out, stream);
            break;
        case KernelKind::Block:
            spmvBlockKernel<<<rows, CUDA_BLOCK_SIZE, 0, stream>>>(
                val, cols, rowDelimiters, vec, rows, out);
            break;
    }
}

bool verifyResults(const double* reference, const double* result, const index_t size) {
    for (index_t i = 0; i < size; ++i) {
        const double ref = reference[i];
        const double res = result[i];
        if (std::abs(ref) < 1e-10) {
            if (std::abs(res) > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e\n",
                            i, ref, res);
                return false;
            }
        } else {
            const double relativeError = std::abs((res - ref) / ref);
            if (relativeError > MAX_RELATIVE_ERROR) {
                std::printf("Validation failed at index %u: reference %.10e, got %.10e "
                            "(rel error: %.10e)\n",
                            i, ref, res, relativeError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of rows/columns in the matrix (default: 1024)\n");
    std::printf("  -s <num>     Sparsity: 1 out of N entries is non-zero (default: 10)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -m <val>     Maximum value for matrix and vector elements (default: 1.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    index_t numRows = 1024;
    index_t sparsity = 10;
    index_t iterations = 10;
    double maxVal = 1.0;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numRows = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sparsity = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = static_cast<index_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            maxVal = std::strtod(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (numRows == 0 || sparsity == 0 || iterations == 0) {
        std::fprintf(stderr, "Error: -n, -s, and -i must all be greater than zero.\n");
        return 1;
    }

    const uint64_t totalEntries = static_cast<uint64_t>(numRows) * numRows;
    const uint64_t nonzeros64 = totalEntries / sparsity;
    if (nonzeros64 > std::numeric_limits<index_t>::max()) {
        std::fprintf(stderr,
                     "Error: the requested matrix has too many nonzeros for 32-bit CSR indices.\n");
        return 1;
    }
    const index_t nonzeros = static_cast<index_t>(nonzeros64);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    std::printf("Sparse Matrix-Vector Multiplication (SpMV) Benchmark\n");
    std::printf("Matrix size: %u x %u\n", numRows, numRows);
    std::printf("Sparsity: 1 out of %u entries is non-zero\n", sparsity);
    std::printf("Non-zero elements: %u (%.2f%% sparse)\n", nonzeros,
                100.0 * (1.0 - static_cast<double>(nonzeros) /
                                   static_cast<double>(totalEntries)));
    std::printf("Iterations: %u\n", iterations);
    std::printf("Max value: %.2f\n", maxVal);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    std::printf("CUDA device: %s\n", deviceProperties.name);

    std::vector<double> hVal(nonzeros);
    std::vector<index_t> hCols(nonzeros);
    std::vector<index_t> hRowDelimiters(static_cast<size_t>(numRows) + 1);
    std::vector<double> hVec(numRows);
    std::vector<double> hOut(numRows);

    std::printf("Initializing data structures...\n");
    fill(hVec.data(), numRows, maxVal);
    fill(hVal.data(), nonzeros, maxVal);
    initRandomMatrix(hCols.data(), hRowDelimiters.data(), nonzeros, numRows);

    std::vector<double> hReference;
    if (validate) {
        std::printf("Computing reference solution...\n");
        hReference.resize(numRows);
        spmvCpu(hVal.data(), hCols.data(), hRowDelimiters.data(),
                hVec.data(), numRows, hReference.data());
    }

    DeviceBuffer<double> dVal(nonzeros);
    DeviceBuffer<index_t> dCols(nonzeros);
    DeviceBuffer<index_t> dRowDelimiters(static_cast<size_t>(numRows) + 1);
    DeviceBuffer<double> dVec(numRows);
    DeviceBuffer<double> dOut(numRows);

    if (nonzeros != 0) {
        CUDA_CHECK(cudaMemcpy(dVal.get(), hVal.data(),
                              static_cast<size_t>(nonzeros) * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCols.get(), hCols.data(),
                              static_cast<size_t>(nonzeros) * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(dRowDelimiters.get(), hRowDelimiters.data(),
                          (static_cast<size_t>(numRows) + 1) * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVec.get(), hVec.data(),
                          static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyHostToDevice));

    const KernelKind kernel = selectKernel(nonzeros, numRows);

    // Warm up the selected kernel so context setup and first-use costs do not
    // contaminate the measured SpMV iterations.
    launchSpmv(kernel, dVal.get(), dCols.get(), dRowDelimiters.get(), dVec.get(),
               numRows, dOut.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("Computing SpMV...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (index_t iteration = 0; iteration < iterations; ++iteration) {
        launchSpmv(kernel, dVal.get(), dCols.get(), dRowDelimiters.get(), dVec.get(),
                   numRows, dOut.get());
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(hOut.data(), dOut.get(),
                          static_cast<size_t>(numRows) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));
    const double averageMs = elapsedMs / static_cast<double>(iterations);
    const double gflops = (2.0 * nonzeros * iterations) /
                          (static_cast<double>(elapsedMs) * 1.0e6);
    std::printf("Average time per iteration: %.3f ms\n", averageMs);
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(hOut, "OutputVector");
    }

    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = verifyResults(hReference.data(), hOut.data(), numRows);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }

    return 0;
}
