#include <cuda_runtime.h>
#include <cstring>

// CUDA kernel for matrix multiplication
__global__ void matmulKernel(const double* A, const double* B, double* C, 
                            size_t N, size_t rowStart, size_t rowEnd) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + rowStart;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rowEnd && j < N) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

extern "C" void matmulGPU(const double* h_A, const double* h_B, double* h_C,
                         size_t N, size_t rowStart, size_t rowEnd) {
    int cudaDeviceCount = 0;
    cudaGetDeviceCount(&cudaDeviceCount);
    
    if (cudaDeviceCount <= 0 || rowStart >= rowEnd) {
        return;
    }
    
    // Set device
    int device = 0;
    cudaSetDevice(device % cudaDeviceCount);
    
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));
    
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((rowEnd - rowStart + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, rowStart, rowEnd);
    cudaDeviceSynchronize();
    
    // Copy only the rows this rank computed
    cudaMemcpy(h_C + rowStart * N, d_C + rowStart * N, 
               (rowEnd - rowStart) * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

