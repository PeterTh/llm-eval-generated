#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void choleskyOffDiagonalKernel(double* d_A, const size_t n, const size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < i && j < n) {
        printf("Processing element [%zu,%zu]\n", i, j);
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += d_A[i * n + k] * d_A[j * n + k];
        }
        printf("  Original d_A[%zu,%zu] = %.6f, sum = %.6f, d_A[%zu,%zu] = %.6f\n", i, j, d_A[i*n+j], sum, j, j, d_A[j*n+j]);
        d_A[i * n + j] = (d_A[i * n + j] - sum) / d_A[j * n + j];
        printf("  New d_A[%zu,%zu] = %.6f\n", i, j, d_A[i*n+j]);
    }
}

int main() {
    size_t n = 4;
    double A[] = {
        4.073697, -0.000946, 0.006269, -0.020953,
        -0.000946, 4.689520, 0.264401, 0.190961,
        0.006269, 0.264401, 4.186988, 0.180028,
        -0.020953, 0.190961, 0.180028, 4.284519
    };
    
    double* d_A;
    cudaMalloc(&d_A, n * n * sizeof(double));
    cudaMemcpy(d_A, A, n * n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Compute diagonals for rows 0 and 1
    for (size_t i = 0; i < 2; ++i) {
        double diag;
        cudaMemcpy(&diag, d_A + i * n + i, sizeof(double), cudaMemcpyDeviceToHost);
        diag = sqrt(diag);
        cudaMemcpy(d_A + i * n + i, &diag, sizeof(double), cudaMemcpyHostToDevice);
    }
    
    // Now run kernel for row 1
    printf("Running kernel for row 1...\n");
    int threads = 256;
    int blocks = (1 + threads - 1) / threads;
    choleskyOffDiagonalKernel<<<blocks, threads>>>(d_A, n, 1);
    cudaDeviceSynchronize();
    
    // Check result
    std::vector<double> result(n * n);
    cudaMemcpy(result.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Result A[1,0] = %.6f\n", result[1 * n + 0]);
    
    cudaFree(d_A);
    return 0;
}
