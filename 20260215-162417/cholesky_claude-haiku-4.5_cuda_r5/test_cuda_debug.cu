#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// Kernel for updating column j
__global__ void updateColumnKernel(double* d_A, const size_t n, const size_t j) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n - j - 1) {
        size_t i = idx + j + 1;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += d_A[i * n + k] * d_A[j * n + k];
        }
        d_A[i * n + j] = (d_A[i * n + j] - sum) / d_A[j * n + j];
    }
}

// Kernel for updating submatrix
__global__ void updateSubmatrixKernel(double* d_A, const size_t n, const size_t j) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n - j - 1 && idy < n - j - 1) {
        size_t i = idx + j + 1;
        size_t k = idy + j + 1;
        d_A[i * n + k] -= d_A[i * n + j] * d_A[k * n + j];
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
    
    // Process column by column
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal
        double diag;
        cudaMemcpy(&diag, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost);
        
        double sum = 0.0;
        if (j > 0) {
            std::vector<double> row(j);
            cudaMemcpy(row.data(), d_A + j * n, j * sizeof(double), cudaMemcpyDeviceToHost);
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }
        }
        
        diag = diag - sum;
        diag = sqrt(diag);
        cudaMemcpy(d_A + j * n + j, &diag, sizeof(double), cudaMemcpyHostToDevice);
        
        printf("Column %zu: diag = %.6f\n", j, diag);
        
        // Update column j
        if (j < n - 1) {
            updateColumnKernel<<<1, 256>>>(d_A, n, j);
            cudaDeviceSynchronize();
            
            // Update submatrix
            updateSubmatrixKernel<<<1, dim3(16, 16)>>>(d_A, n, j);
            cudaDeviceSynchronize();
        }
    }
    
    // Copy back
    std::vector<double> result(n * n);
    cudaMemcpy(result.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Result:\n");
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.6f ", result[i * n + j]);
        }
        printf("\n");
    }
    
    printf("Expected:\n");
    printf("2.018340 0.000000 0.000000 0.000000\n");
    printf("-0.000469 2.165530 0.000000 0.000000\n");
    printf("0.003106 0.122096 2.042565 0.000000\n");
    printf("-0.010381 0.088180 0.082883 2.066341\n");
    
    cudaFree(d_A);
    return 0;
}
