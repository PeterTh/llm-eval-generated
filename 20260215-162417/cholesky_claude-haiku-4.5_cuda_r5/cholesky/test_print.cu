#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void choleskyOffDiagonalKernel(double* d_A, const size_t n, const size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < i && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += d_A[i * n + k] * d_A[j * n + k];
        }
        d_A[i * n + j] = (d_A[i * n + j] - sum) / d_A[j * n + j];
    }
}

__global__ void zeroUpperTriangularRowKernel(double* d_A, const size_t n, const size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j > i && j < n) {
        d_A[i * n + j] = 0.0;
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
    
    for (size_t i = 0; i < n; ++i) {
        // Compute diagonal
        double diag_val;
        cudaMemcpy(&diag_val, d_A + i * n + i, sizeof(double), cudaMemcpyDeviceToHost);
        
        double sum = 0.0;
        if (i > 0) {
            std::vector<double> row_i(i);
            cudaMemcpy(row_i.data(), d_A + i * n, i * sizeof(double), cudaMemcpyDeviceToHost);
            for (size_t k = 0; k < i; ++k) {
                sum += row_i[k] * row_i[k];
            }
        }
        
        diag_val = diag_val - sum;
        diag_val = sqrt(diag_val);
        cudaMemcpy(d_A + i * n + i, &diag_val, sizeof(double), cudaMemcpyHostToDevice);
        
        printf("Row %zu: diag = %.6f\n", i, diag_val);
        
        // Compute off-diagonal
        if (i > 0) {
            int threads = 256;
            int blocks = (i + threads - 1) / threads;
            choleskyOffDiagonalKernel<<<blocks, threads>>>(d_A, n, i);
            cudaDeviceSynchronize();
        }
        
        // Zero upper
        int threads = 256;
        int blocks = (n - i - 1 + threads - 1) / threads;
        if (blocks > 0) {
            zeroUpperTriangularRowKernel<<<blocks, threads>>>(d_A, n, i);
            cudaDeviceSynchronize();
        }
    }
    
    std::vector<double> result(n * n);
    cudaMemcpy(result.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("\nGPU Result:\n");
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            printf("%.6f ", result[i * n + j]);
        }
        printf("\n");
    }
    
    printf("\nExpected:\n");
    printf("2.018340 0.000000 0.000000 0.000000\n");
    printf("-0.000469 2.165530 0.000000 0.000000\n");
    printf("0.003106 0.122096 2.042565 0.000000\n");
    printf("-0.010381 0.088180 0.082883 2.066341\n");
    
    cudaFree(d_A);
    return 0;
}
