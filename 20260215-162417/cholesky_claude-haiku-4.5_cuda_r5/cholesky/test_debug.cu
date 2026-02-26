#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

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
    
    // Manually compute rows 0 and 1
    for (size_t i = 0; i < 2; ++i) {
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
        
        // Compute off-diagonals manually
        if (i > 0) {
            for (size_t j = 0; j < i; ++j) {
                double val;
                cudaMemcpy(&val, d_A + i * n + j, sizeof(double), cudaMemcpyDeviceToHost);
                
                double sum_ij = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    double ik, jk;
                    cudaMemcpy(&ik, d_A + i * n + k, sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&jk, d_A + j * n + k, sizeof(double), cudaMemcpyDeviceToHost);
                    sum_ij += ik * jk;
                }
                
                double ljj;
                cudaMemcpy(&ljj, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost);
                val = (val - sum_ij) / ljj;
                cudaMemcpy(d_A + i * n + j, &val, sizeof(double), cudaMemcpyHostToDevice);
            }
        }
        
        // Zero upper
        for (size_t j = i + 1; j < n; ++j) {
            double zero = 0.0;
            cudaMemcpy(d_A + i * n + j, &zero, sizeof(double), cudaMemcpyHostToDevice);
        }
    }
    
    // Now check row 2
    std::vector<double> row2(2);
    cudaMemcpy(row2.data(), d_A + 2 * n, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Row 2, columns 0-1: %.6f, %.6f\n", row2[0], row2[1]);
    
    double sum2 = row2[0] * row2[0] + row2[1] * row2[1];
    printf("Sum of squares: %.6f\n", sum2);
    
    double orig_a22;
    cudaMemcpy(&orig_a22, d_A + 2 * n + 2, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Original A[2,2]: %.6f\n", orig_a22 + sum2);
    printf("After subtracting sum: %.6f\n", orig_a22);
    printf("Square root: %.6f\n", sqrt(orig_a22));
    
    cudaFree(d_A);
    return 0;
}
