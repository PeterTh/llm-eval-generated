#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky. The panel is factored in dependency order;
// its trailing solve and the Schur-complement update run in parallel.
constexpr int BLOCK_SIZE = 32;

__global__ void factorDiagonal(double* a, size_t n, size_t base, int width, int* ok) {
    if (blockIdx.x || threadIdx.x) return;
    for (int r = 0; r < width; ++r) {
        const size_t i = base + r;
        double x = a[i*n+i];
        for (int k = 0; k < r; ++k) x -= a[i*n+base+k] * a[i*n+base+k];
        if (!(x > 0.0)) { *ok = 0; return; }
        a[i*n+i] = sqrt(x);
        for (int c = r+1; c < width; ++c) {
            double v = a[(base+c)*n+i];
            for (int k = 0; k < r; ++k) v -= a[(base+c)*n+base+k] * a[i*n+base+k];
            a[(base+c)*n+i] = v / a[i*n+i];
        }
    }
    for (int r=0; r<width; ++r)
        for (int c=r+1; c<width; ++c) a[(base+r)*n+base+c]=0.0;
}

__global__ void solvePanelColumn(double* a, size_t n, size_t base, int width, int col) {
    size_t row = base + width + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    const size_t j = base + col;
    double v = a[row*n+j];
    for (int k=0; k<col; ++k) v -= a[row*n+base+k] * a[j*n+base+k];
    a[row*n+j] = v / a[j*n+j];
}

__global__ void updateTrailing(double* a, size_t n, size_t base, int width) {
    size_t j = base + width + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = base + width + (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j > i) return;
    double v = a[i*n+j];
    for (int k=0; k<width; ++k) v -= a[i*n+base+k] * a[j*n+base+k];
    a[i*n+j] = v;
    if (i != j) a[j*n+i] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* dA = nullptr; int* dOk = nullptr; int ok = 1;
    if (cudaMalloc(&dA, A.size()*sizeof(double)) != cudaSuccess || cudaMalloc(&dOk, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "CUDA allocation failed\n"); if(dA) cudaFree(dA); return false;
    }
    cudaMemcpy(dA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dOk, &ok, sizeof(int), cudaMemcpyHostToDevice);
    for (size_t base=0; base<n; base+=BLOCK_SIZE) {
        int width = (int)std::min((size_t)BLOCK_SIZE, n-base);
        factorDiagonal<<<1,1>>>(dA,n,base,width,dOk);
        if (cudaMemcpy(&ok,dOk,sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess || !ok) {
            size_t bad=base; cudaMemcpy(A.data(),dA,A.size()*sizeof(double),cudaMemcpyDeviceToHost);
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",bad);
            cudaFree(dA); cudaFree(dOk); return false;
        }
        for (int col=0; col<width; ++col) {
            int count=(int)(n-base-width);
            if(count>0) solvePanelColumn<<<(count+255)/256,256>>>(dA,n,base,width,col);
        }
        int remain=(int)(n-base-width);
        if(remain>0) updateTrailing<<<dim3((remain+15)/16,(remain+15)/16),dim3(16,16)>>>(dA,n,base,width);
    }
    cudaError_t err=cudaDeviceSynchronize();
    if(err==cudaSuccess) err=cudaMemcpy(A.data(),dA,A.size()*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dOk);
    if(err!=cudaSuccess) { fprintf(stderr,"CUDA Cholesky failed: %s\n",cudaGetErrorString(err)); return false; }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
