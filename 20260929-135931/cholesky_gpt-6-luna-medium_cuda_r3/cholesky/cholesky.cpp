#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return false; } } while (0)

// Factor the small diagonal block. One thread performs the dependent pivots;
// the much larger panel and trailing updates are handled by parallel kernels.
__global__ void diagonalBlock(double* a, size_t n, size_t base, int width, int* failed) {
    if (threadIdx.x || blockIdx.x) return;
    for (int j = 0; j < width; ++j) {
        const size_t d = base + j;
        double v = a[d*n+d];
        for (int k = 0; k < j; ++k) { double x=a[d*n+base+k]; v-=x*x; }
        if (!(v > 0.0)) { *failed = static_cast<int>(d+1); return; }
        a[d*n+d] = sqrt(v);
        for (int r=j+1; r<width; ++r) {
            const size_t row=base+r;
            double x=a[row*n+d];
            for (int k=0;k<j;++k) x-=a[row*n+base+k]*a[d*n+base+k];
            a[row*n+d]=x/a[d*n+d];
        }
    }
}

// Solve L_panel * X^T = A_panel^T, one output element per CUDA thread.
__global__ void solvePanel(double* a, size_t n, size_t base, int width) {
    size_t col = blockIdx.x*blockDim.x + threadIdx.x;
    size_t row = blockIdx.y*blockDim.y + threadIdx.y + base + width;
    if (row >= n || col >= static_cast<size_t>(width)) return;
    for (int k=0;k<width;++k) {
        double x=a[row*n+base+k];
        for (int q=0;q<k;++q) x-=a[row*n+base+q]*a[(base+k)*n+base+q];
        a[row*n+base+k]=x/a[(base+k)*n+base+k];
    }
}

// Symmetric rank-k update of the remaining lower triangle.
__global__ void trailingUpdate(double* a, size_t n, size_t base, int width) {
    size_t j=base+width+blockIdx.x*blockDim.x+threadIdx.x;
    size_t i=base+width+blockIdx.y*blockDim.y+threadIdx.y;
    if (i>=n || j>=n || i<j) return;
    double x=a[i*n+j];
    for (int k=0;k<width;++k) x-=a[i*n+base+k]*a[j*n+base+k];
    a[i*n+j]=x;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n) {
    double* dA=nullptr; int* dFailed=nullptr;
    CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(double)));
    if (cudaMalloc(&dFailed, sizeof(int)) != cudaSuccess) { cudaFree(dA); return false; }
    if (cudaMemcpy(dA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess || cudaMemset(dFailed,0,sizeof(int))!=cudaSuccess) {
        cudaFree(dA); cudaFree(dFailed); return false;
    }
    constexpr int block=32;
    int failed=0;
    for (size_t base=0;base<n;base+=block) {
        int width=static_cast<int>(std::min<size_t>(block,n-base));
        diagonalBlock<<<1,1>>>(dA,n,base,width,dFailed);
        if (base+width<n) {
            dim3 threads(32,8);
            dim3 grid((width+threads.x-1)/threads.x,(n-base-width+threads.y-1)/threads.y);
            solvePanel<<<grid,threads>>>(dA,n,base,width);
            dim3 t2(16,16), g2((n-base-width+t2.x-1)/t2.x,(n-base-width+t2.y-1)/t2.y);
            trailingUpdate<<<g2,t2>>>(dA,n,base,width);
        }
        if (cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess) { cudaFree(dA); cudaFree(dFailed); return false; }
        if (cudaMemcpy(&failed,dFailed,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) { cudaFree(dA); cudaFree(dFailed); return false; }
        if (failed) { printf("Error: Matrix is not positive definite at diagonal element %d\n",failed-1); cudaFree(dA); cudaFree(dFailed); return false; }
    }
    bool ok=cudaMemcpy(A.data(),dA,A.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess;
    cudaFree(dA); cudaFree(dFailed);
    return ok;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    for (size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    for (size_t i=0;i<n;++i) for (size_t j=0;j<n;++j) {
        double s=0; for(size_t k=0;k<n;++k) s+=B[i*n+k]*B[j*n+k]; A[i*n+j]=s;
    }
    for(size_t i=0;i<n;++i) A[i*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& orig,size_t n) {
    double maxError=0,relError=0;
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) {
        double sum=0; for(size_t k=0;k<=std::min(i,j);++k) sum+=L[i*n+k]*L[j*n+k];
        double e=fabs(sum-orig[i*n+j]); maxError=std::max(maxError,e);
        relError=std::max(relError,e/(fabs(orig[i*n+j])+1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxError,relError);
    if(relError>1e-6){printf("Validation failed: relative error too large\n");return false;} return true;
}

void printUsage(const char* p){printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",p);}

int main(int argc,char** argv){
    size_t n=512; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0&&i+1<argc)n=atoi(argv[++i]);
        else if(strcmp(argv[i],"-v")==0)validate=true;
        else if(strcmp(argv[i],"-r")==0)printResults=true;
        else if(strcmp(argv[i],"-h")==0){printUsage(argv[0]);return 0;}
        else {printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);return 1;}
    }
    printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",n,n,validate?"enabled":"disabled");
    std::vector<double>A(n*n),orig; printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A,n);
    if(validate)orig=A;
    printf("Computing Cholesky decomposition...\n"); auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(A,n); auto end=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    if(!success){printf("Cholesky decomposition failed\n");return 1;}
    printf("Computation time: %ld ms\n",duration.count());
    double ops=(double)n*n*n/3.0; double seconds=std::chrono::duration<double>(end-start).count();
    printf("Performance: %.3f GFLOPS\n",seconds>0?ops/seconds/1e9:0.0);
    if(printResults) print_results(A,"CholeskyL");
    if(validate){printf("Validating result...\n");bool valid=validateCholesky(A,orig,n);printf("Validation: %s\n",valid?"PASSED":"FAILED");return valid?0:1;}
    return 0;
}
