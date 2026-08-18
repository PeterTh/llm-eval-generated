#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

static void cuda_check(cudaError_t e, const char *where) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); }
}

__global__ void cholesky_diag(double *a, size_t n, size_t j) {
    extern __shared__ double s[];
    double x = 0.0;
    for (size_t k = threadIdx.x; k < j; k += blockDim.x) { double v = a[j*n+k]; x += v*v; }
    s[threadIdx.x] = x; __syncthreads();
    for (unsigned d = blockDim.x/2; d; d >>= 1) { if (threadIdx.x < d) s[threadIdx.x] += s[threadIdx.x+d]; __syncthreads(); }
    if (threadIdx.x == 0) a[j*n+j] = sqrt(a[j*n+j] - s[0]);
}

__global__ void cholesky_column(double *a, size_t n, size_t k, size_t first, size_t count) {
    size_t i=first+blockIdx.x*blockDim.x+threadIdx.x; if(i>=first+count) return;
    double sum = 0.0;
    for (size_t p=0; p<k; ++p) sum += a[i*n+p]*a[k*n+p];
    a[i*n+k]=(a[i*n+k]-sum)/a[k*n+k];
}
__global__ void cholesky_update(double *a, size_t n, size_t k, size_t first, size_t count) {
    size_t q=blockIdx.x*blockDim.x+threadIdx.x, cols=n-k-1;
    size_t i=first+q/cols, j=k+1+q%cols; if(q>=count*cols) return;
    a[i*n+j]-=a[i*n+k]*a[j*n+k];
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned seed=42;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n*n; ++i) { unsigned s=seed+(unsigned)i*747796405u; B[i]=(rand_r(&s)/(double)RAND_MAX)-.5; }
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i) for (size_t j=0;j<n;++j) { double sum=0; for(size_t k=0;k<n;++k) sum+=B[i*n+k]*B[j*n+k]; A[i*n+j]=sum; }
    #pragma omp parallel for
    for (long long i=0;i<(long long)n;++i) A[i*n+i]+=n;
}

bool choleskyDecomposition(std::vector<double>& a, size_t n, int rank, int nranks) {
    size_t first=(n*rank)/nranks, last=(n*(rank+1))/nranks;
    int devices=0; cuda_check(cudaGetDeviceCount(&devices),"cudaGetDeviceCount"); if(!devices) MPI_Abort(MPI_COMM_WORLD,1); cuda_check(cudaSetDevice(rank%devices),"cudaSetDevice");
    double *d=nullptr; cuda_check(cudaMalloc(&d,n*n*sizeof(double)),"cudaMalloc"); cuda_check(cudaMemcpy(d,a.data(),n*n*sizeof(double),cudaMemcpyHostToDevice),"H2D");
    for(size_t k=0;k<n;++k) {
        int owner=(int)((k*nranks)/n);
        if(rank==owner) { cholesky_diag<<<1,256,256*sizeof(double)>>>(d,n,k); cuda_check(cudaGetLastError(),"diagonal kernel"); cuda_check(cudaMemcpy(a.data()+k*n+k,d+k*n+k,sizeof(double),cudaMemcpyDeviceToHost),"pivot"); }
        MPI_Bcast(a.data()+k*n+k,1,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        cuda_check(cudaMemcpy(d+k*n+k,a.data()+k*n+k,sizeof(double),cudaMemcpyHostToDevice),"pivot H2D");
        if(last>std::max(first,k+1)) { size_t begin=std::max(first,k+1), rows=last-begin; cholesky_column<<<(rows+255)/256,256>>>(d,n,k,begin,rows); cuda_check(cudaGetLastError(),"column kernel"); }
        cuda_check(cudaDeviceSynchronize(),"column sync");
        std::vector<double> col(n); cuda_check(cudaMemcpy2D(col.data()+k+1,sizeof(double),d+(k+1)*n+k,n*sizeof(double),sizeof(double),n-k-1,cudaMemcpyDeviceToHost),"column D2H");
        std::vector<int> cc(nranks),dd(nranks); for(int r=0;r<nranks;++r){cc[r]=(int)((n*(r+1))/nranks-(n*r)/nranks);dd[r]=(int)((n*r)/nranks);} MPI_Allgatherv(col.data()+first,cc[rank],MPI_DOUBLE,col.data(),cc.data(),dd.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        cuda_check(cudaMemcpy2D(d+k+1,n*sizeof(double),col.data()+k+1,sizeof(double),sizeof(double),n-k-1,cudaMemcpyHostToDevice),"column H2D");
        if(last>std::max(first,k+1)) { size_t begin=std::max(first,k+1), rows=last-begin, cols=n-k-1; if(cols) { size_t work=rows*cols; cholesky_update<<<(work+255)/256,256>>>(d,n,k,begin,rows); cuda_check(cudaGetLastError(),"update kernel"); } }
        cuda_check(cudaDeviceSynchronize(),"update sync");
        if(rank==owner) cuda_check(cudaMemcpy(a.data()+k*n+ k+1,d+k*n+k+1,(n-k-1)*sizeof(double),cudaMemcpyDeviceToHost),"pivot row D2H");
        MPI_Bcast(a.data()+k*n+k+1,(int)(n-k-1),MPI_DOUBLE,owner,MPI_COMM_WORLD);
        cuda_check(cudaMemcpy(d+k*n+k+1,a.data()+k*n+k+1,(n-k-1)*sizeof(double),cudaMemcpyHostToDevice),"row H2D");
    }
    cuda_check(cudaMemcpy(a.data(),d,n*n*sizeof(double),cudaMemcpyDeviceToHost),"D2H"); cudaFree(d);
    std::vector<int> counts(nranks), displs(nranks); for(int r=0;r<nranks;++r){size_t b=(n*r)/nranks,e=(n*(r+1))/nranks;counts[r]=(int)((e-b)*n);displs[r]=(int)(b*n);}
    MPI_Allgatherv(a.data()+first*n,counts[rank],MPI_DOUBLE,a.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
    return true;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& A,size_t n) { double me=0,re=0;
    #pragma omp parallel for reduction(max:me,re) schedule(static)
    for(long long i=0;i<(long long)n*n;++i){size_t r=i/n,c=i%n;double s=0;for(size_t k=0;k<n;++k)s+=L[r*n+k]*L[c*n+k];double e=fabs(s-A[i]);me=std::max(me,e);re=std::max(re,e/(fabs(A[i])+1e-10));}
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",me,re); return re<=1e-6;
}
void printUsage(const char*p){printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",p);}

int main(int argc,char**argv){MPI_Init(&argc,&argv);int rank,nr;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);size_t n=512;bool val=false,results=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=strtoull(argv[++i],nullptr,10);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))results=true;else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Finalize();return 1;}}
    if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,val?"enabled":"disabled");}
    std::vector<double>A(n*n),orig; if(!rank)generatePositiveDefiniteMatrix(A,n); MPI_Bcast(A.data(),(int)(n*n),MPI_DOUBLE,0,MPI_COMM_WORLD); if(val)orig=A; MPI_Barrier(MPI_COMM_WORLD); if(!rank)printf("Computing Cholesky decomposition...\n"); auto start=std::chrono::high_resolution_clock::now(); bool ok=choleskyDecomposition(A,n,rank,nr); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(!rank){double sec=std::chrono::duration<double>(end-start).count();printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",(long)(sec*1000),n*n*n/3.0/sec/1e9);if(results)print_results(A,"CholeskyL");if(val){printf("Validating result...\n");ok=validateCholesky(A,orig,n);printf("Validation: %s\n",ok?"PASSED":"FAILED");}} MPI_Bcast(&ok,1,MPI_C_BOOL,0,MPI_COMM_WORLD);MPI_Finalize();return ok?0:1;}
