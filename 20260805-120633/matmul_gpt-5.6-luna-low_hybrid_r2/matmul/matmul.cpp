#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(size_t N, size_t i, size_t j) noexcept { return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N); }
void initMatrix(std::vector<double>& m, size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long p = 0; p < static_cast<long long>(N*N); ++p) m[p] = getPseudoRndValue(N, p/N, p%N);
}
__global__ void matmulKernel(const double* A, const double* B, double* C, size_t rows, size_t N) {
    constexpr int T=32; __shared__ double as[T][T], bs[T][T];
    size_t row=blockIdx.y*T+threadIdx.y, col=blockIdx.x*T+threadIdx.x; double sum=0;
    for (size_t base=0;base<N;base+=T) {
        size_t ak=base+threadIdx.x, bk=base+threadIdx.y;
        as[threadIdx.y][threadIdx.x]=(row<rows&&ak<N)?A[row*N+ak]:0;
        bs[threadIdx.y][threadIdx.x]=(bk<N&&col<N)?B[bk*N+col]:0; __syncthreads();
        #pragma unroll
        for(int k=0;k<T;++k) sum+=as[threadIdx.y][k]*bs[k][threadIdx.x]; __syncthreads();
    }
    if(row<rows&&col<N) C[row*N+col]=sum;
}
static void cudaCheck(cudaError_t e,const char* s) { if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",s,cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,1);} }
void printUsage(const char* p){printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p);}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,nranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    size_t N=512; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)N=strtoull(argv[++i],nullptr,10);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printUsage(argv[0]);MPI_Finalize();return 1;}}
    if(!N){MPI_Abort(MPI_COMM_WORLD,1);} size_t first=N*rank/nranks,last=N*(rank+1)/nranks,rows=last-first;
    std::vector<double>B(N*N),A(rows*N),Cpart(rows*N);
    if(!rank)printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",N,N,validate?"enabled":"disabled");
    #pragma omp parallel for schedule(static)
    for(long long p=0;p<(long long)(N*N);++p)B[p]=getPseudoRndValue(N,p/N,p%N);
    #pragma omp parallel for schedule(static)
    for(long long p=0;p<(long long)(rows*N);++p)A[p]=getPseudoRndValue(N,first+p/N,p%N);
    MPI_Bcast(B.data(),(int)(N*N),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices)==cudaSuccess&&devices?cudaSuccess:cudaErrorNoDevice,"cudaGetDeviceCount"); cudaCheck(cudaSetDevice(rank%devices),"cudaSetDevice");
    double *dA,*dB,*dC; cudaCheck(cudaMalloc(&dA,rows*N*sizeof(double)),"cudaMalloc A");cudaCheck(cudaMalloc(&dB,N*N*sizeof(double)),"cudaMalloc B");cudaCheck(cudaMalloc(&dC,rows*N*sizeof(double)),"cudaMalloc C");
    cudaCheck(cudaMemcpy(dA,A.data(),rows*N*sizeof(double),cudaMemcpyHostToDevice),"copy A");cudaCheck(cudaMemcpy(dB,B.data(),N*N*sizeof(double),cudaMemcpyHostToDevice),"copy B");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); dim3 block(32,32),grid((N+31)/32,(rows+31)/32); matmulKernel<<<grid,block>>>(dA,dB,dC,rows,N); cudaCheck(cudaGetLastError(),"kernel launch");cudaCheck(cudaDeviceSynchronize(),"kernel");cudaCheck(cudaMemcpy(Cpart.data(),dC,rows*N*sizeof(double),cudaMemcpyDeviceToHost),"copy C");auto end=std::chrono::high_resolution_clock::now();
    cudaFree(dA);cudaFree(dB);cudaFree(dC); long long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(),maxms=0;MPI_Reduce(&ms,&maxms,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<double>C; if(!rank)C.resize(N*N); std::vector<int>counts(nranks),displs(nranks);for(int r=0;r<nranks;++r){counts[r]=(int)(N*(N*(r+1)/nranks-N*r/nranks));displs[r]=(int)(N*N*r/nranks);}
    MPI_Gatherv(Cpart.data(),(int)Cpart.size(),MPI_DOUBLE,rank?nullptr:C.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int ok=1;if(!rank){printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n",maxms,2.0*N*N*N/(maxms/1000.0)/1e9);if(printResults)print_results(C,"MatrixC");if(validate){for(size_t i=0;i<N&&ok;++i)for(size_t j=0;j<N&&ok;++j){double s=0;for(size_t k=0;k<N;++k)s+=A[0]*0+B[i*N+k]*B[k*N+j];ok=std::abs((C[i*N+j]-s)/(s+1e-10))<=1e-6;}printf("Validation: %s\n",ok?"PASSED":"FAILED");}}MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);MPI_Finalize();return ok?0:1;
}
