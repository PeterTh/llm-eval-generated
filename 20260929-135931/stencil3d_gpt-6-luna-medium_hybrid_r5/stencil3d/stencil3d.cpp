#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;
__host__ __device__ inline constexpr size_t idx3(size_t x,size_t y,size_t z,size_t nx,size_t ny) { return z*nx*ny+y*nx+x; }

__global__ void stencilKernel(const Real* in, Real* out, size_t nx, size_t ny, size_t lz,
                              size_t firstZ, size_t owned, size_t globalZ) {
    size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = nx * ny * owned;
    if (q >= count) return;
    size_t x=q%nx, y=(q/nx)%ny, oz=q/(nx*ny), z=oz+1;
    size_t i=idx3(x,y,z,nx,ny);
    if (x==0 || x+1==nx || y==0 || y+1==ny || firstZ+oz==0 || firstZ+oz+1==globalZ) out[i]=in[i];
    else out[i]=(in[i]+in[i-1]+in[i+1]+in[i-nx]+in[i+nx]+in[i-nx*ny]+in[i+nx*ny])/7.0;
}

static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2); } }
static void printUsage(const char* p) { printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=128,ny=0,nz=0; int iterations=10; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) printUsage(argv[0]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if(!ny) ny=nx; if(!nz) nz=nx;
    if(nx<3||ny<3||nz<3||iterations<0||size>(int)(nz-2)) { if(rank==0) fprintf(stderr,"Grid dimensions must be >=3 and MPI ranks <= interior Z planes.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    size_t interior=nz-2, base=interior/size, rem=interior%size;
    size_t owned=base+(size_t(rank)<rem), first=1+size_t(rank)*base+std::min(size_t(rank),rem);
    size_t plane=nx*ny, localN=(owned+2)*plane;
    int deviceCount=0; cudaCheck(cudaGetDeviceCount(&deviceCount));
    if(!deviceCount) { if(rank==0) fprintf(stderr,"No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank%deviceCount));
    std::vector<Real> host(localN), gathered;
    #pragma omp parallel for
    for(long long q=0;q<(long long)localN;++q) {
        size_t lz=size_t(q)/plane, remq=size_t(q)%plane;
        size_t gz= lz==0 ? (first?first-1:0) : lz==owned+1 ? first+owned : first+lz-1;
        size_t global=idx3(remq%nx,remq/nx,gz,nx,ny); host[q]=Real(global%19);
    }
    Real *a=nullptr,*b=nullptr; cudaCheck(cudaMalloc(&a,localN*sizeof(Real))); cudaCheck(cudaMalloc(&b,localN*sizeof(Real)));
    cudaCheck(cudaMemcpy(a,host.data(),localN*sizeof(Real),cudaMemcpyHostToDevice));
    size_t sendCount=plane;
    std::vector<int> counts(size),displs(size);
    for(int r=0;r<size;++r) { size_t n=base+(size_t(r)<rem); counts[r]=int(n*plane); size_t f=1+size_t(r)*base+std::min(size_t(r),rem); displs[r]=int(f*plane); }
    if(rank==0) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n",nx,ny,nz,iterations); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    Real *in=a,*out=b;
    for(int it=0;it<iterations;++it) {
        cudaCheck(cudaMemcpy(host.data(),in,localN*sizeof(Real),cudaMemcpyDeviceToHost));
        if(rank>0) MPI_Sendrecv(host.data()+sendCount, int(sendCount), MPI_DOUBLE,rank-1,11,host.data(),int(sendCount),MPI_DOUBLE,rank-1,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank+1<size) MPI_Sendrecv(host.data()+owned*plane, int(sendCount), MPI_DOUBLE,rank+1,12,host.data()+(owned+1)*plane,int(sendCount),MPI_DOUBLE,rank+1,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(in,host.data(),localN*sizeof(Real),cudaMemcpyHostToDevice));
        size_t n=nx*ny*owned; stencilKernel<<<(n+255)/256,256>>>(in,out,nx,ny,owned+2,first,owned,nz); cudaCheck(cudaGetLastError());
        std::swap(in,out);
    }
    cudaCheck(cudaDeviceSynchronize()); auto end=std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpy(host.data(),in,localN*sizeof(Real),cudaMemcpyDeviceToHost));
    std::vector<Real> local(owned*plane); std::copy(host.begin()+plane,host.begin()+(owned+1)*plane,local.begin());
    if(rank==0) {
        gathered.resize(nx*ny*nz);
        #pragma omp parallel for
        for(long long q=0;q<(long long)gathered.size();++q) gathered[q]=Real(size_t(q)%19);
    }
    MPI_Gatherv(local.data(),int(local.size()),MPI_DOUBLE,rank==0?gathered.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        double updates=double((nx-2)*(ny-2)*(nz-2))*iterations;
        printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",long(ms),updates/(ms/1000.0)/1e6);
        if(printResults) print_results(gathered,"Grid");
        if(validate) { bool ok=true; Real mn=gathered[0],mx=mn; for(Real v:gathered) { if(!std::isfinite(v)) ok=false; mn=std::min(mn,v); mx=std::max(mx,v); }
            printf("Value range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,ok&&mx<=1e6&&mn>=-1e6?"PASSED":"FAILED"); if(!ok) MPI_Abort(MPI_COMM_WORLD,1); }
    }
    cudaFree(a); cudaFree(b); MPI_Finalize(); return 0;
}
