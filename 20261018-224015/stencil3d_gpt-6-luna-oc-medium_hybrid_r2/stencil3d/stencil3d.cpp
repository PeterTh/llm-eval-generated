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

__global__ void stencil(const Real* in, Real* out, size_t nx,size_t ny,size_t localNz,size_t globalZ,size_t zStart) {
    size_t x=blockIdx.x*blockDim.x+threadIdx.x+1, y=blockIdx.y*blockDim.y+threadIdx.y+1;
    size_t z=blockIdx.z*blockDim.z+threadIdx.z+1;
    if(x>=nx-1||y>=ny-1||z>=localNz-1) return;
    size_t g=zStart+z, i=idx3(x,y,z,nx,ny);
    if(g==0||g>=globalZ-1) return;
    out[i]=(in[i]+in[i-1]+in[i+1]+in[i-nx]+in[i+nx]+in[i-nx*ny]+in[i+nx*ny])/7.0;
}

static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} }

int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,nranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    size_t nx=128,ny=0,nz=0; int iterations=10; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")){if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0;}
        else {if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    if(!ny)ny=nx; if(!nz)nz=nx;
    if(nx<3||ny<3||nz<3||iterations<0||nz<(size_t)nranks*1){if(rank==0)fprintf(stderr,"Grid dimensions must be >=3 and iterations nonnegative\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices)); if(!devices) MPI_Abort(MPI_COMM_WORLD,2); cudaCheck(cudaSetDevice(rank%devices));
    size_t base=nz/nranks,rem=nz%nranks,localNz=base+(rank<(int)rem),zStart=rank*base+std::min((size_t)rank,rem);
    size_t plane=nx*ny,count=plane*(localNz+2);
    std::vector<Real> a(count),b(count);
    #pragma omp parallel for schedule(static)
    for(long long q=0;q<(long long)(localNz*plane);++q){size_t z=q/plane+1, p=q%plane; a[z*plane+p]=((zStart+z-1)*plane+p)%19;}
    // Install fixed global boundary planes; halo planes are populated by MPI each step.
    if(zStart==0) std::copy(a.begin()+plane,a.begin()+2*plane,a.begin());
    if(zStart+localNz==nz) std::copy(a.begin()+localNz*plane,a.begin()+(localNz+1)*plane,a.begin()+(localNz+1)*plane);
    Real *dA=nullptr,*dB=nullptr; cudaCheck(cudaMalloc(&dA,count*sizeof(Real))); cudaCheck(cudaMalloc(&dB,count*sizeof(Real)));
    cudaCheck(cudaMemcpy(dA,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    if(rank==0) {printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n",nx,ny,nz,iterations);}
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::steady_clock::now();
    dim3 block(32,4,1),grid((nx-2+31)/32,(ny-2+3)/4,localNz);
    for(int it=0;it<iterations;++it){
        // Exchange current slab edges into the one-plane ghost regions.
        MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,10,
                     a.data()+(localNz+1)*plane,plane,MPI_DOUBLE,rank+1<nranks?rank+1:MPI_PROC_NULL,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+localNz*plane,plane,MPI_DOUBLE,rank+1<nranks?rank+1:MPI_PROC_NULL,11,
                     a.data(),plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(dA,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
        stencil<<<grid,block>>>(dA,dB,nx,ny,localNz+2,nz,zStart-1);
        cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(b.data(),dB,count*sizeof(Real),cudaMemcpyDeviceToHost));
        // x/y boundaries and physical z boundaries remain unchanged.
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y){ b[z*plane+y*nx]=a[z*plane+y*nx]; b[z*plane+y*nx+nx-1]=a[z*plane+y*nx+nx-1]; }
        for(size_t z=1;z<=localNz;++z){std::copy(a.begin()+z*plane,a.begin()+z*plane+nx,b.begin()+z*plane); std::copy(a.begin()+z*plane+(ny-1)*nx,a.begin()+z*plane+plane,b.begin()+z*plane+(ny-1)*nx);}
        if(zStart==0) std::copy(a.begin()+plane,a.begin()+2*plane,b.begin()+plane);
        if(zStart+localNz==nz) std::copy(a.begin()+localNz*plane,a.begin()+(localNz+1)*plane,b.begin()+localNz*plane);
        a.swap(b);
    }
    MPI_Barrier(MPI_COMM_WORLD); double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    // Gather owned planes for output and validation.
    std::vector<int> counts(nranks),displs(nranks); for(int r=0;r<nranks;++r){size_t n=nz/nranks+(r<(int)rem); counts[r]=(int)(n*plane); displs[r]=(int)((r*(nz/nranks)+std::min((size_t)r,rem))*plane);}
    std::vector<Real> full(rank==0?nz*plane:0); MPI_Gatherv(a.data()+plane,(int)(localNz*plane),MPI_DOUBLE,full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int result=0;
    if(rank==0){ if(maxElapsed==0)maxElapsed=1e-12; printf("Computation time: %.6f s\nPerformance: %.3f MCellUpdates/s\n",maxElapsed,((double)(nx-2)*(ny-2)*(nz-2)*iterations/maxElapsed/1e6));
        if(printResults) print_results(full,"Grid");
        if(validate){bool ok=true; Real mn=full[0],mx=full[0]; for(Real v:full){if(!std::isfinite(v))ok=false;mn=std::min(mn,v);mx=std::max(mx,v);} printf("Value range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,ok&&mx<=1e6&&mn>=-1e6?"PASSED":"FAILED"); result=ok?0:1;}}
    cudaFree(dA);cudaFree(dB); MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return result;
}
