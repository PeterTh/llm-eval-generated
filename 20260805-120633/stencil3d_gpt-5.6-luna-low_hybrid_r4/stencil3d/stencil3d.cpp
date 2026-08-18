#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

using Real = double;
__host__ __device__ inline constexpr size_t idx3(size_t x,size_t y,size_t z,size_t nx,size_t ny) noexcept { return z*nx*ny+y*nx+x; }

static void checkCuda(cudaError_t e, const char* what) { if(e!=cudaSuccess){ fprintf(stderr,"CUDA error (%s): %s\n",what,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } }

__global__ static void stencil_kernel(const Real* in, Real* out, size_t nx, size_t ny, size_t localNz, size_t globalZ0, size_t globalNz) {
    size_t x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y, z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx || y>=ny || z>=localNz) return;
    size_t p=idx3(x,y,z,nx,ny); size_t gz=(globalZ0==0 && z==0)?0:globalZ0+z-1;
    if(x==0 || x+1==nx || y==0 || y+1==ny || gz==0 || gz+1==globalNz) out[p]=in[p];
    else out[p]=(in[p]+in[p-1]+in[p+1]+in[p-nx]+in[p+nx]+in[p-nx*ny]+in[p+nx*ny])/7.0;
}

static void initialize(std::vector<Real>& a, size_t nx,size_t ny,size_t localNz,size_t globalZ0) {
    #pragma omp parallel for collapse(3) schedule(static)
    for(size_t z=0;z<localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t gz=(globalZ0==0 && z==0)?0:globalZ0+z-1; a[idx3(x,y,z,nx,ny)]=(idx3(x,y,gz,nx,ny)%19)*1.0;
    }
}

static bool validate(const std::vector<Real>& a) {
    if(a.empty()) return true; Real lo=a[0],hi=a[0];
    #pragma omp parallel for reduction(min:lo) reduction(max:hi)
    for(size_t i=0;i<a.size();++i) { if(!std::isfinite(a[i])) lo=-INFINITY; lo=std::min(lo,a[i]); hi=std::max(hi,a[i]); }
    printf("Value range: [%.6f, %.6f]\n",lo,hi); return std::isfinite(lo)&&std::isfinite(hi)&&lo>=-1e6&&hi<=1e6;
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=128,ny=0,nz=0; int iterations=10; bool doValidate=false, printResults=false;
    for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"-x")&&i+1<argc) nx=strtoull(argv[++i],nullptr,10); else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=strtoull(argv[++i],nullptr,10); else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=strtoull(argv[++i],nullptr,10); else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]); else if(!strcmp(argv[i],"-v")) doValidate=true; else if(!strcmp(argv[i],"-r")) printResults=true; else if(!strcmp(argv[i],"-h")){if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0;} else {if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1;} }
    if(!ny)ny=nx;if(!nz)nz=nx; if(nx<3||ny<3||nz<3||iterations<0){if(rank==0)fprintf(stderr,"Grid dimensions must be >= 3 and iterations non-negative\n");MPI_Finalize();return 1;}
    int deviceCount=0; checkCuda(cudaGetDeviceCount(&deviceCount),"device query"); if(!deviceCount){fprintf(stderr,"No CUDA device available on MPI rank %d\n",rank);MPI_Abort(MPI_COMM_WORLD,1);} checkCuda(cudaSetDevice(rank%deviceCount),"device selection");
    size_t owned=nz/size+(rank<(int)(nz%size)); size_t z0=rank*(nz/size)+std::min<size_t>(rank,nz%size); size_t localNz=owned+2; size_t plane=nx*ny, localBytes=localNz*plane*sizeof(Real);
    std::vector<Real> h1(localNz*plane),h2(localNz*plane); initialize(h1,nx,ny,localNz,z0); initialize(h2,nx,ny,localNz,z0);
    Real *d1=nullptr,*d2=nullptr; checkCuda(cudaMalloc(&d1,localBytes),"allocate"); checkCuda(cudaMalloc(&d2,localBytes),"allocate"); checkCuda(cudaMemcpy(d1,h1.data(),localBytes,cudaMemcpyHostToDevice),"initial copy");
    dim3 block(32,4,1), grid((nx+31)/32,(ny+3)/4,1); auto start=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iterations;++it){
        Real *in=(it%2==0)?d1:d2,*out=(it%2==0)?d2:d1; std::vector<Real>& hin=(it%2==0)?h1:h2; std::vector<Real>& hout=(it%2==0)?h2:h1;
        checkCuda(cudaMemcpy(hin.data(),in,localBytes,cudaMemcpyDeviceToHost),"halo copy");
        MPI_Sendrecv(hin.data()+plane,plane,MPI_DOUBLE,rank>0?rank-1:MPI_PROC_NULL,7,hin.data()+(localNz-1)*plane,plane,MPI_DOUBLE,rank+1<size?rank+1:MPI_PROC_NULL,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(hin.data()+(localNz-2)*plane,plane,MPI_DOUBLE,rank+1<size?rank+1:MPI_PROC_NULL,8,hin.data(),plane,MPI_DOUBLE,rank>0?rank-1:MPI_PROC_NULL,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        checkCuda(cudaMemcpy(in,hin.data(),localBytes,cudaMemcpyHostToDevice),"halo upload"); grid.z=localNz; stencil_kernel<<<grid,block>>>(in,out,nx,ny,localNz,z0,nz); checkCuda(cudaGetLastError(),"kernel"); checkCuda(cudaDeviceSynchronize(),"kernel sync"); (void)hout;
    }
    auto end=std::chrono::high_resolution_clock::now(); Real *finalD=(iterations%2==0)?d1:d2; std::vector<Real>& finalH=(iterations%2==0)?h1:h2; checkCuda(cudaMemcpy(finalH.data(),finalD,localBytes,cudaMemcpyDeviceToHost),"final copy");
    std::vector<Real> full; if(rank==0) full.resize(nx*ny*nz); std::vector<int> counts(size),displs(size); for(int r=0;r<size;++r){size_t n=nz/size+(r<(int)(nz%size));counts[r]=(int)(n*plane);displs[r]=(int)((r*(nz/size)+std::min<size_t>(r,nz%size))*plane);} MPI_Gatherv(finalH.data()+plane,(int)(owned*plane),MPI_DOUBLE,rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nComputation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iterations,ms,(double)(nx-2)*(ny-2)*(nz-2)*iterations/(std::max<long long>(1,ms)/1000.0)/1e6); if(printResults) print_results(full,"Grid"); if(doValidate) printf("Validation: %s\n",validate(full)?"PASSED":"FAILED");}
    cudaFree(d1);cudaFree(d2); MPI_Finalize(); return 0;
}
