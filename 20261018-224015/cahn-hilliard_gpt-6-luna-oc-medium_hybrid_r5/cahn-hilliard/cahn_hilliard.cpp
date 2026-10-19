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

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} } while(0)

__device__ inline size_t index3(size_t x,size_t y,size_t z,size_t nx,size_t ny){return z*nx*ny+y*nx+x;}
__device__ inline double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz,double dx,double dy,double dz){
 const size_t xp=x+1<nx?x+1:x, xn=x?x-1:0, yp=y+1<ny?y+1:y, yn=y?y-1:0, zp=z+1, zn=z-1;
 const size_t i=index3(x,y,z,nx,ny);
 return (a[index3(xp,y,z,nx,ny)]+a[index3(xn,y,z,nx,ny)]-2*a[i])/(dx*dx)+(a[index3(x,yp,z,nx,ny)]+a[index3(x,yn,z,nx,ny)]-2*a[i])/(dy*dy)+(a[index3(x,y,zp,nx,ny)]+a[index3(x,y,zn,nx,ny)]-2*a[i])/(dz*dz);
}
__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t nz,double gamma,double dx,double dy,double dz){
 size_t q=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*nz; if(q>=n)return;
 size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1,i=index3(x,y,z,nx,ny); double v=c[i];
 mu[i]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-gamma*lap(c,x,y,z,nx,ny,nz,dx,dy,dz);
}
__global__ void advance(const double* c,const double* mu,double* out,size_t nx,size_t ny,size_t nz,double dt,double D,double dx,double dy,double dz){
 size_t q=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(q>=n)return;
 size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1,i=index3(x,y,z,nx,ny);out[i]=c[i]+dt*D*lap(mu,x,y,z,nx,ny,nz,dx,dy,dz);
}

static void exchange(double* a,size_t plane,size_t localZ,int left,int right,int tag){
 MPI_Sendrecv(a+plane,static_cast<int>(plane),MPI_DOUBLE,left,tag,a+(localZ+1)*plane,static_cast<int>(plane),MPI_DOUBLE,right,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
 MPI_Sendrecv(a+localZ*plane,static_cast<int>(plane),MPI_DOUBLE,right,tag+1,a,static_cast<int>(plane),MPI_DOUBLE,left,tag+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}
static void usage(const char* p){printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n",p);}

int main(int argc,char** argv){
 MPI_Init(&argc,&argv); int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
 size_t nx=64,ny=0,nz=0;int iterations=20;bool validate=false,printResults=false;
 for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)usage(argv[0]);MPI_Abort(MPI_COMM_WORLD,1);}}
 if(!ny)ny=nx;if(!nz)nz=nx;if(nz<static_cast<size_t>(size)){if(!rank)fprintf(stderr,"z dimension must be at least the MPI rank count\n");MPI_Abort(MPI_COMM_WORLD,1);}
 int deviceCount=0;CUDA_CHECK(cudaGetDeviceCount(&deviceCount));if(deviceCount)CUDA_CHECK(cudaSetDevice(rank%deviceCount));else{fprintf(stderr,"Rank %d: no CUDA device available\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}
 const size_t base=nz/size,rem=nz%size,localZ=base+(static_cast<size_t>(rank)<rem),zStart=rank*base+std::min(static_cast<size_t>(rank),rem),plane=nx*ny,vol=plane*localZ,total=vol+2*plane;
 const int left=rank?rank-1:MPI_PROC_NULL,right=rank+1<size?rank+1:MPI_PROC_NULL;double *c,*mu,*next;
 CUDA_CHECK(cudaMallocManaged(&c,total*sizeof(double)));CUDA_CHECK(cudaMallocManaged(&mu,total*sizeof(double)));CUDA_CHECK(cudaMallocManaged(&next,total*sizeof(double)));
 #pragma omp parallel for schedule(static)
 for(long long q=0;q<static_cast<long long>(vol);++q){size_t g=zStart+static_cast<size_t>(q)/plane,remq=static_cast<size_t>(q)%plane,linear=g*plane+remq;double pseudo=(((linear+1)*1299709)% (nx*ny*nz))/static_cast<double>(nx*ny*nz);c[plane+q]=-1+2*pseudo;}
 // Replicate physical boundary planes to implement the original clamped boundary condition.
 if(rank==0)std::copy(c+plane,c+2*plane,c);if(rank==size-1)std::copy(c+localZ*plane,c+(localZ+1)*plane,c+(localZ+1)*plane);
 if(rank==0){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");}
 auto start=std::chrono::high_resolution_clock::now();const int threads=256;const size_t blocks=(vol+threads-1)/threads;
 for(int t=0;t<iterations;++t){
   if(rank==0)std::copy(c+plane,c+2*plane,c);if(rank==size-1)std::copy(c+localZ*plane,c+(localZ+1)*plane,c+(localZ+1)*plane);
   exchange(c,plane,localZ,left,right,10);CUDA_CHECK(cudaDeviceSynchronize());
   chemical<<<blocks,threads>>>(c,mu,nx,ny,localZ,0.5,1,1,1);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
   // Exchange computed chemical-potential halos before applying its Laplacian.
   if(rank==0)std::copy(mu+plane,mu+2*plane,mu);if(rank==size-1)std::copy(mu+localZ*plane,mu+(localZ+1)*plane,mu+(localZ+1)*plane);
   exchange(mu,plane,localZ,left,right,20);CUDA_CHECK(cudaDeviceSynchronize());
    advance<<<blocks,threads>>>(c,mu,next,nx,ny,localZ,0.01,1,1,1,1);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());std::swap(c,next);
 }
 MPI_Barrier(MPI_COMM_WORLD);auto end=std::chrono::high_resolution_clock::now();
 std::vector<int> counts(size),displs(size);for(int r=0;r<size;++r){size_t zc=nz/size+(static_cast<size_t>(r)<nz%size),zs=r*(nz/size)+std::min(static_cast<size_t>(r),nz%size);counts[r]=static_cast<int>(zc*plane);displs[r]=static_cast<int>(zs*plane);}
 std::vector<double> result;if(rank==0)result.resize(nx*ny*nz);MPI_Gatherv(c+plane,static_cast<int>(vol),MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
 if(rank==0){double seconds=std::chrono::duration<double>(end-start).count();printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000));printf("Performance: %.3f MCellUpdates/s\n",seconds>0?static_cast<double>(nx*ny*nz)*iterations/seconds/1e6:0.0);
 if(printResults)print_results(result,"Concentration");if(validate){bool ok=true;double lo=result[0],hi=result[0];for(double v:result){if(!std::isfinite(v))ok=false;lo=std::min(lo,v);hi=std::max(hi,v);}printf("Concentration range: [%.6f, %.6f]\n",lo,hi);if(hi>10||lo< -10)ok=false;printf("Validation: %s\n",ok?"PASSED":"FAILED");if(!ok){CUDA_CHECK(cudaFree(c));CUDA_CHECK(cudaFree(mu));CUDA_CHECK(cudaFree(next));MPI_Finalize();return 1;}}}
 CUDA_CHECK(cudaFree(c));CUDA_CHECK(cudaFree(mu));CUDA_CHECK(cudaFree(next));MPI_Finalize();return 0;
}
