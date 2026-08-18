#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

static inline size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny){return (z*ny+y)*nx+x;}
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);}} while(0)

__device__ inline double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz){
  size_t xp=x+1<nx?x+1:x, xm=x?x-1:0, yp=y+1<ny?y+1:y, ym=y?y-1:0, zp=z+1<nz?z+1:z, zm=z?z-1:0, i=(z*ny+y)*nx+x;
  return (a[(z*ny+y)*nx+xp]+a[(z*ny+y)*nx+xm]-2*a[i])+(a[(z*ny+yp)*nx+x]+a[(z*ny+ym)*nx+x]-2*a[i])+(a[(zp*ny+y)*nx+x]+a[(zm*ny+y)*nx+x]-2*a[i]);
}
__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t nz){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*nz; if(i>=n)return; double v=c[i];
  mu[i]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-.5*lap(c,i%nx,(i/nx)%ny,i/(nx*ny),nx,ny,nz);
}
__global__ void update(double* out,const double* old,const double* mu,size_t nx,size_t ny,size_t nz){
  size_t i=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz; if(i<n) out[i]=old[i]+.01*lap(mu,i%nx,(i/nx)%ny,i/(nx*ny),nx,ny,nz);
}

static void usage(const char* p){printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n",p);}
static bool valid(const std::vector<double>& a){double lo=a[0],hi=a[0]; for(double v:a){if(!std::isfinite(v))return false;lo=std::min(lo,v);hi=std::max(hi,v);} printf("Concentration range: [%.6f, %.6f]\n",lo,hi); return hi<=10&&lo>=-10;}

int main(int argc,char** argv){
  MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
  size_t nx=64,ny=0,nz=0; int iters=20; bool validate=false,results=false;
  for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iters=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))results=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)usage(argv[0]);MPI_Finalize();return 1;}}
  if(!ny)ny=nx;if(!nz)nz=nx; size_t base=nz/size, rem=nz%size, z0=rank*base+std::min<size_t>(rank,rem), local=base+(rank<rem); size_t pitch=nx*ny, n=(local+2)*pitch;
  std::vector<double> h(n); for(size_t z=1;z<=local;z++){
    size_t gz=z0+z-1;
#pragma omp parallel for collapse(2) schedule(static)
    for(size_t y=0;y<ny;y++)for(size_t x=0;x<nx;x++){size_t id=at(x,y,z,nx,ny), g=gz*pitch+y*nx+x; h[id]=-1+2*((((g+1)*1299709)% (nx*ny*nz))/double(nx*ny*nz));}
  }
  double *dold,*dnew,*dmu; CUDA_OK(cudaMalloc(&dold,n*sizeof(double)));CUDA_OK(cudaMalloc(&dnew,n*sizeof(double)));CUDA_OK(cudaMalloc(&dmu,n*sizeof(double)));CUDA_OK(cudaMemcpy(dold,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
  MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); int threads=256;
  for(int t=0;t<iters;t++){
    CUDA_OK(cudaMemcpy(h.data(),dold,n*sizeof(double),cudaMemcpyDeviceToHost));
    MPI_Sendrecv(h.data()+pitch,pitch,MPI_DOUBLE,(rank?rank-1:MPI_PROC_NULL),1,h.data()+(local+1)*pitch,pitch,MPI_DOUBLE,(rank+1<size?rank+1:MPI_PROC_NULL),1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Sendrecv(h.data()+local*pitch,pitch,MPI_DOUBLE,(rank+1<size?rank+1:MPI_PROC_NULL),2,h.data(),pitch,MPI_DOUBLE,(rank?rank-1:MPI_PROC_NULL),2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    if(rank==0) std::copy_n(h.data()+pitch,pitch,h.data());
    if(rank==size-1) std::copy_n(h.data()+local*pitch,pitch,h.data()+(local+1)*pitch);
    CUDA_OK(cudaMemcpy(dold,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    chemical<<<(n+threads-1)/threads,threads>>>(dold,dmu,nx,ny,local+2); CUDA_OK(cudaGetLastError());
    update<<<(n+threads-1)/threads,threads>>>(dnew,dold,dmu,nx,ny,local+2); CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize()); std::swap(dold,dnew);
  }
  CUDA_OK(cudaMemcpy(h.data(),dold,n*sizeof(double),cudaMemcpyDeviceToHost)); auto end=std::chrono::high_resolution_clock::now(); double sec=std::chrono::duration<double>(end-start).count();
  std::vector<double> out; std::vector<int> counts,displs; if(!rank){out.resize(nx*ny*nz);counts.resize(size);displs.resize(size);for(int r=0;r<size;r++){size_t l=base+(r<rem),s=r*base+std::min<size_t>(r,rem);counts[r]=int(l*pitch);displs[r]=int(s*pitch);}}
  MPI_Gatherv(h.data()+pitch,int(local*pitch),MPI_DOUBLE,rank?nullptr:out.data(),rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
  if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\nComputation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iters,size,omp_get_max_threads(),sec*1000,(nx*ny*nz*double(iters)/sec)/1e6);if(results)print_results(out,"Concentration");if(validate)printf("Validation: %s\n",valid(out)?"PASSED":"FAILED");}
  cudaFree(dold);cudaFree(dnew);cudaFree(dmu); MPI_Finalize(); return (!rank&&validate&&!valid(out))?1:0;
}
