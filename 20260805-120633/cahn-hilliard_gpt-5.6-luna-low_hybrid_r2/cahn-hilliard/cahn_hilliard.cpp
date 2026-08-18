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

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } } while(0)
__device__ __forceinline__ size_t I(int x,int y,int z,int nx,int ny){return (size_t)z*nx*ny+(size_t)y*nx+x;}

__global__ void chemical(const double* c,double* m,int nx,int ny,int nz,double gamma,double aa,double bb,double ab){
  size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=(size_t)nx*ny*nz; if(q>=n)return;
  int x=q%nx,y=(q/nx)%ny,z=q/(nx*ny), xp=x<nx-1?x+1:x,xn=x?x-1:0,yp=y<ny-1?y+1:y,yn=y?y-1:0;
  double v=c[q], lap=c[I(xp,y,z,nx,ny)]+c[I(xn,y,z,nx,ny)]+c[I(x,yp,z,nx,ny)]+c[I(x,yn,z,nx,ny)]+c[I(x,y,z+1,nx,ny)]+c[I(x,y,z-1,nx,ny)]-6*v;
  m[q]=4.5*((v+1)*aa+(v-1)*bb-2*v*ab)+3*v+v*v*v-gamma*lap;
}
__global__ void update(double* out,const double* old,const double* m,int nx,int ny,int nz,double dt,double D){
  size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)nx*ny*nz;if(q>=n)return;
  int x=q%nx,y=(q/nx)%ny,z=q/(nx*ny),xp=x<nx-1?x+1:x,xn=x?x-1:0,yp=y<ny-1?y+1:y,yn=y?y-1:0;
  out[q]=old[q]+dt*D*(m[I(xp,y,z,nx,ny)]+m[I(xn,y,z,nx,ny)]+m[I(x,yp,z,nx,ny)]+m[I(x,yn,z,nx,ny)]+m[I(x,y,z+1,nx,ny)]+m[I(x,y,z-1,nx,ny)]-6*m[q]);
}

static void exchange(double* d,int nx,int ny,int nz,int rank,int nr){
  size_t plane=(size_t)nx*ny; std::vector<double> lo(plane),hi(plane),inlo(plane),inhi(plane);
  CUDA_OK(cudaMemcpy(lo.data(),d+plane,plane*sizeof(double),cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(hi.data(),d+(size_t)nz*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
  int down=rank?rank-1:MPI_PROC_NULL, up=rank+1<nr?rank+1:MPI_PROC_NULL;
  MPI_Sendrecv(lo.data(),plane,MPI_DOUBLE,down,0,inhi.data(),plane,MPI_DOUBLE,up,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  MPI_Sendrecv(hi.data(),plane,MPI_DOUBLE,up,1,inlo.data(),plane,MPI_DOUBLE,down,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  if(rank==0) inlo=lo; if(rank==nr-1) inhi=hi;
  CUDA_OK(cudaMemcpy(d,inlo.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d+(size_t)(nz+1)*plane,inhi.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
}

int main(int argc,char**argv){
  MPI_Init(&argc,&argv); int rank,nr; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nr);
  size_t nx=64,ny=0; int nzarg=0,it=20; bool val=false,pr=false;
  for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nzarg=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)it=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))pr=true;else if(!strcmp(argv[i],"-h")){if(!rank)printf("Usage: %s [-x n] [-y n] [-z n] [-i steps] [-v] [-r]\n",argv[0]);MPI_Finalize();return 0;}}
  if(!ny)ny=nx; size_t nz=nzarg?nzarg:nx; if(nz< (size_t)nr){if(!rank)fprintf(stderr,"Z dimension must be >= MPI ranks\n");MPI_Abort(MPI_COMM_WORLD,2);}
  int base=nz/nr, rem=nz%nr, local=base+(rank<rem); size_t plane=nx*ny, cells=(size_t)local*plane;
  double *dc,*dm,*dn; CUDA_OK(cudaMalloc(&dc,(local+2)*plane*sizeof(double)));CUDA_OK(cudaMalloc(&dm,(local+2)*plane*sizeof(double)));CUDA_OK(cudaMalloc(&dn,(local+2)*plane*sizeof(double)));
  std::vector<double> h((local+2)*plane); size_t z0=(size_t)rank*base+std::min(rank,rem);
  #pragma omp parallel for
  for(size_t q=0;q<cells;q++){size_t g=(z0+q/plane)*plane+q%plane; h[plane+q]=-1.0+2.0*((((g+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));}
  CUDA_OK(cudaMemcpy(dc,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice)); exchange(dc,nx,ny,local,rank,nr);
  MPI_Barrier(MPI_COMM_WORLD); double t0=MPI_Wtime(); int threads=256; size_t total=(local+2)*plane;
  for(int t=0;t<it;t++){chemical<<<(total+threads-1)/threads,threads>>>(dc,dm,nx,ny,local+2,.5,-2.0/9,-2.0/9,2.0/9);CUDA_OK(cudaGetLastError());exchange(dm,nx,ny,local,rank,nr);update<<<(cells+threads-1)/threads,threads>>>(dn,dc,dm+plane,nx,ny,local,.01,1.0);CUDA_OK(cudaGetLastError());std::swap(dc,dn);}
  CUDA_OK(cudaDeviceSynchronize()); double elapsed=MPI_Wtime()-t0, worst;MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  std::vector<double> out(cells);CUDA_OK(cudaMemcpy(out.data(),dc+plane,cells*sizeof(double),cudaMemcpyDeviceToHost));
  if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nMPI ranks: %d, OpenMP threads: %d, CUDA: enabled\nComputation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,it,nr,omp_get_max_threads(),worst*1000,(double)nx*ny*nz*it/worst/1e6);}
  std::vector<int> counts(nr),disp(nr);for(int r=0;r<nr;r++){counts[r]=(base+(r<rem))*plane;disp[r]=(r*base+std::min(r,rem))*plane;}
  std::vector<double> all; if(!rank)all.resize(nx*ny*nz);MPI_Gatherv(out.data(),(int)out.size(),MPI_DOUBLE,rank?nullptr:all.data(),counts.data(),disp.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
  int ok=1;if(!rank){for(double v:all)if(!std::isfinite(v)||v>10||v<-10)ok=0;if(val)printf("Validation: %s\n",ok?"PASSED":"FAILED");if(pr)print_results(all,"Concentration");}MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
  cudaFree(dc);cudaFree(dm);cudaFree(dn);MPI_Finalize();return ok?0:1;
}
