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

static inline void ck(cudaError_t e, const char *s) { if (e != cudaSuccess) { fprintf(stderr,"CUDA %s: %s\n",s,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } }
__device__ __forceinline__ size_t at(int x,int y,int z,int nx,int ny) { return (size_t)z*nx*ny+(size_t)y*nx+x; }
__device__ __forceinline__ int clampi(int v,int n) { return v<0?0:(v>=n?n-1:v); }

__global__ void chemical(const double *c,double *mu,int nx,int ny,int nz,double gamma) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y, z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz) return; int X=clampi(x,nx),Y=clampi(y,ny),Z=z+1;
    size_t q=at(X,Y,Z,nx,ny); double v=c[q];
    double lap=(c[at(clampi(x+1,nx),Y,Z,nx,ny)]+c[at(clampi(x-1,nx),Y,Z,nx,ny)]-2*v)
              +(c[at(X,clampi(y+1,ny),Z,nx,ny)]+c[at(X,clampi(y-1,ny),Z,nx,ny)]-2*v)
              +(c[at(X,Y,Z+1,nx,ny)]+c[at(X,Y,Z-1,nx,ny)]-2*v);
    mu[q]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-gamma*lap;
}
__global__ void update(const double *cold,const double *mu,double *out,int nx,int ny,int nz,double dt) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return; int X=clampi(x,nx),Y=clampi(y,ny),Z=z+1; size_t q=at(X,Y,Z,nx,ny);
    double v=mu[q]; double lap=(mu[at(clampi(x+1,nx),Y,Z,nx,ny)]+mu[at(clampi(x-1,nx),Y,Z,nx,ny)]-2*v)
      +(mu[at(X,clampi(y+1,ny),Z,nx,ny)]+mu[at(X,clampi(y-1,ny),Z,nx,ny)]-2*v)
      +(mu[at(X,Y,Z+1,nx,ny)]+mu[at(X,Y,Z-1,nx,ny)]-2*v); out[q]=cold[q]+dt*lap;
}

static void exchange(double *d, int local, size_t plane, int rank, int nr, int nx, int ny) {
    static std::vector<double> send,recv; send.resize(plane); recv.resize(plane);
    auto copy_plane=[&](int z,double *p){ ck(cudaMemcpy(p,d+(size_t)z*plane,plane*sizeof(double),cudaMemcpyDeviceToHost),"D2H halo"); };
    auto put_plane=[&](int z,const double *p){ ck(cudaMemcpy(d+(size_t)z*plane,p,plane*sizeof(double),cudaMemcpyHostToDevice),"H2D halo"); };
    int lo=rank?rank-1:MPI_PROC_NULL, hi=rank+1<nr?rank+1:MPI_PROC_NULL;
    copy_plane(1,send.data()); MPI_Sendrecv(send.data(),(int)plane,MPI_DOUBLE,lo,7,recv.data(),(int)plane,MPI_DOUBLE,hi,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    put_plane(local+1, hi==MPI_PROC_NULL?send.data():recv.data());
    copy_plane(local,send.data()); MPI_Sendrecv(send.data(),(int)plane,MPI_DOUBLE,hi,8,recv.data(),(int)plane,MPI_DOUBLE,lo,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    put_plane(0, lo==MPI_PROC_NULL?send.data():recv.data());
}

static void usage(const char *p){printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p);}
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv); int rank,nr; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nr);
    size_t nx=64,ny=0,nz=0; int iters=20; bool valid=false,results=false;
    for(int i=1;i<argc;i++){ if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]); else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc)iters=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))valid=true; else if(!strcmp(argv[i],"-r"))results=true; else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;} else {if(!rank)usage(argv[0]);MPI_Finalize();return 1;} }
    if(!ny)ny=nx;if(!nz)nz=nx; if(nz<(size_t)nr){if(!rank)fprintf(stderr,"z dimension must be >= MPI ranks\n");MPI_Finalize();return 1;}
    int devs=0; ck(cudaGetDeviceCount(&devs),"device count"); ck(cudaSetDevice(rank%devs),"set device");
    size_t base=nz/nr, rem=nz%nr, local=base+(rank<(int)rem), z0=rank*base+(rank<(int)rem?rank:rem), plane=nx*ny;
    std::vector<double> h(local*plane);
#pragma omp parallel for schedule(static)
    for(size_t q=0;q<h.size();q++){size_t g=z0*plane+q; h[q]=-1.0+2.0*((((g+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));}
    double *c,*mu,*next; ck(cudaMalloc(&c,(local+2)*plane*sizeof(double)),"alloc c");ck(cudaMalloc(&mu,(local+2)*plane*sizeof(double)),"alloc mu");ck(cudaMalloc(&next,(local+2)*plane*sizeof(double)),"alloc next");
    ck(cudaMemcpy(c+plane,h.data(),h.size()*sizeof(double),cudaMemcpyHostToDevice),"initial copy"); exchange(c,local,plane,rank,nr,nx,ny);
    dim3 block(8,8,4), grid((nx+7)/8,(ny+7)/8,(local+3)/4); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iters;t++){ chemical<<<grid,block>>>(c,mu,nx,ny,local,.5); ck(cudaGetLastError(),"chemical launch"); exchange(mu,local,plane,rank,nr,nx,ny); update<<<grid,block>>>(c,mu,next,nx,ny,local,.01); ck(cudaGetLastError(),"update launch"); std::swap(c,next); exchange(c,local,plane,rank,nr,nx,ny); }
    ck(cudaDeviceSynchronize(),"synchronize"); auto end=std::chrono::high_resolution_clock::now(); double sec=std::chrono::duration<double>(end-start).count();
    ck(cudaMemcpy(h.data(),c+plane,h.size()*sizeof(double),cudaMemcpyDeviceToHost),"final copy");
    std::vector<int> counts(nr),displs(nr); for(int r=0;r<nr;r++){size_t n=base+(r<(int)rem);counts[r]=(int)(n*plane);displs[r]=(int)((r*base+(r<(int)rem?r:rem))*plane);} std::vector<double> all(rank?0:nx*ny*nz);
    MPI_Gatherv(h.data(),(int)h.size(),MPI_DOUBLE,rank?nullptr:all.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nComputation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iters,sec*1000,(double)nx*ny*nz*iters/sec/1e6); if(results)print_results(all,"Concentration"); if(valid){double mn=all[0],mx=all[0]; bool ok=true;
#pragma omp parallel for reduction(min:mn) reduction(max:mx)
        for(size_t i=0;i<all.size();i++){if(!std::isfinite(all[i]))ok=false;mn=std::min(mn,all[i]);mx=std::max(mx,all[i]);} printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,(ok&&mn>=-10&&mx<=10)?"PASSED":"FAILED"); if(!(ok&&mn>=-10&&mx<=10)) {cudaFree(c);cudaFree(mu);cudaFree(next);MPI_Finalize();return 1;}} }
    cudaFree(c);cudaFree(mu);cudaFree(next);MPI_Finalize();return 0;
}
