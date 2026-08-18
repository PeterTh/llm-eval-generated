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

inline __device__ __host__ size_t at(int x,int y,int z,int nx,int ny) { return (size_t)z*nx*ny+(size_t)y*nx+x; }

__global__ void chemical(const double* c,double* mu,int nx,int ny,int nz,double gamma,double a,double b,double ab) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y, z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz) return; int X=x+1,Y=y+1,Z=z+1; size_t q=at(X,Y,Z,nx+2,ny+2); double v=c[q];
    double lap=(c[at(X+1,Y,Z,nx+2,ny+2)]+c[at(X-1,Y,Z,nx+2,ny+2)]-2*v)
              +(c[at(X,Y+1,Z,nx+2,ny+2)]+c[at(X,Y-1,Z,nx+2,ny+2)]-2*v)
              +(c[at(X,Y,Z+1,nx+2,ny+2)]+c[at(X,Y,Z-1,nx+2,ny+2)]-2*v);
    mu[q]=4.5*((v+1)*a+(v-1)*b-2*v*ab)+3*v+v*v*v-gamma*lap;
}
__global__ void update(const double* old,double* out,const double* mu,int nx,int ny,int nz,double dt,double D) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,z=blockIdx.z*blockDim.z+threadIdx.z;
    if(x>=nx||y>=ny||z>=nz)return; int X=x+1,Y=y+1,Z=z+1; size_t q=at(X,Y,Z,nx+2,ny+2); double m=mu[q];
    double lap=(mu[at(X+1,Y,Z,nx+2,ny+2)]+mu[at(X-1,Y,Z,nx+2,ny+2)]-2*m)+(mu[at(X,Y+1,Z,nx+2,ny+2)]+mu[at(X,Y-1,Z,nx+2,ny+2)]-2*m)+(mu[at(X,Y,Z+1,nx+2,ny+2)]+mu[at(X,Y,Z-1,nx+2,ny+2)]-2*m);
    out[q]=old[q]+dt*D*lap;
}
static void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,1);}}
static void exchange(double* d,int nx,int ny,int nz,int rank,int size,std::vector<double>& send,std::vector<double>& recv){
    size_t plane=(size_t)(nx+2)*(ny+2); send.resize(plane); recv.resize(plane); int lo=rank?rank-1:MPI_PROC_NULL,hi=rank+1<size?rank+1:MPI_PROC_NULL;
    check(cudaMemcpy(send.data(),d+at(0,0,1,nx+2,ny+2),plane*sizeof(double),cudaMemcpyDeviceToHost));
    MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,lo,7,recv.data(),plane,MPI_DOUBLE,hi,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    if(hi!=MPI_PROC_NULL) check(cudaMemcpy(d+at(0,0,nz+1,nx+2,ny+2),recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
    else check(cudaMemcpy(d+at(0,0,nz+1,nx+2,ny+2),d+at(0,0,nz,nx+2,ny+2),plane*sizeof(double),cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(send.data(),d+at(0,0,nz,nx+2,ny+2),plane*sizeof(double),cudaMemcpyDeviceToHost));
    MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,hi,8,recv.data(),plane,MPI_DOUBLE,lo,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    if(lo!=MPI_PROC_NULL) check(cudaMemcpy(d+at(0,0,0,nx+2,ny+2),recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
    else check(cudaMemcpy(d+at(0,0,0,nx+2,ny+2),d+at(0,0,1,nx+2,ny+2),plane*sizeof(double),cudaMemcpyDeviceToDevice));
}
int main(int argc,char** argv){
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,results=false;
    for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))results=true;else if(!strcmp(argv[i],"-h")){if(!rank)printf("Usage: %s [-x n] [-y n] [-z n] [-i steps] [-v] [-r]\n",argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Finalize();return 1;}}
    if(!ny)ny=nx;if(!nz)nz=nx; if(nz<(size_t)size){if(!rank)fprintf(stderr,"Z dimension must be >= MPI ranks\n");MPI_Abort(MPI_COMM_WORLD,1);}
    size_t base=nz/size, rem=nz%size, local=base+(rank<(int)rem), z0=rank*base+std::min((size_t)rank,rem), plane=(nx+2)*(ny+2), n=(local+2)*plane;
    int dev=rank; int count=0; cudaGetDeviceCount(&count); if(!count){fprintf(stderr,"No CUDA device\n");MPI_Abort(MPI_COMM_WORLD,1);} check(cudaSetDevice(dev%count));
    std::vector<double> h(n);
    #pragma omp parallel for schedule(static)
    for(size_t z=1;z<=local;z++)for(size_t y=1;y<=ny;y++)for(size_t x=1;x<=nx;x++){size_t g=(z0+z-1)*nx*ny+(y-1)*nx+x-1;h[at(x,y,z,nx+2,ny+2)]=-1.0+2.0*((((g+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));}
    #pragma omp parallel for collapse(2)
    for(size_t z=1;z<=local;z++) for(size_t y=0;y<=ny+1;y++){h[at(0,y,z,nx+2,ny+2)]=h[at(1,std::min(y,(size_t)ny),z,nx+2,ny+2)];h[at(nx+1,y,z,nx+2,ny+2)]=h[at(nx,std::min(y,(size_t)ny),z,nx+2,ny+2)];}
    #pragma omp parallel for collapse(2)
    for(size_t z=1;z<=local;z++) for(size_t x=0;x<=nx+1;x++){h[at(x,0,z,nx+2,ny+2)]=h[at(std::min(x,nx),1,z,nx+2,ny+2)];h[at(x,ny+1,z,nx+2,ny+2)]=h[at(std::min(x,nx),ny,z,nx+2,ny+2)];}
    double *c,*mu,*next;check(cudaMalloc(&c,n*sizeof(double)));check(cudaMalloc(&mu,n*sizeof(double)));check(cudaMalloc(&next,n*sizeof(double)));check(cudaMemcpy(c,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    std::vector<double> send,recv; dim3 B(8,8,4),G((nx+7)/8,(ny+7)/8,(local+3)/4); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;t++){exchange(c,nx,ny,local,rank,size,send,recv);chemical<<<G,B>>>(c,mu,nx,ny,local,.5,-2.0/9.0,-2.0/9.0,2.0/9.0);check(cudaDeviceSynchronize());exchange(mu,nx,ny,local,rank,size,send,recv);update<<<G,B>>>(c,next,mu,nx,ny,local,.01,1.0);check(cudaDeviceSynchronize());std::swap(c,next);}
    MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now(); double sec=std::chrono::duration<double>(end-start).count(), maxsec;MPI_Reduce(&sec,&maxsec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    check(cudaMemcpy(h.data(),c,n*sizeof(double),cudaMemcpyDeviceToHost)); std::vector<double> global;std::vector<int> counts(size),displs(size);for(int r=0;r<size;r++){size_t l=base+(r<(int)rem);counts[r]=(int)(l*nx*ny);displs[r]=(int)((r*base+std::min((size_t)r,rem))*nx*ny);}if(!rank)global.resize(nx*ny*nz);std::vector<double> core(local*nx*ny);
    #pragma omp parallel for
    for(size_t z=0;z<local;z++)std::copy_n(h.data()+at(1,1,z+1,nx+2,ny+2),nx*ny,core.data()+z*nx*ny);MPI_Gatherv(core.data(),(int)core.size(),MPI_DOUBLE,global.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nComputation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iterations,maxsec*1000,nx*ny*nz*iterations/maxsec/1e6);if(results)print_results(global,"Concentration");if(validate){bool ok=true;double mn=global[0],mx=global[0];for(double v:global){ok&=std::isfinite(v);mn=std::min(mn,v);mx=std::max(mx,v);}printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,ok&&mx<=10&&mn>=-10?"PASSED":"FAILED");if(!ok||mx>10||mn<-10){MPI_Finalize();return 1;}}}
    cudaFree(c);cudaFree(mu);cudaFree(next);MPI_Finalize();return 0;
}
