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

__device__ __forceinline__ size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny){return (z*ny+y)*nx+x;}
__device__ __forceinline__ double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz){
  size_t xp=x+1<nx?x+1:x,xm=x?x-1:0,yp=y+1<ny?y+1:y,ym=y?y-1:0,zp=z+1<=nz?z+1:z,zm=z?z-1:0; double v=a[at(x,y,z,nx,ny)];
  return a[at(xp,y,z,nx,ny)]+a[at(xm,y,z,nx,ny)]-2*v+a[at(x,yp,z,nx,ny)]+a[at(x,ym,z,nx,ny)]-2*v+a[at(x,y,zp,nx,ny)]+a[at(x,y,zm,nx,ny)]-2*v;
}
__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t nz){size_t q=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(q>=n)return; size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1; double v=c[at(x,y,z,nx,ny)]; mu[at(x,y,z,nx,ny)]=4.5*((v+1)*(-2.0/9)+(v-1)*(-2.0/9)-2*v*(2.0/9))+3*v+v*v*v-.5*lap(c,x,y,z,nx,ny,nz);}
__global__ void update(const double* old,const double* mu,double* next,size_t nx,size_t ny,size_t nz){size_t q=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(q>=n)return;size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1;next[at(x,y,z,nx,ny)]=old[at(x,y,z,nx,ny)]+.01*lap(mu,x,y,z,nx,ny,nz);}
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,2);}}
static void exchange(double* a,size_t nx,size_t ny,size_t lz,int rank,int size){size_t plane=nx*ny; MPI_Request r[4]; int n=0; if(rank){MPI_Irecv(a,plane,MPI_DOUBLE,rank-1,7,MPI_COMM_WORLD,&r[n++]);MPI_Isend(a+plane,plane,MPI_DOUBLE,rank-1,8,MPI_COMM_WORLD,&r[n++]);} if(rank+1<size){MPI_Irecv(a+(lz+1)*plane,plane,MPI_DOUBLE,rank+1,8,MPI_COMM_WORLD,&r[n++]);MPI_Isend(a+lz*plane,plane,MPI_DOUBLE,rank+1,7,MPI_COMM_WORLD,&r[n++]);} MPI_Waitall(n,r,MPI_STATUSES_IGNORE);}
static void usage(const char* p){printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p);}
int main(int argc,char** argv){MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);size_t nx=64,ny=0,nz=0;int it=20;bool val=false,print=false;
 for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)it=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))print=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else {if(!rank)usage(argv[0]);MPI_Finalize();return 1;}} if(!ny)ny=nx;if(!nz)nz=nx;
 int ng=(int)(nz/size)+(rank<(int)(nz%size)); size_t z0=(size_t)rank*(nz/size)+(size_t)std::min(rank,(int)(nz%size)); size_t plane=nx*ny, local=(size_t)ng*plane, total=(size_t)(ng+2)*plane; int devs=0;ck(cudaGetDeviceCount(&devs));if(!devs){fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,3);}ck(cudaSetDevice(rank%devs));
 std::vector<double> h(total), next(total), mu(total);
 #pragma omp parallel for
 for(size_t q=0;q<local;q++){size_t global=q+z0*plane;h[plane+q]=-1+2*((((global+1)*1299709)%((size_t)nx*ny*nz))/(double)((size_t)nx*ny*nz));}
 double *dold,*dnext,*dmu;ck(cudaMalloc(&dold,total*sizeof(double)));ck(cudaMalloc(&dnext,total*sizeof(double)));ck(cudaMalloc(&dmu,total*sizeof(double)));ck(cudaMemcpy(dold,h.data(),total*sizeof(double),cudaMemcpyHostToDevice));
 MPI_Barrier(MPI_COMM_WORLD);auto start=std::chrono::high_resolution_clock::now();int threads=256;size_t blocks=(local+threads-1)/threads;
 for(int t=0;t<it;t++){ck(cudaMemcpy(h.data(),dold,total*sizeof(double),cudaMemcpyDeviceToHost));if(!rank)std::copy_n(h.data()+plane,plane,h.data());if(rank+1==size)std::copy_n(h.data()+ng*plane,plane,h.data()+(ng+1)*plane);exchange(h.data(),nx,ny,ng,rank,size);ck(cudaMemcpy(dold,h.data(),total*sizeof(double),cudaMemcpyHostToDevice));chemical<<<blocks,threads>>>(dold,dmu,nx,ny,ng);ck(cudaGetLastError());ck(cudaDeviceSynchronize());ck(cudaMemcpy(h.data(),dmu,total*sizeof(double),cudaMemcpyDeviceToHost));if(!rank)std::copy_n(h.data()+plane,plane,h.data());if(rank+1==size)std::copy_n(h.data()+ng*plane,plane,h.data()+(ng+1)*plane);exchange(h.data(),nx,ny,ng,rank,size);ck(cudaMemcpy(dmu,h.data(),total*sizeof(double),cudaMemcpyHostToDevice));update<<<blocks,threads>>>(dold,dmu,dnext,nx,ny,ng);ck(cudaGetLastError());std::swap(dold,dnext);}
 ck(cudaMemcpy(h.data(),dold,total*sizeof(double),cudaMemcpyDeviceToHost));MPI_Barrier(MPI_COMM_WORLD);auto ms=std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-start).count();double worst;MPI_Reduce(&ms,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nComputation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,it,worst,(double)nx*ny*nz*it/(worst/1000)/1e6);}
 if(print){std::vector<int> cnt(size),dis(size);int off=0;for(int r=0;r<size;r++){cnt[r]=(int)(((nz/size)+(r<(int)(nz%size)))*plane);dis[r]=off;off+=cnt[r];}std::vector<double> all(rank?0:(size_t)nx*ny*nz);MPI_Gatherv(h.data()+plane,(int)local,MPI_DOUBLE,all.data(),cnt.data(),dis.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);if(!rank)print_results(all,"Concentration");}
 if(val){int bad=0;double lo=1e300,hi=-1e300;for(size_t q=0;q<local;q++){double v=h[plane+q];bad|=!std::isfinite(v);lo=std::min(lo,v);hi=std::max(hi,v);}double glo,ghi;int gbad;MPI_Reduce(&lo,&glo,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);MPI_Reduce(&hi,&ghi,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);MPI_Reduce(&bad,&gbad,1,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);if(!rank){printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",glo,ghi,(!gbad&&glo>=-10&&ghi<=10)?"PASSED":"FAILED");}if(gbad||glo<-10||ghi>10){cudaFree(dold);cudaFree(dnext);cudaFree(dmu);MPI_Finalize();return 1;}}
 cudaFree(dold);cudaFree(dnext);cudaFree(dmu);MPI_Finalize();return 0;}
