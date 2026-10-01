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

__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t localNz,size_t zStart,size_t globalNz,size_t vol){
    size_t q=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*localNz;
    if(q>=n)return;
    size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny), p=(z+1)*nx*ny+q%(nx*ny);
    size_t xm=x?x-1:x,xp=x+1<nx?x+1:x,ym=y?y-1:y,yp=y+1<ny?y+1:y;
    double v=c[p], lap=(c[p-(x-xm)]+c[p+(xp-x)]-2*v)+(c[p-(y-ym)*nx]+c[p+(yp-y)*nx]-2*v);
    size_t gz=zStart+z;
    size_t zm=(z>0)?p-nx*ny:(gz>0?p-nx*ny:p),zp=(z+1<localNz)?p+nx*ny:(gz+1<globalNz?p+nx*ny:p);
    lap += c[zm]+c[zp]-2*v;
    double eAA=-2.0/9.0,eBB=-2.0/9.0,eAB=2.0/9.0;
    (void)vol;
    mu[p]=4.5*((v+1)*eAA+(v-1)*eBB-2*v*eAB)+3*v+v*v*v-0.5*lap;
}

__global__ void update(const double* old,const double* mu,double* next,size_t nx,size_t ny,size_t localNz,size_t zStart,size_t globalNz){
    size_t q=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*localNz;if(q>=n)return;
    size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny),p=(z+1)*nx*ny+q%(nx*ny);
    size_t xm=x?x-1:x,xp=x+1<nx?x+1:x,ym=y?y-1:y,yp=y+1<ny?y+1:y;
    double v=mu[p],lap=mu[p-(x-xm)]+mu[p+(xp-x)]-2*v+mu[p-(y-ym)*nx]+mu[p+(yp-y)*nx]-2*v;
    size_t gz=zStart+z, zm=(z>0||gz>0)?p-nx*ny:p,zp=(z+1<localNz||gz+1<globalNz)?p+nx*ny:p;
    lap+=mu[zm]+mu[zp]-2*v;next[p]=old[p]+0.01*lap;
}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0;int iterations=20;bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Abort(MPI_COMM_WORLD,1);}}
    if(!ny)ny=nx;if(!nz)nz=nx;if(nz<(size_t)size){if(!rank)fprintf(stderr,"Z dimension must be at least the MPI rank count\n");MPI_Abort(MPI_COMM_WORLD,1);}
    MPI_Comm localComm;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&localComm);int localRank;MPI_Comm_rank(localComm,&localRank);int devices=0;CUDA_CHECK(cudaGetDeviceCount(&devices));if(!devices){fprintf(stderr,"No CUDA device available on rank %d\n",rank);MPI_Abort(MPI_COMM_WORLD,1);}CUDA_CHECK(cudaSetDevice(localRank%devices));MPI_Comm_free(&localComm);
    size_t z0=nz*(size_t)rank/(size_t)size,z1=nz*(size_t)(rank+1)/(size_t)size,lz=z1-z0,plane=nx*ny,n=plane*lz;
    if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");}
    std::vector<double> h(n);
    #pragma omp parallel for
    for(long long q=0;q<(long long)n;++q){size_t g=z0*plane+(size_t)q, pseudo=((g+1)*1299709)% (nx*ny*nz);h[q]=-1.0+2.0*(double)pseudo/(double)(nx*ny*nz);}
    size_t alloc=(lz+2)*plane;double *a,*b,*m;CUDA_CHECK(cudaMalloc(&a,alloc*sizeof(double)));CUDA_CHECK(cudaMalloc(&b,alloc*sizeof(double)));CUDA_CHECK(cudaMalloc(&m,alloc*sizeof(double)));CUDA_CHECK(cudaMemcpy(a+plane,h.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    int left=rank?rank-1:MPI_PROC_NULL,right=rank+1<size?rank+1:MPI_PROC_NULL;std::vector<double> send(plane),recv(plane);
    auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t){
        // Exchange concentration halos; outer boundaries copy their nearest interior plane.
        CUDA_CHECK(cudaMemcpy(send.data(),a+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,left,10,recv.data(),plane,MPI_DOUBLE,right,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);if(right!=MPI_PROC_NULL)CUDA_CHECK(cudaMemcpy(a+(lz+1)*plane,recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));else CUDA_CHECK(cudaMemcpy(a+(lz+1)*plane,a+lz*plane,plane*sizeof(double),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(send.data(),a+lz*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,right,11,recv.data(),plane,MPI_DOUBLE,left,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);if(left!=MPI_PROC_NULL)CUDA_CHECK(cudaMemcpy(a,recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));else CUDA_CHECK(cudaMemcpy(a,a+plane,plane*sizeof(double),cudaMemcpyDeviceToDevice));
        chemical<<<(n+255)/256,256>>>(a,m,nx,ny,lz,z0,nz,nx*ny*nz);CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(send.data(),m+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,left,20,recv.data(),plane,MPI_DOUBLE,right,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);if(right!=MPI_PROC_NULL)CUDA_CHECK(cudaMemcpy(m+(lz+1)*plane,recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));else CUDA_CHECK(cudaMemcpy(m+(lz+1)*plane,m+lz*plane,plane*sizeof(double),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(send.data(),m+lz*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));MPI_Sendrecv(send.data(),plane,MPI_DOUBLE,right,21,recv.data(),plane,MPI_DOUBLE,left,21,MPI_COMM_WORLD,MPI_STATUS_IGNORE);if(left!=MPI_PROC_NULL)CUDA_CHECK(cudaMemcpy(m,recv.data(),plane*sizeof(double),cudaMemcpyHostToDevice));else CUDA_CHECK(cudaMemcpy(m,m+plane,plane*sizeof(double),cudaMemcpyDeviceToDevice));
        update<<<(n+255)/256,256>>>(a,m,b,nx,ny,lz,z0,nz);CUDA_CHECK(cudaGetLastError());std::swap(a,b);
    }
    CUDA_CHECK(cudaDeviceSynchronize());auto end=std::chrono::high_resolution_clock::now();double secs=std::chrono::duration<double>(end-start).count(),maxSecs;MPI_Reduce(&secs,&maxSecs,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(h.data(),a+plane,n*sizeof(double),cudaMemcpyDeviceToHost));std::vector<int> counts(size),displs(size);for(int r=0;r<size;++r){size_t s=nz*(size_t)r/(size_t)size,e=nz*(size_t)(r+1)/(size_t)size;counts[r]=(int)((e-s)*plane);displs[r]=(int)(s*plane);}std::vector<double> all;if(!rank)all.resize(nx*ny*nz);MPI_Gatherv(h.data(),(int)n,MPI_DOUBLE,rank?nullptr:all.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(!rank){long ms=(long)(maxSecs*1000);printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",ms,(double)nx*ny*nz*iterations/(maxSecs*1e6));if(printResults)print_results(all,"Concentration");if(validate){double mn=*std::min_element(all.begin(),all.end()),mx=*std::max_element(all.begin(),all.end());printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,(mn>=-10&&mx<=10)?"PASSED":"FAILED");}}
    CUDA_CHECK(cudaFree(a));CUDA_CHECK(cudaFree(b));CUDA_CHECK(cudaFree(m));MPI_Finalize();return 0;
}
