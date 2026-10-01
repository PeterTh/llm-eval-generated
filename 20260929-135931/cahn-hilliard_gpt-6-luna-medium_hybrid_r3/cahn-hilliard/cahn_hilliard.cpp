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

__device__ inline size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny) { return (z*ny+y)*nx+x; }
__device__ inline double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz) {
    size_t xm=x?x-1:x, xp=x+1<nx?x+1:x, ym=y?y-1:y, yp=y+1<ny?y+1:y;
    // Local z=1..nz; halos occupy z=0 and nz+1. Physical edge ranks clamp instead.
    size_t zm=z==1 ? (a[at(x,y,0,nx,ny)]==a[at(x,y,1,nx,ny)] ? 1 : 0) : z-1;
    size_t zp=z==nz ? nz+1 : z+1;
    double c=a[at(x,y,z,nx,ny)];
    return a[at(xp,y,z,nx,ny)]+a[at(xm,y,z,nx,ny)]-2*c + a[at(x,yp,z,nx,ny)]+a[at(x,ym,z,nx,ny)]-2*c + a[at(x,y,zp,nx,ny)]+a[at(x,y,zm,nx,ny)]-2*c;
}
__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t nz,bool first,bool last) {
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*nz; if(q>=n)return;
    size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1, k=at(x,y,z,nx,ny); double v=c[k];
    size_t xm=x?x-1:x,xp=x+1<nx?x+1:x,ym=y?y-1:y,yp=y+1<ny?y+1:y;
    size_t zm=z==1?(first?1:0):z-1,zp=z==nz?(last?nz:nz+1):z+1;
    double l=(c[at(xp,y,z,nx,ny)]+c[at(xm,y,z,nx,ny)]-2*v)+(c[at(x,yp,z,nx,ny)]+c[at(x,ym,z,nx,ny)]-2*v)+(c[at(x,y,zp,nx,ny)]+c[at(x,y,zm,nx,ny)]-2*v);
    mu[k]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-0.5*l;
}
__global__ void update(const double* old,const double* mu,double* out,size_t nx,size_t ny,size_t nz,bool first,bool last) {
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(q>=n)return;
    size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1,k=at(x,y,z,nx,ny);
    size_t xm=x?x-1:x,xp=x+1<nx?x+1:x,ym=y?y-1:y,yp=y+1<ny?y+1:y;
    size_t zm=z==1?(first?1:0):z-1,zp=z==nz?(last?nz:nz+1):z+1;
    double v=mu[k], l=(mu[at(xp,y,z,nx,ny)]+mu[at(xm,y,z,nx,ny)]-2*v)+(mu[at(x,yp,z,nx,ny)]+mu[at(x,ym,z,nx,ny)]-2*v)+(mu[at(x,y,zp,nx,ny)]+mu[at(x,y,zm,nx,ny)]-2*v);
    out[k]=old[k]+0.01*l;
}
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,2);}}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    int deviceCount=0; cudaGetDeviceCount(&deviceCount); if(!deviceCount){if(!rank)fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,2);} ck(cudaSetDevice(rank%deviceCount));
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]);MPI_Finalize();return 0;}else {if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Abort(MPI_COMM_WORLD,1);}}
    if(!ny)ny=nx;if(!nz)nz=nx;
    if(nz<(size_t)size){if(!rank)fprintf(stderr,"z dimension must be at least MPI rank count\n");MPI_Abort(MPI_COMM_WORLD,1);}
    size_t base=nz/size,rem=nz%size,local=base+(rank<(int)rem),zstart=(size_t)rank*base+std::min((size_t)rank,rem), plane=nx*ny, alloc=(local+2)*plane, cells=local*plane;
    std::vector<double> a(alloc),b(alloc),m(alloc);
    #pragma omp parallel for
    for(long long q=0;q<(long long)cells;++q){size_t g=zstart*plane+(size_t)q, pseudo=(((g+1)*1299709)%(nx*ny*nz));a[plane+(size_t)q]=-1.0+2.0*(double)pseudo/(double)(nx*ny*nz);}
    double *da,*dm,*db;ck(cudaMalloc(&da,alloc*sizeof(double)));ck(cudaMalloc(&dm,alloc*sizeof(double)));ck(cudaMalloc(&db,alloc*sizeof(double)));
    if(!rank){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");}
    MPI_Barrier(MPI_COMM_WORLD);auto start=std::chrono::high_resolution_clock::now();
    int prev=rank?rank-1:MPI_PROC_NULL,next=rank+1<size?rank+1:MPI_PROC_NULL;
    for(int t=0;t<iterations;++t){
        MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,prev,10,a.data()+(local+1)*plane,plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+local*plane,plane,MPI_DOUBLE,next,11,a.data(),plane,MPI_DOUBLE,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        ck(cudaMemcpy(da,a.data(),alloc*sizeof(double),cudaMemcpyHostToDevice)); dim3 block(256),grid((cells+255)/256);
        chemical<<<grid,block>>>(da,dm,nx,ny,local,rank==0,rank==size-1);ck(cudaGetLastError());ck(cudaMemcpy(m.data(),dm,alloc*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(m.data()+plane,plane,MPI_DOUBLE,prev,12,m.data()+(local+1)*plane,plane,MPI_DOUBLE,next,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(m.data()+local*plane,plane,MPI_DOUBLE,next,13,m.data(),plane,MPI_DOUBLE,prev,13,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        ck(cudaMemcpy(dm,m.data(),alloc*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(db,a.data(),alloc*sizeof(double),cudaMemcpyHostToDevice));
        update<<<grid,block>>>(db,dm,da,nx,ny,local,rank==0,rank==size-1);ck(cudaGetLastError());ck(cudaMemcpy(b.data(),da,alloc*sizeof(double),cudaMemcpyDeviceToHost));a.swap(b);
    }
    MPI_Barrier(MPI_COMM_WORLD);auto end=std::chrono::high_resolution_clock::now();double secs=std::chrono::duration<double>(end-start).count(),maxsecs;MPI_Reduce(&secs,&maxsecs,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank){printf("Computation time: %ld ms\n",(long)(maxsecs*1000));printf("Performance: %.3f MCellUpdates/s\n",(double)nx*ny*nz*iterations/maxsecs/1e6);}
    std::vector<double> full;if(printResults||validate){std::vector<int> counts(size),displs(size);for(int r=0;r<size;++r){size_t nr=nz/size+(r<(int)(nz%size));counts[r]=(int)(nr*plane);displs[r]=(int)((nz/size*r+std::min((size_t)r,nz%size))*plane);}full.resize(nx*ny*nz);MPI_Gatherv(a.data()+plane,(int)cells,MPI_DOUBLE,full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);}
    int valid=1;if(rank==0&&printResults)print_results(full,"Concentration");
    if(validate){if(rank==0){double lo=full[0],hi=full[0];for(double v:full){if(!std::isfinite(v))valid=0;lo=std::min(lo,v);hi=std::max(hi,v);}printf("Concentration range: [%.6f, %.6f]\n",lo,hi);if(hi>10||lo< -10)valid=0;printf("Validating result...\nValidation: %s\n",valid?"PASSED":"FAILED");}MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD);}
    cudaFree(da);cudaFree(dm);cudaFree(db);MPI_Finalize();return validate&&!valid?1:0;
}
