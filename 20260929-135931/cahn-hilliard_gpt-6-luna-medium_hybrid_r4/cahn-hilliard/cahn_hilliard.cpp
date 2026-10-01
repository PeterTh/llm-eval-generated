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

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

__device__ inline size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny) { return (z*ny+y)*nx+x; }
__device__ inline double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz) {
    size_t xp=x+1<nx?x+1:x, xm=x?x-1:0, yp=y+1<ny?y+1:y, ym=y?y-1:0;
    size_t zp=z+1<nz?z+1:z, zm=z?z-1:0;
    return a[at(xp,y,z,nx,ny)]+a[at(xm,y,z,nx,ny)]-2*a[at(x,y,z,nx,ny)] +
           a[at(x,yp,z,nx,ny)]+a[at(x,ym,z,nx,ny)]-2*a[at(x,y,z,nx,ny)] +
           a[at(x,y,zp,nx,ny)]+a[at(x,y,zm,nx,ny)]-2*a[at(x,y,z,nx,ny)];
}
__global__ void chemical(const double* c,double* mu,size_t nx,size_t ny,size_t nz,size_t nlocal,size_t zstart,size_t globalz) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*nlocal;
    if(i>=n) return;
    size_t x=i%nx,y=(i/nx)%ny,z=i/(nx*ny)+1,gz=zstart+z-1;
    double v=c[at(x,y,z,nx,ny)];
    mu[at(x,y,z,nx,ny)]=-v+v*v*v-0.5*lap(c,x,y,z,nx,ny, nlocal+2);
    (void)globalz; (void)gz;
}
__global__ void update(const double* c,double* out,const double* mu,size_t nx,size_t ny,size_t nlocal,size_t globalz,size_t zstart) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nlocal;
    if(i>=n) return;
    size_t x=i%nx,y=(i/nx)%ny,z=i/(nx*ny)+1,gz=zstart+z-1;
    size_t zp=(gz+1<globalz)?z+1:z, zm=(gz>0)?z-1:z;
    // mu has exchanged halo planes and the same x/y clamped boundaries.
    double l=mu[at(x+1<nx?x+1:x,y,z,nx,ny)]+mu[at(x?x-1:0,y,z,nx,ny)]-2*mu[at(x,y,z,nx,ny)] +
             mu[at(x,y+1<ny?y+1:y,z,nx,ny)]+mu[at(x,y?y-1:0,z,nx,ny)]-2*mu[at(x,y,z,nx,ny)] +
             mu[at(x,y,zp,nx,ny)]+mu[at(x,y,zm,nx,ny)]-2*mu[at(x,y,z,nx,ny)];
    out[at(x,y,z,nx,ny)]=c[at(x,y,z,nx,ny)]+0.01*l;
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")){if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0;}
        else {if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1);}
    }
    if(!ny) ny=nx; if(!nz) nz=nx;
    if(size>(int)nz){if(rank==0) fprintf(stderr,"MPI ranks cannot exceed z dimension\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    int base=(int)(nz/size),rem=(int)(nz%size),nlocal=base+(rank<rem),zstart=rank*base+std::min(rank,rem);
    int devcount=0; CUDA_CHECK(cudaGetDeviceCount(&devcount)); if(!devcount){fprintf(stderr,"No CUDA device available on rank %d\n",rank); MPI_Abort(MPI_COMM_WORLD,1);} CUDA_CHECK(cudaSetDevice(rank%devcount));
    const size_t plane=nx*ny, pitch=(size_t)(nlocal+2)*plane;
    std::vector<double> hC(pitch,0.0),hM(pitch,0.0),hOut(pitch,0.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)nlocal*(long long)plane;++i) { size_t g=(size_t)zstart*plane+(size_t)i, local=(size_t)i+plane; hC[local]=-1.0+2.0*((((g+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz)); }
    double *dC,*dM,*dOut; CUDA_CHECK(cudaMalloc((void**)&dC,pitch*sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&dM,pitch*sizeof(double))); CUDA_CHECK(cudaMalloc((void**)&dOut,pitch*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dC,hC.data(),pitch*sizeof(double),cudaMemcpyHostToDevice));
    if(rank==0) printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::steady_clock::now();
    auto exchange=[&](double* d,std::vector<double>& h){
        CUDA_CHECK(cudaMemcpy(h.data(),d,pitch*sizeof(double),cudaMemcpyDeviceToHost));
        int prev=rank?rank-1:MPI_PROC_NULL,next=rank+1<size?rank+1:MPI_PROC_NULL;
        MPI_Sendrecv(h.data()+plane,plane,MPI_DOUBLE,prev,10,h.data()+(size_t)(nlocal+1)*plane,plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(h.data()+(size_t)nlocal*plane,plane,MPI_DOUBLE,next,11,h.data(),plane,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank==0) memcpy(h.data(),h.data()+plane,plane*sizeof(double));
        if(rank==size-1) memcpy(h.data()+(size_t)(nlocal+1)*plane,h.data()+(size_t)nlocal*plane,plane*sizeof(double));
        CUDA_CHECK(cudaMemcpy(d,h.data(),pitch*sizeof(double),cudaMemcpyHostToDevice));
    };
    for(int t=0;t<iterations;++t) {
        exchange(dC,hC);
        chemical<<<(nlocal*plane+255)/256,256>>>(dC,dM,nx,ny,nz,nlocal,zstart,nz); CUDA_CHECK(cudaGetLastError());
        exchange(dM,hM);
        update<<<(nlocal*plane+255)/256,256>>>(dC,dOut,dM,nx,ny,nlocal,nz,zstart); CUDA_CHECK(cudaGetLastError());
        std::swap(dC,dOut);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),maxElapsed;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(hC.data(),dC,pitch*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<int> counts(size),displs(size); for(int r=0;r<size;++r){int nr=(int)(nz/size)+(r<(int)(nz%size));counts[r]=nr*(int)plane;displs[r]=(r*(int)(nz/size)+std::min(r,(int)(nz%size)))*(int)plane;}
    std::vector<double> result(rank==0?nx*ny*nz:0);
    MPI_Gatherv(hC.data()+plane,nlocal*(int)plane,MPI_DOUBLE,result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){long ms=(long)(maxElapsed*1000); printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",ms,(double)(nx*ny*nz)*iterations/maxElapsed/1e6);
        if(printResults) print_results(result,"Concentration");
        if(validate){double lo=result[0],hi=result[0]; int ok=1;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:ok)
            for(long long i=0;i<(long long)result.size();++i){double v=result[(size_t)i]; if(!std::isfinite(v)) ok=0; lo=std::min(lo,v);hi=std::max(hi,v);}
            printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",lo,hi,(ok&&lo>=-10&&hi<=10)?"PASSED":"FAILED"); if(!(ok&&lo>=-10&&hi<=10)){MPI_Abort(MPI_COMM_WORLD,1);}}
    }
    CUDA_CHECK(cudaFree(dC)); CUDA_CHECK(cudaFree(dM)); CUDA_CHECK(cudaFree(dOut)); MPI_Finalize(); return 0;
}
