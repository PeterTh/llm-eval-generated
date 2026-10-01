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

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

__device__ inline size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny){return z*(nx*ny)+y*nx+x;}
__device__ inline double lap(const double* a,size_t x,size_t y,size_t z,size_t nx,size_t ny,size_t nz){
    const size_t xm=x?x-1:x, xp=x+1<nx?x+1:x, ym=y?y-1:y, yp=y+1<ny?y+1:y;
    return a[at(xp,y,z,nx,ny)]+a[at(xm,y,z,nx,ny)]-2*a[at(x,y,z,nx,ny)] +
           a[at(x,yp,z,nx,ny)]+a[at(x,ym,z,nx,ny)]-2*a[at(x,y,z,nx,ny)] +
           a[at(x,y,z+1,nx,ny)]+a[at(x,y,z-1,nx,ny)]-2*a[at(x,y,z,nx,ny)];
}
__global__ void potential(const double* c,double* mu,size_t nx,size_t ny,size_t nz,size_t lz){
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*lz; if(q>=n)return;
    size_t z=q/(nx*ny)+1, y=(q%(nx*ny))/nx, x=q%nx;
    double v=c[at(x,y,z,nx,ny)];
    mu[at(x,y,z,nx,ny)]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-0.5*lap(c,x,y,z,nx,ny,nz);
}
__global__ void update(double* out,const double* old,const double* mu,size_t nx,size_t ny,size_t lz){
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*lz;if(q>=n)return;
    size_t z=q/(nx*ny)+1,y=(q%(nx*ny))/nx,x=q%nx; size_t i=at(x,y,z,nx,ny);
    out[i]=old[i]+0.01*lap(mu,x,y,z,nx,ny,lz);
}

int main(int argc,char** argv){
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int devices=0; CUDA(cudaGetDeviceCount(&devices)); if(!devices){if(rank==0)fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,2);} CUDA(cudaSetDevice(rank%devices));
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-x")&&i+1<argc)nx=atoi(argv[++i]);else if(!strcmp(argv[i],"-y")&&i+1<argc)ny=atoi(argv[++i]);else if(!strcmp(argv[i],"-z")&&i+1<argc)nz=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iterations=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(rank==0)printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]);MPI_Finalize();return 0;}else{if(rank==0)fprintf(stderr,"Unknown option: %s\n",argv[i]);MPI_Abort(MPI_COMM_WORLD,1);}}
    if(!ny)ny=nx;if(!nz)nz=nx; if(nz<(size_t)size){if(rank==0)fprintf(stderr,"z dimension must be at least MPI rank count\n");MPI_Abort(MPI_COMM_WORLD,1);}
    size_t base=nz/size,rem=nz%size,lz=base+(rank<(int)rem),z0=rank*base+std::min((size_t)rank,rem),plane=nx*ny,vol=nx*ny*lz,alloc=vol+2*plane;
    if(rank==0){printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");}
    std::vector<double> h(alloc,0.0), lower(plane), upper(plane), gathered; double *a,*b,*m; CUDA(cudaMalloc(&a,alloc*sizeof(double)));CUDA(cudaMalloc(&b,alloc*sizeof(double)));CUDA(cudaMalloc(&m,alloc*sizeof(double)));
    #pragma omp parallel for schedule(static)
    for(long long q=0;q<(long long)vol;++q){size_t gz=z0+(size_t)q/plane, remq=(size_t)q%plane, linear=gz*plane+remq; h[plane+(size_t)q]=-1+2.0*((((linear+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));}
    CUDA(cudaMemcpy(a,h.data(),alloc*sizeof(double),cudaMemcpyHostToDevice));
    int left=rank?rank-1:MPI_PROC_NULL,right=rank+1<size?rank+1:MPI_PROC_NULL;
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t){
        CUDA(cudaMemcpy(upper.data(),a+(lz)*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(lower.data(),a+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(upper.data(),(int)plane,MPI_DOUBLE,right,0,lower.data(),(int)plane,MPI_DOUBLE,left,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        // Exchange the lower interior plane in the reverse direction for the upper halo.
        CUDA(cudaMemcpy(h.data(),a+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h.data(),(int)plane,MPI_DOUBLE,left,1,upper.data(),(int)plane,MPI_DOUBLE,right,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank==0) std::copy(h.begin()+plane,h.begin()+2*plane,lower.begin());
        if(rank==size-1) std::copy(h.begin()+plane+vol-plane,h.begin()+plane+vol,upper.begin());
        CUDA(cudaMemcpy(a,lower.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(a+plane+vol,upper.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        potential<<<(vol+255)/256,256>>>(a,m,nx,ny,nz,lz); CUDA(cudaGetLastError());
        CUDA(cudaMemcpy(upper.data(),m+(lz)*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(lower.data(),m+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(upper.data(),(int)plane,MPI_DOUBLE,right,2,lower.data(),(int)plane,MPI_DOUBLE,left,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        CUDA(cudaMemcpy(h.data(),m+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h.data(),(int)plane,MPI_DOUBLE,left,3,upper.data(),(int)plane,MPI_DOUBLE,right,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank==0)std::copy(h.begin(),h.begin()+plane,lower.begin());
        if(rank==size-1) CUDA(cudaMemcpy(upper.data(),m+(lz)*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(m,lower.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(m+plane+vol,upper.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        update<<<(vol+255)/256,256>>>(b,a,m,nx,ny,lz);CUDA(cudaGetLastError()); std::swap(a,b);
    }
    CUDA(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    CUDA(cudaMemcpy(h.data(),a,alloc*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<int> counts(size),displs(size); for(int r=0;r<size;++r){size_t rz=nz/size+(r<(int)rem);counts[r]=(int)(rz*plane);displs[r]=(int)((r*(nz/size)+std::min((size_t)r,rem))*plane);}
    if(rank==0)gathered.resize(nx*ny*nz); MPI_Gatherv(h.data()+plane,(int)vol,MPI_DOUBLE,rank==0?gathered.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();printf("Computation time: %ld ms\n",ms);double mcups=(double)nx*ny*nz*iterations/(ms/1000.0)/1e6;printf("Performance: %.3f MCellUpdates/s\n",mcups);if(printResults)print_results(gathered,"Concentration");if(validate){bool ok=true;double mn=gathered[0],mx=mn;
        #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(&:ok)
        for(long long i=0;i<(long long)gathered.size();++i){double v=gathered[i];if(!std::isfinite(v))ok=false;mn=std::min(mn,v);mx=std::max(mx,v);}printf("Concentration range: [%.6f, %.6f]\n",mn,mx);if(mx>10||mn< -10)ok=false;printf("Validation: %s\n",ok?"PASSED":"FAILED");CUDA(cudaFree(a));CUDA(cudaFree(b));CUDA(cudaFree(m));MPI_Finalize();return ok?0:1;}}
    CUDA(cudaFree(a));CUDA(cudaFree(b));CUDA(cudaFree(m));MPI_Finalize();return 0;
}
