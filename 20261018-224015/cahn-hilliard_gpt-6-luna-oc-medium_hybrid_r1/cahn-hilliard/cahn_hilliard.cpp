#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ if(worldRank==0) std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } } while(0)

__global__ void chemical(const double* c, double* mu, size_t nx, size_t ny, size_t nz,
                         size_t localNz, double gamma, double aa, double bb, double ab) {
    size_t q=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*localNz;
    if(q>=n) return;
    size_t x=q%nx, y=(q/nx)%ny, z=q/(nx*ny)+1, p=z*nx*ny+y*nx+x;
    size_t xp=x+1<nx?x+1:x, xm=x?x-1:x, yp=y+1<ny?y+1:y, ym=y?y-1:y;
    size_t zp=z+1, zm=z-1;
    double v=c[p], lap=(c[p+y*0+(xp-x)*1]+c[p+(xm-x)]-2*v)
      +(c[p+(yp-y)*nx]+c[p+(ym-y)*nx]-2*v)
      +(c[p+(zp-z)*nx*ny]+c[p+(zm-z)*nx*ny]-2*v);
    mu[q]=4.5*((v+1)*aa+(v-1)*bb-2*v*ab)+3*v+v*v*v-gamma*lap;
}

__global__ void update(const double* old, const double* mu, double* next,
                       size_t nx,size_t ny,size_t localNz,double dt) {
    size_t q=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*localNz;
    if(q>=n) return;
    size_t x=q%nx,y=(q/nx)%ny,z=q/(nx*ny)+1,p=z*nx*ny+y*nx+x,mp=q+nx*ny;
    size_t xp=x+1<nx?x+1:x,xm=x?x-1:x,yp=y+1<ny?y+1:y,ym=y?y-1:y;
    size_t zp=z+1,zm=z-1;
    double m=mu[mp];
    next[q]=old[p]+dt*((mu[mp+(yp-y)*nx]+mu[mp+(ym-y)*nx]-2*m)
       +(mu[mp+(xp-x)]+mu[mp+(xm-x)]-2*m)
       +(mu[mp+(zp-z)*nx*ny]+mu[mp+(zm-z)*nx*ny]-2*m));
}

static int worldRank=0;

static void usage(const char* p){ std::printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv){
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); MPI_Comm_rank(MPI_COMM_WORLD,&worldRank);
    int ranks=1; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int deviceCount=0; cudaGetDeviceCount(&deviceCount); if(deviceCount<1){if(worldRank==0) std::fprintf(stderr,"CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    int localRank=0; MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,worldRank,MPI_INFO_NULL,&node); MPI_Comm_rank(node,&localRank); cudaSetDevice(localRank%deviceCount); MPI_Comm_free(&node);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){ if(!std::strcmp(argv[i],"-x")&&i+1<argc) nx=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-y")&&i+1<argc) ny=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-z")&&i+1<argc) nz=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-i")&&i+1<argc) iterations=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-v")) validate=true; else if(!std::strcmp(argv[i],"-r")) printResults=true; else if(!std::strcmp(argv[i],"-h")){if(worldRank==0)usage(argv[0]); MPI_Finalize(); return 0;} else {if(worldRank==0){std::printf("Unknown option: %s\n",argv[i]);usage(argv[0]);} MPI_Finalize(); return 1;} }
    if(!ny)ny=nx; if(!nz)nz=nx;
    if(ranks>static_cast<int>(nz)){if(worldRank==0)std::fprintf(stderr,"MPI ranks cannot exceed z dimension\n"); MPI_Abort(MPI_COMM_WORLD,1);}
    size_t base=nz/ranks,rem=nz%ranks,localNz=base+(worldRank<static_cast<int>(rem)),zStart=worldRank*base+std::min<size_t>(worldRank,rem),plane=nx*ny,n=plane*localNz;
    if(worldRank==0){std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");}
    std::vector<double> old((localNz+2)*plane), next(n), mu(n+2*plane);
    #pragma omp parallel for schedule(static)
    for(long long q=0;q<static_cast<long long>(n);++q){size_t gz=zStart+q/plane, remq=q%plane; size_t linear=gz*plane+remq; double pseudo=(((linear+1)*1299709)% (nx*ny*nz))/static_cast<double>(nx*ny*nz); old[(q/plane+1)*plane+remq]=-1+2*pseudo;}
    double *dc=nullptr,*dm=nullptr,*dn=nullptr; CUDA_CHECK(cudaMalloc(&dc,old.size()*sizeof(double))); CUDA_CHECK(cudaMalloc(&dm,mu.size()*sizeof(double))); CUDA_CHECK(cudaMalloc(&dn,n*sizeof(double)));
    auto exchange=[&](std::vector<double>& a,int tag){int prev=worldRank?worldRank-1:MPI_PROC_NULL,nextRank=worldRank+1<ranks?worldRank+1:MPI_PROC_NULL; MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,prev,tag,a.data()+(localNz+1)*plane,plane,MPI_DOUBLE,nextRank,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE); MPI_Sendrecv(a.data()+localNz*plane,plane,MPI_DOUBLE,nextRank,tag+1,a.data(),plane,MPI_DOUBLE,prev,tag+1,MPI_COMM_WORLD,MPI_STATUS_IGNORE); if(worldRank==0)std::copy_n(a.data()+plane,plane,a.data()); if(worldRank==ranks-1)std::copy_n(a.data()+localNz*plane,plane,a.data()+(localNz+1)*plane);};
    // Physical z boundaries are replicated into the clamped ghost planes.
    exchange(old,10);
    auto start=std::chrono::steady_clock::now();
    for(int t=0;t<iterations;++t){
        CUDA_CHECK(cudaMemcpy(dc,old.data(),old.size()*sizeof(double),cudaMemcpyHostToDevice));
        chemical<<<(n+255)/256,256>>>(dc,dm,nx,ny,nz,localNz,0.5,-2.0/9.0,-2.0/9.0,2.0/9.0); CUDA_CHECK(cudaGetLastError());
        // Chemical potential halos are exchanged on host; local kernels use contiguous slabs.
        CUDA_CHECK(cudaMemcpy(mu.data()+plane,dm,n*sizeof(double),cudaMemcpyDeviceToHost));
        int prev=worldRank?worldRank-1:MPI_PROC_NULL,nextRank=worldRank+1<ranks?worldRank+1:MPI_PROC_NULL;
        MPI_Sendrecv(mu.data()+plane,plane,MPI_DOUBLE,prev,20,mu.data()+(localNz+1)*plane,plane,MPI_DOUBLE,nextRank,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(mu.data()+localNz*plane,plane,MPI_DOUBLE,nextRank,21,mu.data(),plane,MPI_DOUBLE,prev,21,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(worldRank==0)std::copy_n(mu.data()+plane,plane,mu.data()); if(worldRank==ranks-1)std::copy_n(mu.data()+localNz*plane,plane,mu.data()+(localNz+1)*plane);
        // Pad local chemical-potential values into the same ghosted layout expected by update.
        CUDA_CHECK(cudaMemcpy(dm,mu.data(),mu.size()*sizeof(double),cudaMemcpyHostToDevice));
        update<<<(n+255)/256,256>>>(dc,dm,dn,nx,ny,localNz,0.01); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(next.data(),dn,n*sizeof(double),cudaMemcpyDeviceToHost));
        std::copy(next.begin(),next.end(),old.begin()+plane); exchange(old,30);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); auto end=std::chrono::steady_clock::now();
    double localSeconds=std::chrono::duration<double>(end-start).count(), seconds=0.0;
    MPI_Reduce(&localSeconds,&seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<int> counts(ranks),displs(ranks); for(int r=0;r<ranks;++r){size_t rz=nz/ranks+(r<static_cast<int>(nz%ranks)); counts[r]=static_cast<int>(rz*plane); displs[r]=static_cast<int>((r*(nz/ranks)+std::min<size_t>(r,nz%ranks))*plane);}
    std::vector<double> result(worldRank==0?nx*ny*nz:0); MPI_Gatherv(old.data()+plane,static_cast<int>(n),MPI_DOUBLE,result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(worldRank==0){ std::printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000)); std::printf("Performance: %.3f MCellUpdates/s\n",seconds>0?nx*ny*nz*static_cast<double>(iterations)/seconds/1e6:0.0); if(printResults)print_results(result,"Concentration"); if(validate){double mn=*std::min_element(result.begin(),result.end()),mx=*std::max_element(result.begin(),result.end()); bool valid=std::all_of(result.begin(),result.end(),[](double x){return std::isfinite(x);})&&mn>=-10&&mx<=10; std::printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,valid?"PASSED":"FAILED"); if(!valid){cudaFree(dc);cudaFree(dm);cudaFree(dn);MPI_Finalize();return 1;}}}
    cudaFree(dc);cudaFree(dm);cudaFree(dn); MPI_Finalize(); return 0;
}
