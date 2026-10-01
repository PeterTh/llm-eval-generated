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

__device__ __forceinline__ size_t at(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ double lap(const double* a, size_t x, size_t y, size_t z,
                                      size_t nx, size_t ny, size_t nz, double invx2,
                                      double invy2, double invz2) {
    const size_t xp = x + (x + 1 < nx), xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), ym = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zm = z - (z > 0);
    const size_t i = at(x,y,z,nx,ny);
    return (a[at(xp,y,z,nx,ny)] + a[at(xm,y,z,nx,ny)] - 2.0*a[i])*invx2
         + (a[at(x,yp,z,nx,ny)] + a[at(x,ym,z,nx,ny)] - 2.0*a[i])*invy2
         + (a[at(x,y,zp,nx,ny)] + a[at(x,y,zm,nx,ny)] - 2.0*a[i])*invz2;
}

__global__ void chemical(const double* c, double* mu, size_t nx, size_t ny, size_t lz,
                         double gamma, double invx2, double invy2, double invz2) {
    size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = nx * ny * lz;
    if (q >= n) return;
    size_t x=q%nx, y=(q/nx)%ny, z=q/(nx*ny)+1, i=at(x,y,z,nx,ny);
    double v=c[i];
    mu[i] = -v + v*v*v - gamma*lap(c,x,y,z,nx,ny,lz+2,invx2,invy2,invz2);
}

__global__ void update(double* out, const double* old, const double* mu,
                       size_t nx, size_t ny, size_t lz, double factor,
                       double invx2, double invy2, double invz2) {
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*lz;
    if(q>=n) return;
    size_t x=q%nx, y=(q/nx)%ny, z=q/(nx*ny)+1, i=at(x,y,z,nx,ny);
    out[i]=old[i]+factor*lap(mu,x,y,z,nx,ny,lz+2,invx2,invy2,invz2);
}

static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2); } }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false,help=false,bad=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) help=true;
        else bad=true;
    }
    if(ny==0) ny=nx; if(nz==0) nz=nx;
    if(help||bad||!nx||!ny||!nz||iterations<0||ranks>(int)nz) {
        if(rank==0) { if(bad) printf("Invalid option\n");
            printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]); }
        MPI_Finalize(); return bad||!nx||!ny||!nz||iterations<0||ranks>(int)nz?1:0;
    }
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices));
    if(devices<=0) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank%devices));
    size_t base=nz/(size_t)ranks, rem=nz%(size_t)ranks;
    size_t lz=base+(size_t)(rank<(int)rem), z0=(size_t)rank*base+std::min((size_t)rank,rem);
    size_t plane=nx*ny, local=plane*lz, padded=plane*(lz+2);
    std::vector<double> hc(padded), hn(padded), hm(padded);
    #pragma omp parallel for schedule(static)
    for(long long q=0;q<(long long)local;++q) {
        size_t gz=z0+(size_t)q/plane, remq=(size_t)q%plane;
        size_t id=gz*plane+remq;
        hc[plane+(size_t)q]=-1.0+2.0*((((id+1)*1299709)% (nx*ny*nz))/(double)(nx*ny*nz));
    }
    double *dc=nullptr,*dn=nullptr,*dm=nullptr;
    cudaCheck(cudaMalloc(&dc,padded*sizeof(double))); cudaCheck(cudaMalloc(&dn,padded*sizeof(double))); cudaCheck(cudaMalloc(&dm,padded*sizeof(double)));
    cudaCheck(cudaMemcpy(dc,hc.data(),padded*sizeof(double),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dn,hn.data(),padded*sizeof(double),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dm,hm.data(),padded*sizeof(double),cudaMemcpyHostToDevice));
    if(z0==0) cudaCheck(cudaMemcpy(dc,hc.data()+plane,plane*sizeof(double),cudaMemcpyHostToDevice));
    if(z0+lz==nz) cudaCheck(cudaMemcpy(dc+plane*(lz+1),hc.data()+plane*lz,plane*sizeof(double),cudaMemcpyHostToDevice));
    if(rank==0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    const int prev=(rank==0?MPI_PROC_NULL:rank-1), next=(rank==ranks-1?MPI_PROC_NULL:rank+1);
    const double invx2=1.0, invy2=1.0, invz2=1.0;
    for(int t=0;t<iterations;++t) {
        cudaCheck(cudaMemcpy(hc.data()+plane,dc+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hc.data()+plane*lz,dc+plane*lz,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(hc.data()+plane, (int)plane, MPI_DOUBLE, prev, 10, hc.data()+plane*(lz+1),(int)plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(hc.data()+plane*lz,(int)plane,MPI_DOUBLE,next,11,hc.data(),(int)plane,MPI_DOUBLE,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(prev==MPI_PROC_NULL) std::copy_n(hc.data()+plane,plane,hc.data());
        if(next==MPI_PROC_NULL) std::copy_n(hc.data()+plane*lz,plane,hc.data()+plane*(lz+1));
        cudaCheck(cudaMemcpy(dc,hc.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dc+plane*(lz+1),hc.data()+plane*(lz+1),plane*sizeof(double),cudaMemcpyHostToDevice));
        size_t n=local; int blocks=(int)((n+255)/256);
        chemical<<<blocks,256>>>(dc,dm,nx,ny,lz,0.5,invx2,invy2,invz2); cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(hm.data()+plane,dm+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hm.data()+plane*lz,dm+plane*lz,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(hm.data()+plane,(int)plane,MPI_DOUBLE,prev,20,hm.data()+plane*(lz+1),(int)plane,MPI_DOUBLE,next,20,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(hm.data()+plane*lz,(int)plane,MPI_DOUBLE,next,21,hm.data(),(int)plane,MPI_DOUBLE,prev,21,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(prev==MPI_PROC_NULL) std::copy_n(hm.data()+plane,plane,hm.data());
        if(next==MPI_PROC_NULL) std::copy_n(hm.data()+plane*lz,plane,hm.data()+plane*(lz+1));
        cudaCheck(cudaMemcpy(dm,hm.data(),plane*sizeof(double),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dm+plane*(lz+1),hm.data()+plane*(lz+1),plane*sizeof(double),cudaMemcpyHostToDevice));
        update<<<blocks,256>>>(dn,dc,dm,nx,ny,lz,0.01,invx2,invy2,invz2); cudaCheck(cudaGetLastError());
        std::swap(dc,dn);
    }
    cudaCheck(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpy(hc.data()+plane,dc+plane,local*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<int> counts(ranks),displs(ranks); size_t off=0;
    for(int r=0;r<ranks;++r) { size_t rz=base+(size_t)(r<(int)rem); counts[r]=(int)(rz*plane); displs[r]=(int)off; off+=rz*plane; }
    std::vector<double> result(rank==0?nx*ny*nz:0);
    MPI_Gatherv(hc.data()+plane,(int)local,MPI_DOUBLE,result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\n",(long)ms);
        double mcups=(double)(nx*ny*nz)*(double)iterations/((double)ms/1000.0)/1e6;
        printf("Performance: %.3f MCellUpdates/s\n",mcups);
        if(printResults) print_results(result,"Concentration");
        bool valid=true;
        if(validate) { for(double v:result) if(!std::isfinite(v)||v>10.0||v< -10.0) valid=false;
            auto mm=std::minmax_element(result.begin(),result.end());
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\nValidation: %s\n",*mm.first,*mm.second,valid?"PASSED":"FAILED"); }
        cudaFree(dc); cudaFree(dn); cudaFree(dm); MPI_Finalize(); return validate&&!valid?1:0;
    }
    cudaFree(dc); cudaFree(dn); cudaFree(dm); MPI_Finalize(); return 0;
}
