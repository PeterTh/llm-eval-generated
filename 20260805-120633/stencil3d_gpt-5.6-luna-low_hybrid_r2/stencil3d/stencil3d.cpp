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

using Real = double;
__host__ __device__ inline constexpr size_t at(size_t x,size_t y,size_t z,size_t nx,size_t ny) { return z*nx*ny+y*nx+x; }

__global__ void stencil_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                               size_t nx, size_t ny, size_t local_nz, size_t global_z0,
                               size_t global_nz) {
    const size_t x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    const size_t z=blockIdx.z*blockDim.z+threadIdx.z+1;
    if (x>=nx || y>=ny || z>local_nz) return;
    const size_t p=at(x,y,z,nx,ny);
    if (x==0 || x+1==nx || y==0 || y+1==ny || global_z0+z==0 || global_z0+z+1==global_nz)
        out[p]=in[p];
    else
        out[p]=(in[p]+in[p-1]+in[p+1]+in[p-nx]+in[p+nx]+in[p-nx*ny]+in[p+nx*ny])/7.0;
}

static void check_cuda(cudaError_t e, const char* where) {
    if (e!=cudaSuccess) { fprintf(stderr,"CUDA error at %s: %s\n",where,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); }
}

static void initialize(std::vector<Real>& a,size_t nx,size_t ny,size_t nz,size_t z0) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z=0;z<nz;z++) for (size_t y=0;y<ny;y++) for (size_t x=0;x<nx;x++)
        a[at(x,y,z,nx,ny)]=(at(x,y,z+z0,nx,ny)%19)*1.0;
}

static bool valid(const std::vector<Real>& a) {
    Real lo=a[0], hi=a[0];
    for (Real v:a) { if (!std::isfinite(v)) return false; lo=std::min(lo,v); hi=std::max(hi,v); }
    printf("Value range: [%.6f, %.6f]\n",lo,hi); return hi<=1e6 && lo>=-1e6;
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t nx=128,ny=0,nz=0; int iterations=10; bool validate_result=false, print_result=false;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate_result=true; else if(!strcmp(argv[i],"-r")) print_result=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1; }
    }
    if(!ny) ny=nx; if(!nz) nz=nx; if(nx<3||ny<3||nz<3||iterations<0||size>int(nz-2)) { if(rank==0) fprintf(stderr,"Invalid grid or too many MPI ranks\n"); MPI_Finalize(); return 1; }
    const size_t base=nz/size, rem=nz%size, z0=rank*base+std::min<size_t>(rank,rem), owned=base+(size_t(rank)<rem);
    const size_t plane=nx*ny, local_nz=owned+2, bytes=local_nz*plane;
    std::vector<Real> h1(bytes),h2(bytes), send_top(plane),send_bottom(plane),recv_top(plane),recv_bottom(plane);
    initialize(h1,nx,ny,local_nz,z0-1); initialize(h2,nx,ny,local_nz,z0-1);
    check_cuda(cudaSetDevice(rank%std::max(1,[](){int n=0; cudaGetDeviceCount(&n); return n;}())),"cudaSetDevice");
    Real *d1=nullptr,*d2=nullptr; check_cuda(cudaMalloc(&d1,bytes*sizeof(Real)),"cudaMalloc"); check_cuda(cudaMalloc(&d2,bytes*sizeof(Real)),"cudaMalloc");
    check_cuda(cudaMemcpy(d1,h1.data(),bytes*sizeof(Real),cudaMemcpyHostToDevice),"copy"); check_cuda(cudaMemcpy(d2,h2.data(),bytes*sizeof(Real),cudaMemcpyHostToDevice),"copy");
    if(rank==0) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nMPI ranks: %d, OpenMP threads: %d, CUDA: enabled\n",nx,ny,nz,iterations,size,omp_get_max_threads()); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iterations;it++) {
        Real *in=(it%2==0)?d1:d2, *out=(it%2==0)?d2:d1; auto& hin=(it%2==0)?h1:h2;
        check_cuda(cudaMemcpy(hin.data()+plane, in+plane, plane*sizeof(Real), cudaMemcpyDeviceToHost),"halo copy");
        check_cuda(cudaMemcpy(hin.data()+owned*plane, in+owned*plane, plane*sizeof(Real), cudaMemcpyDeviceToHost),"halo copy");
        MPI_Sendrecv(hin.data()+plane,plane,MPI_DOUBLE,rank-1,7,recv_bottom.data(),plane,MPI_DOUBLE,rank+1,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(hin.data()+owned*plane,plane,MPI_DOUBLE,rank+1,8,recv_top.data(),plane,MPI_DOUBLE,rank-1,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank>0) check_cuda(cudaMemcpy(hin.data(),recv_bottom.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo upload");
        if(rank+1<size) check_cuda(cudaMemcpy(hin.data()+(owned+1)*plane,recv_top.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo upload");
        check_cuda(cudaMemcpy(in,hin.data(),bytes*sizeof(Real),cudaMemcpyHostToDevice),"upload");
        dim3 block(16,8,1), grid((nx+15)/16,(ny+7)/8,owned); stencil_kernel<<<grid,block>>>(in,out,nx,ny,owned,z0,nz); check_cuda(cudaGetLastError(),"kernel");
    }
    check_cuda(cudaDeviceSynchronize(),"synchronize"); auto end=std::chrono::high_resolution_clock::now();
    auto& final_local=(iterations%2==0)?h1:h2; Real* df=(iterations%2==0)?d1:d2; check_cuda(cudaMemcpy(final_local.data(),df,bytes*sizeof(Real),cudaMemcpyDeviceToHost),"download");
    std::vector<Real> global; if(rank==0) global.resize(nx*ny*nz); std::vector<int> counts(size),displs(size);
    for(int r=0;r<size;r++){size_t ro=r*base+std::min<size_t>(r,rem), rn=base+(size_t(r)<rem); counts[r]=int(rn*plane); displs[r]=int(ro*plane);}
    MPI_Gatherv(final_local.data()+plane,int(owned*plane),MPI_DOUBLE,rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int validation_ok=1;
    if(rank==0){ auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",ms,double((nx-2)*(ny-2)*(nz-2))*iterations/(ms/1000.0)/1e6); if(print_result) print_results(global,"Grid"); if(validate_result) { validation_ok=valid(global); printf("Validation: %s\n",validation_ok?"PASSED":"FAILED"); } }
    MPI_Bcast(&validation_ok,1,MPI_INT,0,MPI_COMM_WORLD);
    cudaFree(d1); cudaFree(d2); MPI_Finalize(); return validation_ok?0:1;
}
