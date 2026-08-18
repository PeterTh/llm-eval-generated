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
using Real = double;

__global__ void stencil_kernel(const Real* in, Real* out, size_t nx, size_t ny,
                               size_t local_z, size_t global_z0, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_z) return;
    const size_t plane = nx * ny, p = z * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
        global_z0 + z == 0 || global_z0 + z + 1 == nz) {
        out[p] = in[p];
        return;
    }
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] +
              in[p-plane] + in[p+plane]) / 7.0;
}

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static inline size_t index3(size_t x, size_t y, size_t z, size_t nx, size_t ny) { return z * nx * ny + y * nx + x; }

static void usage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -x <num> Grid X (default 128)\n  -y <num> Grid Y (default X)\n  -z <num> Grid Z (default X)\n  -i <num> Iterations (default 10)\n  -v Validate\n  -r Print results\n  -h Help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true; else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) { if(rank==0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<3 || ny<3 || nz<3 || iterations<0 || nranks>static_cast<int>(nz-2)) { if(rank==0) std::fprintf(stderr,"Invalid dimensions or too many MPI ranks\n"); MPI_Finalize(); return 1; }
    const size_t interior=nz-2, base=interior/nranks, extra=interior%nranks;
    const size_t local_z=base+(static_cast<size_t>(rank)<extra), global_z0=1+rank*base+std::min(static_cast<size_t>(rank),extra);
    const size_t plane=nx*ny, local_cells=(local_z+2)*plane;
    std::vector<Real> h0(local_cells), h1(local_cells);
    #pragma omp parallel for collapse(2)
    for (size_t z=0; z<local_z+2; ++z) for (size_t q=0; q<plane; ++q) {
        size_t gz=global_z0+z-1; h0[z*plane+q]=(index3(q%nx,q/nx,gz,nx,ny)%19)*1.0;
    }
    int local_rank=0; MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node); MPI_Comm_rank(node,&local_rank);
    int devices=0; check_cuda(cudaGetDeviceCount(&devices),"device count"); check_cuda(cudaSetDevice(local_rank%devices),"set device");
    Real *d0=nullptr,*d1=nullptr; check_cuda(cudaMalloc(&d0,local_cells*sizeof(Real)),"malloc"); check_cuda(cudaMalloc(&d1,local_cells*sizeof(Real)),"malloc");
    check_cuda(cudaMemcpy(d0,h0.data(),local_cells*sizeof(Real),cudaMemcpyHostToDevice),"initial copy");
    std::vector<Real> send_up(plane),send_down(plane),recv_up(plane),recv_down(plane);
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    dim3 block(16,8,1), grid((nx+15)/16,(ny+7)/8,local_z);
    for(int it=0;it<iterations;++it) {
        stencil_kernel<<<grid,block>>>(d0,d1,nx,ny,local_z,global_z0,nz); check_cuda(cudaGetLastError(),"kernel"); check_cuda(cudaDeviceSynchronize(),"sync");
        if (rank == 0) check_cuda(cudaMemcpy(d1,d0,plane*sizeof(Real),cudaMemcpyDeviceToDevice),"lower physical halo");
        if (rank == nranks-1) check_cuda(cudaMemcpy(d1+(local_z+1)*plane,d0+(local_z+1)*plane,plane*sizeof(Real),cudaMemcpyDeviceToDevice),"upper physical halo");
        if (nranks>1) {
            check_cuda(cudaMemcpy(send_down.data(),d1+plane,plane*sizeof(Real),cudaMemcpyDeviceToHost),"halo down");
            check_cuda(cudaMemcpy(send_up.data(),d1+local_z*plane,plane*sizeof(Real),cudaMemcpyDeviceToHost),"halo up");
            MPI_Sendrecv(send_down.data(),plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,0,recv_up.data(),plane,MPI_DOUBLE,rank<nranks-1?rank+1:MPI_PROC_NULL,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_up.data(),plane,MPI_DOUBLE,rank<nranks-1?rank+1:MPI_PROC_NULL,1,recv_down.data(),plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(rank<nranks-1) check_cuda(cudaMemcpy(d1+(local_z+1)*plane,recv_up.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo receive");
            if(rank) check_cuda(cudaMemcpy(d1,recv_down.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo receive");
        }
        std::swap(d0,d1);
    }
    MPI_Barrier(MPI_COMM_WORLD); double ms=(MPI_Wtime()-start)*1000.0;
    check_cuda(cudaMemcpy(h0.data(),d0,local_cells*sizeof(Real),cudaMemcpyDeviceToHost),"final copy");
    std::vector<Real> result; if(rank==0) result.resize(nx*ny*nz);
    std::vector<int> counts(nranks),displs(nranks); for(int r=0;r<nranks;++r){size_t l=base+(static_cast<size_t>(r)<extra);counts[r]=static_cast<int>(l*plane);displs[r]=static_cast<int>((1+r*base+std::min(static_cast<size_t>(r),extra))*plane);}
    MPI_Gatherv(h0.data()+plane,counts[rank],MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){ for(size_t z=0;z<nz;++z)for(size_t q=0;q<plane;++q)if(z==0||z+1==nz||q%nx==0||q%nx+1==nx||q/nx==0||q/nx+1==ny) result[z*plane+q]=(z*plane+q)%19; std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nComputation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iterations,ms,(double)(nx-2)*(ny-2)*(nz-2)*iterations/(ms/1000.0)/1e6); if(printResults) print_results(result,"Grid"); if(validate){bool ok=true;for(auto v:result)ok&=std::isfinite(v)&&v<=1e6&&v>=-1e6;std::printf("Validation: %s\n",ok?"PASSED":"FAILED");} }
    cudaFree(d0); cudaFree(d1); MPI_Comm_free(&node); MPI_Finalize(); return 0;
}
