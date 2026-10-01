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

__global__ void stencilKernel(const Real* in, Real* out, size_t nx, size_t ny,
                              size_t localNz, size_t zStart, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z >= localNz - 1) return;
    const size_t i = z * nx * ny + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || zStart + z == 0 || zStart + z + 1 == nz) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] + in[i-nx*ny] + in[i+nx*ny]) / 7.0;
    }
}

static void cudaCheck(cudaError_t e) { if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } }

bool validateResult(const std::vector<Real>& g) {
    Real lo = g.empty() ? 0 : g[0], hi = lo;
    for (Real v : g) { if (!std::isfinite(v)) return false; lo = std::min(lo,v); hi = std::max(hi,v); }
    printf("Value range: [%.6f, %.6f]\n", lo, hi);
    return hi <= 1e6 && lo >= -1e6;
}

void printUsage(const char* p) { printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr,"No CUDA devices found\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank % devices));
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nz < 3 || nx < 3 || ny < 3 || nranks > (int)(nz-2)) { if (!rank) fprintf(stderr,"Grid dimensions must be >=3 and ranks <= nz-2\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    // Partition interior planes evenly; rank zero owns global plane zero and the last rank owns nz-1.
    const size_t interior=nz-2, base=interior/nranks, rem=interior%nranks;
    const size_t owned=base+(size_t)(rank<(int)rem);
    const size_t first=1+(size_t)rank*base+std::min((size_t)rank,rem);
    const bool low=rank==0, high=rank==nranks-1;
    const size_t localNz=owned+(low?1:0)+(high?1:0);
    const size_t zStart=first-(low?1:0), plane=nx*ny, count=localNz*plane;
    std::vector<Real> a(count), b(count);
    #pragma omp parallel for schedule(static)
    for (long long q=0;q<(long long)count;++q) {
        size_t z=(size_t)q/plane, y=((size_t)q%plane)/nx, x=(size_t)q%nx;
        size_t global=zStart+z; a[q]=(global*plane+y*nx+x)%19;
    }
    Real *da=nullptr,*db=nullptr; cudaCheck(cudaMalloc(&da,count*sizeof(Real))); cudaCheck(cudaMalloc(&db,count*sizeof(Real)));
    cudaCheck(cudaMemcpy(da,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    int left=rank?rank-1:MPI_PROC_NULL, right=rank+1<nranks?rank+1:MPI_PROC_NULL;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iterations;++it) {
        // Exchange the current owned edge planes into ghost planes before launching the stencil.
        MPI_Sendrecv(a.data()+(low?1:0)*plane,plane,MPI_DOUBLE,left,10,
                     a.data()+(localNz-1-(high?1:0))*plane,plane,MPI_DOUBLE,right,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+(localNz-2-(high?1:0))*plane,plane,MPI_DOUBLE,right,11,
                     a.data()+(low?0:0)*plane,plane,MPI_DOUBLE,left,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(da,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
        dim3 block(32,4,1), grid((nx+31)/32,(ny+3)/4,(localNz-2+0));
        stencilKernel<<<grid,block>>>(da,db,nx,ny,localNz,zStart,nz); cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize()); cudaCheck(cudaMemcpy(b.data(),db,count*sizeof(Real),cudaMemcpyDeviceToHost));
        a.swap(b);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double>(end-start).count(), maxElapsed=0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    // Gather the non-overlapping global planes to rank zero.
    int sendCount=(int)(owned*plane); std::vector<int> counts,displs; std::vector<Real> global;
    if (!rank) { counts.resize(nranks); displs.resize(nranks); global.resize(nx*ny*nz); }
    MPI_Gather(&sendCount,1,MPI_INT,counts.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    if (!rank) { for(int r=0;r<nranks;++r){ size_t z0=1+(size_t)r*base+std::min((size_t)r,rem); displs[r]=(int)(z0*plane); } }
    MPI_Gatherv(a.data()+(low?1:0)*plane,sendCount,MPI_DOUBLE,global.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (!rank) {
        for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) { global[y*nx+x]=(y*nx+x)%19; global[(nz-1)*plane+y*nx+x]=((nz-1)*plane+y*nx+x)%19; }
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n",nx,ny,nz,iterations);
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",maxElapsed*1000,((double)(nx-2)*(ny-2)*(nz-2)*iterations)/maxElapsed/1e6);
        if(printResults) print_results(global,"Grid");
        if(validate) { bool ok=validateResult(global); printf("Validation: %s\n",ok?"PASSED":"FAILED"); if(!ok) MPI_Abort(MPI_COMM_WORLD,1); }
    }
    cudaFree(da); cudaFree(db); MPI_Finalize(); return 0;
}
