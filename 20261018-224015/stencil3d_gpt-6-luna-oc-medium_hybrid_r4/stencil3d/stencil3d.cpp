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
                              size_t localNz, size_t globalZ0, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z >= localNz - 1) return;
    const size_t p = z * nx * ny + y * nx + x;
    const size_t globalZ = globalZ0 + z - 1;
    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || globalZ == 0 || globalZ >= nz - 1) {
        out[p] = in[p];
        return;
    }
    out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] +
              in[p-nx*ny] + in[p+nx*ny]) / 7.0;
}

static void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void printUsage(const char* p) {
    printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t nx=128, ny=0, nz=0;
    int iterations=10;
    bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate=true;
        else if (!strcmp(argv[i], "-r")) printResults=true;
        else if (!strcmp(argv[i], "-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) printUsage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<3 || ny<3 || nz<3 || iterations<0 || static_cast<size_t>(nranks)>nz-2) {
        if(rank==0) fprintf(stderr,"Grid dimensions must be >=3 and provide at least one interior z-plane per MPI rank.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Contiguous z-slab decomposition; each rank owns interior planes and keeps two halos.
    const size_t interior=nz-2, base=interior/nranks, rem=interior%nranks;
    const size_t owned=base+(static_cast<size_t>(rank)<rem), first=1+rank*base+std::min<size_t>(rank,rem);
    const size_t localNz=owned+2, plane=nx*ny, count=localNz*plane;
    int deviceCount=0; checkCuda(cudaGetDeviceCount(&deviceCount));
    if(deviceCount<=0) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    checkCuda(cudaSetDevice(rank%deviceCount));
    Real *a=nullptr,*b=nullptr;
    checkCuda(cudaMallocManaged(&a,count*sizeof(Real))); checkCuda(cudaMallocManaged(&b,count*sizeof(Real)));
    #pragma omp parallel for
    for (long long i=0; i<static_cast<long long>(count); ++i) {
        const size_t z=static_cast<size_t>(i)/plane, inPlane=static_cast<size_t>(i)%plane;
        const size_t gz=(z==0 ? first-1 : (z==localNz-1 ? first+owned : first+z-1));
        const size_t globalIndex=gz*plane+inPlane;
        a[i]=static_cast<Real>(globalIndex%19); b[i]=a[i];
    }
    // Host staging keeps halo exchange portable to MPI implementations without GPU-aware support.
    std::vector<Real> sendLo(plane),sendHi(plane),recvLo(plane),recvHi(plane);
    if(rank==0) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); printf("Initializing grid...\nRunning stencil computation...\n"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start=std::chrono::high_resolution_clock::now();
    for(int it=0; it<iterations; ++it) {
        Real* in=(it%2==0)?a:b; Real* out=(it%2==0)?b:a;
        checkCuda(cudaDeviceSynchronize());
        #pragma omp parallel for
        for(long long j=0;j<static_cast<long long>(plane);++j) { sendLo[j]=in[plane+j]; sendHi[j]=in[owned*plane+j]; }
        MPI_Sendrecv(sendLo.data(),plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,0,
                     recvHi.data(),plane,MPI_DOUBLE,rank+1<nranks?rank+1:MPI_PROC_NULL,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendHi.data(),plane,MPI_DOUBLE,rank+1<nranks?rank+1:MPI_PROC_NULL,1,
                     recvLo.data(),plane,MPI_DOUBLE,rank?rank-1:MPI_PROC_NULL,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank>0) std::copy(recvLo.begin(),recvLo.end(),in);
        if(rank+1<nranks) std::copy(recvHi.begin(),recvHi.end(),in+(owned+1)*plane);
        dim3 block(32,4,1), grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,(owned+block.z-1)/block.z);
        stencilKernel<<<grid,block>>>(in,out,nx,ny,localNz,first,nz);
        checkCuda(cudaGetLastError()); checkCuda(cudaDeviceSynchronize());
    }
    const auto end=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double>(end-start).count(), maxElapsed;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<Real> local(owned*plane);
    Real* final=(iterations%2==0)?a:b;
    std::copy(final+plane,final+(owned+1)*plane,local.begin());
    std::vector<int> counts,displs; std::vector<Real> full;
    if(rank==0) { counts.resize(nranks); displs.resize(nranks); size_t off=0; for(int r=0;r<nranks;++r) { size_t n=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>(off); off+=n*plane; } full.resize(nx*ny*nz); }
    std::vector<Real> interiorAll(rank==0?(nz-2)*plane:0);
    MPI_Gatherv(local.data(),static_cast<int>(local.size()),MPI_DOUBLE,rank==0?interiorAll.data():nullptr,rank==0?counts.data():nullptr,rank==0?displs.data():nullptr,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        #pragma omp parallel for
        for(long long j=0;j<static_cast<long long>(plane);++j) { full[j]=static_cast<Real>(static_cast<size_t>(j)%19); full[(nz-1)*plane+j]=static_cast<Real>(((nz-1)*plane+j)%19); }
        std::copy(interiorAll.begin(),interiorAll.end(),full.begin()+plane);
        printf("Computation time: %ld ms\n",static_cast<long>(maxElapsed*1000));
        double updates=static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations;
        printf("Performance: %.3f MCellUpdates/s\n",updates/maxElapsed/1e6);
        if(printResults) print_results(full,"Grid");
        if(validate) { bool ok=true; Real mn=full[0],mx=full[0]; for(Real v:full) { if(!std::isfinite(v)) ok=false; mn=std::min(mn,v); mx=std::max(mx,v); } printf("Value range: [%.6f, %.6f]\n",mn,mx); if(mx>1e6||mn< -1e6) ok=false; printf("Validation: %s\n",ok?"PASSED":"FAILED"); if(!ok) iterations=-1; }
    }
    checkCuda(cudaFree(a)); checkCuda(cudaFree(b));
    MPI_Bcast(&iterations,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return iterations<0?1:0;
}
