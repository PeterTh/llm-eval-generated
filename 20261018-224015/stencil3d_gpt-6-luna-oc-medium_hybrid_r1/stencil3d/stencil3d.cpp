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

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error on rank %d: %s\n", rank, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void stencilKernel(const Real* in, Real* out, size_t nx, size_t ny,
                              size_t localNz, size_t globalNz, size_t zStart) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z >= localNz - 1) return;
    const size_t plane = nx * ny, i = z * plane + y * nx + x;
    const size_t globalZ = zStart + z - 1;
    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || globalZ == 0 || globalZ == globalNz-1) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] + in[i-plane] + in[i+plane]) / 7.0;
    }
}

static void checkMpi(int rc, int rank) {
    if (rc != MPI_SUCCESS) { fprintf(stderr, "MPI failure on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 3); }
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<3 || ny<3 || nz<3 || iterations<0 || nz < static_cast<size_t>(ranks)) {
        if(rank==0) fprintf(stderr,"Grid dimensions must be >= 3 and Z must be >= MPI rank count; iterations must be nonnegative\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    const size_t base=nz/ranks, extra=nz%ranks;
    const size_t owned=base+(static_cast<size_t>(rank)<extra), zStart=rank*base+std::min(static_cast<size_t>(rank),extra);
    const size_t localNz=owned+2, plane=nx*ny, count=localNz*plane;
    std::vector<Real> hostA(count), hostB(count);
    #pragma omp parallel for schedule(static)
    for (long long j=0;j<static_cast<long long>(count);++j) {
        size_t lz=static_cast<size_t>(j)/plane, rem=static_cast<size_t>(j)%plane;
        size_t gz=zStart+lz-1;
        if(lz==0) gz=(zStart==0?0:zStart-1);
        if(lz==localNz-1) gz=zStart+owned;
        hostA[j]=((gz*plane+rem)%19)*1.0;
    }
    Real *dA=nullptr,*dB=nullptr;
    CUDA_CHECK(cudaMalloc(&dA,count*sizeof(Real))); CUDA_CHECK(cudaMalloc(&dB,count*sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dA,hostA.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hostA.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    if(rank==0) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n",nx,ny,nz,iterations); }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start=std::chrono::high_resolution_clock::now();
    Real *in=dA,*out=dB;
    dim3 block(32,4,1), grid((nx+31)/32,(ny+3)/4,(owned+block.z-1)/block.z);
    for(int iter=0;iter<iterations;++iter) {
        CUDA_CHECK(cudaMemcpy(hostA.data(),in,count*sizeof(Real),cudaMemcpyDeviceToHost));
        MPI_Request req[4]; int nreq=0;
        const int prev=rank==0?MPI_PROC_NULL:rank-1, next=rank==ranks-1?MPI_PROC_NULL:rank+1;
        checkMpi(MPI_Irecv(hostA.data(),plane,MPI_DOUBLE,prev,10,MPI_COMM_WORLD,&req[nreq++]),rank);
        checkMpi(MPI_Irecv(hostA.data()+(localNz-1)*plane,plane,MPI_DOUBLE,next,11,MPI_COMM_WORLD,&req[nreq++]),rank);
        checkMpi(MPI_Isend(hostA.data()+plane,plane,MPI_DOUBLE,prev,11,MPI_COMM_WORLD,&req[nreq++]),rank);
        checkMpi(MPI_Isend(hostA.data()+owned*plane,plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,&req[nreq++]),rank);
        MPI_Waitall(nreq,req,MPI_STATUSES_IGNORE);
        CUDA_CHECK(cudaMemcpy(in,hostA.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
        stencilKernel<<<grid,block>>>(in,out,nx,ny,localNz,nz,zStart);
        CUDA_CHECK(cudaGetLastError());
        std::swap(in,out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end=std::chrono::high_resolution_clock::now();
    long long localMs=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(), ms=0;
    MPI_Reduce(&localMs,&ms,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(hostA.data(),in,count*sizeof(Real),cudaMemcpyDeviceToHost));
    std::vector<Real> local(owned*plane);
    #pragma omp parallel for schedule(static)
    for(long long j=0;j<static_cast<long long>(local.size());++j) local[j]=hostA[plane+static_cast<size_t>(j)];
    std::vector<int> counts(ranks),displs(ranks);
    for(int r=0;r<ranks;++r) { size_t n=(nz/ranks+(static_cast<size_t>(r)<nz%ranks))*plane, off=(r*(nz/ranks)+std::min(static_cast<size_t>(r),nz%ranks))*plane; counts[r]=static_cast<int>(n); displs[r]=static_cast<int>(off); }
    std::vector<Real> finalGrid(rank==0?nx*ny*nz:0);
    MPI_Gatherv(local.data(),static_cast<int>(local.size()),MPI_DOUBLE,finalGrid.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        printf("Computation time: %lld ms\n",ms);
        double updates=static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations;
        printf("Performance: %.3f MCellUpdates/s\n",updates/(ms/1000.0)/1e6);
        if(printResults) print_results(finalGrid,"Grid");
        if(validate) {
            bool valid=true; Real mn=finalGrid[0],mx=finalGrid[0];
            #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(&:valid)
            for(long long j=0;j<static_cast<long long>(finalGrid.size());++j) { Real v=finalGrid[j]; if(!std::isfinite(v)) valid=false; mn=std::min(mn,v); mx=std::max(mx,v); }
            printf("Value range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,(valid&&mx<=1e6&&mn>=-1e6)?"PASSED":"FAILED");
            if(!valid||mx>1e6||mn< -1e6) { cudaFree(dA); cudaFree(dB); MPI_Finalize(); return 1; }
        }
    }
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    MPI_Finalize(); return 0;
}
