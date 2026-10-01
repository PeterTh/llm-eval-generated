#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
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
    if (x >= nx || y >= ny || z > localNz) return;
    const size_t gz = globalZ0 + z - 1;
    const size_t i = (z * ny + y) * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == nz) {
        out[i] = in[i];
    } else {
        out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] +
                  in[i-nx*ny] + in[i+nx*ny]) / 7.0;
    }
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 128)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if ((!strcmp(argv[i],"-x") || !strcmp(argv[i],"-y") || !strcmp(argv[i],"-z") || !strcmp(argv[i],"-i")) && i+1<argc) {
            const char* opt=argv[i++]; const long v=atol(argv[i]);
            if (!strcmp(opt,"-x")) nx=v>0?(size_t)v:0; else if (!strcmp(opt,"-y")) ny=v>0?(size_t)v:0;
            else if (!strcmp(opt,"-z")) nz=v>0?(size_t)v:0; else iterations=(int)v;
        } else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<3 || ny<3 || nz<3 || iterations<0 || nx>INT_MAX || ny>INT_MAX || nz>INT_MAX) {
        if(rank==0) fprintf(stderr,"Grid dimensions must be at least 3 and fit MPI counts; iterations must be nonnegative.\n");
        MPI_Finalize(); return 1;
    }
    if ((size_t)ranks>nz) {
        if(rank==0) fprintf(stderr,"MPI rank count cannot exceed the Z dimension.\n");
        MPI_Finalize(); return 1;
    }
    const size_t base=nz/(size_t)ranks, rem=nz%(size_t)ranks;
    const size_t owned=base+((size_t)rank<rem ? 1 : 0);
    const size_t z0=(size_t)rank*base+std::min((size_t)rank,rem);
    const size_t plane=nx*ny, localNz=owned+2, localSize=localNz*plane;
    if (plane>(size_t)INT_MAX || nx*ny*nz>(size_t)INT_MAX) {
        if(rank==0) fprintf(stderr,"Grid exceeds MPI count limits.\n");
        MPI_Finalize(); return 1;
    }
    int deviceCount=0; cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount<1) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank%deviceCount));
    std::vector<Real> hostA(localSize), hostB(localSize);
    #pragma omp parallel for schedule(static)
    for (long long q=0; q<(long long)localSize; ++q) {
        const size_t lz=(size_t)q/plane, remq=(size_t)q%plane;
        const size_t gy=z0+lz-1;
        hostA[(size_t)q]=(gy<nz && lz>0 && lz<=owned) ? (double)(((gy*plane+remq)%19)) : 0.0;
    }
    // Edge ranks own the physical end planes; initialize their ghost planes from those fixed boundaries.
    if (z0==0) std::copy(hostA.begin()+plane,hostA.begin()+2*plane,hostA.begin());
    if (z0+owned==nz) std::copy(hostA.begin()+owned*plane,hostA.begin()+(owned+1)*plane,hostA.begin()+(owned+1)*plane);
    Real *dA=nullptr,*dB=nullptr; cudaCheck(cudaMalloc(&dA,localSize*sizeof(Real))); cudaCheck(cudaMalloc(&dB,localSize*sizeof(Real)));
    cudaCheck(cudaMemcpy(dA,hostA.data(),localSize*sizeof(Real),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB,hostA.data(),localSize*sizeof(Real),cudaMemcpyHostToDevice));
    const int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<ranks?rank+1:MPI_PROC_NULL;
    if(rank==0) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::high_resolution_clock::now();
    dim3 block(32,4,1), grid((unsigned)((nx+31)/32),(unsigned)((ny+3)/4),(unsigned)((owned+block.z-1)/block.z));
    Real* in=dA; Real* out=dB;
    for(int iter=0;iter<iterations;++iter) {
        cudaCheck(cudaMemcpy(hostA.data()+plane,in+plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(hostA.data()+(owned+1)*plane,in+owned*plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(hostA.data()+plane,(int)plane,MPI_DOUBLE,prev,0,hostA.data()+(owned+1)*plane,(int)plane,MPI_DOUBLE,next,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(hostA.data()+owned*plane,(int)plane,MPI_DOUBLE,next,1,hostA.data(),(int)plane,MPI_DOUBLE,prev,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaCheck(cudaMemcpy(in,hostA.data(),localSize*sizeof(Real),cudaMemcpyHostToDevice));
        if(owned) stencilKernel<<<grid,block>>>(in,out,nx,ny,owned,z0,nz);
        cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize()); std::swap(in,out);
    }
    cudaCheck(cudaMemcpy(hostA.data()+plane,in+plane,owned*plane*sizeof(Real),cudaMemcpyDeviceToHost));
    const auto end=std::chrono::high_resolution_clock::now();
    const double secs=std::chrono::duration<double>(end-start).count();
    std::vector<Real> result;
    if(rank==0) result.resize(nx*ny*nz);
    std::vector<int> counts(ranks),displs(ranks); size_t offset=0;
    for(int r=0;r<ranks;++r) { size_t n=base+(size_t)r<rem?base+1:base; counts[r]=(int)(n*plane); displs[r]=(int)(offset*plane); offset+=n; }
    MPI_Gatherv(hostA.data()+plane,(int)(owned*plane),MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        const long long updates=(long long)(nx-2)*(ny-2)*(nz-2)*iterations;
        printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",(long)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(),secs>0?updates/secs/1e6:0.0);
        if(printResults) print_results(result,"Grid");
        if(validate) { bool valid=true; Real mn=result[0],mx=result[0]; for(Real v:result) { if(!std::isfinite(v)) valid=false; mn=std::min(mn,v); mx=std::max(mx,v); }
            printf("Validating result...\nValue range: [%.6f, %.6f]\n",mn,mx); if(mx>1e6||mn< -1e6) valid=false; printf("Validation: %s\n",valid?"PASSED":"FAILED");
            cudaFree(dA); cudaFree(dB); MPI_Finalize(); return valid?0:1;
        }
    }
    cudaFree(dA); cudaFree(dB); MPI_Finalize(); return 0;
}
