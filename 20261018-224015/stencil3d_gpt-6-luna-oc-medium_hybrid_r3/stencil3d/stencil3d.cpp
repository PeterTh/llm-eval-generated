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
                              size_t localNz, size_t firstGlobalZ, size_t globalNz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || lz > localNz) return;
    const size_t gz = firstGlobalZ + lz - 1;
    const size_t plane = nx * ny;
    const size_t i = lz * plane + y * nx + x;
    if (gz == 0 || gz + 1 == globalNz) { out[i] = in[i]; return; }
    out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] +
              in[i-plane] + in[i+plane]) / 7.0;
}

static void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<2 || ny<2 || nz<2 || iterations<0 || nz<static_cast<size_t>(ranks)) { if (!rank) fprintf(stderr,"Grid dimensions must be >=2 and z >= MPI ranks; iterations must be nonnegative.\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    int devices=0; checkCuda(cudaGetDeviceCount(&devices)); if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    checkCuda(cudaSetDevice(rank % devices));
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t localNz=base+(static_cast<size_t>(rank)<rem);
    const size_t first=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    const size_t plane=nx*ny, count=(localNz+2)*plane;
    std::vector<Real> a(count), b(count);
    #pragma omp parallel for
    for (long long i=0;i<static_cast<long long>(localNz*plane);++i) {
        const size_t z=static_cast<size_t>(i)/plane, remi=static_cast<size_t>(i)%plane;
        a[(z+1)*plane+remi]=((first+z)*plane+remi)%19;
    }
    Real *da=nullptr,*db=nullptr;
    checkCuda(cudaMalloc(&da,count*sizeof(Real))); checkCuda(cudaMalloc(&db,count*sizeof(Real)));
    checkCuda(cudaMemcpy(da,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(db,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    const int prev=rank?rank-1:MPI_PROC_NULL, next=rank+1<ranks?rank+1:MPI_PROC_NULL;
    if (!rank) { printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n",nx,ny,nz,iterations); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::high_resolution_clock::now();
    dim3 block(32,4,1), grid((nx-2+31)/32,(ny-2+3)/4,localNz);
    for (int iter=0;iter<iterations;++iter) {
        Real *src=(iter%2==0)?da:db, *dst=(iter%2==0)?db:da;
        checkCuda(cudaMemcpy(a.data(),src,count*sizeof(Real),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,prev,10,a.data()+(localNz+1)*plane,plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+localNz*plane,plane,MPI_DOUBLE,next,11,a.data(),plane,MPI_DOUBLE,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        checkCuda(cudaMemcpy(src,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
        stencilKernel<<<grid,block>>>(src,dst,nx,ny,localNz,first,nz); checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
    }
    Real *finalDevice=(iterations%2==0)?da:db;
    checkCuda(cudaMemcpy(a.data(),finalDevice,count*sizeof(Real),cudaMemcpyDeviceToHost));
    const auto end=std::chrono::high_resolution_clock::now();
    std::vector<int> counts(ranks), displs(ranks);
    for(int r=0;r<ranks;++r){size_t n=base+(static_cast<size_t>(r)<rem);counts[r]=static_cast<int>(n*plane);displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*plane);}
    std::vector<Real> result(rank==0?nz*plane:0);
    MPI_Gatherv(a.data()+plane,static_cast<int>(localNz*plane),MPI_DOUBLE,result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    const double localMs=std::chrono::duration<double,std::milli>(end-start).count();
    double ms=0.0;
    MPI_Reduce(&localMs,&ms,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %.0f ms\n",ms);
        const double updates=static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations;
        printf("Performance: %.3f MCellUpdates/s\n",updates/(ms/1000.0)/1e6);
        if(printResults) print_results(result,"Grid");
        if(validate){ Real lo=result[0],hi=result[0]; bool ok=true;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:ok)
            for(long long i=0;i<static_cast<long long>(result.size());++i){Real v=result[i];lo=std::min(lo,v);hi=std::max(hi,v);ok=ok&&std::isfinite(v);}
            printf("Value range: [%.6f, %.6f]\nValidation: %s\n",lo,hi,(ok&&hi<=1e6&&lo>=-1e6)?"PASSED":"FAILED");
            if(!ok||hi>1e6||lo< -1e6){cudaFree(da);cudaFree(db);MPI_Finalize();return 1;}
        }
    }
    cudaFree(da); cudaFree(db); MPI_Finalize(); return 0;
}
