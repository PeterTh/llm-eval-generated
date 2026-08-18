#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void initialize_kernel(Real* a, size_t nx, size_t ny, size_t localNz, size_t firstZ) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= localNz + 2) return;
    const size_t i = (z * ny + y) * nx + x;
    if (z == 0 || z == localNz + 1) a[i] = 0.0;
    else a[i] = (((firstZ + z - 1) * ny + y) * nx + x) % 19;
}

__global__ void stencil_kernel(const Real* in, Real* out, size_t nx, size_t ny, size_t localNz,
                               size_t firstZ, size_t zlo, size_t zhi, size_t globalNz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = zlo + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z > zhi || z > localNz) return;
    const size_t i = (z * ny + y) * nx + x;
    const size_t gz = firstZ + z - 1;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == globalNz) out[i] = in[i];
    else out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] + in[i-nx*ny] + in[i+nx*ny]) / 7.0;
}

static void usage(const char* p) { printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, results=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true; else if (!strcmp(argv[i],"-r")) results=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n",argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 || (size_t)ranks > nz) {
        if (!rank) fprintf(stderr,"Grid dimensions must be >= 3, iterations non-negative, and MPI ranks <= Z dimension.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int localRank; MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node,&localRank); int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr,"This executable requires a CUDA device.\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(localRank % devices)); MPI_Comm_free(&node);
    const size_t base=nz/ranks, extra=nz%ranks, localNz=base+(rank<(int)extra), firstZ=rank*base+std::min<size_t>(rank,extra);
    const size_t plane=nx*ny, elems=(localNz+2)*plane;
    Real *a=nullptr,*b=nullptr; CUDA_CHECK(cudaMalloc(&a,elems*sizeof(Real))); CUDA_CHECK(cudaMalloc(&b,elems*sizeof(Real)));
    dim3 block(32,4,2), initGrid((nx+31)/32,(ny+3)/4,(localNz+3)/2);
    initialize_kernel<<<initGrid,block>>>(a,nx,ny,localNz,firstZ); CUDA_CHECK(cudaGetLastError());
    initialize_kernel<<<initGrid,block>>>(b,nx,ny,localNz,firstZ); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    if (!rank) { printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\nMPI ranks: %d\n",nx,ny,nz,iterations,validate?"enabled":"disabled",ranks); printf("Running stencil computation...\n"); }
    std::vector<Real> sendLo(plane),sendHi(plane),recvLo(plane),recvHi(plane);
    const int below=rank?rank-1:MPI_PROC_NULL, above=rank+1<ranks?rank+1:MPI_PROC_NULL;
    dim3 workGrid((nx+31)/32,(ny+3)/4,(localNz+1)/2);
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::steady_clock::now();
    for (int it=0; it<iterations; ++it) {
        Real* in=(it&1)?b:a; Real* out=(it&1)?a:b;
        CUDA_CHECK(cudaMemcpy(sendLo.data(),in+plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sendHi.data(),in+localNz*plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        MPI_Request rq[4]; int n=0;
        if (below!=MPI_PROC_NULL) { MPI_Irecv(recvLo.data(),(int)plane,MPI_DOUBLE,below,11,MPI_COMM_WORLD,&rq[n++]); MPI_Isend(sendLo.data(),(int)plane,MPI_DOUBLE,below,12,MPI_COMM_WORLD,&rq[n++]); }
        if (above!=MPI_PROC_NULL) { MPI_Irecv(recvHi.data(),(int)plane,MPI_DOUBLE,above,12,MPI_COMM_WORLD,&rq[n++]); MPI_Isend(sendHi.data(),(int)plane,MPI_DOUBLE,above,11,MPI_COMM_WORLD,&rq[n++]); }
        // Independent planes run while MPI transfers the boundary halos.
        if (localNz > 2) stencil_kernel<<<workGrid,block>>>(in,out,nx,ny,localNz,firstZ,2,localNz-1,nz);
        MPI_Waitall(n,rq,MPI_STATUSES_IGNORE);
        if (below!=MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(in,recvLo.data(),plane*sizeof(Real),cudaMemcpyHostToDevice));
        if (above!=MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(in+(localNz+1)*plane,recvHi.data(),plane*sizeof(Real),cudaMemcpyHostToDevice));
        stencil_kernel<<<workGrid,block>>>(in,out,nx,ny,localNz,firstZ,1,1,nz);
        if (localNz > 1) stencil_kernel<<<workGrid,block>>>(in,out,nx,ny,localNz,firstZ,localNz,localNz,nz);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD); const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double maxSeconds=0; MPI_Reduce(&seconds,&maxSeconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    Real* finalDevice=(iterations&1)?b:a; std::vector<Real> local(localNz*plane);
    CUDA_CHECK(cudaMemcpy(local.data(),finalDevice+plane,local.size()*sizeof(Real),cudaMemcpyDeviceToHost));
    // This host pass is deliberately threaded: it overlaps no device work and keeps the
    // MPI/CUDA ranks coupled with all CPU cores for final local integrity statistics.
    Real localLo=std::numeric_limits<Real>::max(), localHi=-localLo; int localBad=0;
    #pragma omp parallel for reduction(min:localLo) reduction(max:localHi) reduction(+:localBad)
    for (size_t i=0;i<local.size();++i) { const Real v=local[i]; localBad += !std::isfinite(v); localLo=std::min(localLo,v); localHi=std::max(localHi,v); }
    std::vector<int> counts, displs; std::vector<Real> global;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); for(int r=0;r<ranks;++r) { size_t z=base+(r<(int)extra); counts[r]=(int)(z*plane); displs[r]=(int)((r*base+std::min<size_t>(r,extra))*plane); } global.resize(nx*ny*nz); }
    MPI_Gatherv(local.data(),(int)local.size(),MPI_DOUBLE,rank?nullptr:global.data(),rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int exitCode=0;
    if (!rank) {
        printf("Computation time: %.3f ms\n",maxSeconds*1000.0);
        printf("Performance: %.3f MCellUpdates/s\n",double((nx-2)*(ny-2)*(nz-2))*iterations/maxSeconds/1e6);
        if (results) print_results(global,"Grid");
        if (validate) { Real lo=std::numeric_limits<Real>::max(), hi=-lo; int bad=0;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(+:bad)
            for (size_t i=0;i<global.size();++i) { const Real v=global[i]; bad += !std::isfinite(v); lo=std::min(lo,v); hi=std::max(hi,v); }
            const bool ok=!bad && lo>=-1e6 && hi<=1e6;
            printf("Value range: [%.6f, %.6f]\n",lo,hi); printf("Validation: %s\n",ok?"PASSED":"FAILED");
            exitCode = ok ? 0 : 1;
        }
    }
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); MPI_Bcast(&exitCode,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return exitCode;
}
