#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
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
    const size_t p = nx * ny, i = z * p + y * nx + x;
    const bool boundary = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                          global_z0 + z - 1 == 0 || global_z0 + z == nz - 1;
    if (boundary) out[i] = in[i];
    else out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx] +
                   in[i-p] + in[i+p]) / 7.0;
}

static inline size_t at(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

static void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static void initialize(std::vector<Real>& a, size_t nx, size_t ny, size_t local_z, size_t z0) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_z + 2; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z ? z0 + z - 1 : z0;
                a[at(x,y,z,nx,ny)] = (at(x,y,global_z,nx,ny) % 19) * 1.0;
            }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoll(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoll(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoll(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nx<3 || ny<3 || nz<3 || iterations<0 || ranks>static_cast<int>(nz)) { if(rank==0) fprintf(stderr,"Invalid dimensions or too many MPI ranks\n"); MPI_Finalize(); return 1; }
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t local_z=base+(static_cast<size_t>(rank)<rem), z0=rank*base+std::min(static_cast<size_t>(rank),rem);
    const size_t plane=nx*ny, n=(local_z+2)*plane;
    std::vector<Real> h1(n), h2(n), gathered;
    initialize(h1,nx,ny,local_z,z0); initialize(h2,nx,ny,local_z,z0);
    int device=rank % std::max(1, [](){int n=0; cudaGetDeviceCount(&n); return n;}());
    check_cuda(cudaSetDevice(device),"cudaSetDevice");
    Real *d1=nullptr,*d2=nullptr; check_cuda(cudaMalloc(&d1,n*sizeof(Real)),"cudaMalloc"); check_cuda(cudaMalloc(&d2,n*sizeof(Real)),"cudaMalloc");
    check_cuda(cudaMemcpy(d1,h1.data(),n*sizeof(Real),cudaMemcpyHostToDevice),"initial copy");
    check_cuda(cudaMemcpy(d2,h2.data(),n*sizeof(Real),cudaMemcpyHostToDevice),"initial copy");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    const int up=rank?rank-1:MPI_PROC_NULL, down=rank+1<ranks?rank+1:MPI_PROC_NULL;
    std::vector<Real> send_up(plane),send_down(plane),recv_up(plane),recv_down(plane);
    dim3 block(8,8,4), grid((nx+7)/8,(ny+7)/8,(local_z+3)/4);
    for(int it=0; it<iterations; ++it) {
        Real *in=(it%2==0)?d1:d2, *out=(it%2==0)?d2:d1;
        check_cuda(cudaMemcpy(h1.data(),in,n*sizeof(Real),cudaMemcpyDeviceToHost),"halo copy");
        std::copy(h1.begin()+plane,h1.begin()+2*plane,send_up.begin());
        std::copy(h1.begin()+local_z*plane,h1.begin()+(local_z+1)*plane,send_down.begin());
        MPI_Sendrecv(send_up.data(),plane,MPI_DOUBLE,up,11,recv_down.data(),plane,MPI_DOUBLE,down,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_down.data(),plane,MPI_DOUBLE,down,12,recv_up.data(),plane,MPI_DOUBLE,up,12,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(up!=MPI_PROC_NULL) check_cuda(cudaMemcpy(in,recv_up.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo upload");
        if(down!=MPI_PROC_NULL) check_cuda(cudaMemcpy(in+(local_z+1)*plane,recv_down.data(),plane*sizeof(Real),cudaMemcpyHostToDevice),"halo upload");
        stencil_kernel<<<grid,block>>>(in,out,nx,ny,local_z,z0,nz); check_cuda(cudaGetLastError(),"stencil launch");
    }
    check_cuda(cudaDeviceSynchronize(),"stencil synchronize");
    Real* final_d=(iterations%2==0)?d1:d2; check_cuda(cudaMemcpy(h1.data(),final_d,n*sizeof(Real),cudaMemcpyDeviceToHost),"final copy");
    std::vector<int> counts(ranks), displs(ranks); for(int r=0;r<ranks;++r){size_t l=base+(static_cast<size_t>(r)<rem);counts[r]=static_cast<int>(l*plane);displs[r]=static_cast<int>((r*base+std::min(static_cast<size_t>(r),rem))*plane);}
    if(rank==0) gathered.resize(nx*ny*nz);
    MPI_Gatherv(h1.data()+plane,static_cast<int>(local_z*plane),MPI_DOUBLE,gathered.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now(); double sec=std::chrono::duration<double>(end-start).count();
    if(rank==0){
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nComputation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iterations,sec*1000,((double)(nx-2)*(ny-2)*(nz-2)*iterations)/sec/1e6);
        if(printResults) print_results(gathered,"Grid");
        if(validate){
            bool ok=true;
            #pragma omp parallel for reduction(&:ok)
            for(size_t i=0;i<gathered.size();++i) ok &= std::isfinite(gathered[i]) && gathered[i]<1e6 && gathered[i]>-1e6;
            printf("Validation: %s\n",ok?"PASSED":"FAILED");
            if(!ok){cudaFree(d1);cudaFree(d2);MPI_Finalize();return 1;}
        }
    }
    cudaFree(d1); cudaFree(d2); MPI_Finalize(); return 0;
}
