#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include "../common/results_output.hpp"

using Real = double;

__global__ void stencil_kernel(const Real* in, Real* out, size_t nx, size_t ny,
                               size_t local_nz, size_t global_z0, size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lz = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lz >= local_nz) return;
    const size_t p = lz * nx * ny + y * nx + x;
    const bool boundary = x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                          global_z0 + lz == 0 || global_z0 + lz + 1 == nz;
    if (boundary) out[p] = in[p];
    else out[p] = (in[p] + in[p-1] + in[p+1] + in[p-nx] + in[p+nx] +
                   in[p-nx*ny] + in[p+nx*ny]) / 7.0;
}

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) { return z * nx * ny + y * nx + x; }

static void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t local_nz, size_t global_z0) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
        grid[idx3(x,y,z,nx,ny)] = static_cast<Real>(idx3(x,y,global_z0+z,nx,ny) % 19);
}

static bool validateResult(const std::vector<Real>& grid) {
    Real minVal = grid[0], maxVal = grid[0];
    bool ok = true;
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&:ok)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]); maxVal = std::max(maxVal, grid[i]);
        if (!std::isfinite(grid[i]) || grid[i] > 1e6 || grid[i] < -1e6) ok = false;
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    return ok;
}

static void usage(const char* n) { printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n", n); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    int device_count = 0; cuda_check(cudaGetDeviceCount(&device_count), "device count");
    if (device_count == 0) { if (rank == 0) fprintf(stderr, "No CUDA devices available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cuda_check(cudaSetDevice(local_rank % device_count), "select device");
    size_t nx=128, ny=0, nz=0; int iterations=10; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) { if (!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]); else if (!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]); else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true; else if (!strcmp(argv[i],"-h")) { if(rank==0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; } }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (nz < 3 || nx < 3 || ny < 3 || nranks > static_cast<int>(nz-2)) { if(rank==0) fprintf(stderr,"Grid too small for MPI decomposition\n"); MPI_Finalize(); return 1; }
    const size_t base=(nz-2)/nranks, rem=(nz-2)%nranks;
    const size_t owned=base+(rank<static_cast<int>(rem));
    const size_t z0=1+rank*base+std::min(rank,static_cast<int>(rem));
    const size_t local_nz=owned+2, plane=nx*ny, local_size=local_nz*plane;
    std::vector<Real> a(local_size), b(local_size); initializeGrid(a,nx,ny,local_nz,z0-1);
    Real *da=nullptr,*db=nullptr; cuda_check(cudaMalloc(&da,local_size*sizeof(Real)),"malloc"); cuda_check(cudaMalloc(&db,local_size*sizeof(Real)),"malloc");
    cuda_check(cudaMemcpy(da,a.data(),local_size*sizeof(Real),cudaMemcpyHostToDevice),"upload");
    int up=rank?rank-1:MPI_PROC_NULL, down=rank+1<nranks?rank+1:MPI_PROC_NULL;
    dim3 block(32,4,2), grid((nx+31)/32,(ny+3)/4,(local_nz+1)/2);
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iterations;++it) {
        cuda_check(cudaMemcpy(a.data(),da,local_size*sizeof(Real),cudaMemcpyDeviceToHost),"download halos");
        MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,up,10,a.data(),plane,MPI_DOUBLE,up,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+(local_nz-2)*plane,plane,MPI_DOUBLE,down,11,a.data()+(local_nz-1)*plane,plane,MPI_DOUBLE,down,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cuda_check(cudaMemcpy(da,a.data(),local_size*sizeof(Real),cudaMemcpyHostToDevice),"upload halos");
        stencil_kernel<<<grid,block>>>(da,db,nx,ny,local_nz,z0-1,nz); cuda_check(cudaGetLastError(),"kernel"); cuda_check(cudaDeviceSynchronize(),"sync"); std::swap(da,db);
    }
    cuda_check(cudaMemcpy(a.data(),(iterations%2)?db:da,local_size*sizeof(Real),cudaMemcpyDeviceToHost),"download result"); cudaFree(da); cudaFree(db);
    auto end=std::chrono::high_resolution_clock::now();
    std::vector<Real> result; std::vector<int> counts, displs; if(rank==0){ result.resize(nx*ny*nz); counts.resize(nranks); displs.resize(nranks); for(int r=0;r<nranks;++r){size_t o=base+(r<(int)rem); size_t s=1+r*base+std::min(r,(int)rem); counts[r]=o*plane; displs[r]=s*plane;} }
    MPI_Gatherv(a.data()+plane,owned*plane,MPI_DOUBLE,rank==0?result.data():nullptr,rank==0?counts.data():nullptr,rank==0?displs.data():nullptr,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0){ auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nComputation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n",nx,ny,nz,iterations,ms,(double)(nx-2)*(ny-2)*(nz-2)*iterations/(ms/1000.0)/1e6); if(printResults) print_results(result,"Grid"); if(validate) { printf("Validation: %s\n",validateResult(result)?"PASSED":"FAILED"); } }
    MPI_Comm_free(&local_comm); MPI_Finalize(); return 0;
}
