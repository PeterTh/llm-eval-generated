#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
} while (0)

__device__ __forceinline__ size_t didx(size_t x, size_t y, size_t z,
                                       size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

__global__ void chemical_potential(const double* __restrict__ c,
                                   double* __restrict__ mu, size_t nx,
                                   size_t ny, size_t local_nz, double gamma,
                                   double e_AA, double e_BB, double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t plane = nx * ny;
    const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
    const size_t ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
    const size_t i = didx(x, y, z, nx, plane);
    const double v = c[i];
    const double lap = c[didx(xp,y,z,nx,plane)] + c[didx(xm,y,z,nx,plane)] - 2.0*v
                     + c[didx(x,yp,z,nx,plane)] + c[didx(x,ym,z,nx,plane)] - 2.0*v
                     + c[i + plane] + c[i - plane] - 2.0*v;
    mu[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
          + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void update_field(double* __restrict__ out,
                             const double* __restrict__ in,
                             const double* __restrict__ mu, size_t nx,
                             size_t ny, size_t local_nz, double factor) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t plane = nx * ny;
    const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
    const size_t ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
    const size_t i = didx(x, y, z, nx, plane);
    const double v = mu[i];
    const double lap = mu[didx(xp,y,z,nx,plane)] + mu[didx(xm,y,z,nx,plane)] - 2.0*v
                     + mu[didx(x,yp,z,nx,plane)] + mu[didx(x,ym,z,nx,plane)] - 2.0*v
                     + mu[i + plane] + mu[i - plane] - 2.0*v;
    out[i] = in[i] + factor * lap;
}

// Pinned staging makes this work on CUDA-aware and conventional MPI stacks alike.
static void exchange_halos(double* device, double* lower, double* upper,
                           size_t plane, size_t local_nz, int rank, int ranks,
                           int tag, cudaStream_t stream) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMemcpyAsync(lower, device + plane, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(upper, device + local_nz * plane, bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int prev = rank ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(lower, static_cast<int>(plane), MPI_DOUBLE, prev, tag,
                 upper + plane, static_cast<int>(plane), MPI_DOUBLE, next, tag,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(upper, static_cast<int>(plane), MPI_DOUBLE, next, tag + 1,
                 lower + plane, static_cast<int>(plane), MPI_DOUBLE, prev, tag + 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // MPI_PROC_NULL leaves the receive buffer untouched: duplicate the boundary.
    const double* lo_src = rank ? lower + plane : lower;
    const double* hi_src = rank + 1 < ranks ? upper + plane : upper;
    CUDA_CHECK(cudaMemcpyAsync(device, lo_src, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device + (local_nz + 1) * plane, hi_src, bytes,
                               cudaMemcpyHostToDevice, stream));
}

static bool validateResult(const std::vector<double>& c) {
    int bad = 0;
    double lo = c.empty() ? 0.0 : c[0], hi = lo;
#pragma omp parallel for reduction(+:bad) reduction(min:lo) reduction(max:hi)
    for (long long i = 0; i < static_cast<long long>(c.size()); ++i) {
        bad += !std::isfinite(c[i]); lo = std::min(lo, c[i]); hi = std::max(hi, c[i]);
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", lo, hi);
    return bad == 0 && hi <= 10.0 && lo >= -10.0;
}

static void usage(const char* p) {
    std::printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20;
    bool validate = false, printResults = false, ok = true;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else ok=false;
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    const size_t plane=nx*ny, total=plane*nz;
    if (!nx || !ny || !nz || iterations < 0 || ranks > static_cast<int>(nz) ||
        plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        total > static_cast<size_t>(std::numeric_limits<int>::max())) ok=false;
    if (!ok) { if (!rank) { std::fprintf(stderr,"Invalid arguments (also require MPI ranks <= nz and MPI-sized grid).\n"); usage(argv[0]); } MPI_Finalize(); return 1; }

    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t local_nz=base+(static_cast<size_t>(rank)<rem);
    const size_t z0=rank*base+std::min<size_t>(rank,rem);
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
    int local_rank=0; MPI_Comm_rank(local_comm,&local_rank);
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr,"No CUDA devices found.\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\n",nx,ny,nz,iterations);
        std::printf("Hybrid execution: %d MPI ranks, up to %d OpenMP threads/rank, CUDA GPUs\nValidation: %s\n",ranks,omp_get_max_threads(),validate?"enabled":"disabled");
    }
    const size_t cells=(local_nz+2)*plane, bytes=cells*sizeof(double);
    std::vector<double> host(cells,0.0);
#pragma omp parallel for collapse(2)
    for (long long zz=0; zz<static_cast<long long>(local_nz); ++zz)
        for (long long yy=0; yy<static_cast<long long>(ny); ++yy)
            for (size_t x=0; x<nx; ++x) {
                const size_t gid=(z0+zz)*plane+yy*nx+x;
                host[(zz+1)*plane+yy*nx+x]=-1.0+2.0*((((gid+1)*1299709)%total)/static_cast<double>(total));
            }
    double *cold=nullptr,*cnew=nullptr,*mu=nullptr,*lo=nullptr,*hi=nullptr;
    CUDA_CHECK(cudaMalloc(&cold,bytes)); CUDA_CHECK(cudaMalloc(&cnew,bytes)); CUDA_CHECK(cudaMalloc(&mu,bytes));
    CUDA_CHECK(cudaMallocHost(&lo,2*plane*sizeof(double))); CUDA_CHECK(cudaMallocHost(&hi,2*plane*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold,host.data(),bytes,cudaMemcpyHostToDevice));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    dim3 block(32,4,2), grid((nx+31)/32,(ny+3)/4,(local_nz+1)/2);
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    for (int t=0;t<iterations;++t) {
        exchange_halos(cold,lo,hi,plane,local_nz,rank,ranks,10,stream);
        chemical_potential<<<grid,block,0,stream>>>(cold,mu,nx,ny,local_nz,.5,-(2.0/9.0),-(2.0/9.0),2.0/9.0);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        exchange_halos(mu,lo,hi,plane,local_nz,rank,ranks,20,stream);
        update_field<<<grid,block,0,stream>>>(cnew,cold,mu,nx,ny,local_nz,.01);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
        std::swap(cold,cnew);
    }
    const double elapsed=MPI_Wtime()-start; double max_elapsed=0;
    MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(host.data()+plane,cold+plane,local_nz*plane*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<int> counts(ranks),displs(ranks); for(int r=0;r<ranks;++r){ counts[r]=static_cast<int>((base+(static_cast<size_t>(r)<rem))*plane); displs[r]=static_cast<int>((r*base+std::min<size_t>(r,rem))*plane); }
    std::vector<double> result; if(!rank) result.resize(total);
    MPI_Gatherv(host.data()+plane,counts[rank],MPI_DOUBLE,rank?nullptr:result.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int rc=0;
    if(!rank){ std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",max_elapsed*1000.0,(static_cast<double>(total)*iterations/max_elapsed)/1e6); if(printResults) print_results(result,"Concentration"); if(validate){ std::printf("Validating result...\n"); bool v=validateResult(result); std::printf("Validation: %s\n",v?"PASSED":"FAILED"); rc=v?0:1; } }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);
    cudaStreamDestroy(stream); cudaFree(cold); cudaFree(cnew); cudaFree(mu); cudaFreeHost(lo); cudaFreeHost(hi); MPI_Comm_free(&local_comm); MPI_Finalize(); return rc;
}
