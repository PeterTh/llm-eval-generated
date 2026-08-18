#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
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

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__device__ __forceinline__ size_t at(size_t x, size_t y, size_t z, size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

// z is local, with owned planes [1,lz] and halo planes 0 and lz+1.
__device__ __forceinline__ double lap(const double* __restrict__ a, size_t x, size_t y,
                                      size_t z, size_t nx, size_t ny, size_t plane) {
    const size_t xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
    const size_t ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
    const size_t i = at(x, y, z, nx, plane);
    return a[at(xm,y,z,nx,plane)] + a[at(xp,y,z,nx,plane)]
         + a[at(x,ym,z,nx,plane)] + a[at(x,yp,z,nx,plane)]
         + a[i-plane] + a[i+plane] - 6.0 * a[i];
}

__global__ void chemical(const double* __restrict__ c, double* __restrict__ mu,
                         size_t nx, size_t ny, size_t z0, size_t z1,
                         double gamma, double eaa, double ebb, double eab) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;
    const size_t plane = nx * ny, i = at(x,y,z,nx,plane);
    const double v = c[i];
    mu[i] = 4.5*((v+1.0)*eaa + (v-1.0)*ebb - 2.0*v*eab)
          + 3.0*v + v*v*v - gamma*lap(c,x,y,z,nx,ny,plane);
}

__global__ void update(const double* __restrict__ c, const double* __restrict__ mu,
                       double* __restrict__ out, size_t nx, size_t ny,
                       size_t z0, size_t z1, double scale) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = z0 + blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= z1) return;
    const size_t plane = nx*ny, i = at(x,y,z,nx,plane);
    out[i] = c[i] + scale*lap(mu,x,y,z,nx,ny,plane);
}

static void launch_chemical(const double* c, double* mu, size_t nx, size_t ny,
                            size_t z0, size_t z1, double gamma,
                            double eaa, double ebb, double eab, cudaStream_t stream) {
    if (z0 >= z1) return;
    const dim3 b(32,4,2), g((nx+b.x-1)/b.x, (ny+b.y-1)/b.y, (z1-z0+b.z-1)/b.z);
    chemical<<<g,b,0,stream>>>(c,mu,nx,ny,z0,z1,gamma,eaa,ebb,eab);
}

static void launch_update(const double* c, const double* mu, double* out,
                          size_t nx, size_t ny, size_t z0, size_t z1,
                          double scale, cudaStream_t stream) {
    if (z0 >= z1) return;
    const dim3 b(32,4,2), g((nx+b.x-1)/b.x, (ny+b.y-1)/b.y, (z1-z0+b.z-1)/b.z);
    update<<<g,b,0,stream>>>(c,mu,out,nx,ny,z0,z1,scale);
}

// Exchange GPU-resident boundary planes. CUDA-aware MPI is intentional: it avoids staging
// and allows GPUDirect RDMA. Physical-edge halos reproduce the original clamped boundary.
struct HaloStaging { double *send_lo, *recv_lo, *send_hi, *recv_hi; };

static void begin_halo(double* a, size_t lz, size_t plane, int rank, int nranks,
                       bool gpu_aware, HaloStaging h, MPI_Request req[4]) {
    int n=0;
    if (rank > 0) {
        if (!gpu_aware) CUDA_CHECK(cudaMemcpy(h.send_lo,a+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Irecv(gpu_aware?a:h.recv_lo, (int)plane, MPI_DOUBLE, rank-1, 11, MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(gpu_aware?a+plane:h.send_lo, (int)plane, MPI_DOUBLE, rank-1, 12, MPI_COMM_WORLD, &req[n++]);
    } else CUDA_CHECK(cudaMemcpy(a, a+plane, plane*sizeof(double), cudaMemcpyDeviceToDevice));
    if (rank+1 < nranks) {
        if (!gpu_aware) CUDA_CHECK(cudaMemcpy(h.send_hi,a+lz*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Irecv(gpu_aware?a+(lz+1)*plane:h.recv_hi, (int)plane, MPI_DOUBLE, rank+1, 12, MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(gpu_aware?a+lz*plane:h.send_hi, (int)plane, MPI_DOUBLE, rank+1, 11, MPI_COMM_WORLD, &req[n++]);
    } else CUDA_CHECK(cudaMemcpy(a+(lz+1)*plane, a+lz*plane, plane*sizeof(double), cudaMemcpyDeviceToDevice));
    for (;n<4;++n) req[n]=MPI_REQUEST_NULL;
}

static void finish_halo(double* a, size_t lz, size_t plane, int rank, int nranks,
                        bool gpu_aware, HaloStaging h, MPI_Request req[4]) {
    MPI_Waitall(4,req,MPI_STATUSES_IGNORE);
    if (!gpu_aware && rank>0) CUDA_CHECK(cudaMemcpy(a,h.recv_lo,plane*sizeof(double),cudaMemcpyHostToDevice));
    if (!gpu_aware && rank+1<nranks) CUDA_CHECK(cudaMemcpy(a+(lz+1)*plane,h.recv_hi,plane*sizeof(double),cudaMemcpyHostToDevice));
}

static void usage(const char* p) {
    std::printf("Usage: %s [-x N] [-y N] [-z N] [-i N] [-v] [-r] [-h]\n",p);
}

int main(int argc, char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,nranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD,3);

    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) { if(!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(!rank){std::printf("Unknown option: %s\n",argv[i]);usage(argv[0]);} MPI_Finalize();return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations<0 || (size_t)nranks>nz || nx*ny>(size_t)std::numeric_limits<int>::max()) {
        if(!rank) std::fprintf(stderr,"Invalid grid/iteration count, too many ranks, or MPI plane too large\n");
        MPI_Finalize(); return 1;
    }

    // Contiguous, load-balanced global Z slabs.
    const size_t base=nz/nranks, rem=nz%nranks;
    const size_t lz=base+(size_t(rank)<rem), zoff=rank*base+std::min<size_t>(rank,rem);
    const size_t plane=nx*ny, local=(lz+2)*plane;

    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
    int local_rank=0,devices=0; MPI_Comm_rank(local_comm,&local_rank); CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if(!rank) std::fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,4); }
    CUDA_CHECK(cudaSetDevice(local_rank%devices)); MPI_Comm_free(&local_comm);

    bool gpu_aware=false;
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
    gpu_aware = MPIX_Query_cuda_support()!=0;
    #endif

    std::vector<double> host(lz*plane);
    const size_t volume=nx*ny*nz;
    #pragma omp parallel for schedule(static)
    for (long long q=0;q<(long long)(lz*plane);++q) {
        const size_t gid=zoff*plane+(size_t)q;
        host[q]=-1.0+2.0*((((gid+1)*size_t(1299709))%volume)/double(volume));
    }
    double *cold=nullptr,*cnew=nullptr,*mu=nullptr;
    CUDA_CHECK(cudaMalloc(&cold,local*sizeof(double))); CUDA_CHECK(cudaMalloc(&cnew,local*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu,local*sizeof(double))); CUDA_CHECK(cudaMemcpy(cold+plane,host.data(),host.size()*sizeof(double),cudaMemcpyHostToDevice));
    double* staging=nullptr; CUDA_CHECK(cudaHostAlloc(&staging,4*plane*sizeof(double),cudaHostAllocDefault));
    HaloStaging halos{staging,staging+plane,staging+2*plane,staging+3*plane};
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));

    if(!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",
                    nx,ny,nz,iterations,validate?"enabled":"disabled",nranks,omp_get_max_threads());
        std::printf("MPI GPU transport: %s\n",gpu_aware?"GPUDirect/CUDA-aware":"pinned host staging");
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    constexpr double dt=.01,D=1.0,eaa=-(2.0/9.0),ebb=-(2.0/9.0),eab=2.0/9.0,gamma=.5;
    for(int t=0;t<iterations;++t) {
        MPI_Request rq[4]; begin_halo(cold,lz,plane,rank,nranks,gpu_aware,halos,rq);
        launch_chemical(cold,mu,nx,ny,2,lz,gamma,eaa,ebb,eab,stream);
        finish_halo(cold,lz,plane,rank,nranks,gpu_aware,halos,rq);
        launch_chemical(cold,mu,nx,ny,1,std::min<size_t>(2,lz+1),gamma,eaa,ebb,eab,stream);
        if(lz>1) launch_chemical(cold,mu,nx,ny,lz,lz+1,gamma,eaa,ebb,eab,stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        begin_halo(mu,lz,plane,rank,nranks,gpu_aware,halos,rq);
        launch_update(cold,mu,cnew,nx,ny,2,lz,dt*D,stream);
        finish_halo(mu,lz,plane,rank,nranks,gpu_aware,halos,rq);
        launch_update(cold,mu,cnew,nx,ny,1,std::min<size_t>(2,lz+1),dt*D,stream);
        if(lz>1) launch_update(cold,mu,cnew,nx,ny,lz,lz+1,dt*D,stream);
        CUDA_CHECK(cudaStreamSynchronize(stream)); std::swap(cold,cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed=MPI_Wtime()-start;
    if(!rank) std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",elapsed*1000.0,double(volume)*iterations/elapsed/1e6);

    CUDA_CHECK(cudaMemcpy(host.data(),cold+plane,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    double local_min=std::numeric_limits<double>::infinity(), local_max=-local_min; int local_bad=0;
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(|:local_bad)
    for(long long i=0;i<(long long)host.size();++i) { double v=host[i]; local_min=std::min(local_min,v); local_max=std::max(local_max,v); local_bad |= !std::isfinite(v); }
    double global_min=0,global_max=0; int global_bad=0;
    MPI_Reduce(&local_min,&global_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&local_bad,&global_bad,1,MPI_INT,MPI_LOR,0,MPI_COMM_WORLD);

    if(printResults) {
        std::vector<int> counts(nranks),displs(nranks); for(int r=0;r<nranks;++r){size_t rz=base+(size_t(r)<rem);counts[r]=(int)(rz*plane);displs[r]=(int)((r*base+std::min<size_t>(r,rem))*plane);}
        std::vector<double> all; if(!rank) all.resize(volume);
        MPI_Gatherv(host.data(),(int)host.size(),MPI_DOUBLE,rank?nullptr:all.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(!rank) print_results(all,"Concentration");
    }
    int rc=0; if(validate && !rank) { std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",global_min,global_max); rc=(global_bad||global_max>10.0||global_min< -10.0); std::printf("Validation: %s\n",rc?"FAILED":"PASSED"); }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);
    cudaStreamDestroy(stream); cudaFreeHost(staging); cudaFree(cold); cudaFree(cnew); cudaFree(mu); MPI_Finalize(); return rc;
}
