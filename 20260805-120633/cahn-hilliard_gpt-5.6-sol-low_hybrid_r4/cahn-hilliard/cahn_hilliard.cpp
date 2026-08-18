#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
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

__device__ __forceinline__ size_t didx(int x, int y, int z, int nx, int ny) {
    return (static_cast<size_t>(z) * ny + y) * nx + x;
}

__global__ void chemical_potential(const double* __restrict__ c, double* __restrict__ mu,
                                   int nx, int ny, int nz_local, bool bottom, bool top,
                                   double gamma, double eAA, double eBB, double eAB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz_local) return;
    const int xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
    const int ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
    const int zm = (bottom && z == 1) ? z : z - 1;
    const int zp = (top && z == nz_local) ? z : z + 1;
    const size_t i = didx(x, y, z, nx, ny);
    const double v = c[i];
    const double lap = c[didx(xm,y,z,nx,ny)] + c[didx(xp,y,z,nx,ny)]
                     + c[didx(x,ym,z,nx,ny)] + c[didx(x,yp,z,nx,ny)]
                     + c[didx(x,y,zm,nx,ny)] + c[didx(x,y,zp,nx,ny)] - 6.0 * v;
    mu[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
          + 3.0 * v + v * v * v - gamma * lap;
}

__global__ void update_field(const double* __restrict__ oldc, const double* __restrict__ mu,
                             double* __restrict__ newc, int nx, int ny, int nz_local,
                             bool bottom, bool top, double dtD) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > nz_local) return;
    const int xm = x ? x - 1 : x, xp = x + 1 < nx ? x + 1 : x;
    const int ym = y ? y - 1 : y, yp = y + 1 < ny ? y + 1 : y;
    const int zm = (bottom && z == 1) ? z : z - 1;
    const int zp = (top && z == nz_local) ? z : z + 1;
    const size_t i = didx(x, y, z, nx, ny);
    const double v = mu[i];
    const double lap = mu[didx(xm,y,z,nx,ny)] + mu[didx(xp,y,z,nx,ny)]
                     + mu[didx(x,ym,z,nx,ny)] + mu[didx(x,yp,z,nx,ny)]
                     + mu[didx(x,y,zm,nx,ny)] + mu[didx(x,y,zp,nx,ny)] - 6.0 * v;
    newc[i] = oldc[i] + dtD * lap;
}

static void exchange_faces(double* a, size_t plane, int local_z, int rank, int ranks, double* staging) {
    // Reusable page-locked buffers work with every MPI implementation and allow the
    // CUDA runtime and CUDA-aware MPI transports to use their fastest host path.
    double *send_lo=staging, *send_hi=staging+plane;
    double *recv_lo=staging+2*plane, *recv_hi=staging+3*plane;
    if (rank > 0) CUDA_CHECK(cudaMemcpy(send_lo,a+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
    if (rank+1 < ranks) CUDA_CHECK(cudaMemcpy(send_hi,a+static_cast<size_t>(local_z)*plane,
                                               plane*sizeof(double),cudaMemcpyDeviceToHost));
    MPI_Request req[4]; int n = 0;
    if (rank > 0) {
        MPI_Irecv(recv_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(send_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[n++]);
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(recv_hi, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 12, MPI_COMM_WORLD, &req[n++]);
        MPI_Isend(send_hi, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 11, MPI_COMM_WORLD, &req[n++]);
    }
    if (n) MPI_Waitall(n, req, MPI_STATUSES_IGNORE);
    if (rank > 0) CUDA_CHECK(cudaMemcpy(a,recv_lo,plane*sizeof(double),cudaMemcpyHostToDevice));
    if (rank+1 < ranks) CUDA_CHECK(cudaMemcpy(a+static_cast<size_t>(local_z+1)*plane,recv_hi,
                                               plane*sizeof(double),cudaMemcpyHostToDevice));
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  X size (default 64)\n  -y <num>  Y size\n"
                "  -z <num>  Z size\n  -i <num>  time steps (default 20)\n"
                "  -v        validate\n  -r        print results\n  -h        help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int nx = 64, ny = 0, nz = 0, iterations = 20; bool validate = false, results = false;
    bool bad = false, help = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) results=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else bad=true;
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (help || bad) { if (!rank) usage(argv[0]); MPI_Finalize(); return bad; }
    if (nx<=0 || ny<=0 || nz<=0 || iterations<0 || ranks>nz ||
        static_cast<unsigned long long>(nx)*ny > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr,"Invalid grid/iteration count, or more MPI ranks than Z planes\n");
        MPI_Finalize(); return 1;
    }

    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr,"No CUDA device found\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank=0; MPI_Comm_rank(local_comm,&local_rank); MPI_Comm_free(&local_comm);
    CUDA_CHECK(cudaSetDevice(local_rank % devices));

    const int base=nz/ranks, rem=nz%ranks;
    const int local_z=base+(rank<rem), z0=rank*base+std::min(rank,rem);
    const size_t plane=static_cast<size_t>(nx)*ny, local_n=plane*local_z, alloc_n=plane*(local_z+2);
    std::vector<double> host(local_n);
    const size_t global_n=static_cast<size_t>(nx)*ny*nz;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<static_cast<long long>(local_n); ++i) {
        const size_t gid=static_cast<size_t>(z0)*plane+static_cast<size_t>(i);
        const double pseudo=((gid+1)*1299709ULL % global_n)/static_cast<double>(global_n);
        host[i]=-1.0+2.0*pseudo;
    }
    double *cold=nullptr,*cnew=nullptr,*mu=nullptr,*staging=nullptr;
    CUDA_CHECK(cudaMalloc(&cold,alloc_n*sizeof(double))); CUDA_CHECK(cudaMalloc(&cnew,alloc_n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu,alloc_n*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&staging,4*plane*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold+plane,host.data(),local_n*sizeof(double),cudaMemcpyHostToDevice));
    const dim3 block(32,4,1), grid((nx+31)/32,(ny+3)/4,local_z);
    if (!rank) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %d x %d x %d\nTime steps: %d\nValidation: %s\n",
                    nx,ny,nz,iterations,validate?"enabled":"disabled");
        std::printf("Parallelization: MPI (%d ranks) + OpenMP + CUDA\nRunning Cahn-Hilliard simulation...\n",ranks);
    }
    MPI_Barrier(MPI_COMM_WORLD); const double begin=MPI_Wtime();
    for (int t=0;t<iterations;++t) {
        exchange_faces(cold,plane,local_z,rank,ranks,staging);
        chemical_potential<<<grid,block>>>(cold,mu,nx,ny,local_z,rank==0,rank==ranks-1,
                                           .5,-2.0/9.0,-2.0/9.0,2.0/9.0);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        exchange_faces(mu,plane,local_z,rank,ranks,staging);
        update_field<<<grid,block>>>(cold,mu,cnew,nx,ny,local_z,rank==0,rank==ranks-1,.01);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(cold,cnew);
    }
    const double elapsed=MPI_Wtime()-begin; double max_elapsed=0;
    MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(host.data(),cold+plane,local_n*sizeof(double),cudaMemcpyDeviceToHost));

    std::vector<int> counts,displs; std::vector<double> global;
    if (!rank) { counts.resize(ranks); displs.resize(ranks); global.resize(global_n);
        for(int r=0,off=0;r<ranks;++r){ counts[r]=(base+(r<rem))*static_cast<int>(plane); displs[r]=off; off+=counts[r]; } }
    MPI_Gatherv(host.data(),static_cast<int>(local_n),MPI_DOUBLE,rank?nullptr:global.data(),
                rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int ok=1;
    if (!rank) {
        const double mcups=max_elapsed>0 ? global_n*static_cast<double>(iterations)/max_elapsed/1e6 : 0;
        std::printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",max_elapsed*1000,mcups);
        if (results) print_results(global,"Concentration");
        if (validate) {
            double lo=std::numeric_limits<double>::infinity(), hi=-lo; int finite=1;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(&:finite)
            for(long long i=0;i<static_cast<long long>(global_n);++i){ finite &= std::isfinite(global[i]); lo=std::min(lo,global[i]); hi=std::max(hi,global[i]); }
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",lo,hi);
            ok=finite && lo>=-10.0 && hi<=10.0;
            std::printf("Validation: %s\n",ok?"PASSED":"FAILED");
        }
    }
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
    cudaFree(cold); cudaFree(cnew); cudaFree(mu); cudaFreeHost(staging); MPI_Finalize(); return ok?0:1;
}
