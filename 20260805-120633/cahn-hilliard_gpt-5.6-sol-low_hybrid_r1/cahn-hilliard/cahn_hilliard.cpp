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

#define CUDA_CHECK(call) do {                                                     \
    const cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                     cudaGetErrorString(err_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                              \
    }                                                                              \
} while (0)

__device__ __forceinline__ size_t didx(size_t x, size_t y, size_t z,
                                       size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

__global__ void chemical_potential(const double* __restrict__ c,
                                   double* __restrict__ mu, size_t nx, size_t ny,
                                   size_t local_nz, double gamma, double e_aa,
                                   double e_bb, double e_ab) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t plane = nx * ny;
    const size_t i = didx(x, y, z, nx, plane);
    const size_t xm = x ? x - 1 : x;
    const size_t xp = x + 1 < nx ? x + 1 : x;
    const size_t ym = y ? y - 1 : y;
    const size_t yp = y + 1 < ny ? y + 1 : y;
    const double v = c[i];
    const double lap = c[didx(xm,y,z,nx,plane)] + c[didx(xp,y,z,nx,plane)]
                     + c[didx(x,ym,z,nx,plane)] + c[didx(x,yp,z,nx,plane)]
                     + c[i-plane] + c[i+plane] - 6.0*v;
    mu[i] = 4.5 * ((v + 1.0)*e_aa + (v - 1.0)*e_bb - 2.0*v*e_ab)
          + 3.0*v + v*v*v - gamma*lap;
}

__global__ void update_concentration(double* __restrict__ out,
                                     const double* __restrict__ in,
                                     const double* __restrict__ mu,
                                     size_t nx, size_t ny, size_t local_nz,
                                     double dt_d) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t plane = nx * ny;
    const size_t i = didx(x, y, z, nx, plane);
    const size_t xm = x ? x - 1 : x;
    const size_t xp = x + 1 < nx ? x + 1 : x;
    const size_t ym = y ? y - 1 : y;
    const size_t yp = y + 1 < ny ? y + 1 : y;
    const double v = mu[i];
    const double lap = mu[didx(xm,y,z,nx,plane)] + mu[didx(xp,y,z,nx,plane)]
                     + mu[didx(x,ym,z,nx,plane)] + mu[didx(x,yp,z,nx,plane)]
                     + mu[i-plane] + mu[i+plane] - 6.0*v;
    out[i] = in[i] + dt_d*lap;
}

static void exchange_halos(double* field, size_t local_nz, size_t plane,
                           int rank, int ranks, double* send_lo, double* send_hi,
                           double* recv_lo, double* recv_hi) {
    if (rank == 0)
        CUDA_CHECK(cudaMemcpy(field, field + plane, plane*sizeof(double), cudaMemcpyDeviceToDevice));
    if (rank == ranks - 1)
        CUDA_CHECK(cudaMemcpy(field + (local_nz+1)*plane, field + local_nz*plane,
                              plane*sizeof(double), cudaMemcpyDeviceToDevice));
    const int count = static_cast<int>(plane);
    if (rank) CUDA_CHECK(cudaMemcpy(send_lo,field+plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
    if (rank+1<ranks) CUDA_CHECK(cudaMemcpy(send_hi,field+local_nz*plane,plane*sizeof(double),cudaMemcpyDeviceToHost));
    MPI_Sendrecv(send_lo, count, MPI_DOUBLE,
                 rank ? rank-1 : MPI_PROC_NULL, 10,
                 recv_hi, count, MPI_DOUBLE,
                 rank+1 < ranks ? rank+1 : MPI_PROC_NULL, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_hi, count, MPI_DOUBLE,
                 rank+1 < ranks ? rank+1 : MPI_PROC_NULL, 11,
                 recv_lo, count, MPI_DOUBLE,
                 rank ? rank-1 : MPI_PROC_NULL, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank) CUDA_CHECK(cudaMemcpy(field,recv_lo,plane*sizeof(double),cudaMemcpyHostToDevice));
    if (rank+1<ranks) CUDA_CHECK(cudaMemcpy(field+(local_nz+1)*plane,recv_hi,plane*sizeof(double),cudaMemcpyHostToDevice));
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num> Grid X (default 64)\n"
                "  -y <num> Grid Y (default X)\n  -z <num> Grid Z (default X)\n"
                "  -i <num> Time steps (default 20)\n  -v Validate\n"
                "  -r Print results\n  -h Help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx=64, ny=0, nz=0;
    int iterations=20;
    bool validate=false, print_results_requested=false;
    bool args_ok=true, help=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) print_results_requested=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else args_ok=false;
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (help || !args_ok) {
        if (rank==0) printUsage(argv[0]);
        MPI_Finalize(); return args_ok ? 0 : 1;
    }
    const size_t plane=nx*ny;
    if (!nx || !ny || !nz || iterations < 0 || nz < static_cast<size_t>(ranks) ||
        nx > std::numeric_limits<size_t>::max()/ny ||
        plane > std::numeric_limits<size_t>::max()/nz ||
        plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (print_results_requested && plane*nz > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        if (rank==0) std::fprintf(stderr,"Invalid grid/iterations, too many ranks, or MPI plane too large\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank=0; MPI_Comm_rank(local_comm,&local_rank);
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if(rank==0) std::fprintf(stderr,"No CUDA GPU available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local_comm);

    const size_t base=nz/static_cast<size_t>(ranks), rem=nz%static_cast<size_t>(ranks);
    const size_t local_nz=base+(static_cast<size_t>(rank)<rem);
    const size_t z0=base*rank+std::min(static_cast<size_t>(rank),rem);
    const size_t local_elems=(local_nz+2)*plane;
    std::vector<double> host(local_nz*plane);
    const size_t volume=nx*ny*nz;
#pragma omp parallel for schedule(static)
    for (long long q=0; q<static_cast<long long>(host.size()); ++q) {
        const size_t gid=(z0*plane)+static_cast<size_t>(q);
        const double pseudo=(((gid+1)*size_t{1299709})%volume)/static_cast<double>(volume);
        host[q]=-1.0+2.0*pseudo;
    }
    double *cold=nullptr,*cnew=nullptr,*mu=nullptr;
    double *send_lo=nullptr,*send_hi=nullptr,*recv_lo=nullptr,*recv_hi=nullptr;
    CUDA_CHECK(cudaMalloc(&cold,local_elems*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&cnew,local_elems*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&mu,local_elems*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&send_lo,plane*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&send_hi,plane*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recv_lo,plane*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recv_hi,plane*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold+plane,host.data(),host.size()*sizeof(double),cudaMemcpyHostToDevice));

    if(rank==0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n"
                    "Time steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP max threads: %d\n",
                    nx,ny,nz,iterations,validate?"enabled":"disabled",ranks,omp_get_max_threads());
        std::printf("Running Cahn-Hilliard simulation...\n");
    }
    const dim3 block(32,4,2);
    const dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y,
                    (local_nz+block.z-1)/block.z);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    for(int t=0;t<iterations;++t) {
        exchange_halos(cold,local_nz,plane,rank,ranks,send_lo,send_hi,recv_lo,recv_hi);
        chemical_potential<<<grid,block>>>(cold,mu,nx,ny,local_nz,0.5,-2.0/9.0,-2.0/9.0,2.0/9.0);
        CUDA_CHECK(cudaGetLastError());
        exchange_halos(mu,local_nz,plane,rank,ranks,send_lo,send_hi,recv_lo,recv_hi);
        update_concentration<<<grid,block>>>(cnew,cold,mu,nx,ny,local_nz,0.01);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cold,cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_time=MPI_Wtime()-start;
    double elapsed=0.0;
    MPI_Reduce(&local_time,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) {
        std::printf("Computation time: %ld ms\n",static_cast<long>(elapsed*1000.0));
        const double mcups=elapsed>0 ? (static_cast<double>(volume)*iterations/elapsed/1e6) : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n",mcups);
    }

    int rc=0;
    if(validate || print_results_requested) {
        CUDA_CHECK(cudaMemcpy(host.data(),cold+plane,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    }
    if(validate) {
        int local_bad=0;
        double local_min=std::numeric_limits<double>::infinity();
        double local_max=-std::numeric_limits<double>::infinity();
#pragma omp parallel for reduction(|:local_bad) reduction(min:local_min) reduction(max:local_max)
        for(long long q=0;q<static_cast<long long>(host.size());++q) {
            const double v=host[q]; local_bad |= !std::isfinite(v);
            local_min=std::min(local_min,v); local_max=std::max(local_max,v);
        }
        int bad=0; double minv=0,maxv=0;
        MPI_Reduce(&local_bad,&bad,1,MPI_INT,MPI_LOR,0,MPI_COMM_WORLD);
        MPI_Reduce(&local_min,&minv,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&local_max,&maxv,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if(rank==0) {
            std::printf("Concentration range: [%.6f, %.6f]\n",minv,maxv);
            rc=(bad || maxv>10.0 || minv< -10.0) ? 1 : 0;
            std::printf("Validation: %s\n",rc?"FAILED":"PASSED");
        }
        MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    if(print_results_requested) {
        std::vector<int> counts(ranks),displs(ranks);
        for(int r=0;r<ranks;++r) {
            const size_t rn=base+(static_cast<size_t>(r)<rem);
            counts[r]=static_cast<int>(rn*plane);
            displs[r]=static_cast<int>((base*r+std::min(static_cast<size_t>(r),rem))*plane);
        }
        std::vector<double> global(rank==0?volume:0);
        MPI_Gatherv(host.data(),static_cast<int>(host.size()),MPI_DOUBLE,
                    rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(rank==0) print_results(global,"Concentration");
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    CUDA_CHECK(cudaFreeHost(send_lo)); CUDA_CHECK(cudaFreeHost(send_hi));
    CUDA_CHECK(cudaFreeHost(recv_lo)); CUDA_CHECK(cudaFreeHost(recv_hi));
    MPI_Finalize(); return rc;
}
