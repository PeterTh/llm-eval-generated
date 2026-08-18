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

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
#define MPI_CHECK(call) do { const int e = (call); if (e != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

__global__ void initialize(double *c, size_t plane, size_t local_z, size_t global_z0,
                           size_t volume) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = plane * local_z;
    if (i < n) {
        const size_t id = global_z0 * plane + i;
        c[plane + i] = -1.0 + 2.0 * (((id + 1) * 1299709 % volume) / (double)volume);
    }
}

__device__ __forceinline__ double laplacian(const double *a, size_t i, size_t x,
                                             size_t y, size_t nx, size_t ny, size_t plane) {
    const size_t xm = x ? i - 1 : i, xp = x + 1 < nx ? i + 1 : i;
    const size_t ym = y ? i - nx : i, yp = y + 1 < ny ? i + nx : i;
    return a[xm] + a[xp] + a[ym] + a[yp] + a[i - plane] + a[i + plane] - 6.0 * a[i];
}

__global__ void chemical(const double *c, double *mu, size_t plane, size_t local_z,
                         size_t nx, size_t ny) {
    const size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = plane * local_z;
    if (q < n) {
        const size_t i = plane + q, x = q % nx, y = (q / nx) % ny;
        const double v = c[i];
        // e_AA=e_BB=-2/9 and e_AB=2/9, retained in this algebraically equivalent form.
        mu[i] = -4.0 * v + 3.0 * v + v * v * v - 0.5 * laplacian(c, i, x, y, nx, ny, plane);
    }
}

__global__ void update(const double *cold, const double *mu, double *cnew, size_t plane,
                       size_t local_z, size_t nx, size_t ny) {
    const size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = plane * local_z;
    if (q < n) {
        const size_t i = plane + q, x = q % nx, y = (q / nx) % ny;
        cnew[i] = cold[i] + 0.01 * laplacian(mu, i, x, y, nx, ny, plane);
    }
}

static void exchange_halos(double *d, size_t plane, size_t local_z, int rank, int ranks,
                           double *send_lo, double *send_hi, double *recv_lo, double *recv_hi,
                           cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(send_lo, d + plane, plane * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(send_hi, d + local_z * plane, plane * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Request req[4]; int nreq = 0;
    if (rank > 0) { MPI_CHECK(MPI_Irecv(recv_lo, (int)plane, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[nreq++])); MPI_CHECK(MPI_Isend(send_lo, (int)plane, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[nreq++])); }
    if (rank + 1 < ranks) { MPI_CHECK(MPI_Irecv(recv_hi, (int)plane, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[nreq++])); MPI_CHECK(MPI_Isend(send_hi, (int)plane, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[nreq++])); }
    if (nreq) MPI_CHECK(MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE));
    if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(d, recv_lo, plane * sizeof(double), cudaMemcpyHostToDevice, stream));
    else CUDA_CHECK(cudaMemcpyAsync(d, d + plane, plane * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpyAsync(d + (local_z + 1) * plane, recv_hi, plane * sizeof(double), cudaMemcpyHostToDevice, stream));
    else CUDA_CHECK(cudaMemcpyAsync(d + (local_z + 1) * plane, d + local_z * plane, plane * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void printUsage(const char *p) { printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p); }

int main(int argc, char **argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank)); MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoll(argv[++i]); else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoll(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoll(argv[++i]); else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true; else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < (size_t)ranks || nx*ny > (size_t)std::numeric_limits<int>::max()) { if (!rank) fprintf(stderr,"Invalid dimensions or too many MPI ranks for z decomposition\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm local; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int local_rank; MPI_CHECK(MPI_Comm_rank(local, &local_rank)); int devices; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices)); MPI_CHECK(MPI_Comm_free(&local));
    const size_t plane=nx*ny, base=nz/ranks, remainder=nz%ranks;
    const size_t local_z=base + (rank < (int)remainder), z0=rank*base + std::min((size_t)rank,remainder), volume=plane*nz;
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nMPI ranks: %d\n",nx,ny,nz,iterations,validate?"enabled":"disabled",ranks); }
    double *cold, *cnew, *mu, *sl, *sh, *rl, *rh; const size_t bytes=(local_z+2)*plane*sizeof(double), pbytes=plane*sizeof(double);
    CUDA_CHECK(cudaMalloc(&cold,bytes)); CUDA_CHECK(cudaMalloc(&cnew,bytes)); CUDA_CHECK(cudaMalloc(&mu,bytes));
    CUDA_CHECK(cudaHostAlloc(&sl,pbytes,cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&sh,pbytes,cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&rl,pbytes,cudaHostAllocDefault)); CUDA_CHECK(cudaHostAlloc(&rh,pbytes,cudaHostAllocDefault));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); const int threads=256; const int blocks=(int)((local_z*plane+threads-1)/threads);
    initialize<<<blocks,threads,0,stream>>>(cold,plane,local_z,z0,volume); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); auto start=std::chrono::steady_clock::now();
    for (int t=0;t<iterations;++t) { exchange_halos(cold,plane,local_z,rank,ranks,sl,sh,rl,rh,stream); chemical<<<blocks,threads,0,stream>>>(cold,mu,plane,local_z,nx,ny); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream)); exchange_halos(mu,plane,local_z,rank,ranks,sl,sh,rl,rh,stream); update<<<blocks,threads,0,stream>>>(cold,mu,cnew,plane,local_z,nx,ny); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream)); std::swap(cold,cnew); }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); double max_seconds; MPI_CHECK(MPI_Reduce(&seconds,&max_seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
    std::vector<double> local_values(local_z*plane); CUDA_CHECK(cudaMemcpy(local_values.data(),cold+plane,local_values.size()*sizeof(double),cudaMemcpyDeviceToHost));
    int local_count=(int)local_values.size(); std::vector<int> counts, offsets; std::vector<double> global;
    if (!rank) { counts.resize(ranks); offsets.resize(ranks); } MPI_CHECK(MPI_Gather(&local_count,1,MPI_INT,rank?nullptr:counts.data(),1,MPI_INT,0,MPI_COMM_WORLD));
    if (!rank) { int off=0; for(int r=0;r<ranks;++r){ offsets[r]=off; off+=counts[r]; } global.resize(volume); } MPI_CHECK(MPI_Gatherv(local_values.data(),local_count,MPI_DOUBLE,rank?nullptr:global.data(),rank?nullptr:counts.data(),rank?nullptr:offsets.data(),MPI_DOUBLE,0,MPI_COMM_WORLD));
    int result = 0;
    if (!rank) {
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",max_seconds*1000.0,(double)volume*iterations/max_seconds/1e6);
        if (printResults) print_results(global,"Concentration");
        if (validate) {
            double mn=global[0], mx=global[0]; int bad=0;
#pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(+:bad)
            for(size_t i=0;i<global.size();++i) { const double v=global[i]; bad += !std::isfinite(v); mn=std::min(mn,v); mx=std::max(mx,v); }
            const bool ok = !bad && mx <= 10.0 && mn >= -10.0;
            printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",mn,mx,ok?"PASSED":"FAILED");
            result = ok ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFreeHost(sl)); CUDA_CHECK(cudaFreeHost(sh)); CUDA_CHECK(cudaFreeHost(rl)); CUDA_CHECK(cudaFreeHost(rh)); CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu)); MPI_CHECK(MPI_Finalize()); return result;
}
