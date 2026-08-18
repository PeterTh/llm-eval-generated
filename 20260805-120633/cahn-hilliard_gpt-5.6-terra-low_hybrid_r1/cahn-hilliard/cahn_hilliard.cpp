#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void initialize(double* c, size_t nx, size_t ny, size_t nzLocal,
                           size_t globalZ0, size_t globalNz) {
    const size_t p = nx * ny;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t n = p * nzLocal;
    if (i < n) {
        const size_t z = i / p;
        const size_t linear = (globalZ0 + z) * p + i % p;
        c[i + p] = -1.0 + 2.0 * (((linear + 1) * 1299709ULL) % (p * globalNz)) /
                                  (double)(p * globalNz);
    }
}

__device__ __forceinline__ double lap(const double* a, size_t x, size_t y, size_t z,
                                      size_t nx, size_t ny, size_t nzLocal) {
    const size_t p = nx * ny, q = z * p + y * nx + x;
    const size_t xp = x + (x + 1 < nx), xm = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), ym = y - (y > 0);
    // z is a local index including halo planes.  Global edge halos are filled by reflection.
    return (a[z*p + y*nx + xp] + a[z*p + y*nx + xm] - 2.0*a[q]) +
           (a[z*p + yp*nx + x] + a[z*p + ym*nx + x] - 2.0*a[q]) +
           (a[q+p] + a[q-p] - 2.0*a[q]);
}

__global__ void chemical(const double* c, double* mu, size_t nx, size_t ny, size_t nzLocal) {
    const size_t p = nx * ny, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < p * nzLocal) {
        const size_t z = i / p + 1, q = z * p + i % p;
        const double v = c[q];
        // With the benchmark's parameters, this is the original expression retained verbatim.
        mu[q] = 4.5 * ((v + 1.0) * (-2.0/9.0) + (v - 1.0) * (-2.0/9.0) - 2.0*v*(2.0/9.0))
              + 3.0*v + v*v*v - 0.5 * lap(c, i % nx, (i / nx) % ny, z, nx, ny, nzLocal);
    }
}

__global__ void update(const double* c, const double* mu, double* next,
                       size_t nx, size_t ny, size_t nzLocal) {
    const size_t p = nx * ny, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < p * nzLocal) {
        const size_t z = i / p + 1, q = z * p + i % p;
        next[q] = c[q] + 0.01 * lap(mu, i % nx, (i / nx) % ny, z, nx, ny, nzLocal);
    }
}

static void exchangeHalos(double* d, size_t plane, size_t nzLocal, int rank, int ranks,
                          std::vector<double>& sendLo, std::vector<double>& sendHi,
                          std::vector<double>& recvLo, std::vector<double>& recvHi) {
    CUDA_CHECK(cudaMemcpy(sendLo.data(), d + plane, plane*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sendHi.data(), d + nzLocal*plane, plane*sizeof(double), cudaMemcpyDeviceToHost));
    const int lo = rank ? rank - 1 : MPI_PROC_NULL, hi = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(sendLo.data(), (int)plane, MPI_DOUBLE, lo, 11, recvHi.data(), (int)plane, MPI_DOUBLE, hi, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi.data(), (int)plane, MPI_DOUBLE, hi, 12, recvLo.data(), (int)plane, MPI_DOUBLE, lo, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) std::copy(sendLo.begin(), sendLo.end(), recvLo.begin());
    if (rank == ranks-1) std::copy(sendHi.begin(), sendHi.end(), recvHi.begin());
    CUDA_CHECK(cudaMemcpy(d, recvLo.data(), plane*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d + (nzLocal+1)*plane, recvHi.data(), plane*sizeof(double), cudaMemcpyHostToDevice));
}

static void usage(const char* p) { printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, results=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true; else if (!strcmp(argv[i],"-r")) results=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < (size_t)ranks || nx*ny > (size_t)std::numeric_limits<int>::max()) {
        if (!rank) fprintf(stderr,"Invalid dimensions or more MPI ranks than z planes\\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    const size_t base=nz/ranks, rem=nz%ranks, nzLocal=base + (rank < (int)rem);
    const size_t z0=rank*base + std::min((size_t)rank, rem), plane=nx*ny, localN=(nzLocal+2)*plane;
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\\nGrid size: %zu x %zu x %zu\\nTime steps: %d\\nValidation: %s\\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); }
    double *c, *next, *mu; CUDA_CHECK(cudaMalloc(&c,localN*sizeof(double))); CUDA_CHECK(cudaMalloc(&next,localN*sizeof(double))); CUDA_CHECK(cudaMalloc(&mu,localN*sizeof(double)));
    CUDA_CHECK(cudaMemset(c,0,localN*sizeof(double))); CUDA_CHECK(cudaMemset(next,0,localN*sizeof(double))); CUDA_CHECK(cudaMemset(mu,0,localN*sizeof(double)));
    const int threads=256, blocks=(int)((plane*nzLocal+threads-1)/threads);
    initialize<<<blocks,threads>>>(c,nx,ny,nzLocal,z0,nz); CUDA_CHECK(cudaGetLastError());
    std::vector<double> sl(plane), sh(plane), rl(plane), rh(plane);
    exchangeHalos(c,plane,nzLocal,rank,ranks,sl,sh,rl,rh);
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    for (int t=0;t<iterations;++t) {
        chemical<<<blocks,threads>>>(c,mu,nx,ny,nzLocal); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        exchangeHalos(mu,plane,nzLocal,rank,ranks,sl,sh,rl,rh);
        update<<<blocks,threads>>>(c,mu,next,nx,ny,nzLocal); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        exchangeHalos(next,plane,nzLocal,rank,ranks,sl,sh,rl,rh); std::swap(c,next);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); const double elapsed=MPI_Wtime()-start;
    double maxElapsed=0.0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<double> local(plane*nzLocal); CUDA_CHECK(cudaMemcpy(local.data(),c+plane,local.size()*sizeof(double),cudaMemcpyDeviceToHost));
    int finite=1; double mn=std::numeric_limits<double>::infinity(), mx=-mn;
    #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(&:finite)
    for (size_t i=0;i<local.size();++i) { finite &= std::isfinite(local[i]); mn=std::min(mn,local[i]); mx=std::max(mx,local[i]); }
    int allFinite; double globalMin,globalMax; MPI_Reduce(&finite,&allFinite,1,MPI_INT,MPI_LAND,0,MPI_COMM_WORLD); MPI_Reduce(&mn,&globalMin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&mx,&globalMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (!rank) { printf("Computation time: %.3f ms\\nPerformance: %.3f MCellUpdates/s\\n",maxElapsed*1000.0,(double)(nx*ny*nz)*iterations/maxElapsed/1e6); if(validate) { printf("Concentration range: [%.6f, %.6f]\\nValidation: %s\\n",globalMin,globalMax,(allFinite&&globalMin>=-10&&globalMax<=10)?"PASSED":"FAILED"); } }
    if (results) {
        std::vector<int> counts, displs; std::vector<double> whole;
        if (!rank) { counts.resize(ranks); displs.resize(ranks); for(int r=0;r<ranks;++r){size_t n=base+(r<(int)rem);counts[r]=(int)(n*plane);displs[r]=(int)((r*base+std::min((size_t)r,rem))*plane);} whole.resize(nx*ny*nz); }
        MPI_Gatherv(local.data(),(int)local.size(),MPI_DOUBLE,rank?nullptr:whole.data(),rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if (!rank) print_results(whole,"Concentration");
    }
    int failed = validate && (!allFinite || globalMin < -10 || globalMax > 10);
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(c)); CUDA_CHECK(cudaFree(next)); CUDA_CHECK(cudaFree(mu)); MPI_Finalize();
    return failed ? 1 : 0;
}
