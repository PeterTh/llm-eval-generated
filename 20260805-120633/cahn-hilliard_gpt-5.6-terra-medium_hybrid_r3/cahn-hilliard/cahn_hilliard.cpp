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
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

using index_t = unsigned long long;

__device__ __forceinline__ index_t dindex(const int x, const int y, const int z,
                                           const int nx, const int ny) {
    return (static_cast<index_t>(z) * ny + y) * nx + x;
}

// z is local and includes one ghost plane on each side.  The physical boundary
// is clamped exactly as in the original serial implementation.
__global__ void initialize_kernel(double* c, int nx, int ny, int local_nz, int z0,
                                  index_t global_volume) {
    const index_t n = static_cast<index_t>(nx) * ny * local_nz;
    for (index_t i = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
         i < n; i += static_cast<index_t>(blockDim.x) * gridDim.x) {
        const int x = i % nx;
        const int y = (i / nx) % ny;
        const int z = i / (static_cast<index_t>(nx) * ny);
        const index_t global_id = (static_cast<index_t>(z0 + z) * ny + y) * nx + x;
        c[dindex(x, y, z + 1, nx, ny)] = -1.0 + 2.0 *
            ((static_cast<double>(((global_id + 1) * 1299709ULL) % global_volume)) / global_volume);
    }
}

__device__ __forceinline__ double laplacian(const double* a, int x, int y, int z,
                                             int nx, int ny, int local_nz,
                                             bool low_boundary, bool high_boundary) {
    const int xp = x + (x + 1 < nx), xn = x - (x > 0);
    const int yp = y + (y + 1 < ny), yn = y - (y > 0);
    const int zp = (z == 0 && low_boundary) ? z : z + 1;
    const int zn = (z == local_nz - 1 && high_boundary) ? z : z - 1;
    const int zz = z + 1;
    const index_t q = dindex(x, y, zz, nx, ny);
    return a[dindex(xp,y,zz,nx,ny)] + a[dindex(xn,y,zz,nx,ny)] +
           a[dindex(x,yp,zz,nx,ny)] + a[dindex(x,yn,zz,nx,ny)] +
           a[dindex(x,y,zp + 1,nx,ny)] + a[dindex(x,y,zn + 1,nx,ny)] - 6.0 * a[q];
}

__global__ void chemical_kernel(const double* c, double* mu, int nx, int ny, int local_nz,
                                bool low_boundary, bool high_boundary,
                                double gamma, double eaa, double ebb, double eab) {
    const index_t n = static_cast<index_t>(nx) * ny * local_nz;
    for (index_t i = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
         i < n; i += static_cast<index_t>(blockDim.x) * gridDim.x) {
        const int x = i % nx, y = (i / nx) % ny, z = i / (static_cast<index_t>(nx) * ny);
        const index_t q = dindex(x,y,z + 1,nx,ny);
        const double v = c[q];
        mu[q] = 4.5 * ((v + 1.0) * eaa + (v - 1.0) * ebb - 2.0 * v * eab) +
                3.0 * v + v * v * v - gamma * laplacian(c,x,y,z,nx,ny,local_nz,low_boundary,high_boundary);
    }
}

__global__ void update_kernel(const double* cold, const double* mu, double* cnew,
                              int nx, int ny, int local_nz, bool low_boundary, bool high_boundary,
                              double dt) {
    const index_t n = static_cast<index_t>(nx) * ny * local_nz;
    for (index_t i = blockIdx.x * static_cast<index_t>(blockDim.x) + threadIdx.x;
         i < n; i += static_cast<index_t>(blockDim.x) * gridDim.x) {
        const int x = i % nx, y = (i / nx) % ny, z = i / (static_cast<index_t>(nx) * ny);
        const index_t q = dindex(x,y,z + 1,nx,ny);
        cnew[q] = cold[q] + dt * laplacian(mu,x,y,z,nx,ny,local_nz,low_boundary,high_boundary);
    }
}

static void exchange_halos(double* d_field, std::vector<double>& send_lo, std::vector<double>& send_hi,
                           std::vector<double>& recv_lo, std::vector<double>& recv_hi,
                           int plane, int local_nz, int rank, int ranks) {
    MPI_Request req[4]; int nr = 0;
    if (rank > 0) MPI_Irecv(recv_lo.data(), plane, MPI_DOUBLE, rank - 1, 11, MPI_COMM_WORLD, &req[nr++]);
    if (rank + 1 < ranks) MPI_Irecv(recv_hi.data(), plane, MPI_DOUBLE, rank + 1, 12, MPI_COMM_WORLD, &req[nr++]);
    if (rank > 0) CUDA_CHECK(cudaMemcpy(send_lo.data(), d_field + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpy(send_hi.data(), d_field + static_cast<size_t>(local_nz) * plane,
                                                  plane * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank > 0) MPI_Isend(send_lo.data(), plane, MPI_DOUBLE, rank - 1, 12, MPI_COMM_WORLD, &req[nr++]);
    if (rank + 1 < ranks) MPI_Isend(send_hi.data(), plane, MPI_DOUBLE, rank + 1, 11, MPI_COMM_WORLD, &req[nr++]);
    if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
    if (rank > 0) CUDA_CHECK(cudaMemcpy(d_field, recv_lo.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpy(d_field + static_cast<size_t>(local_nz + 1) * plane,
                                                  recv_hi.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
}

static void usage(const char* p, int rank) {
    if (rank == 0) std::printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) { if (!rank) std::fprintf(stderr, "MPI lacks required thread support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, print_results_flag = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i+1<argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i+1<argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i+1<argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i+1<argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!std::strcmp(argv[i], "-h")) { usage(argv[0], rank); MPI_Finalize(); return 0; }
        else { usage(argv[0], rank); MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx; if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < static_cast<size_t>(ranks) || nx > INT_MAX || ny > INT_MAX || nz > INT_MAX) {
        if (!rank) std::fprintf(stderr, "Invalid grid or more MPI ranks than z planes\n"); MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int device_count = 0; CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank); CUDA_CHECK(cudaSetDevice(local_rank % device_count)); MPI_Comm_free(&local_comm);
    const int base = nz / ranks, extra = nz % ranks;
    const int local_nz = base + (rank < extra); const int z0 = rank * base + std::min(rank, extra);
    const int inx=nx, iny=ny; const int plane = inx * iny;
    const size_t local_count = static_cast<size_t>(plane) * local_nz, alloc_count = static_cast<size_t>(plane) * (local_nz + 2);
    if (rank == 0) std::printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\nGrid size: %zu x %zu x %zu, ranks: %d\n", nx,ny,nz,ranks);
    double *d_cold, *d_cnew, *d_mu; CUDA_CHECK(cudaMalloc(&d_cold, alloc_count*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_cnew, alloc_count*sizeof(double))); CUDA_CHECK(cudaMalloc(&d_mu, alloc_count*sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cold,0,alloc_count*sizeof(double))); CUDA_CHECK(cudaMemset(d_cnew,0,alloc_count*sizeof(double))); CUDA_CHECK(cudaMemset(d_mu,0,alloc_count*sizeof(double)));
    const int threads=256, blocks=std::min(65535, static_cast<int>((local_count + threads-1)/threads));
    initialize_kernel<<<blocks,threads>>>(d_cold,inx,iny,local_nz,z0,static_cast<index_t>(nx)*ny*nz); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> sl(plane), sh(plane), rl(plane), rh(plane);
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::steady_clock::now();
    for (int t=0; t<iterations; ++t) {
        exchange_halos(d_cold,sl,sh,rl,rh,plane,local_nz,rank,ranks);
        chemical_kernel<<<blocks,threads>>>(d_cold,d_mu,inx,iny,local_nz,rank==0,rank==ranks-1,0.5,-2.0/9.0,-2.0/9.0,2.0/9.0); CUDA_CHECK(cudaGetLastError());
        exchange_halos(d_mu,sl,sh,rl,rh,plane,local_nz,rank,ranks);
        update_kernel<<<blocks,threads>>>(d_cold,d_mu,d_cnew,inx,iny,local_nz,rank==0,rank==ranks-1,0.01); CUDA_CHECK(cudaGetLastError());
        std::swap(d_cold,d_cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); const double local_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); double seconds=0;
    MPI_Reduce(&local_seconds,&seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    std::vector<double> local(local_count);
    CUDA_CHECK(cudaMemcpy(local.data(),d_cold+plane,local_count*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<int> counts, displs; std::vector<double> global;
    if (rank==0) { counts.resize(ranks); displs.resize(ranks); for(int r=0;r<ranks;++r) { int n=base+(r<extra); counts[r]=n*plane; displs[r]=(r*base+std::min(r,extra))*plane; } if (validate || print_results_flag) global.resize(nx*ny*nz); }
    if (validate || print_results_flag)
        MPI_Gatherv(local.data(), static_cast<int>(local_count), MPI_DOUBLE, rank==0?global.data():nullptr,
                    rank==0?counts.data():nullptr, rank==0?displs.data():nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank==0) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",seconds*1000.0,(static_cast<double>(nx)*ny*nz*iterations)/(seconds*1e6));
        if (print_results_flag) print_results(global,"Concentration");
        if (validate) { int bad=0; double lo=std::numeric_limits<double>::infinity(), hi=-lo;
            #pragma omp parallel for reduction(min:lo) reduction(max:hi) reduction(|:bad)
            for (long long i=0;i<static_cast<long long>(global.size());++i) { const double v=global[i]; if (!std::isfinite(v)) bad=1; lo=std::min(lo,v); hi=std::max(hi,v); }
            std::printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",lo,hi,(!bad && lo>=-10.0 && hi<=10.0)?"PASSED":"FAILED");
            if (bad || lo < -10.0 || hi > 10.0) { CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu)); MPI_Finalize(); return 1; }
        }
    }
    CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_cnew)); CUDA_CHECK(cudaFree(d_mu)); MPI_Finalize(); return 0;
}
