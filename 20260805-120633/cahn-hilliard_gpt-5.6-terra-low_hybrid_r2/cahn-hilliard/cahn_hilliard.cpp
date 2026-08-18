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

#define CUDA_CHECK(call) do { const cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__device__ __forceinline__ size_t index3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

__global__ void chemical_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                size_t nx, size_t ny, size_t local_nz,
                                double gamma, double eaa, double ebb, double eab) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t i = index3(x, y, z, nx, ny);
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const double cv = c[i];
    const double lap = c[index3(xp,y,z,nx,ny)] + c[index3(xn,y,z,nx,ny)] +
                       c[index3(x,yp,z,nx,ny)] + c[index3(x,yn,z,nx,ny)] +
                       c[index3(x,y,z+1,nx,ny)] + c[index3(x,y,z-1,nx,ny)] - 6.0 * cv;
    mu[i] = 4.5 * ((cv + 1.0) * eaa + (cv - 1.0) * ebb - 2.0 * cv * eab)
          + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void update_kernel(const double* __restrict__ cold, const double* __restrict__ mu,
                              double* __restrict__ cnew, size_t nx, size_t ny, size_t local_nz,
                              double dtD) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) return;
    const size_t i = index3(x, y, z, nx, ny);
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const double lap = mu[index3(xp,y,z,nx,ny)] + mu[index3(xn,y,z,nx,ny)] +
                       mu[index3(x,yp,z,nx,ny)] + mu[index3(x,yn,z,nx,ny)] +
                       mu[index3(x,y,z+1,nx,ny)] + mu[index3(x,y,z-1,nx,ny)] - 6.0 * mu[i];
    cnew[i] = cold[i] + dtD * lap;
}

static void exchange_halos(double* d_field, size_t plane, size_t local_nz, int rank, int nranks,
                           double* send_lo, double* send_hi, double* recv_lo, double* recv_hi) {
    const size_t bytes = plane * sizeof(double);
    CUDA_CHECK(cudaMemcpy(send_lo, d_field + plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(send_hi, d_field + local_nz * plane, bytes, cudaMemcpyDeviceToHost));
    const int lo = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int hi = rank + 1 == nranks ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(send_lo, static_cast<int>(plane), MPI_DOUBLE, lo, 10, recv_hi, static_cast<int>(plane), MPI_DOUBLE, hi, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_hi, static_cast<int>(plane), MPI_DOUBLE, hi, 11, recv_lo, static_cast<int>(plane), MPI_DOUBLE, lo, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Clamped physical boundaries: copy the adjacent interior plane into the ghost plane.
    if (lo == MPI_PROC_NULL) std::memcpy(recv_lo, send_lo, bytes);
    if (hi == MPI_PROC_NULL) std::memcpy(recv_hi, send_hi, bytes);
    CUDA_CHECK(cudaMemcpy(d_field, recv_lo, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_field + (local_nz + 1) * plane, recv_hi, bytes, cudaMemcpyHostToDevice));
}

static void usage(const char* p) {
    printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t nx = 64, ny = 0, nz = 0; int iterations = 20; bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results_flag = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny = nx; if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 || nz < static_cast<size_t>(nranks)) {
        if (!rank) fprintf(stderr, "Grid dimensions must be positive, iterations non-negative, and z >= MPI ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA device found.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const size_t base = nz / nranks, rem = nz % nranks;
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny, interior = local_nz * plane, storage = (local_nz + 2) * plane;
    std::vector<double> initial(interior);
    const size_t volume = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t q = 0; q < interior; ++q) {
        const size_t global_id = (z0 + q / plane) * plane + q % plane;
        initial[q] = -1.0 + 2.0 * ((((global_id + 1) * 1299709ULL) % volume) / static_cast<double>(volume));
    }
    double *cold, *cnew, *mu; CUDA_CHECK(cudaMalloc(&cold, storage * sizeof(double))); CUDA_CHECK(cudaMalloc(&cnew, storage * sizeof(double))); CUDA_CHECK(cudaMalloc(&mu, storage * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(cold + plane, initial.data(), interior * sizeof(double), cudaMemcpyHostToDevice));
    double *send_lo, *send_hi, *recv_lo, *recv_hi;
    CUDA_CHECK(cudaMallocHost(&send_lo, plane * sizeof(double))); CUDA_CHECK(cudaMallocHost(&send_hi, plane * sizeof(double))); CUDA_CHECK(cudaMallocHost(&recv_lo, plane * sizeof(double))); CUDA_CHECK(cudaMallocHost(&recv_hi, plane * sizeof(double)));
    if (!rank) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled", nranks, omp_get_max_threads()); }
    const dim3 block(8, 8, 4), grid((nx+7)/8, (ny+7)/8, (local_nz+3)/4);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto begin = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        exchange_halos(cold, plane, local_nz, rank, nranks, send_lo, send_hi, recv_lo, recv_hi);
        chemical_kernel<<<grid, block>>>(cold, mu, nx, ny, local_nz, 0.5, -2.0/9.0, -2.0/9.0, 2.0/9.0); CUDA_CHECK(cudaGetLastError());
        exchange_halos(mu, plane, local_nz, rank, nranks, send_lo, send_hi, recv_lo, recv_hi);
        update_kernel<<<grid, block>>>(cold, mu, cnew, nx, ny, local_nz, 0.01); CUDA_CHECK(cudaGetLastError());
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double local_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    double seconds; MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    std::vector<double> local(interior); CUDA_CHECK(cudaMemcpy(local.data(), cold + plane, interior * sizeof(double), cudaMemcpyDeviceToHost));
    if (!rank) printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", seconds * 1000.0, (static_cast<double>(volume) * iterations) / seconds / 1.e6);
    int result_status = 0;
    if (validate) {
        double mn = std::numeric_limits<double>::infinity(), mx = -mn; int bad = 0;
        #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(|:bad)
        for (size_t i = 0; i < interior; ++i) { mn = std::min(mn, local[i]); mx = std::max(mx, local[i]); bad |= !std::isfinite(local[i]); }
        double gmn, gmx; int gbad; MPI_Reduce(&mn,&gmn,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&mx,&gmx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); MPI_Reduce(&bad,&gbad,1,MPI_INT,MPI_BOR,0,MPI_COMM_WORLD);
        if (!rank) {
            const bool valid = !gbad && gmn >= -10. && gmx <= 10.;
            printf("Concentration range: [%.6f, %.6f]\n",gmn,gmx);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result_status = valid ? 0 : 1;
        }
        MPI_Bcast(&result_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (print_results_flag) {
        std::vector<int> counts, displs; std::vector<double> all;
        if (!rank) { counts.resize(nranks); displs.resize(nranks); for(int r=0;r<nranks;++r) { const size_t n = base + (static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*plane); } all.resize(volume); }
        MPI_Gatherv(local.data(), static_cast<int>(interior), MPI_DOUBLE, rank ? nullptr : all.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(all, "Concentration");
    }
    CUDA_CHECK(cudaFreeHost(send_lo)); CUDA_CHECK(cudaFreeHost(send_hi)); CUDA_CHECK(cudaFreeHost(recv_lo)); CUDA_CHECK(cudaFreeHost(recv_hi)); CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    MPI_Finalize(); return result_status;
}
