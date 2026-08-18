#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// The z direction is distributed between MPI ranks.  Each rank retains one
// cell of halo on either side; the two halo exchanges are the only MPI traffic
// in a timestep.  x and y use the original clamped boundary conditions.
__host__ __device__ inline constexpr size_t at(const size_t x, const size_t y, const size_t z,
                           const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ size_t clamp_index(const int v, const size_t n) {
    return static_cast<size_t>(v < 0 ? 0 : (v >= static_cast<int>(n) ? n - 1 : v));
}

__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                            const size_t x, const size_t y,
                                            const size_t z, const size_t nx,
                                            const size_t ny, const size_t nz,
                                            const size_t global_z,
                                            const size_t global_nz) {
    const size_t xm = clamp_index(static_cast<int>(x) - 1, nx);
    const size_t xp = clamp_index(static_cast<int>(x) + 1, nx);
    const size_t ym = clamp_index(static_cast<int>(y) - 1, ny);
    const size_t yp = clamp_index(static_cast<int>(y) + 1, ny);
    const size_t zm = (global_z == 0) ? z : z - 1;
    const size_t zp = (global_z + 1 == global_nz) ? z : z + 1;
    const size_t i = at(x, y, z, nx, ny);
    return a[at(xp,y,z,nx,ny)] + a[at(xm,y,z,nx,ny)]
         + a[at(x,yp,z,nx,ny)] + a[at(x,ym,z,nx,ny)]
         + a[at(x,y,zp,nx,ny)] + a[at(x,y,zm,nx,ny)] - 6.0 * a[i];
}

__global__ void chemical_potential(const double* __restrict__ c,
                                   double* __restrict__ mu, size_t nx,
                                   size_t ny, size_t local_nz, size_t z0,
                                   size_t global_nz, double gamma,
                                   double eAA, double eBB, double eAB) {
    const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = nx * ny * local_nz;
    if (q >= n) return;
    const size_t z = q / (nx * ny) + 1;
    const size_t rem = q % (nx * ny);
    const size_t y = rem / nx;
    const size_t x = rem % nx;
    const double cv = c[at(x,y,z,nx,ny)];
    mu[at(x,y,z,nx,ny)] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * laplacian(c, x, y, z, nx, ny, local_nz + 2,
                                             z0 + z - 1, global_nz);
}

__global__ void update_concentration(double* __restrict__ cnew,
                                     const double* __restrict__ cold,
                                     const double* __restrict__ mu, size_t nx,
                                     size_t ny, size_t local_nz, size_t z0,
                                     size_t global_nz, double dtD) {
    const size_t q = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t n = nx * ny * local_nz;
    if (q >= n) return;
    const size_t z = q / (nx * ny) + 1;
    const size_t rem = q % (nx * ny);
    const size_t y = rem / nx;
    const size_t x = rem % nx;
    const size_t i = at(x,y,z,nx,ny);
    cnew[i] = cold[i] + dtD * laplacian(mu, x, y, z, nx, ny, local_nz + 2,
                                        z0 + z - 1, global_nz);
}

static void cuda_check(const cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void exchange(double* d, std::vector<double>& lo, std::vector<double>& hi,
                     std::vector<double>& rlo, std::vector<double>& rhi,
                     size_t nx, size_t ny, size_t local_nz, size_t z0,
                     size_t global_nz, int rank, int nranks, MPI_Comm comm) {
    const size_t plane = nx * ny;
    cuda_check(cudaMemcpy(lo.data(), d + plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "D2H lower");
    cuda_check(cudaMemcpy(hi.data(), d + local_nz * plane, plane * sizeof(double), cudaMemcpyDeviceToHost), "D2H upper");
    if (rank == 0) std::copy(lo.begin(), lo.end(), rlo.begin());
    if (rank == nranks - 1) std::copy(hi.begin(), hi.end(), rhi.begin());
    MPI_Request req[4]; int nreq = 0;
    if (rank > 0) { MPI_Irecv(rlo.data(), static_cast<int>(plane), MPI_DOUBLE, rank-1, 71, comm, &req[nreq++]); MPI_Isend(lo.data(), static_cast<int>(plane), MPI_DOUBLE, rank-1, 72, comm, &req[nreq++]); }
    if (rank + 1 < nranks) { MPI_Irecv(rhi.data(), static_cast<int>(plane), MPI_DOUBLE, rank+1, 72, comm, &req[nreq++]); MPI_Isend(hi.data(), static_cast<int>(plane), MPI_DOUBLE, rank+1, 71, comm, &req[nreq++]); }
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    cuda_check(cudaMemcpy(d, rlo.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "H2D lower halo");
    cuda_check(cudaMemcpy(d + (local_nz + 1) * plane, rhi.data(), plane * sizeof(double), cudaMemcpyHostToDevice), "H2D upper halo");
    (void)z0; (void)global_nz;
}

static void initialize(std::vector<double>& c, size_t nx, size_t ny, size_t local_nz,
                       size_t z0, size_t global_nz) {
    const size_t plane = nx * ny;
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(local_nz * plane); ++q) {
        const size_t gz = z0 + static_cast<size_t>(q) / plane;
        const size_t linear = gz * plane + static_cast<size_t>(q) % plane;
        const size_t pseudo = ((linear + 1) * 1299709ULL) % (nx * ny * global_nz);
        c[plane + static_cast<size_t>(q)] = -1.0 + 2.0 * static_cast<double>(pseudo) / static_cast<double>(nx * ny * global_nz);
    }
}

static void printUsage(const char* prog) {
    printf("Usage: %s [options]\n  -x <num>  X dimension (default 64)\n  -y <num>  Y dimension\n  -z <num>  Z dimension\n  -i <num>  time steps (default 20)\n  -v        validate\n  -r        print results\n  -h        show help\n", prog);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    (void)provided;
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64, ny = 0, nz = 0; int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) fprintf(stderr, "Unknown option: %s\n", argv[i]); MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx; if (nz == 0) nz = nx;
    if (nz < static_cast<size_t>(nranks) || nx == 0 || ny == 0 || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Grid Z must be at least the MPI process count and dimensions must be positive.\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / static_cast<size_t>(nranks), extra = nz % static_cast<size_t>(nranks);
    const size_t local_nz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t z0 = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t plane = nx * ny, local_size = (local_nz + 2) * plane;
    int devices = 0; cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    cuda_check(cudaSetDevice(rank % devices), "cudaSetDevice");

    if (rank == 0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled"); printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices: %d\n", nranks, omp_get_max_threads(), devices); }
    std::vector<double> h_c(local_size), send_lo(plane), send_hi(plane), recv_lo(plane), recv_hi(plane);
    initialize(h_c, nx, ny, local_nz, z0, nz);
    const size_t bytes = local_size * sizeof(double);
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cuda_check(cudaMalloc(&d_cold, bytes), "cudaMalloc cold"); cuda_check(cudaMalloc(&d_cnew, bytes), "cudaMalloc cnew"); cuda_check(cudaMalloc(&d_mu, bytes), "cudaMalloc mu");
    cuda_check(cudaMemcpy(d_cold, h_c.data(), bytes, cudaMemcpyHostToDevice), "initial H2D");
    exchange(d_cold, send_lo, send_hi, recv_lo, recv_hi, nx, ny, local_nz, z0, nz, rank, nranks, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const size_t blocks = (local_nz * plane + 255) / 256;
    for (int t = 0; t < iterations; ++t) {
        chemical_potential<<<blocks, 256>>>(d_cold, d_mu, nx, ny, local_nz, z0, nz, 0.5, -2.0/9.0, -2.0/9.0, 2.0/9.0);
        cuda_check(cudaGetLastError(), "chemical_potential launch"); cuda_check(cudaDeviceSynchronize(), "chemical_potential");
        exchange(d_mu, send_lo, send_hi, recv_lo, recv_hi, nx, ny, local_nz, z0, nz, rank, nranks, MPI_COMM_WORLD);
        update_concentration<<<blocks, 256>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, z0, nz, 0.01);
        cuda_check(cudaGetLastError(), "update launch"); cuda_check(cudaDeviceSynchronize(), "update");
        std::swap(d_cold, d_cnew);
        // The next chemical-potential stencil needs the newly updated c halo.
        if (t + 1 < iterations)
            exchange(d_cold, send_lo, send_hi, recv_lo, recv_hi, nx, ny, local_nz, z0, nz, rank, nranks, MPI_COMM_WORLD);
    }
    cuda_check(cudaMemcpy(h_c.data() + plane, d_cold + plane, local_nz * plane * sizeof(double), cudaMemcpyDeviceToHost), "final D2H");
    const double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double max_elapsed = 0.0; MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) { printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0)); printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(nx*ny*nz) * iterations / max_elapsed / 1e6); }

    std::vector<double> result; std::vector<int> counts, displs;
    if (rank == 0) { result.resize(nx * ny * nz); counts.resize(nranks); displs.resize(nranks); for (int r=0;r<nranks;++r) { const size_t ln=base+(static_cast<size_t>(r)<extra); counts[r]=static_cast<int>(ln*plane); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),extra))*plane); } }
    MPI_Gatherv(h_c.data() + plane, static_cast<int>(local_nz*plane), MPI_DOUBLE, rank == 0 ? result.data() : nullptr, rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0 && printResults) print_results(result, "Concentration");
    int local_ok = 1; double local_min = h_c[plane], local_max = h_c[plane];
    #pragma omp parallel for reduction(&:local_ok) reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (long long q=0;q<static_cast<long long>(local_nz*plane);++q) { const double v=h_c[plane+static_cast<size_t>(q)]; local_ok &= std::isfinite(v) && v <= 10.0 && v >= -10.0; local_min=std::min(local_min,v); local_max=std::max(local_max,v); }
    int global_ok=0; double global_min=0, global_max=0; MPI_Reduce(&local_ok,&global_ok,1,MPI_INT,MPI_LAND,0,MPI_COMM_WORLD); MPI_Reduce(&local_min,&global_min,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (rank == 0 && validate) { printf("Validating result...\nConcentration range: [%.6f, %.6f]\nValidation: %s\n", global_min, global_max, global_ok ? "PASSED" : "FAILED"); }
    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu); MPI_Finalize(); return (validate && rank == 0 && !global_ok) ? 1 : 0;
}
