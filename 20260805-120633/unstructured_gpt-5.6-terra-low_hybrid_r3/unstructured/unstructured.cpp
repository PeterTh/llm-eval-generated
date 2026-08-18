#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } \
} while (0)

// Each rank owns a contiguous set of grid rows plus one halo row at either end.
// Keeping the two dynamic fields as SoA arrays gives coalesced CUDA accesses.
struct DeviceState {
    val_t *energy = nullptr, *flux = nullptr, *next_energy = nullptr, *next_flux = nullptr;
    size_t count = 0;
    void allocate(size_t n) {
        count = n;
        CUDA_CHECK(cudaMalloc(&energy, n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&flux, n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&next_energy, n * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&next_flux, n * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(energy, 0, n * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(flux, 0, n * sizeof(val_t)));
    }
    void swap() { std::swap(energy, next_energy); std::swap(flux, next_flux); }
    void destroy() { cudaFree(energy); cudaFree(flux); cudaFree(next_energy); cudaFree(next_flux); }
};

__global__ void updateKernel(const val_t* energy, const val_t* flux, val_t* next_energy,
                             val_t* next_flux, int width, int local_rows, int global_row0,
                             int global_height) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (col >= width || lr > local_rows) return;
    const int gr = global_row0 + lr - 1;
    const int i = lr * width + col;
    const bool inflow = (gr == 0 && col == 0) || (gr == global_height - 1 && col == width - 1);
    const bool outflow = (gr == 0 && col == width - 1) || (gr == global_height - 1 && col == 0);
    const val_t external = inflow ? 0.5 : (outflow ? -0.5 : 0.0);
    const val_t self = energy[i];
    val_t total = external;
    // Preserve the reference operation order for reproducible double precision results.
    if (gr > 0) total += (energy[i - width] - self) * 0.8 * 1.0 * 0.25;
    if (gr + 1 < global_height) total += (energy[i + width] - self) * 0.8 * 1.0 * 0.25;
    if (col > 0) total += (energy[i - 1] - self) * 0.8 * 1.0 * 0.25;
    if (col + 1 < width) total += (energy[i + 1] - self) * 0.8 * 1.0 * 0.25;
    next_energy[i] = self + total;
    next_flux[i] = flux[i] + fabs(total);
}

static void usage(const char* p) {
    printf("Usage: %s [-n grid_size] [-i iterations] [-v] [-r] [-h]\n", p);
}

static uint64_t hash_local(const std::vector<val_t>& energy, const std::vector<val_t>& flux,
                           int width, int row0) {
    uint64_t h = 0;
    #pragma omp parallel for reduction(^:h) schedule(static)
    for (size_t k = 0; k < energy.size(); ++k) {
        uint64_t e, f;
        memcpy(&e, &energy[k], sizeof(e)); memcpy(&f, &flux[k], sizeof(f));
        const uint64_t global_i = static_cast<uint64_t>(row0) * width + k;
        h ^= (e + global_i) * 0x9e3779b97f4a7c15ULL;
        h ^= (f + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iters = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || n < ranks) { if (!rank) fprintf(stderr, "grid size must be at least the MPI rank count\n"); MPI_Finalize(); return 1; }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const int base = n / ranks, extra = n % ranks;
    const int local_rows = base + (rank < extra ? 1 : 0);
    const int row0 = rank * base + std::min(rank, extra);
    const size_t local_count = static_cast<size_t>(local_rows) * n;
    const size_t alloc_count = static_cast<size_t>(local_rows + 2) * n;
    DeviceState d; d.allocate(alloc_count);
    std::vector<val_t> send_top(n), send_bottom(n), recv_top(n), recv_bottom(n), local_energy(local_count), local_flux(local_count);

    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %lld elements\nIterations: %d\nMPI ranks: %d, OpenMP threads/rank: %d\n", n, n, (long long)n*n, iters, ranks, omp_get_max_threads());
        printf("CUDA devices visible/rank: %d\n\nRunning simulation...\n", devices);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    const dim3 block(32, 8), grid((n + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
    for (int iter = 0; iter < iters; ++iter) {
        // Stage halo data through host memory, so this works with both CUDA-aware and ordinary MPI.
        CUDA_CHECK(cudaMemcpy(send_top.data(), d.energy + n, n * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(send_bottom.data(), d.energy + local_rows * n, n * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_Request req[4]; int nr = 0;
        if (rank) { MPI_Irecv(recv_top.data(), n, MPI_DOUBLE, rank-1, 11, MPI_COMM_WORLD, &req[nr++]); MPI_Isend(send_top.data(), n, MPI_DOUBLE, rank-1, 12, MPI_COMM_WORLD, &req[nr++]); }
        if (rank + 1 < ranks) { MPI_Irecv(recv_bottom.data(), n, MPI_DOUBLE, rank+1, 12, MPI_COMM_WORLD, &req[nr++]); MPI_Isend(send_bottom.data(), n, MPI_DOUBLE, rank+1, 11, MPI_COMM_WORLD, &req[nr++]); }
        MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        if (rank) CUDA_CHECK(cudaMemcpy(d.energy, recv_top.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        if (rank + 1 < ranks) CUDA_CHECK(cudaMemcpy(d.energy + (local_rows + 1) * n, recv_bottom.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        updateKernel<<<grid, block>>>(d.energy, d.flux, d.next_energy, d.next_flux, n, local_rows, row0, n);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); d.swap();
    }
    const double local_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    double elapsed_ms; MPI_Reduce(&local_ms, &elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(local_energy.data(), d.energy + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_flux.data(), d.flux + n, local_count * sizeof(val_t), cudaMemcpyDeviceToHost));
    val_t es = 0, fs = 0, emin = std::numeric_limits<val_t>::max(), emax = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:es,fs) reduction(min:emin) reduction(max:emax) schedule(static)
    for (size_t i = 0; i < local_count; ++i) { es += local_energy[i]; fs += local_flux[i]; emin = std::min(emin, local_energy[i]); emax = std::max(emax, local_energy[i]); }
    val_t ges, gfs, gemin, gemax; MPI_Reduce(&es,&ges,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&fs,&gfs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(&emin,&gemin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&emax,&gemax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    uint64_t lh = hash_local(local_energy, local_flux, n, row0), hash; MPI_Reduce(&lh, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) { const double geps = elapsed_ms > 0 ? (static_cast<double>(n)*n*std::max(iters-1,1))/(elapsed_ms/1000.0)/1e9 : 0; printf("Computation time: %.3f ms\nPerformance: %.4f GigaElements/s, %.4f GFLOPS\n  Result hash: %016llX\n", elapsed_ms, geps, geps*22., (unsigned long long)hash); if (validate) printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", ges,gfs,gemin,gemax,(std::isfinite(ges)&&std::isfinite(gfs)&&std::isfinite(gemin)&&std::isfinite(gemax))?"PASSED":"FAILED"); }
    if (results) { std::vector<int> counts(ranks), displs(ranks); for(int r=0;r<ranks;++r){ const int rows=base+(r<extra); counts[r]=rows*n; displs[r]=(r*base+std::min(r,extra))*n; } std::vector<val_t> all; if(!rank) all.resize(static_cast<size_t>(n)*n); MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, rank?nullptr:all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD); if(!rank) print_results(all, "ElementEnergy"); }
    d.destroy(); MPI_Finalize(); return 0;
}
