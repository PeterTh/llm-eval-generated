#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy, total_flux; };

#define CUDA(call) do { cudaError_t cuda_status_ = (call); if (cuda_status_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_status_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Rows 1..rows are owned; rows 0 and rows+1 contain neighboring ranks' halos.
__global__ void update(const double* energy, const double* flux, double* next,
                       double* next_flux, int n, int rows, int first_row,
                       int begin, int end) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = size_t(end - begin) * n;
    if (k >= count) return;
    int row = begin + int(k / n), y = int(k % n);
    int x = first_row + row - 1;
    size_t p = size_t(row) * n + y;
    double e = energy[p];
    // The original assigns the corners in sequence, including the n=1 case.
    double total = 0.0;
    if ((x == 0 && y == 0) || (x == n-1 && y == n-1)) total = 0.5;
    else if ((x == 0 && y == n-1) || (x == n-1 && y == 0)) total = -0.5;
    if (x + 1 < n) total += (energy[p + n] - e) * 0.8 * 1.0 * 0.25;
    if (x > 0)     total += (energy[p - n] - e) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) total += (energy[p + 1] - e) * 0.8 * 1.0 * 0.25;
    if (y > 0)     total += (energy[p - 1] - e) * 0.8 * 1.0 * 0.25;
    next[p] = e + total;
    next_flux[p] = flux[p] + fabs(total);
    (void)rows;
}

void launch(const double* e, const double* f, double* ne, double* nf,
            int n, int rows, int first, int begin, int end) {
    if (begin >= end) return;
    size_t count = size_t(end - begin) * n;
    update<<<unsigned((count + 255) / 256), 256>>>(e, f, ne, nf, n, rows, first, begin, end);
    CUDA(cudaGetLastError());
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Grid size (NxN elements) (default: 512)\n  -i <num>     Number of simulation iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, iters = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i+1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n < 1 || iters < 0 || size_t(n)*n > INT_MAX) {
        if (!rank) fprintf(stderr, "Grid size must be positive, iterations nonnegative, and element count at most INT_MAX\n");
        MPI_Finalize(); return 1;
    }
    // Excess ranks are excluded so every participating rank owns at least one row.
    int active = std::min(size, n);
    MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) { MPI_Finalize(); return 0; }
    int rows = n / active + (rank < n % active);
    int first = rank * (n / active) + std::min(rank, n % active);
    MPI_Comm local; MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices));
    size_t cells = size_t(rows + 2) * n, bytes = cells * sizeof(double), rowbytes = size_t(n)*sizeof(double);
    double *e, *f, *ne, *nf, *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA(cudaMalloc(&e, bytes)); CUDA(cudaMalloc(&f, bytes)); CUDA(cudaMalloc(&ne, bytes)); CUDA(cudaMalloc(&nf, bytes));
    CUDA(cudaMemset(e, 0, bytes)); CUDA(cudaMemset(f, 0, bytes));
    CUDA(cudaMallocHost(&send_top, rowbytes)); CUDA(cudaMallocHost(&send_bottom, rowbytes));
    CUDA(cudaMallocHost(&recv_top, rowbytes)); CUDA(cudaMallocHost(&recv_bottom, rowbytes));
    cudaStream_t transfer; CUDA(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", n,n,n*n,iters,validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\nRunning simulation...\n");
    }
    MPI_Barrier(comm); double start = MPI_Wtime();
    int up = rank ? rank-1 : MPI_PROC_NULL, down = rank+1 < active ? rank+1 : MPI_PROC_NULL;
    for (int iter = 0; iter < iters; ++iter) {
        // The preceding update must complete before its output is read for halos.
        CUDA(cudaDeviceSynchronize());
        if (up != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_top, e+n, rowbytes, cudaMemcpyDeviceToHost, transfer));
        if (down != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(send_bottom, e+size_t(rows)*n, rowbytes, cudaMemcpyDeviceToHost, transfer));
        launch(e,f,ne,nf,n,rows,first,2,rows); // independent of incoming halos
        CUDA(cudaStreamSynchronize(transfer));
        MPI_Sendrecv(send_top, n, MPI_DOUBLE, up, 0, recv_bottom, n, MPI_DOUBLE, down, 0, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom, n, MPI_DOUBLE, down, 1, recv_top, n, MPI_DOUBLE, up, 1, comm, MPI_STATUS_IGNORE);
        if (up != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(e, recv_top, rowbytes, cudaMemcpyHostToDevice, transfer));
        if (down != MPI_PROC_NULL) CUDA(cudaMemcpyAsync(e+size_t(rows+1)*n, recv_bottom, rowbytes, cudaMemcpyHostToDevice, transfer));
        CUDA(cudaStreamSynchronize(transfer));
        launch(e,f,ne,nf,n,rows,first,1,2);
        if (rows > 1) launch(e,f,ne,nf,n,rows,first,rows,rows+1);
        std::swap(e,ne); std::swap(f,nf);
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime()-start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    std::vector<double> local_energy(size_t(rows)*n), local_flux(size_t(rows)*n);
    CUDA(cudaMemcpy(local_energy.data(), e+n, local_energy.size()*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(local_flux.data(), f+n, local_flux.size()*sizeof(double), cudaMemcpyDeviceToHost));
    uint64_t local_hash = 0, hash = 0;
    #pragma omp parallel for reduction(^:local_hash) schedule(static)
    for (size_t i=0; i<local_energy.size(); ++i) {
        uint64_t a,b; memcpy(&a,&local_energy[i],sizeof(a)); memcpy(&b,&local_flux[i],sizeof(b));
        uint64_t global_i = size_t(first)*n+i;
        local_hash ^= (a+global_i)*0x9e3779b97f4a7c15ULL;
        local_hash ^= (b+global_i)*0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    std::vector<int> counts(active), displs(active);
    for (int r=0; r<active; ++r) { counts[r]=(n/active+(r<n%active))*n; displs[r]=(r*(n/active)+std::min(r,n%active))*n; }
    std::vector<double> energies, fluxes;
    if (!rank && (results || validate)) energies.resize(size_t(n)*n);
    if (!rank && validate) fluxes.resize(size_t(n)*n);
    if (results || validate) MPI_Gatherv(local_energy.data(), counts[rank], MPI_DOUBLE, energies.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    if (validate) MPI_Gatherv(local_flux.data(), counts[rank], MPI_DOUBLE, fluxes.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    int valid = 1;
    if (!rank) {
        long duration_ms = long(max_elapsed*1000); int measured = std::max(iters-1,1);
        double rate = measured * double(n)*n / max_elapsed / 1e9;
        printf("Computation time: %ld ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", duration_ms, max_elapsed*1000/measured, rate, rate*22);
        printf("  Result hash: %016lX\n\n", (unsigned long)hash);
        if (results) print_results(energies, "ElementEnergy");
        if (validate) {
            double sum=0, flux_sum=0, lo=std::numeric_limits<double>::max(), hi=std::numeric_limits<double>::lowest();
            // Preserve the original order of the validation sums.
            for (size_t i=0; i<energies.size(); ++i) { sum+=energies[i]; flux_sum+=fluxes[i]; lo=std::min(lo,energies[i]); hi=std::max(hi,energies[i]); }
            printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",sum,flux_sum,lo,hi);
            if (!std::isfinite(sum)) { printf("  ERROR: Energy sum is not finite\n"); valid=0; }
            if (std::abs(sum)>1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); valid=0; }
            if (!std::isfinite(lo) || !std::isfinite(hi)) { printf("  ERROR: Energy extrema are not finite\n"); valid=0; }
            if (valid) printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid,1,MPI_INT,0,comm);
    CUDA(cudaStreamDestroy(transfer)); CUDA(cudaFreeHost(send_top)); CUDA(cudaFreeHost(send_bottom)); CUDA(cudaFreeHost(recv_top)); CUDA(cudaFreeHost(recv_bottom));
    CUDA(cudaFree(e)); CUDA(cudaFree(f)); CUDA(cudaFree(ne)); CUDA(cudaFree(nf));
    MPI_Comm_free(&comm); MPI_Finalize(); return valid ? 0 : 1;
}
