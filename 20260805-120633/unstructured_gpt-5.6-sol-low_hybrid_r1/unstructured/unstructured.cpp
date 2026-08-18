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
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using val_t = double;

struct ElementDynamic { val_t current_energy, total_flux; };

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

struct Decomposition {
    int first_row, rows;
};

static Decomposition decompose(int n, int rank, int size) {
    const int q = n / size, r = n % size;
    return {rank * q + std::min(rank, r), q + (rank < r)};
}

// The two ghost rows are stored before and after the owned rows.  Computing
// directly from this compact representation avoids transferring the mesh's
// invariant connectivity arrays to every accelerator.
__global__ void updateKernel(const ElementDynamic* __restrict__ in,
                             ElementDynamic* __restrict__ out,
                             int n, int first_row, int rows) {
    const size_t count = static_cast<size_t>(rows) * n;
    for (size_t p = blockIdx.x * blockDim.x + threadIdx.x;
         p < count; p += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int lr = static_cast<int>(p / n) + 1;
        const int y = static_cast<int>(p % n);
        const int x = first_row + lr - 1;
        const size_t i = static_cast<size_t>(lr) * n + y;
        const double e = in[i].current_energy;
        double flux = ((x == 0 || x == n - 1) && (y == 0 || y == n - 1))
                    ? (((x == y) || (x + y == 2 * (n - 1))) ? .5 : -.5) : 0.0;
        // Preserve the reference connection order: +x, -x, +y, -y.
        if (x + 1 < n) flux += (in[i + n].current_energy - e) * .2;
        if (x > 0)     flux += (in[i - n].current_energy - e) * .2;
        if (y + 1 < n) flux += (in[i + 1].current_energy - e) * .2;
        if (y > 0)     flux += (in[i - 1].current_energy - e) * .2;
        out[i].current_energy = e + flux;
        out[i].total_flux = in[i].total_flux + fabs(flux);
    }
}

static void exchangeHalos(ElementDynamic* send_top, ElementDynamic* send_bottom,
                          ElementDynamic* recv_top, ElementDynamic* recv_bottom,
                          int n, int rank, int size) {
    MPI_Request req[4]; int nr = 0;
    const size_t bytes = static_cast<size_t>(n) * sizeof(ElementDynamic);
    if (rank > 0) {
        MPI_Irecv(recv_top, static_cast<int>(bytes), MPI_BYTE, rank - 1, 11, MPI_COMM_WORLD, &req[nr++]);
        MPI_Isend(send_top, static_cast<int>(bytes), MPI_BYTE, rank - 1, 12, MPI_COMM_WORLD, &req[nr++]);
    }
    if (rank + 1 < size) {
        MPI_Irecv(recv_bottom, static_cast<int>(bytes), MPI_BYTE,
                  rank + 1, 12, MPI_COMM_WORLD, &req[nr++]);
        MPI_Isend(send_bottom, static_cast<int>(bytes), MPI_BYTE,
                  rank + 1, 11, MPI_COMM_WORLD, &req[nr++]);
    }
    if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
}

static std::vector<ElementDynamic> runSimulation(int n, int iters, Decomposition part,
                                                  int rank, int size) {
    const size_t padded = static_cast<size_t>(part.rows + 2) * n;
    ElementDynamic *a, *b;
    CUDA_OK(cudaMalloc(&a, padded * sizeof(*a)));
    CUDA_OK(cudaMalloc(&b, padded * sizeof(*b)));
    CUDA_OK(cudaMemset(a, 0, padded * sizeof(*a)));
    CUDA_OK(cudaMemset(b, 0, padded * sizeof(*b)));
    ElementDynamic* halo = nullptr;
    CUDA_OK(cudaMallocHost(&halo, static_cast<size_t>(4) * n * sizeof(*halo)));
    ElementDynamic *send_top=halo, *send_bottom=halo+n, *recv_top=halo+2*n, *recv_bottom=halo+3*n;
    const int threads = 256;
    const size_t owned = static_cast<size_t>(part.rows) * n;
    const int blocks = std::min<size_t>((owned + threads - 1) / threads, 65535);
    for (int iter = 0; iter < iters; ++iter) {
        const size_t row_bytes = static_cast<size_t>(n) * sizeof(*a);
        if (rank > 0) CUDA_OK(cudaMemcpyAsync(send_top, a+n, row_bytes, cudaMemcpyDeviceToHost));
        if (rank+1 < size) CUDA_OK(cudaMemcpyAsync(send_bottom, a+static_cast<size_t>(part.rows)*n, row_bytes, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaStreamSynchronize(nullptr));
        exchangeHalos(send_top, send_bottom, recv_top, recv_bottom, n, rank, size);
        if (rank > 0) CUDA_OK(cudaMemcpyAsync(a, recv_top, row_bytes, cudaMemcpyHostToDevice));
        if (rank+1 < size) CUDA_OK(cudaMemcpyAsync(a+static_cast<size_t>(part.rows+1)*n, recv_bottom, row_bytes, cudaMemcpyHostToDevice));
        if (blocks) updateKernel<<<blocks, threads>>>(a, b, n, part.first_row, part.rows);
        CUDA_OK(cudaGetLastError());
        std::swap(a, b);
    }
    std::vector<ElementDynamic> local(owned);
    CUDA_OK(cudaMemcpy(local.data(), a+n, owned*sizeof(*a), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFreeHost(halo));
    CUDA_OK(cudaFree(a)); CUDA_OK(cudaFree(b));
    return local;
}

static bool validateResults(const std::vector<ElementDynamic>& v) {
    double es = 0, fs = 0, lo = std::numeric_limits<double>::max();
    double hi = std::numeric_limits<double>::lowest();
    #pragma omp parallel for reduction(+:es,fs) reduction(min:lo) reduction(max:hi)
    for (size_t i = 0; i < v.size(); ++i) {
        es += v[i].current_energy; fs += v[i].total_flux;
        lo = std::min(lo, v[i].current_energy); hi = std::max(hi, v[i].current_energy);
    }
    std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", es, fs, lo, hi);
    const bool ok = std::isfinite(es) && std::isfinite(fs) && std::isfinite(lo) && std::isfinite(hi);
    if (std::abs(es) > 1e-8) std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    std::printf("  Validation: %s\n", ok ? "PASSED" : "FAILED");
    return ok;
}

static uint64_t computeHash(const std::vector<ElementDynamic>& v) {
    uint64_t h = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        uint64_t e, f; std::memcpy(&e, &v[i].current_energy, 8); std::memcpy(&f, &v[i].total_flux, 8);
        h ^= (e + i) * 0x9e3779b97f4a7c15ULL; h ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

static void usage(const char* p) { std::printf("Usage: %s [options]\n  -n <num>  Grid size (default: 512)\n  -i <num>  Iterations (default: 10)\n  -v        Validate\n  -r        Print results\n  -h        Help\n", p); }

int main(int argc, char** argv) {
    int provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank); MPI_Comm_free(&local_comm);
    int devices = 0; CUDA_OK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_OK(cudaSetDevice(local_rank % devices));

    int n = 512, iters = 10; bool validate = false, results = false, bad = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad) { if (!rank) usage(argv[0]); MPI_Finalize(); return bad; }
    if (n <= 0 || iters < 0 || size > n) { if (!rank) std::fprintf(stderr, "Require n > 0, iterations >= 0, and MPI ranks <= n\n"); MPI_Finalize(); return 1; }
    const Decomposition part = decompose(n, rank, size);
    if (!rank) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n\nBuilding unstructured mesh...\n",
                    n, n, 1LL*n*n, iters, validate ? "enabled" : "disabled", size, omp_get_max_threads());
        const size_t static_mem = static_cast<size_t>(n)*n*(2*8 + 8 + 8*8 + 8*8);
        const size_t dynamic_mem = static_cast<size_t>(n)*n*sizeof(ElementDynamic)*2;
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\nRunning simulation...\n",
                    (static_mem+dynamic_mem)/1048576., static_mem/1048576., dynamic_mem/1048576.);
    }
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    auto local = runSimulation(n, iters, part, rank, size);
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed = MPI_Wtime() - start;

    std::vector<int> counts(size), displs(size);
    #pragma omp parallel for schedule(static)
    for (int r=0; r<size; ++r) { auto d=decompose(n,r,size); counts[r]=d.rows*n*sizeof(ElementDynamic); displs[r]=d.first_row*n*sizeof(ElementDynamic); }
    std::vector<ElementDynamic> all; if (!rank) all.resize(static_cast<size_t>(n)*n);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()*sizeof(ElementDynamic)), MPI_BYTE,
                rank ? nullptr : all.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    int rc = 0;
    if (!rank) {
        const double ms = elapsed*1000., measured=std::max(iters-1,1), ge=(measured*n*n)/elapsed/1e9;
        std::printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n",
                    ms, ms/measured, ge, ge*22., (unsigned long)computeHash(all));
        if (results) {
            std::vector<double> e(all.size());
            #pragma omp parallel for schedule(static)
            for (size_t i=0;i<all.size();++i) e[i]=all[i].current_energy;
            print_results(e,"ElementEnergy");
        }
        if (validate && !validateResults(all)) rc=1;
    }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
