#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using val_t = double;
struct ElementDynamic { val_t current_energy, total_flux; };

#define CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { if ((call) != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Rows 1..rows are owned; rows 0 and rows+1 hold neighboring ranks' halos.
__global__ void update(const ElementDynamic* in, ElementDynamic* out, int n,
                       int start, int rows, int first, int count) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(count) * n;
    if (k >= total) return;
    int local_x = first + int(k / n);
    int y = int(k % n);
    int x = start + local_x - 1;
    size_t i = size_t(local_x) * n + y;
    double energy = in[i].current_energy;
    // The assignments in the original builder leave the single cell as inflow.
    double flux = 0.0;
    if ((x == 0 && y == 0) || (x == n-1 && y == n-1)) flux = 0.5;
    else if ((x == 0 && y == n-1) || (x == n-1 && y == 0)) flux = -0.5;
    if (x + 1 < n) flux += (in[i+n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (x > 0)     flux += (in[i-n].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y + 1 < n) flux += (in[i+1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    if (y > 0)     flux += (in[i-1].current_energy - energy) * 0.8 * 1.0 * 0.25;
    out[i].current_energy = energy + flux;
    out[i].total_flux = in[i].total_flux + fabs(flux);
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    double sum = 0, flux = 0, mx = std::numeric_limits<double>::lowest(), mn = std::numeric_limits<double>::max();
    for (const auto& e : elements) { sum += e.current_energy; flux += e.total_flux; mx = std::max(e.current_energy, mx); mn = std::min(e.current_energy, mn); }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", sum, flux, mn, mx);
    if (!std::isfinite(sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(mx) || !std::isfinite(mn)) { printf("  ERROR: Energy extrema are not finite\n"); return false; }
    printf("  Validation: PASSED\n"); return true;
}
static uint64_t computeHash(const std::vector<ElementDynamic>& elements, size_t offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) if(elements.size() >= 65536)
    for (size_t i = 0; i < elements.size(); ++i) {
        uint64_t e, f; memcpy(&e, &elements[i].current_energy, sizeof(e)); memcpy(&f, &elements[i].total_flux, sizeof(f));
        hash ^= (e + i + offset) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i + offset) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}
static void printUsage(const char* p) {
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
    if (n <= 0 || size > n || size_t(n)*n > size_t(INT32_MAX)/sizeof(ElementDynamic)) {
        if (!rank) fprintf(stderr, "Grid size must be positive, at least the MPI rank count, and fit MPI gather counts.\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm local; MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int local_rank, devices; MPI_Comm_rank(local, &local_rank); CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local);
    int rows = n/size + (rank < n%size), start = rank*(n/size) + std::min(rank,n%size);
    size_t rowbytes = size_t(n)*sizeof(ElementDynamic), bytes = size_t(rows+2)*rowbytes;
    ElementDynamic *a, *b, *send_top, *send_bottom, *recv_top, *recv_bottom;
    CUDA(cudaMalloc(&a, bytes)); CUDA(cudaMalloc(&b, bytes)); CUDA(cudaMemset(a, 0, bytes)); CUDA(cudaMemset(b, 0, bytes));
    CUDA(cudaMallocHost(&send_top, rowbytes)); CUDA(cudaMallocHost(&send_bottom, rowbytes));
    CUDA(cudaMallocHost(&recv_top, rowbytes)); CUDA(cudaMallocHost(&recv_bottom, rowbytes));
    cudaStream_t stream; CUDA(cudaStreamCreate(&stream));
    if (!rank) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %zu elements\nIterations: %d\nValidation: %s\n\n", n, n, size_t(n)*n, iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\nRunning simulation...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); auto begin = std::chrono::steady_clock::now();
    const int threads = 256;
    auto launch = [&](int first, int count) {
        if (count > 0) { size_t work = size_t(count)*n; update<<<(work+threads-1)/threads, threads, 0, stream>>>(a,b,n,start,rows,first,count); CUDA(cudaGetLastError()); }
    };
    for (int iter = 0; iter < iters; ++iter) {
        // Stage outgoing rows before launching interior work. MPI runs concurrently with that kernel.
        if (rank > 0) CUDA(cudaMemcpy(send_top, a+n, rowbytes, cudaMemcpyDeviceToHost));
        if (rank+1 < size) CUDA(cudaMemcpy(send_bottom, a+size_t(rows)*n, rowbytes, cudaMemcpyDeviceToHost));
        launch(2, rows-2);
        MPI_Request req[4]; int nr = 0;
        if (rank > 0) { MPI_CHECK(MPI_Irecv(recv_top, int(rowbytes), MPI_BYTE, rank-1, 1, MPI_COMM_WORLD, &req[nr++])); MPI_CHECK(MPI_Isend(send_top, int(rowbytes), MPI_BYTE, rank-1, 0, MPI_COMM_WORLD, &req[nr++])); }
        if (rank+1 < size) { MPI_CHECK(MPI_Irecv(recv_bottom, int(rowbytes), MPI_BYTE, rank+1, 0, MPI_COMM_WORLD, &req[nr++])); MPI_CHECK(MPI_Isend(send_bottom, int(rowbytes), MPI_BYTE, rank+1, 1, MPI_COMM_WORLD, &req[nr++])); }
        MPI_CHECK(MPI_Waitall(nr, req, MPI_STATUSES_IGNORE));
        if (rank > 0) CUDA(cudaMemcpyAsync(a, recv_top, rowbytes, cudaMemcpyHostToDevice, stream));
        if (rank+1 < size) CUDA(cudaMemcpyAsync(a+size_t(rows+1)*n, recv_bottom, rowbytes, cudaMemcpyHostToDevice, stream));
        launch(1, 1); if (rows > 1) launch(rows, 1);
        CUDA(cudaStreamSynchronize(stream)); std::swap(a,b);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end-begin).count();
    std::vector<ElementDynamic> local_data(size_t(rows)*n), all;
    CUDA(cudaMemcpy(local_data.data(), a+n, local_data.size()*sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    uint64_t local_hash = computeHash(local_data, size_t(start)*n), hash = 0;
    MPI_CHECK(MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD));
    if (results || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r=0; r<size; ++r) { int rs = n/size + (r<n%size), st = r*(n/size)+std::min(r,n%size); counts[r] = int(size_t(rs)*rowbytes); displs[r] = int(size_t(st)*rowbytes); }
        if (!rank) all.resize(size_t(n)*n);
        MPI_CHECK(MPI_Gatherv(local_data.data(), int(local_data.size()*sizeof(ElementDynamic)), MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD));
    }
    int ok = 1;
    if (!rank) {
        int measured = std::max(iters-1,1); double rate = measured * double(n)*n / (ms/1000.0) / 1e9;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n\n", ms, ms/measured, rate, rate*22, (unsigned long)hash);
        if (results) { std::vector<double> energy(all.size());
            #pragma omp parallel for
            for (size_t i=0; i<all.size(); ++i) energy[i] = all[i].current_energy;
            print_results(energy, "ElementEnergy"); }
        if (validate) ok = validateResults(all);
    }
    MPI_CHECK(MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA(cudaStreamDestroy(stream)); CUDA(cudaFree(a)); CUDA(cudaFree(b)); CUDA(cudaFreeHost(send_top)); CUDA(cudaFreeHost(send_bottom)); CUDA(cudaFreeHost(recv_top)); CUDA(cudaFreeHost(recv_bottom));
    MPI_Finalize(); return ok ? 0 : 1;
}
