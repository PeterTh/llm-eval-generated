#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 8;

struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy, total_flux; };

constexpr idx_t DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                  \
        MPI_Abort(MPI_COMM_WORLD, 2);                                           \
    }                                                                           \
} while (0)

struct Partition {
    int first_row, rows, n;
    size_t owned() const { return static_cast<size_t>(rows) * n; }
    size_t storage() const { return static_cast<size_t>(rows + 2) * n; }
};

static Partition partitionFor(int n, int rank, int ranks) {
    const int base = n / ranks, extra = n % ranks;
    const int rows = base + (rank < extra);
    const int first = rank * base + std::min(rank, extra);
    return {first, rows, n};
}

static void buildLocalMesh(std::vector<ElementStatic>& elems, const Partition& p) {
    elems.resize(p.owned());
    #pragma omp parallel for schedule(static)
    for (int lr = 0; lr < p.rows; ++lr) {
        const int gr = p.first_row + lr;
        for (int col = 0; col < p.n; ++col) {
            ElementStatic e{};
            e.material_idx = ((gr == 0 || gr == p.n - 1) &&
                              (col == 0 || col == p.n - 1))
                ? ((gr == col) ? INFLOW_MAT_ID : OUTFLOW_MAT_ID)
                : DEFAULT_MAT_ID;
            const int local_row = lr + 1; // row zero and rows+1 are halos
            if (gr + 1 < p.n) { e.connected_idx[e.num_connections] = static_cast<idx_t>(local_row + 1) * p.n + col; e.connected_flux[e.num_connections++] = 1.0; }
            if (gr > 0)       { e.connected_idx[e.num_connections] = static_cast<idx_t>(local_row - 1) * p.n + col; e.connected_flux[e.num_connections++] = 1.0; }
            if (col + 1 < p.n){ e.connected_idx[e.num_connections] = static_cast<idx_t>(local_row) * p.n + col + 1; e.connected_flux[e.num_connections++] = 1.0; }
            if (col > 0)      { e.connected_idx[e.num_connections] = static_cast<idx_t>(local_row) * p.n + col - 1; e.connected_flux[e.num_connections++] = 1.0; }
            elems[static_cast<size_t>(lr) * p.n + col] = e;
        }
    }
}

__global__ void updateKernel(const ElementStatic* __restrict__ stat,
                             const ElementDynamic* __restrict__ current,
                             ElementDynamic* __restrict__ next,
                             const Material* __restrict__ materials,
                             size_t begin, size_t end, int n) {
    const size_t owned_i = begin + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (owned_i >= end) return;
    const ElementStatic e = stat[owned_i];
    const size_t local_i = owned_i + n;
    const ElementDynamic self = current[local_i];
    const Material mat = materials[e.material_idx];
    val_t flux = mat.external_flow;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j < static_cast<int>(e.num_connections)) {
            flux += (current[e.connected_idx[j]].current_energy - self.current_energy) *
                    mat.transfer_coeff * e.connected_flux[j] * 0.25;
        }
    }
    next[local_i].current_energy = self.current_energy + flux;
    next[local_i].total_flux = self.total_flux + fabs(flux);
}

static void launchRange(const ElementStatic* s, const ElementDynamic* in,
                        ElementDynamic* out, const Material* m, size_t begin,
                        size_t end, int n, cudaStream_t stream) {
    if (end <= begin) return;
    constexpr int block = 256;
    const unsigned grid = static_cast<unsigned>((end - begin + block - 1) / block);
    updateKernel<<<grid, block, 0, stream>>>(s, in, out, m, begin, end, n);
}

static void runSimulation(const Partition& p, const std::vector<ElementStatic>& stat,
                          std::vector<ElementDynamic>& result, int iters,
                          int rank, int ranks) {
    ElementStatic* ds = nullptr;
    ElementDynamic *da = nullptr, *db = nullptr;
    Material* dm = nullptr;
    ElementDynamic *send_top = nullptr, *send_bottom = nullptr;
    ElementDynamic *recv_top = nullptr, *recv_bottom = nullptr;
    const Material mats[3]{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t row_bytes = static_cast<size_t>(p.n) * sizeof(ElementDynamic);

    CUDA_CHECK(cudaMalloc(&ds, stat.size() * sizeof(ElementStatic)));
    CUDA_CHECK(cudaMalloc(&da, p.storage() * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&db, p.storage() * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMalloc(&dm, sizeof(mats)));
    CUDA_CHECK(cudaMemset(da, 0, p.storage() * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemset(db, 0, p.storage() * sizeof(ElementDynamic)));
    CUDA_CHECK(cudaMemcpy(ds, stat.data(), stat.size() * sizeof(ElementStatic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm, mats, sizeof(mats), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost(&send_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&send_bottom, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_top, row_bytes));
    CUDA_CHECK(cudaMallocHost(&recv_bottom, row_bytes));

    cudaStream_t compute, transfer;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    const int up = rank ? rank - 1 : MPI_PROC_NULL;
    const int down = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;

    for (int iter = 0; iter < iters; ++iter) {
        CUDA_CHECK(cudaMemcpyAsync(send_top, da + p.n, row_bytes, cudaMemcpyDeviceToHost, transfer));
        CUDA_CHECK(cudaMemcpyAsync(send_bottom, da + static_cast<size_t>(p.rows) * p.n,
                                   row_bytes, cudaMemcpyDeviceToHost, transfer));
        if (p.rows > 2)
            launchRange(ds, da, db, dm, p.n, static_cast<size_t>(p.rows - 1) * p.n, p.n, compute);
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        MPI_Sendrecv(send_top, static_cast<int>(row_bytes), MPI_BYTE, up, 10,
                     recv_bottom, static_cast<int>(row_bytes), MPI_BYTE, down, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bottom, static_cast<int>(row_bytes), MPI_BYTE, down, 11,
                     recv_top, static_cast<int>(row_bytes), MPI_BYTE, up, 11,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(da, recv_top, row_bytes, cudaMemcpyHostToDevice, transfer));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(da + static_cast<size_t>(p.rows + 1) * p.n,
                                       recv_bottom, row_bytes, cudaMemcpyHostToDevice, transfer));
        CUDA_CHECK(cudaStreamSynchronize(transfer));
        launchRange(ds, da, db, dm, 0, p.n, p.n, compute);
        if (p.rows > 1)
            launchRange(ds, da, db, dm, static_cast<size_t>(p.rows - 1) * p.n,
                        static_cast<size_t>(p.rows) * p.n, p.n, compute);
        CUDA_CHECK(cudaStreamSynchronize(compute));
        std::swap(da, db);
    }
    result.resize(p.owned());
    CUDA_CHECK(cudaMemcpy(result.data(), da + p.n, p.owned() * sizeof(ElementDynamic), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamDestroy(compute)); CUDA_CHECK(cudaStreamDestroy(transfer));
    CUDA_CHECK(cudaFreeHost(send_top)); CUDA_CHECK(cudaFreeHost(send_bottom));
    CUDA_CHECK(cudaFreeHost(recv_top)); CUDA_CHECK(cudaFreeHost(recv_bottom));
    CUDA_CHECK(cudaFree(ds)); CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dm));
}

static bool validateResults(const std::vector<ElementDynamic>& local, int rank) {
    val_t es = 0.0, fs = 0.0, emin = std::numeric_limits<val_t>::max(), emax = std::numeric_limits<val_t>::lowest();
    #pragma omp parallel for reduction(+:es,fs) reduction(min:emin) reduction(max:emax) schedule(static)
    for (size_t i = 0; i < local.size(); ++i) {
        es += local[i].current_energy; fs += local[i].total_flux;
        emin = std::min(emin, local[i].current_energy); emax = std::max(emax, local[i].current_energy);
    }
    val_t ges, gfs, gmin, gmax;
    MPI_Reduce(&es, &ges, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&fs, &gfs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&emin, &gmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&emax, &gmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", ges, gfs, gmin, gmax);
        valid = std::isfinite(ges) && std::isfinite(gfs) && std::isfinite(gmin) && std::isfinite(gmax);
        if (std::abs(ges) > 1e-8) std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
}

static uint64_t computeLocalHash(const std::vector<ElementDynamic>& a, size_t offset) {
    uint64_t hash = 0;
    #pragma omp parallel for reduction(^:hash) schedule(static)
    for (size_t i = 0; i < a.size(); ++i) {
        uint64_t e, f; std::memcpy(&e, &a[i].current_energy, sizeof(e)); std::memcpy(&f, &a[i].total_flux, sizeof(f));
        const size_t gi = offset + i;
        hash ^= (e + gi) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + gi) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num> Grid size (default: 512)\n  -i <num> Iterations (default: 10)\n  -v Validate\n  -r Print results\n  -h Help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iters = 10; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || ranks > n) {
        if (!rank) std::fprintf(stderr, "Grid size must be positive, iterations nonnegative, and MPI ranks <= grid rows.\n");
        MPI_Finalize(); return 1;
    }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "CUDA device required.\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    int local_rank = 0; MPI_Comm local_comm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank); CUDA_CHECK(cudaSetDevice(local_rank % devices)); MPI_Comm_free(&local_comm);

    const Partition p = partitionFor(n, rank, ranks);
    if (!rank) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        std::printf("Grid size: %d x %d = %lld elements\nIterations: %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n\nBuilding unstructured mesh...\n",
                    n, n, static_cast<long long>(n) * n, iters, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
    }
    std::vector<ElementStatic> stat; buildLocalMesh(stat, p);
    const unsigned long long local_mem = stat.size() * sizeof(ElementStatic) + p.storage() * sizeof(ElementDynamic) * 2;
    unsigned long long total_mem = 0; MPI_Reduce(&local_mem, &total_mem, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Aggregate device data: %.2f MB\n\nRunning simulation...\n", total_mem / (1024.0 * 1024.0));
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    std::vector<ElementDynamic> local; runSimulation(p, stat, local, iters, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed_local = MPI_Wtime() - start, elapsed = [&]{ double x; MPI_Reduce(&elapsed_local, &x, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return x; }();
    const uint64_t lh = computeLocalHash(local, static_cast<size_t>(p.first_row) * n); uint64_t hash = 0;
    MPI_Reduce(&lh, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) {
        const double ms = elapsed * 1000.0, measured = std::max(iters - 1, 1);
        const double geps = (measured * static_cast<double>(n) * n) / elapsed / 1e9;
        std::printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", ms, ms / measured, geps, geps * 22.0, static_cast<unsigned long long>(hash));
    }
    if (printResults) {
        std::vector<int> counts(ranks), displs(ranks); for (int r = 0; r < ranks; ++r) { auto q = partitionFor(n, r, ranks); counts[r] = static_cast<int>(q.owned()); displs[r] = q.first_row * n; }
        std::vector<double> energies(rank == 0 ? static_cast<size_t>(n) * n : 0), le(local.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < local.size(); ++i) le[i] = local[i].current_energy;
        MPI_Gatherv(le.data(), static_cast<int>(le.size()), MPI_DOUBLE, energies.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(energies, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(local, rank);
    MPI_Finalize(); return valid ? 0 : 1;
}
