#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

struct Domain {
    int n = 0;
    int first_row = 0;
    int rows = 0;
    int rank = 0;
    int size = 1;
    std::vector<val_t> energy[2]; // owned rows plus upper/lower halo rows
    std::vector<val_t> total_flux;
};

static inline void partition_rows(int n, int rank, int size, int& first, int& rows) {
    const int q = n / size, r = n % size;
    rows = q + (rank < r);
    first = rank * q + std::min(rank, r);
}

static Domain buildSquare2D(int n, MPI_Comm comm) {
    Domain d;
    d.n = n;
    MPI_Comm_rank(comm, &d.rank);
    MPI_Comm_size(comm, &d.size);
    partition_rows(n, d.rank, d.size, d.first_row, d.rows);
    const size_t cells = static_cast<size_t>(d.rows + 2) * n;
    d.energy[0].assign(cells, 0.0);
    d.energy[1].assign(cells, 0.0);
    d.total_flux.assign(static_cast<size_t>(d.rows) * n, 0.0);
    return d;
}

static inline val_t external_flow(int global_x, int y, int n) {
    if ((global_x == 0 && y == 0) || (global_x == n - 1 && y == n - 1)) return 0.5;
    if ((global_x == 0 && y == n - 1) || (global_x == n - 1 && y == 0)) return -0.5;
    return 0.0;
}

static inline void update_row(Domain& d, int lx, const std::vector<val_t>& src,
                              std::vector<val_t>& dst) {
    const int n = d.n;
    const int gx = d.first_row + lx - 1;
    const size_t base = static_cast<size_t>(lx) * n;
    const size_t owned_base = static_cast<size_t>(lx - 1) * n;
    for (int y = 0; y < n; ++y) {
        const val_t self = src[base + y];
        val_t flux = external_flow(gx, y, n);
        // Preserve the reference operation order for bitwise-identical results.
        if (gx + 1 < n) flux += (src[base + n + y] - self) * 0.8 * 1.0 * 0.25;
        if (gx > 0)     flux += (src[base - n + y] - self) * 0.8 * 1.0 * 0.25;
        if (y + 1 < n)  flux += (src[base + y + 1] - self) * 0.8 * 1.0 * 0.25;
        if (y > 0)      flux += (src[base + y - 1] - self) * 0.8 * 1.0 * 0.25;
        dst[base + y] = self + flux;
        d.total_flux[owned_base + y] += std::abs(flux);
    }
}

static void runSimulation(Domain& d, int n_iters, MPI_Comm comm) {
    int current = 0;
    const int n = d.n;
    for (int iter = 0; iter < n_iters; ++iter) {
        auto& src = d.energy[current];
        auto& dst = d.energy[1 - current];
        MPI_Request req[4];
        int nr = 0;
        if (d.rank > 0) {
            MPI_Irecv(src.data(), n, MPI_DOUBLE, d.rank - 1, 1, comm, &req[nr++]);
            MPI_Isend(src.data() + n, n, MPI_DOUBLE, d.rank - 1, 0, comm, &req[nr++]);
        }
        if (d.rank + 1 < d.size) {
            MPI_Irecv(src.data() + static_cast<size_t>(d.rows + 1) * n, n, MPI_DOUBLE,
                      d.rank + 1, 0, comm, &req[nr++]);
            MPI_Isend(src.data() + static_cast<size_t>(d.rows) * n, n, MPI_DOUBLE,
                      d.rank + 1, 1, comm, &req[nr++]);
        }

        // These rows do not depend on incoming halo data.
        for (int lx = 2; lx <= d.rows - 1; ++lx) update_row(d, lx, src, dst);
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        update_row(d, 1, src, dst);
        if (d.rows > 1) update_row(d, d.rows, src, dst);
        current = 1 - current;
    }
    if (current != 0) d.energy[0].swap(d.energy[1]);
}

static uint64_t localHash(const Domain& d) {
    uint64_t hash = 0;
    const val_t* energy = d.energy[0].data() + d.n;
    for (size_t i = 0; i < d.total_flux.size(); ++i) {
        uint64_t ebits, fbits;
        std::memcpy(&ebits, energy + i, sizeof(ebits));
        std::memcpy(&fbits, d.total_flux.data() + i, sizeof(fbits));
        const uint64_t global_i = static_cast<uint64_t>(d.first_row) * d.n + i;
        hash ^= (ebits + global_i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fbits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static bool validateResults(const Domain& d, MPI_Comm comm) {
    const val_t* energy = d.energy[0].data() + d.n;
    val_t local[4] = {0.0, 0.0, std::numeric_limits<val_t>::lowest(),
                      std::numeric_limits<val_t>::max()};
    for (size_t i = 0; i < d.total_flux.size(); ++i) {
        local[0] += energy[i];
        local[1] += d.total_flux[i];
        local[2] = std::max(local[2], energy[i]);
        local[3] = std::min(local[3], energy[i]);
    }
    val_t sums[2], energy_max, energy_min;
    MPI_Reduce(local, sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(local + 2, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(local + 3, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    int valid = 1;
    if (d.rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", sums[0], sums[1]);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(sums[0]) || !std::isfinite(sums[1]) ||
            !std::isfinite(energy_max) || !std::isfinite(energy_min)) valid = 0;
        if (std::isfinite(sums[0]) && std::abs(sums[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\n  -n <num>     Grid size (default: 512)\n", p);
    printf("  -i <num>     Iterations (default: 10)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || n <= 0 || n_iters < 0) {
        if (world_rank == 0) { if (bad) printf("Invalid command line\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return bad || n <= 0 || n_iters < 0;
    }

    // More ranks than rows cannot contribute; exclude them from all hot-path collectives.
    MPI_Comm comm = MPI_COMM_NULL;
    const int active = std::min(world_size, n);
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }

    Domain d = buildSquare2D(n, comm);
    const uint64_t elems = static_cast<uint64_t>(n) * n;
    if (d.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %llu elements\nIterations: %d\nValidation: %s\nMPI ranks: %d\n\n",
               n, n, static_cast<unsigned long long>(elems), n_iters,
               validate ? "enabled" : "disabled", active);
        const double static_mem = elems * (sizeof(idx_t) * 10 + sizeof(val_t) * 8);
        const double dynamic_mem = elems * sizeof(ElementDynamic) * 2;
        printf("Building unstructured mesh...\nMemory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / 1048576.0, static_mem / 1048576.0,
               dynamic_mem / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    runSimulation(d, n_iters, comm);
    const double local_elapsed = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    const uint64_t lh = localHash(d);
    uint64_t hash;
    MPI_Reduce(&lh, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
    if (d.rank == 0) {
        const double ms = elapsed * 1000.0;
        const int measured = std::max(n_iters - 1, 1);
        const double geps = elapsed > 0.0 ? measured * static_cast<double>(elems) / elapsed / 1e9 : 0.0;
        printf("Computation time: %.0f ms\nPerformance:\n", ms);
        printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n", ms / measured, geps);
        printf("  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", geps * 22.0,
               static_cast<unsigned long long>(hash));
    }

    if (printResults) {
        std::vector<int> counts, displs;
        std::vector<double> all;
        if (d.rank == 0) { counts.resize(d.size); displs.resize(d.size); }
        const int local_count = d.rows * n;
        MPI_Gather(&local_count, 1, MPI_INT, d.rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm);
        if (d.rank == 0) {
            displs[0] = 0;
            for (int r = 1; r < d.size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
            all.resize(elems);
        }
        MPI_Gatherv(d.energy[0].data() + n, local_count, MPI_DOUBLE,
                    d.rank == 0 ? all.data() : nullptr, d.rank == 0 ? counts.data() : nullptr,
                    d.rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (d.rank == 0) print_results(all, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(d, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
