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

using val_t = double;

struct Domain {
    int n = 0;
    int first_row = 0;
    int rows = 0;
    int rank = 0;
    int ranks = 1;
    val_t transfer_coeff = 0.8;
    std::vector<val_t> energy[2];
    std::vector<val_t> total_flux;
};

static void decompose(int n, int rank, int ranks, int& first, int& rows) {
    const int base = n / ranks;
    const int extra = n % ranks;
    rows = base + (rank < extra);
    first = rank * base + std::min(rank, extra);
}

static Domain buildSquare2D(int n, int rank, int ranks) {
    Domain d;
    d.n = n; d.rank = rank; d.ranks = ranks;
    decompose(n, rank, ranks, d.first_row, d.rows);
    const size_t count = static_cast<size_t>(d.rows + 2) * n;
    d.energy[0].assign(count, 0.0);
    d.energy[1].assign(count, 0.0);
    d.total_flux.assign(static_cast<size_t>(d.rows) * n, 0.0);
    return d;
}

static inline val_t externalFlow(int x, int y, int n) {
    if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1))
        return (x == y) ? 0.5 : -0.5;
    return 0.0;
}

// Update owned rows [begin,end).  Ghost rows make remote and local neighbours
// use the same tight loop.  The operation order matches the serial benchmark.
static void updateRows(Domain& d, const std::vector<val_t>& src,
                       std::vector<val_t>& dst, int begin, int end) {
    const int n = d.n;
    for (int lr = begin; lr < end; ++lr) {
        const int x = d.first_row + lr;
        const size_t row = static_cast<size_t>(lr + 1) * n;
        const size_t out = static_cast<size_t>(lr) * n;
        for (int y = 0; y < n; ++y) {
            const val_t here = src[row + y];
            val_t flux = externalFlow(x, y, n);
            if (x + 1 < n) flux += (src[row + n + y] - here) * d.transfer_coeff * 1.0 * 0.25;
            if (x > 0)     flux += (src[row - n + y] - here) * d.transfer_coeff * 1.0 * 0.25;
            if (y + 1 < n) flux += (src[row + y + 1] - here) * d.transfer_coeff * 1.0 * 0.25;
            if (y > 0)     flux += (src[row + y - 1] - here) * d.transfer_coeff * 1.0 * 0.25;
            dst[row + y] = here + flux;
            d.total_flux[out + y] += std::abs(flux);
        }
    }
}

static void runSimulation(Domain& d, int iters) {
    if (d.rows == 0) return;
    const int active = std::min(d.n, d.ranks);
    const int above = d.rank > 0 ? d.rank - 1 : MPI_PROC_NULL;
    const int below = d.rank + 1 < active ? d.rank + 1 : MPI_PROC_NULL;
    int current = 0;
    for (int iter = 0; iter < iters; ++iter) {
        auto& src = d.energy[current];
        auto& dst = d.energy[current ^ 1];
        MPI_Request req[4];
        MPI_Irecv(src.data(), d.n, MPI_DOUBLE, above, 11, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(src.data() + static_cast<size_t>(d.rows + 1) * d.n,
                  d.n, MPI_DOUBLE, below, 10, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(src.data() + d.n, d.n, MPI_DOUBLE, above, 10, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(src.data() + static_cast<size_t>(d.rows) * d.n,
                  d.n, MPI_DOUBLE, below, 11, MPI_COMM_WORLD, &req[3]);

        if (d.rows > 2) updateRows(d, src, dst, 1, d.rows - 1);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        updateRows(d, src, dst, 0, 1);
        if (d.rows > 1) updateRows(d, src, dst, d.rows - 1, d.rows);
        current ^= 1;
    }
    if (current) std::swap(d.energy[0], d.energy[1]);
}

static uint64_t localHash(const Domain& d) {
    uint64_t hash = 0;
    for (int lr = 0; lr < d.rows; ++lr) {
        for (int y = 0; y < d.n; ++y) {
            const uint64_t i = static_cast<uint64_t>(d.first_row + lr) * d.n + y;
            uint64_t e, f;
            const val_t energy = d.energy[0][static_cast<size_t>(lr + 1) * d.n + y];
            const val_t flux = d.total_flux[static_cast<size_t>(lr) * d.n + y];
            std::memcpy(&e, &energy, sizeof(e));
            std::memcpy(&f, &flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

static bool validateResults(const Domain& d) {
    val_t local[4] = {0.0, 0.0, std::numeric_limits<val_t>::lowest(),
                     std::numeric_limits<val_t>::max()};
    for (int lr = 0; lr < d.rows; ++lr)
        for (int y = 0; y < d.n; ++y) {
            const val_t e = d.energy[0][static_cast<size_t>(lr + 1) * d.n + y];
            local[0] += e;
            local[1] += d.total_flux[static_cast<size_t>(lr) * d.n + y];
            local[2] = std::max(local[2], e);
            local[3] = std::min(local[3], e);
        }
    val_t sums[2], maximum, minimum;
    MPI_Reduce(local, sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local + 2, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local + 3, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (d.rank == 0) {
        std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n"
                    "  Energy range: [%.6f, %.6f]\n", sums[0], sums[1], minimum, maximum);
        valid = std::isfinite(sums[0]) && std::isfinite(sums[1]) &&
                std::isfinite(maximum) && std::isfinite(minimum);
        if (std::abs(sums[0]) > 1e-8)
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        std::printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num> Grid size (default: 512)\n"
                "  -i <num> Iterations (default: 10)\n  -v Validate\n"
                "  -r Print results\n  -h Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iters = 10;
    bool validate = false, printResults = false, help = false, bad = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else bad = true;
    }
    if (help || bad || n <= 0 || iters < 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return bad || n <= 0 || iters < 0;
    }

    Domain d = buildSquare2D(n, rank, ranks);
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n"
                    "============================================\n"
                    "Grid size: %d x %d = %lld elements\nIterations: %d\n"
                    "Validation: %s\nMPI ranks: %d\n\nBuilding unstructured mesh...\n",
                    n, n, static_cast<long long>(n) * n, iters,
                    validate ? "enabled" : "disabled", ranks);
        const double bytes = static_cast<double>(n) * n * 3 * sizeof(val_t);
        std::printf("Aggregate dynamic memory: %.2f MB\n\nRunning simulation...\n",
                    bytes / (1024.0 * 1024.0));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(d, iters);
    const double local_time = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    uint64_t lh = localHash(d), hash = 0;
    MPI_Reduce(&lh, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double ms = elapsed * 1000.0;
        const int measured = std::max(iters - 1, 1);
        const double geps = elapsed > 0.0 ? measured * static_cast<double>(n) * n / elapsed / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\nPerformance:\n"
                    "  Time per iteration: %.4f ms\n"
                    "  Elements/sec: %.4f GigaElements/s\n"
                    "  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n",
                    ms, ms / measured, geps, geps * 22.0,
                    static_cast<unsigned long long>(hash));
    }

    if (printResults) {
        const int local_count = d.rows * n;
        std::vector<val_t> packed(local_count);
        for (int lr = 0; lr < d.rows; ++lr)
            std::copy_n(d.energy[0].data() + static_cast<size_t>(lr + 1) * n, n,
                        packed.data() + static_cast<size_t>(lr) * n);
        std::vector<int> counts(ranks), displs(ranks);
        for (int r = 0; r < ranks; ++r) {
            int first, rows; decompose(n, r, ranks, first, rows);
            counts[r] = rows * n; displs[r] = first * n;
        }
        std::vector<val_t> all(rank == 0 ? static_cast<size_t>(n) * n : 0);
        MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE, all.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(all, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(d);
    MPI_Finalize();
    return valid ? 0 : 1;
}
