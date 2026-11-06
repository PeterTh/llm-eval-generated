#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using val_t = double;

// Divide a dimension into contiguous, nearly equal pieces.
struct Partition {
    int start;
    int size;
};

static Partition partition(int n, int parts, int coord) {
    const int base = n / parts;
    const int extra = n % parts;
    return {coord * base + std::min(coord, extra), base + (coord < extra)};
}

struct Tile {
    int x, y, rows, cols, stride;
    int north, south, west, east;
    std::vector<val_t> energy, next, flux;
    MPI_Datatype column = MPI_DATATYPE_NULL;
};

// The generated mesh is a square grid. Store its connectivity implicitly and
// retain a one-cell energy halo around each rank's rectangular subdomain.
static Tile makeTile(int n, int rank, int px, int py) {
    const int rx = rank / py, ry = rank % py;
    const Partition x = partition(n, px, rx), y = partition(n, py, ry);
    Tile t{x.start, y.start, x.size, y.size, y.size + 2,
           rx ? rank - py : MPI_PROC_NULL,
           rx + 1 < px ? rank + py : MPI_PROC_NULL,
           ry ? rank - 1 : MPI_PROC_NULL,
           ry + 1 < py ? rank + 1 : MPI_PROC_NULL, {}, {}, {}};
    const size_t padded = (static_cast<size_t>(t.rows) + 2) * t.stride;
    t.energy.assign(padded, 0.0);
    t.next.assign(padded, 0.0);
    t.flux.assign(static_cast<size_t>(t.rows) * t.cols, 0.0);
    MPI_Type_vector(t.rows, 1, t.stride, MPI_DOUBLE, &t.column);
    MPI_Type_commit(&t.column);
    return t;
}

static inline void updateCell(Tile& t, int n, int i, int j) {
    const int x = t.x + i - 1, y = t.y + j - 1;
    const size_t pos = static_cast<size_t>(i) * t.stride + j;
    const val_t e = t.energy[pos];
    val_t total = 0.0;
    if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1)) {
        // The last corner assignment in the original mesh also handles n=1.
        total = ((x == 0 && y == 0) || (x == n - 1 && y == n - 1)) ? 0.5 : -0.5;
    }
    // Match the original connection order and floating-point operations.
    if (x + 1 < n) total += (t.energy[pos + t.stride] - e) * 0.8 * 0.25;
    if (x > 0)     total += (t.energy[pos - t.stride] - e) * 0.8 * 0.25;
    if (y + 1 < n) total += (t.energy[pos + 1] - e) * 0.8 * 0.25;
    if (y > 0)     total += (t.energy[pos - 1] - e) * 0.8 * 0.25;
    t.next[pos] = e + total;
    t.flux[static_cast<size_t>(i - 1) * t.cols + j - 1] += std::abs(total);
}

// All cells strictly inside a tile have four local neighbors and no corner
// source. Keeping this loop branch-free lets the compiler vectorize it.
static void updateInterior(Tile& t) {
    const val_t* __restrict__ energy = t.energy.data();
    val_t* __restrict__ next = t.next.data();
    val_t* __restrict__ flux = t.flux.data();
    const int s = t.stride;
    for (int i = 2; i < t.rows; ++i) {
        for (int j = 2; j < t.cols; ++j) {
            const size_t p = static_cast<size_t>(i) * s + j;
            const size_t f = static_cast<size_t>(i - 1) * t.cols + j - 1;
            const val_t e = energy[p];
            val_t total = 0.0;
            total += (energy[p + s] - e) * 0.8 * 0.25;
            total += (energy[p - s] - e) * 0.8 * 0.25;
            total += (energy[p + 1] - e) * 0.8 * 0.25;
            total += (energy[p - 1] - e) * 0.8 * 0.25;
            next[p] = e + total;
            flux[f] += std::abs(total);
        }
    }
}

static void runSimulation(Tile& t, int n, int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[8];
        val_t* e = t.energy.data();
        const int s = t.stride;
        MPI_Irecv(e + 1, t.cols, MPI_DOUBLE, t.north, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(e + static_cast<size_t>(t.rows + 1) * s + 1, t.cols, MPI_DOUBLE, t.south, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(e + s, 1, t.column, t.west, 2, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(e + s + t.cols + 1, 1, t.column, t.east, 3, MPI_COMM_WORLD, &req[3]);
        MPI_Isend(e + static_cast<size_t>(t.rows) * s + 1, t.cols, MPI_DOUBLE, t.south, 0, MPI_COMM_WORLD, &req[4]);
        MPI_Isend(e + s + 1, t.cols, MPI_DOUBLE, t.north, 1, MPI_COMM_WORLD, &req[5]);
        MPI_Isend(e + s + t.cols, 1, t.column, t.east, 2, MPI_COMM_WORLD, &req[6]);
        MPI_Isend(e + s + 1, 1, t.column, t.west, 3, MPI_COMM_WORLD, &req[7]);

        updateInterior(t);
        MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
        for (int j = 1; j <= t.cols; ++j) updateCell(t, n, 1, j);
        if (t.rows > 1)
            for (int j = 1; j <= t.cols; ++j) updateCell(t, n, t.rows, j);
        for (int i = 2; i < t.rows; ++i) {
            updateCell(t, n, i, 1);
            if (t.cols > 1) updateCell(t, n, i, t.cols);
        }
        t.energy.swap(t.next);
    }
}

static uint64_t localHash(const Tile& t, int n) {
    uint64_t hash_value = 0;
    for (int i = 0; i < t.rows; ++i) {
        for (int j = 0; j < t.cols; ++j) {
            const size_t local = static_cast<size_t>(i) * t.cols + j;
            const uint64_t global = static_cast<uint64_t>(t.x + i) * n + t.y + j;
            uint64_t energy_bits, flux_bits;
            const val_t energy = t.energy[static_cast<size_t>(i + 1) * t.stride + j + 1];
            std::memcpy(&energy_bits, &energy, sizeof(energy_bits));
            std::memcpy(&flux_bits, &t.flux[local], sizeof(flux_bits));
            hash_value ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
            hash_value ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash_value;
}

// Reconstruct global row-major order for the order-sensitive result printer and
// the original sequential validation sums. Called only when requested.
static std::vector<val_t> gatherField(const Tile& t, int n, int rank, int ranks,
                                      int px, int py, bool active, bool flux) {
    std::vector<val_t> local;
    if (active) {
        local.resize(static_cast<size_t>(t.rows) * t.cols);
        if (flux) local = t.flux;
        else for (int i = 0; i < t.rows; ++i)
            std::copy_n(t.energy.data() + static_cast<size_t>(i + 1) * t.stride + 1,
                        t.cols, local.data() + static_cast<size_t>(i) * t.cols);
    }
    std::vector<int> counts, displs;
    std::vector<val_t> packed, global;
    if (rank == 0) {
        counts.resize(ranks, 0);
        displs.resize(ranks, 0);
        int offset = 0;
        for (int r = 0; r < px * py; ++r) {
            const auto x = partition(n, px, r / py);
            const auto y = partition(n, py, r % py);
            counts[r] = x.size * y.size;
            displs[r] = offset;
            offset += counts[r];
        }
        packed.resize(static_cast<size_t>(n) * n);
        global.resize(packed.size());
    }
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int r = 0; r < px * py; ++r) {
            const auto x = partition(n, px, r / py);
            const auto y = partition(n, py, r % py);
            for (int i = 0; i < x.size; ++i)
                std::copy_n(packed.data() + displs[r] + static_cast<size_t>(i) * y.size,
                            y.size, global.data() + static_cast<size_t>(x.start + i) * n + y.start);
        }
    }
    return global;
}

static bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, iterations = 10, exit_code = 0;
    bool validate = false, print_results_flag = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); exit_code = 1; break; }
    }
    if (help || exit_code) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return exit_code;
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and have at most INT_MAX elements\n");
        MPI_Finalize();
        return 1;
    }
    // Use most available ranks, while allowing a few to remain idle if this
    // avoids thin tiles and their disproportionately expensive halo exchanges.
    int max_active = 1;
    for (int x = 1; x <= std::min(n, ranks); ++x)
        max_active = std::max(max_active, x * std::min(n, ranks / x));
    int px = 1, py = 1;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= std::min(n, ranks); ++x) {
        const int y = std::min(n, ranks / x);
        const int count = x * y;
        if (count < max_active - max_active / 10) continue;
        // Approximate local work plus the cost of communicating tile edges.
        const double cost = static_cast<double>(n) * n / count +
                            8.0 * n * (1.0 / x + 1.0 / y);
        if (cost < best_cost) { best_cost = cost; px = x; py = y; }
    }
    const bool active = rank < px * py;
    const int n_elems = n * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\nValidation: %s\n\n", iterations, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    Tile tile{};
    if (active) tile = makeTile(n, rank, px, py);
    const uint64_t local_bytes = active ? (tile.energy.size() + tile.next.size() + tile.flux.size()) * sizeof(val_t) : 0;
    uint64_t total_bytes = 0;
    MPI_Reduce(&local_bytes, &total_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (distributed state, including halos)\n\n", total_bytes / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) runSimulation(tile, n, iterations);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = active ? localHash(tile, n) : 0, result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double ms = max_elapsed * 1000.0;
        const int measured = std::max(iterations - 1, 1);
        const double rate = measured * static_cast<double>(n_elems) / max_elapsed / 1e9;
        printf("Computation time: %ld ms\n", static_cast<long>(ms));
        printf("Performance:\n  Time per iteration: %.4f ms\n", ms / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }
    std::vector<val_t> energy, flux;
    if (print_results_flag || validate) energy = gatherField(tile, n, rank, ranks, px, py, active, false);
    if (validate) flux = gatherField(tile, n, rank, ranks, px, py, active, true);
    if (rank == 0) {
        if (print_results_flag) print_results(energy, "ElementEnergy");
        if (validate && !validateResults(energy, flux)) exit_code = 1;
    }
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (active) MPI_Type_free(&tile.column);
    MPI_Finalize();
    return exit_code;
}
