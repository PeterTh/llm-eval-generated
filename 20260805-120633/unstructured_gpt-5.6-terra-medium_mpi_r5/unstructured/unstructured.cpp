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

#include "../common/results_output.hpp"

using val_t = double;
constexpr int DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

struct Material { val_t transfer_coeff, external_flow; };
struct ElementDynamic { val_t current_energy, total_flux; };
static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "ElementDynamic must be tightly packed");

struct World {
    std::vector<Material> materials;
    std::vector<unsigned char> material;
    std::vector<ElementDynamic> current, next;
    int rows = 0, cols = 0, first_row = 0, first_col = 0;
};

static int block_size(int n, int parts, int coordinate) {
    const int base = n / parts, remainder = n % parts;
    return base + (coordinate < remainder);
}
static int block_start(int n, int parts, int coordinate) {
    const int base = n / parts, remainder = n % parts;
    return coordinate * base + std::min(coordinate, remainder);
}

// Select the largest rectangular process grid that gives every active rank work.
static int choose_process_grid(int requested, int n, int dims[2]) {
    for (int count = std::min(requested, n * n); count > 0; --count) {
        int best_rows = 0, best_cols = 0;
        for (int rows = 1; rows * rows <= count; ++rows) {
            if (count % rows) continue;
            const int cols = count / rows;
            if (rows <= n && cols <= n && (!best_rows || cols - rows < best_cols - best_rows)) {
                best_rows = rows; best_cols = cols;
            }
        }
        if (best_rows) { dims[0] = best_rows; dims[1] = best_cols; return count; }
    }
    return 0;
}

static void buildSquare2D(World& world, int n, const int dims[2], const int coords[2]) {
    world.rows = block_size(n, dims[0], coords[0]);
    world.cols = block_size(n, dims[1], coords[1]);
    world.first_row = block_start(n, dims[0], coords[0]);
    world.first_col = block_start(n, dims[1], coords[1]);
    const size_t count = static_cast<size_t>(world.rows) * world.cols;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.material.assign(count, DEFAULT_MAT_ID);
    world.current.assign(count, {0.0, 0.0});
    world.next.resize(count);
    for (int r = 0; r < world.rows; ++r) for (int c = 0; c < world.cols; ++c) {
        const int gr = world.first_row + r, gc = world.first_col + c;
        if ((gr == 0 && gc == 0) || (gr == n - 1 && gc == n - 1))
            world.material[static_cast<size_t>(r) * world.cols + c] = INFLOW_MAT_ID;
        else if ((gr == 0 && gc == n - 1) || (gr == n - 1 && gc == 0))
            world.material[static_cast<size_t>(r) * world.cols + c] = OUTFLOW_MAT_ID;
    }
}

static inline void update_cell(World& w, int n, int r, int c,
                               const std::vector<ElementDynamic>& north,
                               const std::vector<ElementDynamic>& south,
                               const std::vector<ElementDynamic>& west,
                               const std::vector<ElementDynamic>& east) {
    const size_t i = static_cast<size_t>(r) * w.cols + c;
    const ElementDynamic& center = w.current[i];
    const Material& mat = w.materials[w.material[i]];
    val_t flux = mat.external_flow;
    const auto add = [&](const ElementDynamic& other) {
        flux += (other.current_energy - center.current_energy) * mat.transfer_coeff * 0.25;
    };
    const int gr = w.first_row + r, gc = w.first_col + c;
    if (gr > 0) add(r ? w.current[i - w.cols] : north[c]);
    if (gr + 1 < n) add(r + 1 < w.rows ? w.current[i + w.cols] : south[c]);
    if (gc > 0) add(c ? w.current[i - 1] : west[r]);
    if (gc + 1 < n) add(c + 1 < w.cols ? w.current[i + 1] : east[r]);
    w.next[i] = {center.current_energy + flux, center.total_flux + std::abs(flux)};
}

static void runSimulation(World& w, int n, int iters, MPI_Comm comm, const int neighbors[4]) {
    std::vector<ElementDynamic> north(w.cols), south(w.cols), west(w.rows), east(w.rows);
    MPI_Datatype column;
    MPI_Type_vector(w.rows, 2, 2 * w.cols, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    for (int iter = 0; iter < iters; ++iter) {
        MPI_Request requests[8]; int nr = 0;
        // Direction order: north, south, west, east. Tags identify the transmitted edge.
        if (neighbors[0] != MPI_PROC_NULL) {
            MPI_Irecv(north.data(), 2 * w.cols, MPI_DOUBLE, neighbors[0], 1, comm, &requests[nr++]);
            MPI_Isend(w.current.data(), 2 * w.cols, MPI_DOUBLE, neighbors[0], 0, comm, &requests[nr++]);
        }
        if (neighbors[1] != MPI_PROC_NULL) {
            MPI_Irecv(south.data(), 2 * w.cols, MPI_DOUBLE, neighbors[1], 0, comm, &requests[nr++]);
            MPI_Isend(w.current.data() + static_cast<size_t>(w.rows - 1) * w.cols, 2 * w.cols, MPI_DOUBLE, neighbors[1], 1, comm, &requests[nr++]);
        }
        if (neighbors[2] != MPI_PROC_NULL) {
            MPI_Irecv(west.data(), 2 * w.rows, MPI_DOUBLE, neighbors[2], 3, comm, &requests[nr++]);
            MPI_Isend(w.current.data(), 1, column, neighbors[2], 2, comm, &requests[nr++]);
        }
        if (neighbors[3] != MPI_PROC_NULL) {
            MPI_Irecv(east.data(), 2 * w.rows, MPI_DOUBLE, neighbors[3], 2, comm, &requests[nr++]);
            MPI_Isend(w.current.data() + (w.cols - 1), 1, column, neighbors[3], 3, comm, &requests[nr++]);
        }
        // The interior does not depend on received data and overlaps communication.
        for (int r = 1; r + 1 < w.rows; ++r)
            for (int c = 1; c + 1 < w.cols; ++c) update_cell(w, n, r, c, north, south, west, east);
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
        for (int r = 0; r < w.rows; ++r) for (int c = 0; c < w.cols; ++c)
            if (r == 0 || r + 1 == w.rows || c == 0 || c + 1 == w.cols)
                update_cell(w, n, r, c, north, south, west, east);
        w.current.swap(w.next);
    }
    MPI_Type_free(&column);
}

static uint64_t computeHash(const World& w, int n) {
    uint64_t hash = 0;
    for (int r = 0; r < w.rows; ++r) for (int c = 0; c < w.cols; ++c) {
        const size_t local = static_cast<size_t>(r) * w.cols + c;
        const uint64_t global = static_cast<uint64_t>(w.first_row + r) * n + w.first_col + c;
        uint64_t e, f;
        std::memcpy(&e, &w.current[local].current_energy, sizeof(e));
        std::memcpy(&f, &w.current[local].total_flux, sizeof(f));
        hash ^= (e + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* p) { std::printf("Usage: %s [-n grid] [-i iterations] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int rank = world_rank;
    int n = 512, iters = 10; bool validate = false, results = false, bad_args = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else bad_args = true;
    }
    if (n <= 0 || iters < 0 || bad_args) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    int dims[2] = {0, 0};
    const int active_count = choose_process_grid(ranks, n, dims);
    MPI_Comm active; MPI_Comm_split(MPI_COMM_WORLD, rank < active_count ? 0 : MPI_UNDEFINED, rank, &active);
    World world; int coords[2] = {0, 0}, neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    double elapsed = 0.0;
    if (rank < active_count) {
        MPI_Comm cart; int periods[2] = {0, 0}; MPI_Cart_create(active, 2, dims, periods, 0, &cart);
        MPI_Comm_rank(cart, &rank); MPI_Cart_coords(cart, rank, 2, coords);
        MPI_Cart_shift(cart, 0, 1, &neighbors[0], &neighbors[1]); MPI_Cart_shift(cart, 1, 1, &neighbors[2], &neighbors[3]);
        buildSquare2D(world, n, dims, coords);
        MPI_Barrier(cart); const double start = MPI_Wtime();
        runSimulation(world, n, iters, cart, neighbors);
        elapsed = MPI_Wtime() - start;
        MPI_Comm_free(&cart); MPI_Comm_free(&active);
    }
    double max_elapsed = 0.0; MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = world_rank < active_count ? computeHash(world, n) : 0, hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    double local_sum = 0, local_flux = 0, local_min = std::numeric_limits<double>::max(), local_max = std::numeric_limits<double>::lowest();
    for (const auto& e : world.current) { local_sum += e.current_energy; local_flux += e.total_flux; local_min = std::min(local_min, e.current_energy); local_max = std::max(local_max, e.current_energy); }
    double sum, flux, emin, emax;
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &emin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); MPI_Reduce(&local_max, &emax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (!rank) {
        const double ms = max_elapsed * 1000.0, measured = std::max(iters - 1, 1);
        const double rate = max_elapsed > 0.0 ? (measured * static_cast<double>(n) * n) / max_elapsed / 1e9 : 0.0;
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nMPI ranks: %d (%d active)\n\n", n, n, n * n, iters, ranks, active_count);
        std::printf("Computation time: %.0f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n", ms, ms / measured, rate, rate * 22.0, static_cast<unsigned long long>(hash));
        if (validate) {
            valid = std::isfinite(sum) && std::isfinite(flux) && std::isfinite(emin) && std::isfinite(emax);
            std::printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n", sum, flux, emin, emax, valid ? "PASSED" : "FAILED");
        }
    }
    // External result output is deliberately root-only; gather in global row-major order.
    if (results) {
        const int local_count = static_cast<int>(world.current.size());
        int metadata[4] = {world.first_row, world.first_col, world.rows, world.cols};
        std::vector<int> counts, displacements, all_metadata;
        if (!world_rank) { counts.resize(ranks); all_metadata.resize(4 * ranks); }
        MPI_Gather(&local_count, 1, MPI_INT, world_rank ? nullptr : counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(metadata, 4, MPI_INT, world_rank ? nullptr : all_metadata.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
        if (!world_rank) { displacements.resize(ranks); for (int p = 1; p < ranks; ++p) displacements[p] = displacements[p - 1] + counts[p - 1]; }
        std::vector<double> local_energy(local_count), gathered, energy;
        for (int i = 0; i < local_count; ++i) local_energy[i] = world.current[i].current_energy;
        if (!world_rank) { gathered.resize(static_cast<size_t>(n) * n); energy.assign(static_cast<size_t>(n) * n, 0.0); }
        MPI_Gatherv(local_energy.data(), local_count, MPI_DOUBLE, world_rank ? nullptr : gathered.data(), world_rank ? nullptr : counts.data(), world_rank ? nullptr : displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!world_rank) {
            for (int p = 0; p < ranks; ++p) for (int r = 0; r < all_metadata[4 * p + 2]; ++r) for (int c = 0; c < all_metadata[4 * p + 3]; ++c)
                energy[static_cast<size_t>(all_metadata[4 * p] + r) * n + all_metadata[4 * p + 1] + c] = gathered[displacements[p] + r * all_metadata[4 * p + 3] + c];
            print_results(energy, "ElementEnergy");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
