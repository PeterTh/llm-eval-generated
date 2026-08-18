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

// Each rank owns a contiguous set of grid rows.  The two ghost entries in
// read_state contain the adjacent rank's boundary energies when needed.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
    std::vector<val_t> ghost_energy;
    std::vector<val_t> halo_send;
    int root = 0, first_row = 0, local_rows = 0;
};

constexpr idx_t DEFAULT_MAT_ID = 0, INFLOW_MAT_ID = 1, OUTFLOW_MAT_ID = 2;

static void buildSquare2D(World& world, int root, int rank, int ranks) {
    world.root = root;
    const int base = root / ranks, remainder = root % ranks;
    world.local_rows = base + (rank < remainder ? 1 : 0);
    world.first_row = rank * base + std::min(rank, remainder);
    const size_t local_count = static_cast<size_t>(world.local_rows) * root;

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(local_count);
    world.elements_dynamic.assign(local_count, {0.0, 0.0});
    world.elements_dynamic_swap.resize(local_count);
    world.ghost_energy.assign(static_cast<size_t>(2) * root, 0.0);
    world.halo_send.resize(static_cast<size_t>(2) * root);

    // Connected indices address local state, with local_count/local_count+1
    // representing the top/bottom ghost rows respectively.
    for (int lr = 0; lr < world.local_rows; ++lr) {
        const int global_row = world.first_row + lr;
        for (int col = 0; col < root; ++col) {
            const size_t i = static_cast<size_t>(lr) * root + col;
            ElementStatic& e = world.elements_static[i];
            e.material_idx = (global_row == 0 && col == 0) ||
                             (global_row == root - 1 && col == root - 1) ? INFLOW_MAT_ID :
                             ((global_row == 0 && col == root - 1) ||
                              (global_row == root - 1 && col == 0) ? OUTFLOW_MAT_ID : DEFAULT_MAT_ID);
            e.num_connections = 0;
            const auto add = [&](idx_t index) {
                e.connected_idx[e.num_connections] = index;
                e.connected_flux[e.num_connections++] = 1.0;
            };
            if (global_row + 1 < root)
                add(lr + 1 < world.local_rows ? i + root : local_count + root + col);
            if (global_row > 0)
                add(lr > 0 ? i - root : local_count + col);
            if (col + 1 < root) add(i + 1);
            if (col > 0) add(i - 1);
        }
    }
}

inline val_t readEnergy(const World& world, idx_t index) {
    const size_t local_count = world.elements_dynamic.size();
    return index < local_count ? world.elements_dynamic[index].current_energy
                               : world.ghost_energy[index - local_count];
}

static void updateRows(World& world, int begin, int end) {
    const int root = world.root;
    for (int lr = begin; lr < end; ++lr) {
        const size_t begin_i = static_cast<size_t>(lr) * root;
        for (int col = 0; col < root; ++col) {
            const size_t i = begin_i + col;
            const ElementStatic& e = world.elements_static[i];
            const ElementDynamic& d = world.elements_dynamic[i];
            val_t flux = world.materials[e.material_idx].external_flow;
            const val_t coeff = world.materials[e.material_idx].transfer_coeff * 0.25;
            for (idx_t j = 0; j < e.num_connections; ++j)
                flux += (readEnergy(world, e.connected_idx[j]) - d.current_energy) * coeff * e.connected_flux[j];
            world.elements_dynamic_swap[i] = {d.current_energy + flux, d.total_flux + std::abs(flux)};
        }
    }
}

static void runSimulation(World& world, int iterations, int rank, int ranks) {
    const int top = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int bottom = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int width = world.root;
    for (int iter = 0; iter < iterations; ++iter) {
        // ElementDynamic also carries accumulated flux, but halos need only
        // current energy.  Packing makes the message half the size and avoids
        // a non-contiguous MPI datatype in the hot path.
        for (int col = 0; col < width; ++col) {
            world.halo_send[col] = world.elements_dynamic[col].current_energy;
            world.halo_send[width + col] = world.elements_dynamic[static_cast<size_t>(world.local_rows - 1) * width + col].current_energy;
        }
        MPI_Request req[4];
        MPI_Irecv(world.ghost_energy.data(), width, MPI_DOUBLE, top, 1, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(world.ghost_energy.data() + width, width, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(world.halo_send.data(), width, MPI_DOUBLE, top, 0, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(world.halo_send.data() + width, width, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &req[3]);

        // Interior rows do not depend on communication, hiding halo latency.
        updateRows(world, 1, std::max(1, world.local_rows - 1));
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (world.local_rows == 1) updateRows(world, 0, 1);
        else {
            updateRows(world, 0, 1);
            updateRows(world, world.local_rows - 1, world.local_rows);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

static bool validateResults(const World& world, int rank) {
    val_t local_energy = 0.0, local_flux = 0.0;
    val_t local_min = std::numeric_limits<val_t>::max();
    val_t local_max = std::numeric_limits<val_t>::lowest();
    for (const auto& e : world.elements_dynamic) {
        local_energy += e.current_energy; local_flux += e.total_flux;
        local_min = std::min(local_min, e.current_energy); local_max = std::max(local_max, e.current_energy);
    }
    val_t energy, flux, min_energy, max_energy;
    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &min_energy, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &max_energy, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const int local_ok = std::isfinite(local_energy) && std::isfinite(local_flux) && std::isfinite(local_min) && std::isfinite(local_max);
    int ok; MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy, flux, min_energy, max_energy);
        if (std::abs(energy) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: %s\n", ok ? "PASSED" : "FAILED");
    }
    return ok != 0;
}

static uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    const uint64_t start = static_cast<uint64_t>(world.first_row) * world.root;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        uint64_t e, f;
        std::memcpy(&e, &world.elements_dynamic[i].current_energy, sizeof(e));
        std::memcpy(&f, &world.elements_dynamic[i].total_flux, sizeof(f));
        hash ^= (e + start + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + start + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n  -n <num>     Grid size (NxN elements) (default: 512)\n  -i <num>     Number of simulation iterations (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int root = 512, iterations = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) root = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (root < ranks || root <= 0 || iterations < 0) { if (!rank) fprintf(stderr, "Grid size must be positive and at least the MPI rank count; iterations must be non-negative.\n"); MPI_Finalize(); return 1; }
    const uint64_t elements = static_cast<uint64_t>(root) * root;
    if (!rank) printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %llu elements\nIterations: %d\nValidation: %s\nMPI ranks: %d\n\nBuilding distributed unstructured mesh...\n", root, root, static_cast<unsigned long long>(elements), iterations, validate ? "enabled" : "disabled", ranks);
    World world; buildSquare2D(world, root, rank, ranks);
    if (!rank) {
        const size_t sm = static_cast<size_t>(elements) * sizeof(ElementStatic), dm = static_cast<size_t>(elements) * sizeof(ElementDynamic) * 2;
        printf("Global memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\nRunning simulation...\n", (sm + dm) / 1048576.0, sm / 1048576.0, dm / 1048576.0);
    }
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    runSimulation(world, iterations, rank, ranks);
    const double local_elapsed = MPI_Wtime() - start; double elapsed;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) {
        const double ms = elapsed * 1000.0, measured = std::max(iterations - 1, 1);
        const double geps = elapsed > 0.0 ? measured * elements / elapsed / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", ms, ms / measured, geps, geps * 22.0, static_cast<unsigned long long>(hash));
    }
    if (results) {
        std::vector<val_t> all;
        if (!rank) all.resize(static_cast<size_t>(elements));
        const int count = world.local_rows * root;
        std::vector<val_t> local_energy(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) local_energy[i] = world.elements_dynamic[i].current_energy;
        std::vector<int> counts, offsets;
        if (!rank) {
            counts.resize(ranks); offsets.resize(ranks);
            const int base = root / ranks, remainder = root % ranks;
            int offset = 0;
            for (int r = 0; r < ranks; ++r) {
                counts[r] = (base + (r < remainder ? 1 : 0)) * root;
                offsets[r] = offset; offset += counts[r];
            }
        }
        MPI_Gatherv(local_energy.data(), count, MPI_DOUBLE, rank ? nullptr : all.data(),
                    rank ? nullptr : counts.data(), rank ? nullptr : offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(all, "ElementEnergy");
    }
    const bool ok = !validate || validateResults(world, rank);
    MPI_Finalize();
    return ok ? 0 : 1;
}
