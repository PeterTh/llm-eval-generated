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

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 8;
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy; val_t total_flux; };

// Dynamic arrays include one halo row on each side.  Static entries describe
// owned rows only; their connection indices address the halo-padded arrays.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int width = 0;
    int first_row = 0;
    int local_rows = 0;
    int rank = 0;
    int ranks = 1;
};

inline val_t computeFlux(const Material& mat, const ElementDynamic& self,
                         val_t connection_flux, const ElementDynamic& other) {
    return (other.current_energy - self.current_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

void buildSquare2D(World& world, int root_rows, int rank, int ranks) {
    world.width = root_rows;
    world.rank = rank;
    world.ranks = ranks;
    const int base = root_rows / ranks;
    const int extra = root_rows % ranks;
    world.local_rows = base + (rank < extra ? 1 : 0);
    world.first_row = rank * base + std::min(rank, extra);

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t local_elems = static_cast<size_t>(world.local_rows) * root_rows;
    world.elements_static.resize(local_elems);
    const size_t padded_elems = static_cast<size_t>(world.local_rows + 2) * root_rows;
    world.elements_dynamic.assign(padded_elems, {0.0, 0.0});
    world.elements_dynamic_swap.assign(padded_elems, {0.0, 0.0});

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int global_x = world.first_row + local_x;
        for (int y = 0; y < root_rows; ++y) {
            const size_t owned = static_cast<size_t>(local_x) * root_rows + y;
            ElementStatic& elem = world.elements_static[owned];
            elem.material_idx = DEFAULT_MAT_ID;
            if ((global_x == 0 || global_x == root_rows - 1) && (y == 0 || y == root_rows - 1))
                elem.material_idx = ((global_x == 0 && y == 0) ||
                                     (global_x == root_rows - 1 && y == root_rows - 1)) ? INFLOW_MAT_ID : OUTFLOW_MAT_ID;
            const int physical = (local_x + 1) * root_rows + y;
            elem.num_connections = 0;
            // Preserve the original neighbor order: down, up, right, left.
            if (global_x + 1 < root_rows) { elem.connected_idx[elem.num_connections] = physical + root_rows; elem.connected_flux[elem.num_connections++] = 1.0; }
            if (global_x > 0)             { elem.connected_idx[elem.num_connections] = physical - root_rows; elem.connected_flux[elem.num_connections++] = 1.0; }
            if (y + 1 < root_rows)        { elem.connected_idx[elem.num_connections] = physical + 1; elem.connected_flux[elem.num_connections++] = 1.0; }
            if (y > 0)                    { elem.connected_idx[elem.num_connections] = physical - 1; elem.connected_flux[elem.num_connections++] = 1.0; }
        }
    }
}

void beginHaloExchange(World& world, MPI_Datatype dynamic_type, MPI_Request requests[4]) {
    const int n = world.width;
    const int above = world.rank == 0 ? MPI_PROC_NULL : world.rank - 1;
    const int below = world.rank == world.ranks - 1 ? MPI_PROC_NULL : world.rank + 1;
    MPI_Irecv(world.elements_dynamic.data(), n, dynamic_type, above, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(world.elements_dynamic.data() + static_cast<size_t>(world.local_rows + 1) * n, n, dynamic_type, below, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(world.elements_dynamic.data() + n, n, dynamic_type, above, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(world.elements_dynamic.data() + static_cast<size_t>(world.local_rows) * n, n, dynamic_type, below, 1, MPI_COMM_WORLD, &requests[3]);
}

void runSimulation(World& world, int n_iters, MPI_Datatype dynamic_type) {
    const int n = world.width;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        beginHaloExchange(world, dynamic_type, requests);
        const auto update_row = [&](int x) {
            for (int y = 0; y < n; ++y) {
                const size_t owned = static_cast<size_t>(x) * n + y;
                const size_t physical = static_cast<size_t>(x + 1) * n + y;
                const ElementStatic& stat = world.elements_static[owned];
                const ElementDynamic& current = world.elements_dynamic[physical];
                val_t flux = world.materials[stat.material_idx].external_flow;
                for (idx_t j = 0; j < stat.num_connections; ++j)
                    flux += computeFlux(world.materials[stat.material_idx], current, stat.connected_flux[j], world.elements_dynamic[stat.connected_idx[j]]);
                ElementDynamic& next = world.elements_dynamic_swap[physical];
                next.current_energy = current.current_energy + flux;
                next.total_flux = current.total_flux + std::abs(flux);
            }
        };
        // Interior rows depend only on local state, so hide halo latency behind
        // their computation.  Boundary rows are completed after the exchange.
        for (int x = 1; x + 1 < world.local_rows; ++x) update_row(x);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (world.local_rows > 0) update_row(0);
        if (world.local_rows > 1) update_row(world.local_rows - 1);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (int x = 0; x < world.local_rows; ++x) for (int y = 0; y < world.width; ++y) {
        const size_t global = static_cast<size_t>(world.first_row + x) * world.width + y;
        const ElementDynamic& e = world.elements_dynamic[static_cast<size_t>(x + 1) * world.width + y];
        uint64_t ebits, fbits;
        std::memcpy(&ebits, &e.current_energy, sizeof(ebits));
        std::memcpy(&fbits, &e.total_flux, sizeof(fbits));
        hash ^= (ebits + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (fbits + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
}

bool validateResults(const World& world, int rank) {
    val_t local_energy = 0.0, local_flux = 0.0;
    val_t local_min = std::numeric_limits<val_t>::max(), local_max = std::numeric_limits<val_t>::lowest();
    for (int x = 0; x < world.local_rows; ++x) for (int y = 0; y < world.width; ++y) {
        const auto& e = world.elements_dynamic[static_cast<size_t>(x + 1) * world.width + y];
        local_energy += e.current_energy; local_flux += e.total_flux;
        local_min = std::min(local_min, e.current_energy); local_max = std::max(local_max, e.current_energy);
    }
    val_t energy, flux, minimum, maximum;
    MPI_Reduce(&local_energy, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux, &flux, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n", energy, flux, minimum, maximum);
        if (!std::isfinite(energy) || !std::isfinite(flux) || !std::isfinite(minimum) || !std::isfinite(maximum)) { printf("  ERROR: non-finite result\n"); return false; }
        if (std::abs(energy) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        printf("  Validation: PASSED\n");
    }
    return true;
}

void printUsage(const char* p) { printf("Usage: %s [-n grid] [-i iterations] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n = 512, iters = 10; bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n <= 0 || iters < 0 || world_size > n) { if (!rank) fprintf(stderr, "Grid size must be positive and at least the number of MPI ranks.\n"); MPI_Finalize(); return 1; }
    World world; buildSquare2D(world, n, rank, world_size);
    MPI_Datatype dynamic_type; MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type); MPI_Type_commit(&dynamic_type);
    if (!rank) {
        const size_t total = static_cast<size_t>(n) * n;
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %zu elements\nIterations: %d\nMPI ranks: %d\nValidation: %s\n\n", n, n, total, iters, world_size, validate ? "enabled" : "disabled");
        printf("Building distributed unstructured mesh...\nGlobal memory (logical): %.2f MB\n\nRunning simulation...\n", total * (sizeof(ElementStatic) + 2 * sizeof(ElementDynamic)) / (1024.0 * 1024.0));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime(); runSimulation(world, iters, dynamic_type); const double elapsed = MPI_Wtime() - start;
    double duration; MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world), hash = 0; MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (!rank) {
        const int measured = std::max(iters - 1, 1); const double ms = duration * 1000.0;
        const double geps = duration > 0 ? (static_cast<double>(measured) * n * n) / duration / 1e9 : 0.0;
        printf("Computation time: %.3f ms\nPerformance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016llX\n\n", ms, ms / measured, geps, geps * 22.0, static_cast<unsigned long long>(hash));
    }
    if (results) {
        std::vector<double> local(static_cast<size_t>(world.local_rows) * n), global;
        for (int x = 0; x < world.local_rows; ++x) for (int y = 0; y < n; ++y) local[static_cast<size_t>(x) * n + y] = world.elements_dynamic[static_cast<size_t>(x + 1) * n + y].current_energy;
        std::vector<int> counts, displs;
        if (!rank) { counts.resize(world_size); displs.resize(world_size); for (int r = 0; r < world_size; ++r) { const int rows = n / world_size + (r < n % world_size); counts[r] = rows * n; displs[r] = (r * (n / world_size) + std::min(r, n % world_size)) * n; } global.resize(static_cast<size_t>(n) * n); }
        MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, rank ? nullptr : global.data(), rank ? nullptr : counts.data(), rank ? nullptr : displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (!rank) print_results(global, "ElementEnergy");
    }
    bool ok = validate ? validateResults(world, rank) : true; int all_ok = ok ? 1 : 0; MPI_Bcast(&all_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&dynamic_type); MPI_Finalize(); return all_ok ? 0 : 1;
}
