#include <mpi.h>

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

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Local indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;      // owned elements only
    std::vector<ElementDynamic> elements_dynamic;    // owned + halo rows
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// MPI decomposition state
static int MPI_RANK = 0;
static int MPI_SIZE = 1;
static int X_START = 0;
static int X_END = 0;
static int LOCAL_NX = 0;
static int LOCAL_NELEMS = 0;
static int TOTAL_LOCAL = 0;
static int RANK_ABOVE = -1;
static int RANK_BELOW = -1;

// Map global (gx, gy) to dynamic index in elements_dynamic.
// Layout: [halo_above(0..n_root-1), owned(n_root..), halo_below(..)]
static inline int g2l(int gx, int gy, int n_root) {
    if (gx == X_START - 1) return gy;
    if (gx >= X_START && gx < X_END)
        return n_root + (gx - X_START) * n_root + gy;
    if (gx == X_END)
        return n_root + LOCAL_NX * n_root + gy;
    return -1;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Build local portion of the 2D grid mesh
void buildLocalMesh(World& world, const int n_root) {
    // Materials are identical on every rank
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(LOCAL_NELEMS);
    world.elements_dynamic.resize(TOTAL_LOCAL);
    world.elements_dynamic_swap.resize(TOTAL_LOCAL);

    // Zero-initialise all dynamic data
    for (int i = 0; i < TOTAL_LOCAL; ++i) {
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity for owned elements
    // Static index: (x - X_START) * n_root + y  (range 0..LOCAL_NELEMS-1)
    // Dynamic index: n_root + static_index
    for (int x = X_START; x < X_END; ++x) {
        for (int y = 0; y < n_root; ++y) {
            const int si = (x - X_START) * n_root + y;
            ElementStatic& elem = world.elements_static[si];
            elem.material_idx = 0;
            elem.num_connections = 0;

            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_root && ny >= 0 && ny < n_root) {
                    const int ld = g2l(nx, ny, n_root);
                    elem.connected_idx[elem.num_connections] = static_cast<idx_t>(ld);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Assign corner materials (only if the corner lives on this rank)
    const int last = n_root - 1;
    auto set_mat = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= X_START && gx < X_END && gy >= 0 && gy < n_root) {
            const int si = (gx - X_START) * n_root + gy;
            world.elements_static[si].material_idx = mat_id;
        }
    };
    set_mat(0, 0, 1);         // INFLOW
    set_mat(0, last, 2);      // OUTFLOW
    set_mat(last, 0, 2);      // OUTFLOW
    set_mat(last, last, 1);   // INFLOW
}

// Exchange one row of current_energy with neighbour ranks (non-blocking)
static inline void exchangeHalos(World& world, const int n_root) {
    MPI_Request reqs[4];
    int nreq = 0;

    if (RANK_ABOVE >= 0) {
        MPI_Isend(&world.elements_dynamic[n_root].current_energy, n_root,
                  MPI_DOUBLE, RANK_ABOVE, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(&world.elements_dynamic[0].current_energy, n_root,
                  MPI_DOUBLE, RANK_ABOVE, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    if (RANK_BELOW < MPI_SIZE) {
        const int last_owned = n_root + LOCAL_NELEMS - n_root;
        MPI_Isend(&world.elements_dynamic[last_owned].current_energy, n_root,
                  MPI_DOUBLE, RANK_BELOW, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(&world.elements_dynamic[n_root + LOCAL_NELEMS].current_energy, n_root,
                  MPI_DOUBLE, RANK_BELOW, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    if (nreq > 0)
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// Run simulation for n_iters iterations (MPI-aware)
void runSimulation(World& world, const int n_iters, const int n_root) {
    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world, n_root);

        for (int i = 0; i < LOCAL_NELEMS; ++i) {
            const ElementStatic& es = world.elements_static[i];
            const ElementDynamic& ed = world.elements_dynamic[n_root + i];
            const Material& mat = world.materials[es.material_idx];

            val_t total_flux = mat.external_flow;

            for (idx_t j = 0; j < es.num_connections; ++j) {
                const idx_t ni = es.connected_idx[j];
                const ElementDynamic& nd = world.elements_dynamic[ni];
                total_flux += computeFlux(mat, ed, es.connected_flux[j], nd);
            }

            ElementDynamic& ew = world.elements_dynamic_swap[n_root + i];
            ew.current_energy = ed.current_energy + total_flux;
            ew.total_flux = ed.total_flux + std::abs(total_flux);
        }

        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate = 0;
    int printResults = 0;

    // ── Parse arguments on rank 0, broadcast to all ranks ──
    if (MPI_RANK == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                n_elems_root = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)
                n_iters = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0)
                validate = 1;
            else if (strcmp(argv[i], "-r") == 0)
                printResults = 1;
            else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_elems = n_elems_root * n_elems_root;

    // ── 1-D domain decomposition along the x-axis ──
    const int base_nx = n_elems_root / MPI_SIZE;
    const int remainder = n_elems_root % MPI_SIZE;
    X_START = MPI_RANK * base_nx + std::min(MPI_RANK, remainder);
    X_END   = (MPI_RANK + 1) * base_nx + std::min(MPI_RANK + 1, remainder);
    LOCAL_NX   = X_END - X_START;
    LOCAL_NELEMS = LOCAL_NX * n_elems_root;
    TOTAL_LOCAL  = LOCAL_NELEMS + 2 * n_elems_root;  // owned + 2 halo rows
    RANK_ABOVE = MPI_RANK - 1;
    RANK_BELOW = MPI_RANK + 1;

    // ── Print header (rank 0) ──
    if (MPI_RANK == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", MPI_SIZE);
        printf("Grid size: %d x %d = %d elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Build local mesh ──
    World world;
    buildLocalMesh(world, n_elems_root);

    // ── Memory report (rank 0) ──
    if (MPI_RANK == 0) {
        const size_t total_static  = n_elems * sizeof(ElementStatic);
        const size_t total_dynamic = n_elems * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (total_static + total_dynamic) / (1024.0 * 1024.0),
               total_static / (1024.0 * 1024.0),
               total_dynamic / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Timed simulation ──
    const double t_start = MPI_Wtime();
    runSimulation(world, n_iters, n_elems_root);
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_end = MPI_Wtime();
    const double duration_s = t_end - t_start;
    const long duration_ms = static_cast<long>(duration_s * 1000.0);

    // ── Gather owned results to rank 0 ──
    std::vector<val_t> all_energy(n_elems);
    std::vector<val_t> all_flux(n_elems);

    std::vector<int> counts(MPI_SIZE);
    std::vector<int> displs(MPI_SIZE);
    if (MPI_RANK == 0) {
        const int bnx = n_elems_root / MPI_SIZE;
        const int rem = n_elems_root % MPI_SIZE;
        for (int r = 0; r < MPI_SIZE; ++r) {
            const int rs = r * bnx + std::min(r, rem);
            const int re = (r + 1) * bnx + std::min(r + 1, rem);
            counts[r] = (re - rs) * n_elems_root;
            displs[r] = rs * n_elems_root;
        }
    }

    MPI_Gatherv(&world.elements_dynamic[n_elems_root].current_energy,
                LOCAL_NELEMS, MPI_DOUBLE,
                all_energy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(&world.elements_dynamic[n_elems_root].total_flux,
                LOCAL_NELEMS, MPI_DOUBLE,
                all_flux.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ── Rank 0: hash, performance, validation ──
    if (MPI_RANK == 0) {
        // Compute hash (identical algorithm to serial version)
        uint64_t hash = 0;
        for (int i = 0; i < n_elems; ++i) {
            const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&all_energy[i]);
            const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&all_flux[i]);
            hash ^= (*e_ptr + static_cast<uint64_t>(i)) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + static_cast<uint64_t>(i)) * 0xbf58476d1ce4e5b9ULL;
        }

        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            static_cast<double>(n_measured_iters * n_elems) / duration_s / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            print_results(all_energy, "ElementEnergy");
        }

        // Validation
        if (validate) {
            val_t energy_sum = 0.0;
            val_t flux_sum = 0.0;
            val_t energy_max = std::numeric_limits<val_t>::lowest();
            val_t energy_min = std::numeric_limits<val_t>::max();

            for (int i = 0; i < n_elems; ++i) {
                energy_sum += all_energy[i];
                flux_sum   += all_flux[i];
                energy_max = std::max(all_energy[i], energy_max);
                energy_min = std::min(all_energy[i], energy_min);
            }

            printf("Validation results:\n");
            printf("  Energy sum: %.12f\n", energy_sum);
            printf("  Flux sum: %.2f\n", flux_sum);
            printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

            bool valid = true;
            constexpr val_t energy_epsilon = 1e-8;

            if (!std::isfinite(energy_sum)) {
                printf("  ERROR: Energy sum is not finite\n");
                valid = false;
            }
            if (std::abs(energy_sum) > energy_epsilon) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = false;
            }
            if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = false;
            }

            printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
