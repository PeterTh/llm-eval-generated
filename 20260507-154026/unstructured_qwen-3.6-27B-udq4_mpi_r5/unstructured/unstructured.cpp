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

// World state (per-rank local copy)
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// ---- MPI state ----
static int mpi_rank = 0;
static int mpi_size = 1;

// Domain decomposition (row-based)
static int g_n_elems_root = 512;
static int g_row_start = 0;   // first owned row
static int g_row_end = 0;     // last owned row (exclusive)
static int g_halo_row_start = 0;
static int g_halo_row_end = 0;
static int g_rank_above = -1;
static int g_rank_below = -1;

// MPI datatype for ElementDynamic (two contiguous doubles)
static MPI_Datatype mpi_element_dynamic_type = MPI_DATATYPE_NULL;

// ---- helpers ----

static inline idx_t local_idx(int x, int y) {
    return static_cast<idx_t>(x - g_halo_row_start) * g_n_elems_root + y;
}

// Row-based domain decomposition: each rank gets a contiguous set of rows.
// Halos of 1 row are added above/below for neighbour communication.
static void setup_domain(int n_elems_root, int rank, int size) {
    const int rows_per_rank = n_elems_root / size;
    const int remainder = n_elems_root % size;

    g_row_start = rank * rows_per_rank + std::min(rank, remainder);
    g_row_end   = (rank + 1) * rows_per_rank + std::min(rank + 1, remainder);

    g_halo_row_start = std::max(0, g_row_start - 1);
    g_halo_row_end   = std::min(n_elems_root, g_row_end + 1);

    // Find neighbour ranks (only those that actually own rows)
    g_rank_above = -1;
    g_rank_below = -1;
    for (int r = 0; r < size; ++r) {
        const int rs = r * rows_per_rank + std::min(r, remainder);
        const int re = (r + 1) * rows_per_rank + std::min(r + 1, remainder);
        if (r == rank || rs == re) continue;
        if (re == g_row_start)  g_rank_above = r;
        if (rs == g_row_end)    g_rank_below = r;
    }
}

// ---- mesh build (local) ----

void buildSquare2D(World& world, int n_elems_root) {
    const int local_rows = g_halo_row_end - g_halo_row_start;
    const idx_t n_local  = static_cast<idx_t>(local_rows) * n_elems_root;

    // Materials (replicated on every rank)
    world.materials.reserve(3);
    world.materials.emplace_back(Material{0.8, 0.0});   // default
    world.materials.emplace_back(Material{0.8, 0.5});   // inflow
    world.materials.emplace_back(Material{0.8, -0.5});  // outflow

    world.elements_static.resize(n_local);
    world.elements_dynamic.resize(n_local);
    world.elements_dynamic_swap.resize(n_local);

    for (idx_t i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx  = 0;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux     = 0.0;
    }

    // Build connectivity for owned elements only
    for (int x = g_row_start; x < g_row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t li = local_idx(x, y);
            ElementStatic& elem = world.elements_static[li];

            const int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    elem.connected_idx[elem.num_connections] = local_idx(nx, ny);
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Corner materials (only if the corner row belongs to this rank)
    const int last = n_elems_root - 1;
    auto set_mat = [&](int x, int y, idx_t id) {
        const idx_t li = local_idx(x, y);
        if (li < n_local) world.elements_static[li].material_idx = id;
    };
    set_mat(0, 0, 1);           // inflow
    set_mat(0, last, 2);        // outflow
    set_mat(last, 0, 2);        // outflow
    set_mat(last, last, 1);     // inflow
}

// ---- flux kernel ----

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// ---- halo exchange (non-blocking, overlapping top+bottom) ----
//
// Each boundary pair uses a deterministic tag (min of the two ranks) so both
// sides of the exchange agree on the tag value.

static void exchangeHalo(World& world, int n_elems_root) {
    if (mpi_size == 1 || g_row_start >= g_row_end) return;

    const int ncols = n_elems_root;
    MPI_Request reqs[4];
    int nreq = 0;

    // Top boundary: exchange with rank_above
    if (g_rank_above >= 0) {
        const idx_t recv_off = local_idx(g_halo_row_start, 0);
        const idx_t send_off = local_idx(g_row_start, 0);
        const int tag = std::min(g_rank_above, mpi_rank);
        MPI_Irecv(&world.elements_dynamic[recv_off], ncols, mpi_element_dynamic_type,
                  g_rank_above, tag, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(&world.elements_dynamic[send_off], ncols, mpi_element_dynamic_type,
                  g_rank_above, tag, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    // Bottom boundary: exchange with rank_below
    if (g_rank_below >= 0) {
        const idx_t recv_off = local_idx(g_halo_row_end - 1, 0);
        const idx_t send_off = local_idx(g_row_end - 1, 0);
        const int tag = std::min(g_rank_below, mpi_rank);
        MPI_Irecv(&world.elements_dynamic[recv_off], ncols, mpi_element_dynamic_type,
                  g_rank_below, tag, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Isend(&world.elements_dynamic[send_off], ncols, mpi_element_dynamic_type,
                  g_rank_below, tag, MPI_COMM_WORLD, &reqs[nreq++]);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

// ---- simulation (parallel) ----

void runSimulation(World& world, int n_iters, int n_elems_root) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // Compute phase — each rank updates its owned elements
        for (int x = g_row_start; x < g_row_end; ++x) {
            for (int y = 0; y < n_elems_root; ++y) {
                const idx_t li = local_idx(x, y);
                const ElementStatic&  es = world.elements_static[li];
                const ElementDynamic& ed = world.elements_dynamic[li];
                const Material&       mat = world.materials[es.material_idx];

                val_t total_flux = mat.external_flow;

                for (idx_t j = 0; j < es.num_connections; ++j) {
                    const idx_t ni = es.connected_idx[j];
                    total_flux += computeFlux(mat, ed, es.connected_flux[j],
                                              world.elements_dynamic[ni]);
                }

                ElementDynamic& ew = world.elements_dynamic_swap[li];
                ew.current_energy = ed.current_energy + total_flux;
                ew.total_flux     = ed.total_flux + std::abs(total_flux);
            }
        }

        // Swap double-buffer
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);

        // Exchange boundary rows so neighbours have fresh data
        exchangeHalo(world, n_elems_root);
    }
}

// ---- validation (collective) ----

bool validateResults(const World& world, int n_elems_root) {
    val_t l_energy_sum = 0.0, l_flux_sum = 0.0;
    val_t l_energy_max = std::numeric_limits<val_t>::lowest();
    val_t l_energy_min = std::numeric_limits<val_t>::max();

    for (int x = g_row_start; x < g_row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t li = local_idx(x, y);
            const val_t e = world.elements_dynamic[li].current_energy;
            l_energy_sum += e;
            l_flux_sum   += world.elements_dynamic[li].total_flux;
            l_energy_max  = std::max(e, l_energy_max);
            l_energy_min  = std::min(e, l_energy_min);
        }
    }

    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Allreduce(&l_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l_flux_sum,   &flux_sum,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&l_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&l_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    bool valid = true;
    if (mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n"); valid = false;
        }
        if (std::abs(energy_sum) > 1e-8) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n"); valid = false;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n"); valid = false;
        }
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

// ---- hash (XOR-reduce across ranks) ----

static void xor_op_func(void* incoming, void* inout, int* len, MPI_Datatype*) {
    for (int i = 0; i < *len; i++)
        ((uint64_t*)inout)[i] ^= ((uint64_t*)incoming)[i];
}

static uint64_t computeHash(const World& world, int n_elems_root) {
    uint64_t local_hash = 0;

    for (int x = g_row_start; x < g_row_end; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const idx_t li = local_idx(x, y);
            const idx_t gi = static_cast<idx_t>(x) * n_elems_root + y;

            const uint64_t* ep = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[li].current_energy);
            const uint64_t* fp = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[li].total_flux);
            local_hash ^= (*ep + gi) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (*fp + gi) * 0xbf58476d1ce4e5b9ULL;
        }
    }

    uint64_t global_hash;
    MPI_Op op;
    MPI_Op_create(xor_op_func, true, &op);
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, op, MPI_COMM_WORLD);
    MPI_Op_free(&op);
    return global_hash;
}

// ---- usage ----

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---- main ----

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int n_elems_root = 512;
    int n_iters      = 10;
    int validate     = 0;
    int printResults = 0;

    // Parse args on rank 0
    if (mpi_rank == 0) {
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

    // Broadcast parameters to all ranks
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_elems = n_elems_root * n_elems_root;
    g_n_elems_root = n_elems_root;

    // Domain decomposition
    setup_domain(n_elems_root, mpi_rank, mpi_size);

    // MPI datatype for halo exchange
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_element_dynamic_type);
    MPI_Type_commit(&mpi_element_dynamic_type);

    // ---- header (rank 0) ----
    if (mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Grid size: %d x %d = %d elements\n",
               n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // ---- build mesh ----
    if (mpi_rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root);

    // ---- memory (rank 0, global view) ----
    if (mpi_rank == 0) {
        const size_t sm = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dm = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               (sm + dm) / (1024.0 * 1024.0),
               sm / (1024.0 * 1024.0),
               dm / (1024.0 * 1024.0));
        printf("\n");
    }

    // ---- simulation ----
    if (mpi_rank == 0) printf("Running simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    runSimulation(world, n_iters, n_elems_root);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_duration_ms = (MPI_Wtime() - t0) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- performance (rank 0) ----
    if (mpi_rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        const int n_measured = std::max(n_iters - 1, 1);
        const double tpi = duration_ms / n_measured;
        const double ges = (static_cast<double>(n_measured) * n_elems) /
                           (duration_ms / 1000.0) / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", tpi);
        printf("  Elements/sec: %.4f GigaElements/s\n", ges);
        printf("  Performance: %.4f GFLOPS\n", ges * 22.0);
    }

    // ---- hash ----
    const uint64_t hash = computeHash(world, n_elems_root);
    if (mpi_rank == 0) {
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }

    // ---- print results (-r) ----
    if (printResults) {
        const int rows_per_rank = n_elems_root / mpi_size;
        const int remainder     = n_elems_root % mpi_size;

        const int n_owned = (g_row_end - g_row_start) * n_elems_root;
        std::vector<double> local_energy(n_owned);
        for (int x = g_row_start; x < g_row_end; ++x)
            for (int y = 0; y < n_elems_root; ++y) {
                const idx_t li = local_idx(x, y);
                local_energy[(x - g_row_start) * n_elems_root + y] =
                    world.elements_dynamic[li].current_energy;
            }

        std::vector<int> recvcounts(mpi_size), displs(mpi_size);
        for (int r = 0; r < mpi_size; ++r) {
            const int rs = r * rows_per_rank + std::min(r, remainder);
            const int re = (r + 1) * rows_per_rank + std::min(r + 1, remainder);
            recvcounts[r] = (re - rs) * n_elems_root;
            displs[r]     = rs * n_elems_root;
        }

        std::vector<double> all_energy;
        if (mpi_rank == 0) all_energy.resize(n_elems);

        const double* sbuf = n_owned > 0 ? local_energy.data() : nullptr;
        MPI_Gatherv(sbuf, n_owned, MPI_DOUBLE,
                    all_energy.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0)
            print_results(all_energy, "ElementEnergy");
    }

    // ---- validation ----
    int global_valid = 1;
    if (validate) {
        const bool valid = validateResults(world, n_elems_root);
        int local_valid = valid ? 1 : 0;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    // cleanup
    if (mpi_element_dynamic_type != MPI_DATATYPE_NULL)
        MPI_Type_free(&mpi_element_dynamic_type);

    MPI_Finalize();
    return global_valid ? 0 : 1;
}
