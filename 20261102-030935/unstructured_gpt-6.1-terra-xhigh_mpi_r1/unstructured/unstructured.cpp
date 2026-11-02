#include <algorithm>
#include <cmath>
#include <cstddef>
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

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

static_assert(sizeof(ElementDynamic) == 2 * sizeof(val_t) &&
              offsetof(ElementDynamic, total_flux) == sizeof(val_t));

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // One ghost row at each end. Only current_energy is exchanged.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    int n;
    int first_row;
    int rows;
};

// Contiguous row blocks keep the mesh and its output in global index order.
void buildSquare2D(World& world, int n, int rank, int size) {
    world.n = n;
    world.rows = n / size + (rank < n % size);
    world.first_row = rank * (n / size) + std::min(rank, n % size);
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(static_cast<size_t>(world.rows) * n);
    world.elements_dynamic.resize(static_cast<size_t>(world.rows + 2) * n);
    world.elements_dynamic_swap.resize(static_cast<size_t>(world.rows + 2) * n);

    for (int local_x = 0; local_x < world.rows; ++local_x) {
        const int x = world.first_row + local_x;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[static_cast<size_t>(local_x) * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            // Preserve the original ordering: down, up, right, left.
            const auto connect = [&](size_t neighbor) {
                elem.connected_idx[elem.num_connections] = neighbor;
                elem.connected_flux[elem.num_connections++] = 1.0;
            };
            const size_t center = static_cast<size_t>(local_x + 1) * n + y;
            if (x + 1 < n) connect(center + n);
            if (x > 0) connect(center - n);
            if (y + 1 < n) connect(center + 1);
            if (y > 0) connect(center - 1);

            // Assign in the same order as the original, including n == 1.
            if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (x == 0 && y == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n - 1 && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (x == n - 1 && y == n - 1) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                         val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void updateRows(World& world, int first, int last) {
    const int n = world.n;
    for (size_t i = static_cast<size_t>(first) * n; i < static_cast<size_t>(last) * n; ++i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i + n];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn = world.elements_dynamic[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[i + n];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    }
}

void runSimulation(World& world, int n_iters, int rank, int size) {
    if (world.rows == 0) return;
    const int n = world.n;
    const int active_ranks = std::min(n, size);
    if (active_ranks == 1) {
        for (int iter = 0; iter < n_iters; ++iter) {
            updateRows(world, 0, world.rows);
            std::swap(world.elements_dynamic, world.elements_dynamic_swap);
        }
        return;
    }
    const int prev = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < active_ranks ? rank + 1 : MPI_PROC_NULL;
    // A row contains interleaved energy and cumulative flux; only energy is needed by neighbors.
    MPI_Datatype energy_row;
    MPI_Type_vector(n, 1, 2, MPI_DOUBLE, &energy_row);
    MPI_Type_commit(&energy_row);
    for (int iter = 0; iter < n_iters; ++iter) {
        auto* data = world.elements_dynamic.data();
        MPI_Request requests[4];
        MPI_Irecv(&data[0].current_energy, 1, energy_row, prev, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&data[static_cast<size_t>(world.rows + 1) * n].current_energy, 1, energy_row,
                  next, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Isend(&data[n].current_energy, 1, energy_row, prev, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Isend(&data[static_cast<size_t>(world.rows) * n].current_energy, 1, energy_row,
                  next, 0, MPI_COMM_WORLD, &requests[3]);
        if (world.rows > 2) updateRows(world, 1, world.rows - 1);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        updateRows(world, 0, 1);
        if (world.rows > 1) updateRows(world, world.rows - 1, world.rows);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
    MPI_Type_free(&energy_row);
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const World& world) {
    uint64_t result = 0;
    const size_t count = static_cast<size_t>(world.rows) * world.n;
    const size_t global_start = static_cast<size_t>(world.first_row) * world.n;
    for (size_t i = 0; i < count; ++i) {
        const ElementDynamic& elem = world.elements_dynamic[i + world.n];
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &elem.current_energy, sizeof(energy_bits));
        std::memcpy(&flux_bits, &elem.total_flux, sizeof(flux_bits));
        result ^= (energy_bits + global_start + i) * 0x9e3779b97f4a7c15ULL;
        result ^= (flux_bits + global_start + i) * 0xbf58476d1ce4e5b9ULL;
    }
    return result;
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    // MPI_Gatherv uses int counts and displacements, as did the original grid indexing.
    if (n <= 0 || static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Invalid grid size: %d\n", n);
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n, rank, size);
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = (static_cast<size_t>(n_elems) + 2ULL * n * size) *
                                   sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters, rank, size);
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const uint64_t local_hash = computeHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = elapsed * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / elapsed / 1e9;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
    }
    int valid = 1;
    if (printResults || validate) {
        std::vector<int> counts, displacements;
        std::vector<ElementDynamic> all_elements;
        if (rank == 0) {
            counts.resize(size);
            displacements.resize(size);
            for (int r = 0; r < size; ++r) {
                counts[r] = (n / size + (r < n % size)) * n;
                displacements[r] = (r * (n / size) + std::min(r, n % size)) * n;
            }
            all_elements.resize(n_elems);
        }
        MPI_Datatype element_type;
        MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
        MPI_Type_commit(&element_type);
        MPI_Gatherv(world.elements_dynamic.data() + n, world.rows * n, element_type,
                    all_elements.data(), counts.data(), displacements.data(), element_type, 0, MPI_COMM_WORLD);
        MPI_Type_free(&element_type);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all_elements.size());
                for (const auto& elem : all_elements) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(all_elements) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
