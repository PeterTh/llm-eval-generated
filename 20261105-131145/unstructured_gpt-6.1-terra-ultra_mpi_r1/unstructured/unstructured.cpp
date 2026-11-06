#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
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
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // The elements following the owned range are halo entries. Only their energy is used.
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    idx_t global_begin = 0;
    std::vector<int> recv_counts, recv_displs, send_counts, send_displs;
    std::vector<int> recv_peers, send_peers;
    std::vector<idx_t> send_indices;
    std::vector<val_t> send_buffer;
    std::vector<MPI_Datatype> recv_types;
    std::vector<std::pair<size_t, size_t>> interior_ranges;
    std::vector<size_t> boundary;
};
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Contiguous ownership preserves global element order and also works when a
// rank owns less than a complete row of the grid.
static idx_t partitionBegin(idx_t total, int rank, int size) {
    return (total / size) * rank + std::min<idx_t>(total % size, rank);
}
static int ownerOf(idx_t index, idx_t total, int size) {
    const idx_t q = total / size, r = total % size;
    const idx_t large_end = (q + 1) * r;
    return index < large_end ? static_cast<int>(index / (q + 1))
                             : static_cast<int>(r + (index - large_end) / q);
}

void buildSquare2D(World& world, int n, int rank, int size) {
    const idx_t total = static_cast<idx_t>(n) * n;
    const idx_t begin = partitionBegin(total, rank, size);
    const idx_t end = partitionBegin(total, rank + 1, size);
    const size_t local = static_cast<size_t>(end - begin);
    world.global_begin = begin;
    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    world.elements_static.resize(local);
    std::vector<std::vector<idx_t>> requests(size);
    for (size_t i = 0; i < local; ++i) {
        const idx_t global = begin + i;
        const idx_t x = global / n, y = global % n;
        ElementStatic& elem = world.elements_static[i];
        elem.material_idx = DEFAULT_MAT_ID;
        // Preserve the original assignment order, including the single-cell case.
        if (x == 0 && y == 0) elem.material_idx = INFLOW_MAT_ID;
        if (x == 0 && y == static_cast<idx_t>(n - 1)) elem.material_idx = OUTFLOW_MAT_ID;
        if (x == static_cast<idx_t>(n - 1) && y == 0) elem.material_idx = OUTFLOW_MAT_ID;
        if (x == static_cast<idx_t>(n - 1) && y == static_cast<idx_t>(n - 1)) elem.material_idx = INFLOW_MAT_ID;
        elem.num_connections = 0;
        auto add = [&](idx_t neighbor) {
            elem.connected_idx[elem.num_connections] = neighbor;
            elem.connected_flux[elem.num_connections++] = 1.0;
            if (neighbor < begin || neighbor >= end)
                requests[ownerOf(neighbor, total, size)].push_back(neighbor);
        };
        if (x + 1 < static_cast<idx_t>(n)) add(global + n);
        if (x > 0) add(global - n);
        if (y + 1 < static_cast<idx_t>(n)) add(global + 1);
        if (y > 0) add(global - 1);
    }
    world.recv_counts.resize(size);
    world.recv_displs.resize(size);
    world.send_counts.resize(size);
    world.send_displs.resize(size);
    std::vector<idx_t> remote_indices;
    for (int p = 0; p < size; ++p) {
        auto& indices = requests[p];
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        world.recv_displs[p] = static_cast<int>(remote_indices.size());
        world.recv_counts[p] = static_cast<int>(indices.size());
        remote_indices.insert(remote_indices.end(), indices.begin(), indices.end());
    }
    MPI_Alltoall(world.recv_counts.data(), 1, MPI_INT, world.send_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int nsend = 0;
    for (int p = 0; p < size; ++p) {
        world.send_displs[p] = nsend;
        nsend += world.send_counts[p];
    }
    world.send_indices.resize(nsend);
    MPI_Alltoallv(remote_indices.data(), world.recv_counts.data(), world.recv_displs.data(), MPI_UINT64_T,
                  world.send_indices.data(), world.send_counts.data(), world.send_displs.data(), MPI_UINT64_T,
                  MPI_COMM_WORLD);
    for (auto& index : world.send_indices) index -= begin;
    world.send_buffer.resize(nsend);
    world.elements_dynamic.resize(local + remote_indices.size());
    world.elements_dynamic_swap.resize(local + remote_indices.size());
    world.recv_types.resize(size, MPI_DATATYPE_NULL);
    for (int p = 0; p < size; ++p) {
        if (world.recv_counts[p]) {
            world.recv_peers.push_back(p);
            // The receive lands directly in the energy fields of the halo slots.
            MPI_Type_vector(world.recv_counts[p], 1, sizeof(ElementDynamic) / sizeof(val_t), MPI_DOUBLE,
                            &world.recv_types[p]);
            MPI_Type_commit(&world.recv_types[p]);
        }
        if (world.send_counts[p]) world.send_peers.push_back(p);
    }
    size_t range_start = 0;
    for (size_t i = 0; i < local; ++i) {
        auto& elem = world.elements_static[i];
        bool boundary = false;
        for (idx_t j = 0; j < elem.num_connections; ++j) {
            const idx_t neighbor = elem.connected_idx[j];
            if (neighbor >= begin && neighbor < end) {
                elem.connected_idx[j] = neighbor - begin;
            } else {
                const int p = ownerOf(neighbor, total, size);
                const auto& indices = requests[p];
                const auto pos = std::lower_bound(indices.begin(), indices.end(), neighbor) - indices.begin();
                elem.connected_idx[j] = local + world.recv_displs[p] + pos;
                boundary = true;
            }
        }
        if (boundary) {
            if (range_start < i) world.interior_ranges.emplace_back(range_start, i);
            world.boundary.push_back(i);
            range_start = i + 1;
        }
    }
    if (range_start < local) world.interior_ranges.emplace_back(range_start, local);
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& world, int n_iters) {
    std::vector<MPI_Request> requests;
    requests.reserve(world.recv_peers.size() + world.send_peers.size());
    const size_t local = world.elements_static.size();
    auto update = [&](size_t i) {
        const ElementStatic& elem_static = world.elements_static[i];
        const ElementDynamic& elem_dyn = world.elements_dynamic[i];
        const Material& mat = world.materials[elem_static.material_idx];
        val_t total_flux = mat.external_flow;
        for (idx_t j = 0; j < elem_static.num_connections; ++j) {
            const ElementDynamic& neighbor_dyn = world.elements_dynamic[elem_static.connected_idx[j]];
            total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
        }
        ElementDynamic& elem_write = world.elements_dynamic_swap[i];
        elem_write.current_energy = elem_dyn.current_energy + total_flux;
        elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
    };
    for (int iter = 0; iter < n_iters; ++iter) {
        requests.clear();
        for (int p : world.recv_peers) {
            MPI_Request req;
            MPI_Irecv(&world.elements_dynamic[local + world.recv_displs[p]].current_energy,
                      1, world.recv_types[p], p, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        for (size_t i = 0; i < world.send_indices.size(); ++i)
            world.send_buffer[i] = world.elements_dynamic[world.send_indices[i]].current_energy;
        for (int p : world.send_peers) {
            MPI_Request req;
            MPI_Isend(world.send_buffer.data() + world.send_displs[p], world.send_counts[p],
                      MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        for (const auto& range : world.interior_ranges)
            for (size_t i = range.first; i < range.second; ++i) update(i);
        if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        for (size_t i : world.boundary) update(i);
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
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
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        uint64_t energy, flux;
        std::memcpy(&energy, &world.elements_dynamic[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &world.elements_dynamic[i].total_flux, sizeof(flux));
        const uint64_t global = world.global_begin + i;
        hash ^= (energy + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global) * 0xbf58476d1ce4e5b9ULL;
    }
    return hash;
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
    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_elems_root = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n_elems_root <= 0 || static_cast<int64_t>(n_elems_root) * n_elems_root > std::numeric_limits<int>::max()) {
        if (rank == 0) printf("Grid size must be positive and its square must fit in an int\n");
        MPI_Finalize(); return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, rank, size);
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t dynamic_mem = static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_time = MPI_Wtime() - start;
    double elapsed = 0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = computeHash(world), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = elapsed * 1000.0 / n_measured_iters;
        const double giga_elems_per_sec = elapsed > 0.0
            ? (static_cast<double>(n_measured_iters) * n_elems) / elapsed / 1e9 : 0.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016lX\n\n", static_cast<unsigned long>(hash));
    }
    int valid = 1;
    if (validate || printResults) {
        std::vector<int> counts(size), displs(size);
        for (int p = 0; p < size; ++p) {
            counts[p] = static_cast<int>(partitionBegin(n_elems, p + 1, size) - partitionBegin(n_elems, p, size));
            displs[p] = static_cast<int>(partitionBegin(n_elems, p, size));
        }
        std::vector<ElementDynamic> all;
        if (rank == 0) all.resize(n_elems);
        MPI_Datatype dynamic_type;
        MPI_Type_contiguous(2, MPI_DOUBLE, &dynamic_type);
        MPI_Type_commit(&dynamic_type);
        MPI_Gatherv(world.elements_dynamic.data(), static_cast<int>(world.elements_static.size()), dynamic_type,
                    all.data(), counts.data(), displs.data(), dynamic_type, 0, MPI_COMM_WORLD);
        MPI_Type_free(&dynamic_type);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energyData;
                energyData.reserve(all.size());
                for (const auto& elem : all) energyData.push_back(elem.current_energy);
                print_results(energyData, "ElementEnergy");
            }
            if (validate) valid = validateResults(all) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    for (auto& type : world.recv_types) if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
    MPI_Finalize();
    return valid ? 0 : 1;
}
