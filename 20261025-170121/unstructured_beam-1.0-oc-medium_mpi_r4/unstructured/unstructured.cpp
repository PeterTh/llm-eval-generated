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
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
    std::vector<ElementDynamic> ghost_top;
    std::vector<ElementDynamic> ghost_bottom;
    std::vector<ElementDynamic> ghost_top_swap;
    std::vector<ElementDynamic> ghost_bottom_swap;
    int x_start = 0;
    int x_end = 0;
    int n_root = 0;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void buildSquare2D(World& world, const int n_elems_root, int rank, int size) {
    world.n_root = n_elems_root;
    const int base = n_elems_root / size;
    const int rem = n_elems_root % size;
    int xs = rank * base + std::min(rank, rem);
    int xe = xs + base + (rank < rem ? 1 : 0);
    world.x_start = xs;
    world.x_end = xe;
    const int n_local_rows = xe - xs;
    const int n_local = n_local_rows * n_elems_root;

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    world.elements_static.resize(n_local);
    world.elements_dynamic.assign(n_local, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.resize(n_local);
    world.ghost_top.assign(n_elems_root, ElementDynamic{0.0, 0.0});
    world.ghost_bottom.assign(n_elems_root, ElementDynamic{0.0, 0.0});
    world.ghost_top_swap.assign(n_elems_root, ElementDynamic{0.0, 0.0});
    world.ghost_bottom_swap.assign(n_elems_root, ElementDynamic{0.0, 0.0});

    for (int i = 0; i < n_local; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }

    for (int x = xs; x < xe; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = (x - xs) * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    if (rank == 0) {
        const int last = n_elems_root - 1;
        world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    }
    // corner assignments by global index
    // handle generically: set materials for the four corners based on ownership
    auto setmat = [&](int gx, int gy, idx_t mat) {
        if (gx >= xs && gx < xe) {
            world.elements_static[(gx - xs) * n_elems_root + gy].material_idx = mat;
        }
    };
    const int last = n_elems_root - 1;
    setmat(0, 0, INFLOW_MAT_ID);
    setmat(0, last, OUTFLOW_MAT_ID);
    setmat(last, 0, OUTFLOW_MAT_ID);
    setmat(last, last, INFLOW_MAT_ID);
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void exchangeHalos(World& world, int rank, int size) {
    int n_root = world.n_root;
    MPI_Request reqs[4];
    int nreq = 0;
    // send top row (owned first row) to rank-1, receive into ghost_top from rank-1
    if (rank > 0) {
        MPI_Isend(world.elements_dynamic.data(), n_root * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(world.ghost_top.data(), n_root * sizeof(ElementDynamic), MPI_BYTE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    if (rank < size - 1) {
        int last_row_offset = (world.x_end - world.x_start - 1) * n_root;
        MPI_Isend(world.elements_dynamic.data() + last_row_offset, n_root * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreq++]);
        MPI_Irecv(world.ghost_bottom.data(), n_root * sizeof(ElementDynamic), MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreq++]);
    }
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
}

void runSimulation(World& world, const int n_iters, int rank, int size) {
    const size_t n_local = world.elements_static.size();
    const int n_root = world.n_root;
    const int xs = world.x_start;
    const int xe = world.x_end;

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world, rank, size);
        for (size_t i = 0; i < n_local; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            val_t total_flux = mat.external_flow;
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                int nx = (int)(neighbor_idx / n_root);
                int ny = (int)(neighbor_idx % n_root);
                const ElementDynamic* neighbor_dyn;
                if (nx >= xs && nx < xe) {
                    neighbor_dyn = &world.elements_dynamic[(nx - xs) * n_root + ny];
                } else if (nx == xs - 1) {
                    neighbor_dyn = &world.ghost_top[ny];
                } else { // nx == xe
                    neighbor_dyn = &world.ghost_bottom[ny];
                }
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], *neighbor_dyn);
            }
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : world.elements_dynamic) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > energy_epsilon) { printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n"); }
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) { printf("  ERROR: Energy extrema are not finite\n"); return false; }
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        hash ^= (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
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

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n_elems_root, rank, size);

    if (rank == 0) {
        const size_t static_mem = (size_t)n_elems * sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    runSimulation(world, n_iters, rank, size);
    auto end = std::chrono::high_resolution_clock::now();
    const long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather
    int n_local = (int)world.elements_dynamic.size();
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&n_local, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> bcounts(size, 0), bdispls(size, 0);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + counts[i-1];
        for (int i = 0; i < size; ++i) bcounts[i] = counts[i] * (int)sizeof(ElementDynamic);
        bdispls[0] = 0;
        for (int i = 1; i < size; ++i) bdispls[i] = bdispls[i-1] + bcounts[i-1];
    }
    std::vector<ElementDynamic> global_dyn;
    if (rank == 0) global_dyn.resize(n_elems);
    MPI_Gatherv(world.elements_dynamic.data(), n_local * (int)sizeof(ElementDynamic), MPI_BYTE,
                global_dyn.data(), bcounts.data(), bdispls.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        const uint64_t hash = computeHash(global_dyn);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(global_dyn.size());
            for (const auto& elem : global_dyn) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }
        if (validate) {
            World wv;
            wv.elements_dynamic = global_dyn;
            if (!validateResults(wv)) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
