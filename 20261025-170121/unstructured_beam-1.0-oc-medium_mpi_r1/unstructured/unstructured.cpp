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
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

int world_rank, world_size;

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

void buildLocal(World& world, const int n, int& local_x_start, int& local_nx) {
    const int rows_per = (n + world_size - 1) / world_size;
    local_x_start = world_rank * rows_per;
    int local_x_end = std::min(local_x_start + rows_per, n);
    local_nx = local_x_end - local_x_start;

    world.materials.clear();
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    const int ext_rows = local_nx + 2;
    world.elements_static.assign(local_nx * n, ElementStatic{});
    world.elements_dynamic.assign(ext_rows * n, ElementDynamic{0.0, 0.0});
    world.elements_dynamic_swap.assign(ext_rows * n, ElementDynamic{0.0, 0.0});

    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    for (int lx = 0; lx < local_nx; ++lx) {
        const int gx = local_x_start + lx;
        for (int y = 0; y < n; ++y) {
            ElementStatic& elem = world.elements_static[lx * n + y];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            for (int k = 0; k < MAX_CONNECTIONS; ++k) { elem.connected_idx[k]=0; elem.connected_flux[k]=0.0; }
            for (int nn = 0; nn < 4; ++nn) {
                const int nx = gx + offsets[nn][0];
                const int ny = y + offsets[nn][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const int ext_row = nx - local_x_start + 1;
                    const int neighbor_ext = ext_row * n + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_ext;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    if (local_x_start == 0) {
        world.elements_static[0 * n + 0].material_idx = INFLOW_MAT_ID;
        world.elements_static[0 * n + (n-1)].material_idx = OUTFLOW_MAT_ID;
    }
    if (local_x_end == n) {
        const int last = local_nx - 1;
        world.elements_static[last * n + 0].material_idx = OUTFLOW_MAT_ID;
        world.elements_static[last * n + (n-1)].material_idx = INFLOW_MAT_ID;
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& world, const int n_iters, int local_nx, int n) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // halo exchange current_energy (whole struct rows)
        if (world_rank > 0) {
            MPI_Sendrecv(&world.elements_dynamic[1*n], n*sizeof(ElementDynamic), MPI_BYTE,
                         world_rank-1, 0,
                         &world.elements_dynamic[0*n], n*sizeof(ElementDynamic), MPI_BYTE,
                         world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (world_rank < world_size - 1) {
            MPI_Sendrecv(&world.elements_dynamic[local_nx*n], n*sizeof(ElementDynamic), MPI_BYTE,
                         world_rank+1, 1,
                         &world.elements_dynamic[(local_nx+1)*n], n*sizeof(ElementDynamic), MPI_BYTE,
                         world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        for (int lx = 0; lx < local_nx; ++lx) {
            for (int y = 0; y < n; ++y) {
                const ElementStatic& elem_static = world.elements_static[lx*n + y];
                const int i = (lx+1)*n + y;
                const ElementDynamic& elem_dyn = world.elements_dynamic[i];
                const Material& mat = world.materials[elem_static.material_idx];

                val_t total_flux = mat.external_flow;
                for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                    const idx_t neighbor_idx = elem_static.connected_idx[j];
                    const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                }
                ElementDynamic& elem_write = world.elements_dynamic_swap[i];
                elem_write.current_energy = elem_dyn.current_energy + total_flux;
                elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
            }
        }
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
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

bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            const int n = n_elems_root;

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n*n);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\nBuilding unstructured mesh...\n");
    }

    int local_x_start, local_nx;
    World world;
    buildLocal(world, n, local_x_start, local_nx);

    if (world_rank == 0) {
        const size_t static_mem = (size_t)n*n*sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n*n*sizeof(ElementDynamic)*2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0*1024.0),
               static_mem / (1024.0*1024.0),
               dynamic_mem / (1024.0*1024.0));
        printf("\nRunning simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, local_nx, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // gather
    std::vector<int> counts(world_size), displs(world_size);
    int local_count = local_nx * n;
    int local_bytes = local_count * (int)sizeof(ElementDynamic);
    MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r-1] + counts[r-1];
    int total_bytes = displs[world_size-1] + counts[world_size-1];

    std::vector<char> global_buf;
    if (world_rank == 0) global_buf.resize(total_bytes);

    std::vector<char> sendbuf(local_bytes);
    for (int lx = 0; lx < local_nx; ++lx)
        for (int y = 0; y < n; ++y)
            *reinterpret_cast<ElementDynamic*>(&sendbuf[(lx*n+y)*sizeof(ElementDynamic)]) = world.elements_dynamic[(lx+1)*n + y];

    MPI_Gatherv(sendbuf.data(), local_bytes, MPI_BYTE,
                global_buf.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        World w;
        w.materials = world.materials;
        // build static global not needed for results
        w.elements_dynamic.resize((size_t)n*n);
        for (int i = 0; i < n*n; ++i)
            w.elements_dynamic[i] = *reinterpret_cast<ElementDynamic*>(&global_buf[i*sizeof(ElementDynamic)]);
        w.elements_dynamic_swap = w.elements_dynamic;

        auto duration_ms2 = duration_ms;
        printf("Computation time: %ld ms\n", duration_ms2);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms2) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (size_t)n*n) / (duration_ms2 / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(w.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(w.elements_dynamic.size());
            for (const auto& elem : w.elements_dynamic) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResults(w);
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    const int n = n_elems_root;

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n, n, n*n);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\nBuilding unstructured mesh...\n");
    }

    int local_x_start, local_nx;
    World world;
    buildLocal(world, n, local_x_start, local_nx);

    if (world_rank == 0) {
        const size_t static_mem = (size_t)n*n*sizeof(ElementStatic);
        const size_t dynamic_mem = (size_t)n*n*sizeof(ElementDynamic)*2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0*1024.0),
               static_mem / (1024.0*1024.0),
               dynamic_mem / (1024.0*1024.0));
        printf("\nRunning simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters, local_nx, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // gather
    std::vector<int> counts(world_size), displs(world_size);
    int local_count = local_nx * n;
    int local_bytes = local_count * (int)sizeof(ElementDynamic);
    MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r-1] + counts[r-1];
    int total_bytes = displs[world_size-1] + counts[world_size-1];

    std::vector<char> sendbuf(local_bytes);
    for (int lx = 0; lx < local_nx; ++lx)
        for (int y = 0; y < n; ++y)
            *reinterpret_cast<ElementDynamic*>(&sendbuf[(lx*n+y)*sizeof(ElementDynamic)]) = world.elements_dynamic[(lx+1)*n + y];

    std::vector<char> global_buf;
    if (world_rank == 0) global_buf.resize(total_bytes);

    MPI_Gatherv(sendbuf.data(), local_bytes, MPI_BYTE,
                global_buf.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        World w;
        w.materials = world.materials;
        w.elements_dynamic.resize((size_t)n*n);
        for (int i = 0; i < n*n; ++i)
            w.elements_dynamic[i] = *reinterpret_cast<ElementDynamic*>(&global_buf[i*sizeof(ElementDynamic)]);
        w.elements_dynamic_swap = w.elements_dynamic;

        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (size_t)n*n) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        const uint64_t hash = computeHash(w.elements_dynamic);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(w.elements_dynamic.size());
            for (const auto& elem : w.elements_dynamic) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResults(w);
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
