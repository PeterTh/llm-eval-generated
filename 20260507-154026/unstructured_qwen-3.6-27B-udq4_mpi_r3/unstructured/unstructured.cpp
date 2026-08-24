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
    idx_t connected_idx[MAX_CONNECTIONS];     // Global indices of connected elements
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
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build local portion of a 2D square grid as an unstructured mesh
void buildSquare2D(World& world, int n_elems_root, int local_start_x, int local_rows) {
    const int n_local_elems = local_rows * n_elems_root;

    // Initialize materials (identical on all ranks)
    world.materials.reserve(3);
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    // Allocate local elements
    world.elements_static.resize(n_local_elems);
    world.elements_dynamic.resize(n_local_elems);
    world.elements_dynamic_swap.resize(n_local_elems);

    // Initialize all local elements with default material and zero energy
    for (int i = 0; i < n_local_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }

    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int lx = 0; lx < local_rows; ++lx) {
        const int gx = local_start_x + lx;
        for (int y = 0; y < n_elems_root; ++y) {
            const int local_idx = lx * n_elems_root + y;
            ElementStatic& elem = world.elements_static[local_idx];

            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

            for (int n = 0; n < 4; ++n) {
                const int nx = gx + offsets[n][0];
                const int ny = y + offsets[n][1];

                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = (idx_t)neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    auto set_corner = [&](int gx, int gy, idx_t mat_id) {
        if (gx >= local_start_x && gx < local_start_x + local_rows &&
            gy >= 0 && gy < n_elems_root) {
            const int li = (gx - local_start_x) * n_elems_root + gy;
            world.elements_static[li].material_idx = mat_id;
        }
    };
    set_corner(0, 0, INFLOW_MAT_ID);
    set_corner(0, last, OUTFLOW_MAT_ID);
    set_corner(last, 0, OUTFLOW_MAT_ID);
    set_corner(last, last, INFLOW_MAT_ID);
}

// Run simulation for n_iters iterations with MPI halo exchange
void runSimulation(World& world, int n_iters, int n_elems_root,
                   int local_start_x, int local_rows, int rank, int num_ranks) {
    const int local_end_x = local_start_x + local_rows;
    const size_t n_local_elems = world.elements_static.size();

    // Halo buffers for boundary rows
    std::vector<val_t> top_halo(n_elems_root, 0.0);
    std::vector<val_t> bottom_halo(n_elems_root, 0.0);
    std::vector<val_t> top_send(n_elems_root, 0.0);
    std::vector<val_t> bottom_send(n_elems_root, 0.0);

    // Communication partners
    const int top_partner    = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int bottom_partner = (rank < num_ranks - 1) ? rank + 1 : MPI_PROC_NULL;

    // Element computation kernel
    auto compute_element = [&](size_t i) {
        const ElementStatic& es = world.elements_static[i];
        const ElementDynamic& ed = world.elements_dynamic[i];
        const Material& mat = world.materials[es.material_idx];

        val_t total_flux = mat.external_flow;

        for (idx_t j = 0; j < es.num_connections; ++j) {
            const idx_t nidx = es.connected_idx[j];
            const int nx = static_cast<int>(nidx / n_elems_root);
            const int ny = static_cast<int>(nidx % n_elems_root);

            val_t n_energy;
            if (nx == local_start_x - 1) {
                n_energy = top_halo[ny];
            } else if (nx == local_end_x) {
                n_energy = bottom_halo[ny];
            } else {
                // Convert global index to local index
                const size_t local_nidx = static_cast<size_t>(nx - local_start_x) * n_elems_root + ny;
                n_energy = world.elements_dynamic[local_nidx].current_energy;
            }

            total_flux += (n_energy - ed.current_energy) *
                          mat.transfer_coeff * es.connected_flux[j] * 0.25;
        }

        ElementDynamic& ew = world.elements_dynamic_swap[i];
        ew.current_energy = ed.current_energy + total_flux;
        ew.total_flux = ed.total_flux + std::abs(total_flux);
    };

    for (int iter = 0; iter < n_iters; ++iter) {
        // Prepare halo data from boundary rows
        if (n_local_elems > 0) {
            if (rank > 0) {
                for (int y = 0; y < n_elems_root; ++y) {
                    top_send[y] = world.elements_dynamic[y].current_energy;
                }
            }
            if (rank < num_ranks - 1) {
                const int llr = (local_rows - 1) * n_elems_root;
                for (int y = 0; y < n_elems_root; ++y) {
                    bottom_send[y] = world.elements_dynamic[llr + y].current_energy;
                }
            }
        }

        // Post non-blocking halo exchange
        MPI_Request requests[4];
        int nreq = 0;

        if (rank > 0)
            MPI_Isend(top_send.data(), n_elems_root, MPI_DOUBLE, top_partner, 0,
                      MPI_COMM_WORLD, &requests[nreq++]);
        if (rank < num_ranks - 1)
            MPI_Irecv(bottom_halo.data(), n_elems_root, MPI_DOUBLE, bottom_partner, 0,
                      MPI_COMM_WORLD, &requests[nreq++]);
        if (rank < num_ranks - 1)
            MPI_Isend(bottom_send.data(), n_elems_root, MPI_DOUBLE, bottom_partner, 1,
                      MPI_COMM_WORLD, &requests[nreq++]);
        if (rank > 0)
            MPI_Irecv(top_halo.data(), n_elems_root, MPI_DOUBLE, top_partner, 1,
                      MPI_COMM_WORLD, &requests[nreq++]);

        // Compute interior elements (overlap computation with communication)
        if (n_local_elems > 0 && local_rows > 2) {
            for (int lx = 1; lx < local_rows - 1; ++lx) {
                const size_t row_start = static_cast<size_t>(lx) * n_elems_root;
                for (int y = 0; y < n_elems_root; ++y) {
                    compute_element(row_start + y);
                }
            }
        }

        // Wait for halo exchange to complete
        if (nreq > 0)
            MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

        // Compute boundary elements (now have halo data)
        if (n_local_elems > 0) {
            // Top boundary (first local row)
            for (int y = 0; y < n_elems_root; ++y) {
                compute_element(static_cast<size_t>(y));
            }
            // Bottom boundary (last local row)
            if (local_rows > 1) {
                const size_t last_row = static_cast<size_t>(local_rows - 1) * n_elems_root;
                for (int y = 0; y < n_elems_root; ++y) {
                    compute_element(last_row + y);
                }
            }
        }

        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results with MPI reduction
bool validateResults(const World& world, int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();

    for (const auto& elem : world.elements_dynamic) {
        local_energy_sum += elem.current_energy;
        local_flux_sum += elem.total_flux;
        local_energy_max = std::max(elem.current_energy, local_energy_max);
        local_energy_min = std::min(elem.current_energy, local_energy_min);
    }

    val_t energy_sum, flux_sum, energy_max, energy_min;

    MPI_Allreduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    bool valid = true;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

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

        if (valid) {
            printf("  Validation: PASSED\n");
        }
    }

    return valid;
}

// Compute hash with MPI reduction (XOR is commutative/associative)
uint64_t computeHash(const World& world, int n_elems_root, int local_start_x) {
    uint64_t local_hash = 0;
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        const int gx = static_cast<int>(i / n_elems_root) + local_start_x;
        const int gy = static_cast<int>(i % n_elems_root);
        const idx_t global_idx = static_cast<idx_t>(gx) * n_elems_root + gy;

        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&world.elements_dynamic[i].total_flux);
        local_hash ^= (*e_ptr + global_idx) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (*f_ptr + global_idx) * 0xbf58476d1ce4e5b9ULL;
    }

    uint64_t global_hash;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
    return global_hash;
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int n_elems_root = 512;
    int n_iters = 10;
    int validate_flag = 0;
    int printResults_flag = 0;

    // Parse command line arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n_elems_root = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                n_iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_flag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_flag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
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
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = validate_flag != 0;
    const bool printResults = printResults_flag != 0;
    const int n_elems = n_elems_root * n_elems_root;

    // Domain decomposition: 1D block along rows
    const int rows_per_rank = n_elems_root / num_ranks;
    const int remainder = n_elems_root % num_ranks;
    int local_start_x = 0;
    for (int r = 0; r < rank; ++r) {
        local_start_x += rows_per_rank + (r < remainder ? 1 : 0);
    }
    const int local_rows = rows_per_rank + (rank < remainder ? 1 : 0);
    const int n_local_elems = local_rows * n_elems_root;

    // Print info on rank 0
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("MPI ranks: %d\n", num_ranks);
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    buildSquare2D(world, n_elems_root, local_start_x, local_rows);

    // Calculate total memory usage across all ranks
    const size_t local_static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t local_dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    size_t total_static_mem = 0, total_dynamic_mem = 0;
    MPI_Allreduce(&local_static_mem, &total_static_mem, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_dynamic_mem, &total_dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    const size_t total_mem = total_static_mem + total_dynamic_mem;

    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               total_static_mem / (1024.0 * 1024.0),
               total_dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation with synchronized timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    runSimulation(world, n_iters, n_elems_root, local_start_x, local_rows, rank, num_ranks);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    const double duration_ms = (end_time - start_time) * 1000.0;
    double max_duration_ms;
    MPI_Allreduce(&duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Calculate performance metrics (using total grid size)
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = max_duration_ms / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * static_cast<double>(n_elems)) /
                                      (max_duration_ms / 1000.0) / 1e9;
    const double gflops = giga_elems_per_sec * 22.0;

    // Compute hash
    const uint64_t hash = computeHash(world, n_elems_root, local_start_x);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_duration_ms);

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", static_cast<unsigned long>(hash));
        printf("\n");
    }

    // Print results for external validation (gather to rank 0)
    if (printResults) {
        // Compute recvcounts and displs for MPI_Gatherv (same on all ranks)
        std::vector<int> recvcounts(num_ranks);
        std::vector<int> displs(num_ranks);
        int disp = 0;
        for (int r = 0; r < num_ranks; ++r) {
            const int rows_r = n_elems_root / num_ranks + (r < n_elems_root % num_ranks ? 1 : 0);
            recvcounts[r] = rows_r * n_elems_root;
            displs[r] = disp;
            disp += recvcounts[r];
        }

        // Extract local energies
        std::vector<double> local_energies(n_local_elems);
        for (int i = 0; i < n_local_elems; ++i) {
            local_energies[i] = world.elements_dynamic[i].current_energy;
        }

        if (rank == 0) {
            std::vector<double> energyData(n_elems);
            MPI_Gatherv(local_energies.data(), n_local_elems, MPI_DOUBLE,
                        energyData.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
            print_results(energyData, "ElementEnergy");
        } else {
            MPI_Gatherv(local_energies.data(), n_local_elems, MPI_DOUBLE,
                        nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                        0, MPI_COMM_WORLD);
        }
    }

    // Validation
    int global_valid = 1;
    if (validate) {
        global_valid = validateResults(world, rank) ? 1 : 0;
    }
    int local_valid_flag = global_valid;
    MPI_Allreduce(&local_valid_flag, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_valid ? 0 : 1;
}
