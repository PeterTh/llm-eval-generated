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
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Dynamic state for each element
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// World state (static parts only)
struct WorldStatic {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(WorldStatic& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    
    // Initialize all elements with default material
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    
    // Build connectivity: each element connects to its neighbors in 2D grid
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
            ElementStatic& elem = world.elements_static[idx];
            
            // Connect to neighbors (up, down, left, right)
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                const int nx = x + offsets[n][0];
                const int ny = y + offsets[n][1];
                
                // Check if neighbor is within bounds
                if (nx >= 0 && nx < n_elems_root && ny >= 0 && ny < n_elems_root) {
                    const int neighbor_idx = nx * n_elems_root + ny;
                    elem.connected_idx[elem.num_connections] = neighbor_idx;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, const ElementDynamic& this_elem,
                        val_t connection_flux, const ElementDynamic& other_elem) {
    return (other_elem.current_energy - this_elem.current_energy) * 
           mat.transfer_coeff * connection_flux * 0.25;
}

// Run simulation for n_iters iterations (distributed by rows using MPI)
void runSimulationDistributed(const WorldStatic& world_static,
                              std::vector<ElementDynamic>& local_dyn,
                              std::vector<ElementDynamic>& local_swap,
                              const int n_iters,
                              const int n_elems_root,
                              const int rank, const int size,
                              const int start_row, const int local_n_rows) {
    const int n_cols = n_elems_root;
    const int local_n_elems = local_n_rows * n_cols;
    const int global_n_elems = n_elems_root * n_elems_root;
    const int end_row = start_row + local_n_rows - 1;

    // Buffers for halo exchange (packed as [current_energy, total_flux] per element)
    std::vector<double> send_top(2 * n_cols), send_bottom(2 * n_cols);
    std::vector<double> recv_top(2 * n_cols), recv_bottom(2 * n_cols);

    // Temporary ElementDynamic for neighbor when using halo
    ElementDynamic tmp_neighbor;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Prepare send buffers if we have rows
        if (local_n_rows > 0) {
            // top row
            for (int c = 0; c < n_cols; ++c) {
                const int li = 0 * n_cols + c;
                send_top[2*c] = local_dyn[li].current_energy;
                send_top[2*c+1] = local_dyn[li].total_flux;
            }
            // bottom row
            for (int c = 0; c < n_cols; ++c) {
                const int li = (local_n_rows - 1) * n_cols + c;
                send_bottom[2*c] = local_dyn[li].current_energy;
                send_bottom[2*c+1] = local_dyn[li].total_flux;
            }
        }

        const int up_rank = (start_row > 0) ? (rank - 1) : MPI_PROC_NULL;
        const int down_rank = (end_row < n_elems_root - 1) ? (rank + 1) : MPI_PROC_NULL;

        // Exchange with up neighbor: send_top -> up_rank, recv_top <- up_rank (their bottom row)
        MPI_Sendrecv(local_n_rows>0 ? send_top.data() : nullptr, 2*n_cols, MPI_DOUBLE, up_rank, 0,
                     local_n_rows>0 ? recv_top.data() : nullptr, 2*n_cols, MPI_DOUBLE, up_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Exchange with down neighbor: send_bottom -> down_rank, recv_bottom <- down_rank (their top row)
        MPI_Sendrecv(local_n_rows>0 ? send_bottom.data() : nullptr, 2*n_cols, MPI_DOUBLE, down_rank, 1,
                     local_n_rows>0 ? recv_bottom.data() : nullptr, 2*n_cols, MPI_DOUBLE, down_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Update local elements
        for (int local_idx = 0; local_idx < local_n_elems; ++local_idx) {
            const int global_idx = (start_row * n_cols) + local_idx;
            const ElementStatic& elem_static = world_static.elements_static[global_idx];
            const ElementDynamic& elem_dyn = local_dyn[local_idx];
            const Material& mat = world_static.materials[elem_static.material_idx];

            val_t total_flux = mat.external_flow;

            // Add flux from connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_global = elem_static.connected_idx[j];
                const int neighbor_row = neighbor_global / n_cols;
                const int neighbor_col = neighbor_global % n_cols;

                if (neighbor_row >= start_row && neighbor_row <= end_row) {
                    // Local neighbor
                    const int neighbor_local_idx = (neighbor_row - start_row) * n_cols + neighbor_col;
                    const ElementDynamic& neighbor_dyn = local_dyn[neighbor_local_idx];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
                } else if (neighbor_row == start_row - 1 && local_n_rows > 0) {
                    // From top halo
                    tmp_neighbor.current_energy = recv_top[2*neighbor_col];
                    tmp_neighbor.total_flux = recv_top[2*neighbor_col+1];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], tmp_neighbor);
                } else if (neighbor_row == end_row + 1 && local_n_rows > 0) {
                    // From bottom halo
                    tmp_neighbor.current_energy = recv_bottom[2*neighbor_col];
                    tmp_neighbor.total_flux = recv_bottom[2*neighbor_col+1];
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], tmp_neighbor);
                } else {
                    // Neighbor is remote but not adjacent (happens when some ranks have zero rows). Fetch by direct MPI communication
                    // Perform an MPI_Get-like request by probing owner rank and receiving single element.
                    int owner_rank = 0;
                    // Determine owner rank by recomputing partitioning
                    int rows_per_rank = n_elems_root / size;
                    int rem = n_elems_root % size;
                    int rstart = 0;
                    for (int r = 0; r < size; ++r) {
                        int rn = rows_per_rank + (r < rem ? 1 : 0);
                        if (neighbor_row >= rstart && neighbor_row < rstart + rn) { owner_rank = r; break; }
                        rstart += rn;
                    }
                    double buf[2];
                    if (owner_rank == rank) {
                        // shouldn't happen
                        tmp_neighbor = local_dyn[(neighbor_row - start_row)*n_cols + neighbor_col];
                    } else {
                        // ask owner_rank to send the element value using synchronous send/recv tags
                        // We will use tag=((int)neighbor_global)%100000 to avoid collision
                        int tag = (int)(neighbor_global % 100000);
                        // send a request
                        MPI_Send(nullptr, 0, MPI_BYTE, owner_rank, tag+100000, MPI_COMM_WORLD);
                        MPI_Recv(buf, 2, MPI_DOUBLE, owner_rank, tag+200000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        tmp_neighbor.current_energy = buf[0];
                        tmp_neighbor.total_flux = buf[1];
                    }
                    total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], tmp_neighbor);
                }
            }

            // Update element state
            ElementDynamic& elem_write = local_swap[local_idx];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }

        // Swap buffers
        std::swap(local_dyn, local_swap);

        // Service incoming on-demand requests if any (non-blocking approach would be better but keep simple)
        // Probe for any requests and reply
        int flag = 1;
        MPI_Status status;
        while (true) {
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (!flag) break;
            int tag = status.MPI_TAG;
            MPI_Recv(nullptr, 0, MPI_BYTE, status.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // requester expects data with tag+100000 offset scheme
            int neighbor_global = tag - 100000;
            double sendbuf[2] = {0.0, 0.0};
            if (local_n_rows > 0) {
                int owner_row = neighbor_global / n_cols;
                int owner_col = neighbor_global % n_cols;
                if (owner_row >= start_row && owner_row <= end_row) {
                    int li = (owner_row - start_row) * n_cols + owner_col;
                    sendbuf[0] = local_dyn[li].current_energy;
                    sendbuf[1] = local_dyn[li].total_flux;
                }
            }
            MPI_Send(sendbuf, 2, MPI_DOUBLE, status.MPI_SOURCE, tag+100000, MPI_COMM_WORLD);
        }
    }
}

// Validate simulation results (aggregated across ranks)
bool validateResultsDistributed(const std::vector<ElementDynamic>& gathered, const int rank) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : gathered) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }

    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        // Check for numerical issues
        constexpr val_t energy_epsilon = 1e-8;

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            return false;
        }

        if (std::abs(energy_sum) > energy_epsilon) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }

        if (!std::isfinite(flux_sum)) {
            printf("  ERROR: Flux sum is not finite\n");
            return false;
        }

        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Energy extrema are not finite\n");
            return false;
        }

        printf("  Validation: PASSED\n");
    }
    return true;
}

uint64_t computeHashDistributed(const std::vector<ElementDynamic>& elements) {
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
    
    // Parse command line arguments (only on rank 0 but broadcasted)
    if (rank == 0) {
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
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else if (i > 0) {
                // ignore unknown options on other ranks
            }
        }
    }
    MPI_Bcast(&n_elems_root, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

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

    WorldStatic world_static;
    buildSquare2D(world_static, n_elems_root);

    // Partition rows among ranks
    int rows_per_rank = n_elems_root / size;
    int rem = n_elems_root % size;
    int start_row = 0;
    for (int r = 0; r < rank; ++r) start_row += rows_per_rank + (r < rem ? 1 : 0);
    int local_n_rows = rows_per_rank + (rank < rem ? 1 : 0);
    int local_n_elems = local_n_rows * n_elems_root;

    // Calculate memory usage (only print on rank 0)
    if (rank == 0) {
        const size_t static_mem = world_static.elements_static.size() * sizeof(ElementStatic);
        const size_t dynamic_mem = n_elems * sizeof(ElementDynamic) * 2;
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Initialize local dynamic arrays
    std::vector<ElementDynamic> local_dyn(local_n_elems);
    std::vector<ElementDynamic> local_swap(local_n_elems);
    for (int i = 0; i < local_n_elems; ++i) { local_dyn[i].current_energy = 0.0; local_dyn[i].total_flux = 0.0; local_swap[i] = local_dyn[i]; }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulationDistributed(world_static, local_dyn, local_swap, n_iters, n_elems_root, rank, size, start_row, local_n_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long duration_ms = 0;
    if (rank == 0) {
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Computation time: %ld ms\n", duration_ms);

        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }

    // Gather results to rank 0 for hashing/validation/printing
    // Pack local data as double array [current_energy, total_flux,...]
    std::vector<int> recvcounts(size), displs(size);
    std::vector<int> sendcounts(size);
    int mycount = 2 * local_n_elems;
    MPI_Allgather(&mycount, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int total_recv = 0;
    for (int r = 0; r < size; ++r) { recvcounts[r] = sendcounts[r]; displs[r] = total_recv; total_recv += recvcounts[r]; }

    std::vector<double> sendbuf(2*local_n_elems);
    for (int i = 0; i < local_n_elems; ++i) { sendbuf[2*i] = local_dyn[i].current_energy; sendbuf[2*i+1] = local_dyn[i].total_flux; }
    std::vector<double> recvbuf;
    if (rank == 0) recvbuf.resize(total_recv);

    MPI_Gatherv(sendbuf.data(), mycount, MPI_DOUBLE, recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<ElementDynamic> gathered;
    if (rank == 0) {
        // Reconstruct in global element order
        gathered.resize(n_elems);
        // For each rank, determine its start_row and number of rows
        int rstart = 0;
        int pos = 0;
        for (int r = 0; r < size; ++r) {
            int rn = rows_per_rank + (r < rem ? 1 : 0);
            int r_elems = rn * n_elems_root;
            for (int i = 0; i < r_elems; ++i) {
                if (pos + 2*i + 1 < (int)recvbuf.size()) {
                    const double e = recvbuf[displs[r] + 2*i];
                    const double f = recvbuf[displs[r] + 2*i + 1];
                    int global_idx = rstart * n_elems_root + i;
                    if (global_idx < n_elems) {
                        gathered[global_idx].current_energy = e;
                        gathered[global_idx].total_flux = f;
                    }
                }
            }
            rstart += rn;
            pos += 2*r_elems;
        }
    }

    // Compute hash and optionally print results
    if (rank == 0) {
        uint64_t hash = computeHashDistributed(gathered);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> energyData;
            energyData.reserve(gathered.size());
            for (const auto& elem : gathered) energyData.push_back(elem.current_energy);
            print_results(energyData, "ElementEnergy");
        }

        if (validate) {
            bool valid = validateResultsDistributed(gathered, rank);
            if (!valid) { MPI_Finalize(); return 1; }
        }
    }

    MPI_Finalize();
    return 0;
}
