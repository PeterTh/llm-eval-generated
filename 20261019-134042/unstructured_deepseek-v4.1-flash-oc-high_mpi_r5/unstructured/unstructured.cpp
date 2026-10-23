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
constexpr int LOCAL_MAX_CONNECTIONS = 4;

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

// Compact per-element data used by the distributed solver
struct LocalElement {
    val_t transfer_coeff;
    val_t external_flow;
    uint32_t num_connections;
    uint32_t connected_idx[LOCAL_MAX_CONNECTIONS];
    val_t connected_flux[LOCAL_MAX_CONNECTIONS];
};

struct Range {
    int start;
    int len;
};

inline Range partitionRange(int n, int parts, int coord) {
    const int base = n / parts;
    const int rem = n % parts;
    const int start = coord * base + std::min(coord, rem);
    const int len = base + (coord < rem ? 1 : 0);
    return Range{start, len};
}

// Distributed world state: elements are owned in a 2D block decomposition.
struct World {
    std::vector<Material> materials;
    std::vector<LocalElement> elements_static;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux;
    std::vector<val_t> flux_swap;

    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[2] = {1, 1};
    int coord[2] = {0, 0};
    int nbr[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int n = 0;
    int x0 = 0;
    int y0 = 0;
    int lx = 0;
    int ly = 0;
    int dimx = 0;
    int dimy = 0;
    std::vector<val_t> sbuf[4];
    std::vector<val_t> rbuf[4];
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

inline void addConnection(LocalElement& elem, uint32_t neighbor_idx) {
    elem.connected_idx[elem.num_connections] = neighbor_idx;
    elem.connected_flux[elem.num_connections] = 1.0;
    ++elem.num_connections;
}

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n = n_elems_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    int dims[2] = {0, 0};
    MPI_Dims_create(world.size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &world.cart);
    MPI_Comm_rank(world.cart, &world.rank);
    MPI_Comm_size(world.cart, &world.size);
    world.dims[0] = dims[0];
    world.dims[1] = dims[1];
    MPI_Cart_coords(world.cart, world.rank, 2, world.coord);
    MPI_Cart_shift(world.cart, 0, 1, &world.nbr[0], &world.nbr[1]);
    MPI_Cart_shift(world.cart, 1, 1, &world.nbr[2], &world.nbr[3]);

    const Range rx = partitionRange(n, world.dims[0], world.coord[0]);
    const Range ry = partitionRange(n, world.dims[1], world.coord[1]);
    world.n = n;
    world.x0 = rx.start;
    world.y0 = ry.start;
    world.lx = rx.len;
    world.ly = ry.len;
    world.dimx = world.lx + 2;
    world.dimy = world.ly + 2;

    const size_t local_size = static_cast<size_t>(world.dimx) * world.dimy;
    world.elements_static.assign(local_size, LocalElement{});
    world.energy.assign(local_size, 0.0);
    world.energy_swap.assign(local_size, 0.0);
    world.flux.assign(local_size, 0.0);
    world.flux_swap.assign(local_size, 0.0);

    for (int i = 1; i <= world.lx; ++i) {
        const int gx = world.x0 + i - 1;
        for (int j = 1; j <= world.ly; ++j) {
            const int gy = world.y0 + j - 1;
            LocalElement& elem =
                world.elements_static[static_cast<size_t>(i) * world.dimy + j];

            idx_t mat_id = DEFAULT_MAT_ID;
            if ((gx == 0 && gy == 0) || (gx == n - 1 && gy == n - 1)) {
                mat_id = INFLOW_MAT_ID;
            } else if ((gx == 0 && gy == n - 1) || (gx == n - 1 && gy == 0)) {
                mat_id = OUTFLOW_MAT_ID;
            }
            const Material& mat = world.materials[mat_id];

            elem.transfer_coeff = mat.transfer_coeff;
            elem.external_flow = mat.external_flow;
            elem.num_connections = 0;

            // Keep the original connection order: +x, -x, +y, -y
            if (gx + 1 < n) {
                addConnection(elem, static_cast<uint32_t>(
                    static_cast<size_t>(i + 1) * world.dimy + j));
            }
            if (gx - 1 >= 0) {
                addConnection(elem, static_cast<uint32_t>(
                    static_cast<size_t>(i - 1) * world.dimy + j));
            }
            if (gy + 1 < n) {
                addConnection(elem, static_cast<uint32_t>(
                    static_cast<size_t>(i) * world.dimy + (j + 1)));
            }
            if (gy - 1 >= 0) {
                addConnection(elem, static_cast<uint32_t>(
                    static_cast<size_t>(i) * world.dimy + (j - 1)));
            }
        }
    }

    const int row_len = std::max(world.ly, 1);
    const int col_len = std::max(world.lx, 1);
    world.sbuf[0].resize(row_len);
    world.sbuf[1].resize(row_len);
    world.rbuf[0].resize(row_len);
    world.rbuf[1].resize(row_len);
    world.sbuf[2].resize(col_len);
    world.sbuf[3].resize(col_len);
    world.rbuf[2].resize(col_len);
    world.rbuf[3].resize(col_len);
}

// Exchange the current energy values across block boundaries
void exchangeHalos(World& world) {
    const int dimy = world.dimy;
    const int lx = world.lx;
    const int ly = world.ly;
    val_t* energy = world.energy.data();

    MPI_Request reqs[8];
    int num_reqs = 0;

    MPI_Irecv(energy + 1, ly, MPI_DOUBLE, world.nbr[0], 0, world.cart,
              &reqs[num_reqs++]);
    MPI_Irecv(energy + static_cast<size_t>(lx + 1) * dimy + 1, ly, MPI_DOUBLE,
              world.nbr[1], 1, world.cart, &reqs[num_reqs++]);
    MPI_Irecv(world.rbuf[2].data(), lx, MPI_DOUBLE, world.nbr[2], 2,
              world.cart, &reqs[num_reqs++]);
    MPI_Irecv(world.rbuf[3].data(), lx, MPI_DOUBLE, world.nbr[3], 3,
              world.cart, &reqs[num_reqs++]);

    for (int i = 1; i <= lx; ++i) {
        const size_t p = static_cast<size_t>(i) * dimy;
        world.sbuf[2][i - 1] = energy[p + 1];
        world.sbuf[3][i - 1] = energy[p + ly];
    }

    MPI_Isend(energy + dimy + 1, ly, MPI_DOUBLE, world.nbr[0], 1, world.cart,
              &reqs[num_reqs++]);
    const int down_row = (lx > 0 ? lx : 0);
    MPI_Isend(energy + static_cast<size_t>(down_row) * dimy + 1, ly, MPI_DOUBLE,
              world.nbr[1], 0, world.cart, &reqs[num_reqs++]);
    MPI_Isend(world.sbuf[2].data(), lx, MPI_DOUBLE, world.nbr[2], 3,
              world.cart, &reqs[num_reqs++]);
    MPI_Isend(world.sbuf[3].data(), lx, MPI_DOUBLE, world.nbr[3], 2,
              world.cart, &reqs[num_reqs++]);

    MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

    for (int i = 1; i <= lx; ++i) {
        const size_t p = static_cast<size_t>(i) * dimy;
        energy[p] = world.rbuf[2][i - 1];
        energy[p + ly + 1] = world.rbuf[3][i - 1];
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const int dimy = world.dimy;
    const int lx = world.lx;
    const int ly = world.ly;

    for (int iter = 0; iter < n_iters; ++iter) {
        exchangeHalos(world);

        const val_t* energy = world.energy.data();
        const val_t* flux = world.flux.data();
        val_t* energy_out = world.energy_swap.data();
        val_t* flux_out = world.flux_swap.data();
        const LocalElement* elements = world.elements_static.data();

        for (int i = 1; i <= lx; ++i) {
            size_t p = static_cast<size_t>(i) * dimy + 1;
            for (int j = 0; j < ly; ++j, ++p) {
                const LocalElement& elem = elements[p];
                const val_t self = energy[p];

                // Start with external flow
                val_t total_flux = elem.external_flow;

                // Add flux from all connected elements
                const uint32_t nconn = elem.num_connections;
                for (uint32_t c = 0; c < nconn; ++c) {
                    const val_t other = energy[elem.connected_idx[c]];
                    total_flux += (other - self) * elem.transfer_coeff *
                                  elem.connected_flux[c] * 0.25;
                }

                // Update element state
                energy_out[p] = self + total_flux;
                flux_out[p] = flux[p] + std::abs(total_flux);
            }
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

std::vector<val_t> packInterior(const World& world,
                                const std::vector<val_t>& data) {
    std::vector<val_t> packed(static_cast<size_t>(world.lx) * world.ly);
    size_t k = 0;
    for (int i = 1; i <= world.lx; ++i) {
        size_t p = static_cast<size_t>(i) * world.dimy + 1;
        for (int j = 0; j < world.ly; ++j, ++p) {
            packed[k++] = data[p];
        }
    }
    return packed;
}

void gatherArray(const World& world, const std::vector<val_t>& local,
                 std::vector<val_t>& global) {
    const int count = static_cast<int>(local.size());
    std::vector<int> counts(world.size);
    std::vector<int> displs(world.size);
    int offset = 0;
    for (int r = 0; r < world.size; ++r) {
        int c[2];
        MPI_Cart_coords(world.cart, r, 2, c);
        const Range rx = partitionRange(world.n, world.dims[0], c[0]);
        const Range ry = partitionRange(world.n, world.dims[1], c[1]);
        counts[r] = rx.len * ry.len;
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<val_t> recv;
    if (world.rank == 0) {
        recv.resize(static_cast<size_t>(world.n) * world.n);
    }

    MPI_Gatherv(local.data(), count, MPI_DOUBLE,
                world.rank == 0 ? recv.data() : nullptr,
                world.rank == 0 ? counts.data() : nullptr,
                world.rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, world.cart);

    if (world.rank == 0) {
        global.resize(static_cast<size_t>(world.n) * world.n);
        for (int r = 0; r < world.size; ++r) {
            int c[2];
            MPI_Cart_coords(world.cart, r, 2, c);
            const Range rx = partitionRange(world.n, world.dims[0], c[0]);
            const Range ry = partitionRange(world.n, world.dims[1], c[1]);
            const val_t* src = recv.data() + displs[r];
            for (int i = 0; i < rx.len; ++i) {
                for (int j = 0; j < ry.len; ++j) {
                    global[static_cast<size_t>(rx.start + i) * world.n +
                           (ry.start + j)] = *src++;
                }
            }
        }
    }
}

// Validate simulation results
bool validateResults(const std::vector<val_t>& energy,
                     const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();

    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }

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
        // Don't fail validation as this can happen with external flows
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
    return true;
}

// Compute a simple hash of the local results for verification
uint64_t computeLocalHash(const World& world) {
    uint64_t hash = 0;
    for (int i = 1; i <= world.lx; ++i) {
        const int gx = world.x0 + i - 1;
        size_t p = static_cast<size_t>(i) * world.dimy + 1;
        for (int j = 0; j < world.ly; ++j, ++p) {
            const int gy = world.y0 + j;
            const size_t idx = static_cast<size_t>(gx) * world.n + gy;
            const uint64_t* e_ptr =
                reinterpret_cast<const uint64_t*>(&world.energy[p]);
            const uint64_t* f_ptr =
                reinterpret_cast<const uint64_t*>(&world.flux[p]);
            hash ^= (*e_ptr + idx) * 0x9e3779b97f4a7c15ULL;
            hash ^= (*f_ptr + idx) * 0xbf58476d1ce4e5b9ULL;
        }
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

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    const int n_elems = n_elems_root * n_elems_root;

    if (world_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root,
               n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
        printf("Building unstructured mesh...\n");
    }

    // Build the distributed unstructured mesh
    World world;
    world.rank = world_rank;
    world.size = world_size;
    buildSquare2D(world, n_elems_root);

    // Calculate memory usage
    const size_t static_mem =
        static_cast<size_t>(n_elems) * sizeof(ElementStatic);
    const size_t dynamic_mem =
        static_cast<size_t>(n_elems) * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (world.rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runSimulation(world, n_iters);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    MPI_Allreduce(MPI_IN_PLACE, &duration_ms, 1, MPI_LONG, MPI_MAX, world.cart);

    uint64_t local_hash = computeLocalHash(world);
    uint64_t global_hash = 0;
    MPI_Allreduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR,
                  world.cart);

    if (world.rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter =
            static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec =
            (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;

        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);

        // Compute hash for verification
        printf("  Result hash: %016lX\n", global_hash);
        printf("\n");
    }

    std::vector<val_t> global_energy;
    std::vector<val_t> global_flux;
    if (printResults || validate) {
        const std::vector<val_t> local_energy = packInterior(world, world.energy);
        gatherArray(world, local_energy, global_energy);
        if (validate) {
            const std::vector<val_t> local_flux = packInterior(world, world.flux);
            gatherArray(world, local_flux, global_flux);
        }
    }

    // Print results for external validation
    if (printResults && world.rank == 0) {
        print_results(global_energy, "ElementEnergy");
    }

    // Validation
    if (validate) {
        int valid_local = 1;
        if (world.rank == 0) {
            valid_local = validateResults(global_energy, global_flux) ? 1 : 0;
        }
        int valid_global = 0;
        MPI_Bcast(&valid_local, 1, MPI_INT, 0, world.cart);
        valid_global = valid_local;
        if (!valid_global) {
            MPI_Comm_free(&world.cart);
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Comm_free(&world.cart);
    MPI_Finalize();
    return 0;
}
