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
using lidx_t = int32_t;  // Rank-local element index (owned + ghost)

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 8;

// Material properties for energy transfer
struct Material {
    val_t transfer_coeff;  // Energy transfer coefficient
    val_t external_flow;   // External energy source/sink
};

// Static connectivity information for each element (global indices)
struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];     // Indices of connected elements
    val_t connected_flux[MAX_CONNECTIONS];    // Flux coefficients for each connection
};

// Contiguous range of local elements [begin, end)
struct Range {
    lidx_t begin;
    lidx_t end;
};

// Halo exchange with a single peer rank
struct Neighbor {
    int rank;
    std::vector<lidx_t> send_idx;   // Local owned elements to send to this peer
    lidx_t recv_offset;             // Start of this peer's ghosts in the energy array
    lidx_t recv_count;
};

// Distributed world state: each rank owns a contiguous block of global elements.
// Connectivity is stored in compressed (CSR) form with rank-local indices;
// ghost elements (owned by other ranks) are stored after the owned elements.
struct World {
    std::vector<Material> materials;

    idx_t n_global = 0;
    idx_t first = 0;         // Global index of first owned element
    lidx_t n_local = 0;      // Number of owned elements
    lidx_t n_ghost = 0;      // Number of ghost elements

    std::vector<uint32_t> material_idx;   // per owned element
    std::vector<lidx_t> conn_offset;      // n_local + 1
    std::vector<lidx_t> conn_idx;         // local indices of connected elements
    std::vector<val_t> conn_flux;         // flux coefficient per connection

    // Dynamic state (SoA): energy includes ghost slots, flux only owned
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> flux;
    std::vector<val_t> flux_swap;

    std::vector<Neighbor> neighbors;
    std::vector<Range> interior;   // elements depending only on owned data
    std::vector<Range> boundary;   // elements depending on ghost data
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// Block distribution of global elements over ranks
static inline idx_t blockStart(idx_t n, int nranks, int r) {
    const idx_t base = n / nranks, rem = n % nranks;
    return base * r + std::min<idx_t>(r, rem);
}

static inline int blockOwner(idx_t n, int nranks, idx_t g) {
    const idx_t base = n / nranks, rem = n % nranks;
    const idx_t split = rem * (base + 1);
    if (g < split) return static_cast<int>(g / (base + 1));
    return static_cast<int>(rem + (g - split) / base);
}

// Build global connectivity of a single element of the 2D square grid
static void buildSquare2DElement(ElementStatic& elem, const idx_t idx, const idx_t n_elems_root) {
    const int64_t n = static_cast<int64_t>(n_elems_root);
    const int64_t x = static_cast<int64_t>(idx / n_elems_root);
    const int64_t y = static_cast<int64_t>(idx % n_elems_root);
    elem.material_idx = DEFAULT_MAT_ID;
    elem.num_connections = 0;

    // Connect to neighbors (up, down, left, right)
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int k = 0; k < 4; ++k) {
        const int64_t nx = x + offsets[k][0];
        const int64_t ny = y + offsets[k][1];
        if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
            elem.connected_idx[elem.num_connections] = static_cast<idx_t>(nx * n + ny);
            elem.connected_flux[elem.num_connections] = 1.0;
            elem.num_connections++;
        }
    }

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int64_t last = n - 1;
    if ((x == 0 && y == 0) || (x == last && y == last)) elem.material_idx = INFLOW_MAT_ID;
    else if ((x == 0 && y == last) || (x == last && y == 0)) elem.material_idx = OUTFLOW_MAT_ID;
}

// Build the locally owned part of a 2D square grid as an unstructured mesh,
// and set up the halo exchange pattern with other ranks.
void buildSquare2D(World& world, const int n_elems_root, MPI_Comm comm) {
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    const idx_t n_root = static_cast<idx_t>(n_elems_root);
    const idx_t n_elems = n_root * n_root;

    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    world.n_global = n_elems;
    world.first = blockStart(n_elems, nranks, rank);
    const idx_t last_excl = blockStart(n_elems, nranks, rank + 1);
    const lidx_t n_local = static_cast<lidx_t>(last_excl - world.first);
    world.n_local = n_local;

    // Build global connectivity of owned elements
    std::vector<ElementStatic> elems(n_local);
    for (lidx_t i = 0; i < n_local; ++i) {
        buildSquare2DElement(elems[i], world.first + i, n_root);
    }

    // Collect ghost (remote) element indices, sorted => grouped by owner rank
    std::vector<idx_t> ghosts;
    for (lidx_t i = 0; i < n_local; ++i) {
        for (idx_t j = 0; j < elems[i].num_connections; ++j) {
            const idx_t g = elems[i].connected_idx[j];
            if (g < world.first || g >= last_excl) ghosts.push_back(g);
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    world.n_ghost = static_cast<lidx_t>(ghosts.size());

    auto toLocal = [&](idx_t g) -> lidx_t {
        if (g >= world.first && g < last_excl) return static_cast<lidx_t>(g - world.first);
        const auto it = std::lower_bound(ghosts.begin(), ghosts.end(), g);
        return n_local + static_cast<lidx_t>(it - ghosts.begin());
    };

    // Compressed local connectivity, classify interior/boundary elements
    world.material_idx.resize(n_local);
    world.conn_offset.resize(n_local + 1);
    world.conn_offset[0] = 0;
    std::vector<char> is_boundary(n_local, 0);
    for (lidx_t i = 0; i < n_local; ++i) {
        world.material_idx[i] = static_cast<uint32_t>(elems[i].material_idx);
        for (idx_t j = 0; j < elems[i].num_connections; ++j) {
            const lidx_t l = toLocal(elems[i].connected_idx[j]);
            if (l >= n_local) is_boundary[i] = 1;
            world.conn_idx.push_back(l);
            world.conn_flux.push_back(elems[i].connected_flux[j]);
        }
        world.conn_offset[i + 1] = static_cast<lidx_t>(world.conn_idx.size());
    }
    elems.clear();
    elems.shrink_to_fit();

    for (lidx_t i = 0; i < n_local;) {
        lidx_t j = i;
        while (j < n_local && is_boundary[j] == is_boundary[i]) ++j;
        (is_boundary[i] ? world.boundary : world.interior).push_back(Range{i, j});
        i = j;
    }

    // Determine which ghosts come from which rank
    std::vector<int> recv_counts(nranks, 0), recv_displs(nranks, 0);
    for (const idx_t g : ghosts) recv_counts[blockOwner(n_elems, nranks, g)]++;
    for (int r = 1; r < nranks; ++r) recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];

    // Exchange request lists: tell owners which of their elements we need
    std::vector<int> send_counts(nranks, 0), send_displs(nranks, 0);
    MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, comm);
    for (int r = 1; r < nranks; ++r) send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    const int total_send = nranks > 0 ? send_displs[nranks - 1] + send_counts[nranks - 1] : 0;
    std::vector<idx_t> requested(total_send);
    static_assert(sizeof(idx_t) == sizeof(uint64_t));
    MPI_Alltoallv(ghosts.data(), recv_counts.data(), recv_displs.data(), MPI_UINT64_T,
                  requested.data(), send_counts.data(), send_displs.data(), MPI_UINT64_T, comm);

    for (int r = 0; r < nranks; ++r) {
        if (recv_counts[r] == 0 && send_counts[r] == 0) continue;
        Neighbor nb;
        nb.rank = r;
        nb.recv_offset = n_local + recv_displs[r];
        nb.recv_count = recv_counts[r];
        nb.send_idx.resize(send_counts[r]);
        for (int k = 0; k < send_counts[r]; ++k) {
            nb.send_idx[k] = static_cast<lidx_t>(requested[send_displs[r] + k] - world.first);
        }
        world.neighbors.push_back(std::move(nb));
    }

    // Initialize all elements with zero energy
    world.energy.assign(static_cast<size_t>(n_local) + world.n_ghost, 0.0);
    world.energy_swap.assign(static_cast<size_t>(n_local) + world.n_ghost, 0.0);
    world.flux.assign(n_local, 0.0);
    world.flux_swap.assign(n_local, 0.0);
}

// Compute energy flux between two elements
inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

// Update a contiguous range of owned elements
static inline void updateRange(const World& world, const Range rg,
                               const val_t* __restrict energy, const val_t* __restrict flux,
                               val_t* __restrict energy_out, val_t* __restrict flux_out) {
    const Material* __restrict materials = world.materials.data();
    const uint32_t* __restrict mat_idx = world.material_idx.data();
    const lidx_t* __restrict off = world.conn_offset.data();
    const lidx_t* __restrict cidx = world.conn_idx.data();
    const val_t* __restrict cflux = world.conn_flux.data();

    for (lidx_t i = rg.begin; i < rg.end; ++i) {
        const Material& mat = materials[mat_idx[i]];
        const val_t e = energy[i];

        // Start with external flow
        val_t total_flux = mat.external_flow;

        // Add flux from all connected elements
        for (lidx_t j = off[i]; j < off[i + 1]; ++j) {
            total_flux += computeFlux(mat, e, cflux[j], energy[cidx[j]]);
        }

        // Update element state
        energy_out[i] = e + total_flux;
        flux_out[i] = flux[i] + std::abs(total_flux);
    }
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters, MPI_Comm comm) {
    const size_t n_nb = world.neighbors.size();
    std::vector<std::vector<val_t>> send_bufs(n_nb);
    for (size_t k = 0; k < n_nb; ++k) send_bufs[k].resize(world.neighbors[k].send_idx.size());
    std::vector<MPI_Request> reqs(2 * n_nb);

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* energy = world.energy.data();

        // Start halo exchange of current energies
        for (size_t k = 0; k < n_nb; ++k) {
            const Neighbor& nb = world.neighbors[k];
            MPI_Irecv(energy + nb.recv_offset, nb.recv_count, MPI_DOUBLE, nb.rank, 0, comm, &reqs[k]);
        }
        for (size_t k = 0; k < n_nb; ++k) {
            const Neighbor& nb = world.neighbors[k];
            val_t* buf = send_bufs[k].data();
            const size_t cnt = nb.send_idx.size();
            for (size_t s = 0; s < cnt; ++s) buf[s] = energy[nb.send_idx[s]];
            MPI_Isend(buf, static_cast<int>(cnt), MPI_DOUBLE, nb.rank, 0, comm, &reqs[n_nb + k]);
        }

        // Update elements that only depend on owned data while messages are in flight
        for (const Range& rg : world.interior) {
            updateRange(world, rg, energy, world.flux.data(),
                        world.energy_swap.data(), world.flux_swap.data());
        }

        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);

        // Update elements depending on ghost data
        for (const Range& rg : world.boundary) {
            updateRange(world, rg, energy, world.flux.data(),
                        world.energy_swap.data(), world.flux_swap.data());
        }

        // Swap buffers
        std::swap(world.energy, world.energy_swap);
        std::swap(world.flux, world.flux_swap);
    }
}

// Validate simulation results (on gathered global data, sequential order)
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
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

// Compute a simple hash of the owned results (combined across ranks with XOR)
uint64_t computeHash(const World& world) {
    uint64_t hash = 0;
    for (lidx_t l = 0; l < world.n_local; ++l) {
        const uint64_t i = world.first + static_cast<uint64_t>(l);
        uint64_t e_bits, f_bits;
        std::memcpy(&e_bits, &world.energy[l], sizeof(e_bits));
        std::memcpy(&f_bits, &world.flux[l], sizeof(f_bits));
        hash ^= (e_bits + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f_bits + i) * 0xbf58476d1ce4e5b9ULL;
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    const bool root = (rank == 0);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identically on all ranks)
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (root) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
        printf("\n");
        
        // Build the unstructured mesh
        printf("Building unstructured mesh...\n");
    }
    World world;
    buildSquare2D(world, n_elems_root, comm);
    
    // Calculate memory usage (summed over all ranks)
    {
        double local_mem[2];
        local_mem[0] = static_cast<double>(world.material_idx.size() * sizeof(uint32_t) +
                                           world.conn_offset.size() * sizeof(lidx_t) +
                                           world.conn_idx.size() * sizeof(lidx_t) +
                                           world.conn_flux.size() * sizeof(val_t));
        local_mem[1] = static_cast<double>((world.energy.size() + world.energy_swap.size() +
                                            world.flux.size() + world.flux_swap.size()) * sizeof(val_t));
        double mem[2];
        MPI_Reduce(local_mem, mem, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
        if (root) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
                   (mem[0] + mem[1]) / (1024.0 * 1024.0),
                   mem[0] / (1024.0 * 1024.0),
                   mem[1] / (1024.0 * 1024.0));
            printf("\n");
            
            // Run simulation
            printf("Running simulation...\n");
            fflush(stdout);
        }
    }
    
    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();
    
    runSimulation(world, n_iters, comm);
    
    double elapsed = MPI_Wtime() - t_start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
    const long duration_ms = static_cast<long>(elapsed * 1000.0);
    
    // Compute hash for verification (XOR is order independent)
    uint64_t local_hash = computeHash(world);
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    if (root) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (static_cast<double>(n_measured_iters) * n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Gather global results on root for output / validation
    int ret = 0;
    if (printResults || validate) {
        std::vector<int> counts(nranks), displs(nranks);
        const int my_count = world.n_local;
        MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
        std::vector<val_t> energy, flux;
        if (root) {
            for (int r = 1; r < nranks; ++r) displs[r] = displs[r - 1] + counts[r - 1];
            energy.resize(world.n_global);
            if (validate) flux.resize(world.n_global);
        }
        MPI_Gatherv(world.energy.data(), my_count, MPI_DOUBLE,
                    energy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        if (validate) {
            MPI_Gatherv(world.flux.data(), my_count, MPI_DOUBLE,
                        flux.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
        }
        if (root) {
            // Print results for external validation
            if (printResults) {
                print_results(energy, "ElementEnergy");
            }
            
            // Validation
            if (validate && !validateResults(energy, flux)) {
                ret = 1;
            }
        }
        MPI_Bcast(&ret, 1, MPI_INT, 0, comm);
    }
    
    MPI_Finalize();
    return ret;
}
