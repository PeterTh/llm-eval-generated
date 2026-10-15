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

// Static connectivity information for each locally owned element.
// Connected indices are local: [0, n_local) owned, [n_local, n_local + n_ghost) ghosts.
struct ElementStatic {
    uint32_t material_idx;
    uint32_t num_connections;
    uint32_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Material type IDs
constexpr uint32_t DEFAULT_MAT_ID = 0;
constexpr uint32_t INFLOW_MAT_ID = 1;
constexpr uint32_t OUTFLOW_MAT_ID = 2;

struct HaloLink {
    int rank;
    size_t recv_offset;               // offset in energy array (ghost region) for receive
    size_t count_recv;
    std::vector<uint32_t> send_idx;   // local indices to send
    std::vector<val_t> send_buf;
};

// Distributed world state: contiguous block of grid rows per rank
struct World {
    int rank = 0, nprocs = 1;
    int n_root = 0;
    idx_t first = 0;      // first owned global element index
    size_t n_local = 0;
    size_t n_ghost = 0;
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> energy, energy_swap;   // size n_local + n_ghost (ghosts at tail)
    std::vector<val_t> flux;                  // size n_local
    std::vector<uint8_t> is_boundary;
    std::vector<HaloLink> links;
};

static void rowRange(int n_root, int nprocs, int r, int& r0, int& r1) {
    const int base = n_root / nprocs, rem = n_root % nprocs;
    r0 = r * base + std::min(r, rem);
    r1 = r0 + base + (r < rem ? 1 : 0);
}

static int rowOwner(int n_root, int nprocs, int row) {
    const int base = n_root / nprocs, rem = n_root % nprocs;
    const int split = rem * (base + 1);
    if (row < split) return row / (base + 1);
    return rem + (row - split) / base;
}

// Build the local part of a 2D square grid as an unstructured mesh
void buildSquare2D(World& world, const int n_elems_root) {
    const int n = n_elems_root;
    world.n_root = n;

    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material

    int r0, r1;
    rowRange(n, world.nprocs, world.rank, r0, r1);
    world.first = (idx_t)r0 * n;
    const size_t nl = (size_t)(r1 - r0) * n;
    world.n_local = nl;

    world.elements_static.resize(nl);

    // Neighbor lists in global indices, in original connection order
    std::vector<idx_t> ghosts;
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    std::vector<idx_t> gconn(nl * 4);
    for (int x = r0; x < r1; ++x) {
        for (int y = 0; y < n; ++y) {
            const size_t li = (size_t)(x - r0) * n + y;
            ElementStatic& elem = world.elements_static[li];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            for (int k = 0; k < 4; ++k) {
                const int nx = x + offsets[k][0];
                const int ny = y + offsets[k][1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n) {
                    const idx_t g = (idx_t)nx * n + ny;
                    gconn[li * 4 + elem.num_connections] = g;
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                    if (nx < r0 || nx >= r1) ghosts.push_back(g);
                }
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    world.n_ghost = ghosts.size();

    // Remap connections to local indices
    world.is_boundary.assign(nl, 0);
    for (size_t li = 0; li < nl; ++li) {
        ElementStatic& elem = world.elements_static[li];
        for (uint32_t j = 0; j < elem.num_connections; ++j) {
            const idx_t g = gconn[li * 4 + j];
            if (g >= world.first && g < world.first + nl) {
                elem.connected_idx[j] = (uint32_t)(g - world.first);
            } else {
                const size_t pos = std::lower_bound(ghosts.begin(), ghosts.end(), g) - ghosts.begin();
                elem.connected_idx[j] = (uint32_t)(nl + pos);
                world.is_boundary[li] = 1;
            }
        }
    }

    // Halo links (adjacency is symmetric, so send sets mirror receive sets)
    for (size_t pos = 0; pos < ghosts.size();) {
        const int owner = rowOwner(n, world.nprocs, (int)(ghosts[pos] / n));
        size_t end = pos;
        while (end < ghosts.size() && rowOwner(n, world.nprocs, (int)(ghosts[end] / n)) == owner) ++end;
        HaloLink link;
        link.rank = owner;
        link.recv_offset = nl + pos;
        link.count_recv = end - pos;
        int s0, s1;
        rowRange(n, world.nprocs, owner, s0, s1);
        for (size_t li = 0; li < nl; ++li) {
            if (!world.is_boundary[li]) continue;
            const ElementStatic& elem = world.elements_static[li];
            for (uint32_t j = 0; j < elem.num_connections; ++j) {
                const idx_t g = gconn[li * 4 + j];
                const int row = (int)(g / n);
                if (row >= s0 && row < s1) { link.send_idx.push_back((uint32_t)li); break; }
            }
        }
        link.send_buf.resize(link.send_idx.size());
        world.links.push_back(std::move(link));
        pos = end;
    }

    world.energy.assign(nl + world.n_ghost, 0.0);
    world.energy_swap.assign(nl + world.n_ghost, 0.0);
    world.flux.assign(nl, 0.0);

    // Set corner elements as inflow/outflow to create interesting dynamics
    const int last = n - 1;
    auto setMat = [&](int x, int y, uint32_t m) {
        if (x >= r0 && x < r1) world.elements_static[(size_t)(x - r0) * n + y].material_idx = m;
    };
    setMat(0, 0, INFLOW_MAT_ID);
    setMat(0, last, OUTFLOW_MAT_ID);
    setMat(last, 0, OUTFLOW_MAT_ID);
    setMat(last, last, INFLOW_MAT_ID);
}

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_local = world.n_local;
    const size_t nl_links = world.links.size();
    std::vector<MPI_Request> reqs(2 * nl_links);

    for (int iter = 0; iter < n_iters; ++iter) {
        val_t* cur = world.energy.data();
        val_t* nxt = world.energy_swap.data();
        val_t* flux = world.flux.data();
        const ElementStatic* st = world.elements_static.data();
        const Material* mats = world.materials.data();
        const uint8_t* bnd = world.is_boundary.data();

        // Post halo exchange
        for (size_t l = 0; l < nl_links; ++l) {
            HaloLink& lk = world.links[l];
            MPI_Irecv(cur + lk.recv_offset, (int)lk.count_recv, MPI_DOUBLE, lk.rank, 0,
                      MPI_COMM_WORLD, &reqs[l]);
        }
        for (size_t l = 0; l < nl_links; ++l) {
            HaloLink& lk = world.links[l];
            for (size_t k = 0; k < lk.send_idx.size(); ++k) lk.send_buf[k] = cur[lk.send_idx[k]];
            MPI_Isend(lk.send_buf.data(), (int)lk.send_buf.size(), MPI_DOUBLE, lk.rank, 0,
                      MPI_COMM_WORLD, &reqs[nl_links + l]);
        }

        auto update = [&](size_t i) {
            const ElementStatic& es = st[i];
            const Material& mat = mats[es.material_idx];
            const val_t e = cur[i];
            val_t total_flux = mat.external_flow;
            for (uint32_t j = 0; j < es.num_connections; ++j) {
                total_flux += (cur[es.connected_idx[j]] - e) * mat.transfer_coeff *
                              es.connected_flux[j] * 0.25;
            }
            nxt[i] = e + total_flux;
            flux[i] = flux[i] + std::abs(total_flux);
        };

        // Interior while communication is in flight
        for (size_t i = 0; i < n_local; ++i) if (!bnd[i]) update(i);

        if (nl_links) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        for (size_t i = 0; i < n_local; ++i) if (bnd[i]) update(i);

        std::swap(world.energy, world.energy_swap);
    }
}

// Validate simulation results (energy/flux hold the full global arrays)
bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& fluxv) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += fluxv[i];
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

// Compute a simple hash of the local results (XOR-combinable across ranks)
uint64_t computeHash(const World& w) {
    uint64_t hash = 0;
    for (size_t li = 0; li < w.n_local; ++li) {
        const uint64_t i = w.first + li;
        uint64_t e, f;
        memcpy(&e, &w.energy[li], sizeof e);
        memcpy(&f, &w.flux[li], sizeof f);
        hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
        hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (n_elems_root < 1) {
        if (rank == 0) printf("Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n_elems_root * n_elems_root;
    
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
        printf("\n");
        printf("Building unstructured mesh...\n");
    }
    World world;
    world.rank = rank;
    world.nprocs = nprocs;
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage (summed over ranks)
    unsigned long long loc_mem[2] = {
        (unsigned long long)world.elements_static.size() * sizeof(ElementStatic),
        (unsigned long long)(world.energy.size() + world.energy_swap.size() + world.flux.size()) *
            sizeof(val_t)};
    unsigned long long tot_mem[2];
    MPI_Reduce(loc_mem, tot_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const size_t static_mem = tot_mem[0];
        const size_t dynamic_mem = tot_mem[1];
        const size_t total_mem = static_mem + dynamic_mem;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
               total_mem / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0),
               dynamic_mem / (1024.0 * 1024.0));
        printf("\n");
        printf("Running simulation...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t0;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long duration_ms = (long)(elapsed * 1000.0);
    
    // Compute hash for verification
    uint64_t local_hash = computeHash(world), hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    
    // Gather global arrays only when needed
    std::vector<val_t> energyData, fluxData;
    if (validate || printResults) {
        std::vector<int> counts(nprocs), displs(nprocs);
        for (int r = 0; r < nprocs; ++r) {
            int a, b;
            rowRange(n_elems_root, nprocs, r, a, b);
            counts[r] = (b - a) * n_elems_root;
            displs[r] = a * n_elems_root;
        }
        if (rank == 0) {
            energyData.resize(n_elems);
            fluxData.resize(n_elems);
        }
        MPI_Gatherv(world.energy.data(), (int)world.n_local, MPI_DOUBLE, energyData.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(world.flux.data(), (int)world.n_local, MPI_DOUBLE, fluxData.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * (double)n_elems) / (duration_ms / 1000.0) / 1e9;
        
        // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016lX\n", (unsigned long)hash);
        printf("\n");
        
        if (printResults) {
            print_results(energyData, "ElementEnergy");
        }
        
        if (validate && !validateResults(energyData, fluxData)) ret = 1;
    }
    
    MPI_Finalize();
    return ret;
}
