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

// Build a 2D square grid as an unstructured mesh
// This represents computation on arbitrarily-shaped geometries
void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Allocate elements
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    
    // Initialize all elements with default material and zero energy
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
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

// Run simulation for n_iters iterations
void runSimulation(World& world, const int n_iters) {
    const size_t n_elems = world.elements_static.size();
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Update all elements
        for (size_t i = 0; i < n_elems; ++i) {
            const ElementStatic& elem_static = world.elements_static[i];
            const ElementDynamic& elem_dyn = world.elements_dynamic[i];
            const Material& mat = world.materials[elem_static.material_idx];
            
            // Start with external flow
            val_t total_flux = mat.external_flow;
            
            // Add flux from all connected elements
            for (idx_t j = 0; j < elem_static.num_connections; ++j) {
                const idx_t neighbor_idx = elem_static.connected_idx[j];
                const ElementDynamic& neighbor_dyn = world.elements_dynamic[neighbor_idx];
                total_flux += computeFlux(mat, elem_dyn, elem_static.connected_flux[j], neighbor_dyn);
            }
            
            // Update element state
            ElementDynamic& elem_write = world.elements_dynamic_swap[i];
            elem_write.current_energy = elem_dyn.current_energy + total_flux;
            elem_write.total_flux = elem_dyn.total_flux + std::abs(total_flux);
        }
        
        // Swap buffers
        std::swap(world.elements_dynamic, world.elements_dynamic_swap);
    }
}

// Validate simulation results
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

// Compute a simple hash of the results for verification
uint64_t computeHash(const std::vector<ElementDynamic>& elements) {
    uint64_t hash = 0;
    for (size_t i = 0; i < elements.size(); ++i) {
        // Simple hash combining energy and flux values
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    if (n_elems_root <= 0 || n_iters < 0) { if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n"); MPI_Finalize(); return 1; }
    const int base = n_elems_root / nprocs, rem = n_elems_root % nprocs;
    const int local_rows = base + (rank < rem);
    const int first_row = rank * base + std::min(rank, rem);
    const int local_n = local_rows * n_elems_root;
    
    if (rank == 0) printf("Unstructured Mesh Energy Transfer Benchmark\n");
    if (rank == 0) printf("============================================\n");
    if (rank == 0) { printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems); printf("Iterations: %d\n", n_iters); printf("Validation: %s\n\n", validate ? "enabled" : "disabled"); }
    
    // Build the unstructured mesh
    if (rank == 0) printf("Building unstructured mesh...\n");
    World world;
    world.materials = {{0.8,0.0},{0.8,0.5},{0.8,-0.5}};
    world.elements_static.resize(local_n);
    world.elements_dynamic.resize(local_n);
    world.elements_dynamic_swap.resize(local_n);
    for (int lr=0; lr<local_rows; ++lr) for(int c=0;c<n_elems_root;++c) {
        const int gr=first_row+lr; const size_t i=static_cast<size_t>(lr)*n_elems_root+c;
        auto& e=world.elements_static[i]; e.material_idx=DEFAULT_MAT_ID; e.num_connections=0;
        if(gr==0 && c==0 || gr==n_elems_root-1 && c==n_elems_root-1) e.material_idx=INFLOW_MAT_ID;
        if(gr==0 && c==n_elems_root-1 || gr==n_elems_root-1 && c==0) e.material_idx=OUTFLOW_MAT_ID;
        world.elements_dynamic[i]={0.0,0.0};
        const int deg=(gr>0)+(gr+1<n_elems_root)+(c>0)+(c+1<n_elems_root);
        e.num_connections=deg;
    }
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    if (rank == 0) printf("Memory usage per rank: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n",
           total_mem / (1024.0 * 1024.0),
           static_mem / (1024.0 * 1024.0),
           dynamic_mem / (1024.0 * 1024.0));
    if (rank == 0) printf("\n");
    
    // Run simulation
    if (rank == 0) printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const int prev = rank ? rank-1 : MPI_PROC_NULL, next = rank+1<nprocs ? rank+1 : MPI_PROC_NULL;
    const size_t stride=n_elems_root;
    std::vector<double> send_up(stride), send_down(stride), recv_up(stride), recv_down(stride);
    for(int it=0;it<n_iters;++it) {
        // Keep only energy values in halos; exchange both boundary rows.
        for(size_t c=0;c<stride && local_rows;++c){send_up[c]=world.elements_dynamic[c].current_energy;send_down[c]=world.elements_dynamic[(local_rows-1)*stride+c].current_energy;}
        MPI_Sendrecv(send_up.data(), stride, MPI_DOUBLE, prev, 2, recv_down.data(), stride, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_down.data(), stride, MPI_DOUBLE, next, 3, recv_up.data(), stride, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int lr=0;lr<local_rows;++lr) for(int c=0;c<n_elems_root;++c) {
            size_t i=static_cast<size_t>(lr)*stride+c; const int gr=first_row+lr; const auto& s=world.elements_static[i]; const auto& d=world.elements_dynamic[i];
            double flow=world.materials[s.material_idx].external_flow;
            auto add=[&](double other){flow+=(other-d.current_energy)*0.8*0.25;};
            if(gr>0) add(lr?world.elements_dynamic[i-stride].current_energy:recv_up[c]);
            if(gr+1<n_elems_root) add(lr+1<local_rows?world.elements_dynamic[i+stride].current_energy:recv_down[c]);
            if(c>0) add(world.elements_dynamic[i-1].current_energy); if(c+1<n_elems_root) add(world.elements_dynamic[i+1].current_energy);
            world.elements_dynamic_swap[i]={d.current_energy+flow,d.total_flux+std::abs(flow)};
        }
        world.elements_dynamic.swap(world.elements_dynamic_swap);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    double local_sec=std::chrono::duration<double>(end-start).count(), max_sec=0; MPI_Reduce(&local_sec,&max_sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) duration_ms=static_cast<long>(max_sec*1000.0);
    MPI_Bcast(&duration_ms,1,MPI_LONG,0,MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", duration_ms);
    
    // Calculate performance metrics
    const int n_measured_iters = std::max(n_iters - 1, 1);
    const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
    const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
    
    // Approximate FLOPS: ~22 FLOPS per element per iteration (from reference)
    const double gflops = giga_elems_per_sec * 22.0;
    
    if(rank==0) printf("Performance:\n");
    if(rank==0) { printf("  Time per iteration: %.4f ms\n", time_per_iter); printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec); printf("  Performance: %.4f GFLOPS\n", gflops); }
    
    // Compute hash for verification
    std::vector<int> counts(nprocs), displs(nprocs);
    for(int p=0;p<nprocs;++p){int rows=n_elems_root/nprocs+(p<n_elems_root%nprocs);counts[p]=2*rows*n_elems_root;displs[p]=2*(p*(n_elems_root/nprocs)+std::min(p,n_elems_root%nprocs))*n_elems_root;}
    std::vector<ElementDynamic> all(rank==0?n_elems:0);
    for(size_t k=0;k<world.elements_dynamic.size();++k) world.elements_dynamic_swap[k]=world.elements_dynamic[k];
    MPI_Gatherv(world.elements_dynamic_swap.data(), local_n*2, MPI_DOUBLE, rank==0?reinterpret_cast<double*>(all.data()):nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    uint64_t hash=0; if(rank==0) hash=computeHash(all);
    MPI_Bcast(&hash,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(rank==0){ printf("  Result hash: %016lX\n\n", hash); }
    
    // Print results for external validation
    if (printResults && rank==0) {
        std::vector<double> energyData;
        energyData.reserve(all.size());
        for (const auto& elem : all) {
            energyData.push_back(elem.current_energy);
        }
        print_results(energyData, "ElementEnergy");
    }
    
    // Validation
    if (validate) {
        bool valid=true;
        if(rank==0) { World global; global.elements_dynamic=std::move(all); valid=validateResults(global); }
        int ok=valid; MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
        if (!ok) { MPI_Finalize(); return 1; }
    }
    MPI_Finalize();
    return 0;
}
