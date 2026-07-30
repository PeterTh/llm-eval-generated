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
#include <omp.h>
#include <cuda_runtime.h>

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
    std::vector<ElementDynamic> elements_dynamic;  // Interior + ghost cells
    
    // MPI domain decomposition
    int mpi_rank = 0;
    int mpi_size = 1;
    int proc_grid_x = 1;
    int proc_grid_y = 1;
    int proc_x = 0;
    int proc_y = 0;
    int local_nx = 0;
    int local_ny = 0;
    int global_x_offset = 0;
    int global_y_offset = 0;
    int n_interior = 0;
    int n_ghosts = 0;
    int n_total = 0;
    
    // Ghost cell offsets in elements_dynamic array
    int ghost_left_offset = 0;   // y-1 direction
    int ghost_right_offset = 0;  // y+1 direction
    int ghost_top_offset = 0;    // x-1 direction
    int ghost_bottom_offset = 0; // x+1 direction
    
    // Neighbor ranks
    int rank_left = MPI_PROC_NULL;
    int rank_right = MPI_PROC_NULL;
    int rank_top = MPI_PROC_NULL;
    int rank_bottom = MPI_PROC_NULL;
    
    // CUDA device pointers
    ElementStatic* d_elements_static = nullptr;
    ElementDynamic* d_elements_dynamic = nullptr;
    ElementDynamic* d_elements_dynamic_swap = nullptr;
    Material* d_materials = nullptr;
    val_t* d_boundary_left = nullptr;
    val_t* d_boundary_right = nullptr;
    val_t* d_boundary_top = nullptr;
    val_t* d_boundary_bottom = nullptr;
    
    // MPI buffers
    std::vector<val_t> recv_left, recv_right, recv_top, recv_bottom;
};

// Material type IDs
constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

// CUDA kernel for simulation
__global__ void simulation_kernel(
    const ElementStatic* __restrict__ es_arr,
    const ElementDynamic* __restrict__ ed_arr,
    ElementDynamic* __restrict__ ed_swap,
    const Material* __restrict__ mat_arr,
    int n_interior)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_interior) return;
    
    const ElementStatic& es = es_arr[i];
    const ElementDynamic& ed = ed_arr[i];
    const Material& mat = mat_arr[es.material_idx];
    
    val_t total_flux = mat.external_flow;
    
    for (idx_t j = 0; j < es.num_connections; ++j) {
        const idx_t ni = es.connected_idx[j];
        const ElementDynamic& nd = ed_arr[ni];
        total_flux += (nd.current_energy - ed.current_energy) * 
                      mat.transfer_coeff * es.connected_flux[j] * 0.25;
    }
    
    ed_swap[i].current_energy = ed.current_energy + total_flux;
    ed_swap[i].total_flux = ed.total_flux + fabs(total_flux);
}

// CUDA kernel to extract boundary elements
__global__ void extract_boundary_kernel(
    const ElementDynamic* __restrict__ ed,
    val_t* __restrict__ boundary_left,
    val_t* __restrict__ boundary_right,
    val_t* __restrict__ boundary_top,
    val_t* __restrict__ boundary_bottom,
    int local_nx, int local_ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < local_nx) {
        boundary_left[i] = ed[i * local_ny].current_energy;
        boundary_right[i] = ed[i * local_ny + local_ny - 1].current_energy;
    }
    
    if (i < local_ny) {
        boundary_top[i] = ed[i].current_energy;
        boundary_bottom[i] = ed[(local_nx - 1) * local_ny + i].current_energy;
    }
}

// Compute 2D processor grid dimensions
void computeProcessorGrid(int size, int& grid_x, int& grid_y) {
    grid_x = static_cast<int>(std::sqrt(static_cast<double>(size)));
    while (size % grid_x != 0) grid_x--;
    grid_y = size / grid_x;
}

// Build a 2D square grid as an unstructured mesh with MPI domain decomposition
void buildSquare2D(World& world, const int n_elems_root) {
    // Initialize materials
    world.materials.emplace_back(Material{0.8, 0.0});    // Default material
    world.materials.emplace_back(Material{0.8, 0.5});    // Inflow material
    world.materials.emplace_back(Material{0.8, -0.5});   // Outflow material
    
    // Compute domain decomposition
    computeProcessorGrid(world.mpi_size, world.proc_grid_x, world.proc_grid_y);
    world.proc_x = world.mpi_rank / world.proc_grid_y;
    world.proc_y = world.mpi_rank % world.proc_grid_y;
    
    // Compute local domain size
    int base_nx = n_elems_root / world.proc_grid_x;
    int base_ny = n_elems_root / world.proc_grid_y;
    int rem_x = n_elems_root % world.proc_grid_x;
    int rem_y = n_elems_root % world.proc_grid_y;
    
    world.local_nx = base_nx + (world.proc_x < rem_x ? 1 : 0);
    world.local_ny = base_ny + (world.proc_y < rem_y ? 1 : 0);
    
    world.global_x_offset = world.proc_x * base_nx + std::min(world.proc_x, rem_x);
    world.global_y_offset = world.proc_y * base_ny + std::min(world.proc_y, rem_y);
    
    world.n_interior = world.local_nx * world.local_ny;
    
    // Ghost cell layout (appended to elements_dynamic)
    world.ghost_left_offset = world.n_interior;
    world.ghost_right_offset = world.n_interior + world.local_nx;
    world.ghost_top_offset = world.n_interior + 2 * world.local_nx;
    world.ghost_bottom_offset = world.n_interior + 2 * world.local_nx + world.local_ny;
    world.n_ghosts = 2 * world.local_nx + 2 * world.local_ny;
    world.n_total = world.n_interior + world.n_ghosts;
    
    // Neighbor ranks
    if (world.proc_y > 0)
        world.rank_left = world.proc_x * world.proc_grid_y + (world.proc_y - 1);
    if (world.proc_y < world.proc_grid_y - 1)
        world.rank_right = world.proc_x * world.proc_grid_y + (world.proc_y + 1);
    if (world.proc_x > 0)
        world.rank_top = (world.proc_x - 1) * world.proc_grid_y + world.proc_y;
    if (world.proc_x < world.proc_grid_x - 1)
        world.rank_bottom = (world.proc_x + 1) * world.proc_grid_y + world.proc_y;
    
    // Allocate elements
    world.elements_static.resize(world.n_interior);
    world.elements_dynamic.resize(world.n_total, ElementDynamic{0.0, 0.0});
    
    // Initialize all elements with default material
    for (int i = 0; i < world.n_interior; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
    }
    
    // Build connectivity
    for (int x = 0; x < world.local_nx; ++x) {
        for (int y = 0; y < world.local_ny; ++y) {
            int gx = world.global_x_offset + x;
            int gy = world.global_y_offset + y;
            int idx = x * world.local_ny + y;
            ElementStatic& elem = world.elements_static[idx];
            
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            
            for (int n = 0; n < 4; ++n) {
                int ngx = gx + offsets[n][0];
                int ngy = gy + offsets[n][1];
                
                if (ngx >= 0 && ngx < n_elems_root && ngy >= 0 && ngy < n_elems_root) {
                    int lx = ngx - world.global_x_offset;
                    int ly = ngy - world.global_y_offset;
                    
                    if (lx >= 0 && lx < world.local_nx && ly >= 0 && ly < world.local_ny) {
                        // Interior neighbor
                        elem.connected_idx[elem.num_connections] = lx * world.local_ny + ly;
                    } else {
                        // Ghost cell neighbor
                        int ghost_idx;
                        if (ngx < world.global_x_offset) {
                            ghost_idx = world.ghost_top_offset + ly;
                        } else if (ngx >= world.global_x_offset + world.local_nx) {
                            ghost_idx = world.ghost_bottom_offset + ly;
                        } else if (ngy < world.global_y_offset) {
                            ghost_idx = world.ghost_left_offset + lx;
                        } else {
                            ghost_idx = world.ghost_right_offset + lx;
                        }
                        elem.connected_idx[elem.num_connections] = ghost_idx;
                    }
                    elem.connected_flux[elem.num_connections] = 1.0;
                    elem.num_connections++;
                }
            }
        }
    }
    
    // Set corner elements as inflow/outflow
    if (world.global_x_offset == 0 && world.global_y_offset == 0)
        world.elements_static[0].material_idx = INFLOW_MAT_ID;
    if (world.global_x_offset == 0 && world.global_y_offset + world.local_ny == n_elems_root)
        world.elements_static[world.local_ny - 1].material_idx = OUTFLOW_MAT_ID;
    if (world.global_x_offset + world.local_nx == n_elems_root && world.global_y_offset == 0)
        world.elements_static[(world.local_nx - 1) * world.local_ny].material_idx = OUTFLOW_MAT_ID;
    if (world.global_x_offset + world.local_nx == n_elems_root && 
        world.global_y_offset + world.local_ny == n_elems_root)
        world.elements_static[(world.local_nx - 1) * world.local_ny + world.local_ny - 1].material_idx = INFLOW_MAT_ID;
    
    // Allocate MPI receive buffers
    world.recv_left.resize(world.local_nx);
    world.recv_right.resize(world.local_nx);
    world.recv_top.resize(world.local_ny);
    world.recv_bottom.resize(world.local_ny);
}

// Exchange ghost cells using MPI
void exchangeGhostCells(World& world, val_t* host_left, val_t* host_right, 
                        val_t* host_top, val_t* host_bottom) {
    // Unpack received ghost cells
    #pragma omp parallel for
    for (int x = 0; x < world.local_nx; ++x) {
        world.elements_dynamic[world.ghost_left_offset + x].current_energy = world.recv_left[x];
        world.elements_dynamic[world.ghost_right_offset + x].current_energy = world.recv_right[x];
    }
    
    #pragma omp parallel for
    for (int y = 0; y < world.local_ny; ++y) {
        world.elements_dynamic[world.ghost_top_offset + y].current_energy = world.recv_top[y];
        world.elements_dynamic[world.ghost_bottom_offset + y].current_energy = world.recv_bottom[y];
    }
    
    // MPI exchange
    MPI_Request requests[8];
    int n_req = 0;
    
    if (world.rank_left != MPI_PROC_NULL) {
        MPI_Isend(host_left, world.local_nx, MPI_DOUBLE, world.rank_left, 0, MPI_COMM_WORLD, &requests[n_req++]);
        MPI_Irecv(world.recv_left.data(), world.local_nx, MPI_DOUBLE, world.rank_left, 0, MPI_COMM_WORLD, &requests[n_req++]);
    }
    if (world.rank_right != MPI_PROC_NULL) {
        MPI_Isend(host_right, world.local_nx, MPI_DOUBLE, world.rank_right, 0, MPI_COMM_WORLD, &requests[n_req++]);
        MPI_Irecv(world.recv_right.data(), world.local_nx, MPI_DOUBLE, world.rank_right, 0, MPI_COMM_WORLD, &requests[n_req++]);
    }
    if (world.rank_top != MPI_PROC_NULL) {
        MPI_Isend(host_top, world.local_ny, MPI_DOUBLE, world.rank_top, 0, MPI_COMM_WORLD, &requests[n_req++]);
        MPI_Irecv(world.recv_top.data(), world.local_ny, MPI_DOUBLE, world.rank_top, 0, MPI_COMM_WORLD, &requests[n_req++]);
    }
    if (world.rank_bottom != MPI_PROC_NULL) {
        MPI_Isend(host_bottom, world.local_ny, MPI_DOUBLE, world.rank_bottom, 0, MPI_COMM_WORLD, &requests[n_req++]);
        MPI_Irecv(world.recv_bottom.data(), world.local_ny, MPI_DOUBLE, world.rank_bottom, 0, MPI_COMM_WORLD, &requests[n_req++]);
    }
    
    MPI_Waitall(n_req, requests, MPI_STATUSES_IGNORE);
}

// Initialize CUDA
void initCUDA(World& world) {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    cudaSetDevice(world.mpi_rank % num_gpus);
    
    cudaMalloc(&world.d_elements_static, world.n_interior * sizeof(ElementStatic));
    cudaMalloc(&world.d_elements_dynamic, world.n_total * sizeof(ElementDynamic));
    cudaMalloc(&world.d_elements_dynamic_swap, world.n_total * sizeof(ElementDynamic));
    cudaMalloc(&world.d_materials, world.materials.size() * sizeof(Material));
    
    int max_dim = std::max(world.local_nx, world.local_ny);
    cudaMalloc(&world.d_boundary_left, max_dim * sizeof(val_t));
    cudaMalloc(&world.d_boundary_right, max_dim * sizeof(val_t));
    cudaMalloc(&world.d_boundary_top, max_dim * sizeof(val_t));
    cudaMalloc(&world.d_boundary_bottom, max_dim * sizeof(val_t));
    
    cudaMemcpy(world.d_elements_static, world.elements_static.data(), 
               world.n_interior * sizeof(ElementStatic), cudaMemcpyHostToDevice);
    cudaMemcpy(world.d_elements_dynamic, world.elements_dynamic.data(), 
               world.n_total * sizeof(ElementDynamic), cudaMemcpyHostToDevice);
    cudaMemcpy(world.d_materials, world.materials.data(), 
               world.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
}

// Run simulation
void runSimulation(World& world, const int n_iters) {
    initCUDA(world);
    
    int blockSize = 256;
    int gridSize_sim = (world.n_interior + blockSize - 1) / blockSize;
    int gridSize_boundary = (std::max(world.local_nx, world.local_ny) + blockSize - 1) / blockSize;
    
    ElementDynamic* d_current = world.d_elements_dynamic;
    ElementDynamic* d_swap = world.d_elements_dynamic_swap;
    
    // Host buffers for boundary data
    std::vector<val_t> host_left(world.local_nx);
    std::vector<val_t> host_right(world.local_nx);
    std::vector<val_t> host_top(world.local_ny);
    std::vector<val_t> host_bottom(world.local_ny);
    
    for (int iter = 0; iter < n_iters; ++iter) {
        // Extract boundary elements on GPU
        extract_boundary_kernel<<<gridSize_boundary, blockSize>>>(
            d_current,
            world.d_boundary_left,
            world.d_boundary_right,
            world.d_boundary_top,
            world.d_boundary_bottom,
            world.local_nx, world.local_ny);
        
        // Copy boundary data from GPU to CPU
        cudaMemcpy(host_left.data(), world.d_boundary_left, 
                   world.local_nx * sizeof(val_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_right.data(), world.d_boundary_right, 
                   world.local_nx * sizeof(val_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_top.data(), world.d_boundary_top, 
                   world.local_ny * sizeof(val_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_bottom.data(), world.d_boundary_bottom, 
                   world.local_ny * sizeof(val_t), cudaMemcpyDeviceToHost);
        
        // MPI ghost cell exchange
        exchangeGhostCells(world, host_left.data(), host_right.data(), 
                          host_top.data(), host_bottom.data());
        
        // Upload ghost cells to GPU
        cudaMemcpy(world.d_elements_dynamic + world.n_interior,
                   world.elements_dynamic.data() + world.n_interior,
                   world.n_ghosts * sizeof(ElementDynamic),
                   cudaMemcpyHostToDevice);
        
        // Launch simulation kernel
        simulation_kernel<<<gridSize_sim, blockSize>>>(
            world.d_elements_static,
            d_current,
            d_swap,
            world.d_materials,
            world.n_interior);
        
        // Swap pointers
        ElementDynamic* tmp = d_current;
        d_current = d_swap;
        d_swap = tmp;
    }
    
    // Final download of results
    cudaMemcpy(world.elements_dynamic.data(),
               d_current,
               world.n_interior * sizeof(ElementDynamic),
               cudaMemcpyDeviceToHost);
    
    world.d_elements_dynamic = d_current;
    world.d_elements_dynamic_swap = d_swap;
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_energy_max = std::numeric_limits<val_t>::lowest();
    val_t local_energy_min = std::numeric_limits<val_t>::max();
    
    #pragma omp parallel for reduction(+:local_energy_sum,local_flux_sum) \
        reduction(max:local_energy_max) reduction(min:local_energy_min)
    for (int i = 0; i < world.n_interior; ++i) {
        local_energy_sum += world.elements_dynamic[i].current_energy;
        local_flux_sum += world.elements_dynamic[i].total_flux;
        local_energy_max = std::max(local_energy_max, world.elements_dynamic[i].current_energy);
        local_energy_min = std::min(local_energy_min, world.elements_dynamic[i].current_energy);
    }
    
    val_t energy_sum, flux_sum, energy_max, energy_min;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_energy_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (world.mpi_rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        
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

// Compute hash on rank 0 after gathering all elements
uint64_t computeHashGlobal(const std::vector<ElementDynamic>& elements) {
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
    
    World world;
    MPI_Comm_rank(MPI_COMM_WORLD, &world.mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.mpi_size);
    
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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world.mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int n_elems = n_elems_root * n_elems_root;
    
    if (world.mpi_rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world.mpi_size);
        printf("\n");
    }
    
    // Build the unstructured mesh
    if (world.mpi_rank == 0) printf("Building unstructured mesh...\n");
    buildSquare2D(world, n_elems_root);
    
    // Calculate memory usage
    const size_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
    const size_t dynamic_mem = world.elements_dynamic.size() * sizeof(ElementDynamic) * 2;
    const size_t total_mem = static_mem + dynamic_mem;
    
    size_t global_total_mem;
    MPI_Reduce(&total_mem, &global_total_mem, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (world.mpi_rank == 0) {
        printf("Memory usage: %.2f MB\n", global_total_mem / (1024.0 * 1024.0));
        printf("\n");
    }
    
    // Run simulation
    if (world.mpi_rank == 0) printf("Running simulation...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(world, n_iters);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (world.mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);
        
        // Calculate performance metrics
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = static_cast<double>(duration_ms) / n_measured_iters;
        const double giga_elems_per_sec = (n_measured_iters * n_elems) / (duration_ms / 1000.0) / 1e9;
        const double gflops = giga_elems_per_sec * 22.0;
        
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
    }
    
    // Gather results to rank 0 for hash computation
    std::vector<ElementDynamic> global_elements;
    if (world.mpi_rank == 0) {
        global_elements.resize(n_elems);
    }
    
    // Compute base dimensions for all ranks
    int base_nx = n_elems_root / world.proc_grid_x;
    int base_ny = n_elems_root / world.proc_grid_y;
    int rem_x = n_elems_root % world.proc_grid_x;
    int rem_y = n_elems_root % world.proc_grid_y;
    
    // Send local elements to rank 0
    if (world.mpi_rank != 0) {
        MPI_Send(world.elements_dynamic.data(), world.n_interior * 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    if (world.mpi_rank == 0) {
        // Copy rank 0's elements
        for (int x = 0; x < world.local_nx; ++x)
            for (int y = 0; y < world.local_ny; ++y) {
                int gx = world.global_x_offset + x;
                int gy = world.global_y_offset + y;
                global_elements[gx * n_elems_root + gy] = world.elements_dynamic[x * world.local_ny + y];
            }
        
        // Receive from other ranks
        for (int rank = 1; rank < world.mpi_size; ++rank) {
            int sender_proc_x = rank / world.proc_grid_y;
            int sender_proc_y = rank % world.proc_grid_y;
            int sender_local_nx = base_nx + (sender_proc_x < rem_x ? 1 : 0);
            int sender_local_ny = base_ny + (sender_proc_y < rem_y ? 1 : 0);
            int sender_global_x_offset = sender_proc_x * base_nx + std::min(sender_proc_x, rem_x);
            int sender_global_y_offset = sender_proc_y * base_ny + std::min(sender_proc_y, rem_y);
            
            std::vector<ElementDynamic> sender_elements(sender_local_nx * sender_local_ny);
            MPI_Recv(sender_elements.data(), sender_local_nx * sender_local_ny * 2, 
                    MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int x = 0; x < sender_local_nx; ++x)
                for (int y = 0; y < sender_local_ny; ++y) {
                    int gx = sender_global_x_offset + x;
                    int gy = sender_global_y_offset + y;
                    global_elements[gx * n_elems_root + gy] = sender_elements[x * sender_local_ny + y];
                }
        }
        
        const uint64_t hash = computeHashGlobal(global_elements);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");
    }
    
    // Print results for external validation
    if (printResults) {
        if (world.mpi_rank == 0) {
            std::vector<double> energyData;
            energyData.reserve(n_elems);
            for (int i = 0; i < n_elems; ++i)
                energyData.push_back(global_elements[i].current_energy);
            print_results(energyData, "ElementEnergy");
        }
    }
    
    // Validation
    if (validate) {
        bool valid = validateResults(world);
        if (!valid) {
            MPI_Finalize();
            return 1;
        }
    }
    
    // Cleanup CUDA
    cudaFree(world.d_elements_static);
    cudaFree(world.d_elements_dynamic);
    cudaFree(world.d_elements_dynamic_swap);
    cudaFree(world.d_materials);
    cudaFree(world.d_boundary_left);
    cudaFree(world.d_boundary_right);
    cudaFree(world.d_boundary_top);
    cudaFree(world.d_boundary_bottom);
    
    MPI_Finalize();
    return 0;
}
