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
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic { idx_t material_idx; idx_t num_connections; idx_t connected_idx[MAX_CONNECTIONS]; val_t connected_flux[MAX_CONNECTIONS]; };
struct ElementDynamic { val_t current_energy; val_t total_flux; };

// device kernels
__global__ void runIterKernel(const ElementStatic* es, const Material* mats, const ElementDynamic* in, ElementDynamic* out, size_t n) {
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ElementStatic& elem_static = es[i];
    const ElementDynamic& elem_dyn = in[i];
    const Material& mat = mats[elem_static.material_idx];
    val_t total_flux = mat.external_flow;
    for (idx_t j = 0; j < elem_static.num_connections; ++j) {
        idx_t neighbor_idx = elem_static.connected_idx[j];
        const ElementDynamic& neighbor_dyn = in[neighbor_idx];
        val_t f = (neighbor_dyn.current_energy - elem_dyn.current_energy) * mat.transfer_coeff * elem_static.connected_flux[j] * 0.25;
        total_flux += f;
    }
    out[i].current_energy = elem_dyn.current_energy + total_flux;
    out[i].total_flux = elem_dyn.total_flux + fabs(total_flux);
}

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

void buildSquare2D(World& world, const int n_elems_root) {
    const int n_elems = n_elems_root * n_elems_root;
    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});
    world.elements_static.resize(n_elems);
    world.elements_dynamic.resize(n_elems);
    world.elements_dynamic_swap.resize(n_elems);
    #pragma omp parallel for
    for (int i = 0; i < n_elems; ++i) {
        world.elements_static[i].material_idx = DEFAULT_MAT_ID;
        world.elements_static[i].num_connections = 0;
        world.elements_dynamic[i].current_energy = 0.0;
        world.elements_dynamic[i].total_flux = 0.0;
    }
    for (int x = 0; x < n_elems_root; ++x) {
        for (int y = 0; y < n_elems_root; ++y) {
            const int idx = x * n_elems_root + y;
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
    const int last = n_elems_root - 1;
    world.elements_static[0 * n_elems_root + 0].material_idx = INFLOW_MAT_ID;
    world.elements_static[0 * n_elems_root + last].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + 0].material_idx = OUTFLOW_MAT_ID;
    world.elements_static[last * n_elems_root + last].material_idx = INFLOW_MAT_ID;
}

bool validateResults(const World& world) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    #pragma omp parallel for reduction(+:energy_sum,flux_sum) reduction(max:energy_max) reduction(min:energy_min)
    for (size_t i = 0; i < world.elements_dynamic.size(); ++i) {
        energy_sum += world.elements_dynamic[i].current_energy;
        flux_sum += world.elements_dynamic[i].total_flux;
        energy_max = std::max(world.elements_dynamic[i].current_energy, energy_max);
        energy_min = std::min(world.elements_dynamic[i].current_energy, energy_min);
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
    #pragma omp parallel for reduction(^:hash)
    for (size_t i = 0; i < elements.size(); ++i) {
        const uint64_t* e_ptr = reinterpret_cast<const uint64_t*>(&elements[i].current_energy);
        const uint64_t* f_ptr = reinterpret_cast<const uint64_t*>(&elements[i].total_flux);
        uint64_t h = (*e_ptr + i) * 0x9e3779b97f4a7c15ULL;
        h ^= (*f_ptr + i) * 0xbf58476d1ce4e5b9ULL;
        // reduction on xor is order dependent; do atomic-ish via critical? Simpler serial fallback for hash correctness
    }
    // Recompute serially to preserve exact hash
    hash = 0;
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
    printf("Options:\n  -n <num>  Grid size\n  -i <num>  Iterations\n  -v        Validate\n  -r        Print results\n  -h        Help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_elems_root = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
    }
    if (rank != 0) { MPI_Finalize(); return 0; }

    const int n_elems = n_elems_root * n_elems_root;
    printf("Unstructured Mesh Energy Transfer Benchmark\n");
    printf("Grid size: %d x %d = %d elements\n", n_elems_root, n_elems_root, n_elems);
    printf("Iterations: %d\n", n_iters);

    World world;
    buildSquare2D(world, n_elems_root);

    // CUDA device buffers
    ElementStatic* d_es; Material* d_mats; ElementDynamic *d_in, *d_out;
    cudaMalloc(&d_es, world.elements_static.size()*sizeof(ElementStatic));
    cudaMalloc(&d_mats, world.materials.size()*sizeof(Material));
    cudaMalloc(&d_in, world.elements_dynamic.size()*sizeof(ElementDynamic));
    cudaMalloc(&d_out, world.elements_dynamic.size()*sizeof(ElementDynamic));
    cudaMemcpy(d_es, world.elements_static.data(), world.elements_static.size()*sizeof(ElementStatic), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mats, world.materials.data(), world.materials.size()*sizeof(Material), cudaMemcpyHostToDevice);

    printf("Running simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_in, world.elements_dynamic.data(), world.elements_dynamic.size()*sizeof(ElementDynamic), cudaMemcpyHostToDevice);
    int threads=256;
    int blocks=(n_elems+threads-1)/threads;
    ElementDynamic* cur=d_in; ElementDynamic* nxt=d_out;
    for (int iter=0; iter<n_iters; ++iter) {
        runIterKernel<<<blocks,threads>>>(d_es,d_mats,cur,nxt,n_elems);
        cudaDeviceSynchronize();
        std::swap(cur,nxt);
    }
    // ensure final in cur? after n_iters swaps, cur points to last output if n_iters odd? Let's copy back cur.
    cudaMemcpy(world.elements_dynamic.data(), cur, world.elements_dynamic.size()*sizeof(ElementDynamic), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    printf("Computation time: %ld ms\n", duration_ms);

    const uint64_t hash = computeHash(world.elements_dynamic);
    printf("  Result hash: %016lX\n", hash);

    if (printResults) {
        std::vector<double> energyData;
        energyData.reserve(world.elements_dynamic.size());
        for (const auto& elem : world.elements_dynamic) energyData.push_back(elem.current_energy);
        print_results(energyData, "ElementEnergy");
    }
    if (validate) {
        if (!validateResults(world)) { MPI_Finalize(); return 1; }
    }
    cudaFree(d_es); cudaFree(d_mats); cudaFree(d_in); cudaFree(d_out);
    MPI_Finalize();
    return 0;
}
