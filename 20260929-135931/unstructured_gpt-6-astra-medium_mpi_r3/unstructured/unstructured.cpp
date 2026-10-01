#include <algorithm>
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
struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic {
    idx_t material_idx, num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy, total_flux; };
struct World {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank, n, nx, ny, x0, y0, dims[2], coords[2], neighbors[4];
    size_t ghost[4];
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
    std::vector<size_t> interior, boundary;
    std::vector<double> send[4], receive[4];
};

// Construct only the owned adjacency lists, with remote neighbors mapped to ghosts.
void buildSquare2D(World& w, int n) {
    w.n = n;
    int periods[2];
    MPI_Comm_rank(w.comm, &w.rank);
    MPI_Cart_get(w.comm, 2, w.dims, periods, w.coords);
    w.x0 = int(int64_t(n) * w.coords[0] / w.dims[0]);
    w.y0 = int(int64_t(n) * w.coords[1] / w.dims[1]);
    w.nx = int(int64_t(n) * (w.coords[0] + 1) / w.dims[0]) - w.x0;
    w.ny = int(int64_t(n) * (w.coords[1] + 1) / w.dims[1]) - w.y0;
    MPI_Cart_shift(w.comm, 0, 1, &w.neighbors[1], &w.neighbors[0]);
    MPI_Cart_shift(w.comm, 1, 1, &w.neighbors[3], &w.neighbors[2]);
    const size_t owned = size_t(w.nx) * w.ny;
    size_t storage = owned;
    for (int d = 0; d < 4; ++d) {
        w.ghost[d] = storage;
        if (w.neighbors[d] == MPI_PROC_NULL) continue;
        int count = d < 2 ? w.ny : w.nx;
        w.send[d].resize(count);
        w.receive[d].resize(count);
        storage += count;
    }
    w.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    w.elements_static.resize(owned);
    w.elements_dynamic.resize(storage);
    w.elements_dynamic_swap.resize(storage);
    const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int x = 0; x < w.nx; ++x) {
        for (int y = 0; y < w.ny; ++y) {
            size_t i = size_t(x) * w.ny + y;
            auto& e = w.elements_static[i];
            int gx = w.x0 + x, gy = w.y0 + y;
            // Preserve assignment order for overlapping corners on a 1x1 grid.
            if (gx == 0 && gy == 0) e.material_idx = 1;
            if (gx == 0 && gy == n - 1) e.material_idx = 2;
            if (gx == n - 1 && gy == 0) e.material_idx = 2;
            if (gx == n - 1 && gy == n - 1) e.material_idx = 1;
            bool remote = false;
            for (int d = 0; d < 4; ++d) {
                int xx = x + offsets[d][0], yy = y + offsets[d][1];
                if (gx + offsets[d][0] < 0 || gx + offsets[d][0] >= n ||
                    gy + offsets[d][1] < 0 || gy + offsets[d][1] >= n) continue;
                size_t neighbor;
                if (xx >= 0 && xx < w.nx && yy >= 0 && yy < w.ny)
                    neighbor = size_t(xx) * w.ny + yy;
                else {
                    neighbor = w.ghost[d] + (d < 2 ? y : x);
                    remote = true;
                }
                size_t j = e.num_connections++;
                e.connected_idx[j] = neighbor;
                e.connected_flux[j] = 1.0;
            }
            (remote ? w.boundary : w.interior).push_back(i);
        }
    }
}

inline val_t computeFlux(const Material& mat, const ElementDynamic& self,
                         val_t connection_flux, const ElementDynamic& other) {
    return (other.current_energy - self.current_energy) *
           mat.transfer_coeff * connection_flux * 0.25;
}

void runSimulation(World& w, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        // Transfer only energy: accumulated flux is never read by neighbors.
        for (int d = 0; d < 4; ++d) {
            if (w.neighbors[d] == MPI_PROC_NULL) continue;
            auto& buffer = w.send[d];
            for (size_t j = 0; j < buffer.size(); ++j) {
                size_t i = d == 0 ? size_t(w.nx - 1) * w.ny + j :
                           d == 1 ? j : d == 2 ? j * w.ny + w.ny - 1 : j * w.ny;
                buffer[j] = w.elements_dynamic[i].current_energy;
            }
            MPI_Irecv(w.receive[d].data(), int(buffer.size()), MPI_DOUBLE,
                      w.neighbors[d], d ^ 1, w.comm, &requests[count++]);
            MPI_Isend(buffer.data(), int(buffer.size()), MPI_DOUBLE,
                      w.neighbors[d], d, w.comm, &requests[count++]);
        }
        const auto update = [&](const std::vector<size_t>& indices) {
            for (size_t i : indices) {
                const auto& e = w.elements_static[i];
                const auto& state = w.elements_dynamic[i];
                const auto& mat = w.materials[e.material_idx];
                val_t flux = mat.external_flow;
                // Keep the original neighbor and arithmetic order for exact results.
                for (idx_t j = 0; j < e.num_connections; ++j)
                    flux += computeFlux(mat, state, e.connected_flux[j],
                                        w.elements_dynamic[e.connected_idx[j]]);
                auto& out = w.elements_dynamic_swap[i];
                out.current_energy = state.current_energy + flux;
                out.total_flux = state.total_flux + std::abs(flux);
            }
        };
        update(w.interior);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int d = 0; d < 4; ++d)
            for (size_t j = 0; j < w.receive[d].size(); ++j)
                w.elements_dynamic[w.ghost[d] + j].current_energy = w.receive[d][j];
        update(w.boundary);
        w.elements_dynamic.swap(w.elements_dynamic_swap);
    }
}

// Validate simulation results
bool validateResults(const World& world) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        const auto& elem = world.elements_dynamic[i];
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    
    double sums[2] = {energy_sum, flux_sum};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    energy_sum = sums[0];
    flux_sum = sums[1];
    MPI_Allreduce(MPI_IN_PLACE, &energy_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(MPI_IN_PLACE, &energy_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    if (world.rank != 0)
        return std::isfinite(energy_sum) && std::isfinite(flux_sum) &&
               std::isfinite(energy_min) && std::isfinite(energy_max);
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

// XOR is independent of rank order, but the hash index must remain global.
uint64_t computeHash(const World& w) {
    uint64_t hash = 0, result = 0;
    for (size_t i = 0; i < w.elements_static.size(); ++i) {
        uint64_t global = (w.x0 + i / w.ny) * uint64_t(w.n) + w.y0 + i % w.ny;
        uint64_t energy, flux;
        std::memcpy(&energy, &w.elements_dynamic[i].current_energy, sizeof(energy));
        std::memcpy(&flux, &w.elements_dynamic[i].total_flux, sizeof(flux));
        hash ^= (energy + global) * 0x9e3779b97f4a7c15ULL;
        hash ^= (flux + global) * 0xbf58476d1ce4e5b9ULL;
    }
    MPI_Reduce(&hash, &result, 1, MPI_UINT64_T, MPI_BXOR, 0, w.comm);
    return result;
}

// Gathering is needed only for -r: the existing output uses an ordered hash
// and Kahan sum. Receive directly into global rows without a second full copy.
void printDistributedResults(const World& w) {
    std::vector<double> local(w.elements_static.size());
    for (size_t i = 0; i < local.size(); ++i)
        local[i] = w.elements_dynamic[i].current_energy;
    if (w.rank != 0) {
        for (int x = 0; x < w.nx; ++x)
            MPI_Send(local.data() + size_t(x) * w.ny, w.ny, MPI_DOUBLE, 0, 10, w.comm);
        return;
    }
    std::vector<double> energy(size_t(w.n) * w.n);
    int ranks;
    MPI_Comm_size(w.comm, &ranks);
    for (int r = 0; r < ranks; ++r) {
        int coords[2];
        MPI_Cart_coords(w.comm, r, 2, coords);
        int x0 = int(int64_t(w.n) * coords[0] / w.dims[0]);
        int x1 = int(int64_t(w.n) * (coords[0] + 1) / w.dims[0]);
        int y0 = int(int64_t(w.n) * coords[1] / w.dims[1]);
        int y1 = int(int64_t(w.n) * (coords[1] + 1) / w.dims[1]);
        for (int x = x0; x < x1; ++x) {
            double* row = energy.data() + size_t(x) * w.n + y0;
            if (r == 0)
                std::copy_n(local.data() + size_t(x - x0) * w.ny, y1 - y0, row);
            else
                MPI_Recv(row, y1 - y0, MPI_DOUBLE, r, 10, w.comm, MPI_STATUS_IGNORE);
        }
    }
    print_results(energy, "ElementEnergy");
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
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
    if (n <= 0 || n_iters < 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and iterations nonnegative.\n");
        MPI_Finalize();
        return 1;
    }
    // Maximize the number of nonempty blocks, then minimize their perimeter.
    int dims[2] = {1, 1};
    for (int x = 1; x <= std::min(n, ranks); ++x) {
        int y = std::min(n, ranks / x);
        if (x * y > dims[0] * dims[1] ||
            (x * y == dims[0] * dims[1] && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < dims[0] * dims[1] ? 0 : MPI_UNDEFINED,
                   rank, &active);
    if (active == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    World world;
    int periods[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active);
    uint64_t n_elems = uint64_t(n) * n;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid: %d x %d)\n", dims[0] * dims[1], dims[0], dims[1]);
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n);
    uint64_t memory[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        world.elements_dynamic.size() * sizeof(ElementDynamic) * 2
    };
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (memory[0] + memory[1]) / (1024.0 * 1024.0),
               memory[0] / (1024.0 * 1024.0), memory[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(world.comm);
    double start = MPI_Wtime();
    runSimulation(world, n_iters);
    double elapsed = MPI_Wtime() - start, maximum_elapsed = 0;
    MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    uint64_t hash = computeHash(world);
    if (rank == 0) {
        double duration_ms = maximum_elapsed * 1000.0;
        double rate = maximum_elapsed > 0 ? double(n_iters) * n_elems / maximum_elapsed / 1e9 : 0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / std::max(n_iters, 1));
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }
    if (printResults) printDistributedResults(world);
    bool valid = !validate || validateResults(world);
    MPI_Comm_free(&world.comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
