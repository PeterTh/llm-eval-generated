#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using val_t = double;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};
struct ElementStatic {
    uint8_t material_idx;
    uint8_t neighbors;
};
struct ElementDynamic {
    val_t current_energy;
    val_t total_flux;
};

// Block distribution, including the remainder in the first blocks.
static int blockSize(int n, int blocks, int coord) {
    return n / blocks + (coord < n % blocks);
}
static int blockStart(int n, int blocks, int coord) {
    return coord * (n / blocks) + std::min(coord, n % blocks);
}

struct World {
    int nx, ny, x0, y0, pitch;
    int xminus, xplus, yminus, yplus;
    MPI_Comm comm;
    MPI_Datatype row_type, column_type;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic, elements_dynamic_swap;
};

static void buildSquare2D(World& world, int n, const int dims[2], int rank, MPI_Comm comm) {
    const int cx = rank / dims[1], cy = rank % dims[1];
    world.nx = blockSize(n, dims[0], cx);
    world.ny = blockSize(n, dims[1], cy);
    world.x0 = blockStart(n, dims[0], cx);
    world.y0 = blockStart(n, dims[1], cy);
    world.pitch = world.ny + 2;
    world.comm = comm;
    MPI_Cart_shift(comm, 0, 1, &world.xminus, &world.xplus);
    MPI_Cart_shift(comm, 1, 1, &world.yminus, &world.yplus);
    const size_t size = static_cast<size_t>(world.nx + 2) * world.pitch;
    world.elements_static.resize(size);
    world.elements_dynamic.resize(size);
    world.elements_dynamic_swap.resize(size);
    for (int x = 1; x <= world.nx; ++x) {
        for (int y = 1; y <= world.ny; ++y) {
            const int gx = world.x0 + x - 1, gy = world.y0 + y - 1;
            const size_t i = static_cast<size_t>(x) * world.pitch + y;
            ElementStatic& elem = world.elements_static[i];
            elem.material_idx = 0;
            // Match the original sequence of corner assignments, including n=1.
            if (gx == 0 && gy == 0) elem.material_idx = 1;
            if (gx == 0 && gy == n - 1) elem.material_idx = 2;
            if (gx == n - 1 && gy == 0) elem.material_idx = 2;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = 1;
            // The bits preserve the original +x, -x, +y, -y order.
            elem.neighbors = (gx + 1 < n ? 1 : 0) | (gx > 0 ? 2 : 0) |
                             (gy + 1 < n ? 4 : 0) | (gy > 0 ? 8 : 0);
        }
    }
    // Only current_energy is communicated. Both types describe energies in the
    // padded array of ElementDynamic structs, without sending accumulated flux.
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double), "unexpected element layout");
    MPI_Type_vector(world.ny, 1, 2, MPI_DOUBLE, &world.row_type);
    MPI_Type_commit(&world.row_type);
    MPI_Type_vector(world.nx, 1, 2 * world.pitch, MPI_DOUBLE, &world.column_type);
    MPI_Type_commit(&world.column_type);
}

static inline void updateElement(World& world, size_t i) {
    static constexpr Material materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const ElementStatic& elem = world.elements_static[i];
    const ElementDynamic& state = world.elements_dynamic[i];
    const Material& mat = materials[elem.material_idx];
    val_t flux = mat.external_flow;
    const auto* states = world.elements_dynamic.data();
    const auto add = [&](size_t neighbor) {
        flux += (states[neighbor].current_energy - state.current_energy) *
                mat.transfer_coeff * 1.0 * 0.25;
    };
    if (elem.neighbors & 1) add(i + world.pitch);
    if (elem.neighbors & 2) add(i - world.pitch);
    if (elem.neighbors & 4) add(i + 1);
    if (elem.neighbors & 8) add(i - 1);
    world.elements_dynamic_swap[i].current_energy = state.current_energy + flux;
    world.elements_dynamic_swap[i].total_flux = state.total_flux + std::abs(flux);
}

static void runSimulation(World& w, int n_iters) {
    const int p = w.pitch;
    for (int iter = 0; iter < n_iters; ++iter) {
        auto* a = w.elements_dynamic.data();
        MPI_Request requests[8];
        MPI_Irecv(&a[1].current_energy, 1, w.row_type, w.xminus, 0, w.comm, &requests[0]);
        MPI_Irecv(&a[static_cast<size_t>(w.nx + 1) * p + 1].current_energy, 1, w.row_type, w.xplus, 1, w.comm, &requests[1]);
        MPI_Irecv(&a[p].current_energy, 1, w.column_type, w.yminus, 2, w.comm, &requests[2]);
        MPI_Irecv(&a[p + w.ny + 1].current_energy, 1, w.column_type, w.yplus, 3, w.comm, &requests[3]);
        MPI_Isend(&a[static_cast<size_t>(w.nx) * p + 1].current_energy, 1, w.row_type, w.xplus, 0, w.comm, &requests[4]);
        MPI_Isend(&a[p + 1].current_energy, 1, w.row_type, w.xminus, 1, w.comm, &requests[5]);
        MPI_Isend(&a[p + w.ny].current_energy, 1, w.column_type, w.yplus, 2, w.comm, &requests[6]);
        MPI_Isend(&a[p + 1].current_energy, 1, w.column_type, w.yminus, 3, w.comm, &requests[7]);
        for (int x = 2; x < w.nx; ++x)
            for (int y = 2; y < w.ny; ++y)
                updateElement(w, static_cast<size_t>(x) * p + y);
        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        for (int y = 1; y <= w.ny; ++y) {
            updateElement(w, static_cast<size_t>(p) + y);
            if (w.nx > 1) updateElement(w, static_cast<size_t>(w.nx) * p + y);
        }
        for (int x = 2; x < w.nx; ++x) {
            updateElement(w, static_cast<size_t>(x) * p + 1);
            if (w.ny > 1) updateElement(w, static_cast<size_t>(x) * p + w.ny);
        }
        w.elements_dynamic.swap(w.elements_dynamic_swap);
    }
}

static uint64_t computeHash(const World& w, int n) {
    uint64_t hash = 0;
    for (int x = 1; x <= w.nx; ++x) {
        for (int y = 1; y <= w.ny; ++y) {
            const auto& elem = w.elements_dynamic[static_cast<size_t>(x) * w.pitch + y];
            const uint64_t i = static_cast<uint64_t>(w.x0 + x - 1) * n + w.y0 + y - 1;
            uint64_t e, f;
            std::memcpy(&e, &elem.current_energy, sizeof(e));
            std::memcpy(&f, &elem.total_flux, sizeof(f));
            hash ^= (e + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (f + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Gather only for the optional reports, whose sums and hashes are order dependent.
static std::vector<ElementDynamic> gatherResults(const World& w, bool active, int n,
                                                  const int dims[2], int active_count,
                                                  int rank, int ranks) {
    std::vector<ElementDynamic> local;
    if (active) {
        local.reserve(static_cast<size_t>(w.nx) * w.ny);
        for (int x = 1; x <= w.nx; ++x)
            for (int y = 1; y <= w.ny; ++y)
                local.push_back(w.elements_dynamic[static_cast<size_t>(x) * w.pitch + y]);
    }
    std::vector<int> counts, offsets;
    std::vector<ElementDynamic> packed, result;
    if (rank == 0) {
        counts.resize(ranks, 0);
        offsets.resize(ranks, 0);
        int offset = 0;
        for (int r = 0; r < active_count; ++r) {
            counts[r] = blockSize(n, dims[0], r / dims[1]) * blockSize(n, dims[1], r % dims[1]);
            offsets[r] = offset;
            offset += counts[r];
        }
        packed.resize(offset);
        result.resize(offset);
    }
    MPI_Datatype element_type;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element_type);
    MPI_Type_commit(&element_type);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), element_type,
                packed.data(), counts.data(), offsets.data(), element_type, 0, MPI_COMM_WORLD);
    MPI_Type_free(&element_type);
    if (rank == 0) {
        for (int r = 0; r < active_count; ++r) {
            const int cx = r / dims[1], cy = r % dims[1];
            const int nx = blockSize(n, dims[0], cx), ny = blockSize(n, dims[1], cy);
            const int x0 = blockStart(n, dims[0], cx), y0 = blockStart(n, dims[1], cy);
            for (int x = 0; x < nx; ++x)
                std::copy_n(packed.begin() + offsets[r] + x * ny, ny,
                            result.begin() + static_cast<size_t>(x0 + x) * n + y0);
        }
    }
    return result;
}

static bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (const auto& elem : elements) {
        energy_sum += elem.current_energy;
        flux_sum += elem.total_flux;
        energy_max = std::max(elem.current_energy, energy_max);
        energy_min = std::min(elem.current_energy, energy_min);
    }
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n",
           energy_sum, flux_sum, energy_min, energy_max);
    if (!std::isfinite(energy_sum)) { printf("  ERROR: Energy sum is not finite\n"); return false; }
    if (std::abs(energy_sum) > 1e-8) printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(flux_sum)) { printf("  ERROR: Flux sum is not finite\n"); return false; }
    if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
        printf("  ERROR: Energy extrema are not finite\n"); return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512, n_iters = 10, status = 0;
    bool validate = false, printResults = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); status = 1; break; }
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return status;
    }
    if (n <= 0 || static_cast<int64_t>(n) * n > std::numeric_limits<int>::max()) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive and contain at most INT_MAX elements\n");
        MPI_Finalize();
        return 1;
    }
    const int n_elems = n * n;
    // Prefer balanced tiles, allowing a few ranks to be idle when a prime
    // process count would otherwise create a much more communication-heavy grid.
    const int max_active = std::min(ranks, n_elems);
    const int min_active = std::max(1, max_active - max_active / 10);
    int active_count = 1, dims[2] = {1, 1};
    double best_cost = std::numeric_limits<double>::infinity();
    for (int count = max_active; count >= min_active; --count) {
        int d = static_cast<int>(std::sqrt(static_cast<double>(count)));
        while (d > 0 && (count % d != 0 || count / d > n)) --d;
        if (d == 0 || d > n) continue;
        const int other = count / d;
        const double cost = static_cast<double>(n_elems) / count +
                            4.0 * n * (1.0 / d + 1.0 / other);
        if (cost < best_cost) {
            best_cost = cost;
            active_count = count;
            dims[0] = d;
            dims[1] = other;
        }
    }
    // A valid factorization always exists by count <= n*n, but it may lie
    // outside the preferred range when the grid has very few elements.
    if (!std::isfinite(best_cost)) {
        for (int count = min_active - 1; count >= 1; --count) {
            int d = static_cast<int>(std::sqrt(static_cast<double>(count)));
            while (d > 0 && (count % d != 0 || count / d > n)) --d;
            if (d > 0 && d <= n) {
                active_count = count;
                dims[0] = d;
                dims[1] = count / d;
                break;
            }
        }
    }
    const bool active = rank < active_count;
    MPI_Comm split, cart = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &split);
    World world{};
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
        printf("Grid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n", n, n, n_elems, n_iters, validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    if (active) {
        int periods[2] = {0, 0};
        MPI_Cart_create(split, 2, dims, periods, 0, &cart);
        buildSquare2D(world, n, dims, rank, cart);
    }
    unsigned long long local_static = active ? world.elements_static.size() * sizeof(ElementStatic) : 0;
    unsigned long long local_dynamic = active ? world.elements_dynamic.size() * sizeof(ElementDynamic) * 2 : 0;
    unsigned long long local_mem[2] = {local_static, local_dynamic}, global_mem[2] = {0, 0};
    MPI_Reduce(local_mem, global_mem, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (global_mem[0] + global_mem[1]) / 1048576.0, global_mem[0] / 1048576.0, global_mem[1] / 1048576.0);
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (active) runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    uint64_t local_hash = active ? computeHash(world, n) : 0, hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        const int measured = std::max(n_iters - 1, 1);
        const double per_iter = max_elapsed * 1000.0 / measured;
        const double rate = (static_cast<double>(measured) * n_elems) / max_elapsed / 1e9;
        printf("Computation time: %ld ms\nPerformance:\n", duration_ms);
        printf("  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", per_iter, rate, rate * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }
    if (validate || printResults) {
        auto result = gatherResults(world, active, n, dims, active_count, rank, ranks);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> energies;
                energies.reserve(result.size());
                for (const auto& elem : result) energies.push_back(elem.current_energy);
                print_results(energies, "ElementEnergy");
            }
            if (validate && !validateResults(result)) status = 1;
        }
    }
    if (active) {
        MPI_Type_free(&world.row_type);
        MPI_Type_free(&world.column_type);
        MPI_Comm_free(&cart);
        MPI_Comm_free(&split);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
