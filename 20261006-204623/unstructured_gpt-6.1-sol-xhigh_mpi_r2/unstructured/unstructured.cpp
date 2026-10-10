#include <mpi.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

// The generated mesh has at most four edges per element, all with unit flux.
constexpr int MAX_CONNECTIONS = 4;
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Indices address local energy storage, including remote halo elements.
struct ElementStatic {
    size_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx = DEFAULT_MAT_ID;
    uint8_t num_connections = 0;
};

struct World {
    Material materials[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    MPI_Request halo_requests[2][8]{};
    int request_count = 0;
    int rank = 0;
    int dims[2]{};
    int coords[2]{};
    int neighbors[4]{}; // x-, x+, y-, y+
    int grid_size = 0;
    int x_begin = 0, y_begin = 0;
    int nx = 0, ny = 0;
    size_t stride = 0;
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> current_energy;
    std::vector<val_t> next_energy;
    // Flux depends only on its own previous value, so it needs no halo or swap.
    std::vector<val_t> total_flux;

    ~World() {
        // A rank that fails allocation must reach MPI_Abort immediately, even
        // when its peers are already waiting in a collective or halo exchange.
        if (std::uncaught_exceptions() != 0) return;
        for (auto& requests : halo_requests) {
            for (int i = 0; i < request_count; ++i) {
                MPI_Request_free(&requests[i]);
            }
        }
        if (column_type != MPI_DATATYPE_NULL) MPI_Type_free(&column_type);
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }
};

// Balance both dimensions, using all ranks whenever a nonempty rectangular
// process grid fits. Excess ranks on tiny meshes wait outside the simulation.
void chooseProcessGrid(int ranks, int n, int dims[2]) {
    int best_count = 0;
    dims[0] = dims[1] = 1;
    for (int small = 1; small <= n && small <= ranks / small; ++small) {
        const int large = std::min(n, ranks / small);
        const int count = small * large;
        if (count > best_count ||
            (count == best_count && large - small < dims[0] - dims[1])) {
            best_count = count;
            dims[0] = large;
            dims[1] = small;
        }
    }
}

void partitionDimension(int n, int parts, int coord, int& begin, int& count) {
    const int base = n / parts;
    const int remainder = n % parts;
    count = base + (coord < remainder);
    begin = coord * base + std::min(coord, remainder);
}

void buildSquare2D(World& world, int n) {
    world.grid_size = n;
    partitionDimension(n, world.dims[0], world.coords[0], world.x_begin, world.nx);
    partitionDimension(n, world.dims[1], world.coords[1], world.y_begin, world.ny);
    world.stride = static_cast<size_t>(world.ny) + 2;
    const size_t rows = static_cast<size_t>(world.nx) + 2;
    if (rows > std::numeric_limits<size_t>::max() / world.stride) {
        throw std::length_error("Local mesh size exceeds the address space");
    }
    const size_t storage_size = rows * world.stride;
    world.elements_static.resize(storage_size);
    world.current_energy.resize(storage_size, 0.0);
    world.next_energy.resize(storage_size, 0.0);
    world.total_flux.resize(storage_size, 0.0);

    for (int x = 1; x <= world.nx; ++x) {
        const int gx = world.x_begin + x - 1;
        for (int y = 1; y <= world.ny; ++y) {
            const int gy = world.y_begin + y - 1;
            const size_t i = static_cast<size_t>(x) * world.stride + y;
            ElementStatic& elem = world.elements_static[i];
            // Preserve the original x+, x-, y+, y- accumulation order.
            if (gx + 1 < n) elem.connected_idx[elem.num_connections++] = i + world.stride;
            if (gx > 0) elem.connected_idx[elem.num_connections++] = i - world.stride;
            if (gy + 1 < n) elem.connected_idx[elem.num_connections++] = i + 1;
            if (gy > 0) elem.connected_idx[elem.num_connections++] = i - 1;

            // The last corner assignment wins for the original 1x1 mesh too.
            if ((gx == 0 && gy == 0) || (gx == n - 1 && gy == n - 1)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((gx == 0 && gy == n - 1) || (gx == n - 1 && gy == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            }
        }
    }
}

void initializeHaloExchange(World& world) {
    MPI_Cart_shift(world.comm, 0, 1, &world.neighbors[0], &world.neighbors[1]);
    MPI_Cart_shift(world.comm, 1, 1, &world.neighbors[2], &world.neighbors[3]);
    if (world.dims[1] > 1) {
        MPI_Type_vector(world.nx, 1, static_cast<int>(world.stride), MPI_DOUBLE,
                        &world.column_type);
        MPI_Type_commit(&world.column_type);
    }

    // Persistent requests reference each fixed energy allocation separately.
    // Rows are contiguous; derived datatypes transfer columns without copying
    // total flux, static connectivity, or a full mesh to another rank.
    for (int buffer = 0; buffer < 2; ++buffer) {
        val_t* energy = buffer == 0 ? world.current_energy.data() : world.next_energy.data();
        int count = 0;
        for (int dir = 0; dir < 4; ++dir) {
            if (world.neighbors[dir] == MPI_PROC_NULL) continue;
            size_t send_offset, recv_offset;
            int length;
            MPI_Datatype type;
            if (dir < 2) {
                send_offset = (dir == 0 ? world.stride : world.nx * world.stride) + 1;
                recv_offset = (dir == 0 ? 0 : (world.nx + 1) * world.stride) + 1;
                length = world.ny;
                type = MPI_DOUBLE;
            } else {
                send_offset = world.stride + (dir == 2 ? 1 : world.ny);
                recv_offset = world.stride + (dir == 2 ? 0 : world.ny + 1);
                length = 1;
                type = world.column_type;
            }
            MPI_Recv_init(energy + recv_offset, length, type, world.neighbors[dir],
                          dir ^ 1, world.comm, &world.halo_requests[buffer][count++]);
            MPI_Send_init(energy + send_offset, length, type, world.neighbors[dir],
                          dir, world.comm, &world.halo_requests[buffer][count++]);
        }
        world.request_count = count;
    }
}

template<bool full_connectivity>
inline void updateCells(World& world, int x, int y_begin, int y_end) {
    const size_t row = static_cast<size_t>(x) * world.stride;
    const val_t* energy = world.current_energy.data();
    val_t* next = world.next_energy.data();
    val_t* flux = world.total_flux.data();
    for (int y = y_begin; y < y_end; ++y) {
        const size_t i = row + y;
        const ElementStatic& elem = world.elements_static[i];
        const Material& mat = world.materials[full_connectivity ? DEFAULT_MAT_ID : elem.material_idx];
        const val_t current = energy[i];
        val_t total = mat.external_flow;
        const int connections = full_connectivity ? MAX_CONNECTIONS : elem.num_connections;
        for (int j = 0; j < connections; ++j) {
            // Unit connection flux; retain the original floating-point order.
            total += (energy[elem.connected_idx[j]] - current) * mat.transfer_coeff * 0.25;
        }
        next[i] = current + total;
        flux[i] += std::abs(total);
    }
}

inline void updateRow(World& world, int x, int y_begin, int y_end) {
    if (y_begin >= y_end) return;
    const int gx = world.x_begin + x - 1;
    if (gx == 0 || gx == world.grid_size - 1) {
        updateCells<false>(world, x, y_begin, y_end);
        return;
    }
    if (y_begin == 1 && world.y_begin == 0) {
        updateCells<false>(world, x, y_begin, y_begin + 1);
        ++y_begin;
    }
    if (y_begin < y_end && y_end == world.ny + 1 &&
        world.y_begin + world.ny == world.grid_size) {
        --y_end;
        updateCells<false>(world, x, y_end, y_end + 1);
    }
    // Every nonphysical-boundary cell has four edges and default material.
    // Specializing this loop removes per-cell material and loop-bound loads.
    updateCells<true>(world, x, y_begin, y_end);
}

void runSimulation(World& world, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // All owned and halo energies start at zero, so iteration zero needs
        // no messages. Later iterations exchange the previous state only.
        if (world.request_count == 0 || iter == 0) {
            for (int x = 1; x <= world.nx; ++x) updateRow(world, x, 1, world.ny + 1);
        } else {
            MPI_Request* requests = world.halo_requests[iter & 1];
            MPI_Startall(world.request_count, requests);
            // Interior cells read no remote data and hide communication latency.
            int completed = 0;
            for (int x = 2; x < world.nx; ++x) {
                updateRow(world, x, 2, world.ny);
                // Drive rendezvous/datatype progress on MPI implementations
                // without an asynchronous progress engine, amortized by rows.
                if (!completed && (x & 31) == 0) {
                    MPI_Testall(world.request_count, requests, &completed, MPI_STATUSES_IGNORE);
                }
            }
            MPI_Waitall(world.request_count, requests, MPI_STATUSES_IGNORE);

            updateRow(world, 1, 1, world.ny + 1);
            if (world.nx > 1) updateRow(world, world.nx, 1, world.ny + 1);
            for (int x = 2; x < world.nx; ++x) {
                updateRow(world, x, 1, 2);
                if (world.ny > 1) updateRow(world, x, world.ny, world.ny + 1);
            }
        }
        world.current_energy.swap(world.next_energy);
    }
}

uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (int x = 1; x <= world.nx; ++x) {
        const idx_t global_row = static_cast<idx_t>(world.x_begin + x - 1) * world.grid_size;
        const size_t row = static_cast<size_t>(x) * world.stride;
        for (int y = 1; y <= world.ny; ++y) {
            const idx_t global_index = global_row + world.y_begin + y - 1;
            const size_t i = row + y;
            local_hash ^= (std::bit_cast<uint64_t>(world.current_energy[i]) + global_index) *
                          0x9e3779b97f4a7c15ULL;
            local_hash ^= (std::bit_cast<uint64_t>(world.total_flux[i]) + global_index) *
                          0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
}

// Only the explicit external-validation output needs the global energy array.
// Place each block in serial row-major order to preserve its Kahan sum and FNV
// hash exactly. A datatype count of one also avoids MPI's int count limit.
void printEnergyResults(const World& world) {
    MPI_Datatype local_block;
    MPI_Type_vector(world.nx, world.ny, static_cast<int>(world.stride), MPI_DOUBLE,
                    &local_block);
    MPI_Type_commit(&local_block);
    const val_t* first = world.current_energy.data() + world.stride + 1;
    if (world.rank == 0) {
        const size_t n = static_cast<size_t>(world.grid_size);
        std::vector<val_t> energies(n * n);
        for (int x = 0; x < world.nx; ++x) {
            std::copy_n(first + static_cast<size_t>(x) * world.stride, world.ny,
                        energies.data() + static_cast<size_t>(world.x_begin + x) * n + world.y_begin);
        }
        int ranks;
        MPI_Comm_size(world.comm, &ranks);
        for (int source = 1; source < ranks; ++source) {
            int coords[2], x_begin, y_begin, nx, ny;
            MPI_Cart_coords(world.comm, source, 2, coords);
            partitionDimension(world.grid_size, world.dims[0], coords[0], x_begin, nx);
            partitionDimension(world.grid_size, world.dims[1], coords[1], y_begin, ny);
            MPI_Datatype target_block;
            MPI_Type_vector(nx, ny, world.grid_size, MPI_DOUBLE, &target_block);
            MPI_Type_commit(&target_block);
            MPI_Recv(energies.data() + static_cast<size_t>(x_begin) * n + y_begin,
                     1, target_block, source, 4, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&target_block);
        }
        print_results(energies, "ElementEnergy");
    } else {
        MPI_Send(first, 1, local_block, 0, 4, world.comm);
    }
    MPI_Type_free(&local_block);
}

bool validateResults(const World& world) {
    val_t local_sums[2] = {0.0, 0.0};
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    for (int x = 1; x <= world.nx; ++x) {
        const size_t row = static_cast<size_t>(x) * world.stride;
        for (int y = 1; y <= world.ny; ++y) {
            const size_t i = row + y;
            local_sums[0] += world.current_energy[i];
            local_sums[1] += world.total_flux[i];
            local_max = std::max(world.current_energy[i], local_max);
            local_min = std::min(world.current_energy[i], local_min);
        }
    }
    val_t sums[2], energy_max, energy_min;
    // Only diagnostic sum rounding depends on MPI's reduction order.
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, world.comm);
    if (world.rank != 0) return true;

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", sums[0]);
    printf("  Flux sum: %.2f\n", sums[1]);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
    if (!std::isfinite(sums[0])) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(sums[0]) > 1e-8) {
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    }
    if (!std::isfinite(sums[1])) {
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseNumber(const char* text, int minimum, int maximum, int& value) {
    char* end;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int runBenchmark(int n, int n_iters, bool validate, bool printResults, int world_rank,
                 int world_size) {
    int dims[2];
    chooseProcessGrid(world_size, n, dims);
    const int active_ranks = dims[0] * dims[1];
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (active_comm == MPI_COMM_NULL) return 0;

    World world;
    world.dims[0] = dims[0];
    world.dims[1] = dims[1];
    const int periods[2] = {0, 0};
    MPI_Cart_create(active_comm, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active_comm);
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Cart_coords(world.comm, world.rank, 2, world.coords);
    const idx_t n_elems = static_cast<idx_t>(n) * n;
    if (world.rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (active: %d, process grid: %d x %d)\n\n",
               world_size, active_ranks, dims[0], dims[1]);
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n);
    initializeHaloExchange(world);

    const uint64_t local_mem[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        (world.current_energy.size() + world.next_energy.size() + world.total_flux.size()) * sizeof(val_t)
    };
    uint64_t global_mem[2];
    MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (world.rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (global_mem[0] + global_mem[1]) / (1024.0 * 1024.0),
               global_mem[0] / (1024.0 * 1024.0), global_mem[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(world.comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    const uint64_t result_hash = computeHash(world);
    if (world.rank == 0) {
        seconds = std::max(seconds, MPI_Wtick());
        const double duration_ms = seconds * 1000.0;
        const int n_measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = static_cast<double>(n_measured_iters) * n_elems / seconds / 1e9;
        printf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }
    if (printResults) printEnergyResults(world);
    return validate && !validateResults(world) ? 1 : 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // Parse once, then give every rank the same options and exit status.
    // n, iterations, validation, result output, early exit, status
    int options[6] = {512, 10, 0, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
                const bool grid = strcmp(argv[i], "-n") == 0;
                if (!parseNumber(argv[++i], grid ? 1 : 0, grid ? INT_MAX - 2 : INT_MAX,
                                 options[grid ? 0 : 1])) {
                    printf("Invalid %s: %s\n", grid ? "grid size" : "iteration count", argv[i]);
                    printUsage(argv[0]);
                    options[4] = options[5] = 1;
                    break;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[3] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[4] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[4] = options[5] = 1;
                break;
            }
        }
    }
    MPI_Bcast(options, 6, MPI_INT, 0, MPI_COMM_WORLD);
    int status = options[5];
    if (!options[4]) {
        try {
            status = runBenchmark(options[0], options[1], options[2], options[3], rank, ranks);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
