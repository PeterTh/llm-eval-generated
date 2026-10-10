#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

// The generated mesh has at most four connections, all with unit flux.
constexpr int MAX_CONNECTIONS = 4;
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Connectivity uses indices into the local energy array, including its halo.
struct ElementStatic {
    idx_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx = DEFAULT_MAT_ID;
    uint8_t num_connections = 0;
};

struct World {
    std::array<Material, 3> materials{{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}}};
    std::vector<ElementStatic> elements_static;
    // Only energy is needed by neighboring elements. Accumulated flux can be
    // updated in place, so it needs neither a halo nor a second buffer.
    std::vector<val_t> energy, energy_swap, total_flux;
    std::array<std::vector<val_t>, 2> column_send, column_recv;
    MPI_Comm comm = MPI_COMM_NULL;
    int n = 0, nx = 0, ny = 0, x_begin = 0, y_begin = 0;
    int dims[2] = {0, 0};
    int peers[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t stride = 0;
};

// Choose the largest rectangular process grid that fits the mesh. This also
// handles launches with more ranks than cells without creating empty blocks.
int chooseProcessGrid(int n, int ranks, int dims[2]) {
    int active = 0;
    int best_span = INT_MAX;
    for (int x = 1; x <= std::min(n, ranks); ++x) {
        const int y = std::min(n, ranks / x);
        const int count = x * y;
        const int span = x + y;
        if (count > active || (count == active && span < best_span)) {
            active = count;
            best_span = span;
            dims[0] = x;
            dims[1] = y;
        }
    }
    // Keep the longer local dimension contiguous in memory when possible.
    if (dims[0] < dims[1]) std::swap(dims[0], dims[1]);
    return active;
}

void blockExtent(int n, int parts, int coord, int& begin, int& length) {
    const int base = n / parts;
    const int remainder = n % parts;
    length = base + (coord < remainder);
    begin = coord * base + std::min(coord, remainder);
}

void buildSquare2D(World& world, int n, int rank, int active) {
    world.n = n;
    MPI_Comm participants;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED,
                   rank, &participants);
    if (participants == MPI_COMM_NULL) return;

    const int periods[2] = {0, 0};
    MPI_Cart_create(participants, 2, world.dims, periods, 0, &world.comm);
    MPI_Comm_free(&participants);
    int coords[2];
    MPI_Cart_coords(world.comm, rank, 2, coords);
    blockExtent(n, world.dims[0], coords[0], world.x_begin, world.nx);
    blockExtent(n, world.dims[1], coords[1], world.y_begin, world.ny);
    MPI_Cart_shift(world.comm, 0, 1, &world.peers[0], &world.peers[1]);
    MPI_Cart_shift(world.comm, 1, 1, &world.peers[2], &world.peers[3]);

    world.stride = static_cast<size_t>(world.ny) + 2;
    const size_t local_count = static_cast<size_t>(world.nx) * world.ny;
    const size_t padded_count = (static_cast<size_t>(world.nx) + 2) * world.stride;
    world.elements_static.resize(local_count);
    world.energy.resize(padded_count, 0.0);
    world.energy_swap.resize(padded_count, 0.0);
    world.total_flux.resize(local_count, 0.0);
    for (int side = 0; side < 2; ++side) {
        if (world.peers[side + 2] != MPI_PROC_NULL) {
            world.column_send[side].resize(world.nx);
            world.column_recv[side].resize(world.nx);
        }
    }

    for (int x = 0; x < world.nx; ++x) {
        const int gx = world.x_begin + x;
        for (int y = 0; y < world.ny; ++y) {
            const int gy = world.y_begin + y;
            const size_t p = (static_cast<size_t>(x) + 1) * world.stride + y + 1;
            auto& elem = world.elements_static[static_cast<size_t>(x) * world.ny + y];
            // Preserve the serial accumulation order: +x, -x, +y, -y.
            if (gx < n - 1) elem.connected_idx[elem.num_connections++] = p + world.stride;
            if (gx > 0)     elem.connected_idx[elem.num_connections++] = p - world.stride;
            if (gy < n - 1) elem.connected_idx[elem.num_connections++] = p + 1;
            if (gy > 0)     elem.connected_idx[elem.num_connections++] = p - 1;

            // The assignment order is significant for a one-element mesh.
            if (gx == 0 && gy == 0)         elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1)     elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0)     elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

inline val_t computeFlux(const Material& mat, val_t energy, val_t other_energy) {
    return (other_energy - energy) * mat.transfer_coeff * 0.25;
}

// Updating a row range keeps both state and connectivity accesses sequential.
inline void updateRow(World& world, int x, int y_begin, int y_end) {
    size_t i = static_cast<size_t>(x - 1) * world.ny + y_begin - 1;
    size_t p = static_cast<size_t>(x) * world.stride + y_begin;
    const val_t* energy = world.energy.data();
    for (int y = y_begin; y < y_end; ++y, ++i, ++p) {
        const auto& elem = world.elements_static[i];
        const auto& mat = world.materials[elem.material_idx];
        const val_t current = energy[p];
        val_t flux = mat.external_flow;
        // Unroll the overwhelmingly common four-connection case, retaining
        // the original floating-point operations and addition order.
        if (elem.num_connections == 4) {
            flux += computeFlux(mat, current, energy[elem.connected_idx[0]]);
            flux += computeFlux(mat, current, energy[elem.connected_idx[1]]);
            flux += computeFlux(mat, current, energy[elem.connected_idx[2]]);
            flux += computeFlux(mat, current, energy[elem.connected_idx[3]]);
        } else {
            for (int j = 0; j < elem.num_connections; ++j)
                flux += computeFlux(mat, current, energy[elem.connected_idx[j]]);
        }
        world.energy_swap[p] = current + flux;
        world.total_flux[i] += std::abs(flux);
    }
}

void runSimulation(World& world, int n_iters) {
    if (world.comm == MPI_COMM_NULL) return;
    const size_t s = world.stride;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        val_t* energy = world.energy.data();
        // Receives and sends use only current energy, never cumulative flux.
        if (world.peers[0] != MPI_PROC_NULL)
            MPI_Irecv(energy + 1, world.ny, MPI_DOUBLE, world.peers[0], 1,
                      world.comm, &requests[count++]);
        if (world.peers[1] != MPI_PROC_NULL)
            MPI_Irecv(energy + (static_cast<size_t>(world.nx) + 1) * s + 1,
                      world.ny, MPI_DOUBLE, world.peers[1], 0,
                      world.comm, &requests[count++]);
        for (int side = 0; side < 2; ++side) {
            if (world.peers[side + 2] == MPI_PROC_NULL) continue;
            MPI_Irecv(world.column_recv[side].data(), world.nx, MPI_DOUBLE,
                      world.peers[side + 2], 3 - side, world.comm, &requests[count++]);
            const size_t column = side == 0 ? 1 : world.ny;
            for (int x = 0; x < world.nx; ++x)
                world.column_send[side][x] = energy[(static_cast<size_t>(x) + 1) * s + column];
        }
        if (world.peers[0] != MPI_PROC_NULL)
            MPI_Isend(energy + s + 1, world.ny, MPI_DOUBLE, world.peers[0], 0,
                      world.comm, &requests[count++]);
        if (world.peers[1] != MPI_PROC_NULL)
            MPI_Isend(energy + static_cast<size_t>(world.nx) * s + 1,
                      world.ny, MPI_DOUBLE, world.peers[1], 1,
                      world.comm, &requests[count++]);
        for (int side = 0; side < 2; ++side) {
            if (world.peers[side + 2] != MPI_PROC_NULL)
                MPI_Isend(world.column_send[side].data(), world.nx, MPI_DOUBLE,
                          world.peers[side + 2], 2 + side, world.comm, &requests[count++]);
        }

        // This region never reads a halo, so computation can overlap transfers.
        for (int x = 2; x < world.nx; ++x)
            updateRow(world, x, 2, world.ny);

        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int side = 0; side < 2; ++side) {
            if (world.peers[side + 2] == MPI_PROC_NULL) continue;
            const size_t column = side == 0 ? 0 : static_cast<size_t>(world.ny) + 1;
            for (int x = 0; x < world.nx; ++x)
                energy[(static_cast<size_t>(x) + 1) * s + column] = world.column_recv[side][x];
        }
        // Cover the boundary exactly once, including blocks one cell wide.
        updateRow(world, 1, 1, world.ny + 1);
        if (world.nx > 1) updateRow(world, world.nx, 1, world.ny + 1);
        for (int x = 2; x < world.nx; ++x) {
            updateRow(world, x, 1, 2);
            if (world.ny > 1) updateRow(world, x, world.ny, world.ny + 1);
        }
        world.energy.swap(world.energy_swap);
    }
}

uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const size_t i = static_cast<size_t>(x) * world.ny + y;
            const size_t p = (static_cast<size_t>(x) + 1) * world.stride + y + 1;
            const idx_t global_idx = static_cast<idx_t>(world.x_begin + x) * world.n + world.y_begin + y;
            local_hash ^= (std::bit_cast<uint64_t>(world.energy[p]) + global_idx) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (std::bit_cast<uint64_t>(world.total_flux[i]) + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return global_hash;
}

// Full ordered output is optional. Use subarray receive types to assemble the
// original global ordering directly, without staging or int-sized displacements.
std::vector<val_t> gatherEnergies(const World& world, int rank, int active) {
    std::vector<val_t> result;
    if (world.comm == MPI_COMM_NULL) return result;
    if (rank == 0) {
        result.resize(static_cast<size_t>(world.n) * world.n);
        for (int x = 0; x < world.nx; ++x)
            std::copy_n(world.energy.data() + (static_cast<size_t>(x) + 1) * world.stride + 1,
                        world.ny, result.data() + static_cast<size_t>(world.x_begin + x) * world.n + world.y_begin);
        const int sizes[2] = {world.n, world.n};
        for (int source = 1; source < active; ++source) {
            int coords[2], starts[2], lengths[2];
            MPI_Cart_coords(world.comm, source, 2, coords);
            for (int d = 0; d < 2; ++d)
                blockExtent(world.n, world.dims[d], coords[d], starts[d], lengths[d]);
            MPI_Datatype block;
            MPI_Type_create_subarray(2, sizes, lengths, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data(), 1, block, source, 4, world.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
    } else {
        MPI_Datatype block;
        MPI_Type_create_hvector(world.nx, world.ny,
                               static_cast<MPI_Aint>(world.stride * sizeof(val_t)), MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(world.energy.data() + world.stride + 1, 1, block, 0, 4, world.comm);
        MPI_Type_free(&block);
    }
    return result;
}

bool validateResults(const World& world, int rank) {
    val_t sums[2] = {0.0, 0.0};
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    for (int x = 0; x < world.nx; ++x) {
        for (int y = 0; y < world.ny; ++y) {
            const val_t energy = world.energy[(static_cast<size_t>(x) + 1) * world.stride + y + 1];
            sums[0] += energy;
            sums[1] += world.total_flux[static_cast<size_t>(x) * world.ny + y];
            energy_min = std::min(energy, energy_min);
            energy_max = std::max(energy, energy_max);
        }
    }
    val_t global_sums[2], global_min = 0.0, global_max = 0.0;
    MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_sums[0]);
        printf("  Flux sum: %.2f\n", global_sums[1]);
        printf("  Energy range: [%.6f, %.6f]\n", global_min, global_max);
        if (!std::isfinite(global_sums[0])) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else if (std::abs(global_sums[0]) > 1e-8) {
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(global_sums[1])) {
            printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(global_min) || !std::isfinite(global_max)) {
            printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) printf("  Validation: PASSED\n");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
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

bool parseInteger(const char* value, int& result) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < INT_MIN || parsed > INT_MAX) return false;
    result = static_cast<int>(parsed);
    return true;
}

int benchmarkMain(int argc, char** argv, int rank, int ranks) {
    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            int& value = strcmp(argv[i], "-n") == 0 ? n_elems_root : n_iters;
            if (!parseInteger(argv[++i], value)) {
                if (rank == 0) printf("Invalid integer: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (n_elems_root <= 0 || n_elems_root > INT_MAX - 2) {
        if (rank == 0) printf("Grid size must be between 1 and %d\n", INT_MAX - 2);
        return 1;
    }
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    World world;
    const int active = chooseProcessGrid(n_elems_root, ranks, world.dims);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (active: %d, process grid: %d x %d)\n\n",
               ranks, active, world.dims[0], world.dims[1]);
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n_elems_root, rank, active);

    uint64_t memory[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        (world.energy.size() + world.energy_swap.size() + world.total_flux.size()) * sizeof(val_t)};
    for (int side = 0; side < 2; ++side)
        memory[1] += (world.column_send[side].size() + world.column_recv[side].size()) * sizeof(val_t);
    uint64_t global_memory[2];
    MPI_Reduce(memory, global_memory, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (global_memory[0] + global_memory[1]) / (1024.0 * 1024.0),
               global_memory[0] / (1024.0 * 1024.0), global_memory[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const uint64_t result_hash = computeHash(world);
    if (rank == 0) {
        const double duration_ms = duration * 1000.0;
        const int measured_iters = std::max(n_iters, 0);
        const double time_per_iter = measured_iters ? duration_ms / measured_iters : 0.0;
        const double giga_elems_per_sec = duration > 0.0 ? measured_iters * static_cast<double>(n_elems) / duration / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }
    if (printResults) {
        const auto energies = gatherEnergies(world, rank, active);
        if (rank == 0) print_results(energies, "ElementEnergy");
    }
    const bool valid = !validate || validateResults(world, rank);
    if (world.comm != MPI_COMM_NULL) MPI_Comm_free(&world.comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = benchmarkMain(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
