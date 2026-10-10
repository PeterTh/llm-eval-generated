#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using val_t = double;
constexpr int MAX_CONNECTIONS = 4;
constexpr unsigned char DEFAULT_MAT_ID = 0;
constexpr unsigned char INFLOW_MAT_ID = 1;
constexpr unsigned char OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Connectivity uses local indices, including the one-element halo. All
// connections in this mesh have unit flux, so no coefficient array is needed.
struct ElementStatic {
    size_t connected_idx[MAX_CONNECTIONS];
    unsigned char material_idx = DEFAULT_MAT_ID;
    unsigned char num_connections = 0;
};

struct World {
    std::array<Material, 3> materials{{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}}};
    std::vector<ElementStatic> elements_static;
    // Separate arrays let MPI exchange only energy; cumulative flux is local.
    std::vector<val_t> energy, energy_swap, total_flux;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype column = MPI_DATATYPE_NULL;
    int rank = 0, size = 1;
    int n = 0, rows = 0, cols = 0, x0 = 0, y0 = 0, pitch = 0;
    int neighbors[4] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
};

// Quotient/remainder partitioning keeps every active rank nonempty.
void block(int n, int parts, int coord, int& first, int& count) {
    count = n / parts + (coord < n % parts);
    first = coord * (n / parts) + std::min(coord, n % parts);
}

void buildSquare2D(World& world, int n, MPI_Comm comm) {
    world.comm = comm;
    world.n = n;
    MPI_Comm_rank(comm, &world.rank);
    MPI_Comm_size(comm, &world.size);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm, 2, dims, periods, coords);
    block(n, dims[0], coords[0], world.x0, world.rows);
    block(n, dims[1], coords[1], world.y0, world.cols);
    world.pitch = world.cols + 2;
    MPI_Cart_shift(comm, 0, 1, &world.neighbors[1], &world.neighbors[0]);
    MPI_Cart_shift(comm, 1, 1, &world.neighbors[3], &world.neighbors[2]);

    world.elements_static.resize(static_cast<size_t>(world.rows) * world.cols);
    const size_t storage = (static_cast<size_t>(world.rows) + 2) * world.pitch;
    world.energy.assign(storage, 0.0);
    world.energy_swap.assign(storage, 0.0);
    world.total_flux.assign(world.elements_static.size(), 0.0);
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            auto& elem = world.elements_static[static_cast<size_t>(x) * world.cols + y];
            const size_t i = (static_cast<size_t>(x) + 1) * world.pitch + y + 1;
            const int gx = world.x0 + x, gy = world.y0 + y;
            // Preserve the original accumulation order: +x, -x, +y, -y.
            if (gx + 1 < n) elem.connected_idx[elem.num_connections++] = i + world.pitch;
            if (gx > 0) elem.connected_idx[elem.num_connections++] = i - world.pitch;
            if (gy + 1 < n) elem.connected_idx[elem.num_connections++] = i + 1;
            if (gy > 0) elem.connected_idx[elem.num_connections++] = i - 1;
            if (gx == 0 && gy == 0) elem.material_idx = INFLOW_MAT_ID;
            if (gx == 0 && gy == n - 1) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = INFLOW_MAT_ID;
        }
    }
    MPI_Type_vector(world.rows, 1, world.pitch, MPI_DOUBLE, &world.column);
    MPI_Type_commit(&world.column);
}

inline void updateElement(World& world, int x, int y) {
    const size_t local = static_cast<size_t>(x) * world.cols + y;
    const size_t i = (static_cast<size_t>(x) + 1) * world.pitch + y + 1;
    const auto& elem = world.elements_static[local];
    const auto& mat = world.materials[elem.material_idx];
    const val_t current = world.energy[i];
    val_t flux = mat.external_flow;
    for (unsigned char j = 0; j < elem.num_connections; ++j) {
        flux += (world.energy[elem.connected_idx[j]] - current) * mat.transfer_coeff * 0.25;
    }
    world.energy_swap[i] = current + flux;
    world.total_flux[local] += std::abs(flux);
}

void runSimulation(World& world, int n_iters) {
    const size_t pitch = world.pitch;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        val_t* e = world.energy.data();
        // Direction tags distinguish the two faces even in narrow domains.
        auto exchange = [&](int neighbor, val_t* recv, val_t* send,
                            int length, MPI_Datatype type, int recv_tag, int send_tag) {
            if (neighbor == MPI_PROC_NULL) return;
            MPI_Irecv(recv, length, type, neighbor, recv_tag, world.comm, &requests[count++]);
            MPI_Isend(send, length, type, neighbor, send_tag, world.comm, &requests[count++]);
        };
        exchange(world.neighbors[0], e + (world.rows + 1) * pitch + 1,
                 e + world.rows * pitch + 1, world.cols, MPI_DOUBLE, 1, 0);
        exchange(world.neighbors[1], e + 1, e + pitch + 1,
                 world.cols, MPI_DOUBLE, 0, 1);
        exchange(world.neighbors[2], e + pitch + world.cols + 1,
                 e + pitch + world.cols, 1, world.column, 3, 2);
        exchange(world.neighbors[3], e + pitch, e + pitch + 1,
                 1, world.column, 2, 3);

        // Physical boundary cells need no halo and can also be computed early.
        const int x_begin = world.neighbors[1] == MPI_PROC_NULL ? 0 : 1;
        const int x_end = world.rows - (world.neighbors[0] == MPI_PROC_NULL ? 0 : 1);
        const int y_begin = world.neighbors[3] == MPI_PROC_NULL ? 0 : 1;
        const int y_end = world.cols - (world.neighbors[2] == MPI_PROC_NULL ? 0 : 1);
        for (int x = x_begin; x < x_end; ++x)
            for (int y = y_begin; y < y_end; ++y) updateElement(world, x, y);

        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Each boundary cell is updated exactly once, including one-cell blocks.
        for (int x = 0; x < world.rows; ++x) {
            if (x < x_begin || x >= x_end) {
                for (int y = 0; y < world.cols; ++y) updateElement(world, x, y);
            } else {
                for (int y = 0; y < std::min(y_begin, world.cols); ++y) updateElement(world, x, y);
                for (int y = std::max(y_begin, y_end); y < world.cols; ++y) updateElement(world, x, y);
            }
        }
        world.energy.swap(world.energy_swap);
    }
}

uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0, global_hash = 0;
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const size_t local = static_cast<size_t>(x) * world.cols + y;
            const size_t i = (static_cast<size_t>(x) + 1) * world.pitch + y + 1;
            const uint64_t global = static_cast<uint64_t>(world.x0 + x) * world.n + world.y0 + y;
            uint64_t energy_bits, flux_bits;
            // memcpy preserves the original bit hash without aliasing violations.
            std::memcpy(&energy_bits, &world.energy[i], sizeof(energy_bits));
            std::memcpy(&flux_bits, &world.total_flux[local], sizeof(flux_bits));
            local_hash ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
}

bool validateResults(const World& world) {
    val_t sums[2] = {0.0, 0.0}, global_sums[2];
    val_t energy_min = std::numeric_limits<val_t>::max();
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    int finite = 1, all_finite;
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const val_t energy = world.energy[(static_cast<size_t>(x) + 1) * world.pitch + y + 1];
            const val_t flux = world.total_flux[static_cast<size_t>(x) * world.cols + y];
            sums[0] += energy;
            sums[1] += flux;
            energy_min = std::min(energy, energy_min);
            energy_max = std::max(energy, energy_max);
            finite &= std::isfinite(energy) && std::isfinite(flux);
        }
    }
    val_t global_min, global_max;
    MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, world.comm);
    MPI_Allreduce(&energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, world.comm);
    MPI_Allreduce(&energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, world.comm);
    MPI_Allreduce(&finite, &all_finite, 1, MPI_INT, MPI_MIN, world.comm);
    const bool valid = all_finite && std::isfinite(global_sums[0]) && std::isfinite(global_sums[1]) &&
                       std::isfinite(global_min) && std::isfinite(global_max);
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", global_sums[0]);
        printf("  Flux sum: %.2f\n", global_sums[1]);
        printf("  Energy range: [%.6f, %.6f]\n", global_min, global_max);
        if (std::abs(global_sums[0]) > 1e-8)
            printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        if (!valid) printf("  ERROR: Simulation results are not finite\n");
        printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

// Ordered output needs the global array only when -r is explicitly requested.
// Chunk messages to avoid MPI's int count limit for large local partitions.
void printDistributedResults(const World& world) {
    std::vector<val_t> packed(static_cast<size_t>(world.rows) * world.cols);
    for (int x = 0; x < world.rows; ++x)
        std::copy_n(world.energy.data() + (static_cast<size_t>(x) + 1) * world.pitch + 1,
                    world.cols, packed.data() + static_cast<size_t>(x) * world.cols);
    if (world.rank != 0) {
        for (size_t offset = 0; offset < packed.size();) {
            const int count = static_cast<int>(std::min(packed.size() - offset,
                                                       static_cast<size_t>(std::numeric_limits<int>::max())));
            MPI_Send(packed.data() + offset, count, MPI_DOUBLE, 0, 10, world.comm);
            offset += count;
        }
        return;
    }
    std::vector<val_t> global(static_cast<size_t>(world.n) * world.n);
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    for (int rank = 0; rank < world.size; ++rank) {
        MPI_Cart_coords(world.comm, rank, 2, coords);
        int x0, y0, rows, cols;
        block(world.n, dims[0], coords[0], x0, rows);
        block(world.n, dims[1], coords[1], y0, cols);
        if (rank != 0) {
            packed.resize(static_cast<size_t>(rows) * cols);
            for (size_t offset = 0; offset < packed.size();) {
                const int count = static_cast<int>(std::min(packed.size() - offset,
                                                           static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Recv(packed.data() + offset, count, MPI_DOUBLE, rank, 10, world.comm, MPI_STATUS_IGNORE);
                offset += count;
            }
        }
        for (int x = 0; x < rows; ++x)
            std::copy_n(packed.data() + static_cast<size_t>(x) * cols, cols,
                        global.data() + static_cast<size_t>(x0 + x) * world.n + y0);
    }
    print_results(global, "ElementEnergy");
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

bool parseInteger(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max() - 2)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n_elems_root = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const bool grid = strcmp(argv[i], "-n") == 0;
            if (!parseInteger(argv[++i], grid ? n_elems_root : n_iters) || (grid && n_elems_root == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid %s: %s\n", grid ? "grid size" : "iteration count", argv[i]);
                MPI_Finalize();
                return 1;
            }
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

    // Choose the largest nonempty rectangular process grid, preferring square
    // blocks. Excess ranks (e.g. more ranks than elements) do no mesh work.
    int dims[2] = {1, 1}, active = 1;
    for (int x = 1; x <= std::min(n_elems_root, size); ++x) {
        const int y = std::min(n_elems_root, size / x);
        const int product = x * y;
        if (product > active || (product == active && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            active = product;
            dims[0] = x;
            dims[1] = y;
        }
    }
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (rank >= active) {
        MPI_Finalize();
        return 0;
    }
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 2, dims, periods, 0, &cart);
    MPI_Comm_free(&workers);

    int status = 0;
    try {
        const uint64_t n_elems = static_cast<uint64_t>(n_elems_root) * n_elems_root;
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %llu elements\n", n_elems_root, n_elems_root,
                   static_cast<unsigned long long>(n_elems));
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d active of %d (%d x %d blocks)\n\n", active, size, dims[0], dims[1]);
            printf("Building unstructured mesh...\n");
        }
        World world;
        buildSquare2D(world, n_elems_root, cart);
        const uint64_t static_mem = world.elements_static.size() * sizeof(ElementStatic);
        const uint64_t dynamic_mem = (world.energy.size() * 2 + world.total_flux.size()) * sizeof(val_t);
        uint64_t local_mem[2] = {static_mem, dynamic_mem}, global_mem[2];
        MPI_Reduce(local_mem, global_mem, 2, MPI_UINT64_T, MPI_SUM, 0, cart);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (global_mem[0] + global_mem[1]) / (1024.0 * 1024.0),
                   global_mem[0] / (1024.0 * 1024.0), global_mem[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        const uint64_t result_hash = computeHash(world);
        if (rank == 0) {
            const double duration_ms = seconds * 1000.0;
            const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
            const double giga_elems_per_sec = seconds > 0 ? static_cast<double>(n_iters) * n_elems / seconds / 1e9 : 0.0;
            printf("Computation time: %.3f ms\n", duration_ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", time_per_iter);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
            printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(result_hash));
        }
        if (printResults) printDistributedResults(world);
        if (validate && !validateResults(world)) status = 1;
        MPI_Type_free(&world.column);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return status;
}
