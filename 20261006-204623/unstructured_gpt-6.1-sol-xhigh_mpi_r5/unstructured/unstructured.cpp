#include <algorithm>
#include <bit>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 4;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Only the partition perimeter needs explicit connectivity. Interior elements
// have the same four offsets; storing them implicitly avoids streaming a large
// adjacency array through memory on every iteration.
struct ElementStatic {
    size_t local_idx;
    size_t connected_idx[MAX_CONNECTIONS];
    Material material;
    int num_connections;
};

struct World {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int grid_size = 0;
    int dims[2] = {1, 1};
    int rows = 0;
    int cols = 0;
    int first_row = 0;
    int first_col = 0;
    int up = MPI_PROC_NULL;
    int down = MPI_PROC_NULL;
    int left = MPI_PROC_NULL;
    int right = MPI_PROC_NULL;
    size_t stride = 0;
    std::vector<ElementStatic> elements_static;
    // Structure of arrays: only energy is read by neighbors or sent over MPI.
    // Accumulated flux can be updated in place because it has no dependencies.
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux;
    std::vector<val_t> send_left, send_right, recv_left, recv_right;
};

int blockSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}

int blockStart(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts);
}

// Maximize the number of nonempty ranks, then minimize the perimeter of each
// block. This also handles prime process counts and more ranks than elements.
void chooseProcessGrid(int n, int processes, int dims[2]) {
    int best_count = 0;
    int best_difference = INT_MAX;
    for (int rows = 1; rows <= std::min(n, processes); ++rows) {
        const int cols = std::min(n, processes / rows);
        const int count = rows * cols;
        const int difference = std::abs(rows - cols);
        if (count > best_count || (count == best_count && difference < best_difference)) {
            best_count = count;
            best_difference = difference;
            dims[0] = rows;
            dims[1] = cols;
        }
    }
}

void buildSquare2D(World& world, int n) {
    world.grid_size = n;
    int coordinates[2];
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    MPI_Cart_coords(world.comm, world.rank, 2, coordinates);
    MPI_Cart_shift(world.comm, 0, 1, &world.up, &world.down);
    MPI_Cart_shift(world.comm, 1, 1, &world.left, &world.right);
    world.rows = blockSize(n, world.dims[0], coordinates[0]);
    world.cols = blockSize(n, world.dims[1], coordinates[1]);
    world.first_row = blockStart(n, world.dims[0], coordinates[0]);
    world.first_col = blockStart(n, world.dims[1], coordinates[1]);
    world.stride = static_cast<size_t>(world.cols) + 2;
    const size_t cells = (static_cast<size_t>(world.rows) + 2) * world.stride;
    world.energy.resize(cells, 0.0);
    world.energy_swap.resize(cells, 0.0);
    world.total_flux.resize(cells, 0.0);
    if (world.left != MPI_PROC_NULL || world.right != MPI_PROC_NULL) {
        world.send_left.resize(world.rows);
        world.send_right.resize(world.rows);
        world.recv_left.resize(world.rows);
        world.recv_right.resize(world.rows);
    }

    const size_t boundary_count = static_cast<size_t>(world.rows) * world.cols -
        static_cast<size_t>(std::max(world.rows - 2, 0)) * std::max(world.cols - 2, 0);
    world.elements_static.reserve(boundary_count);
    const auto addBoundary = [&](int x, int y) {
        const int global_x = world.first_row + x - 1;
        const int global_y = world.first_col + y - 1;
        ElementStatic elem{};
        elem.local_idx = static_cast<size_t>(x) * world.stride + y;
        elem.material = {0.8, 0.0};
        // Preserve the original corner-assignment order, including the 1x1 case.
        if (global_x == 0 && global_y == 0) elem.material.external_flow = 0.5;
        if (global_x == 0 && global_y == n - 1) elem.material.external_flow = -0.5;
        if (global_x == n - 1 && global_y == 0) elem.material.external_flow = -0.5;
        if (global_x == n - 1 && global_y == n - 1) elem.material.external_flow = 0.5;
        // Same accumulation order as the serial mesh: +x, -x, +y, -y.
        if (global_x + 1 < n)
            elem.connected_idx[elem.num_connections++] = elem.local_idx + world.stride;
        if (global_x > 0)
            elem.connected_idx[elem.num_connections++] = elem.local_idx - world.stride;
        if (global_y + 1 < n)
            elem.connected_idx[elem.num_connections++] = elem.local_idx + 1;
        if (global_y > 0)
            elem.connected_idx[elem.num_connections++] = elem.local_idx - 1;
        world.elements_static.push_back(elem);
    };
    for (int x = 1; x <= world.rows; ++x) {
        if (x == 1 || x == world.rows) {
            for (int y = 1; y <= world.cols; ++y) addBoundary(x, y);
        } else {
            addBoundary(x, 1);
            if (world.cols > 1) addBoundary(x, world.cols);
        }
    }
}

inline val_t computeFlux(const Material& mat, val_t energy, val_t other_energy) {
    // Every connection in buildSquare2D has coefficient 1.0. Keep the original
    // multiplication and summation order; do not reassociate into a Laplacian.
    return (other_energy - energy) * mat.transfer_coeff * 1.0 * 0.25;
}

void updateInterior(World& world, MPI_Request requests[], int count) {
    const val_t* __restrict current = world.energy.data();
    val_t* __restrict next = world.energy_swap.data();
    val_t* __restrict accumulated = world.total_flux.data();
    const size_t stride = world.stride;
    const Material material{0.8, 0.0};
    int communication_complete = count == 0;
    for (int x = 2; x < world.rows; ++x) {
        const size_t first = static_cast<size_t>(x) * stride;
        // Contiguous arrays let the compiler vectorize all four neighbor reads.
        for (int y = 2; y < world.cols; ++y) {
            const size_t i = first + y;
            const val_t energy = current[i];
            val_t flux = material.external_flow;
            flux += computeFlux(material, energy, current[i + stride]);
            flux += computeFlux(material, energy, current[i - stride]);
            flux += computeFlux(material, energy, current[i + 1]);
            flux += computeFlux(material, energy, current[i - 1]);
            next[i] = energy + flux;
            accumulated[i] += std::abs(flux);
        }
        // Make progress on MPI implementations without a background progress
        // thread, especially when large halos use the rendezvous protocol.
        if (!communication_complete && (x & 31) == 0)
            MPI_Testall(count, requests, &communication_complete, MPI_STATUSES_IGNORE);
    }
}

void runSimulation(World& world, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        val_t* current = world.energy.data();
        const size_t stride = world.stride;
        // Receive directly into the row ghosts; columns use contiguous buffers.
        if (world.up != MPI_PROC_NULL)
            MPI_Irecv(current + 1, world.cols, MPI_DOUBLE, world.up, 0,
                      world.comm, &requests[count++]);
        if (world.down != MPI_PROC_NULL)
            MPI_Irecv(current + (static_cast<size_t>(world.rows) + 1) * stride + 1,
                      world.cols, MPI_DOUBLE, world.down, 1, world.comm, &requests[count++]);
        if (world.left != MPI_PROC_NULL)
            MPI_Irecv(world.recv_left.data(), world.rows, MPI_DOUBLE, world.left, 2,
                      world.comm, &requests[count++]);
        if (world.right != MPI_PROC_NULL)
            MPI_Irecv(world.recv_right.data(), world.rows, MPI_DOUBLE, world.right, 3,
                      world.comm, &requests[count++]);
        if (world.left != MPI_PROC_NULL || world.right != MPI_PROC_NULL) {
            for (int x = 1; x <= world.rows; ++x) {
                const size_t first = static_cast<size_t>(x) * stride;
                if (world.left != MPI_PROC_NULL) world.send_left[x - 1] = current[first + 1];
                if (world.right != MPI_PROC_NULL) world.send_right[x - 1] = current[first + world.cols];
            }
        }
        if (world.down != MPI_PROC_NULL)
            MPI_Isend(current + static_cast<size_t>(world.rows) * stride + 1,
                      world.cols, MPI_DOUBLE, world.down, 0, world.comm, &requests[count++]);
        if (world.up != MPI_PROC_NULL)
            MPI_Isend(current + stride + 1, world.cols, MPI_DOUBLE, world.up, 1,
                      world.comm, &requests[count++]);
        if (world.right != MPI_PROC_NULL)
            MPI_Isend(world.send_right.data(), world.rows, MPI_DOUBLE, world.right, 2,
                      world.comm, &requests[count++]);
        if (world.left != MPI_PROC_NULL)
            MPI_Isend(world.send_left.data(), world.rows, MPI_DOUBLE, world.left, 3,
                      world.comm, &requests[count++]);

        updateInterior(world, requests, count);
        if (count != 0) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int x = 1; x <= world.rows; ++x) {
            const size_t first = static_cast<size_t>(x) * stride;
            if (world.left != MPI_PROC_NULL) current[first] = world.recv_left[x - 1];
            if (world.right != MPI_PROC_NULL) current[first + world.cols + 1] = world.recv_right[x - 1];
        }
        for (const ElementStatic& elem : world.elements_static) {
            const size_t i = elem.local_idx;
            const val_t energy = current[i];
            val_t flux = elem.material.external_flow;
            for (int j = 0; j < elem.num_connections; ++j)
                flux += computeFlux(elem.material, energy, current[elem.connected_idx[j]]);
            world.energy_swap[i] = energy + flux;
            world.total_flux[i] += std::abs(flux);
        }
        world.energy.swap(world.energy_swap);
    }
}

bool validateResults(const World& world) {
    val_t local_sums[2] = {0.0, 0.0};
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    int local_finite = 1;
    for (int x = 1; x <= world.rows; ++x) {
        for (int y = 1; y <= world.cols; ++y) {
            const size_t i = static_cast<size_t>(x) * world.stride + y;
            const val_t energy = world.energy[i];
            const val_t flux = world.total_flux[i];
            local_sums[0] += energy;
            local_sums[1] += flux;
            local_max = std::max(energy, local_max);
            local_min = std::min(energy, local_min);
            local_finite &= std::isfinite(energy) && std::isfinite(flux);
        }
    }
    val_t sums[2] = {0.0, 0.0}, energy_max = 0.0, energy_min = 0.0;
    int finite = 0;
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, world.comm);
    MPI_Reduce(&local_finite, &finite, 1, MPI_INT, MPI_MIN, 0, world.comm);
    int valid = 1;
    if (world.rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", sums[0]);
        printf("  Flux sum: %.2f\n", sums[1]);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!finite || !std::isfinite(sums[0]) || !std::isfinite(sums[1]) ||
            !std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            printf("  ERROR: Simulation results are not finite\n");
            valid = 0;
        } else {
            if (std::abs(sums[0]) > 1e-8)
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            printf("  Validation: PASSED\n");
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, world.comm);
    return valid != 0;
}

uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    for (int x = 1; x <= world.rows; ++x) {
        const idx_t global_row = static_cast<idx_t>(world.first_row + x - 1) * world.grid_size;
        for (int y = 1; y <= world.cols; ++y) {
            const size_t i = static_cast<size_t>(x) * world.stride + y;
            const idx_t global_idx = global_row + world.first_col + y - 1;
            local_hash ^= (std::bit_cast<uint64_t>(world.energy[i]) + global_idx) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (std::bit_cast<uint64_t>(world.total_flux[i]) + global_idx) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return hash;
}

// Gather only when -r is requested. Vector datatypes place each rectangular
// block directly in global row-major order, with no second full-grid buffer
// and no MPI int count limit on the total number of elements.
void printGlobalResults(const World& world) {
    if (world.rank != 0) {
        MPI_Datatype block;
        MPI_Type_vector(world.rows, world.cols, static_cast<int>(world.stride), MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(world.energy.data() + world.stride + 1, 1, block, 0, 4, world.comm);
        MPI_Type_free(&block);
        return;
    }
    std::vector<double> energyData(static_cast<size_t>(world.grid_size) * world.grid_size);
    for (int x = 0; x < world.rows; ++x) {
        std::copy_n(world.energy.data() + (static_cast<size_t>(x) + 1) * world.stride + 1,
                    world.cols, energyData.data() + static_cast<size_t>(world.first_row + x) *
                    world.grid_size + world.first_col);
    }
    for (int rank = 1; rank < world.size; ++rank) {
        int coordinates[2];
        MPI_Cart_coords(world.comm, rank, 2, coordinates);
        const int rows = blockSize(world.grid_size, world.dims[0], coordinates[0]);
        const int cols = blockSize(world.grid_size, world.dims[1], coordinates[1]);
        const int first_row = blockStart(world.grid_size, world.dims[0], coordinates[0]);
        const int first_col = blockStart(world.grid_size, world.dims[1], coordinates[1]);
        MPI_Datatype block;
        MPI_Type_vector(rows, cols, world.grid_size, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(energyData.data() + static_cast<size_t>(first_row) * world.grid_size + first_col,
                 1, block, rank, 4, world.comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(energyData, "ElementEnergy");
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

bool parseInteger(const char* argument, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = strtol(argument, &end, 10);
    if (errno != 0 || end == argument || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX)
        return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            int& value = strcmp(argv[i], "-n") == 0 ? n_elems_root : n_iters;
            if (!parseInteger(argv[++i], value)) {
                if (rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
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
    if (n_elems_root < 1 || n_elems_root > INT_MAX - 2) {
        if (rank == 0) fprintf(stderr, "Grid size must be between 1 and %d\n", INT_MAX - 2);
        MPI_Finalize();
        return 1;
    }

    World world;
    chooseProcessGrid(n_elems_root, processes, world.dims);
    const int active_processes = world.dims[0] * world.dims[1];
    MPI_Comm active_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active_processes ? 0 : MPI_UNDEFINED, rank, &active_comm);
    if (active_comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const int periods[2] = {0, 0};
    MPI_Cart_create(active_comm, 2, world.dims, periods, 0, &world.comm);
    MPI_Comm_free(&active_comm);
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (process grid: %d x %d)\n\n",
               active_processes, world.dims[0], world.dims[1]);
        printf("Building unstructured mesh...\n");
    }
    int exit_code = 0;
    try {
        buildSquare2D(world, n_elems_root);
        const uint64_t local_memory[2] = {
            world.elements_static.capacity() * sizeof(ElementStatic),
            (world.energy.capacity() + world.energy_swap.capacity() + world.total_flux.capacity() +
             world.send_left.capacity() + world.send_right.capacity() +
             world.recv_left.capacity() + world.recv_right.capacity()) * sizeof(val_t)
        };
        uint64_t memory[2] = {0, 0};
        MPI_Reduce(local_memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (memory[0] + memory[1]) / (1024.0 * 1024.0),
                   memory[0] / (1024.0 * 1024.0), memory[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(world.comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters);
        const double local_duration = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&local_duration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
        const uint64_t hash = computeHash(world);
        if (rank == 0) {
            const double duration_ms = duration * 1000.0;
            const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
            const double giga_elems_per_sec = duration > 0.0 ?
                static_cast<double>(std::max(n_iters, 0)) * static_cast<double>(n_elems) /
                duration / 1e9 : 0.0;
            printf("Computation time: %.3f ms\n", duration_ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", time_per_iter);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (printResults) printGlobalResults(world);
        if (validate && !validateResults(world)) exit_code = 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Comm_free(&world.comm);
    MPI_Finalize();
    return exit_code;
}
