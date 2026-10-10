#include <algorithm>
#include <mpi.h>
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

// Types to represent unstructured mesh elements
using idx_t = uint64_t;
using val_t = double;

// Maximum number of connections per element (for a 2D grid: 4 neighbors)
constexpr int MAX_CONNECTIONS = 4;

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

// Each rank owns a rectangular tile; energy arrays include a one-cell halo.
// Connectivity uses local padded indices, so no ownership lookup is needed
// in the iteration loop. Only energy (not accumulated flux) crosses ranks.
struct World {
    Material materials[3] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> energy, energy_swap, total_flux;
    int rows, cols, x_begin, y_begin;
    size_t stride;
};

int blockSize(int n, int parts, int coordinate) {
    return n / parts + (coordinate < n % parts);
}

int blockBegin(int n, int parts, int coordinate) {
    return coordinate * (n / parts) + std::min(coordinate, n % parts);
}

void buildSquare2D(World& world, int n, const int dims[2], const int coords[2]) {
    world.rows = blockSize(n, dims[0], coords[0]);
    world.cols = blockSize(n, dims[1], coords[1]);
    world.x_begin = blockBegin(n, dims[0], coords[0]);
    world.y_begin = blockBegin(n, dims[1], coords[1]);
    world.stride = static_cast<size_t>(world.cols) + 2;
    const size_t count = static_cast<size_t>(world.rows) * world.cols;
    world.elements_static.resize(count);
    world.total_flux.resize(count, 0.0);
    world.energy.resize((static_cast<size_t>(world.rows) + 2) * world.stride, 0.0);
    world.energy_swap.resize(world.energy.size(), 0.0);
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            const int gx = world.x_begin + x, gy = world.y_begin + y;
            const size_t index = (static_cast<size_t>(x) + 1) * world.stride + y + 1;
            auto& elem = world.elements_static[static_cast<size_t>(x) * world.cols + y];
            elem.material_idx = 0;
            // Preserve the original assignment order, including the 1x1 case.
            if (gx == 0 && gy == 0) elem.material_idx = 1;
            if (gx == 0 && gy == n - 1) elem.material_idx = 2;
            if (gx == n - 1 && gy == 0) elem.material_idx = 2;
            if (gx == n - 1 && gy == n - 1) elem.material_idx = 1;
            elem.num_connections = 0;
            auto connect = [&](size_t neighbor) {
                elem.connected_idx[elem.num_connections] = neighbor;
                elem.connected_flux[elem.num_connections++] = 1.0;
            };
            // Keep the reference's +x, -x, +y, -y accumulation order.
            if (gx + 1 < n) connect(index + world.stride);
            if (gx > 0) connect(index - world.stride);
            if (gy + 1 < n) connect(index + 1);
            if (gy > 0) connect(index - 1);
        }
    }
}

inline void updateElement(World& world, int x, int y) {
    const size_t owned = static_cast<size_t>(x - 1) * world.cols + y - 1;
    const size_t index = static_cast<size_t>(x) * world.stride + y;
    const auto& elem = world.elements_static[owned];
    const auto& mat = world.materials[elem.material_idx];
    const val_t energy = world.energy[index];
    val_t flux = mat.external_flow;
    for (idx_t j = 0; j < elem.num_connections; ++j) {
        flux += (world.energy[elem.connected_idx[j]] - energy) *
                mat.transfer_coeff * elem.connected_flux[j] * 0.25;
    }
    world.energy_swap[index] = energy + flux;
    world.total_flux[owned] += std::abs(flux);
}

void runSimulation(World& world, int n_iters, MPI_Comm comm) {
    int minus_x, plus_x, minus_y, plus_y;
    MPI_Cart_shift(comm, 0, 1, &minus_x, &plus_x);
    MPI_Cart_shift(comm, 1, 1, &minus_y, &plus_y);
    MPI_Datatype column;
    MPI_Type_vector(world.rows, 1, static_cast<int>(world.stride), MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    const size_t stride = world.stride;
    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        auto receive = [&](int peer, size_t offset, int length, MPI_Datatype type, int tag) {
            if (peer != MPI_PROC_NULL)
                MPI_Irecv(world.energy.data() + offset, length, type, peer, tag,
                          comm, &requests[count++]);
        };
        auto send = [&](int peer, size_t offset, int length, MPI_Datatype type, int tag) {
            if (peer != MPI_PROC_NULL)
                MPI_Isend(world.energy.data() + offset, length, type, peer, tag,
                          comm, &requests[count++]);
        };
        receive(minus_x, 1, world.cols, MPI_DOUBLE, 0);
        receive(plus_x, (static_cast<size_t>(world.rows) + 1) * stride + 1,
                world.cols, MPI_DOUBLE, 1);
        receive(minus_y, stride, 1, column, 2);
        receive(plus_y, stride + world.cols + 1, 1, column, 3);
        send(plus_x, static_cast<size_t>(world.rows) * stride + 1, world.cols, MPI_DOUBLE, 0);
        send(minus_x, stride + 1, world.cols, MPI_DOUBLE, 1);
        send(plus_y, stride + world.cols, 1, column, 2);
        send(minus_y, stride + 1, 1, column, 3);

        // Interior updates do not depend on incoming halo data.
        for (int x = 2; x < world.rows; ++x)
            for (int y = 2; y < world.cols; ++y)
                updateElement(world, x, y);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int y = 1; y <= world.cols; ++y) {
            updateElement(world, 1, y);
            if (world.rows > 1) updateElement(world, world.rows, y);
        }
        for (int x = 2; x < world.rows; ++x) {
            updateElement(world, x, 1);
            if (world.cols > 1) updateElement(world, x, world.cols);
        }
        world.energy.swap(world.energy_swap);
    }
    MPI_Type_free(&column);
}

// XOR is order-independent; global indices make this identical to the serial hash.
uint64_t computeHash(const World& world, int n) {
    uint64_t result = 0;
    for (int x = 0; x < world.rows; ++x) {
        for (int y = 0; y < world.cols; ++y) {
            uint64_t energy_bits, flux_bits;
            const size_t local = static_cast<size_t>(x) * world.cols + y;
            const size_t padded = (static_cast<size_t>(x) + 1) * world.stride + y + 1;
            const uint64_t global = static_cast<uint64_t>(world.x_begin + x) * n + world.y_begin + y;
            std::memcpy(&energy_bits, &world.energy[padded], sizeof(energy_bits));
            std::memcpy(&flux_bits, &world.total_flux[local], sizeof(flux_bits));
            result ^= (energy_bits + global) * 0x9e3779b97f4a7c15ULL;
            result ^= (flux_bits + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return result;
}

// Gather only on explicit output/validation requests. Restore row-major order
// to preserve the reference's sequential sums and order-sensitive output hash.
std::vector<ElementDynamic> gatherResults(const World& world, int n,
                                         const int dims[2], MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<ElementDynamic> tile(world.total_flux.size());
    for (int x = 0; x < world.rows; ++x)
        for (int y = 0; y < world.cols; ++y) {
            const size_t i = static_cast<size_t>(x) * world.cols + y;
            tile[i] = {world.energy[(static_cast<size_t>(x) + 1) * world.stride + y + 1],
                       world.total_flux[i]};
        }
    MPI_Datatype element;
    MPI_Type_contiguous(2, MPI_DOUBLE, &element);
    MPI_Type_commit(&element);
    static_assert(sizeof(ElementDynamic) == 2 * sizeof(double));
    std::vector<ElementDynamic> result;
    if (rank == 0) result.resize(static_cast<size_t>(n) * n);
    for (int source = 0; source < size; ++source) {
        if (rank != 0 && rank != source) continue;
        int coords[2];
        MPI_Cart_coords(comm, source, 2, coords);
        const int rows = blockSize(n, dims[0], coords[0]);
        const int cols = blockSize(n, dims[1], coords[1]);
        const size_t count = static_cast<size_t>(rows) * cols;
        if (rank == 0 && source != 0) tile.resize(count);
        // Chunk messages to avoid MPI's int count limit on large grids.
        for (size_t offset = 0; offset < count; ) {
            const int length = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
            if (source != 0) {
                if (rank == 0) MPI_Recv(tile.data() + offset, length, element, source, 4,
                                        comm, MPI_STATUS_IGNORE);
                else MPI_Send(tile.data() + offset, length, element, 0, 4, comm);
            }
            offset += length;
        }
        if (rank == 0) {
            const int xb = blockBegin(n, dims[0], coords[0]);
            const int yb = blockBegin(n, dims[1], coords[1]);
            for (int x = 0; x < rows; ++x)
                std::copy_n(tile.data() + static_cast<size_t>(x) * cols, cols,
                            result.data() + static_cast<size_t>(xb + x) * n + yb);
        }
    }
    MPI_Type_free(&element);
    return result;
}

// Validate simulation results
bool validateResults(const std::vector<ElementDynamic>& elements) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    
    for (const auto& elem : elements) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 512, n_iters = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const bool grid = strcmp(argv[i], "-n") == 0;
            char* end;
            const long value = strtol(argv[++i], &end, 10);
            if (*end != '\0' || end == argv[i] || value < (grid ? 1 : 0) || value > INT_MAX - 2) {
                if (rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            if (grid) n = static_cast<int>(value);
            else n_iters = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    // Use the largest rectangular process grid that has no empty tiles,
    // preferring a square shape to minimize the communication surface.
    int dims[2] = {1, 1};
    int active = 1;
    for (int x = 1; x <= std::min(n, size); ++x) {
        const int y = std::min(n, size / x);
        const int used = x * y;
        if (used > active || (used == active && std::abs(x - y) < std::abs(dims[0] - dims[1]))) {
            active = used;
            dims[0] = x; dims[1] = y;
        }
    }
    MPI_Comm workers, comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (workers == MPI_COMM_NULL) { MPI_Finalize(); return 0; }
    int periods[2] = {0, 0}, coords[2];
    MPI_Cart_create(workers, 2, dims, periods, 0, &comm);
    MPI_Comm_free(&workers);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    const size_t n_elems = static_cast<size_t>(n) * n;

    try {
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %zu elements\n", n, n, n_elems);
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (process grid: %d x %d)\n\n", active, dims[0], dims[1]);
            printf("Building unstructured mesh...\n");
        }
        World world;
        buildSquare2D(world, n, dims, coords);
        const uint64_t local_mem[2] = {
            world.elements_static.size() * sizeof(ElementStatic),
            (world.energy.size() * 2 + world.total_flux.size()) * sizeof(val_t)
        };
        uint64_t memory[2];
        MPI_Reduce(local_mem, memory, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (memory[0] + memory[1]) / (1024.0 * 1024.0),
                   memory[0] / (1024.0 * 1024.0), memory[1] / (1024.0 * 1024.0));
            printf("Running simulation...\n");
        }
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters, comm);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        const uint64_t local_hash = computeHash(world, n);
        uint64_t result_hash;
        MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);
        if (rank == 0) {
            const double duration_ms = seconds * 1000.0;
            const double rate = seconds > 0 ? static_cast<double>(n_iters) * n_elems / seconds / 1e9 : 0;
            printf("Computation time: %.3f ms\n", duration_ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", duration_ms / std::max(n_iters, 1));
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
        }
        bool valid = true;
        if (printResults || validate) {
            const auto elements = gatherResults(world, n, dims, comm);
            if (rank == 0) {
                if (printResults) {
                    std::vector<double> energyData;
                    energyData.reserve(elements.size());
                    for (const auto& elem : elements) energyData.push_back(elem.current_energy);
                    print_results(energyData, "ElementEnergy");
                }
                if (validate) valid = validateResults(elements);
            }
        }
        int status = valid ? 0 : 1;
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
        MPI_Comm_free(&comm);
        MPI_Finalize();
        return status;
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
