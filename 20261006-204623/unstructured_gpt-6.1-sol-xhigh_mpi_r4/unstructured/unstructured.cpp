#include <algorithm>
#include <array>
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

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

// The mesh builder produces at most four connections, all with unit flux.
constexpr int MAX_CONNECTIONS = 4;
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Connectivity uses local energy-buffer indices, including halo cells. Unit
// connection coefficients are shared instead of stored once per edge.
struct ElementStatic {
    idx_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx;
    uint8_t num_connections;
};

struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    // Separate energy from cumulative flux: only energy crosses rank boundaries.
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> total_flux;
    int current_buffer = 0;
    int grid_size = 0;
    int rows = 0;
    int cols = 0;
    int x_begin = 0;
    int y_begin = 0;
    size_t stride = 0;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    std::array<std::vector<MPI_Request>, 2> halo_requests;
};

// Choose the largest rectangular process grid that fits the mesh, then minimize
// its aspect ratio. Very small grids can leave surplus ranks inactive, avoiding
// empty blocks and communication through ranks with no owned elements.
std::array<int, 2> chooseProcessGrid(int grid_size, int ranks) {
    std::array<int, 2> dims{1, 1};
    int best_count = 1;
    int best_difference = 0;
    for (int x = 1; x <= std::min(grid_size, ranks); ++x) {
        const int y = std::min(grid_size, ranks / x);
        const int count = x * y;
        const int difference = std::abs(x - y);
        if (count > best_count ||
            (count == best_count && difference < best_difference)) {
            dims = {x, y};
            best_count = count;
            best_difference = difference;
        }
    }
    return dims;
}

void partitionDimension(int n, int partitions, int coordinate,
                        int& begin, int& count) {
    const int base = n / partitions;
    const int remainder = n % partitions;
    begin = coordinate * base + std::min(coordinate, remainder);
    count = base + (coordinate < remainder ? 1 : 0);
}

void buildSquare2D(World& world, int n_elems_root) {
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(world.comm, 2, dims, periods, coords);
    world.grid_size = n_elems_root;
    partitionDimension(n_elems_root, dims[0], coords[0], world.x_begin, world.rows);
    partitionDimension(n_elems_root, dims[1], coords[1], world.y_begin, world.cols);
    world.stride = static_cast<size_t>(world.cols) + 2;

    world.materials = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
    const size_t n_local = static_cast<size_t>(world.rows) * world.cols;
    const size_t n_with_halo = (static_cast<size_t>(world.rows) + 2) * world.stride;
    world.elements_static.resize(n_local);
    world.total_flux.assign(n_local, 0.0);
    for (auto& buffer : world.energy) {
        buffer.assign(n_with_halo, 0.0);
    }

    for (int x = 0; x < world.rows; ++x) {
        const int global_x = world.x_begin + x;
        for (int y = 0; y < world.cols; ++y) {
            const int global_y = world.y_begin + y;
            ElementStatic& elem = world.elements_static[static_cast<size_t>(x) * world.cols + y];
            const size_t index = (static_cast<size_t>(x) + 1) * world.stride + y + 1;
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            // Preserve the serial neighbor order: +x, -x, +y, -y. Cross-rank
            // connections point into the corresponding single-cell halo.
            if (global_x + 1 < n_elems_root) {
                elem.connected_idx[elem.num_connections++] = index + world.stride;
            }
            if (global_x > 0) {
                elem.connected_idx[elem.num_connections++] = index - world.stride;
            }
            if (global_y + 1 < n_elems_root) {
                elem.connected_idx[elem.num_connections++] = index + 1;
            }
            if (global_y > 0) {
                elem.connected_idx[elem.num_connections++] = index - 1;
            }

            // Keep assignment order, including the one-element grid where all
            // four corners coincide and the final assignment is inflow.
            const int last = n_elems_root - 1;
            if (global_x == 0 && global_y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (global_x == 0 && global_y == last) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == last && global_y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == last && global_y == last) elem.material_idx = INFLOW_MAT_ID;
        }
    }
}

void prepareHaloExchange(World& world) {
    int north, south, west, east;
    MPI_Cart_shift(world.comm, 0, 1, &north, &south);
    MPI_Cart_shift(world.comm, 1, 1, &west, &east);
    const int stride = static_cast<int>(world.stride);
    MPI_Type_vector(world.rows, 1, stride, MPI_DOUBLE, &world.column_type);
    MPI_Type_commit(&world.column_type);

    constexpr int TO_NORTH = 0, TO_SOUTH = 1, TO_WEST = 2, TO_EAST = 3;
    for (int buffer = 0; buffer < 2; ++buffer) {
        auto& requests = world.halo_requests[buffer];
        requests.reserve(8);
        val_t* energy = world.energy[buffer].data();
        auto receive = [&](val_t* address, int count, MPI_Datatype datatype,
                           int peer, int tag) {
            if (peer != MPI_PROC_NULL) {
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Recv_init(address, count, datatype, peer, tag, world.comm, &requests.back());
            }
        };
        auto send = [&](val_t* address, int count, MPI_Datatype datatype,
                        int peer, int tag) {
            if (peer != MPI_PROC_NULL) {
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Send_init(address, count, datatype, peer, tag, world.comm, &requests.back());
            }
        };
        // Rows are contiguous; column datatypes avoid application-side packing.
        receive(energy + 1, world.cols, MPI_DOUBLE, north, TO_SOUTH);
        receive(energy + (static_cast<size_t>(world.rows) + 1) * world.stride + 1,
                world.cols, MPI_DOUBLE, south, TO_NORTH);
        receive(energy + world.stride, 1, world.column_type, west, TO_EAST);
        receive(energy + world.stride + world.cols + 1, 1, world.column_type, east, TO_WEST);
        send(energy + world.stride + 1, world.cols, MPI_DOUBLE, north, TO_NORTH);
        send(energy + static_cast<size_t>(world.rows) * world.stride + 1,
             world.cols, MPI_DOUBLE, south, TO_SOUTH);
        send(energy + world.stride + 1, 1, world.column_type, west, TO_WEST);
        send(energy + world.stride + world.cols, 1, world.column_type, east, TO_EAST);
    }
}

inline val_t computeFlux(const Material& mat, val_t this_energy, val_t other_energy) {
    // Retain the original operation order; no reassociation or fast-math.
    return (other_energy - this_energy) * mat.transfer_coeff * CONNECTION_FLUX * 0.25;
}

// Strictly local interior elements always have four connections and default
// material. A separate instantiation exposes the fixed loop bound to the compiler.
template <bool Interior>
void updateRegion(World& world, int x_begin, int x_end, int y_begin, int y_end) {
    const val_t* current = world.energy[world.current_buffer].data();
    val_t* next = world.energy[world.current_buffer ^ 1].data();
    for (int x = x_begin; x < x_end; ++x) {
        size_t local_index = static_cast<size_t>(x - 1) * world.cols + y_begin - 1;
        size_t energy_index = static_cast<size_t>(x) * world.stride + y_begin;
        for (int y = y_begin; y < y_end; ++y, ++local_index, ++energy_index) {
            const ElementStatic& elem = world.elements_static[local_index];
            const Material& mat = world.materials[Interior ? DEFAULT_MAT_ID : elem.material_idx];
            const val_t energy = current[energy_index];
            val_t total_flux = mat.external_flow;
            const int connections = Interior ? MAX_CONNECTIONS : elem.num_connections;
            for (int j = 0; j < connections; ++j) {
                total_flux += computeFlux(mat, energy, current[elem.connected_idx[j]]);
            }
            next[energy_index] = energy + total_flux;
            world.total_flux[local_index] += std::abs(total_flux);
        }
    }
}

void runSimulation(World& world, int n_iters) {
    for (int iter = 0; iter < n_iters; ++iter) {
        auto& requests = world.halo_requests[world.current_buffer];
        if (!requests.empty()) {
            MPI_Startall(static_cast<int>(requests.size()), requests.data());
        }

        // Work that does not read incoming halos overlaps the communication.
        if (world.rows > 2 && world.cols > 2) {
            updateRegion<true>(world, 2, world.rows, 2, world.cols);
        }
        if (!requests.empty()) {
            // Sends must complete as well: this buffer is overwritten next step.
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }

        if (world.rows <= 2 || world.cols <= 2) {
            updateRegion<false>(world, 1, world.rows + 1, 1, world.cols + 1);
        } else {
            updateRegion<false>(world, 1, 2, 1, world.cols + 1);
            updateRegion<false>(world, world.rows, world.rows + 1, 1, world.cols + 1);
            updateRegion<false>(world, 2, world.rows, 1, 2);
            updateRegion<false>(world, 2, world.rows, world.cols, world.cols + 1);
        }
        world.current_buffer ^= 1;
    }
}

void freeHaloExchange(World& world) {
    for (auto& requests : world.halo_requests) {
        for (MPI_Request& request : requests) {
            MPI_Request_free(&request);
        }
    }
    MPI_Type_free(&world.column_type);
}

uint64_t computeHash(const World& world) {
    uint64_t hash_value = 0;
    const auto& energy = world.energy[world.current_buffer];
    for (int x = 0; x < world.rows; ++x) {
        size_t local_index = static_cast<size_t>(x) * world.cols;
        size_t energy_index = (static_cast<size_t>(x) + 1) * world.stride + 1;
        idx_t global_index = static_cast<idx_t>(world.x_begin + x) * world.grid_size + world.y_begin;
        for (int y = 0; y < world.cols; ++y, ++local_index, ++energy_index, ++global_index) {
            uint64_t energy_bits, flux_bits;
            // memcpy avoids the strict-aliasing violation in the serial hash.
            std::memcpy(&energy_bits, &energy[energy_index], sizeof(energy_bits));
            std::memcpy(&flux_bits, &world.total_flux[local_index], sizeof(flux_bits));
            hash_value ^= (energy_bits + global_index) * 0x9e3779b97f4a7c15ULL;
            hash_value ^= (flux_bits + global_index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash_value;
}

// Gather only for explicitly requested output/validation, outside the timed
// simulation. Receive each block directly into global row-major order so both
// the FNV hash and the original serial summation order are preserved exactly.
void gatherResults(const World& world, bool gather_flux,
                   std::vector<val_t>& energy, std::vector<val_t>& flux) {
    int rank, ranks;
    MPI_Comm_rank(world.comm, &rank);
    MPI_Comm_size(world.comm, &ranks);
    constexpr int ENERGY_TAG = 4, FLUX_TAG = 5;
    if (rank == 0) {
        const size_t n_global = static_cast<size_t>(world.grid_size) * world.grid_size;
        energy.resize(n_global);
        if (gather_flux) flux.resize(n_global);
        const auto& local_energy = world.energy[world.current_buffer];
        for (int x = 0; x < world.rows; ++x) {
            const size_t destination = static_cast<size_t>(world.x_begin + x) * world.grid_size + world.y_begin;
            std::copy_n(local_energy.data() + (static_cast<size_t>(x) + 1) * world.stride + 1,
                        world.cols, energy.data() + destination);
            if (gather_flux) {
                std::copy_n(world.total_flux.data() + static_cast<size_t>(x) * world.cols,
                            world.cols, flux.data() + destination);
            }
        }
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(world.comm, 2, dims, periods, coords);
        for (int peer = 1; peer < ranks; ++peer) {
            MPI_Cart_coords(world.comm, peer, 2, coords);
            int x_begin, rows, y_begin, cols;
            partitionDimension(world.grid_size, dims[0], coords[0], x_begin, rows);
            partitionDimension(world.grid_size, dims[1], coords[1], y_begin, cols);
            MPI_Datatype block_type;
            MPI_Type_vector(rows, cols, world.grid_size, MPI_DOUBLE, &block_type);
            MPI_Type_commit(&block_type);
            const size_t destination = static_cast<size_t>(x_begin) * world.grid_size + y_begin;
            MPI_Recv(energy.data() + destination, 1, block_type, peer, ENERGY_TAG,
                     world.comm, MPI_STATUS_IGNORE);
            if (gather_flux) {
                MPI_Recv(flux.data() + destination, 1, block_type, peer, FLUX_TAG,
                         world.comm, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&block_type);
        }
    } else {
        MPI_Datatype owned_type;
        MPI_Type_vector(world.rows, world.cols, static_cast<int>(world.stride),
                        MPI_DOUBLE, &owned_type);
        MPI_Type_commit(&owned_type);
        MPI_Send(world.energy[world.current_buffer].data() + world.stride + 1,
                 1, owned_type, 0, ENERGY_TAG, world.comm);
        MPI_Type_free(&owned_type);
        if (gather_flux) {
            // Using a datatype also supports local arrays exceeding INT_MAX.
            MPI_Datatype flux_type;
            MPI_Type_vector(world.rows, world.cols, world.cols, MPI_DOUBLE, &flux_type);
            MPI_Type_commit(&flux_type);
            MPI_Send(world.total_flux.data(), 1, flux_type, 0, FLUX_TAG, world.comm);
            MPI_Type_free(&flux_type);
        }
    }
}

bool validateResults(const std::vector<val_t>& energy, const std::vector<val_t>& flux) {
    val_t energy_sum = 0.0;
    val_t flux_sum = 0.0;
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < energy.size(); ++i) {
        energy_sum += energy[i];
        flux_sum += flux[i];
        energy_max = std::max(energy[i], energy_max);
        energy_min = std::min(energy[i], energy_min);
    }
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

bool parseInteger(const char* text, int minimum, int maximum, int& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

int runBenchmark(int argc, char** argv, int world_rank, int world_ranks) {
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseInteger(argv[++i], 1, INT_MAX - 2, n_elems_root)) {
                if (world_rank == 0) fprintf(stderr, "Invalid grid size: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseInteger(argv[++i], 0, INT_MAX, n_iters)) {
                if (world_rank == 0) fprintf(stderr, "Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    const auto process_grid = chooseProcessGrid(n_elems_root, world_ranks);
    const int active_ranks = process_grid[0] * process_grid[1];
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &active_comm);
    if (active_comm == MPI_COMM_NULL) return 0;

    World world;
    const int dims[2] = {process_grid[0], process_grid[1]};
    const int periods[2] = {0, 0};
    MPI_Cart_create(active_comm, 2, dims, periods, 0, &world.comm);
    MPI_Comm_free(&active_comm);
    int rank;
    MPI_Comm_rank(world.comm, &rank);
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d/%d (process grid: %d x %d)\n\n",
               active_ranks, world_ranks, dims[0], dims[1]);
        printf("Building unstructured mesh...\n");
    }
    buildSquare2D(world, n_elems_root);
    prepareHaloExchange(world);
    const uint64_t local_memory[2] = {
        world.elements_static.size() * sizeof(ElementStatic),
        (world.energy[0].size() + world.energy[1].size() + world.total_flux.size()) * sizeof(val_t)
    };
    uint64_t global_memory[2] = {};
    MPI_Reduce(local_memory, global_memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
    if (rank == 0) {
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (global_memory[0] + global_memory[1]) / (1024.0 * 1024.0),
               global_memory[0] / (1024.0 * 1024.0), global_memory[1] / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    MPI_Barrier(world.comm);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_seconds, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    const uint64_t local_hash = computeHash(world);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);

    if (rank == 0) {
        const double duration_ms = duration_seconds * 1000.0;
        const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
        const double giga_elems_per_sec = duration_seconds > 0.0
            ? (static_cast<double>(n_iters) * n_elems) / duration_seconds / 1e9 : 0.0;
        printf("Computation time: %.3f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }

    int valid = 1;
    if (printResults || validate) {
        std::vector<val_t> energy, flux;
        gatherResults(world, validate, energy, flux);
        if (rank == 0) {
            if (printResults) print_results(energy, "ElementEnergy");
            if (validate) valid = validateResults(energy, flux) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, world.comm);
    }
    freeHaloExchange(world);
    MPI_Comm_free(&world.comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        // An allocation failure must not strand peers in a collective.
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
