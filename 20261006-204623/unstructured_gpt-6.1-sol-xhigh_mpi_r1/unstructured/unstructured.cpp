#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
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
constexpr val_t CONNECTION_FLUX = 1.0;
constexpr uint8_t DEFAULT_MAT_ID = 0;
constexpr uint8_t INFLOW_MAT_ID = 1;
constexpr uint8_t OUTFLOW_MAT_ID = 2;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

// Indices address local energies, including the one-cell ghost layer.
// Unit connection coefficients are shared rather than repeated per element.
struct ElementStatic {
    idx_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx;
    uint8_t num_connections;
};

struct World {
    std::array<Material, 3> materials{{{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}}};
    std::vector<ElementStatic> elements_static;
    std::array<std::vector<val_t>, 2> energy;
    // Neighbors only read energy, so cumulative flux can be updated in place.
    std::vector<val_t> total_flux;
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    std::array<std::array<MPI_Request, 8>, 2> requests;
    int request_count = 0;
    int current_buffer = 0;
    int rank = 0;
    int size = 1;
    int grid_size = 0;
    int dims[2] = {1, 1};
    int coords[2] = {0, 0};
    int start[2] = {0, 0};
    int extent[2] = {0, 0};
    size_t stride = 0;
};

void partition(const int n, const int parts, const int coordinate,
               int& start, int& extent) {
    const int base = n / parts;
    const int remainder = n % parts;
    extent = base + (coordinate < remainder);
    start = coordinate * base + std::min(coordinate, remainder);
}

// Use as many nonempty rectangular blocks as possible, preferring a square
// process grid. Extra processes participate in world-level finalization only.
int chooseProcessGrid(const int n, const int processes, int dims[2]) {
    int active = 0;
    int best_difference = std::numeric_limits<int>::max();
    for (int x = 1; x <= std::min(n, processes); ++x) {
        const int y = std::min(n, processes / x);
        const int count = x * y;
        const int difference = std::abs(x - y);
        // Favor contiguous row exchanges when both orientations are equivalent.
        if (count > active || (count == active &&
            (difference < best_difference || (difference == best_difference && x > dims[0])))) {
            active = count;
            best_difference = difference;
            dims[0] = x;
            dims[1] = y;
        }
    }
    return active;
}

void setupHaloExchange(World& world) {
    int x_minus, x_plus, y_minus, y_plus;
    MPI_Cart_shift(world.comm, 0, 1, &x_minus, &x_plus);
    MPI_Cart_shift(world.comm, 1, 1, &y_minus, &y_plus);
    const int nx = world.extent[0];
    const int ny = world.extent[1];
    const size_t stride = world.stride;

    if (y_minus != MPI_PROC_NULL || y_plus != MPI_PROC_NULL) {
        MPI_Type_vector(nx, 1, static_cast<int>(stride), MPI_DOUBLE, &world.column_type);
        MPI_Type_commit(&world.column_type);
    }

    // Persistent requests amortize setup and refer to each fixed energy buffer.
    // Tags identify the direction in which the sender's data travels.
    for (int buffer = 0; buffer < 2; ++buffer) {
        val_t* energy = world.energy[buffer].data();
        auto& requests = world.requests[buffer];
        int count = 0;
        if (x_minus != MPI_PROC_NULL)
            MPI_Recv_init(energy + 1, ny, MPI_DOUBLE, x_minus, 0,
                          world.comm, &requests[count++]);
        if (x_plus != MPI_PROC_NULL)
            MPI_Recv_init(energy + (nx + 1) * stride + 1, ny, MPI_DOUBLE, x_plus, 1,
                          world.comm, &requests[count++]);
        if (y_minus != MPI_PROC_NULL)
            MPI_Recv_init(energy + stride, 1, world.column_type, y_minus, 2,
                          world.comm, &requests[count++]);
        if (y_plus != MPI_PROC_NULL)
            MPI_Recv_init(energy + stride + ny + 1, 1, world.column_type, y_plus, 3,
                          world.comm, &requests[count++]);
        if (x_minus != MPI_PROC_NULL)
            MPI_Send_init(energy + stride + 1, ny, MPI_DOUBLE, x_minus, 1,
                          world.comm, &requests[count++]);
        if (x_plus != MPI_PROC_NULL)
            MPI_Send_init(energy + nx * stride + 1, ny, MPI_DOUBLE, x_plus, 0,
                          world.comm, &requests[count++]);
        if (y_minus != MPI_PROC_NULL)
            MPI_Send_init(energy + stride + 1, 1, world.column_type, y_minus, 3,
                          world.comm, &requests[count++]);
        if (y_plus != MPI_PROC_NULL)
            MPI_Send_init(energy + stride + ny, 1, world.column_type, y_plus, 2,
                          world.comm, &requests[count++]);
        world.request_count = count;
    }
}

void freeHaloExchange(World& world) {
    for (auto& requests : world.requests)
        for (int i = 0; i < world.request_count; ++i)
            MPI_Request_free(&requests[i]);
    if (world.column_type != MPI_DATATYPE_NULL)
        MPI_Type_free(&world.column_type);
}

// Build only the owned portion of the original unstructured square mesh.
void buildSquare2D(World& world, const int n_elems_root) {
    world.grid_size = n_elems_root;
    MPI_Comm_rank(world.comm, &world.rank);
    MPI_Comm_size(world.comm, &world.size);
    int periods[2];
    MPI_Cart_get(world.comm, 2, world.dims, periods, world.coords);
    for (int axis = 0; axis < 2; ++axis)
        partition(n_elems_root, world.dims[axis], world.coords[axis],
                  world.start[axis], world.extent[axis]);
    const int nx = world.extent[0];
    const int ny = world.extent[1];
    world.stride = static_cast<size_t>(ny) + 2;
    const size_t storage = (static_cast<size_t>(nx) + 2) * world.stride;
    world.elements_static.resize(static_cast<size_t>(nx) * ny);
    for (auto& energy : world.energy)
        energy.assign(storage, 0.0);
    world.total_flux.assign(storage, 0.0);

    for (int x = 1; x <= nx; ++x) {
        const int global_x = world.start[0] + x - 1;
        for (int y = 1; y <= ny; ++y) {
            const int global_y = world.start[1] + y - 1;
            const size_t i = static_cast<size_t>(x) * world.stride + y;
            ElementStatic& elem = world.elements_static[static_cast<size_t>(x - 1) * ny + y - 1];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;
            // Preserve the original summation order: +x, -x, +y, -y.
            if (global_x + 1 < n_elems_root)
                elem.connected_idx[elem.num_connections++] = i + world.stride;
            if (global_x > 0)
                elem.connected_idx[elem.num_connections++] = i - world.stride;
            if (global_y + 1 < n_elems_root)
                elem.connected_idx[elem.num_connections++] = i + 1;
            if (global_y > 0)
                elem.connected_idx[elem.num_connections++] = i - 1;

            const int last = n_elems_root - 1;
            // Separate assignments also preserve the 1x1 mesh's final inflow.
            if (global_x == 0 && global_y == 0) elem.material_idx = INFLOW_MAT_ID;
            if (global_x == 0 && global_y == last) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == last && global_y == 0) elem.material_idx = OUTFLOW_MAT_ID;
            if (global_x == last && global_y == last) elem.material_idx = INFLOW_MAT_ID;
        }
    }
    setupHaloExchange(world);
}

inline val_t computeFlux(const Material& mat, val_t energy, val_t other_energy) {
    return (other_energy - energy) * mat.transfer_coeff * CONNECTION_FLUX * 0.25;
}

template <bool full_connectivity>
inline void updateElement(World& world, const val_t* energy, val_t* next,
                          const size_t element_index, const size_t energy_index) {
    const ElementStatic& elem = world.elements_static[element_index];
    const Material& mat = world.materials[elem.material_idx];
    const val_t current = energy[energy_index];
    val_t flux = mat.external_flow;
    const int connections = full_connectivity ? MAX_CONNECTIONS : elem.num_connections;
    for (int j = 0; j < connections; ++j)
        flux += computeFlux(mat, current, energy[elem.connected_idx[j]]);
    next[energy_index] = current + flux;
    world.total_flux[energy_index] += std::abs(flux);
}

void runSimulation(World& world, const int n_iters) {
    const int nx = world.extent[0];
    const int ny = world.extent[1];
    const size_t stride = world.stride;
    for (int iter = 0; iter < n_iters; ++iter) {
        const int buffer = world.current_buffer;
        const val_t* energy = world.energy[buffer].data();
        val_t* next = world.energy[1 - buffer].data();
        bool pending = world.request_count != 0;
        if (pending)
            MPI_Startall(world.request_count, world.requests[buffer].data());

        // No ghost values are read here, allowing communication to overlap work.
        for (int x = 2; x < nx; ++x) {
            const size_t row = static_cast<size_t>(x) * stride;
            const size_t element_row = static_cast<size_t>(x - 1) * ny;
            for (int y = 2; y < ny; ++y)
                updateElement<true>(world, energy, next, element_row + y - 1, row + y);
            // Drive progress on MPI implementations without a progress thread.
            if (pending && (x & 31) == 0) {
                int completed;
                MPI_Testall(world.request_count, world.requests[buffer].data(),
                            &completed, MPI_STATUSES_IGNORE);
                pending = !completed;
            }
        }
        if (pending)
            MPI_Waitall(world.request_count, world.requests[buffer].data(), MPI_STATUSES_IGNORE);

        // Update the block perimeter exactly once, including thin 1-cell blocks.
        for (int y = 1; y <= ny; ++y)
            updateElement<false>(world, energy, next, y - 1, stride + y);
        if (nx > 1) {
            const size_t row = static_cast<size_t>(nx) * stride;
            const size_t element_row = static_cast<size_t>(nx - 1) * ny;
            for (int y = 1; y <= ny; ++y)
                updateElement<false>(world, energy, next, element_row + y - 1, row + y);
        }
        for (int x = 2; x < nx; ++x) {
            const size_t row = static_cast<size_t>(x) * stride;
            const size_t element_row = static_cast<size_t>(x - 1) * ny;
            updateElement<false>(world, energy, next, element_row, row + 1);
            if (ny > 1)
                updateElement<false>(world, energy, next, element_row + ny - 1, row + ny);
        }
        world.current_buffer = 1 - buffer;
    }
}

bool validateResults(const World& world) {
    val_t sums[2] = {0.0, 0.0};
    val_t energy_max = std::numeric_limits<val_t>::lowest();
    val_t energy_min = std::numeric_limits<val_t>::max();
    const auto& energy = world.energy[world.current_buffer];
    for (int x = 1; x <= world.extent[0]; ++x) {
        const size_t row = static_cast<size_t>(x) * world.stride;
        for (int y = 1; y <= world.extent[1]; ++y) {
            const size_t i = row + y;
            sums[0] += energy[i];
            sums[1] += world.total_flux[i];
            energy_max = std::max(energy[i], energy_max);
            energy_min = std::min(energy[i], energy_min);
        }
    }
    val_t global_sums[2], global_max, global_min;
    MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, world.comm);
    MPI_Reduce(&energy_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
    MPI_Reduce(&energy_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, world.comm);
    if (world.rank != 0) return true;

    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", global_sums[0]);
    printf("  Flux sum: %.2f\n", global_sums[1]);
    printf("  Energy range: [%.6f, %.6f]\n", global_min, global_max);
    constexpr val_t energy_epsilon = 1e-8;
    if (!std::isfinite(global_sums[0])) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(global_sums[0]) > energy_epsilon)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(global_sums[1])) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(global_max) || !std::isfinite(global_min)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const World& world) {
    uint64_t local_hash = 0;
    const auto& energy = world.energy[world.current_buffer];
    for (int x = 1; x <= world.extent[0]; ++x) {
        const idx_t global_row = static_cast<idx_t>(world.start[0] + x - 1) * world.grid_size;
        const size_t row = static_cast<size_t>(x) * world.stride;
        for (int y = 1; y <= world.extent[1]; ++y) {
            const size_t local_i = row + y;
            const idx_t global_i = global_row + world.start[1] + y - 1;
            const uint64_t e_bits = std::bit_cast<uint64_t>(energy[local_i]);
            const uint64_t f_bits = std::bit_cast<uint64_t>(world.total_flux[local_i]);
            local_hash ^= (e_bits + global_i) * 0x9e3779b97f4a7c15ULL;
            local_hash ^= (f_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, world.comm);
    return global_hash;
}

// Only -r needs a global array: preserve print_results' row-major Kahan sum,
// samples and order-sensitive FNV hash. Datatypes place tiles directly into it.
void printDistributedResults(const World& world) {
    constexpr int result_tag = 4;
    const auto& energy = world.energy[world.current_buffer];
    if (world.rank != 0) {
        MPI_Datatype tile;
        MPI_Type_vector(world.extent[0], world.extent[1], static_cast<int>(world.stride),
                        MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Send(energy.data() + world.stride + 1, 1, tile, 0, result_tag, world.comm);
        MPI_Type_free(&tile);
        return;
    }

    const int n = world.grid_size;
    std::vector<val_t> results(static_cast<size_t>(n) * n);
    std::vector<MPI_Request> requests;
    requests.reserve(world.size - 1);
    for (int rank = 1; rank < world.size; ++rank) {
        int coords[2], start[2], extent[2];
        MPI_Cart_coords(world.comm, rank, 2, coords);
        for (int axis = 0; axis < 2; ++axis)
            partition(n, world.dims[axis], coords[axis], start[axis], extent[axis]);
        MPI_Datatype tile;
        MPI_Type_vector(extent[0], extent[1], n, MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Request request;
        MPI_Irecv(results.data() + static_cast<size_t>(start[0]) * n + start[1],
                  1, tile, rank, result_tag, world.comm, &request);
        requests.push_back(request);
        MPI_Type_free(&tile);
    }
    for (int x = 0; x < world.extent[0]; ++x)
        std::copy_n(energy.data() + static_cast<size_t>(x + 1) * world.stride + 1,
                    world.extent[1],
                    results.data() + static_cast<size_t>(world.start[0] + x) * n + world.start[1]);
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    print_results(results, "ElementEnergy");
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
    char* end;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max())
        return false;
    value = static_cast<int>(parsed);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    // grid size, iterations, validation, external results, parse status
    int options[5] = {512, 10, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
                const int option = strcmp(argv[i], "-n") == 0 ? 0 : 1;
                if (!parseInteger(argv[++i], options[option])) {
                    printf("Invalid numeric argument: %s\n", argv[i]);
                    options[4] = 1;
                    break;
                }
            } else if (strcmp(argv[i], "-v") == 0) {
                options[2] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[3] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                options[4] = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                options[4] = 1;
                break;
            }
        }
        if (options[4] == 0 && (options[0] < 1 || options[0] > std::numeric_limits<int>::max() - 2)) {
            printf("Grid size must be positive and leave room for ghost cells.\n");
            options[4] = 1;
        }
        if (options[4] != 0) printUsage(argv[0]);
    }
    MPI_Bcast(options, 5, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[4] != 0) {
        MPI_Finalize();
        return options[4] == 2 ? 0 : 1;
    }

    const int n_elems_root = options[0];
    const int n_iters = options[1];
    const idx_t n_elems = static_cast<idx_t>(n_elems_root) * n_elems_root;
    int dims[2];
    const int active_processes = chooseProcessGrid(n_elems_root, processes, dims);
    MPI_Comm active_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active_processes ? 0 : MPI_UNDEFINED, rank, &active_comm);
    int exit_status = 0;
    try {
        if (rank < active_processes) {
            World world;
            const int periods[2] = {0, 0};
            // Retain rank zero as the sole output process.
            MPI_Cart_create(active_comm, 2, dims, periods, 0, &world.comm);
            MPI_Comm_free(&active_comm);
            if (rank == 0) {
                printf("Unstructured Mesh Energy Transfer Benchmark\n");
                printf("============================================\n");
                printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
                printf("Iterations: %d\n", n_iters);
                printf("Validation: %s\n", options[2] ? "enabled" : "disabled");
                printf("MPI processes: %d (%d active, %d x %d blocks)\n\n",
                       processes, active_processes, dims[0], dims[1]);
                printf("Building unstructured mesh...\n");
            }
            buildSquare2D(world, n_elems_root);
            const uint64_t memory[2] = {
                world.elements_static.size() * sizeof(ElementStatic),
                (world.energy[0].size() * 2 + world.total_flux.size()) * sizeof(val_t)
            };
            uint64_t global_memory[2];
            MPI_Reduce(memory, global_memory, 2, MPI_UINT64_T, MPI_SUM, 0, world.comm);
            if (rank == 0) {
                printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                       (global_memory[0] + global_memory[1]) / (1024.0 * 1024.0),
                       global_memory[0] / (1024.0 * 1024.0),
                       global_memory[1] / (1024.0 * 1024.0));
                printf("Running simulation...\n");
            }
            MPI_Barrier(world.comm);
            const double start_time = MPI_Wtime();
            runSimulation(world, n_iters);
            const double local_time = MPI_Wtime() - start_time;
            double elapsed;
            MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, world.comm);
            freeHaloExchange(world);
            const uint64_t result_hash = computeHash(world);
            if (rank == 0) {
                const double duration_ms = elapsed * 1000.0;
                const double time_per_iter = duration_ms / std::max(n_iters, 1);
                const double giga_elems_per_sec = elapsed > 0.0
                    ? static_cast<double>(n_iters) * n_elems / elapsed / 1e9 : 0.0;
                printf("Computation time: %.3f ms\n", duration_ms);
                printf("Performance:\n");
                printf("  Time per iteration: %.4f ms\n", time_per_iter);
                printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
                printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
                printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
            }
            if (options[3]) printDistributedResults(world);
            if (options[2] && !validateResults(world)) exit_status = 1;
            MPI_Comm_free(&world.comm);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Includes idle ranks and propagates root's validation status to every rank.
    MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_status;
}
