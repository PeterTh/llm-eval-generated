#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;

constexpr int MAX_CONNECTIONS = 8;

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

struct ElementStatic {
    idx_t material_idx;
    idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS];
    val_t connected_flux[MAX_CONNECTIONS];
};

// Owned elements are contiguous and followed by two halo rows in energy.
// The structure-of-arrays dynamic state keeps the communicated rows contiguous
// and avoids transmitting the accumulated flux, which is never read remotely.
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<val_t> energy;
    std::vector<val_t> energy_swap;
    std::vector<val_t> total_flux;
    std::vector<val_t> total_flux_swap;
    int grid_size = 0;
    int first_row = 0;
    int local_rows = 0;
    int previous_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;
};

constexpr idx_t DEFAULT_MAT_ID = 0;
constexpr idx_t INFLOW_MAT_ID = 1;
constexpr idx_t OUTFLOW_MAT_ID = 2;

static void rowPartition(int n, int size, int rank, int& first, int& count) {
    const int base = n / size;
    const int remainder = n % size;
    count = base + (rank < remainder ? 1 : 0);
    first = rank * base + std::min(rank, remainder);
}

void buildSquare2D(World& world, int n, int rank, int size) {
    world.grid_size = n;
    rowPartition(n, size, rank, world.first_row, world.local_rows);
    if (world.local_rows != 0) {
        world.previous_rank = world.first_row == 0 ? MPI_PROC_NULL : rank - 1;
        world.next_rank = world.first_row + world.local_rows == n ? MPI_PROC_NULL : rank + 1;
    }

    world.materials.emplace_back(Material{0.8, 0.0});
    world.materials.emplace_back(Material{0.8, 0.5});
    world.materials.emplace_back(Material{0.8, -0.5});

    const size_t owned = static_cast<size_t>(world.local_rows) * n;
    const size_t halo = world.local_rows == 0 ? 0 : static_cast<size_t>(2) * n;
    world.elements_static.resize(owned);
    world.energy.assign(owned + halo, 0.0);
    world.energy_swap.resize(owned + halo);
    world.total_flux.assign(owned, 0.0);
    world.total_flux_swap.resize(owned);

    const idx_t previous_halo = owned;
    const idx_t next_halo = owned + n;
    const int last = n - 1;

    for (int local_x = 0; local_x < world.local_rows; ++local_x) {
        const int x = world.first_row + local_x;
        for (int y = 0; y < n; ++y) {
            const size_t local_idx = static_cast<size_t>(local_x) * n + y;
            ElementStatic& elem = world.elements_static[local_idx];
            elem.material_idx = DEFAULT_MAT_ID;
            elem.num_connections = 0;

            if ((x == 0 && y == 0) || (x == last && y == last)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((x == 0 && y == last) || (x == last && y == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            }

            // Preserve the original connection and floating-point evaluation order:
            // x+1, x-1, y+1, y-1.
            if (x + 1 < n) {
                elem.connected_idx[elem.num_connections] =
                    local_x + 1 < world.local_rows ? local_idx + n : next_halo + y;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (x > 0) {
                elem.connected_idx[elem.num_connections] =
                    local_x > 0 ? local_idx - n : previous_halo + y;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (y + 1 < n) {
                elem.connected_idx[elem.num_connections] = local_idx + 1;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
            if (y > 0) {
                elem.connected_idx[elem.num_connections] = local_idx - 1;
                elem.connected_flux[elem.num_connections++] = 1.0;
            }
        }
    }
}

inline val_t computeFlux(const Material& mat, val_t this_energy,
                         val_t connection_flux, val_t other_energy) {
    return (other_energy - this_energy) * mat.transfer_coeff * connection_flux * 0.25;
}

static inline void updateRows(World& world, int begin_row, int end_row) {
    const int n = world.grid_size;
    for (int local_x = begin_row; local_x < end_row; ++local_x) {
        const size_t begin = static_cast<size_t>(local_x) * n;
        const size_t end = begin + n;
        for (size_t i = begin; i < end; ++i) {
            const ElementStatic& elem = world.elements_static[i];
            const Material& mat = world.materials[elem.material_idx];
            const val_t this_energy = world.energy[i];
            val_t flux = mat.external_flow;
            for (idx_t j = 0; j < elem.num_connections; ++j) {
                flux += computeFlux(mat, this_energy, elem.connected_flux[j],
                                    world.energy[elem.connected_idx[j]]);
            }
            world.energy_swap[i] = this_energy + flux;
            world.total_flux_swap[i] = world.total_flux[i] + std::abs(flux);
        }
    }
}

void runSimulation(World& world, int n_iters) {
    const int n = world.grid_size;
    const size_t owned = world.elements_static.size();

    for (int iter = 0; iter < n_iters; ++iter) {
        MPI_Request requests[4];
        int request_count = 0;
        if (world.local_rows != 0) {
            if (world.previous_rank != MPI_PROC_NULL) {
                MPI_Irecv(world.energy.data() + owned, n, MPI_DOUBLE,
                          world.previous_rank, 101, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.energy.data(), n, MPI_DOUBLE,
                          world.previous_rank, 100, MPI_COMM_WORLD, &requests[request_count++]);
            }
            if (world.next_rank != MPI_PROC_NULL) {
                MPI_Irecv(world.energy.data() + owned + n, n, MPI_DOUBLE,
                          world.next_rank, 100, MPI_COMM_WORLD, &requests[request_count++]);
                MPI_Isend(world.energy.data() + owned - n, n, MPI_DOUBLE,
                          world.next_rank, 101, MPI_COMM_WORLD, &requests[request_count++]);
            }

            // Interior rows have no dependency on halo traffic.
            if (world.local_rows > 2) {
                updateRows(world, 1, world.local_rows - 1);
            }
            if (request_count != 0) {
                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
            }
            updateRows(world, 0, 1);
            if (world.local_rows > 1) {
                updateRows(world, world.local_rows - 1, world.local_rows);
            }
        }
        std::swap(world.energy, world.energy_swap);
        std::swap(world.total_flux, world.total_flux_swap);
    }
}

bool validateResults(const World& world, int rank) {
    val_t local_energy_sum = 0.0;
    val_t local_flux_sum = 0.0;
    val_t local_max = std::numeric_limits<val_t>::lowest();
    val_t local_min = std::numeric_limits<val_t>::max();
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        local_energy_sum += world.energy[i];
        local_flux_sum += world.total_flux[i];
        local_max = std::max(local_max, world.energy[i]);
        local_min = std::min(local_min, world.energy[i]);
    }

    val_t energy_sum = 0.0, flux_sum = 0.0;
    val_t energy_max = 0.0, energy_min = 0.0;
    MPI_Reduce(&local_energy_sum, &energy_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_flux_sum, &flux_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        std::printf("Validation results:\n");
        std::printf("  Energy sum: %.12f\n", energy_sum);
        std::printf("  Flux sum: %.2f\n", flux_sum);
        std::printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);
        if (!std::isfinite(energy_sum)) {
            std::printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        }
        if (std::abs(energy_sum) > 1e-8) {
            std::printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
        }
        if (!std::isfinite(flux_sum)) {
            std::printf("  ERROR: Flux sum is not finite\n");
            valid = 0;
        }
        if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
            std::printf("  ERROR: Energy extrema are not finite\n");
            valid = 0;
        }
        if (valid) std::printf("  Validation: PASSED\n");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid != 0;
}

uint64_t computeDistributedHash(const World& world, int rank) {
    uint64_t local_hash = 0;
    const uint64_t global_offset = static_cast<uint64_t>(world.first_row) * world.grid_size;
    for (size_t i = 0; i < world.elements_static.size(); ++i) {
        uint64_t energy_bits, flux_bits;
        std::memcpy(&energy_bits, &world.energy[i], sizeof(energy_bits));
        std::memcpy(&flux_bits, &world.total_flux[i], sizeof(flux_bits));
        const uint64_t global_i = global_offset + i;
        local_hash ^= (energy_bits + global_i) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (flux_bits + global_i) * 0xbf58476d1ce4e5b9ULL;
    }
    uint64_t global_hash = 0;
    MPI_Reduce(&local_hash, &global_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    return rank == 0 ? global_hash : 0;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Grid size (NxN elements) (default: 512)\n");
    std::printf("  -i <num>     Number of simulation iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    int parse_status = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) n_iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parse_status = 2;
        else parse_status = 1;
    }
    if (parse_status != 0 || n <= 0 || n_iters < 0) {
        if (rank == 0) {
            if (parse_status == 1) std::printf("Unknown or incomplete option\n");
            else if (parse_status == 0) std::printf("Grid size must be positive and iterations nonnegative\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_status == 2 ? 0 : 1;
    }

    const uint64_t n_elements = static_cast<uint64_t>(n) * n;
    if (rank == 0) {
        std::printf("Unstructured Mesh Energy Transfer Benchmark\n");
        std::printf("============================================\n");
        std::printf("Grid size: %d x %d = %llu elements\n", n, n,
                    static_cast<unsigned long long>(n_elements));
        std::printf("Iterations: %d\n", n_iters);
        std::printf("MPI ranks: %d\n", size);
        std::printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        std::printf("Building unstructured mesh...\n");
    }

    World world;
    buildSquare2D(world, n, rank, size);
    const uint64_t static_mem = n_elements * sizeof(ElementStatic);
    const uint64_t dynamic_mem = n_elements * sizeof(val_t) * 4;
    if (rank == 0) {
        std::printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                    (static_mem + dynamic_mem) / (1024.0 * 1024.0),
                    static_mem / (1024.0 * 1024.0),
                    dynamic_mem / (1024.0 * 1024.0));
        std::printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const uint64_t result_hash = computeDistributedHash(world, rank);
    if (rank == 0) {
        const double duration_ms = seconds * 1000.0;
        const int measured_iters = std::max(n_iters - 1, 1);
        const double time_per_iter = duration_ms / measured_iters;
        const double giga_elements_per_second = seconds > 0.0
            ? (static_cast<double>(measured_iters) * n_elements) / seconds / 1e9 : 0.0;
        std::printf("Computation time: %.3f ms\n", duration_ms);
        std::printf("Performance:\n");
        std::printf("  Time per iteration: %.4f ms\n", time_per_iter);
        std::printf("  Elements/sec: %.4f GigaElements/s\n", giga_elements_per_second);
        std::printf("  Performance: %.4f GFLOPS\n", giga_elements_per_second * 22.0);
        std::printf("  Result hash: %016llX\n\n",
                    static_cast<unsigned long long>(result_hash));
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> global_energy;
        const int local_count = static_cast<int>(world.elements_static.size());
        if (rank == 0) {
            counts.resize(size);
            displacements.resize(size);
            for (int r = 0; r < size; ++r) {
                int first, rows;
                rowPartition(n, size, r, first, rows);
                counts[r] = rows * n;
                displacements[r] = first * n;
            }
            global_energy.resize(static_cast<size_t>(n_elements));
        }
        MPI_Gatherv(world.energy.data(), local_count, MPI_DOUBLE,
                    rank == 0 ? global_energy.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results(global_energy, "ElementEnergy");
    }

    const bool valid = !validate || validateResults(world, rank);
    MPI_Finalize();
    return valid ? 0 : 1;
}
