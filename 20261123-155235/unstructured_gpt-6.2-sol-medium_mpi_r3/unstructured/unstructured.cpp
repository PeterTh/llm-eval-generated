#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using val_t = double;

// The generated mesh has up to four neighbors, one transfer coefficient and
// unit connection weights. Missing connections point back to the cell itself.
struct ElementStatic {
    std::array<uint32_t, 4> neighbors;
    val_t external_flow;
};

struct Tile {
    int x;
    int y;
    int rows;
    int cols;
};

struct World {
    std::vector<ElementStatic> elements;
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> flux;
    int current = 0;
};

using ProcessGrid = std::array<int, 2>;

ProcessGrid chooseGrid(int side, int ranks) {
    ProcessGrid dims{1, 1};
    // Use as many nonempty tiles as possible, preferring a square process
    // grid and contiguous row exchanges when only a strip fits.
    for (int cols = 1; cols <= side && cols <= ranks / cols; ++cols) {
        const int rows = std::min(side, ranks / cols);
        if (rows * cols >= dims[0] * dims[1]) {
            dims = {rows, cols};
        }
    }
    return dims;
}

Tile tileForRank(MPI_Comm mesh, const ProcessGrid& dims, int side, int rank) {
    int coords[2];
    MPI_Cart_coords(mesh, rank, 2, coords);
    const auto offset = [side](int coord, int parts) {
        return coord * (side / parts) + std::min(coord, side % parts);
    };
    const int x = offset(coords[0], dims[0]);
    const int y = offset(coords[1], dims[1]);
    return {x, y, offset(coords[0] + 1, dims[0]) - x,
                  offset(coords[1] + 1, dims[1]) - y};
}

void buildSquare2D(World& world, const Tile& tile, int side) {
    const size_t local_size = static_cast<size_t>(tile.rows) * tile.cols;
    const int stride = tile.cols + 2;
    const size_t padded_size = static_cast<size_t>(tile.rows + 2) * stride;
    world.elements.resize(local_size);
    world.flux.resize(local_size);
    for (auto& energy : world.energy) {
        energy.resize(padded_size);
    }

    for (int i = 0; i < tile.rows; ++i) {
        const int x = tile.x + i;
        for (int j = 0; j < tile.cols; ++j) {
            const int y = tile.y + j;
            const auto own = static_cast<uint32_t>(static_cast<size_t>(i + 1) * stride + j + 1);
            auto& elem = world.elements[static_cast<size_t>(i) * tile.cols + j];
            elem.neighbors = {x + 1 < side ? own + stride : own,
                              x > 0        ? own - stride : own,
                              y + 1 < side ? own + 1 : own,
                              y > 0        ? own - 1 : own};
            if ((x == 0 || x == side - 1) && (y == 0 || y == side - 1)) {
                elem.external_flow = x == y ? 0.5 : -0.5;
            }
        }
    }
}

void updateTile(World& world, int cols, int row_begin, int row_end, int col_begin, int col_end) {
    const auto* energy = world.energy[world.current].data();
    auto* next = world.energy[1 - world.current].data();
    const int stride = cols + 2;
    for (int i = row_begin; i < row_end; ++i) {
        const size_t row = static_cast<size_t>(i) * cols;
        const size_t padded_row = static_cast<size_t>(i + 1) * stride + 1;
        for (int j = col_begin; j < col_end; ++j) {
            const auto& elem = world.elements[row + j];
            const val_t current = energy[padded_row + j];
            val_t total_flux = elem.external_flow;
            for (uint32_t neighbor : elem.neighbors) {
                total_flux += (energy[neighbor] - current) * 0.8 * 0.25;
            }
            next[padded_row + j] = current + total_flux;
            world.flux[row + j] += std::abs(total_flux);
        }
    }
}

struct HaloExchange {
    enum Direction { North, South, West, East, Count };
    std::array<int, Count> neighbors;
    std::array<std::array<MPI_Request, 2 * Count>, 2> requests;
    int count = 0;
    MPI_Datatype column = MPI_DATATYPE_NULL;
};

void prepareHalo(HaloExchange& halo, World& world, const Tile& tile, MPI_Comm mesh) {
    using Dir = HaloExchange;
    MPI_Cart_shift(mesh, 0, 1, &halo.neighbors[Dir::North], &halo.neighbors[Dir::South]);
    MPI_Cart_shift(mesh, 1, 1, &halo.neighbors[Dir::West], &halo.neighbors[Dir::East]);

    const size_t stride = tile.cols + 2;
    if (halo.neighbors[Dir::West] != MPI_PROC_NULL || halo.neighbors[Dir::East] != MPI_PROC_NULL) {
        MPI_Type_vector(tile.rows, 1, static_cast<int>(stride), MPI_DOUBLE, &halo.column);
        MPI_Type_commit(&halo.column);
    }
    const size_t last_row = tile.rows * stride;
    const std::array<size_t, Dir::Count> outgoing{stride + 1, last_row + 1, stride + 1, stride + tile.cols};
    const std::array<size_t, Dir::Count> incoming{1, last_row + stride + 1, stride, stride + tile.cols + 1};
    constexpr std::array<int, Dir::Count> opposite{Dir::South, Dir::North, Dir::East, Dir::West};
    for (int dir = 0; dir < Dir::Count; ++dir) {
        const int peer = halo.neighbors[dir];
        if (peer == MPI_PROC_NULL) {
            continue;
        }
        const bool row = dir < Dir::West;
        const int count = row ? tile.cols : 1;
        const MPI_Datatype type = row ? MPI_DOUBLE : halo.column;
        for (size_t buffer = 0; buffer < world.energy.size(); ++buffer) {
            auto* energy = world.energy[buffer].data();
            auto& requests = halo.requests[buffer];
            MPI_Recv_init(energy + incoming[dir], count, type, peer, opposite[dir], mesh, &requests[halo.count]);
            MPI_Send_init(energy + outgoing[dir], count, type, peer, dir, mesh, &requests[halo.count + 1]);
        }
        halo.count += 2;
    }
}

void freeHalo(HaloExchange& halo) {
    for (auto& requests : halo.requests) {
        for (int i = 0; i < halo.count; ++i) {
            MPI_Request_free(&requests[i]);
        }
    }
    if (halo.column != MPI_DATATYPE_NULL) {
        MPI_Type_free(&halo.column);
    }
}

void runSimulation(World& world, const Tile& tile, HaloExchange& halo, int n_iters) {
    using Dir = HaloExchange;
    const int top = halo.neighbors[Dir::North] != MPI_PROC_NULL;
    const int bottom = tile.rows - (halo.neighbors[Dir::South] != MPI_PROC_NULL);
    const int left = halo.neighbors[Dir::West] != MPI_PROC_NULL;
    const int right = tile.cols - (halo.neighbors[Dir::East] != MPI_PROC_NULL);
    const bool has_interior = top < bottom && left < right;

    for (int iter = 0; iter < n_iters; ++iter) {
        // Both owned and ghost energies start at zero.
        if (iter == 0 || halo.count == 0) {
            updateTile(world, tile.cols, 0, tile.rows, 0, tile.cols);
        } else {
            auto* requests = halo.requests[world.current].data();
            MPI_Startall(halo.count, requests);
            if (has_interior) {
                updateTile(world, tile.cols, top, bottom, left, right);
            }
            MPI_Waitall(halo.count, requests, MPI_STATUSES_IGNORE);

            if (has_interior) {
                updateTile(world, tile.cols, 0, top, 0, tile.cols);
                updateTile(world, tile.cols, bottom, tile.rows, 0, tile.cols);
                if (left > 0) {
                    updateTile(world, tile.cols, top, bottom, 0, left);
                }
                if (right < tile.cols) {
                    updateTile(world, tile.cols, top, bottom, right, tile.cols);
                }
            } else {
                updateTile(world, tile.cols, 0, tile.rows, 0, tile.cols);
            }
        }
        world.current = 1 - world.current;
    }
}

// Iterate through owned cells in global row-major order within each tile.
template <typename Visit>
void visitElements(const World& world, const Tile& tile, Visit visit) {
    const size_t stride = tile.cols + 2;
    const auto& energy = world.energy[world.current];
    for (int i = 0; i < tile.rows; ++i) {
        for (int j = 0; j < tile.cols; ++j) {
            visit(i, j, energy[(i + 1) * stride + j + 1],
                  world.flux[static_cast<size_t>(i) * tile.cols + j]);
        }
    }
}

uint64_t computeHash(const World& world, const Tile& tile, int side, MPI_Comm mesh) {
    uint64_t local_hash = 0;
    visitElements(world, tile, [&](int i, int j, val_t energy, val_t flux) {
        const uint64_t index = static_cast<uint64_t>(tile.x + i) * side + tile.y + j;
        local_hash ^= (std::bit_cast<uint64_t>(energy) + index) * 0x9e3779b97f4a7c15ULL;
        local_hash ^= (std::bit_cast<uint64_t>(flux) + index) * 0xbf58476d1ce4e5b9ULL;
    });
    uint64_t hash = 0;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, mesh);
    return hash;
}

void printGlobalResults(const World& world, const Tile& tile, int side, const ProcessGrid& dims,
                        int rank, MPI_Comm mesh) {
    constexpr int result_tag = HaloExchange::Count;
    const auto& energy = world.energy[world.current];
    const int stride = tile.cols + 2;
    if (rank != 0) {
        MPI_Datatype owned;
        MPI_Type_vector(tile.rows, tile.cols, stride, MPI_DOUBLE, &owned);
        MPI_Type_commit(&owned);
        MPI_Send(energy.data() + stride + 1, 1, owned, 0, result_tag, mesh);
        MPI_Type_free(&owned);
        return;
    }

    // Place tiles directly in the global order required by print_results.
    std::vector<val_t> global_energy(static_cast<size_t>(side) * side);
    for (int i = 0; i < tile.rows; ++i) {
        std::copy_n(energy.data() + static_cast<size_t>(i + 1) * stride + 1, tile.cols,
                    global_energy.data() + static_cast<size_t>(i) * side);
    }
    for (int peer = 1; peer < dims[0] * dims[1]; ++peer) {
        const Tile source = tileForRank(mesh, dims, side, peer);
        MPI_Datatype destination;
        MPI_Type_vector(source.rows, source.cols, side, MPI_DOUBLE, &destination);
        MPI_Type_commit(&destination);
        MPI_Recv(global_energy.data() + static_cast<size_t>(source.x) * side + source.y,
                 1, destination, peer, result_tag, mesh, MPI_STATUS_IGNORE);
        MPI_Type_free(&destination);
    }
    print_results(global_energy, "ElementEnergy");
}

bool validateResults(const World& world, const Tile& tile, int rank, MPI_Comm mesh) {
    val_t local_sums[2]{};
    val_t local_min = std::numeric_limits<val_t>::max();
    val_t local_max = std::numeric_limits<val_t>::lowest();
    visitElements(world, tile, [&](int, int, val_t energy, val_t flux) {
        local_sums[0] += energy;
        local_sums[1] += flux;
        local_min = std::min(local_min, energy);
        local_max = std::max(local_max, energy);
    });

    val_t sums[2]{};
    val_t energy_min = 0.0;
    val_t energy_max = 0.0;
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, mesh);
    MPI_Reduce(&local_min, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, mesh);
    MPI_Reduce(&local_max, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);
    if (rank != 0) {
        return true;
    }

    const val_t energy_sum = sums[0];
    const val_t flux_sum = sums[1];
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int side = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long value = std::strtol(argv[++i], nullptr, 10);
            // The original benchmark represents the global cell count as int.
            if (value <= 0 || value > std::numeric_limits<int>::max() / value) {
                if (rank == 0) {
                    printf("Invalid grid size: %s\n", argv[i]);
                }
                return 1;
            }
            side = static_cast<int>(value);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            n_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    const ProcessGrid dims = chooseGrid(side, ranks);
    constexpr int periods[2]{0, 0};
    MPI_Comm mesh;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods, 0, &mesh);
    if (mesh == MPI_COMM_NULL) {
        return 0;
    }

    const int n_elems = side * side;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n", side, side, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    const Tile tile = tileForRank(mesh, dims, side, rank);
    World world;
    buildSquare2D(world, tile, side);
    HaloExchange halo;
    prepareHalo(halo, world, tile, mesh);
    if (rank == 0) {
        const size_t static_mem = static_cast<size_t>(n_elems) * sizeof(ElementStatic);
        const size_t padded_size = static_cast<size_t>(side + 2 * dims[0]) * (side + 2 * dims[1]);
        const size_t dynamic_mem = (static_cast<size_t>(n_elems) + 2 * padded_size) * sizeof(val_t);
        constexpr double bytes_per_mb = 1024.0 * 1024.0;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / bytes_per_mb,
               static_mem / bytes_per_mb, dynamic_mem / bytes_per_mb);
        printf("Running simulation...\n");
    }

    MPI_Barrier(mesh);
    const double start = MPI_Wtime();
    runSimulation(world, tile, halo, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);
    freeHalo(halo);

    const uint64_t hash = computeHash(world, tile, side, mesh);
    if (rank == 0) {
        const double duration_ms = seconds * 1000.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));

        const double time_per_iter = duration_ms / std::max(n_iters, 1);
        const double giga_elems_per_sec = n_iters > 0 && seconds > 0.0
            ? static_cast<double>(n_iters) * n_elems / seconds / 1e9 : 0.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }

    if (printResults) {
        printGlobalResults(world, tile, side, dims, rank, mesh);
    }
    const bool valid = !validate || validateResults(world, tile, rank, mesh);
    MPI_Comm_free(&mesh);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
