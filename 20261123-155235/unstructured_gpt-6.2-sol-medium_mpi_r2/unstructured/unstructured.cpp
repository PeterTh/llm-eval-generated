#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using idx_t = uint32_t;
using val_t = double;

struct Tile {
    int x_begin = 0;
    int x_end = 0;
    int y_begin = 0;
    int y_end = 0;

    int rows() const { return x_end - x_begin; }
    int cols() const { return y_end - y_begin; }
    size_t size() const { return static_cast<size_t>(rows()) * cols(); }
};

// Use a compact two-dimensional partition even when the rank count is prime.
// Adjacent stripes can have different numbers of tiles.
struct Partition {
    int side;
    int active_ranks;
    int stripes;

    Partition(int side, int ranks)
        : side(side),
          active_ranks(static_cast<int>(std::min<int64_t>(ranks, int64_t{side} * side))),
          stripes(static_cast<int>(std::sqrt(active_ranks))) {
        // Prefer aligned faces when a sufficiently balanced factorization exists.
        int divisor = stripes;
        while (divisor > 1 && active_ranks % divisor != 0 &&
               int64_t{divisor} * divisor * 2 >= active_ranks) {
            --divisor;
        }
        if (active_ranks % divisor == 0 && int64_t{divisor} * divisor * 2 >= active_ranks) {
            stripes = divisor;
        }
        stripes = std::max(stripes, 1 + (active_ranks - 1) / side);
    }

    static int split(int length, int part, int parts) {
        return static_cast<int>(int64_t{length} * part / parts);
    }

    int firstRank(int stripe) const {
        return stripe * (active_ranks / stripes) + std::min(stripe, active_ranks % stripes);
    }

    int columns(int stripe) const { return firstRank(stripe + 1) - firstRank(stripe); }

    int stripeOf(int rank) const {
        const int base = active_ranks / stripes;
        const int extra = active_ranks % stripes;
        const int wide_ranks = extra * (base + 1LL);
        return rank < wide_ranks ? rank / (base + 1) : (rank - extra) / base;
    }

    int rowStart(int stripe) const {
        return stripe + split(side - stripes, firstRank(stripe), active_ranks);
    }

    Tile tile(int stripe, int column) const {
        const int cols = columns(stripe);
        return {rowStart(stripe), rowStart(stripe + 1),
                split(side, column, cols), split(side, column + 1, cols)};
    }
};

struct Element {
    idx_t neighbors[4];  // +x, -x, +y, -y, omitting physical boundaries
    uint8_t degree = 0;
    int8_t source = 0;    // External flow in units of 0.5
};

struct Halo {
    int peer;
    int count;
    idx_t send_offset;
    idx_t receive_offset;
    bool column;
};

struct World {
    Tile tile;
    std::vector<Element> elements;
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> total_flux;
    std::vector<Halo> halos;
};

void buildSquare2D(World& world, const Partition& partition, int stripe, int column) {
    const Tile& tile = world.tile;
    const int side = partition.side;
    const int rows = tile.rows();
    const int cols = tile.cols();
    const int rank = partition.firstRank(stripe) + column;

    idx_t next_ghost = tile.size();
    std::array<idx_t, 4> ghost{};
    auto addRowHalo = [&](int direction, int peer_stripe, idx_t send_offset) {
        ghost[direction] = next_ghost;
        next_ghost += cols;
        const int peer_cols = partition.columns(peer_stripe);
        // Find the peer tile containing the first column of this face.
        int peer_col = (int64_t{tile.y_begin + 1} * peer_cols - 1) / side;
        for (; peer_col < peer_cols; ++peer_col) {
            const int start = std::max(tile.y_begin, Partition::split(side, peer_col, peer_cols));
            const int end = std::min(tile.y_end, Partition::split(side, peer_col + 1, peer_cols));
            if (start >= end) {
                break;
            }
            const int offset = start - tile.y_begin;
            world.halos.push_back({partition.firstRank(peer_stripe) + peer_col, end - start,
                                   send_offset + offset, ghost[direction] + offset, false});
        }
    };
    auto addColumnHalo = [&](int direction, int peer, idx_t send_offset) {
        ghost[direction] = next_ghost;
        next_ghost += rows;
        world.halos.push_back({peer, rows, send_offset, ghost[direction], true});
    };

    if (tile.x_end < side) {
        addRowHalo(0, stripe + 1, tile.size() - cols);
    }
    if (tile.x_begin > 0) {
        addRowHalo(1, stripe - 1, 0);
    }
    if (tile.y_end < side) {
        addColumnHalo(2, rank + 1, cols - 1);
    }
    if (tile.y_begin > 0) {
        addColumnHalo(3, rank - 1, 0);
    }

    world.elements.resize(tile.size());
    world.total_flux.resize(tile.size());
    for (auto& buffer : world.energy) {
        buffer.resize(next_ghost);
    }

    for (int x = 0; x < rows; ++x) {
        const int gx = tile.x_begin + x;
        for (int y = 0; y < cols; ++y) {
            const int gy = tile.y_begin + y;
            const idx_t i = static_cast<idx_t>(x) * cols + y;
            Element& elem = world.elements[i];
            if (gx + 1 < side) {
                elem.neighbors[elem.degree++] = x + 1 < rows ? i + cols : ghost[0] + y;
            }
            if (gx > 0) {
                elem.neighbors[elem.degree++] = x > 0 ? i - cols : ghost[1] + y;
            }
            if (gy + 1 < side) {
                elem.neighbors[elem.degree++] = y + 1 < cols ? i + 1 : ghost[2] + x;
            }
            if (gy > 0) {
                elem.neighbors[elem.degree++] = y > 0 ? i - 1 : ghost[3] + x;
            }
            if ((gx == 0 || gx == side - 1) && (gy == 0 || gy == side - 1)) {
                elem.source = gx == gy ? 1 : -1;
            }
        }
    }
}

// Interior elements have four neighbors and zero external flow. Keep the
// connection order and arithmetic of the original element update.
template <bool interior>
inline void updateElement(World& world, idx_t i, const val_t* current, val_t* next) {
    const Element& elem = world.elements[i];
    const val_t own_energy = current[i];
    val_t flux = interior ? 0.0 : elem.source * 0.5;
    const int degree = interior ? 4 : elem.degree;
    for (int j = 0; j < degree; ++j) {
        flux += (current[elem.neighbors[j]] - own_energy) * 0.8 * 0.25;
    }
    next[i] = own_energy + flux;
    world.total_flux[i] += std::abs(flux);
}

double runSimulation(World& world, int n_iters) {
    // A column send reads strided owned cells directly. Receives, including
    // segments from differently tiled stripes, write into contiguous ghosts.
    MPI_Datatype column_type = MPI_DATATYPE_NULL;
    if (std::any_of(world.halos.begin(), world.halos.end(), [](const Halo& h) { return h.column; })) {
        MPI_Type_vector(world.tile.rows(), 1, world.tile.cols(), MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);
    }

    // Persistent requests are bound to one energy buffer per iteration parity.
    std::array<std::vector<MPI_Request>, 2> exchanges;
    for (int buffer = 0; buffer < 2; ++buffer) {
        val_t* energy = world.energy[buffer].data();
        auto& requests = exchanges[buffer];
        requests.resize(2 * world.halos.size());
        size_t request = 0;
        for (const Halo& halo : world.halos) {
            MPI_Recv_init(energy + halo.receive_offset, halo.count, MPI_DOUBLE, halo.peer,
                          0, MPI_COMM_WORLD, &requests[request++]);
        }
        for (const Halo& halo : world.halos) {
            MPI_Send_init(energy + halo.send_offset, halo.column ? 1 : halo.count,
                          halo.column ? column_type : MPI_DOUBLE, halo.peer,
                          0, MPI_COMM_WORLD, &requests[request++]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    const int rows = world.tile.rows();
    const int cols = world.tile.cols();
    for (int iter = 0; rows > 0 && iter < n_iters; ++iter) {
        const val_t* current = world.energy[iter % 2].data();
        val_t* next = world.energy[1 - iter % 2].data();
        auto& requests = exchanges[iter % 2];
        // Initial energies (including ghosts) are already zero everywhere.
        const bool exchange = iter > 0 && !requests.empty();
        if (exchange) {
            MPI_Startall(static_cast<int>(requests.size()), requests.data());
        }

        for (int x = 1; x < rows - 1; ++x) {
            const idx_t row = static_cast<idx_t>(x) * cols;
            for (int y = 1; y < cols - 1; ++y) {
                updateElement<true>(world, row + y, current, next);
            }
        }

        if (exchange) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        }

        for (int y = 0; y < cols; ++y) {
            updateElement<false>(world, y, current, next);
        }
        for (int x = 1; x < rows - 1; ++x) {
            const idx_t row = static_cast<idx_t>(x) * cols;
            updateElement<false>(world, row, current, next);
            if (cols > 1) {
                updateElement<false>(world, row + cols - 1, current, next);
            }
        }
        if (rows > 1) {
            const idx_t row = world.tile.size() - cols;
            for (int y = 0; y < cols; ++y) {
                updateElement<false>(world, row + y, current, next);
            }
        }
    }
    const double elapsed = MPI_Wtime() - start;

    for (auto& requests : exchanges) {
        for (MPI_Request& request : requests) {
            MPI_Request_free(&request);
        }
    }
    if (column_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&column_type);
    }
    if (n_iters > 0 && n_iters % 2 != 0) {
        world.energy[0].swap(world.energy[1]);
    }

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_elapsed;
}

uint64_t computeHash(const World& world, int side) {
    uint64_t result = 0;
    const Tile& tile = world.tile;
    size_t local = 0;
    for (int x = tile.x_begin; x < tile.x_end; ++x) {
        uint64_t global = static_cast<uint64_t>(x) * side + tile.y_begin;
        for (int y = tile.y_begin; y < tile.y_end; ++y, ++global, ++local) {
            result ^= (std::bit_cast<uint64_t>(world.energy[0][local]) + global) *
                      0x9e3779b97f4a7c15ULL;
            result ^= (std::bit_cast<uint64_t>(world.total_flux[local]) + global) *
                      0xbf58476d1ce4e5b9ULL;
        }
    }
    return result;
}

// The external report's Kahan sum and FNV hash depend on global row-major order.
void printGlobalResults(const World& world, const Partition& partition, int rank) {
    const auto& energy = world.energy[0];
    const Tile& local = world.tile;
    if (rank != 0) {
        if (rank < partition.active_ranks) {
            MPI_Send(energy.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                     0, 1, MPI_COMM_WORLD);
        }
        return;
    }

    const int side = partition.side;
    if (partition.active_ranks == 1) {
        print_results(energy, "ElementEnergy");
        return;
    }
    std::vector<val_t> global(static_cast<size_t>(side) * side);
    std::vector<MPI_Request> requests(partition.active_ranks - 1);
    for (int stripe = 0; stripe < partition.stripes; ++stripe) {
        for (int column = 0; column < partition.columns(stripe); ++column) {
            const int peer = partition.firstRank(stripe) + column;
            if (peer == 0) {
                continue;
            }
            const Tile tile = partition.tile(stripe, column);
            MPI_Datatype placement;
            MPI_Type_vector(tile.rows(), tile.cols(), side, MPI_DOUBLE, &placement);
            MPI_Type_commit(&placement);
            MPI_Irecv(global.data() + static_cast<size_t>(tile.x_begin) * side + tile.y_begin,
                      1, placement, peer, 1, MPI_COMM_WORLD, &requests[peer - 1]);
            MPI_Type_free(&placement);
        }
    }
    for (int x = 0; x < local.rows(); ++x) {
        std::copy_n(energy.data() + static_cast<size_t>(x) * local.cols(), local.cols(),
                    global.data() + static_cast<size_t>(x) * side);
    }
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    print_results(global, "ElementEnergy");
}

bool validateResults(const World& world, int rank) {
    val_t local_sums[2]{};
    val_t local_extrema[2]{std::numeric_limits<val_t>::max(), std::numeric_limits<val_t>::max()};
    for (size_t i = 0; i < world.tile.size(); ++i) {
        const val_t energy = world.energy[0][i];
        local_sums[0] += energy;
        local_sums[1] += world.total_flux[i];
        local_extrema[0] = std::min(energy, local_extrema[0]);
        local_extrema[1] = std::min(-energy, local_extrema[1]);
    }

    val_t sums[2]{};
    val_t extrema[2]{};
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_extrema, extrema, 2, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", sums[0]);
        printf("  Flux sum: %.2f\n", sums[1]);
        printf("  Energy range: [%.6f, %.6f]\n", extrema[0], -extrema[1]);

        if (!std::isfinite(sums[0])) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else {
            if (std::abs(sums[0]) > 1e-8) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(sums[1])) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = 0;
            } else if (!std::isfinite(extrema[0]) || !std::isfinite(extrema[1])) {
                printf("  ERROR: Energy extrema are not finite\n");
                valid = 0;
            } else {
                printf("  Validation: PASSED\n");
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid;
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
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_elems_root = atoi(argv[++i]);
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

    if (n_elems_root <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Grid size must be positive\n");
        }
        return 1;
    }

    const size_t n_elems = static_cast<size_t>(n_elems_root) * n_elems_root;
    const Partition partition(n_elems_root, ranks);
    World world;
    int stripe = 0;
    int column = 0;
    if (rank < partition.active_ranks) {
        stripe = partition.stripeOf(rank);
        column = rank - partition.firstRank(stripe);
        world.tile = partition.tile(stripe, column);
    }
    const Tile& tile = world.tile;
    const int row_faces = (tile.x_begin > 0) + (tile.x_end < n_elems_root);
    const int column_faces = (tile.y_begin > 0) + (tile.y_end < n_elems_root);
    const uint64_t ghost_count = static_cast<uint64_t>(tile.cols()) * row_faces +
                                 static_cast<uint64_t>(tile.rows()) * column_faces;
    const int tile_too_large = tile.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                               tile.size() + ghost_count > std::numeric_limits<idx_t>::max();
    int any_tile_too_large;
    MPI_Allreduce(&tile_too_large, &any_tile_too_large, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_tile_too_large) {
        if (rank == 0) {
            fprintf(stderr, "Grid tiles exceed the supported size per MPI rank\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    if (rank < partition.active_ranks) {
        buildSquare2D(world, partition, stripe, column);
    }

    const uint64_t local_dynamic_mem =
        (world.energy[0].size() + world.energy[1].size() + world.total_flux.size()) * sizeof(val_t);
    uint64_t dynamic_mem = 0;
    MPI_Reduce(&local_dynamic_mem, &dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const size_t static_mem = n_elems * sizeof(Element);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (static_mem + dynamic_mem) / (1024.0 * 1024.0),
               static_mem / (1024.0 * 1024.0), dynamic_mem / (1024.0 * 1024.0));
        printf("Running simulation...\n");
    }

    const double elapsed = runSimulation(world, n_iters);
    const uint64_t local_hash = computeHash(world, n_elems_root);
    uint64_t result_hash = 0;
    MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const int n_measured_iters = std::max(n_iters, 2) - 1;
        const double giga_elems_per_sec = n_measured_iters * static_cast<double>(n_elems) /
                                          elapsed / 1e9;
        printf("Computation time: %" PRId64 " ms\n", static_cast<int64_t>(elapsed * 1000.0));
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", elapsed * 1000.0 / n_measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
    }

    if (printResults) {
        printGlobalResults(world, partition, rank);
    }
    return validate && !validateResults(world, rank) ? 1 : 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int status = runBenchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
