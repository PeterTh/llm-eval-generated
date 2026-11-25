#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = std::size_t;
using val_t = double;

struct Band {
    int first_rank;
    int rank_count;
    int first_row;
    int end_row;
};

struct Tile {
    int first_row = 0;
    int first_col = 0;
    int rows = 0;
    int cols = 0;

    idx_t size() const {
        return static_cast<idx_t>(rows) * cols;
    }
};

// Split into contiguous pieces, giving one extra item to each of the first pieces.
int splitStart(int size, int parts, int part) {
    return part * (size / parts) + std::min(part, size % parts);
}

int splitOwner(int size, int parts, int index) {
    const int extra = size % parts;
    const int cut = splitStart(size, parts, extra);
    const int width = size / parts;
    return index < cut ? index / (width + 1) : extra + (index - cut) / width;
}

struct Partition {
    int grid_size;
    int active_ranks;
    int bands;

    Partition(int grid_size, int ranks)
        : grid_size(grid_size),
          active_ranks(static_cast<int>(
              std::min<idx_t>(ranks, static_cast<idx_t>(grid_size) * grid_size))),
          bands(std::max(static_cast<int>(std::sqrt(active_ranks)),
                         1 + (active_ranks - 1) / grid_size)) {}

    int firstRow(int band, int first_rank) const {
        // Reserve one row for each band and apportion the rest by its rank count.
        return band + static_cast<idx_t>(grid_size - bands) * first_rank / active_ranks;
    }

    Band band(int index) const {
        const int first = splitStart(active_ranks, bands, index);
        const int end = splitStart(active_ranks, bands, index + 1);
        return {first, end - first, firstRow(index, first), firstRow(index + 1, end)};
    }

    Tile tile(int rank) const {
        const Band owner = band(splitOwner(active_ranks, bands, rank));
        const int column = rank - owner.first_rank;
        const int first = splitStart(grid_size, owner.rank_count, column);
        const int end = splitStart(grid_size, owner.rank_count, column + 1);
        return {owner.first_row, first, owner.end_row - owner.first_row, end - first};
    }
};

struct Material {
    val_t transfer_coeff;
    val_t external_flow;
};

constexpr Material materials[] = {{0.8, 0.0}, {0.8, 0.5}, {0.8, -0.5}};
constexpr int DEFAULT_MAT_ID = 0;
constexpr int INFLOW_MAT_ID = 1;
constexpr int OUTFLOW_MAT_ID = 2;
constexpr int MAX_CONNECTIONS = 4;
// All connections in the generated mesh have the same weight.
constexpr val_t CONNECTION_FLUX = 1.0;

struct ElementStatic {
    idx_t connected_idx[MAX_CONNECTIONS];
    uint8_t material_idx;
    uint8_t num_connections;
};

struct World {
    Tile tile;
    std::vector<ElementStatic> elements;
    // Both buffers have the owned elements first, followed by the receive halos.
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> total_flux;
    std::array<std::vector<MPI_Request>, 2> halo_requests;
    int current_buffer = 0;
};

void buildSquare2D(World& world, const Partition& partition, int rank) {
    if (rank >= partition.active_ranks) {
        return;
    }

    world.tile = partition.tile(rank);
    const auto [first_row, first_col, rows, cols] = world.tile;
    const int grid_size = partition.grid_size;
    const int end_row = first_row + rows;
    const int end_col = first_col + cols;
    const idx_t owned = world.tile.size();

    idx_t buffer_size = owned;
    const idx_t down = buffer_size;
    buffer_size += end_row < grid_size ? cols : 0;
    const idx_t up = buffer_size;
    buffer_size += first_row > 0 ? cols : 0;
    const idx_t right = buffer_size;
    buffer_size += end_col < grid_size ? rows : 0;
    const idx_t left = buffer_size;
    buffer_size += first_col > 0 ? rows : 0;

    world.elements.resize(owned);
    world.total_flux.resize(owned);
    for (auto& buffer : world.energy) {
        buffer.resize(buffer_size);
    }

    // Retain the serial connection order for floating-point accumulation.
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const idx_t index = static_cast<idx_t>(row) * cols + col;
            ElementStatic& elem = world.elements[index];
            const auto connect = [&](idx_t neighbor) {
                elem.connected_idx[elem.num_connections++] = neighbor;
            };

            const int x = first_row + row;
            const int y = first_col + col;
            if (x + 1 < grid_size) {
                connect(row + 1 < rows ? index + cols : down + col);
            }
            if (x > 0) {
                connect(row > 0 ? index - cols : up + col);
            }
            if (y + 1 < grid_size) {
                connect(col + 1 < cols ? index + 1 : right + row);
            }
            if (y > 0) {
                connect(col > 0 ? index - 1 : left + row);
            }

            const int last = grid_size - 1;
            if ((x == 0 && y == 0) || (x == last && y == last)) {
                elem.material_idx = INFLOW_MAT_ID;
            } else if ((x == 0 && y == last) || (x == last && y == 0)) {
                elem.material_idx = OUTFLOW_MAT_ID;
            } else {
                elem.material_idx = DEFAULT_MAT_ID;
            }
        }
    }

    const auto addHalo = [&](int peer, idx_t send_offset, idx_t recv_offset,
                             int count, int stride) {
        MPI_Datatype send_type = MPI_DOUBLE;
        int send_count = count;
        if (stride > 1) {
            MPI_Type_vector(count, 1, stride, MPI_DOUBLE, &send_type);
            MPI_Type_commit(&send_type);
            send_count = 1;
        }

        for (int buffer = 0; buffer < 2; ++buffer) {
            auto* energy = world.energy[buffer].data();
            auto& requests = world.halo_requests[buffer];
            MPI_Request recv_request, send_request;
            MPI_Recv_init(energy + recv_offset, count, MPI_DOUBLE, peer, 0,
                          MPI_COMM_WORLD, &recv_request);
            MPI_Send_init(energy + send_offset, send_count, send_type, peer, 0,
                          MPI_COMM_WORLD, &send_request);
            requests.push_back(recv_request);
            requests.push_back(send_request);
        }

        if (stride > 1) {
            MPI_Type_free(&send_type);
        }
    };

    // Adjacent bands can have different tile widths. Exchange each shared segment.
    const int band_index = splitOwner(partition.active_ranks, partition.bands, rank);
    const auto addRowHalos = [&](int neighbor_band, idx_t send_offset, idx_t recv_offset) {
        const Band neighbor = partition.band(neighbor_band);
        int col = first_col;
        int peer_col = splitOwner(grid_size, neighbor.rank_count, col);
        while (col < end_col) {
            const int next = std::min(end_col,
                                      splitStart(grid_size, neighbor.rank_count, peer_col + 1));
            const int offset = col - first_col;
            addHalo(neighbor.first_rank + peer_col, send_offset + offset,
                    recv_offset + offset, next - col, 1);
            col = next;
            ++peer_col;
        }
    };

    if (end_row < grid_size) {
        addRowHalos(band_index + 1, owned - cols, down);
    }
    if (first_row > 0) {
        addRowHalos(band_index - 1, 0, up);
    }
    if (end_col < grid_size) {
        addHalo(rank + 1, cols - 1, right, rows, cols);
    }
    if (first_col > 0) {
        addHalo(rank - 1, 0, left, rows, cols);
    }
}

inline void updateElements(const ElementStatic* elements, const val_t* current,
                           val_t* next, val_t* total_flux, idx_t first, idx_t end) {
    for (idx_t i = first; i < end; ++i) {
        const ElementStatic& elem = elements[i];
        const Material& mat = materials[elem.material_idx];
        const val_t energy = current[i];
        val_t flux = mat.external_flow;
        for (int j = 0; j < MAX_CONNECTIONS; ++j) {
            if (j == elem.num_connections) {
                break;
            }
            flux += (current[elem.connected_idx[j]] - energy) *
                    mat.transfer_coeff * CONNECTION_FLUX * 0.25;
        }
        next[i] = energy + flux;
        total_flux[i] += std::abs(flux);
    }
}

void runSimulation(World& world, int grid_size, int n_iters) {
    const auto [first_row, first_col, rows, cols] = world.tile;
    const idx_t owned = world.tile.size();
    if (owned == 0) {
        return;
    }

    const int inner_row = first_row > 0 ? 1 : 0;
    const int inner_col = first_col > 0 ? 1 : 0;
    const int end_inner_row = std::max(inner_row, rows - (first_row + rows < grid_size));
    const int end_inner_col = std::max(inner_col, cols - (first_col + cols < grid_size));

    for (int iter = 0; iter < n_iters; ++iter) {
        const int buffer = world.current_buffer;
        auto& requests = world.halo_requests[buffer];
        const auto update = [&](idx_t begin, idx_t end) {
            updateElements(world.elements.data(), world.energy[buffer].data(),
                           world.energy[1 - buffer].data(), world.total_flux.data(), begin, end);
        };

        if (iter == 0 || requests.empty()) {
            // The initial energies, including all ghost values, are zero.
            update(0, owned);
        } else {
            const int count = static_cast<int>(requests.size());
            MPI_Startall(count, requests.data());
            for (int row = inner_row; row < end_inner_row; ++row) {
                const idx_t start = static_cast<idx_t>(row) * cols;
                update(start + inner_col, start + end_inner_col);
            }

            MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
            update(0, static_cast<idx_t>(inner_row) * cols);
            for (int row = inner_row; row < end_inner_row; ++row) {
                const idx_t start = static_cast<idx_t>(row) * cols;
                update(start, start + inner_col);
                update(start + end_inner_col, start + cols);
            }
            update(static_cast<idx_t>(end_inner_row) * cols, owned);
        }
        world.current_buffer = 1 - buffer;
    }
}

uint64_t computeHash(const World& world, int grid_size) {
    const auto [first_row, first_col, rows, cols] = world.tile;
    const auto& energy = world.energy[world.current_buffer];
    uint64_t hash = 0;
    for (int row = 0; row < rows; ++row) {
        const idx_t local_start = static_cast<idx_t>(row) * cols;
        const uint64_t global_start = static_cast<uint64_t>(first_row + row) * grid_size + first_col;
        for (int col = 0; col < cols; ++col) {
            const idx_t local = local_start + col;
            const uint64_t global = global_start + col;
            hash ^= (std::bit_cast<uint64_t>(energy[local]) + global) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(world.total_flux[local]) + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

bool validateResults(const World& world, int rank) {
    const auto& energy = world.energy[world.current_buffer];
    val_t local_sums[2] = {0.0, 0.0};
    val_t local_extrema[2] = {std::numeric_limits<val_t>::lowest(),
                             std::numeric_limits<val_t>::lowest()};
    for (idx_t i = 0; i < world.tile.size(); ++i) {
        local_sums[0] += energy[i];
        local_sums[1] += world.total_flux[i];
        local_extrema[0] = std::max(energy[i], local_extrema[0]);
        local_extrema[1] = std::max(-energy[i], local_extrema[1]);
    }

    val_t sums[2], extrema[2];
    MPI_Reduce(local_sums, sums, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_extrema, extrema, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (rank == 0) {
        const val_t energy_sum = sums[0];
        const val_t flux_sum = sums[1];
        const val_t energy_max = extrema[0];
        const val_t energy_min = -extrema[1];

        printf("Validation results:\n");
        printf("  Energy sum: %.12f\n", energy_sum);
        printf("  Flux sum: %.2f\n", flux_sum);
        printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

        if (!std::isfinite(energy_sum)) {
            printf("  ERROR: Energy sum is not finite\n");
            valid = 0;
        } else {
            if (std::abs(energy_sum) > 1e-8) {
                printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
            }
            if (!std::isfinite(flux_sum)) {
                printf("  ERROR: Flux sum is not finite\n");
                valid = 0;
            } else if (!std::isfinite(energy_max) || !std::isfinite(energy_min)) {
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

void printDistributedResults(const World& world, const Partition& partition, int rank) {
    if (rank >= partition.active_ranks) {
        return;
    }

    const int grid_size = partition.grid_size;
    const auto& energy = world.energy[world.current_buffer];
    if (rank == 0) {
        // The output helper hashes in global row-major order. Receive tiles
        // directly into that layout so only one global energy array is needed.
        std::vector<val_t> global(static_cast<idx_t>(grid_size) * grid_size);
        std::vector<MPI_Request> requests(partition.active_ranks - 1);
        for (int peer = 1; peer < partition.active_ranks; ++peer) {
            const Tile tile = partition.tile(peer);
            MPI_Datatype layout;
            MPI_Type_vector(tile.rows, tile.cols, grid_size, MPI_DOUBLE, &layout);
            MPI_Type_commit(&layout);
            const idx_t offset = static_cast<idx_t>(tile.first_row) * grid_size + tile.first_col;
            MPI_Irecv(global.data() + offset, 1, layout, peer, 1, MPI_COMM_WORLD, &requests[peer - 1]);
            MPI_Type_free(&layout);
        }
        for (int row = 0; row < world.tile.rows; ++row) {
            std::copy_n(energy.data() + static_cast<idx_t>(row) * world.tile.cols, world.tile.cols,
                        global.data() + static_cast<idx_t>(row) * grid_size);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        print_results(global, "ElementEnergy");
    } else if (world.tile.size() <= static_cast<idx_t>(std::numeric_limits<int>::max())) {
        MPI_Send(energy.data(), static_cast<int>(world.tile.size()), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    } else {
        MPI_Datatype layout;
        MPI_Type_contiguous(world.tile.rows, MPI_DOUBLE, &layout);
        MPI_Type_commit(&layout);
        MPI_Send(energy.data(), world.tile.cols, layout, 0, 1, MPI_COMM_WORLD);
        MPI_Type_free(&layout);
    }
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
    int grid_size = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            grid_size = atoi(argv[++i]);
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

    if (grid_size <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Grid size must be positive\n");
        }
        return 1;
    }

    const idx_t n_elems = static_cast<idx_t>(grid_size) * grid_size;
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", grid_size, grid_size, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }

    World world;
    const Partition partition(grid_size, ranks);
    buildSquare2D(world, partition, rank);

    const uint64_t local_mem[2] = {world.elements.size() * sizeof(ElementStatic),
                                   (world.energy[0].size() + world.energy[1].size() +
                                    world.total_flux.size()) * sizeof(val_t)};
    uint64_t memory[2];
    MPI_Reduce(local_mem, memory, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        constexpr double bytes_per_mb = 1024.0 * 1024.0;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (memory[0] + memory[1]) / bytes_per_mb, memory[0] / bytes_per_mb,
               memory[1] / bytes_per_mb);
        printf("Running simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    runSimulation(world, grid_size, n_iters);
    const double local_seconds = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    for (auto& requests : world.halo_requests) {
        for (auto& request : requests) {
            MPI_Request_free(&request);
        }
    }

    const uint64_t local_hash = computeHash(world, grid_size);
    uint64_t hash;
    MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double duration_ms = seconds * 1000.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));

        const int n_measured_iters = n_iters > 1 ? n_iters - 1 : 1;
        const double time_per_iter = duration_ms / n_measured_iters;
        const double giga_elems_per_sec = seconds > 0.0 ? n_measured_iters * (n_elems / 1e9) / seconds : 0.0;
        const double gflops = giga_elems_per_sec * 22.0;

        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", time_per_iter);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", gflops);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }

    if (printResults) {
        printDistributedResults(world, partition, rank);
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
