#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/hash.hpp"

using val_t = double;

enum Face { North, South, West, East, FaceCount };

struct Span {
    int begin;
    int end;

    int size() const { return end - begin; }
};

Span partition(int side, int parts, int coord) {
    const int base = side / parts;
    const int remainder = side % parts;
    const int begin = coord * base + std::min(coord, remainder);
    return {begin, begin + base + (coord < remainder)};
}

struct Tile {
    Span rows;
    Span cols;

    size_t size() const { return size_t(rows.size()) * cols.size(); }
};

Tile tileForRank(int side, const int dims[2], int rank) {
    return {partition(side, dims[0], rank / dims[1]),
            partition(side, dims[1], rank % dims[1])};
}

struct Cell {
    std::array<size_t, FaceCount> neighbors;
    int degree;
    int source;  // External energy flow in units of 0.5
};

struct World {
    Tile tile;
    std::array<int, FaceCount> peers;
    std::array<size_t, FaceCount> ghosts;
    std::vector<Cell> cells;
    std::array<std::vector<val_t>, 2> energy;
    std::vector<val_t> flux;
    int current = 0;
};

World buildSquare2D(int side, Tile tile, MPI_Comm mesh) {
    World world{};
    world.tile = tile;
    MPI_Cart_shift(mesh, 0, 1, &world.peers[North], &world.peers[South]);
    MPI_Cart_shift(mesh, 1, 1, &world.peers[West], &world.peers[East]);

    const int rows = tile.rows.size();
    const int cols = tile.cols.size();
    const size_t owned = tile.size();
    size_t storage = owned;
    for (int face = 0; face < FaceCount; ++face) {
        world.ghosts[face] = storage;
        if (world.peers[face] != MPI_PROC_NULL) {
            storage += face == North || face == South ? cols : rows;
        }
    }
    world.cells.resize(owned);
    for (auto& buffer : world.energy) {
        buffer.resize(storage);
    }
    world.flux.resize(owned);

    // Store local and halo connections in the original flux accumulation order.
    for (int row = 0; row < rows; ++row) {
        const int x = tile.rows.begin + row;
        for (int col = 0; col < cols; ++col) {
            const int y = tile.cols.begin + col;
            const size_t i = size_t(row) * cols + col;
            Cell& cell = world.cells[i];
            if (x < side - 1) {
                cell.neighbors[cell.degree++] = row + 1 < rows ? i + cols : world.ghosts[South] + col;
            }
            if (x > 0) {
                cell.neighbors[cell.degree++] = row > 0 ? i - cols : world.ghosts[North] + col;
            }
            if (y < side - 1) {
                cell.neighbors[cell.degree++] = col + 1 < cols ? i + 1 : world.ghosts[East] + row;
            }
            if (y > 0) {
                cell.neighbors[cell.degree++] = col > 0 ? i - 1 : world.ghosts[West] + row;
            }
            const bool at_x_edge = x == 0 || x == side - 1;
            const bool at_y_edge = y == 0 || y == side - 1;
            if (at_x_edge && at_y_edge) {
                cell.source = x == y ? 1 : -1;
            }
        }
    }
    return world;
}

struct Exchange {
    std::array<MPI_Request, 2 * FaceCount> requests{};
    int count = 0;
};

Exchange setupExchange(const World& world, val_t* energy,
                       std::array<std::vector<val_t>, 2>& columns, MPI_Comm mesh) {
    const int cols = world.tile.cols.size();
    const int rows = world.tile.rows.size();
    Exchange exchange;

    for (int face = 0; face < FaceCount; ++face) {
        if (world.peers[face] == MPI_PROC_NULL) {
            continue;
        }
        const bool row_face = face == North || face == South;
        const int length = row_face ? cols : rows;
        const val_t* outgoing = row_face ? energy + (face == North ? 0 : world.tile.size() - cols)
                                         : columns[face - West].data();
        MPI_Recv_init(energy + world.ghosts[face], length, MPI_DOUBLE,
                      world.peers[face], face ^ 1, mesh, &exchange.requests[exchange.count++]);
        MPI_Send_init(outgoing, length, MPI_DOUBLE,
                      world.peers[face], face, mesh, &exchange.requests[exchange.count++]);
    }
    return exchange;
}

inline val_t connectionFlux(val_t energy, val_t neighbor) {
    // The generated mesh uses a transfer coefficient of 0.8 and unit connection weights.
    return (neighbor - energy) * 0.8 * 0.25;
}

void updateCells(const World& world, const val_t* old_energy, val_t* new_energy, val_t* flux,
                 Span rows, Span cols) {
    const int stride = world.tile.cols.size();
    if (rows.size() == 0 || cols.size() == 0) {
        return;
    }
    for (int row = rows.begin; row < rows.end; ++row) {
        const size_t end = size_t(row) * stride + cols.end;
        for (size_t i = size_t(row) * stride + cols.begin; i < end; ++i) {
            const Cell& cell = world.cells[i];
            const val_t energy = old_energy[i];
            val_t transfer = cell.source * 0.5;
            if (cell.degree == FaceCount) {
                transfer += connectionFlux(energy, old_energy[cell.neighbors[0]]);
                transfer += connectionFlux(energy, old_energy[cell.neighbors[1]]);
                transfer += connectionFlux(energy, old_energy[cell.neighbors[2]]);
                transfer += connectionFlux(energy, old_energy[cell.neighbors[3]]);
            } else {
                for (int j = 0; j < cell.degree; ++j) {
                    transfer += connectionFlux(energy, old_energy[cell.neighbors[j]]);
                }
            }
            new_energy[i] = energy + transfer;
            flux[i] += std::abs(transfer);
        }
    }
}

struct HaloExchange {
    std::array<Exchange, 2> buffers;
    std::array<std::vector<val_t>, 2> columns;

    HaloExchange(World& world, MPI_Comm mesh) {
        for (int face : {West, East}) {
            if (world.peers[face] != MPI_PROC_NULL) {
                columns[face - West].resize(world.tile.rows.size());
            }
        }
        for (size_t i = 0; i < buffers.size(); ++i) {
            buffers[i] = setupExchange(world, world.energy[i].data(), columns, mesh);
        }
    }

    void packColumns(const val_t* energy, int stride) {
        for (int face : {West, East}) {
            auto& column = columns[face - West];
            const size_t offset = face == West ? 0 : stride - 1;
            for (size_t row = 0; row < column.size(); ++row) {
                column[row] = energy[row * stride + offset];
            }
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        for (auto& exchange : buffers) {
            for (int i = 0; i < exchange.count; ++i) {
                MPI_Request_free(&exchange.requests[i]);
            }
        }
    }
};

void runSimulation(World& world, int iterations, HaloExchange& halo) {
    const int rows = world.tile.rows.size();
    const int cols = world.tile.cols.size();
    const int first_row = world.peers[North] != MPI_PROC_NULL;
    const int first_col = world.peers[West] != MPI_PROC_NULL;
    const Span inner_rows{first_row, std::max(first_row, rows - (world.peers[South] != MPI_PROC_NULL))};
    const Span inner_cols{first_col, std::max(first_col, cols - (world.peers[East] != MPI_PROC_NULL))};

    for (int iter = 0; iter < iterations; ++iter) {
        const val_t* old_energy = world.energy[world.current].data();
        val_t* new_energy = world.energy[1 - world.current].data();
        val_t* flux = world.flux.data();
        Exchange& exchange = halo.buffers[world.current];

        if (exchange.count != 0) {
            halo.packColumns(old_energy, cols);
            MPI_Startall(exchange.count, exchange.requests.data());
        }
        updateCells(world, old_energy, new_energy, flux, inner_rows, inner_cols);
        if (exchange.count != 0) {
            MPI_Waitall(exchange.count, exchange.requests.data(), MPI_STATUSES_IGNORE);
            updateCells(world, old_energy, new_energy, flux, {0, inner_rows.begin}, {0, cols});
            updateCells(world, old_energy, new_energy, flux, {inner_rows.end, rows}, {0, cols});
            updateCells(world, old_energy, new_energy, flux, inner_rows, {0, inner_cols.begin});
            updateCells(world, old_energy, new_energy, flux, inner_rows, {inner_cols.end, cols});
        }
        world.current = 1 - world.current;
    }
}

struct Summary {
    uint64_t result_hash = 0;
    std::array<val_t, 2> sums{};
    val_t min_energy = std::numeric_limits<val_t>::max();
    val_t max_energy = std::numeric_limits<val_t>::lowest();
};

template <bool validate>
Summary summarize(const World& world, int side) {
    Summary summary;
    const auto& energy = world.energy[world.current];
    const int cols = world.tile.cols.size();
    size_t i = 0;
    for (int row = world.tile.rows.begin; row < world.tile.rows.end; ++row) {
        uint64_t global_index = uint64_t(row) * side + world.tile.cols.begin;
        for (int col = 0; col < cols; ++col, ++i, ++global_index) {
            summary.result_hash ^= (std::bit_cast<uint64_t>(energy[i]) + global_index) * 0x9e3779b97f4a7c15ULL;
            summary.result_hash ^= (std::bit_cast<uint64_t>(world.flux[i]) + global_index) * 0xbf58476d1ce4e5b9ULL;
            if constexpr (validate) {
                summary.sums[0] += energy[i];
                summary.sums[1] += world.flux[i];
                summary.min_energy = std::min(energy[i], summary.min_energy);
                summary.max_energy = std::max(energy[i], summary.max_energy);
            }
        }
    }
    return summary;
}

bool validateResults(const Summary& summary) {
    const auto [energy_sum, flux_sum] = summary.sums;
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", summary.min_energy, summary.max_energy);

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
    if (!std::isfinite(summary.max_energy) || !std::isfinite(summary.min_energy)) {
        printf("  ERROR: Energy extrema are not finite\n");
        return false;
    }
    printf("  Validation: PASSED\n");
    return true;
}

void printDistributedResults(const World& world, int side, const int dims[2], int rank, MPI_Comm mesh) {
    const auto& energy = world.energy[world.current];
    const auto& tile = world.tile;
    constexpr int target_batch_elements = 256 * 1024;
    const int batch_rows = std::max(1, target_batch_elements / side);
    constexpr int results_tag = FaceCount;
    if (rank != 0) {
        for (int row = 0; row < tile.rows.size();) {
            const int rows = std::min(batch_rows, tile.rows.size() - row);
            MPI_Send(energy.data() + size_t(row) * tile.cols.size(), rows * tile.cols.size(),
                     MPI_DOUBLE, 0, results_tag, mesh);
            row += rows;
        }
        return;
    }

    // Stream the same row-major sequence used by the serial Kahan sum and FNV-1a hash.
    const size_t elements = size_t(side) * side;
    const size_t last_quartile = elements - elements / 4 - (elements % 4 != 0);
    const std::array<size_t, 5> sample_indices{0, elements / 4, elements / 2,
                                               last_quartile, elements - 1};
    std::array<val_t, sample_indices.size()> samples{};
    size_t next_sample = 0;
    hash digest;
    val_t sum = 0.0, correction = 0.0;
    val_t min_energy = energy[0], max_energy = energy[0];
    std::vector<val_t> batch(size_t(std::min(batch_rows, tile.rows.size())) * side);
    std::vector<MPI_Request> requests(dims[1]);
    std::vector<Span> col_spans;
    col_spans.reserve(dims[1]);
    for (int col = 0; col < dims[1]; ++col) {
        col_spans.push_back(partition(side, dims[1], col));
    }
    for (int process_row = 0; process_row < dims[0]; ++process_row) {
        const Span row_span = partition(side, dims[0], process_row);
        for (int begin = row_span.begin; begin < row_span.end;) {
            const int rows = std::min(batch_rows, row_span.end - begin);
            int request_count = 0;
            for (int process_col = 0; process_col < dims[1]; ++process_col) {
                const int peer = process_row * dims[1] + process_col;
                const Span cols = col_spans[process_col];
                val_t* block = batch.data() + size_t(rows) * cols.begin;
                const int count = rows * cols.size();
                if (peer == 0) {
                    const size_t local_row = begin - tile.rows.begin;
                    std::copy_n(energy.data() + local_row * cols.size(), count, block);
                } else {
                    MPI_Irecv(block, count, MPI_DOUBLE, peer, results_tag,
                              mesh, &requests[request_count++]);
                }
            }
            MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE);

            for (int row = 0; row < rows; ++row) {
                const size_t row_start = size_t(begin + row) * side;
                for (const Span cols : col_spans) {
                    const val_t* segment = batch.data() + size_t(rows) * cols.begin
                                                       + size_t(row) * cols.size();
                    for (int col = 0; col < cols.size(); ++col) {
                        const val_t value = segment[col];
                        const val_t y = value - correction;
                        const val_t next_sum = sum + y;
                        correction = (next_sum - sum) - y;
                        sum = next_sum;
                        min_energy = std::min(min_energy, value);
                        max_energy = std::max(max_energy, value);
                        digest.add(value);
                    }
                    while (next_sample < samples.size() && sample_indices[next_sample] < row_start + cols.end) {
                        samples[next_sample] = segment[sample_indices[next_sample] - row_start - cols.begin];
                        ++next_sample;
                    }
                }
            }
            begin += rows;
        }
    }

    printf("=== RESULTS ===\n");
    printf("Name: ElementEnergy\n");
    printf("Elements: %zu\n", elements);
    printf("Sum: %.17e\n", sum);
    printf("Min: %.17e\n", min_energy);
    printf("Max: %.17e\n", max_energy);
    for (size_t i = 0; i < samples.size(); ++i) {
        printf("Sample[%zu]: %.17e\n", sample_indices[i], samples[i]);
    }
    printf("Hash: %016" PRIx64 "\n", digest.get());
    printf("=== END RESULTS ===\n");
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

int benchmark(int argc, char** argv, int rank, int ranks) {
    int side = 512;
    int iterations = 10;
    bool validate = false;
    bool print_results_flag = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            side = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            print_results_flag = true;
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
    if (side <= 0) {
        if (rank == 0) {
            printf("Grid size must be positive\n");
        }
        return 1;
    }

    const uint64_t elements = uint64_t(side) * side;
    int active_ranks = static_cast<int>(std::min<uint64_t>(ranks, elements));
    int dims[2];
    for (;; --active_ranks) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(active_ranks, 2, dims);
        if (dims[0] <= side && dims[1] <= side) {
            break;
        }
    }

    const int periods[2] = {0, 0};
    MPI_Comm mesh;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &mesh);
    if (mesh == MPI_COMM_NULL) {
        return 0;
    }

    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %" PRIu64 " elements\n", side, side, elements);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    const Tile tile = tileForRank(side, dims, rank);
    World world = buildSquare2D(side, tile, mesh);
    const size_t packed_columns = size_t(tile.rows.size()) *
        ((world.peers[West] != MPI_PROC_NULL) + (world.peers[East] != MPI_PROC_NULL));
    const uint64_t local_dynamic = (2 * world.energy[0].size() + world.flux.size() + packed_columns)
                                   * sizeof(val_t);
    uint64_t dynamic_mem;
    MPI_Reduce(&local_dynamic, &dynamic_mem, 1, MPI_UINT64_T, MPI_SUM, 0, mesh);
    if (rank == 0) {
        const double static_mb = elements * sizeof(Cell) / (1024.0 * 1024.0);
        const double dynamic_mb = dynamic_mem / (1024.0 * 1024.0);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               static_mb + dynamic_mb, static_mb, dynamic_mb);
        printf("Running simulation...\n");
    }

    double local_seconds;
    {
        HaloExchange halo(world, mesh);
        MPI_Barrier(mesh);
        const double start = MPI_Wtime();
        runSimulation(world, iterations, halo);
        local_seconds = MPI_Wtime() - start;
    }
    double seconds;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);

    const Summary local = validate ? summarize<true>(world, side) : summarize<false>(world, side);
    Summary global;
    MPI_Reduce(&local.result_hash, &global.result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, mesh);

    if (rank == 0) {
        const double measured_iters = iterations > 1 ? iterations - 1 : 1;
        const double giga_elems_per_sec = measured_iters * elements / seconds / 1e9;
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", seconds * 1000.0 / measured_iters);
        printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
        printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", global.result_hash);
    }

    if (print_results_flag) {
        printDistributedResults(world, side, dims, rank, mesh);
    }
    bool valid = true;
    if (validate) {
        MPI_Reduce(local.sums.data(), global.sums.data(), 2, MPI_DOUBLE, MPI_SUM, 0, mesh);
        MPI_Reduce(&local.min_energy, &global.min_energy, 1, MPI_DOUBLE, MPI_MIN, 0, mesh);
        MPI_Reduce(&local.max_energy, &global.max_energy, 1, MPI_DOUBLE, MPI_MAX, 0, mesh);
        if (rank == 0) {
            valid = validateResults(global);
        }
    }
    MPI_Comm_free(&mesh);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status;
    try {
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
