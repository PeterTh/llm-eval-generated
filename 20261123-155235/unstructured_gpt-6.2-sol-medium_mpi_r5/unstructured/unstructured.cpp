#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
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

enum { X_MINUS, X_PLUS, Y_MINUS, Y_PLUS, NUM_FACES };

struct Axis {
    int begin;
    int count;
};

Axis partitionAxis(int n, int parts, int coord) {
    const int extra = n % parts;
    return {coord * (n / parts) + std::min(coord, extra),
            n / parts + (coord < extra)};
}

struct Domain {
    Axis x;
    Axis y;
    std::array<int, NUM_FACES> peers;
};

struct ProcessGrid {
    int rows = 1;
    int cols = 1;

    ProcessGrid(int n, int ranks) {
        // Use as many nonempty tiles as possible, then balance the two axes.
        // On a tie, favor longer contiguous rows over strided columns.
        const int tiles = std::min(ranks, n * n);
        for (int r = 1; r <= n && r <= tiles / r; ++r) {
            const int c = std::min(n, tiles / r);
            if (r * c >= rows * cols) {
                rows = c;
                cols = r;
            }
        }
    }

    Domain domain(int n, int rank) const {
        const int x = rank / cols;
        const int y = rank % cols;
        return {partitionAxis(n, rows, x), partitionAxis(n, cols, y),
                {x > 0 ? rank - cols : MPI_PROC_NULL,
                 x + 1 < rows ? rank + cols : MPI_PROC_NULL,
                 y > 0 ? rank - 1 : MPI_PROC_NULL,
                 y + 1 < cols ? rank + 1 : MPI_PROC_NULL}};
    }
};

// The generated mesh has up to four connections, all with unit flux coefficient.
// All materials have transfer coefficient 0.8 and sources are multiples of 0.5.
using Connections = std::array<idx_t, 4>;

// Elements at the physical edge have fewer neighbors and may carry a source.
struct EdgeElement {
    idx_t index;
    uint8_t degree;
    int8_t source;
};

struct Range {
    idx_t begin;
    idx_t end;
};

struct Work {
    std::vector<Range> full_degree;
    std::vector<EdgeElement> edges;
};

void appendRange(std::vector<Range>& ranges, idx_t begin, idx_t end) {
    if (begin == end) {
        return;
    }
    if (!ranges.empty() && ranges.back().end == begin) {
        ranges.back().end = end;
    } else {
        ranges.push_back({begin, end});
    }
}

struct Face {
    int peer;
    int count;
    idx_t send_offset;
    idx_t ghost_offset = 0;
    std::vector<val_t> packed{};
};

struct World {
    Domain domain;
    std::vector<Connections> connections;
    std::array<Face, NUM_FACES> faces;
    Work interior;
    Work boundary;
    std::vector<val_t> energy;
    std::vector<val_t> next_energy;
    std::vector<val_t> total_flux;
};

World buildSquare2D(const Domain& domain, int n) {
    const int h = domain.x.count;
    const int w = domain.y.count;
    const auto& peers = domain.peers;
    const idx_t count = h * w;
    World world{};
    world.domain = domain;
    world.connections.resize(count);
    world.faces = {{{peers[X_MINUS], w, 0},
                    {peers[X_PLUS], w, static_cast<idx_t>((h - 1) * w)},
                    {peers[Y_MINUS], h, 0},
                    {peers[Y_PLUS], h, static_cast<idx_t>(w - 1)}}};
    world.total_flux.resize(count);

    idx_t energy_size = count;
    for (int f = 0; f < NUM_FACES; ++f) {
        Face& face = world.faces[f];
        if (face.peer != MPI_PROC_NULL) {
            face.ghost_offset = energy_size;
            energy_size += face.count;
            if (f >= Y_MINUS) {
                face.packed.resize(face.count);
            }
        }
    }
    world.energy.resize(energy_size);
    world.next_energy.resize(energy_size);

    const int first_y = peers[Y_MINUS] != MPI_PROC_NULL;
    const int last_y = std::max(first_y, w - (peers[Y_PLUS] != MPI_PROC_NULL));
    for (int x = 0; x < h; ++x) {
        const idx_t row = x * w;
        const bool halo_row = (x == 0 && peers[X_MINUS] != MPI_PROC_NULL)
                           || (x == h - 1 && peers[X_PLUS] != MPI_PROC_NULL);
        const int gx = domain.x.begin + x;
        for (int y = 0; y < w; ++y) {
            const int gy = domain.y.begin + y;
            const idx_t i = row + y;
            Connections& neighbors = world.connections[i];
            uint8_t degree = 0;

            // Preserve the serial accumulation order (+x, -x, +y, -y).
            if (gx + 1 < n) {
                neighbors[degree++] =
                    x + 1 < h ? i + w : world.faces[X_PLUS].ghost_offset + y;
            }
            if (gx > 0) {
                neighbors[degree++] =
                    x > 0 ? i - w : world.faces[X_MINUS].ghost_offset + y;
            }
            if (gy + 1 < n) {
                neighbors[degree++] =
                    y + 1 < w ? i + 1 : world.faces[Y_PLUS].ghost_offset + x;
            }
            if (gy > 0) {
                neighbors[degree++] =
                    y > 0 ? i - 1 : world.faces[Y_MINUS].ghost_offset + x;
            }

            Work& work = (halo_row || y < first_y || y >= last_y) ? world.boundary : world.interior;
            if (degree == 4) {
                appendRange(work.full_degree, i, i + 1);
            } else {
                int8_t source = 0;
                if ((gx == 0 || gx == n - 1) && (gy == 0 || gy == n - 1)) {
                    source = gx == gy ? 1 : -1;
                }
                work.edges.push_back({i, degree, source});
            }
        }
    }
    return world;
}

inline val_t computeFlux(val_t current, val_t neighbor) {
    return ((neighbor - current) * 0.8) * 0.25;
}

void updateElements(World& world, const Work& work) {
    const val_t* energy = world.energy.data();
    for (const Range& range : work.full_degree) {
        for (idx_t i = range.begin; i < range.end; ++i) {
            const Connections& neighbors = world.connections[i];
            const val_t current = energy[i];
            val_t flux = 0.0;
            flux += computeFlux(current, energy[neighbors[0]]);
            flux += computeFlux(current, energy[neighbors[1]]);
            flux += computeFlux(current, energy[neighbors[2]]);
            flux += computeFlux(current, energy[neighbors[3]]);
            world.next_energy[i] = current + flux;
            world.total_flux[i] += std::abs(flux);
        }
    }
    for (const EdgeElement& edge : work.edges) {
        const idx_t i = edge.index;
        const val_t current = energy[i];
        val_t flux = edge.source * 0.5;
        for (int j = 0; j < edge.degree; ++j) {
            flux += computeFlux(current, energy[world.connections[i][j]]);
        }
        world.next_energy[i] = current + flux;
        world.total_flux[i] += std::abs(flux);
    }
}

void runSimulation(World& world, int n_iters, MPI_Comm comm) {
    for (int iter = 0; iter < n_iters; ++iter) {
        // At iteration zero both owned and ghost energies are already initialized.
        MPI_Request requests[2 * NUM_FACES];
        int receives = 0;
        int pending = 0;
        if (iter > 0) {
            for (int f = 0; f < NUM_FACES; ++f) {
                const Face& face = world.faces[f];
                if (face.peer != MPI_PROC_NULL) {
                    MPI_Irecv(world.energy.data() + face.ghost_offset, face.count, MPI_DOUBLE,
                              face.peer, f ^ 1, comm, &requests[receives++]);
                }
            }
            pending = receives;
            for (int f = 0; f < NUM_FACES; ++f) {
                Face& face = world.faces[f];
                if (face.peer == MPI_PROC_NULL) {
                    continue;
                }
                const val_t* send = world.energy.data() + face.send_offset;
                if (f >= Y_MINUS) {
                    const size_t stride = world.domain.y.count;
                    for (int x = 0; x < face.count; ++x) {
                        face.packed[x] = send[x * stride];
                    }
                    send = face.packed.data();
                }
                MPI_Isend(send, face.count, MPI_DOUBLE, face.peer, f, comm, &requests[pending++]);
            }
        }

        updateElements(world, world.interior);
        if (receives > 0) {
            MPI_Waitall(receives, requests, MPI_STATUSES_IGNORE);
        }
        updateElements(world, world.boundary);
        if (pending > receives) {
            MPI_Waitall(pending - receives, requests + receives, MPI_STATUSES_IGNORE);
        }
        world.energy.swap(world.next_energy);
    }
}

uint64_t computeHash(const World& world, int n) {
    uint64_t hash = 0;
    size_t i = 0;
    for (int x = 0; x < world.domain.x.count; ++x) {
        const uint64_t row = uint64_t(world.domain.x.begin + x) * n + world.domain.y.begin;
        for (int y = 0; y < world.domain.y.count; ++y, ++i) {
            const uint64_t index = row + y;
            hash ^= (std::bit_cast<uint64_t>(world.energy[i]) + index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(world.total_flux[i]) + index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

void printDistributedResults(const World& world, const ProcessGrid& grid, int n,
                             int rank, MPI_Comm comm) {
    constexpr int results_tag = NUM_FACES;
    const int ranks = grid.rows * grid.cols;
    if (ranks == 1) {
        print_results(world.energy, "ElementEnergy");
        return;
    }
    if (rank != 0) {
        MPI_Send(world.energy.data(), static_cast<int>(world.connections.size()), MPI_DOUBLE,
                 0, results_tag, comm);
        return;
    }

    // The external hash and compensated sum are order-sensitive. Place each
    // contiguous tile in its global row-major positions before calling the helper.
    std::vector<val_t> energy(size_t(n) * n);
    if (grid.cols == 1) {
        std::copy_n(world.energy.data(), world.connections.size(), energy.data());
        for (int peer = 1; peer < ranks; ++peer) {
            const Axis rows = partitionAxis(n, grid.rows, peer);
            MPI_Recv(energy.data() + size_t(rows.begin) * n, rows.count * n, MPI_DOUBLE,
                     peer, results_tag, comm, MPI_STATUS_IGNORE);
        }
    } else {
        // The first tile is at least as large as every other tile.
        std::vector<val_t> packed(world.connections.size());
        for (int received = 0; received < ranks; ++received) {
            int peer = 0;
            const val_t* data = world.energy.data();
            if (received != 0) {
                MPI_Status status;
                MPI_Recv(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                         MPI_ANY_SOURCE, results_tag, comm, &status);
                peer = status.MPI_SOURCE;
                data = packed.data();
            }
            const Domain tile = grid.domain(n, peer);
            const int w = tile.y.count;
            for (int x = 0; x < tile.x.count; ++x) {
                std::copy_n(data + size_t(x) * w, w,
                            energy.data() + size_t(tile.x.begin + x) * n + tile.y.begin);
            }
        }
    }
    print_results(energy, "ElementEnergy");
}

bool validateResults(const World& world, int rank, MPI_Comm comm) {
    val_t sums[2] = {};
    val_t extrema[2] = {std::numeric_limits<val_t>::lowest(),
                        std::numeric_limits<val_t>::lowest()};
    for (size_t i = 0; i < world.connections.size(); ++i) {
        const val_t energy = world.energy[i];
        sums[0] += energy;
        sums[1] += world.total_flux[i];
        extrema[0] = std::max(energy, extrema[0]);
        extrema[1] = std::max(-energy, extrema[1]);
    }
    val_t global_sums[2];
    val_t global_extrema[2];
    MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(extrema, global_extrema, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) {
        return true;
    }

    const val_t energy_sum = global_sums[0];
    const val_t flux_sum = global_sums[1];
    const val_t energy_min = -global_extrema[1];
    const val_t energy_max = global_extrema[0];
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n", energy_sum);
    printf("  Flux sum: %.2f\n", flux_sum);
    printf("  Energy range: [%.6f, %.6f]\n", energy_min, energy_max);

    if (!std::isfinite(energy_sum)) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(energy_sum) > 1e-8) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    enum { GRID_SIZE, ITERATIONS, VALIDATE, RESULTS, EXIT_STATUS, NUM_OPTIONS };
    int options[NUM_OPTIONS] = {512, 10, 0, 0, -1};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                options[GRID_SIZE] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                options[ITERATIONS] = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                options[VALIDATE] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                options[RESULTS] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                options[EXIT_STATUS] = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                options[EXIT_STATUS] = 1;
                break;
            }
        }
        const int n = options[GRID_SIZE];
        if (options[EXIT_STATUS] < 0 && (n <= 0 || n > INT_MAX / n)) {
            fprintf(stderr, "Grid size must be positive and its square must fit in an int.\n");
            options[EXIT_STATUS] = 1;
        }
    }
    MPI_Bcast(options, NUM_OPTIONS, MPI_INT, 0, MPI_COMM_WORLD);
    if (options[EXIT_STATUS] >= 0) {
        MPI_Finalize();
        return options[EXIT_STATUS];
    }

    const int n = options[GRID_SIZE];
    const int n_iters = options[ITERATIONS];
    const int n_elems = n * n;
    const ProcessGrid grid(n, ranks);
    const int active_ranks = grid.rows * grid.cols;
    MPI_Comm comm = MPI_COMM_WORLD;
    if (active_ranks != ranks) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < active_ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    }

    int status = 0;
    if (comm != MPI_COMM_NULL) {
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %d elements\n", n, n, n_elems);
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n\n", options[VALIDATE] ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
        }

        World world = buildSquare2D(grid.domain(n, rank), n);
        uint64_t local_memory[2] = {
            world.connections.size() * sizeof(Connections),
            (world.energy.size() + world.next_energy.size() + world.total_flux.size()) * sizeof(val_t)
        };
        for (const Work* work : {&world.interior, &world.boundary}) {
            local_memory[0] += work->full_degree.capacity() * sizeof(Range)
                             + work->edges.capacity() * sizeof(EdgeElement);
        }
        for (const Face& face : world.faces) {
            local_memory[1] += face.packed.size() * sizeof(val_t);
        }
        uint64_t memory[2];
        MPI_Reduce(local_memory, memory, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
        if (rank == 0) {
            constexpr double mb = 1024.0 * 1024.0;
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (memory[0] + memory[1]) / mb, memory[0] / mb, memory[1] / mb);
            printf("Running simulation...\n");
        }

        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        runSimulation(world, n_iters, comm);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0.0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        const uint64_t local_hash = computeHash(world, n);
        uint64_t result_hash = 0;
        MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

        if (rank == 0) {
            const double duration_ms = seconds * 1000.0;
            const double time_per_iter = n_iters > 0 ? duration_ms / n_iters : 0.0;
            const double giga_elems_per_sec = seconds > 0.0
                ? double(std::max(n_iters, 0)) * n_elems / seconds / 1e9 : 0.0;
            printf("Computation time: %lld ms\n", static_cast<long long>(duration_ms));
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", time_per_iter);
            printf("  Elements/sec: %.4f GigaElements/s\n", giga_elems_per_sec);
            printf("  Performance: %.4f GFLOPS\n", giga_elems_per_sec * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
        }

        if (options[RESULTS]) {
            printDistributedResults(world, grid, n, rank, comm);
        }
        if (options[VALIDATE] && !validateResults(world, rank, comm)) {
            status = 1;
        }
        if (comm != MPI_COMM_WORLD) {
            MPI_Comm_free(&comm);
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
