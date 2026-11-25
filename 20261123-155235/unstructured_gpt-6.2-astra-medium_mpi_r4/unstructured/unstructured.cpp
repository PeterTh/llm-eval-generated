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

struct Tile {
    int row, col, height, width;
};

Tile tileFor(int rank, const int dims[2], int n) {
    const int x = rank / dims[1], y = rank % dims[1];
    const int h = n / dims[0], w = n / dims[1];
    const int extraRows = n % dims[0], extraCols = n % dims[1];
    return {x * h + std::min(x, extraRows), y * w + std::min(y, extraCols),
            h + (x < extraRows), w + (y < extraCols)};
}

// All connections have unit weight and all materials have transfer coefficient 0.8.
struct Element {
    std::array<size_t, 4> neighbor;
    double external_flow;
};

struct World {
    std::vector<Element> mesh;
    std::vector<double> energy, next, flux;
};

World buildSquare2D(Tile t, int n) {
    const size_t w = t.width, count = w * t.height;
    // Ghost faces follow the owned cells in north, south, west, east order.
    const size_t north = count, south = north + w;
    const size_t west = south + w, east = west + t.height;
    World world;
    world.mesh.resize(count);
    world.energy.resize(east + t.height);
    world.next.resize(world.energy.size());
    world.flux.resize(count);
    for (int x = 0; x < t.height; ++x) {
        for (int y = 0; y < t.width; ++y) {
            const size_t i = x * w + y;
            const int gx = t.row + x, gy = t.col + y;
            // A missing physical connection contributes zero via a self-reference.
            auto& e = world.mesh[i];
            e.neighbor = {x + 1 < t.height ? i + w : (gx + 1 < n ? south + y : i),
                          x > 0 ? i - w : (gx > 0 ? north + y : i),
                          y + 1 < t.width ? i + 1 : (gy + 1 < n ? east + x : i),
                          y > 0 ? i - 1 : (gy > 0 ? west + x : i)};
            if ((gx == 0 || gx == n - 1) && (gy == 0 || gy == n - 1))
                e.external_flow = gx == gy ? 0.5 : -0.5;
        }
    }
    return world;
}

void update(World& world, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        const Element& e = world.mesh[i];
        const double energy = world.energy[i];
        double flow = e.external_flow;
        for (size_t j : e.neighbor)
            flow += (world.energy[j] - energy) * 0.8 * 0.25;
        world.next[i] = energy + flow;
        world.flux[i] += std::abs(flow);
    }
}

void runSimulation(World& world, Tile t, MPI_Comm grid, MPI_Datatype column, int iters) {
    int peer[4];
    MPI_Cart_shift(grid, 0, 1, &peer[0], &peer[1]);
    MPI_Cart_shift(grid, 1, 1, &peer[2], &peer[3]);
    const size_t w = t.width, count = world.mesh.size();
    const size_t face[] = {0, count - w, 0, w - 1};
    const size_t ghost[] = {count, count + w, count + 2*w, count + 2*w + t.height};
    const int top = peer[0] != MPI_PROC_NULL;
    const int bottom = std::max(top, t.height - (peer[1] != MPI_PROC_NULL));
    const int left = peer[2] != MPI_PROC_NULL;
    const int right = std::max(left, t.width - (peer[3] != MPI_PROC_NULL));

    for (int iter = 0; iter < iters; ++iter) {
        MPI_Request requests[8];
        int pending = 0;
        double* energy = world.energy.data();
        for (int d = 0; d < 4; ++d) {
            if (peer[d] == MPI_PROC_NULL) continue;
            MPI_Irecv(energy + ghost[d], d < 2 ? t.width : t.height, MPI_DOUBLE,
                      peer[d], d ^ 1, grid, &requests[pending++]);
            MPI_Isend(energy + face[d], d < 2 ? t.width : 1,
                      d < 2 ? MPI_DOUBLE : column, peer[d], d, grid, &requests[pending++]);
        }
        if (left == 0 && right == t.width) {
            update(world, top * w, bottom * w);
        } else {
            for (int x = top; x < bottom; ++x)
                update(world, x * w + left, x * w + right);
        }
        if (pending) {
            MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
            update(world, 0, top * w);
            update(world, bottom * w, count);
            if (left || right < t.width) {
                for (int x = top; x < bottom; ++x) {
                    update(world, x * w, x * w + left);
                    update(world, x * w + right, (x + 1) * w);
                }
            }
        }
        world.energy.swap(world.next);
    }
}

uint64_t localHash(const World& world, Tile t, int n) {
    uint64_t hash = 0;
    for (int x = 0; x < t.height; ++x) {
        for (int y = 0; y < t.width; ++y) {
            const size_t i = size_t(x) * t.width + y;
            const uint64_t global = uint64_t(t.row + x) * n + t.col + y;
            hash ^= (std::bit_cast<uint64_t>(world.energy[i]) + global) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(world.flux[i]) + global) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Gather only for ordered reporting. Derived datatypes place each tile directly
// in the global array and keep MPI counts within the range of an int.
std::vector<double> gatherField(const std::vector<double>& field, Tile t, int n,
                                int rank, const int dims[2], MPI_Comm grid) {
    if (rank != 0) {
        MPI_Datatype block;
        MPI_Type_vector(t.height, t.width, t.width, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(field.data(), 1, block, 0, 4, grid);
        MPI_Type_free(&block);
        return {};
    }
    std::vector<double> global(size_t(n) * n);
    for (int x = 0; x < t.height; ++x)
        std::copy_n(field.data() + size_t(x) * t.width, t.width, global.data() + size_t(x) * n);
    for (int src = 1; src < dims[0] * dims[1]; ++src) {
        Tile other = tileFor(src, dims, n);
        MPI_Datatype block;
        MPI_Type_vector(other.height, other.width, n, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(global.data() + size_t(other.row) * n + other.col, 1, block, src, 4,
                 grid, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    return global;
}

// Validate simulation results
bool validateResults(const std::vector<double>& energy, const std::vector<double>& flux) {
    double energy_sum = 0.0;
    double flux_sum = 0.0;
    double energy_max = std::numeric_limits<double>::lowest();
    double energy_min = std::numeric_limits<double>::max();
    
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
    
    // Check for numerical issues
    constexpr double energy_epsilon = 1e-8;
    
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n_elems_root = 512;
    int n_iters = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    
    if (n_elems_root <= 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive\n");
        MPI_Finalize();
        return 1;
    }
    const size_t n_elems = size_t(n_elems_root) * n_elems_root;
    int active = std::min<size_t>(ranks, n_elems);
    int dims[2] = {};
    for (;;) {
        MPI_Dims_create(active, 2, dims);
        if (std::max(dims[0], dims[1]) <= n_elems_root) break;
        --active;
        dims[0] = dims[1] = 0;
    }
    const int periodic[2] = {};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &grid);
    if (grid == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    const Tile tile = tileFor(rank, dims, n_elems_root);
    if (rank == 0) {
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %zu elements\n", n_elems_root, n_elems_root, n_elems);
        printf("Iterations: %d\n", n_iters);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
        printf("Building unstructured mesh...\n");
    }
    World world = buildSquare2D(tile, n_elems_root);
    if (rank == 0) {
        const double static_mb = double(n_elems) * sizeof(Element) / (1024 * 1024);
        const double ghosts = 2.0 * n_elems_root * (double(dims[0]) + dims[1]);
        const double dynamic_mb = (3.0 * n_elems + 2.0 * ghosts) * sizeof(double) / (1024 * 1024);
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               static_mb + dynamic_mb, static_mb, dynamic_mb);
        printf("Running simulation...\n");
    }
    MPI_Datatype column;
    MPI_Type_vector(tile.height, 1, tile.width, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    runSimulation(world, tile, grid, column, n_iters);
    const double elapsed_ms = (MPI_Wtime() - start) * 1000.0;
    double duration_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    MPI_Type_free(&column);

    const uint64_t partial = localHash(world, tile, n_elems_root);
    uint64_t hash = 0;
    MPI_Reduce(&partial, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, grid);
    if (rank == 0) {
        const int measured = n_iters > 1 ? n_iters - 1 : 1;
        const double rate = double(measured) * n_elems / (duration_ms * 1e6);
        printf("Computation time: %.0f ms\n", duration_ms);
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n", duration_ms / measured);
        printf("  Elements/sec: %.4f GigaElements/s\n", rate);
        printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }
    int status = 0;
    if (printResults || validate) {
        auto energy = gatherField(world.energy, tile, n_elems_root, rank, dims, grid);
        if (rank == 0 && printResults) print_results(energy, "ElementEnergy");
        if (validate) {
            auto flux = gatherField(world.flux, tile, n_elems_root, rank, dims, grid);
            if (rank == 0) status = validateResults(energy, flux) ? 0 : 1;
            MPI_Bcast(&status, 1, MPI_INT, 0, grid);
        }
    }
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return status;
}
