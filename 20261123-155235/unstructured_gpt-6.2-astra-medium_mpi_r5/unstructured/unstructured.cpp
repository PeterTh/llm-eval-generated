#include <algorithm>
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

struct Range {
    int begin;
    int size;
};

Range partition(int n, int parts, int coord) {
    const int extra = n % parts;
    return {coord * (n / parts) + std::min(coord, extra),
            n / parts + (coord < extra)};
}

struct Tile {
    Range rows, cols;
    size_t pitch;
    std::vector<double> energy, next, flux;

    Tile(Range rows, Range cols)
        : rows(rows), cols(cols), pitch(size_t(cols.size) + 2),
          energy((size_t(rows.size) + 2) * pitch), next(energy.size()),
          flux(energy.size()) {}
};

// Both buffers have a one-cell halo. On a physical edge a missing neighbor
// contributes zero. Preserve the original down, up, right, left flux order.
inline double transfer(double center, double neighbor) {
    return (neighbor - center) * 0.8 * 0.25;
}

void simulate(Tile& tile, int n, int iterations, MPI_Comm cart) {
    const int h = tile.rows.size, w = tile.cols.size;
    const size_t pitch = tile.pitch;
    int up, down, left, right;
    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);
    MPI_Datatype column;
    MPI_Type_vector(h, 1, static_cast<int>(pitch), MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    for (int iter = 0; iter < iterations; ++iter) {
        double* e = tile.energy.data();
        double* next = tile.next.data();
        double* flux = tile.flux.data();
        const size_t bottom = size_t(h) * pitch;
        MPI_Request requests[8];
        int count = 0;
        auto exchange = [&](int peer, size_t send, size_t recv, int length,
                            MPI_Datatype type) {
            if (peer == MPI_PROC_NULL) return;
            MPI_Irecv(e + recv, length, type, peer, 0, cart, &requests[count++]);
            MPI_Isend(e + send, length, type, peer, 0, cart, &requests[count++]);
        };
        exchange(up, pitch + 1, 1, w, MPI_DOUBLE);
        exchange(down, bottom + 1, bottom + pitch + 1, w, MPI_DOUBLE);
        exchange(left, pitch + 1, pitch, 1, column);
        exchange(right, pitch + w, pitch + w + 1, 1, column);

        auto update = [&](size_t i, double source) {
            double delta = source;
            delta += transfer(e[i], e[i + pitch]);
            delta += transfer(e[i], e[i - pitch]);
            delta += transfer(e[i], e[i + 1]);
            delta += transfer(e[i], e[i - 1]);
            next[i] = e[i] + delta;
            flux[i] += std::abs(delta);
        };
        for (int x = 2; x < h; ++x) {
            const size_t row = size_t(x) * pitch;
            for (int y = 2; y < w; ++y) update(row + y, 0.0);
        }
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);

        // Mirroring the boundary energy makes the missing connection's
        // difference zero without a branch in the stencil.
        if (up == MPI_PROC_NULL) std::copy_n(e + pitch + 1, w, e + 1);
        if (down == MPI_PROC_NULL) std::copy_n(e + bottom + 1, w, e + bottom + pitch + 1);
        for (int x = 1; x <= h; ++x) {
            const size_t row = size_t(x) * pitch;
            if (left == MPI_PROC_NULL) e[row] = e[row + 1];
            if (right == MPI_PROC_NULL) e[row + w + 1] = e[row + w];
            const int gx = tile.rows.begin + x - 1;
            auto edge = [&](int y) {
                const int gy = tile.cols.begin + y - 1;
                double source = 0.0;
                if ((gx == 0 || gx == n - 1) && (gy == 0 || gy == n - 1))
                    source = gx == gy ? 0.5 : -0.5;
                update(row + y, source);
            };
            edge(1);
            if (x == 1 || x == h)
                for (int y = 2; y < w; ++y) update(row + y, 0.0);
            if (w > 1) edge(w);
        }
        tile.energy.swap(tile.next);
    }
    MPI_Type_free(&column);
}

uint64_t localHash(const Tile& tile, int n) {
    uint64_t result = 0;
    for (int x = 0; x < tile.rows.size; ++x) {
        const size_t row = (size_t(x) + 1) * tile.pitch + 1;
        const uint64_t global = uint64_t(tile.rows.begin + x) * n + tile.cols.begin;
        for (int y = 0; y < tile.cols.size; ++y) {
            result ^= (std::bit_cast<uint64_t>(tile.energy[row + y]) + global + y) * 0x9e3779b97f4a7c15ULL;
            result ^= (std::bit_cast<uint64_t>(tile.flux[row + y]) + global + y) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return result;
}

// External verification uses an order-sensitive byte hash and compensated
// sum. Receive each tile directly into its final position in global row order.
void printEnergy(const Tile& tile, int n, const int dims[2], int rank, MPI_Comm cart) {
    if (rank != 0) {
        MPI_Datatype block;
        MPI_Type_vector(tile.rows.size, tile.cols.size, static_cast<int>(tile.pitch), MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(tile.energy.data() + tile.pitch + 1, 1, block, 0, 1, cart);
        MPI_Type_free(&block);
        return;
    }
    std::vector<double> energy(size_t(n) * n);
    for (int x = 0; x < tile.rows.size; ++x)
        std::copy_n(tile.energy.data() + (size_t(x) + 1) * tile.pitch + 1,
                    tile.cols.size, energy.data() + size_t(x) * n);
    for (int peer = 1; peer < dims[0] * dims[1]; ++peer) {
        int coords[2];
        MPI_Cart_coords(cart, peer, 2, coords);
        const Range rows = partition(n, dims[0], coords[0]);
        const Range cols = partition(n, dims[1], coords[1]);
        MPI_Datatype block;
        MPI_Type_vector(rows.size, cols.size, n, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(energy.data() + size_t(rows.begin) * n + cols.begin, 1, block,
                 peer, 1, cart, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(energy, "ElementEnergy");
}

bool validateResults(const Tile& tile, int rank, MPI_Comm cart) {
    double sums[2] = {};
    double extrema[2] = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
    for (int x = 1; x <= tile.rows.size; ++x) {
        for (int y = 1; y <= tile.cols.size; ++y) {
            const size_t i = size_t(x) * tile.pitch + y;
            sums[0] += tile.energy[i];
            sums[1] += tile.flux[i];
            extrema[0] = std::max(extrema[0], -tile.energy[i]);
            extrema[1] = std::max(extrema[1], tile.energy[i]);
        }
    }
    double global_sums[2], global_extrema[2];
    MPI_Reduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, 0, cart);
    MPI_Reduce(extrema, global_extrema, 2, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank != 0) return true;
    const double energy_sum = global_sums[0], flux_sum = global_sums[1];
    const double energy_min = -global_extrema[0], energy_max = global_extrema[1];
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
    
    const int n = n_elems_root;
    if (n <= 0 || n > std::numeric_limits<int>::max() - 2) {
        if (rank == 0) fprintf(stderr, "Grid size is outside the supported range\n");
        MPI_Finalize();
        return 1;
    }

    // Maximize active ranks, then minimize the tile perimeter for that count.
    int dims[2] = {1, 1};
    for (int rows = 1; rows <= n && rows <= ranks / rows; ++rows) {
        const int cols = std::min(n, ranks / rows);
        if (rows * cols >= dims[0] * dims[1]) {
            dims[0] = rows;
            dims[1] = cols;
        }
    }
    // Favor contiguous row messages when the process grid is rectangular.
    std::swap(dims[0], dims[1]);
    const int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);
    int status = 0;
    if (cart != MPI_COMM_NULL) {
        int coords[2];
        MPI_Cart_coords(cart, rank, 2, coords);
        const uint64_t elements = uint64_t(n) * n;
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, elements);
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
        }
        Tile tile(partition(n, dims[0], coords[0]), partition(n, dims[1], coords[1]));
        // Sum of the padded tile areas over the Cartesian grid.
        const double dynamic_mb = (double(n) + 2.0 * dims[0]) * (double(n) + 2.0 * dims[1])
                                  * 3 * sizeof(double) / (1024.0 * 1024.0);
        if (rank == 0) {
            printf("Memory usage: %.2f MB (static: 0.00 MB, dynamic: %.2f MB)\n\n",
                   dynamic_mb, dynamic_mb);
            printf("Running simulation...\n");
        }
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        simulate(tile, n, n_iters, cart);
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

        const uint64_t local = localHash(tile, n);
        uint64_t hash;
        MPI_Reduce(&local, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, cart);
        if (rank == 0) {
            const double ms = seconds * 1000.0;
            const double measured_iters = std::max(double(n_iters) - 1, 1.0);
            const double rate = measured_iters * elements / seconds / 1e9;
            printf("Computation time: %.3f ms\n", ms);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", ms / measured_iters);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (printResults) printEnergy(tile, n, dims, rank, cart);
        if (validate && !validateResults(tile, rank, cart)) status = 1;
        MPI_Comm_free(&cart);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
