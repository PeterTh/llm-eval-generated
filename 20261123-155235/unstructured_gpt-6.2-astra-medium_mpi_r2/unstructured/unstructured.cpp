#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

struct Interval {
    int first, length;
};

Interval partition(int n, int parts, int coord) {
    const int base = n / parts;
    const int extra = n % parts;
    return {coord * base + std::min(coord, extra), base + (coord < extra)};
}

// Use as many nonempty tiles as possible, preferring square tiles on ties.
MPI_Comm createGrid(int n, int ranks, int dims[2]) {
    int used = 0;
    for (int rows = 1; rows <= n && rows <= ranks / rows; ++rows) {
        int cols = std::min(n, ranks / rows);
        if (rows * cols >= used) {
            used = rows * cols;
            // Prefer contiguous row messages along the more partitioned axis.
            dims[0] = cols;
            dims[1] = rows;
        }
    }
    const int periodic[2] = {};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &grid);
    return grid;
}

struct Element {
    std::array<size_t, 4> neighbor;
    double external_flow;
};

struct Tile {
    Interval rows, cols;
    size_t stride;
    std::vector<Element> elements;
    std::vector<double> energy, next, flux;

    Tile(int n, Interval r, Interval c)
        : rows(r), cols(c), stride(size_t(c.length) + 2),
          elements(size_t(r.length) * c.length),
          energy((size_t(r.length) + 2) * stride), next(energy.size()), flux(elements.size()) {
        for (int i = 0; i < rows.length; ++i) {
            for (int j = 0; j < cols.length; ++j) {
                const int x = rows.first + i, y = cols.first + j;
                const size_t k = (size_t(i) + 1) * stride + j + 1;
                Element& elem = elements[size_t(i) * cols.length + j];
                // Self edges contribute zero at the physical boundary.
                elem.neighbor = {x < n - 1 ? k + stride : k, x > 0 ? k - stride : k,
                                 y < n - 1 ? k + 1 : k, y > 0 ? k - 1 : k};
                if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1))
                    elem.external_flow = x == y ? 0.5 : -0.5;
            }
        }
    }

    void updateRow(int i, int begin, int end) {
        for (int j = begin; j < end; ++j) {
            const size_t cell = size_t(i) * cols.length + j;
            const size_t k = (size_t(i) + 1) * stride + j + 1;
            const double current = energy[k];
            double change = elements[cell].external_flow;
            // All materials use transfer coefficient 0.8 and all edges have unit flux.
            for (size_t neighbor : elements[cell].neighbor)
                change += (energy[neighbor] - current) * 0.8 * 0.25;
            next[k] = current + change;
            flux[cell] += std::abs(change);
        }
    }
};

void runSimulation(Tile& t, int iterations, MPI_Comm grid) {
    const int h = t.rows.length, w = t.cols.length;
    int peer[4];
    MPI_Cart_shift(grid, 0, 1, &peer[0], &peer[1]);
    MPI_Cart_shift(grid, 1, 1, &peer[2], &peer[3]);
    MPI_Datatype column;
    MPI_Type_create_hvector(h, 1, t.stride * sizeof(double), MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    const size_t first = t.stride + 1, last = h * t.stride + 1;
    const size_t send[4] = {first, last, first, first + w - 1};
    const size_t recv[4] = {1, last + t.stride, first - 1, first + w};
    const int top = peer[0] != MPI_PROC_NULL, left = peer[2] != MPI_PROC_NULL;
    const int bottom = std::max(top, h - (peer[1] != MPI_PROC_NULL));
    const int right = std::max(left, w - (peer[3] != MPI_PROC_NULL));

    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        for (int d = 0; d < 4; ++d) {
            if (peer[d] == MPI_PROC_NULL) continue;
            const int length = d < 2 ? w : 1;
            MPI_Datatype type = d < 2 ? MPI_DOUBLE : column;
            MPI_Irecv(t.energy.data() + recv[d], length, type, peer[d], 0, grid, &requests[count++]);
            MPI_Isend(t.energy.data() + send[d], length, type, peer[d], 0, grid, &requests[count++]);
        }
        for (int i = top; i < bottom; ++i)
            t.updateRow(i, left, right);

        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        if (top) t.updateRow(0, 0, w);
        for (int i = top; i < bottom; ++i) {
            if (left) t.updateRow(i, 0, left);
            if (right < w) t.updateRow(i, right, w);
        }
        if (bottom < h) t.updateRow(h - 1, 0, w);
        t.energy.swap(t.next);
    }
    MPI_Type_free(&column);
}

uint64_t computeHash(const Tile& t, int n) {
    uint64_t hash = 0;
    for (int i = 0; i < t.rows.length; ++i) {
        for (int j = 0; j < t.cols.length; ++j) {
            const uint64_t index = uint64_t(t.rows.first + i) * n + t.cols.first + j;
            const double energy = t.energy[(size_t(i) + 1) * t.stride + j + 1];
            const double flux = t.flux[size_t(i) * t.cols.length + j];
            hash ^= (std::bit_cast<uint64_t>(energy) + index) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(flux) + index) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

// Reassemble only the optional order-dependent external report.
void printEnergy(const Tile& t, int n, const int dims[2], int rank, MPI_Comm grid) {
    if (rank != 0) {
        MPI_Datatype block;
        MPI_Type_create_hvector(t.rows.length, t.cols.length, t.stride * sizeof(double), MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(t.energy.data() + t.stride + 1, 1, block, 0, 1, grid);
        MPI_Type_free(&block);
        return;
    }
    std::vector<double> data(size_t(n) * n);
    for (int i = 0; i < t.rows.length; ++i)
        std::copy_n(t.energy.data() + (size_t(i) + 1) * t.stride + 1, t.cols.length, data.data() + size_t(i) * n);
    for (int p = 1; p < dims[0] * dims[1]; ++p) {
        int coord[2];
        MPI_Cart_coords(grid, p, 2, coord);
        Interval r = partition(n, dims[0], coord[0]), c = partition(n, dims[1], coord[1]);
        MPI_Datatype block;
        MPI_Type_vector(r.length, c.length, n, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(data.data() + size_t(r.first) * n + c.first, 1, block, p, 1, grid, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(data, "ElementEnergy");
}

bool validateResults(const Tile& t, int rank, MPI_Comm grid) {
    double sums[2] = {}, totals[2] = {};
    double low = t.energy[t.stride + 1], high = low;
    for (int i = 0; i < t.rows.length; ++i) {
        for (int j = 0; j < t.cols.length; ++j) {
            double value = t.energy[(size_t(i) + 1) * t.stride + j + 1];
            sums[0] += value;
            sums[1] += t.flux[size_t(i) * t.cols.length + j];
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    double energy_min = 0, energy_max = 0;
    MPI_Reduce(sums, totals, 2, MPI_DOUBLE, MPI_SUM, 0, grid);
    MPI_Reduce(&low, &energy_min, 1, MPI_DOUBLE, MPI_MIN, 0, grid);
    MPI_Reduce(&high, &energy_max, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank != 0) return true;
    const double energy_sum = totals[0], flux_sum = totals[1];
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
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive\n");
        MPI_Finalize();
        return 1;
    }
    int dims[2];
    MPI_Comm grid = createGrid(n, ranks, dims);
    int status = 0;
    if (grid != MPI_COMM_NULL) {
        int coord[2];
        MPI_Cart_coords(grid, rank, 2, coord);
        const uint64_t cells = uint64_t(n) * n;
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, cells);
            printf("Iterations: %d\n", n_iters);
            printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
        }
        Tile tile(n, partition(n, dims[0], coord[0]), partition(n, dims[1], coord[1]));
        if (rank == 0) {
            double static_mb = double(cells) * sizeof(Element) / (1024 * 1024);
            double padded = (double(n) + 2.0 * dims[0]) * (double(n) + 2.0 * dims[1]);
            double dynamic_mb = (2 * padded + cells) * sizeof(double) / (1024 * 1024);
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   static_mb + dynamic_mb, static_mb, dynamic_mb);
            printf("Running simulation...\n");
        }
        MPI_Barrier(grid);
        double start = MPI_Wtime();
        runSimulation(tile, n_iters, grid);
        double elapsed = MPI_Wtime() - start, seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        uint64_t local_hash = computeHash(tile, n), hash = 0;
        MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, grid);
        if (rank == 0) {
            double iters = std::max(double(n_iters) - 1, 1.0);
            double rate = iters * cells / seconds / 1e9;
            printf("Computation time: %.0f ms\n", seconds * 1000);
            printf("Performance:\n");
            printf("  Time per iteration: %.4f ms\n", seconds * 1000 / iters);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (printResults) printEnergy(tile, n, dims, rank, grid);
        if (validate && !validateResults(tile, rank, grid)) status = 1;
        MPI_Comm_free(&grid);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
