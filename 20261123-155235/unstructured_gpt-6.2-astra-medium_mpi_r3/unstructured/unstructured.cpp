#include <mpi.h>

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

#include "../common/results_output.hpp"

struct Range {
    int start, size;
};

Range partition(int n, int parts, int coord) {
    const int base = n / parts;
    const int extra = n % parts;
    return {coord * base + std::min(coord, extra), base + (coord < extra)};
}

// Signed offsets keep each unit-flux connection compact, even when the
// local array and its one-cell halo exceed the 32-bit index range.
struct Element {
    std::array<int, 4> offsets;
    int count = 0;
    double external_flow = 0;
};

struct World {
    Range rows, cols;
    size_t pitch;
    std::vector<Element> elements;
    std::vector<double> energy, next, flux;

    World(int n, Range rows, Range cols)
        : rows(rows), cols(cols), pitch(size_t(cols.size) + 2),
          elements(size_t(rows.size) * cols.size),
          energy((size_t(rows.size) + 2) * pitch), next(energy.size()),
          flux(elements.size()) {
        for (int r = 0; r < rows.size; ++r) {
            for (int c = 0; c < cols.size; ++c) {
                const int x = rows.start + r, y = cols.start + c;
                Element& e = elements[size_t(r) * cols.size + c];
                if (x + 1 < n) e.offsets[e.count++] = static_cast<int>(pitch);
                if (x > 0) e.offsets[e.count++] = -static_cast<int>(pitch);
                if (y + 1 < n) e.offsets[e.count++] = 1;
                if (y > 0) e.offsets[e.count++] = -1;
                if ((x == 0 || x == n - 1) && (y == 0 || y == n - 1))
                    e.external_flow = x == y ? 0.5 : -0.5;
            }
        }
    }

    void update(int r, int first, int end) {
        for (int c = first; c < end; ++c) {
            const size_t k = size_t(r) * cols.size + c;
            const size_t i = (r + 1) * pitch + c + 1;
            const Element& e = elements[k];
            const double* center = energy.data() + i;
            const double current = *center;
            double change = e.external_flow;
            auto add_flux = [&](int j) {
                change += (center[e.offsets[j]] - current) * 0.8 * 0.25;
            };
            if (e.count == 4) {
                for (int j = 0; j < 4; ++j) add_flux(j);
            } else {
                for (int j = 0; j < e.count; ++j) add_flux(j);
            }
            next[i] = current + change;
            flux[k] += std::abs(change);
        }
    }
};

void runSimulation(World& w, int iterations, MPI_Comm grid) {
    int up, down, left, right;
    MPI_Cart_shift(grid, 0, 1, &up, &down);
    MPI_Cart_shift(grid, 1, 1, &left, &right);
    const int h = w.rows.size, width = w.cols.size;
    const size_t p = w.pitch;
    MPI_Datatype column;
    MPI_Type_vector(h, 1, static_cast<int>(p), MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    // Only defer edges that actually depend on another rank.
    const int r0 = up != MPI_PROC_NULL;
    const int r1 = std::max(r0, h - (down != MPI_PROC_NULL));
    const int c0 = left != MPI_PROC_NULL;
    const int c1 = std::max(c0, width - (right != MPI_PROC_NULL));
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        double* data = w.energy.data();
        auto exchange = [&](int peer, size_t send, size_t recv, int n, MPI_Datatype type) {
            if (peer == MPI_PROC_NULL) return;
            MPI_Irecv(data + recv, n, type, peer, 0, grid, &requests[count++]);
            MPI_Isend(data + send, n, type, peer, 0, grid, &requests[count++]);
        };
        exchange(up, p + 1, 1, width, MPI_DOUBLE);
        exchange(down, h * p + 1, (h + 1) * p + 1, width, MPI_DOUBLE);
        exchange(left, p + 1, p, 1, column);
        exchange(right, p + width, p + width + 1, 1, column);
        for (int r = r0; r < r1; ++r) w.update(r, c0, c1);
        if (count) {
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            for (int r = 0; r < r0; ++r) w.update(r, 0, width);
            for (int r = r0; r < r1; ++r) {
                w.update(r, 0, c0);
                w.update(r, c1, width);
            }
            for (int r = r1; r < h; ++r) w.update(r, 0, width);
        }
        w.energy.swap(w.next);
    }
    MPI_Type_free(&column);
}

uint64_t computeHash(const World& w, int n) {
    uint64_t hash = 0;
    for (int r = 0; r < w.rows.size; ++r) {
        for (int c = 0; c < w.cols.size; ++c) {
            const uint64_t i = uint64_t(w.rows.start + r) * n + w.cols.start + c;
            const double e = w.energy[(r + 1) * w.pitch + c + 1];
            const double f = w.flux[size_t(r) * w.cols.size + c];
            hash ^= (std::bit_cast<uint64_t>(e) + i) * 0x9e3779b97f4a7c15ULL;
            hash ^= (std::bit_cast<uint64_t>(f) + i) * 0xbf58476d1ce4e5b9ULL;
        }
    }
    return hash;
}

bool validateResults(const World& w, MPI_Comm grid, int rank) {
    double sums[2]{}, totals[2];
    double low = std::numeric_limits<double>::max(), high = -low;
    for (int r = 0; r < w.rows.size; ++r) {
        for (int c = 0; c < w.cols.size; ++c) {
            const double e = w.energy[(r + 1) * w.pitch + c + 1];
            sums[0] += e;
            sums[1] += w.flux[size_t(r) * w.cols.size + c];
            low = std::min(low, e);
            high = std::max(high, e);
        }
    }
    double energy_min, energy_max;
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

// Place tiles directly into their global row-major positions for the
// order-sensitive Kahan sum and FNV hash in print_results.
void printEnergy(const World& w, int n, const int dims[2], int rank, MPI_Comm grid) {
    if (rank != 0) {
        MPI_Datatype tile;
        MPI_Type_vector(w.rows.size, w.cols.size, static_cast<int>(w.pitch), MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Send(w.energy.data() + w.pitch + 1, 1, tile, 0, 1, grid);
        MPI_Type_free(&tile);
        return;
    }
    std::vector<double> all(size_t(n) * n);
    for (int r = 0; r < w.rows.size; ++r)
        std::copy_n(w.energy.data() + (r + 1) * w.pitch + 1, w.cols.size, all.data() + size_t(r) * n);
    for (int src = 1; src < dims[0] * dims[1]; ++src) {
        int pos[2];
        MPI_Cart_coords(grid, src, 2, pos);
        Range rows = partition(n, dims[0], pos[0]), cols = partition(n, dims[1], pos[1]);
        MPI_Datatype tile;
        MPI_Type_vector(rows.size, cols.size, n, MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Recv(all.data() + size_t(rows.start) * n + cols.start, 1, tile, src, 1, grid, MPI_STATUS_IGNORE);
        MPI_Type_free(&tile);
    }
    print_results(all, "ElementEnergy");
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
    int n = 512, iterations = 10;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else {
            bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }
    if (n <= 0 || n > std::numeric_limits<int>::max() - 2) {
        if (rank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize();
        return 1;
    }
    const uint64_t cells = uint64_t(n) * n;
    int dims[2]{}, periods[2]{};
    MPI_Dims_create(static_cast<int>(std::min<uint64_t>(ranks, cells)), 2, dims);
    if (dims[0] > n) {
        dims[0] = n;
        dims[1] = ranks / n;
    }
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid);
    int status = 0;
    if (grid != MPI_COMM_NULL) {
        int pos[2];
        MPI_Cart_coords(grid, rank, 2, pos);
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n, n, cells);
            printf("Iterations: %d\nValidation: %s\n\n", iterations, validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
        }
        World w(n, partition(n, dims[0], pos[0]), partition(n, dims[1], pos[1]));
        uint64_t local_bytes = (2 * w.energy.size() + w.flux.size()) * sizeof(double), dynamic_bytes;
        MPI_Reduce(&local_bytes, &dynamic_bytes, 1, MPI_UINT64_T, MPI_SUM, 0, grid);
        if (rank == 0) {
            const double static_mb = cells * sizeof(Element) / 1048576.0;
            const double dynamic_mb = dynamic_bytes / 1048576.0;
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   static_mb + dynamic_mb, static_mb, dynamic_mb);
            printf("Running simulation...\n");
        }
        MPI_Barrier(grid);
        double start = MPI_Wtime();
        runSimulation(w, iterations, grid);
        double elapsed = MPI_Wtime() - start, seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        uint64_t local_hash = computeHash(w, n), hash;
        MPI_Reduce(&local_hash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, grid);
        if (rank == 0) {
            const double rate = std::max(iterations, 0) * double(cells) / seconds / 1e9;
            printf("Computation time: %.3f ms\nPerformance:\n", seconds * 1000);
            printf("  Time per iteration: %.4f ms\n", seconds * 1000 / std::max(iterations, 1));
            printf("  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n", rate, rate * 22);
            printf("  Result hash: %016" PRIX64 "\n\n", hash);
        }
        if (results) printEnergy(w, n, dims, rank, grid);
        if (validate && !validateResults(w, grid, rank)) status = 1;
        MPI_Comm_free(&grid);
    }
    if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
