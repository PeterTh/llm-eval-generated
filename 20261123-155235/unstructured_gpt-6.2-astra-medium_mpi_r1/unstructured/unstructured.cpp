#include <mpi.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

struct Block {
    int x, y, rows, cols;
};

Block blockFor(MPI_Comm grid, const int dims[2], int n, int rank) {
    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);
    // Put the remainder in the first blocks along each axis.
    auto begin = [n](int c, int parts) { return c * (n / parts) + std::min(c, n % parts); };
    int x = begin(coords[0], dims[0]), y = begin(coords[1], dims[1]);
    return {x, y, begin(coords[0] + 1, dims[0]) - x, begin(coords[1] + 1, dims[1]) - y};
}

// Signed offsets address neighbors in the local energy buffer, including its halo.
// A missing physical edge points to the element itself and contributes zero flux.
struct Links {
    int down, up, right, left;
    double source;
};

struct Mesh : Block {
    int pitch;
    std::vector<Links> links;
    std::vector<double> energy, next, flux;

    Mesh(Block block, int n) : Block(block), pitch(cols + 2), links(size_t(rows) * cols),
        energy((size_t(rows) + 2) * pitch), next(energy.size()), flux(links.size()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int gx = x + i, gy = y + j;
                double source = 0.0;
                if ((gx == 0 || gx == n - 1) && (gy == 0 || gy == n - 1))
                    source = gx == gy ? 0.5 : -0.5;
                links[size_t(i) * cols + j] = {gx < n - 1 ? pitch : 0, gx ? -pitch : 0,
                                               gy < n - 1 ? 1 : 0, gy ? -1 : 0, source};
            }
        }
    }

    size_t index(int i, int j) const { return size_t(i + 1) * pitch + j + 1; }

    void update(int i, int first, int end) {
        for (int j = first; j < end; ++j) {
            size_t local = size_t(i) * cols + j, k = index(i, j);
            const Links& neighbors = links[local];
            const double* e = &energy[k];
            // All materials have transfer coefficient 0.8 and every edge has unit weight.
            double f = neighbors.source;
            f += (e[neighbors.down] - *e) * 0.8 * 0.25;
            f += (e[neighbors.up] - *e) * 0.8 * 0.25;
            f += (e[neighbors.right] - *e) * 0.8 * 0.25;
            f += (e[neighbors.left] - *e) * 0.8 * 0.25;
            next[k] = *e + f;
            flux[local] += std::abs(f);
        }
    }

    uint64_t resultHash(int n) const {
        uint64_t result = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                uint64_t global = uint64_t(x + i) * n + y + j;
                result ^= (std::bit_cast<uint64_t>(energy[index(i,j)]) + global) * 0x9e3779b97f4a7c15ULL;
                result ^= (std::bit_cast<uint64_t>(flux[size_t(i) * cols + j]) + global) * 0xbf58476d1ce4e5b9ULL;
            }
        }
        return result;
    }
};

void runSimulation(Mesh& m, MPI_Comm grid, MPI_Datatype column, int iters) {
    int top, bottom, left, right;
    MPI_Cart_shift(grid, 0, 1, &top, &bottom);
    MPI_Cart_shift(grid, 1, 1, &left, &right);
    int firstRow = top != MPI_PROC_NULL, endRow = m.rows - (bottom != MPI_PROC_NULL);
    int firstCol = left != MPI_PROC_NULL, endCol = m.cols - (right != MPI_PROC_NULL);
    for (int iter = 0; iter < iters; ++iter) {
        MPI_Request requests[8];
        int count = 0;
        auto exchange = [&](int peer, size_t send, size_t recv, int len, MPI_Datatype type) {
            if (peer == MPI_PROC_NULL) return;
            MPI_Irecv(m.energy.data() + recv, len, type, peer, 0, grid, &requests[count++]);
            MPI_Isend(m.energy.data() + send, len, type, peer, 0, grid, &requests[count++]);
        };
        exchange(top, m.index(0,0), m.index(-1,0), m.cols, MPI_DOUBLE);
        exchange(bottom, m.index(m.rows-1,0), m.index(m.rows,0), m.cols, MPI_DOUBLE);
        exchange(left, m.index(0,0), m.index(0,-1), 1, column);
        exchange(right, m.index(0,m.cols-1), m.index(0,m.cols), 1, column);

        for (int i = firstRow; i < endRow; ++i) m.update(i, firstCol, endCol);
        if (count) {
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            if (firstRow) m.update(0, 0, m.cols);
            if (endRow < m.rows && endRow >= firstRow) m.update(endRow, 0, m.cols);
            for (int i = firstRow; i < endRow; ++i) {
                if (firstCol) m.update(i, 0, 1);
                if (endCol < m.cols && endCol >= firstCol) m.update(i, endCol, m.cols);
            }
        }
        m.energy.swap(m.next);
    }
}

// Only the optional external report needs all energies in serial traversal order.
void reportEnergy(const Mesh& m, MPI_Comm grid, const int dims[2], int n, int rank, int active) {
    std::vector<double> global;
    if (rank == 0) {
        global.resize(size_t(n) * n);
        for (int i = 0; i < m.rows; ++i)
            std::copy_n(&m.energy[m.index(i,0)], m.cols, &global[size_t(i) * n]);
        for (int peer = 1; peer < active; ++peer) {
            Block b = blockFor(grid, dims, n, peer);
            MPI_Datatype tile;
            MPI_Type_vector(b.rows, b.cols, n, MPI_DOUBLE, &tile);
            MPI_Type_commit(&tile);
            MPI_Recv(&global[size_t(b.x) * n + b.y], 1, tile, peer, 1, grid, MPI_STATUS_IGNORE);
            MPI_Type_free(&tile);
        }
        print_results(global, "ElementEnergy");
    } else {
        MPI_Datatype tile;
        MPI_Type_vector(m.rows, m.cols, m.pitch, MPI_DOUBLE, &tile);
        MPI_Type_commit(&tile);
        MPI_Send(&m.energy[m.index(0,0)], 1, tile, 0, 1, grid);
        MPI_Type_free(&tile);
    }
}

bool validateResults(const Mesh& m, MPI_Comm grid, int rank) {
    double sums[2] = {}, lo = std::numeric_limits<double>::max(), hi = -lo;
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            double e = m.energy[m.index(i,j)];
            sums[0] += e;
            sums[1] += m.flux[size_t(i) * m.cols + j];
            lo = std::min(e, lo);
            hi = std::max(e, hi);
        }
    }
    double total[2], minimum, maximum;
    MPI_Reduce(sums, total, 2, MPI_DOUBLE, MPI_SUM, 0, grid);
    MPI_Reduce(&lo, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, grid);
    MPI_Reduce(&hi, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
    if (rank != 0) return true;
    printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n", total[0], total[1]);
    printf("  Energy range: [%.6f, %.6f]\n", minimum, maximum);
    if (!std::isfinite(total[0])) {
        printf("  ERROR: Energy sum is not finite\n");
        return false;
    }
    if (std::abs(total[0]) > 1e-8)
        printf("  WARNING: Energy sum diverged from 0 (expected conservation)\n");
    if (!std::isfinite(total[1])) {
        printf("  ERROR: Flux sum is not finite\n");
        return false;
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    
    if (n_elems_root < 1) {
        if (rank == 0) fprintf(stderr, "Grid size must be positive\n");
        MPI_Finalize();
        return 1;
    }

    const uint64_t n_elems = uint64_t(n_elems_root) * n_elems_root;
    int active = std::min<uint64_t>(size, n_elems);
    int dims[2];
    do {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(active, 2, dims);
        if (std::max(dims[0], dims[1]) <= n_elems_root) break;
        --active;
    } while (true);
    int periodic[2] = {};
    MPI_Comm grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &grid);
    int valid = 1;
    if (grid != MPI_COMM_NULL) {
        if (rank == 0) {
            printf("Unstructured Mesh Energy Transfer Benchmark\n");
            printf("============================================\n");
            printf("Grid size: %d x %d = %" PRIu64 " elements\n", n_elems_root, n_elems_root, n_elems);
            printf("Iterations: %d\nValidation: %s\n\n", n_iters, validate ? "enabled" : "disabled");
            printf("Building unstructured mesh...\n");
        }
        Mesh mesh(blockFor(grid, dims, n_elems_root, rank), n_elems_root);
        uint64_t local_bytes[] = {mesh.links.size() * sizeof(Links),
                                  (mesh.energy.size() * 2 + mesh.flux.size()) * sizeof(double)};
        uint64_t bytes[2];
        MPI_Reduce(local_bytes, bytes, 2, MPI_UINT64_T, MPI_SUM, 0, grid);
        if (rank == 0) {
            constexpr double mb = 1024.0 * 1024.0;
            printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
                   (bytes[0] + bytes[1]) / mb, bytes[0] / mb, bytes[1] / mb);
            printf("Running simulation...\n");
        }

        MPI_Datatype column;
        MPI_Type_vector(mesh.rows, 1, mesh.pitch, MPI_DOUBLE, &column);
        MPI_Type_commit(&column);
        MPI_Barrier(grid);
        double start = MPI_Wtime();
        runSimulation(mesh, grid, column, n_iters);
        double local_ms = (MPI_Wtime() - start) * 1000.0, ms;
        MPI_Reduce(&local_ms, &ms, 1, MPI_DOUBLE, MPI_MAX, 0, grid);
        MPI_Type_free(&column);
        uint64_t local_hash = mesh.resultHash(n_elems_root), result_hash;
        MPI_Reduce(&local_hash, &result_hash, 1, MPI_UINT64_T, MPI_BXOR, 0, grid);
        if (rank == 0) {
            const double measured = std::max(1.0, double(n_iters) - 1);
            const double rate = measured * n_elems / ms / 1e6;
            printf("Computation time: %.0f ms\nPerformance:\n", ms);
            printf("  Time per iteration: %.4f ms\n", ms / measured);
            printf("  Elements/sec: %.4f GigaElements/s\n", rate);
            printf("  Performance: %.4f GFLOPS\n", rate * 22.0);
            printf("  Result hash: %016" PRIX64 "\n\n", result_hash);
        }
        if (printResults) reportEnergy(mesh, grid, dims, n_elems_root, rank, active);
        if (validate) valid = validateResults(mesh, grid, rank);
        MPI_Comm_free(&grid);
    }
    if (validate) MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
