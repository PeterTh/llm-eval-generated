#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

// MPI Cartesian coordinates and array extents use storage order (z, y, x).
using Shape = std::array<int, 3>;

Shape processGrid(const Shape &global, int ranks) {
    Shape best{1, 1, 1};
    int used = 0;
    double cost = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= std::min(global[0], ranks); ++z) {
        for (int y = 1; y <= std::min(global[1], ranks / z); ++y) {
            int x = std::min(global[2], ranks / z / y);
            int count = z * y * x;
            // Minimize total internal face area, preferring contiguous
            // transfers and longer vector loops for otherwise similar shapes.
            double area = double(z - 1) * global[1] * global[2] +
                          double(y - 1) * global[0] * global[2] * 1.05 +
                          double(x - 1) * global[0] * global[1] * 1.5;
            if (count > used || (count == used && area < cost)) {
                best = {z, y, x};
                used = count;
                cost = area;
            }
        }
    }
    return best;
}

struct Block {
    Shape size, start;
    Block(const Shape &global, const Shape &grid, const Shape &coords) {
        for (int d = 0; d < 3; ++d) {
            int base = global[d] / grid[d], extra = global[d] % grid[d];
            size[d] = base + (coords[d] < extra);
            start[d] = coords[d] * base + std::min(coords[d], extra);
        }
    }
};

MPI_Datatype subarray(const Shape &full, const Shape &size, const Shape &start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, full.data(), size.data(), start.data(), MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    MPI_Comm comm;
    Block block;
    Shape full;
    size_t row, plane;
    int neighbor[3][2];
    MPI_Datatype face[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};

    Domain(MPI_Comm cart, const Block &b) : comm(cart), block(b), full(b.size) {
        for (int &n : full)
            n += 2;
        row = full[2];
        plane = row * full[1];
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(comm, d, 1, &neighbor[d][0], &neighbor[d][1]);
            if (neighbor[d][0] != MPI_PROC_NULL || neighbor[d][1] != MPI_PROC_NULL) {
                Shape extent = block.size;
                extent[d] = 1;
                face[d] = subarray(full, extent, {0, 0, 0});
            }
        }
    }

    ~Domain() {
        for (auto &type : face)
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
    }

    size_t index(int z, int y, int x) const { return z * plane + y * row + x; }

    // Only face interiors are needed by a seven-point stencil.
    template <class Kernel> void stencil(std::vector<double> &field, Kernel kernel) const {
        const Shape &n = block.size;
        const size_t strides[] = {plane, row, 1};
        Shape lo{1, 1, 1}, hi{n[0] + 1, n[1] + 1, n[2] + 1};
        MPI_Request req[12];
        int count = 0;
        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                int peer = neighbor[d][side];
                if (peer == MPI_PROC_NULL) continue;
                size_t edge = index(1, 1, 1) + (side ? n[d] - 1 : 0) * strides[d];
                size_t ghost = side ? edge + strides[d] : edge - strides[d];
                MPI_Irecv(field.data() + ghost, 1, face[d], peer, 2 * d + 1 - side, comm, &req[count++]);
                MPI_Isend(field.data() + edge, 1, face[d], peer, 2 * d + side, comm, &req[count++]);
                if (side)
                    --hi[d];
                else
                    ++lo[d];
            }
            hi[d] = std::max(lo[d], hi[d]);
        }

        // Clamp physical boundaries by reusing the edge row or plane.
        // In x, fill the two ghost cells to keep the inner loop branch-free.
        auto apply = [&](int z0, int z1, int y0, int y1, int x0, int x1) {
            for (int z = z0; z < z1; ++z) {
                size_t zm = z == 1 && neighbor[0][0] == MPI_PROC_NULL ? 0 : plane;
                size_t zp = z == n[0] && neighbor[0][1] == MPI_PROC_NULL ? 0 : plane;
                for (int y = y0; y < y1; ++y) {
                    size_t ym = y == 1 && neighbor[1][0] == MPI_PROC_NULL ? 0 : row;
                    size_t yp = y == n[1] && neighbor[1][1] == MPI_PROC_NULL ? 0 : row;
                    double *p = field.data() + index(z, y, 1);
                    if (neighbor[2][0] == MPI_PROC_NULL) p[-1] = p[0];
                    if (neighbor[2][1] == MPI_PROC_NULL) p[n[2]] = p[n[2] - 1];
                    size_t end = index(z, y, x1);
                    for (size_t i = index(z, y, x0); i < end; ++i) {
                        double twice = 2.0 * field[i];
                        double xx = (field[i + 1] + field[i - 1]) - twice;
                        double yy = (field[i + yp] + field[i - ym]) - twice;
                        double zz = (field[i + zp] + field[i - zm]) - twice;
                        kernel(i, xx + yy + zz);
                    }
                }
            }
        };
        apply(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        if (!count) return;
        MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
        // Disjoint boundary boxes, including blocks only one cell thick.
        apply(1, lo[0], 1, n[1] + 1, 1, n[2] + 1);
        apply(hi[0], n[0] + 1, 1, n[1] + 1, 1, n[2] + 1);
        apply(lo[0], hi[0], 1, lo[1], 1, n[2] + 1);
        apply(lo[0], hi[0], hi[1], n[1] + 1, 1, n[2] + 1);
        if (lo[2] > 1) apply(lo[0], hi[0], lo[1], hi[1], 1, lo[2]);
        if (hi[2] <= n[2]) apply(lo[0], hi[0], lo[1], hi[1], hi[2], n[2] + 1);
    }
};

void outputField(const Domain &d, const Shape &global, const Shape &grid, const std::vector<double> &field,
                 int rank, int ranks) {
    MPI_Datatype interior = subarray(d.full, d.block.size, {1, 1, 1});
    MPI_Request send;
    MPI_Isend(field.data(), 1, interior, 0, 0, d.comm, &send);
    if (rank == 0) {
        std::vector<double> result(size_t(global[0]) * global[1] * global[2]);
        for (int src = 0; src < ranks; ++src) {
            Shape coords;
            MPI_Cart_coords(d.comm, src, 3, coords.data());
            Block b(global, grid, coords);
            MPI_Datatype dest = subarray(global, b.size, b.start);
            MPI_Recv(result.data(), 1, dest, src, 0, d.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&dest);
        }
        print_results(result, "Concentration");
    }
    MPI_Wait(&send, MPI_STATUS_IGNORE);
    MPI_Type_free(&interior);
}

bool validateField(const Domain &d, const std::vector<double> &c, int rank) {
    double local[] = {INFINITY, INFINITY, 1.0}, result[3];
    for (int z = 1; z <= d.block.size[0]; ++z)
        for (int y = 1; y <= d.block.size[1]; ++y)
            for (int x = 1; x <= d.block.size[2]; ++x) {
                double v = c[d.index(z, y, x)];
                local[0] = std::min(local[0], v);
                local[1] = std::min(local[1], -v);
                if (!std::isfinite(v)) local[2] = 0;
            }
    // Negate maxima so all three statistics share one reduction.
    MPI_Allreduce(local, result, 3, MPI_DOUBLE, MPI_MIN, d.comm);
    bool valid = result[2] && result[0] >= -10 && result[1] >= -10;
    if (rank == 0) {
        printf("Validating result...\n");
        if (!result[2])
            printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", result[0], -result[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char **argv, int rank, int ranks) {
    int nx = 64;
    int ny = 0;
    int nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx <= 0 || ny <= 0 || nz <= 0 || std::max({nx, ny, nz}) > std::numeric_limits<int>::max() - 2) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }

    Shape global{nz, ny, nx}, grid = processGrid(global, ranks);
    Shape periodic{}, coords;
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, grid.data(), periodic.data(), 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    Block b(global, grid, coords);
    int status = 0;
    {
        Domain d(cart, b);
        const size_t volume = size_t(nx) * ny * nz;
        std::vector<double> c(d.plane * d.full[0]), mu(c.size());
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %d x %d x %d\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (int z = 0; z < b.size[0]; ++z)
            for (int y = 0; y < b.size[1]; ++y) {
                size_t base = (size_t(b.start[0] + z) * ny + b.start[1] + y) * nx + b.start[2];
                for (int x = 0; x < b.size[2]; ++x) {
                    double pseudo = ((base + x + 1) * 1299709 % volume) / double(volume);
                    c[d.index(z + 1, y + 1, x + 1)] = -1.0 + 2.0 * pseudo;
                }
            }

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(cart);
        double start = MPI_Wtime();
        constexpr double e_AA = -2.0 / 9.0, e_BB = -2.0 / 9.0, e_AB = 2.0 / 9.0;
        for (int t = 0; t < iterations; ++t) {
            d.stencil(c, [&](size_t i, double lap) {
                double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv +
                        cv * cv * cv - 0.5 * lap;
            });
            // All spatial dependence now resides in mu, so c can update in place.
            d.stencil(mu, [&](size_t i, double lap) { c[i] += 0.01 * lap; });
        }
        double elapsed = MPI_Wtime() - start, duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / duration / 1e6);
        }
        if (printResults) outputField(d, global, grid, c, rank, grid[0] * grid[1] * grid[2]);
        if (validate && !validateField(d, c, rank)) status = 1;
    }
    MPI_Comm_free(&cart);
    return status;
}

} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
