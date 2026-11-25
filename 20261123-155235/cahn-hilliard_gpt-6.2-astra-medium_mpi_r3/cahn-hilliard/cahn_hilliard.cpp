#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Shape = std::array<int, 3>; // z, y, x (C storage order)

// Minimize exchanged face area, weighting strided transfers more heavily.
Shape processGrid(const Shape& grid, int ranks) {
    const double volume = double(grid[0]) * grid[1] * grid[2];
    for (int n = static_cast<int>(std::min(double(ranks), volume)); n > 0; --n) {
        Shape best{};
        double cost = std::numeric_limits<double>::infinity();
        for (int z = 1; z <= std::min(n, grid[0]); ++z) {
            if (n % z)
                continue;
            for (int y = 1; y <= std::min(n / z, grid[1]); ++y) {
                if ((n / z) % y)
                    continue;
                const int x = n / z / y;
                if (x > grid[2])
                    continue;
                const double area = (z - 1.0) / grid[0] + 1.1 * (y - 1) / grid[1] + 1.5 * (x - 1) / grid[2];
                if (area < cost) {
                    best = {z, y, x};
                    cost = area;
                }
            }
        }
        if (best[0])
            return best;
    }
    return {1, 1, 1};
}

struct Block {
    Shape size, start;
};

Block blockFor(int rank, MPI_Comm cart, const Shape& grid, const Shape& parts) {
    Shape coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    Block b;
    for (int d = 0; d < 3; ++d) {
        const int base = grid[d] / parts[d], extra = grid[d] % parts[d];
        b.size[d] = base + (coords[d] < extra);
        b.start[d] = coords[d] * base + std::min(coords[d], extra);
    }
    return b;
}

MPI_Datatype subarray(const Shape& shape, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, shape.data(), size.data(), start.data(), MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    MPI_Comm cart;
    Shape n, padded, lo, hi;
    std::array<int, 6> peer;
    std::array<MPI_Datatype, 3> face{};
    size_t row, plane;

    Domain(MPI_Comm comm, Shape size) : cart(comm), n(size), padded(n) {
        for (int& extent : padded)
            extent += 2;
        row = padded[2];
        plane = row * padded[1];
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart, d, 1, &peer[2 * d], &peer[2 * d + 1]);
            lo[d] = 1 + (peer[2 * d] != MPI_PROC_NULL);
            hi[d] = n[d] + 1 - (peer[2 * d + 1] != MPI_PROC_NULL);
            hi[d] = std::max(lo[d], hi[d]);
            if (peer[2 * d] != MPI_PROC_NULL || peer[2 * d + 1] != MPI_PROC_NULL) {
                Shape extent = n;
                extent[d] = 1;
                face[d] = subarray(padded, extent, {0, 0, 0});
            }
        }
    }

    ~Domain() {
        for (int d = 0; d < 3; ++d)
            if (peer[2 * d] != MPI_PROC_NULL || peer[2 * d + 1] != MPI_PROC_NULL)
                MPI_Type_free(&face[d]);
    }

    size_t at(int z, int y, int x) const { return z * plane + y * row + x; }

    template <bool chemical>
    void stage(std::vector<double>& input, std::vector<double>& output,
               const std::vector<double>& cold) const {
        MPI_Request requests[12];
        int count = 0;
        const std::array<size_t, 3> stride{plane, row, 1};
        for (int side = 0; side < 6; ++side) {
            if (peer[side] == MPI_PROC_NULL)
                continue;
            const int d = side / 2;
            const size_t step = stride[d];
            const size_t first = at(1, 1, 1);
            const size_t send = first + (side % 2 ? (n[d] - 1) * step : 0);
            const size_t recv = side % 2 ? send + step : send - step;
            MPI_Irecv(input.data() + recv, 1, face[d], peer[side], side ^ 1, cart, &requests[count++]);
            MPI_Isend(input.data() + send, 1, face[d], peer[side], side, cart, &requests[count++]);
        }

        // Clamp physical x faces so every row has a branch-free inner loop.
        if (peer[4] == MPI_PROC_NULL || peer[5] == MPI_PROC_NULL) {
            for (int z = 1; z <= n[0]; ++z) {
                for (int y = 1; y <= n[1]; ++y) {
                    double* line = input.data() + at(z, y, 0);
                    if (peer[4] == MPI_PROC_NULL)
                        line[0] = line[1];
                    if (peer[5] == MPI_PROC_NULL)
                        line[n[2] + 1] = line[n[2]];
                }
            }
        }

        auto rows = [&](int z0, int z1, int y0, int y1, int x0, int x1) {
            if (x0 >= x1 || y0 >= y1)
                return;
            for (int z = z0; z < z1; ++z) {
                const size_t back = z == 1 && peer[0] == MPI_PROC_NULL ? 0 : plane;
                const size_t front = z == n[0] && peer[1] == MPI_PROC_NULL ? 0 : plane;
                for (int y = y0; y < y1; ++y) {
                    const size_t down = y == 1 && peer[2] == MPI_PROC_NULL ? 0 : row;
                    const size_t up = y == n[1] && peer[3] == MPI_PROC_NULL ? 0 : row;
                    const size_t end = at(z, y, x1);
                    for (size_t i = at(z, y, x0); i < end; ++i) {
                        const double cv = input[i], twice = 2.0 * cv;
                        const double xx = input[i + 1] + input[i - 1] - twice;
                        const double yy = input[i + up] + input[i - down] - twice;
                        const double zz = input[i + front] + input[i - back] - twice;
                        const double lap = xx + yy + zz; // unit grid spacing
                        if constexpr (chemical) {
                            constexpr double aa = -2.0 / 9.0, bb = aa, ab = -aa;
                            output[i] = 4.5 * ((cv + 1.0) * aa + (cv - 1.0) * bb - 2.0 * cv * ab) +
                                        3.0 * cv + cv * cv * cv - 0.5 * lap;
                        } else {
                            output[i] = cold[i] + 0.01 * lap;
                        }
                    }
                }
            }
        };

        rows(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        if (count == 0)
            return;
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Disjoint strips cover the remaining cells even for one-cell blocks.
        rows(1, lo[0], 1, n[1] + 1, 1, n[2] + 1);
        rows(hi[0], n[0] + 1, 1, n[1] + 1, 1, n[2] + 1);
        rows(lo[0], hi[0], 1, lo[1], 1, n[2] + 1);
        rows(lo[0], hi[0], hi[1], n[1] + 1, 1, n[2] + 1);
        rows(lo[0], hi[0], lo[1], hi[1], 1, lo[2]);
        rows(lo[0], hi[0], lo[1], hi[1], hi[2], n[2] + 1);
    }
};

void report(const Domain& domain, const std::vector<double>& c, int rank, const Shape& grid,
            const Shape& parts) {
    auto owned = subarray(domain.padded, domain.n, {1, 1, 1});
    if (rank == 0) {
        std::vector<double> global(size_t(grid[0]) * grid[1] * grid[2]);
        int ranks;
        MPI_Comm_size(domain.cart, &ranks);
        for (int r = 0; r < ranks; ++r) {
            const Block b = blockFor(r, domain.cart, grid, parts);
            auto dest = subarray(grid, b.size, b.start);
            if (r == 0)
                MPI_Sendrecv(c.data(), 1, owned, 0, 0, global.data(), 1, dest, 0, 0, domain.cart,
                             MPI_STATUS_IGNORE);
            else
                MPI_Recv(global.data(), 1, dest, r, 0, domain.cart, MPI_STATUS_IGNORE);
            MPI_Type_free(&dest);
        }
        print_results(global, "Concentration");
    } else {
        MPI_Send(c.data(), 1, owned, 0, 0, domain.cart);
    }
    MPI_Type_free(&owned);
}

int validateResult(const Domain& domain, const std::vector<double>& c, int rank) {
    // Reduce -min, max, and a nonfinite flag together.
    double local[3]{-INFINITY, -INFINITY, 0}, global[3];
    for (int z = 1; z <= domain.n[0]; ++z) {
        for (int y = 1; y <= domain.n[1]; ++y) {
            for (int x = 1; x <= domain.n[2]; ++x) {
                double v = c[domain.at(z, y, x)];
                local[0] = std::max(local[0], -v);
                local[1] = std::max(local[1], v);
                if (!std::isfinite(v))
                    local[2] = 1;
            }
        }
    }
    MPI_Reduce(local, global, 3, MPI_DOUBLE, MPI_MAX, 0, domain.cart);
    int status = 0;
    if (rank == 0) {
        printf("Validating result...\n");
        if (global[2]) {
            printf("Validation failed: found NaN or Inf value\n");
            status = 1;
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", -global[0], global[1]);
            if (global[0] > 10 || global[1] > 10) {
                printf("Validation failed: values out of expected range\n");
                status = 1;
            }
        }
        printf("Validation: %s\n", status ? "FAILED" : "PASSED");
    }
    return status;
}

void printUsage(const char* progName) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
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
            if (rank == 0)
                printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0)
                printf("Unknown option: %s\n", argv[i]);
            if (rank == 0)
                printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0)
        ny = nx;
    if (nz == 0)
        nz = nx;

    if (nx == 0 || ny == 0 || nz == 0 ||
        std::max({nx, ny, nz}) > size_t(std::numeric_limits<int>::max() - 2)) {
        if (rank == 0)
            std::fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }

    Shape grid{int(nz), int(ny), int(nx)};
    Shape parts = processGrid(grid, ranks);
    int periods[3]{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods, 0, &cart);
    int status = 0;
    if (cart != MPI_COMM_NULL) {
        const Block b = blockFor(rank, cart, grid, parts);
        Domain domain(cart, b.size);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        const size_t vol = nx * ny * nz;
        const size_t storage = domain.padded[0] * domain.plane;
        std::vector<double> cold(storage), cnew(storage), mu(storage);
        for (int z = 0; z < b.size[0]; ++z) {
            for (int y = 0; y < b.size[1]; ++y) {
                size_t id = ((b.start[0] + z) * ny + b.start[1] + y) * nx + b.start[2];
                for (int x = 0; x < b.size[2]; ++x, ++id) {
                    double pseudo = ((id + 1) * 1299709 % vol) / static_cast<double>(vol);
                    cold[domain.at(z + 1, y + 1, x + 1)] = -1.0 + 2.0 * pseudo;
                }
            }
        }

        if (rank == 0)
            printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(cart);
        double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            domain.stage<true>(cold, mu, cold);
            domain.stage<false>(mu, cnew, cold);
            cold.swap(cnew);
        }
        double localTime = MPI_Wtime() - start, elapsed;
        MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", long(elapsed * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(vol) * iterations / elapsed / 1e6);
        }
        if (printResults)
            report(domain, cold, rank, grid, parts);
        if (validate) {
            status = validateResult(domain, cold, rank);
        }
    }
    if (cart != MPI_COMM_NULL)
        MPI_Comm_free(&cart);
    if (validate)
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
