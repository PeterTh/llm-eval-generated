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

// MPI subarray dimensions are in storage order (z, y, x).
using Shape = std::array<int, 3>;

Shape processGrid(const Shape& grid, int ranks) {
    Shape best{1, 1, 1};
    int used = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= std::min(grid[0], ranks); ++z) {
        for (int y = 1; y <= std::min(grid[1], ranks / z); ++y) {
            int x = std::min(grid[2], ranks / (z * y));
            int count = x * y * z;
            // Weight cuts through strided faces more heavily than cuts between planes.
            double cost = double(z - 1) * grid[1] * grid[2] + double(y - 1) * grid[0] * grid[2] * 1.1 +
                          double(x - 1) * grid[0] * grid[1] * 1.5;
            if (count > used || (count == used && cost < bestCost)) {
                used = count;
                bestCost = cost;
                best = {z, y, x};
            }
        }
    }
    return best;
}

struct Block {
    Shape start, size;
};

Block blockFor(int rank, MPI_Comm cart, const Shape& dims, const Shape& grid) {
    Shape pos;
    MPI_Cart_coords(cart, rank, 3, pos.data());
    Block block;
    for (int d = 0; d < 3; ++d) {
        int base = grid[d] / dims[d], extra = grid[d] % dims[d];
        block.start[d] = pos[d] * base + std::min(pos[d], extra);
        block.size[d] = base + (pos[d] < extra);
    }
    return block;
}

MPI_Datatype subarray(const Shape& full, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, full.data(), size.data(), start.data(), MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    Shape n, padded;
    size_t row, plane;
    MPI_Comm cart;
    Shape first, end; // half-open box independent of incoming halos
    std::array<bool, 3> upperBoundary;
    struct Face {
        int peer, tag;
        size_t send, recv;
        MPI_Datatype type;
    };
    std::vector<Face> faces;

    Domain(Shape size, MPI_Comm comm)
        : n(size), padded{n[0] + 2, n[1] + 2, n[2] + 2}, row(padded[2]), plane(row * padded[1]), cart(comm) {
        const size_t stride[] = {plane, row, 1};
        for (int d = 0; d < 3; ++d) {
            int peer[2];
            MPI_Cart_shift(cart, d, 1, &peer[0], &peer[1]);
            upperBoundary[d] = peer[1] == MPI_PROC_NULL;
            first[d] = 1 + (peer[0] != MPI_PROC_NULL);
            end[d] = std::max(first[d], n[d] + 1 - (peer[1] != MPI_PROC_NULL));
            for (int side = 0; side < 2; ++side) {
                if (peer[side] == MPI_PROC_NULL) continue;
                Shape face = n;
                face[d] = 1;
                size_t send = plane + row + 1 + (side ? (n[d] - 1) * stride[d] : 0);
                faces.push_back({peer[side], 2 * d + side, send, side ? send + stride[d] : send - stride[d],
                                 subarray(padded, face, {0, 0, 0})});
            }
        }
    }

    ~Domain() {
        for (auto& f : faces)
            MPI_Type_free(&f.type);
    }

    template <class Update>
    void stencil(std::vector<double>& field, std::vector<double>& out, Update update) const {
        MPI_Request requests[12];
        int nr = 0;
        for (const auto& f : faces)
            MPI_Irecv(field.data() + f.recv, 1, f.type, f.peer, f.tag ^ 1, cart, &requests[nr++]);
        for (const auto& f : faces)
            MPI_Isend(field.data() + f.send, 1, f.type, f.peer, f.tag, cart, &requests[nr++]);

        // Replicate the center value across each physical boundary.
        for (int z = 1; z <= n[0]; ++z) {
            double* slice = field.data() + z * plane;
            for (int y = 1; y <= n[1]; ++y) {
                double* line = slice + y * row;
                if (first[2] == 1) line[0] = line[1];
                if (upperBoundary[2]) line[n[2] + 1] = line[n[2]];
            }
            if (first[1] == 1) std::copy_n(slice + row + 1, n[2], slice + 1);
            if (upperBoundary[1]) std::copy_n(slice + n[1] * row + 1, n[2], slice + (n[1] + 1) * row + 1);
        }
        for (int y = 1; y <= n[1]; ++y) {
            double* line = field.data() + y * row + 1;
            if (first[0] == 1) std::copy_n(line + plane, n[2], line);
            if (upperBoundary[0]) std::copy_n(line + n[0] * plane, n[2], line + (n[0] + 1) * plane);
        }

        auto lineUpdate = [&](int z, int y, int begin, int limit) {
            const size_t offset = z * plane + y * row;
            for (int x = begin; x < limit; ++x) {
                const size_t i = offset + x;
                const double v = field[i];
                const double xx = field[i + 1] + field[i - 1] - 2.0 * v;
                const double yy = field[i + row] + field[i - row] - 2.0 * v;
                const double zz = field[i + plane] + field[i - plane] - 2.0 * v;
                out[i] = update(i, v, xx + yy + zz);
            }
        };
        for (int z = first[0]; z < end[0]; ++z)
            for (int y = first[1]; y < end[1]; ++y)
                lineUpdate(z, y, first[2], end[2]);
        if (!nr) return;
        MPI_Waitall(nr, requests, MPI_STATUSES_IGNORE);
        for (int z = 1; z <= n[0]; ++z) {
            for (int y = 1; y <= n[1]; ++y) {
                if (z < first[0] || z >= end[0] || y < first[1] || y >= end[1])
                    lineUpdate(z, y, 1, n[2] + 1);
                else {
                    lineUpdate(z, y, 1, first[2]);
                    lineUpdate(z, y, end[2], n[2] + 1);
                }
            }
        }
    }
};

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

int run(int argc, char** argv, int rank, int ranks) {
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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    const size_t maxExtent = std::numeric_limits<int>::max() - 2;
    if (!nx || nx > maxExtent || ny > maxExtent || nz > maxExtent ||
        nz > std::vector<double>().max_size() / nx / ny) {
        if (rank == 0) fprintf(stderr, "Unsupported grid dimensions\n");
        return 1;
    }
    const size_t volume = nx * ny * nz;
    const Shape grid{int(nz), int(ny), int(nx)};
    const Shape dims = processGrid(grid, ranks);
    const int periods[3]{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;
    const Block block = blockFor(rank, cart, dims, grid);
    const Shape n = block.size;
    const Domain domain(n, cart);
    std::vector<double> concentration(domain.padded[0] * domain.plane);
    std::vector<double> mu(concentration.size());

    if (rank == 0) printf("Initializing concentration field...\n");
    for (int z = 0; z < n[0]; ++z) {
        for (int y = 0; y < n[1]; ++y) {
            size_t global = ((block.start[0] + z) * ny + block.start[1] + y) * nx + block.start[2];
            size_t local = (z + 1) * domain.plane + (y + 1) * domain.row + 1;
            for (int x = 0; x < n[2]; ++x) {
                double pseudo = ((global + x + 1) * 1299709 % volume) / double(volume);
                concentration[local + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        domain.stencil(concentration, mu, [](size_t, double cv, double laplacian) {
            constexpr double e_AA = -2.0 / 9.0, e_BB = -2.0 / 9.0, e_AB = 2.0 / 9.0;
            return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                   0.5 * laplacian;
        });
        // Each update needs only its own old concentration once mu is complete.
        domain.stencil(mu, concentration, [&](size_t i, double, double laplacian) {
            return concentration[i] + 0.01 * laplacian;
        });
    }
    double elapsed = MPI_Wtime() - start, duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", long(duration * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", volume * double(iterations) / duration / 1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(volume);
            for (int z = 0; z < n[0]; ++z)
                for (int y = 0; y < n[1]; ++y)
                    std::copy_n(concentration.data() + (z + 1) * domain.plane + (y + 1) * domain.row + 1,
                                n[2], result.data() + (z * ny + y) * nx);
            int active;
            MPI_Comm_size(cart, &active);
            for (int peer = 1; peer < active; ++peer) {
                Block other = blockFor(peer, cart, dims, grid);
                MPI_Datatype target = subarray(grid, other.size, other.start);
                MPI_Recv(result.data(), 1, target, peer, 0, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
            print_results(result, "Concentration");
        } else {
            MPI_Datatype source = subarray(domain.padded, n, {1, 1, 1});
            MPI_Send(concentration.data(), 1, source, 0, 0, cart);
            MPI_Type_free(&source);
        }
    }

    int status = 0;
    if (validate) {
        // Reduce both extrema and the nonfinite flag in a single collective.
        double stats[] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                          0.0};
        for (int z = 1; z <= n[0]; ++z) {
            for (int y = 1; y <= n[1]; ++y) {
                const double* line = concentration.data() + z * domain.plane + y * domain.row + 1;
                for (int x = 0; x < n[2]; ++x) {
                    stats[0] = std::min(stats[0], line[x]);
                    stats[1] = std::min(stats[1], -line[x]);
                    if (!std::isfinite(line[x])) stats[2] = -1.0;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, stats, 3, MPI_DOUBLE, MPI_MIN, cart);
        status = stats[2] < 0 || stats[0] < -10.0 || -stats[1] > 10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (stats[2] < 0)
                printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", stats[0], -stats[1]);
                if (status) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Comm_free(&cart);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
