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

using Shape = std::array<int, 3>;  // z, y, x in C storage order

// Keep blocks nonempty and minimize the total area of internal faces.
Shape processGrid(const Shape& global, int ranks) {
    const size_t volume = size_t(global[0]) * global[1] * global[2];
    for (int count = std::min<size_t>(volume, ranks); count > 0; --count) {
        Shape best{};
        double bestArea = std::numeric_limits<double>::infinity();
        for (int z = 1; z <= std::min(global[0], count); ++z) {
            if (count % z) continue;
            const int plane = count / z;
            for (int y = 1; y <= std::min(global[1], plane); ++y) {
                if (plane % y || plane / y > global[2]) continue;
                const int x = plane / y;
                const double area = double(z - 1) / global[0] +
                                    double(y - 1) / global[1] +
                                    double(x - 1) / global[2];
                // Prefer contiguous z faces when surface areas tie.
                if (area <= bestArea) {
                    best = {z, y, x};
                    bestArea = area;
                }
            }
        }
        if (best[0]) return best;
    }
    return {1, 1, 1};
}

struct Block {
    Shape start, size;
};

Block partition(const Shape& global, const Shape& dims, const Shape& coords) {
    Block block;
    for (int d = 0; d < 3; ++d) {
        const int base = global[d] / dims[d], extra = global[d] % dims[d];
        block.size[d] = base + (coords[d] < extra);
        block.start[d] = base * coords[d] + std::min(coords[d], extra);
    }
    return block;
}

MPI_Datatype subarray(const Shape& storage, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, storage.data(), size.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    MPI_Comm comm;
    Shape n, storage, innerBegin, innerEnd;
    std::array<int, 6> peer;
    std::array<MPI_Datatype, 6> face;
    std::array<size_t, 6> sendOffset, recvOffset;

    size_t index(int z, int y, int x) const {
        return (size_t(z) * storage[1] + y) * storage[2] + x;
    }

    Domain(MPI_Comm cart, Shape size) : comm(cart), n(size) {
        for (int d = 0; d < 3; ++d) storage[d] = n[d] + 2;
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(comm, d, 1, &peer[2*d], &peer[2*d+1]);
            innerBegin[d] = 1 + (peer[2*d] != MPI_PROC_NULL);
            innerEnd[d] = std::max(innerBegin[d], n[d]+1 - (peer[2*d+1] != MPI_PROC_NULL));
            Shape extent = n;
            extent[d] = 1;
            for (int side = 0; side < 2; ++side) {
                const int f = 2*d + side;
                if (peer[f] == MPI_PROC_NULL) continue;
                face[f] = subarray(storage, extent, {0, 0, 0});
                Shape pos{1, 1, 1};
                pos[d] = side ? n[d] : 1;
                sendOffset[f] = index(pos[0], pos[1], pos[2]);
                pos[d] += side ? 1 : -1;
                recvOffset[f] = index(pos[0], pos[1], pos[2]);
            }
        }
    }

    ~Domain() {
        for (int f = 0; f < 6; ++f)
            if (peer[f] != MPI_PROC_NULL) MPI_Type_free(&face[f]);
    }

    // Apply one stencil while the input's six faces travel to neighboring ranks.
    template<class Update>
    void stencil(std::vector<double>& field, Update update) const {
        MPI_Request requests[12];
        int count = 0;
        double* v = field.data();
        for (int f = 0; f < 6; ++f) {
            if (peer[f] == MPI_PROC_NULL) continue;
            MPI_Irecv(v + recvOffset[f], 1, face[f], peer[f], f ^ 1, comm, &requests[count++]);
            MPI_Isend(v + sendOffset[f], 1, face[f], peer[f], f, comm, &requests[count++]);
        }

        auto row = [&](int z, int y, int begin, int end) {
            const size_t i = index(z, y, 0);
            const size_t stride = storage[2], plane = stride * storage[1];
            const double* center = v + i;
            const double* yp = center + ((y == n[1] && peer[3] == MPI_PROC_NULL) ? 0 : stride);
            const double* ym = center - ((y == 1 && peer[2] == MPI_PROC_NULL) ? 0 : stride);
            const double* zp = center + ((z == n[0] && peer[1] == MPI_PROC_NULL) ? 0 : plane);
            const double* zm = center - ((z == 1 && peer[0] == MPI_PROC_NULL) ? 0 : plane);
            // Replicate physical x endpoints in the unused ghost slots.
            // Only this row reads those slots.
            if (peer[4] == MPI_PROC_NULL) v[i] = center[1];
            if (peer[5] == MPI_PROC_NULL) v[i+n[2]+1] = center[n[2]];
            for (int x = begin; x < end; ++x) {
                const double twice = 2.0 * center[x];
                const double lap = (center[x+1] + center[x-1] - twice) +
                                   (yp[x] + ym[x] - twice) + (zp[x] + zm[x] - twice);
                update(i+x, center[x], lap);
            }
        };

        for (int z = innerBegin[0]; z < innerEnd[0]; ++z)
            for (int y = innerBegin[1]; y < innerEnd[1]; ++y)
                row(z, y, innerBegin[2], innerEnd[2]);

        if (!count) return;
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (int z = 1; z <= n[0]; ++z) {
            for (int y = 1; y <= n[1]; ++y) {
                if (z < innerBegin[0] || z >= innerEnd[0] ||
                    y < innerBegin[1] || y >= innerEnd[1]) {
                    row(z, y, 1, n[2]+1);
                } else {
                    if (innerBegin[2] > 1) row(z, y, 1, innerBegin[2]);
                    if (innerEnd[2] <= n[2]) row(z, y, innerEnd[2], n[2]+1);
                }
            }
        }
    }
};

void reportField(const Domain& domain, const std::vector<double>& field, const Shape& global,
                 const Shape& dims, int rank) {
    MPI_Datatype owned = subarray(domain.storage, domain.n, {1, 1, 1});
    if (rank == 0) {
        std::vector<double> result(size_t(global[0]) * global[1] * global[2]);
        int ranks;
        MPI_Comm_size(domain.comm, &ranks);
        for (int r = 0; r < ranks; ++r) {
            Shape coords;
            MPI_Cart_coords(domain.comm, r, 3, coords.data());
            const Block block = partition(global, dims, coords);
            MPI_Datatype target = subarray(global, block.size, block.start);
            if (r == 0) {
                MPI_Sendrecv(field.data(), 1, owned, 0, 0, result.data(), 1, target,
                             0, 0, domain.comm, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(result.data(), 1, target, r, 0, domain.comm, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&target);
        }
        print_results(result, "Concentration");
    } else {
        MPI_Send(field.data(), 1, owned, 0, 0, domain.comm);
    }
    MPI_Type_free(&owned);
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
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const Shape global{nz, ny, nx};
    const auto maxSize = std::numeric_limits<size_t>::max();
    if (std::min({nx, ny, nz}) <= 0 ||
        std::max({nx, ny, nz}) > std::numeric_limits<int>::max() - 2 ||
        size_t(nx) > maxSize / ny / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t volume = size_t(nx) * ny * nz;
    const Shape dims = processGrid(global, ranks);
    const int periodic[3]{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periodic, 0, &cart);
    int valid = 1;
    if (cart != MPI_COMM_NULL) {
        Shape coords;
        MPI_Cart_coords(cart, rank, 3, coords.data());
        const Block block = partition(global, dims, coords);
        const Domain domain(cart, block.size);
        const size_t cells = size_t(domain.storage[0]) * domain.storage[1] * domain.storage[2];
        std::vector<double> c(cells), mu(cells);

        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %d x %d x %d\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (int z = 0; z < block.size[0]; ++z) {
            for (int y = 0; y < block.size[1]; ++y) {
                const size_t first = (size_t(block.start[0]+z) * ny + block.start[1]+y) * nx + block.start[2];
                const size_t local = domain.index(z+1, y+1, 1);
                for (int x = 0; x < block.size[2]; ++x) {
                    const double pseudo = ((first+x+1) * 1299709 % volume) / double(volume);
                    c[local+x] = -1.0 + 2.0 * pseudo;
                }
            }
        }

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
        constexpr double gamma = 0.5, dt = 0.01, D = 1.0;
        for (int t = 0; t < iterations; ++t) {
            domain.stencil(c, [&](size_t i, double cv, double lap) {
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - gamma * lap;
            });
            // The update needs only the old concentration at the same site.
            domain.stencil(mu, [&](size_t i, double, double lap) {
                c[i] += dt * D * lap;
            });
        }
        const double localTime = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&localTime, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(1000 * seconds));
            printf("Performance: %.3f MCellUpdates/s\n", double(volume) * iterations / seconds / 1e6);
        }

        if (printResults) reportField(domain, c, global, dims, rank);
        if (validate) {
            // Negating the minimum lets one MAX reduction collect both bounds.
            double stats[3]{-std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(), 0.0};
            for (int z = 1; z <= block.size[0]; ++z) {
                for (int y = 1; y <= block.size[1]; ++y) {
                    const double* row = c.data() + domain.index(z, y, 1);
                    for (int x = 0; x < block.size[2]; ++x) {
                        stats[0] = std::max(stats[0], -row[x]);
                        stats[1] = std::max(stats[1], row[x]);
                        if (!std::isfinite(row[x])) stats[2] = 1.0;
                    }
                }
            }
            MPI_Allreduce(MPI_IN_PLACE, stats, 3, MPI_DOUBLE, MPI_MAX, cart);
            valid = stats[2] == 0 && stats[0] <= 10.0 && stats[1] <= 10.0;
            if (rank == 0) {
                printf("Validating result...\n");
                if (stats[2]) {
                    printf("Validation failed: found NaN or Inf value\n");
                } else {
                    printf("Concentration range: [%.6f, %.6f]\n", -stats[0], stats[1]);
                    if (!valid) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_Comm_free(&cart);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
