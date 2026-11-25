#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Coordinates are ordered z, y, x to match MPI_ORDER_C.
using Shape = std::array<int, 3>;

struct Box {
    Shape lo, hi;  // half-open intervals
};

// Prefer full occupancy, then minimize the total area of internal faces.
Shape processGrid(const Shape& global, int ranks) {
    Shape best{1, 1, 1};
    int used = 1;
    double bestArea = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= std::min(global[0], ranks); ++z) {
        for (int y = 1; y <= std::min(global[1], ranks / z); ++y) {
            int x = std::min(global[2], ranks / (z * y));
            int count = z * y * x;
            double area = double(z - 1) * global[1] * global[2]
                        + double(y - 1) * global[0] * global[2]
                        + double(x - 1) * global[0] * global[1];
            if (count > used || (count == used && area < bestArea)) {
                used = count;
                bestArea = area;
                best = {z, y, x};
            }
        }
    }
    return best;
}

Box partition(const Shape& global, const Shape& dims, const Shape& coords) {
    Box b;
    for (int d = 0; d < 3; ++d) {
        int base = global[d] / dims[d], extra = global[d] % dims[d];
        b.lo[d] = coords[d] * base + std::min(coords[d], extra);
        b.hi[d] = b.lo[d] + base + (coords[d] < extra);
    }
    return b;
}

MPI_Datatype subarray(const Shape& full, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, full.data(), size.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Grid {
    Shape n, padded;
    size_t row, plane;

    size_t index(int z, int y, int x) const {
        return size_t(z) * plane + size_t(y) * row + x;
    }
};

struct Halos {
    MPI_Comm comm;
    Box owned, interior;
    int peer[6];
    MPI_Datatype face[6]{}, ghost[6]{};

    Halos(MPI_Comm cart, const Grid& g) : comm(cart) {
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart, d, 1, &peer[2*d], &peer[2*d+1]);
            owned.lo[d] = 1;
            owned.hi[d] = g.n[d] + 1;
            interior.lo[d] = 1 + (peer[2*d] != MPI_PROC_NULL);
            interior.hi[d] = std::max(interior.lo[d],
                                      owned.hi[d] - (peer[2*d+1] != MPI_PROC_NULL));
        }
        for (int side = 0; side < 6; ++side) {
            if (peer[side] == MPI_PROC_NULL) continue;
            int d = side / 2;
            Shape size = g.n, start{1, 1, 1};
            size[d] = 1;
            start[d] = side % 2 ? g.n[d] : 1;
            face[side] = subarray(g.padded, size, start);
            start[d] += side % 2 ? 1 : -1;
            ghost[side] = subarray(g.padded, size, start);
        }
    }

    ~Halos() {
        for (int i = 0; i < 6; ++i) {
            if (peer[i] != MPI_PROC_NULL) {
                MPI_Type_free(&face[i]);
                MPI_Type_free(&ghost[i]);
            }
        }
    }

    template<class Kernel>
    void apply(std::vector<double>& field, const Kernel& compute) const {
        MPI_Request requests[12];
        int count = 0;
        for (int i = 0; i < 6; ++i) {
            if (peer[i] == MPI_PROC_NULL) continue;
            MPI_Irecv(field.data(), 1, ghost[i], peer[i], i ^ 1, comm, &requests[count++]);
            MPI_Isend(field.data(), 1, face[i], peer[i], i, comm, &requests[count++]);
        }
        compute(interior);
        if (!count) return;
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Peel disjoint boundary slabs, including one-cell-wide partitions.
        Box remaining = owned;
        for (int d = 0; d < 3; ++d) {
            Box slab = remaining;
            if (remaining.lo[d] < interior.lo[d]) {
                slab.hi[d] = interior.lo[d];
                compute(slab);
            }
            if (remaining.hi[d] > interior.hi[d]) {
                slab = remaining;
                slab.lo[d] = interior.hi[d];
                compute(slab);
            }
            remaining.lo[d] = interior.lo[d];
            remaining.hi[d] = interior.hi[d];
        }
    }
};

// Both stencil passes share unit spacing and clamped physical boundaries.
template<class Update>
void stencil(const Grid& g, const Halos& h, std::vector<double>& input,
             double* __restrict__ output, const Update& update) {
    h.apply(input, [&](const Box& b) {
        for (int z = b.lo[0]; z < b.hi[0]; ++z) {
            size_t zm = z == 1 && h.peer[0] == MPI_PROC_NULL ? 0 : g.plane;
            size_t zp = z == g.n[0] && h.peer[1] == MPI_PROC_NULL ? 0 : g.plane;
            for (int y = b.lo[1]; y < b.hi[1]; ++y) {
                size_t ym = y == 1 && h.peer[2] == MPI_PROC_NULL ? 0 : g.row;
                size_t yp = y == g.n[1] && h.peer[3] == MPI_PROC_NULL ? 0 : g.row;
                // Extend the row at physical faces so the hot loop has unit-stride loads.
                double* line = &input[g.index(z, y, 0)];
                if (b.lo[2] == 1 && h.peer[4] == MPI_PROC_NULL) line[0] = line[1];
                if (b.hi[2] > g.n[2] && h.peer[5] == MPI_PROC_NULL)
                    line[g.n[2]+1] = line[g.n[2]];
                for (int x = b.lo[2]; x < b.hi[2]; ++x) {
                    size_t i = g.index(z, y, x);
                    double c = input[i];
                    double xx = input[i+1] + input[i-1] - 2.0*c;
                    double yy = input[i+yp] + input[i-ym] - 2.0*c;
                    double zz = input[i+zp] + input[i-zm] - 2.0*c;
                    output[i] = update(i, c, xx + yy + zz);
                }
            }
        }
    });
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
    struct Finalize { ~Finalize() { MPI_Finalize(); } } finalize;
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
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    Shape global{nz, ny, nx};
    size_t volume = 1;
    for (int length : global) {
        if (length < 1 || length > INT_MAX - 2 ||
            volume > std::numeric_limits<size_t>::max() / (length + 2)) {
            if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
            return 1;
        }
        volume *= length + 2;
    }
    volume = size_t(nx) * ny * nz;
    Shape dims = processGrid(global, ranks), periods{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    Shape coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    Box block = partition(global, dims, coords);
    Grid g;
    for (int d = 0; d < 3; ++d) {
        g.n[d] = block.hi[d] - block.lo[d];
        g.padded[d] = g.n[d] + 2;
    }
    g.row = g.padded[2];
    g.plane = g.row * g.padded[1];
    size_t storage = g.plane * g.padded[0];
    std::vector<double> cold(storage), cnew(storage), mu(storage);
    Halos halos(cart, g);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    for (int z = 1; z <= g.n[0]; ++z) {
        for (int y = 1; y <= g.n[1]; ++y) {
            size_t first = (size_t(block.lo[0] + z - 1) * ny + block.lo[1] + y - 1) * nx
                         + block.lo[2];
            for (int x = 1; x <= g.n[2]; ++x) {
                double pseudo = ((first + x) * 1299709 % volume) / double(volume);
                cold[g.index(z, y, x)] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        stencil(g, halos, cold, mu.data(), [](size_t, double cv, double lap) {
            constexpr double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
            return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv - 0.5 * lap;
        });
        stencil(g, halos, mu, cnew.data(), [&](size_t i, double, double lap) {
            return cold[i] + 0.01 * lap;
        });
        cold.swap(cnew);
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", double(volume)*iterations/duration/1e6);
    }

    if (printResults) {
        if (rank == 0) {
            std::vector<double> result(volume);
            int active;
            MPI_Comm_size(cart, &active);
            for (int src = 0; src < active; ++src) {
                Shape location;
                MPI_Cart_coords(cart, src, 3, location.data());
                Box region = partition(global, dims, location);
                if (src == 0) {
                    for (int z = 0; z < g.n[0]; ++z) {
                        for (int y = 0; y < g.n[1]; ++y) {
                            size_t offset = (size_t(z) * ny + y) * nx;
                            std::copy_n(&cold[g.index(z+1, y+1, 1)], g.n[2], &result[offset]);
                        }
                    }
                } else {
                    Shape size;
                    for (int d = 0; d < 3; ++d) size[d] = region.hi[d] - region.lo[d];
                    MPI_Datatype dest = subarray(global, size, region.lo);
                    MPI_Recv(result.data(), 1, dest, src, 0, cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&dest);
                }
            }
            print_results(result, "Concentration");
        } else {
            MPI_Datatype src = subarray(g.padded, g.n, {1, 1, 1});
            MPI_Send(cold.data(), 1, src, 0, 0, cart);
            MPI_Type_free(&src);
        }
    }

    int failed = 0;
    if (validate) {
        // MPI_MIN on the negated maximum combines both extrema in a single reduction.
        double local[3] = {INFINITY, INFINITY, 0}, bounds[3];
        for (int z = 1; z <= g.n[0]; ++z) {
            for (int y = 1; y <= g.n[1]; ++y) {
                for (int x = 1; x <= g.n[2]; ++x) {
                    double v = cold[g.index(z, y, x)];
                    local[0] = std::min(local[0], v);
                    local[1] = std::min(local[1], -v);
                    if (!std::isfinite(v)) local[2] = -1;
                }
            }
        }
        MPI_Allreduce(local, bounds, 3, MPI_DOUBLE, MPI_MIN, cart);
        failed = bounds[2] < 0 || bounds[0] < -10 || -bounds[1] > 10;
        if (rank == 0) {
            printf("Validating result...\n");
            if (bounds[2] < 0) {
                printf("Validation failed: found NaN or Inf value\n");
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", bounds[0], -bounds[1]);
                if (failed) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    MPI_Comm_free(&cart);
    return failed;
}
