#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

// Choose a feasible Cartesian topology with small communication surface area.
// Unlike MPI_Dims_create, this also accounts for non-cubic global grids.
int chooseTopology(int ranks, const int global[3], int dims[3]) {
    for (int active = ranks; active > 0; --active) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(active, global[0]); ++x) {
            if (active % x) continue;
            const int rest = active / x;
            for (int y = 1; y <= std::min(rest, global[1]); ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (z > global[2]) continue;
                // Total internal face area, normalized by global volume.
                const double surface = double(x - 1) / global[0]
                                     + double(y - 1) / global[1]
                                     + double(z - 1) / global[2];
                if (surface < best) {
                    best = surface;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return active;
    }
    return 1;
}

void blockForRank(MPI_Comm comm, int rank, const int global[3],
                  const int dims[3], int local[3], int offset[3]) {
    int coords[3];
    MPI_Cart_coords(comm, rank, 3, coords);
    for (int d = 0; d < 3; ++d) {
        const int base = global[d] / dims[d];
        const int extra = global[d] % dims[d];
        local[d] = base + (coords[d] < extra);
        offset[d] = coords[d] * base + std::min(coords[d], extra);
    }
}

MPI_Datatype subarray(const int sizes[3], const int subsizes[3], const int starts[3]) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Half-open box bounds. Keeping x innermost gives unit-stride vectorizable work.
void stencilBox(const Real* input, Real* output, size_t sx, size_t sy,
                int xb, int xe, int yb, int ye, int zb, int ze) {
    const size_t plane = sx * sy;
    for (int z = zb; z < ze; ++z) {
        for (int y = yb; y < ye; ++y) {
            const size_t row = idx3(0, y, z, sx, sy);
            for (int x = xb; x < xe; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i-1] + input[i+1]
                           + input[i-sx] + input[i+sx]
                           + input[i-plane] + input[i+plane]) / 7.0;
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int worldRank, int worldSize) {
    int global[3] = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < 0 ||
                value > std::numeric_limits<int>::max() - 2) {
                if (!worldRank) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (option == 'i') iterations = static_cast<int>(value);
            else global[option == 'x' ? 0 : option == 'y' ? 1 : 2] = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "-r")) {
            printResults = true;
        } else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    if (!global[0]) {
        if (!worldRank) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    int dims[3];
    const int active = chooseTopology(worldSize, global, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    // Excess ranks need no storage or iteration-time collectives.
    if (workers == MPI_COMM_NULL) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm comm;
    MPI_Cart_create(workers, 3, dims, periods, 0, &comm);
    MPI_Comm_free(&workers);
    int rank;
    MPI_Comm_rank(comm, &rank);
    int local[3], offset[3];
    blockForRank(comm, rank, global, dims, local, offset);
    const size_t sx = size_t(local[0]) + 2, sy = size_t(local[1]) + 2;
    const size_t sz = size_t(local[2]) + 2;
    if (sx > std::numeric_limits<size_t>::max() / sy ||
        sx * sy > std::vector<Real>().max_size() / sz) {
        throw std::bad_alloc();
    }
    std::vector<Real> grid1(sx * sy * sz), grid2(sx * sy * sz);
    if (!rank) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d x %d)\n", active, dims[0], dims[1], dims[2]);
        printf("Initializing grid...\n");
    }
    for (int z = 1; z <= local[2]; ++z)
        for (int y = 1; y <= local[1]; ++y)
            for (int x = 1; x <= local[0]; ++x) {
                // Compute the global index modulo 19 without overflowing size_t.
                const size_t indexMod = (((size_t(offset[2] + z - 1) % 19)
                    * (global[1] % 19) + size_t(offset[1] + y - 1) % 19)
                    * (global[0] % 19) + size_t(offset[0] + x - 1) % 19) % 19;
                grid1[idx3(x, y, z, sx, sy)] = Real(indexMod);
            }
    // Physical boundary values are constant; initialize both buffers once.
    grid2 = grid1;

    const int sizes[3] = {local[2] + 2, local[1] + 2, local[0] + 2};
    const int zero[3] = {0, 0, 0};
    MPI_Datatype faces[3];
    int neighbor[6];
    size_t sendOffset[6], recvOffset[6];
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        int face[3] = {local[2], local[1], local[0]};
        face[2-d] = 1;
        faces[d] = subarray(sizes, face, zero);
        int pos[3] = {1, 1, 1};
        sendOffset[2*d] = idx3(pos[0], pos[1], pos[2], sx, sy);
        pos[d] = local[d];
        sendOffset[2*d+1] = idx3(pos[0], pos[1], pos[2], sx, sy);
        pos[d] = 0;
        recvOffset[2*d] = idx3(pos[0], pos[1], pos[2], sx, sy);
        pos[d] = local[d] + 1;
        recvOffset[2*d+1] = idx3(pos[0], pos[1], pos[2], sx, sy);
    }
    int begin[3], end[3], coreBegin[3], coreEnd[3];
    for (int d = 0; d < 3; ++d) {
        begin[d] = 1 + (offset[d] == 0);
        end[d] = local[d] + 1 - (offset[d] + local[d] == global[d]);
        // Empty interiors (including dimensions of size one) are valid.
        end[d] = std::max(begin[d], end[d]);
        coreBegin[d] = std::min(end[d], std::max(begin[d], 2));
        coreEnd[d] = std::max(coreBegin[d], std::min(end[d], local[d]));
    }
    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request receives[6], sends[6];
        for (int f = 0; f < 6; ++f)
            MPI_Irecv(grid1.data() + recvOffset[f], 1, faces[f/2], neighbor[f],
                      f ^ 1, comm, &receives[f]);
        for (int f = 0; f < 6; ++f)
            MPI_Isend(grid1.data() + sendOffset[f], 1, faces[f/2], neighbor[f],
                      f, comm, &sends[f]);

        const auto box = [&](int xb, int xe, int yb, int ye, int zb, int ze) {
            stencilBox(grid1.data(), grid2.data(), sx, sy, xb, xe, yb, ye, zb, ze);
        };
        box(coreBegin[0], coreEnd[0], coreBegin[1], coreEnd[1], coreBegin[2], coreEnd[2]);
        MPI_Waitall(6, receives, MPI_STATUSES_IGNORE);
        // Six disjoint boxes cover the remaining shell, even for thin blocks.
        box(begin[0], end[0], begin[1], end[1], begin[2], coreBegin[2]);
        box(begin[0], end[0], begin[1], end[1], coreEnd[2], end[2]);
        box(begin[0], end[0], begin[1], coreBegin[1], coreBegin[2], coreEnd[2]);
        box(begin[0], end[0], coreEnd[1], end[1], coreBegin[2], coreEnd[2]);
        box(begin[0], coreBegin[0], coreBegin[1], coreEnd[1], coreBegin[2], coreEnd[2]);
        box(coreEnd[0], end[0], coreBegin[1], coreEnd[1], coreBegin[2], coreEnd[2]);
        MPI_Waitall(6, sends, MPI_STATUSES_IGNORE);
        grid1.swap(grid2);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        const double updates = double(std::max(global[0]-2, 0))
            * std::max(global[1]-2, 0) * std::max(global[2]-2, 0) * iterations;
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }

    // Gather only on request, directly into global x-fastest order. Derived
    // datatypes avoid packing buffers and MPI int count/displacement limits.
    if (printResults) {
        const int interior[3] = {local[2], local[1], local[0]};
        const int one[3] = {1, 1, 1};
        MPI_Datatype owned = subarray(sizes, interior, one);
        if (!rank) {
            const size_t gx = global[0], gy = global[1], gz = global[2];
            if (gx > std::numeric_limits<size_t>::max() / gy ||
                gx * gy > std::vector<Real>().max_size() / gz) throw std::bad_alloc();
            std::vector<Real> result(gx * gy * gz);
            for (int z = 0; z < local[2]; ++z)
                for (int y = 0; y < local[1]; ++y)
                    std::copy_n(grid1.data() + idx3(1, y+1, z+1, sx, sy), local[0],
                        result.data() + idx3(offset[0], offset[1]+y, offset[2]+z, gx, gy));
            const int full[3] = {global[2], global[1], global[0]};
            for (int r = 1; r < active; ++r) {
                int n[3], o[3];
                blockForRank(comm, r, global, dims, n, o);
                const int count[3] = {n[2], n[1], n[0]}, at[3] = {o[2], o[1], o[0]};
                MPI_Datatype target = subarray(full, count, at);
                MPI_Recv(result.data(), 1, target, r, 6, comm, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
            print_results(result, "Grid");
        } else {
            MPI_Send(grid1.data(), 1, owned, 0, 6, comm);
        }
        MPI_Type_free(&owned);
    }

    int valid = 1;
    if (validate) {
        Real minimum = std::numeric_limits<Real>::infinity();
        Real maximum = -minimum;
        int finite = 1;
        for (int z = 1; z <= local[2]; ++z)
            for (int y = 1; y <= local[1]; ++y)
                for (int x = 1; x <= local[0]; ++x) {
                    const Real v = grid1[idx3(x, y, z, sx, sy)];
                    finite &= std::isfinite(v);
                    minimum = std::min(minimum, v);
                    maximum = std::max(maximum, v);
                }
        Real globalMin, globalMax;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        valid = valid && globalMax <= 1e6 && globalMin >= -1e6;
        if (!rank) {
            printf("Validating result...\n");
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (auto& face : faces) MPI_Type_free(&face);
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 1;
    try {
        status = run(argc, argv, rank, size);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
