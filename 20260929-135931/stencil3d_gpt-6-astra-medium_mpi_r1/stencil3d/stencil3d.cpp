#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// Choose the largest usable process grid, then minimize its halo surface.
// Geometry-aware factoring also handles thin grids and more ranks than cells.
std::array<int, 3> processGrid(int ranks, const std::array<int, 3>& global) {
    for (int count = ranks; count > 0; --count) {
        std::array<int, 3> best{};
        double cost = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(count, global[0]); ++x) {
            if (count % x) continue;
            const int rest = count / x;
            for (int y = 1; y <= std::min(rest, global[1]); ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (z > global[2]) continue;
                const double lx = double(global[0]) / x;
                const double ly = double(global[1]) / y;
                const double lz = double(global[2]) / z;
                const double surface = (x > 1 ? ly * lz : 0) +
                                       (y > 1 ? lx * lz : 0) +
                                       (z > 1 ? lx * ly : 0);
                if (surface < cost) {
                    cost = surface;
                    best = {x, y, z};
                }
            }
        }
        if (best[0]) return best;
    }
    return {1, 1, 1};
}

struct Domain {
    std::array<int, 3> n, offset;
    size_t row, plane;
    explicit Domain(const std::array<int, 3>& global,
                    const std::array<int, 3>& dims, const int* coords) {
        for (int d = 0; d < 3; ++d) {
            const int base = global[d] / dims[d];
            const int extra = global[d] % dims[d];
            n[d] = base + (coords[d] < extra);
            offset[d] = coords[d] * base + std::min(coords[d], extra);
        }
        row = size_t(n[0]) + 2;
        plane = row * (size_t(n[1]) + 2);
    }
    size_t index(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * row + x;
    }
    MPI_Datatype type(const std::array<int, 3>& start,
                      const std::array<int, 3>& extent) const {
        const int sizes[3] = {n[2] + 2, n[1] + 2, n[0] + 2};
        const int subs[3] = {extent[2], extent[1], extent[0]};
        const int starts[3] = {start[2], start[1], start[0]};
        MPI_Datatype result;
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                                 MPI_DOUBLE, &result);
        MPI_Type_commit(&result);
        return result;
    }
};

// Preserve the serial expression's operation order; no fast-math reassociation.
inline void updateRow(const Real* __restrict__ in, Real* __restrict__ out,
                      size_t begin, size_t end, size_t row, size_t plane) {
    for (size_t i = begin; i < end; ++i) {
        out[i] = (in[i] + in[i-1] + in[i+1] + in[i-row] + in[i+row] +
                  in[i-plane] + in[i+plane]) / 7.0;
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

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    std::array<int, 3> global = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "-x") || !strcmp(arg, "-y") ||
             !strcmp(arg, "-z") || !strcmp(arg, "-i")) && i + 1 < argc) {
            char* end;
            const long value = std::strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX - 2) {
                if (!rank) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (!strcmp(arg, "-i")) iterations = int(value);
            else global[arg[1] - 'x'] = int(value);
        } else if (!strcmp(arg, "-v")) validate = true;
        else if (!strcmp(arg, "-r")) printResults = true;
        else if (!strcmp(arg, "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", arg); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    size_t capacity = 1;
    for (int n : global) {
        if (!n || capacity > std::numeric_limits<size_t>::max() / sizeof(Real) / (size_t(n) + 2)) {
            if (!rank) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
            return 1;
        }
        capacity *= size_t(n) + 2;
    }
    const auto dims = processGrid(ranks, global);
    const int active = dims[0] * dims[1] * dims[2];
    MPI_Comm group;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &group);
    if (rank >= active) return 0;
    MPI_Comm cart;
    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(group, 3, dims.data(), periods, 0, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    const Domain domain(global, dims, coords);
    const auto& n = domain.n;
    std::vector<Real> grid1(domain.plane * (size_t(n[2]) + 2));
    if (!rank) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid: %d x %d x %d)\n", active, dims[0], dims[1], dims[2]);
        printf("Initializing grid...\n");
    }
    for (int z = 1; z <= n[2]; ++z)
        for (int y = 1; y <= n[1]; ++y)
            for (int x = 1; x <= n[0]; ++x) {
                const size_t index = (size_t(z - 1 + domain.offset[2]) * global[1] +
                                     y - 1 + domain.offset[1]) * global[0] + x - 1 + domain.offset[0];
                grid1[domain.index(x, y, z)] = Real(index % 19);
            }
    // Fixed global boundaries are initialized in both buffers once.
    std::vector<Real> grid2 = grid1;
    std::array<int, 6> neighbor;
    std::array<MPI_Datatype, 6> sendTypes, receiveTypes;
    sendTypes.fill(MPI_DATATYPE_NULL);
    receiveTypes.fill(MPI_DATATYPE_NULL);
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        for (int side = 0; side < 2; ++side) {
            const int face = 2*d + side;
            if (neighbor[face] == MPI_PROC_NULL) continue;
            auto extent = n;
            extent[d] = 1;
            std::array<int, 3> start = {1, 1, 1};
            start[d] = side ? n[d] : 1;
            sendTypes[face] = domain.type(start, extent);
            start[d] = side ? n[d] + 1 : 0;
            receiveTypes[face] = domain.type(start, extent);
        }
    }
    std::array<int, 3> lo, hi, coreLo, coreHi;
    for (int d = 0; d < 3; ++d) {
        lo[d] = domain.offset[d] == 0 ? 2 : 1;
        hi[d] = n[d] + 1 - (domain.offset[d] + n[d] == global[d]);
        coreLo[d] = std::max(lo[d], 2);
        coreHi[d] = std::min(hi[d], n[d]);
    }
    if (!rank) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double startTime = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (int face = 0; face < 6; ++face)
            if (neighbor[face] != MPI_PROC_NULL)
                MPI_Irecv(grid1.data(), 1, receiveTypes[face], neighbor[face], face ^ 1,
                          cart, &requests[count++]);
        for (int face = 0; face < 6; ++face)
            if (neighbor[face] != MPI_PROC_NULL)
                MPI_Isend(grid1.data(), 1, sendTypes[face], neighbor[face], face,
                          cart, &requests[count++]);
        for (int z = coreLo[2]; z < coreHi[2]; ++z)
            for (int y = coreLo[1]; y < coreHi[1]; ++y)
                updateRow(grid1.data(), grid2.data(), domain.index(coreLo[0], y, z),
                          domain.index(coreHi[0], y, z), domain.row, domain.plane);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // The remaining shell is disjoint from the already updated core.
        for (int z = lo[2]; z < hi[2]; ++z)
            for (int y = lo[1]; y < hi[1]; ++y) {
                if (z >= coreLo[2] && z < coreHi[2] &&
                    y >= coreLo[1] && y < coreHi[1] && coreLo[0] < coreHi[0]) {
                    updateRow(grid1.data(), grid2.data(), domain.index(lo[0], y, z),
                              domain.index(coreLo[0], y, z), domain.row, domain.plane);
                    updateRow(grid1.data(), grid2.data(), domain.index(coreHi[0], y, z),
                              domain.index(hi[0], y, z), domain.row, domain.plane);
                } else {
                    updateRow(grid1.data(), grid2.data(), domain.index(lo[0], y, z),
                              domain.index(hi[0], y, z), domain.row, domain.plane);
                }
            }
        grid1.swap(grid2);
    }
    const double elapsed = MPI_Wtime() - startTime;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (!rank) {
        const double updates = double(std::max(global[0]-2, 0)) *
                               std::max(global[1]-2, 0) * std::max(global[2]-2, 0) * iterations;
        printf("Computation time: %.3f ms\n", duration * 1000);
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }
    if (printResults) {
        // Gather directly into global row-major order, without int-sized counts
        // or an additional packed copy of each process's block.
        MPI_Datatype owned = domain.type({1, 1, 1}, n);
        if (rank) MPI_Send(grid1.data(), 1, owned, 0, 6, cart);
        else {
            std::vector<Real> result(size_t(global[0]) * global[1] * global[2]);
            for (int source = 0; source < active; ++source) {
                int sourceCoords[3];
                MPI_Cart_coords(cart, source, 3, sourceCoords);
                const Domain block(global, dims, sourceCoords);
                const int sizes[3] = {global[2], global[1], global[0]};
                const int subs[3] = {block.n[2], block.n[1], block.n[0]};
                const int starts[3] = {block.offset[2], block.offset[1], block.offset[0]};
                MPI_Datatype target;
                MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &target);
                MPI_Type_commit(&target);
                if (source == 0)
                    MPI_Sendrecv(grid1.data(), 1, owned, 0, 6, result.data(), 1, target, 0, 6,
                                 cart, MPI_STATUS_IGNORE);
                else MPI_Recv(result.data(), 1, target, source, 6, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
            print_results(result, "Grid");
        }
        MPI_Type_free(&owned);
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity(), localMax = -localMin;
        for (int z = 1; z <= n[2]; ++z)
            for (int y = 1; y <= n[1]; ++y)
                for (int x = 1; x <= n[0]; ++x) {
                    const Real value = grid1[domain.index(x, y, z)];
                    if (!std::isfinite(value)) valid = 0;
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
        if (localMin < -1e6 || localMax > 1e6) valid = 0;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, cart);
        Real minimum, maximum;
        MPI_Reduce(&localMin, &minimum, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&localMax, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (!rank) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", minimum, maximum);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (int face = 0; face < 6; ++face) {
        if (sendTypes[face] != MPI_DATATYPE_NULL) MPI_Type_free(&sendTypes[face]);
        if (receiveTypes[face] != MPI_DATATYPE_NULL) MPI_Type_free(&receiveTypes[face]);
    }
    MPI_Comm_free(&cart);
    MPI_Comm_free(&group);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "stencil3d: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
