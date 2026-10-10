#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

// Axis order is Z, Y, X, so contiguous X rows remain innermost.
struct Block {
    int n[3];
    size_t start[3];
};

Block blockFor(const int global[3], const int dims[3], const int coords[3]) {
    Block b;
    for (int a = 0; a < 3; ++a) {
        const int base = global[a] / dims[a], extra = global[a] % dims[a];
        b.n[a] = base + (coords[a] < extra);
        b.start[a] = size_t(coords[a]) * base + std::min(coords[a], extra);
    }
    return b;
}

// Use as many nonempty ranks as possible, then minimize halo surface/volume.
// Unlike MPI_Dims_create, this also accounts for elongated global grids.
void chooseTopology(const int global[3], int ranks, int dims[3]) {
    int bestRanks = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= std::min(global[2], ranks); ++px) {
        for (int py = 1; py <= std::min(global[1], ranks / px); ++py) {
            const int pz = std::min(global[0], ranks / (px * py));
            const int used = px * py * pz;
            const double cost = double(px - 1) / global[2]
                              + double(py - 1) / global[1]
                              + double(pz - 1) / global[0];
            if (used > bestRanks || (used == bestRanks && cost < bestCost)) {
                bestRanks = used;
                bestCost = cost;
                dims[0] = pz; dims[1] = py; dims[2] = px;
            }
        }
    }
}

// Bounds are half-open. The addition order matches the serial implementation;
// do not enable fast-math, which would change the external validation hash.
void updateBox(const Real* __restrict__ input, Real* __restrict__ output,
               size_t sx, size_t sy, const int lo[3], const int hi[3]) {
    const size_t plane = sx * sy;
    for (int z = lo[0]; z < hi[0]; ++z) {
        for (int y = lo[1]; y < hi[1]; ++y) {
            const size_t row = idx3(0, y, z, sx, sy);
            for (int x = lo[2]; x < hi[2]; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1]
                           + input[i - sx] + input[i + sx]
                           + input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

// Gather only for -r. Chunked transfers avoid MPI's int count limit, and root
// reconstructs global row-major ordering for the original sum/samples/hash.
void printDistributedResults(const std::vector<Real>& grid, const Block& b,
                             const int global[3], const int dims[3],
                             MPI_Comm cart, int rank, int ranks) {
    const size_t sx = size_t(b.n[2]) + 2, sy = size_t(b.n[1]) + 2;
    std::vector<Real> packed(size_t(b.n[0]) * b.n[1] * b.n[2]);
    size_t offset = 0;
    for (int z = 1; z <= b.n[0]; ++z)
        for (int y = 1; y <= b.n[1]; ++y) {
            std::copy_n(grid.data() + idx3(1, y, z, sx, sy), b.n[2], packed.data() + offset);
            offset += b.n[2];
        }
    constexpr size_t chunk = 1 << 20;
    if (rank != 0) {
        for (size_t i = 0; i < packed.size(); i += chunk)
            MPI_Send(packed.data() + i, int(std::min(chunk, packed.size() - i)),
                     MPI_DOUBLE, 0, 20, cart);
        return;
    }
    std::vector<Real> result(size_t(global[0]) * global[1] * global[2]);
    for (int r = 0; r < ranks; ++r) {
        int coords[3];
        MPI_Cart_coords(cart, r, 3, coords);
        const Block remote = blockFor(global, dims, coords);
        if (r != 0) {
            packed.resize(size_t(remote.n[0]) * remote.n[1] * remote.n[2]);
            for (size_t i = 0; i < packed.size(); i += chunk)
                MPI_Recv(packed.data() + i, int(std::min(chunk, packed.size() - i)),
                         MPI_DOUBLE, r, 20, cart, MPI_STATUS_IGNORE);
        }
        offset = 0;
        for (int z = 0; z < remote.n[0]; ++z)
            for (int y = 0; y < remote.n[1]; ++y) {
                const size_t dest = idx3(remote.start[2], remote.start[1] + y,
                                         remote.start[0] + z, global[2], global[1]);
                std::copy_n(packed.data() + offset, remote.n[2], result.data() + dest);
                offset += remote.n[2];
            }
    }
    print_results(result, "Grid");
}

bool validateDistributed(const std::vector<Real>& grid, const Block& b,
                         MPI_Comm cart, int rank) {
    Real bounds[2] = {std::numeric_limits<Real>::infinity(),
                     -std::numeric_limits<Real>::infinity()};
    int finite = 1;
    const size_t sx = size_t(b.n[2]) + 2, sy = size_t(b.n[1]) + 2;
    for (int z = 1; z <= b.n[0]; ++z)
        for (int y = 1; y <= b.n[1]; ++y)
            for (int x = 1; x <= b.n[2]; ++x) {
                const Real value = grid[idx3(x, y, z, sx, sy)];
                finite &= std::isfinite(value);
                bounds[0] = std::min(bounds[0], value);
                bounds[1] = std::max(bounds[1], value);
            }
    Real minimum, maximum;
    int allFinite;
    MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, cart);
    MPI_Allreduce(bounds, &minimum, 1, MPI_DOUBLE, MPI_MIN, cart);
    MPI_Allreduce(bounds + 1, &maximum, 1, MPI_DOUBLE, MPI_MAX, cart);
    const bool valid = allFinite && minimum >= -1e6 && maximum <= 1e6;
    if (rank == 0) {
        if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
        printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
        if (minimum < -1e6 || maximum > 1e6)
            printf("Validation failed: values out of expected range\n");
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
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
    int nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if ((!strcmp(option, "-x") || !strcmp(option, "-y") ||
             !strcmp(option, "-z") || !strcmp(option, "-i")) && i + 1 < argc) {
            char* end;
            errno = 0;
            const long value = strtol(argv[++i], &end, 10);
            const bool dimension = strcmp(option, "-i") != 0;
            if (errno || *end || end == argv[i] || value < 0 ||
                value > INT_MAX - 2 || (dimension && !strcmp(option, "-x") && value == 0)) {
                if (worldRank == 0) fprintf(stderr, "Invalid value for %s: %s\n", option, argv[i]);
                return 1;
            }
            if (!strcmp(option, "-x")) nx = int(value);
            else if (!strcmp(option, "-y")) ny = int(value);
            else if (!strcmp(option, "-z")) nz = int(value);
            else iterations = int(value);
        } else if (!strcmp(option, "-v")) validate = true;
        else if (!strcmp(option, "-r")) printResults = true;
        else if (!strcmp(option, "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const int global[3] = {nz, ny, nx};
    if (size_t(nx) > std::numeric_limits<size_t>::max() / size_t(ny) / size_t(nz) / sizeof(Real))
        throw std::runtime_error("Grid size exceeds addressable memory");
    int dims[3];
    chooseTopology(global, worldSize, dims);
    const int ranks = dims[0] * dims[1] * dims[2];
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < ranks ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (active == MPI_COMM_NULL) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&active);
    int rank, coords[3];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, coords);
    const Block b = blockFor(global, dims, coords);
    const size_t sx = size_t(b.n[2]) + 2, sy = size_t(b.n[1]) + 2;
    if (sx > std::numeric_limits<size_t>::max() / sy / (size_t(b.n[0]) + 2) / sizeof(Real))
        throw std::runtime_error("Local grid size exceeds addressable memory");
    std::vector<Real> grids[2];
    grids[0].resize(sx * sy * (size_t(b.n[0]) + 2));
    for (int z = 1; z <= b.n[0]; ++z)
        for (int y = 1; y <= b.n[1]; ++y)
            for (int x = 1; x <= b.n[2]; ++x) {
                const size_t index = idx3(b.start[2] + x - 1, b.start[1] + y - 1,
                                          b.start[0] + z - 1, nx, ny);
                grids[0][idx3(x, y, z, sx, sy)] = Real(index % 19);
            }
    // Fixed physical boundaries are initialized in BOTH buffers, avoiding a
    // full-volume boundary-copy pass every iteration.
    grids[1] = grids[0];

    int lo[3], hi[3], coreLo[3], coreHi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = 1 + (b.start[a] == 0);
        hi[a] = b.n[a] + 1 - (b.start[a] + b.n[a] == size_t(global[a]));
        hi[a] = std::max(lo[a], hi[a]);
        coreLo[a] = std::min(hi[a], std::max(lo[a], 2));
        coreHi[a] = std::max(coreLo[a], std::min(hi[a], b.n[a]));
    }

    // Persistent requests amortize setup across iterations. Faces exclude
    // edges/corners because the seven-point stencil never needs them.
    MPI_Datatype types[12];
    int typeCount = 0;
    MPI_Request requests[2][12];
    int requestCount = 0;
    const int sizes[3] = {b.n[0] + 2, b.n[1] + 2, b.n[2] + 2};
    for (int a = 0; a < 3; ++a) {
        int neighbors[2];
        MPI_Cart_shift(cart, a, 1, neighbors, neighbors + 1);
        for (int side = 0; side < 2; ++side) {
            if (neighbors[side] == MPI_PROC_NULL) continue;
            int face[3] = {b.n[0], b.n[1], b.n[2]};
            face[a] = 1;
            int sendStart[3] = {1, 1, 1}, recvStart[3] = {1, 1, 1};
            sendStart[a] = side ? b.n[a] : 1;
            recvStart[a] = side ? b.n[a] + 1 : 0;
            MPI_Datatype& sendType = types[typeCount++];
            MPI_Datatype& recvType = types[typeCount++];
            MPI_Type_create_subarray(3, sizes, face, sendStart, MPI_ORDER_C, MPI_DOUBLE, &sendType);
            MPI_Type_create_subarray(3, sizes, face, recvStart, MPI_ORDER_C, MPI_DOUBLE, &recvType);
            MPI_Type_commit(&sendType);
            MPI_Type_commit(&recvType);
            for (int buffer = 0; buffer < 2; ++buffer) {
                MPI_Recv_init(grids[buffer].data(), 1, recvType, neighbors[side],
                              2 * a + 1 - side, cart, &requests[buffer][requestCount]);
                MPI_Send_init(grids[buffer].data(), 1, sendType, neighbors[side],
                              2 * a + side, cart, &requests[buffer][requestCount + 1]);
            }
            requestCount += 2;
        }
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\n", nx, ny, nz, iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d x %d)\n", ranks, dims[2], dims[1], dims[0]);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        const int buffer = iter % 2;
        const Real* input = grids[buffer].data();
        Real* output = grids[1 - buffer].data();
        if (requestCount) MPI_Startall(requestCount, requests[buffer]);
        updateBox(input, output, sx, sy, coreLo, coreHi);
        if (requestCount) MPI_Waitall(requestCount, requests[buffer], MPI_STATUSES_IGNORE);
        // Six disjoint boxes cover the remaining shell, even for one-cell
        // local blocks, without a conditional in the innermost cell loop.
        int shellLo[3], shellHi[3];
        std::copy_n(lo, 3, shellLo);
        std::copy_n(hi, 3, shellHi);
        for (int a = 0; a < 3; ++a) {
            shellHi[a] = coreLo[a];
            updateBox(input, output, sx, sy, shellLo, shellHi);
            shellLo[a] = coreHi[a];
            shellHi[a] = hi[a];
            updateBox(input, output, sx, sy, shellLo, shellHi);
            shellLo[a] = coreLo[a];
            shellHi[a] = coreHi[a];
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const double updates = double(std::max(0, nx - 2)) * std::max(0, ny - 2)
                             * std::max(0, nz - 2) * iterations;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    const auto& finalGrid = grids[iterations % 2];
    if (printResults) printDistributedResults(finalGrid, b, global, dims, cart, rank, ranks);
    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateDistributed(finalGrid, b, cart, rank);
    }
    for (int buffer = 0; buffer < 2; ++buffer)
        for (int r = 0; r < requestCount; ++r) MPI_Request_free(&requests[buffer][r]);
    for (int t = 0; t < typeCount; ++t) MPI_Type_free(&types[t]);
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
