#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

struct Block {
    size_t begin;
    size_t size;
};

static Block splitBlock(const size_t n, const int parts, const int coord) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    return {static_cast<size_t>(coord) * base + std::min(static_cast<size_t>(coord), extra),
            base + (static_cast<size_t>(coord) < extra)};
}

// Select the feasible Cartesian process grid with the least total interface
// area.  Unlike MPI_Dims_create, this also accounts for non-cubic grids.
static bool chooseProcessGrid(const int ranks, const size_t nx, const size_t ny,
                              const size_t nz, int dims[3]) {
    long double best = std::numeric_limits<long double>::infinity();
    bool found = false;
    for (int dz = 1; dz <= ranks; ++dz) {
        if (ranks % dz != 0 || static_cast<size_t>(dz) > nz) continue;
        const int remaining = ranks / dz;
        for (int dy = 1; dy <= remaining; ++dy) {
            if (remaining % dy != 0 || static_cast<size_t>(dy) > ny) continue;
            const int dx = remaining / dy;
            if (static_cast<size_t>(dx) > nx) continue;
            const long double cost =
                static_cast<long double>(dx - 1) * ny * nz +
                static_cast<long double>(dy - 1) * nx * nz +
                static_cast<long double>(dz - 1) * nx * ny;
            if (cost < best) {
                best = cost;
                dims[0] = dx;
                dims[1] = dy;
                dims[2] = dz;
                found = true;
            }
        }
    }
    return found;
}

class LocalGrid {
public:
    LocalGrid(size_t nx, size_t ny, size_t nz)
        : nx_(nx), ny_(ny), nz_(nz), sx_(nx + 2), sy_(ny + 2),
          values_((nx + 2) * (ny + 2) * (nz + 2)) {}

    size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return (z * sy_ + y) * sx_ + x;
    }
    Real* at(const size_t x, const size_t y, const size_t z) {
        return values_.data() + index(x, y, z);
    }
    const Real* at(const size_t x, const size_t y, const size_t z) const {
        return values_.data() + index(x, y, z);
    }
    Real& operator()(const size_t x, const size_t y, const size_t z) {
        return values_[index(x, y, z)];
    }
    const Real& operator()(const size_t x, const size_t y, const size_t z) const {
        return values_[index(x, y, z)];
    }
    size_t nx() const { return nx_; }
    size_t ny() const { return ny_; }
    size_t nz() const { return nz_; }
    int sx() const { return static_cast<int>(sx_); }
    int plane() const { return static_cast<int>(sx_ * sy_); }

private:
    size_t nx_, ny_, nz_, sx_, sy_;
    std::vector<Real> values_;
};

static void initializeGrid(LocalGrid& grid, const Block& xb, const Block& yb,
                           const Block& zb, const size_t globalNx,
                           const size_t globalNy) {
    for (size_t z = 1; z <= grid.nz(); ++z) {
        const size_t gz = zb.begin + z - 1;
        for (size_t y = 1; y <= grid.ny(); ++y) {
            const size_t gy = yb.begin + y - 1;
            size_t globalIndex = (gz * globalNy + gy) * globalNx + xb.begin;
            for (size_t x = 1; x <= grid.nx(); ++x, ++globalIndex)
                grid(x, y, z) = static_cast<Real>(globalIndex % 19);
        }
    }
}

struct FaceTypes {
    MPI_Datatype x = MPI_DATATYPE_NULL;
    MPI_Datatype y = MPI_DATATYPE_NULL;
    MPI_Datatype z = MPI_DATATYPE_NULL;

    explicit FaceTypes(const LocalGrid& g) {
        MPI_Datatype yColumn;
        MPI_Type_vector(static_cast<int>(g.ny()), 1, g.sx(), MPI_DOUBLE, &yColumn);
        MPI_Type_create_hvector(static_cast<int>(g.nz()), 1,
                                static_cast<MPI_Aint>(g.plane()) * sizeof(Real),
                                yColumn, &x);
        MPI_Type_commit(&x);
        MPI_Type_free(&yColumn);

        MPI_Type_vector(static_cast<int>(g.nz()), static_cast<int>(g.nx()),
                        g.plane(), MPI_DOUBLE, &y);
        MPI_Type_commit(&y);
        MPI_Type_vector(static_cast<int>(g.ny()), static_cast<int>(g.nx()),
                        g.sx(), MPI_DOUBLE, &z);
        MPI_Type_commit(&z);
    }
    ~FaceTypes() {
        release();
    }
    void release() {
        if (x != MPI_DATATYPE_NULL) MPI_Type_free(&x);
        if (y != MPI_DATATYPE_NULL) MPI_Type_free(&y);
        if (z != MPI_DATATYPE_NULL) MPI_Type_free(&z);
    }
};

static int startHaloExchange(LocalGrid& input, const FaceTypes& faces,
                             const int neighbors[6], MPI_Comm comm,
                             MPI_Request requests[12]) {
    int n = 0;
    MPI_Irecv(input.at(0, 1, 1), 1, faces.x, neighbors[0], 10, comm, &requests[n++]);
    MPI_Irecv(input.at(input.nx() + 1, 1, 1), 1, faces.x, neighbors[1], 10, comm, &requests[n++]);
    MPI_Irecv(input.at(1, 0, 1), 1, faces.y, neighbors[2], 11, comm, &requests[n++]);
    MPI_Irecv(input.at(1, input.ny() + 1, 1), 1, faces.y, neighbors[3], 11, comm, &requests[n++]);
    MPI_Irecv(input.at(1, 1, 0), 1, faces.z, neighbors[4], 12, comm, &requests[n++]);
    MPI_Irecv(input.at(1, 1, input.nz() + 1), 1, faces.z, neighbors[5], 12, comm, &requests[n++]);

    MPI_Isend(input.at(1, 1, 1), 1, faces.x, neighbors[0], 10, comm, &requests[n++]);
    MPI_Isend(input.at(input.nx(), 1, 1), 1, faces.x, neighbors[1], 10, comm, &requests[n++]);
    MPI_Isend(input.at(1, 1, 1), 1, faces.y, neighbors[2], 11, comm, &requests[n++]);
    MPI_Isend(input.at(1, input.ny(), 1), 1, faces.y, neighbors[3], 11, comm, &requests[n++]);
    MPI_Isend(input.at(1, 1, 1), 1, faces.z, neighbors[4], 12, comm, &requests[n++]);
    MPI_Isend(input.at(1, 1, input.nz()), 1, faces.z, neighbors[5], 12, comm, &requests[n++]);
    return n;
}

static inline void computeBox(const LocalGrid& input, LocalGrid& output,
                              const size_t x0, const size_t x1,
                              const size_t y0, const size_t y1,
                              const size_t z0, const size_t z1) {
    for (size_t z = z0; z < z1; ++z) {
        for (size_t y = y0; y < y1; ++y) {
            for (size_t x = x0; x < x1; ++x) {
                output(x, y, z) =
                    (input(x, y, z) + input(x - 1, y, z) + input(x + 1, y, z) +
                     input(x, y - 1, z) + input(x, y + 1, z) + input(x, y, z - 1) +
                     input(x, y, z + 1)) / 7.0;
            }
        }
    }
}

static void stencilIteration(LocalGrid& input, LocalGrid& output,
                             const FaceTypes& faces, const int neighbors[6],
                             MPI_Comm comm, const Block& xb, const Block& yb,
                             const Block& zb, const size_t nx, const size_t ny,
                             const size_t nz) {
    MPI_Request requests[12];
    const int requestCount = startHaloExchange(input, faces, neighbors, comm, requests);

    // Exclude fixed global boundaries from the update range.
    const size_t x0 = xb.begin == 0 ? 2 : 1;
    const size_t x1 = xb.begin + xb.size == nx ? xb.size : xb.size + 1;
    const size_t y0 = yb.begin == 0 ? 2 : 1;
    const size_t y1 = yb.begin + yb.size == ny ? yb.size : yb.size + 1;
    const size_t z0 = zb.begin == 0 ? 2 : 1;
    const size_t z1 = zb.begin + zb.size == nz ? zb.size : zb.size + 1;

    // The strictly local core overlaps useful work with halo transfers.
    const size_t cx0 = std::max<size_t>(x0, 2);
    const size_t cx1 = std::max(cx0, std::min(x1, xb.size));
    const size_t cy0 = std::max<size_t>(y0, 2);
    const size_t cy1 = std::max(cy0, std::min(y1, yb.size));
    const size_t cz0 = std::max<size_t>(z0, 2);
    const size_t cz1 = std::max(cz0, std::min(z1, zb.size));
    const bool hasCore = cx0 < cx1 && cy0 < cy1 && cz0 < cz1;
    if (hasCore) computeBox(input, output, cx0, cx1, cy0, cy1, cz0, cz1);

    MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);

    if (!hasCore) {
        computeBox(input, output, x0, x1, y0, y1, z0, z1);
        return;
    }
    // Six non-overlapping boxes cover the halo-dependent shell.
    computeBox(input, output, x0, x1, y0, y1, z0, cz0);
    computeBox(input, output, x0, x1, y0, y1, cz1, z1);
    computeBox(input, output, x0, x1, y0, cy0, cz0, cz1);
    computeBox(input, output, x0, x1, cy1, y1, cz0, cz1);
    computeBox(input, output, x0, cx0, cy0, cy1, cz0, cz1);
    computeBox(input, output, cx1, x1, cy0, cy1, cz0, cz1);
}

static std::vector<Real> gatherGrid(const LocalGrid& local, const Block& xb,
                                    const Block& yb, const Block& zb,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const int dims[3], MPI_Comm comm, const int rank,
                                    const int ranks) {
    const size_t localCountSize = xb.size * yb.size * zb.size;
    if (localCountSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Result block is too large for MPI_Gatherv\n");
        MPI_Abort(comm, 2);
    }
    const int localCount = static_cast<int>(localCountSize);
    std::vector<Real> packed(localCountSize);
    size_t p = 0;
    for (size_t z = 1; z <= zb.size; ++z)
        for (size_t y = 1; y <= yb.size; ++y)
            for (size_t x = 1; x <= xb.size; ++x)
                packed[p++] = local(x, y, z);

    std::vector<int> counts(rank == 0 ? ranks : 0);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    std::vector<int> displacements(rank == 0 ? ranks : 0);
    int total = 0;
    if (rank == 0) {
        for (int r = 0; r < ranks; ++r) {
            displacements[r] = total;
            if (counts[r] > std::numeric_limits<int>::max() - total) {
                std::fprintf(stderr, "Global result is too large for MPI_Gatherv\n");
                MPI_Abort(comm, 2);
            }
            total += counts[r];
        }
    }
    std::vector<Real> gathered(rank == 0 ? static_cast<size_t>(total) : 0);
    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, gathered.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, comm);

    std::vector<Real> global;
    if (rank == 0) {
        global.resize(nx * ny * nz);
        for (int r = 0; r < ranks; ++r) {
            int c[3];
            MPI_Cart_coords(comm, r, 3, c);
            const Block rx = splitBlock(nx, dims[0], c[0]);
            const Block ry = splitBlock(ny, dims[1], c[1]);
            const Block rz = splitBlock(nz, dims[2], c[2]);
            size_t q = static_cast<size_t>(displacements[r]);
            for (size_t z = 0; z < rz.size; ++z)
                for (size_t y = 0; y < ry.size; ++y)
                    for (size_t x = 0; x < rx.size; ++x)
                        global[((rz.begin + z) * ny + ry.begin + y) * nx + rx.begin + x] =
                            gathered[q++];
        }
    }
    return global;
}

static bool validateResult(const LocalGrid& grid, MPI_Comm comm, const int rank) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int localValid = 1;
    for (size_t z = 1; z <= grid.nz(); ++z)
        for (size_t y = 1; y <= grid.ny(); ++y)
            for (size_t x = 1; x <= grid.nx(); ++x) {
                const Real value = grid(x, y, z);
                localValid &= std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
    Real globalMin, globalMax;
    int globalValid;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, comm);
    if (rank == 0) {
        if (!globalValid) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            globalValid = 0;
        }
    }
    MPI_Bcast(&globalValid, 1, MPI_INT, 0, comm);
    return globalValid != 0;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-x") == 0 || std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0 || std::strcmp(argv[i], "-i") == 0) &&
            i + 1 < argc) {
            char* end = nullptr;
            const long long value = std::strtoll(argv[++i], &end, 10);
            if (*end != '\0' || value < 0) parseError = true;
            if (std::strcmp(argv[i - 1], "-x") == 0) nx = static_cast<size_t>(value);
            else if (std::strcmp(argv[i - 1], "-y") == 0) ny = static_cast<size_t>(value);
            else if (std::strcmp(argv[i - 1], "-z") == 0) nz = static_cast<size_t>(value);
            else iterations = static_cast<int>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else parseError = true;
    }
    if (help) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0) parseError = true;
    if (parseError) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Invalid command line or grid dimensions (each must be at least 2).\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int dims[3];
    if (!chooseProcessGrid(ranks, nx, ny, nz, dims)) {
        if (worldRank == 0)
            std::fprintf(stderr, "Cannot map %d ranks onto a %zu x %zu x %zu grid.\n",
                         ranks, nx, ny, nz);
        MPI_Finalize();
        return 1;
    }
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int rank, coords[3];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, coords);
    int neighbors[6];
    MPI_Cart_shift(cart, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(cart, 1, 1, &neighbors[2], &neighbors[3]);
    MPI_Cart_shift(cart, 2, 1, &neighbors[4], &neighbors[5]);

    const Block xb = splitBlock(nx, dims[0], coords[0]);
    const Block yb = splitBlock(ny, dims[1], coords[1]);
    const Block zb = splitBlock(nz, dims[2], coords[2]);
    if (xb.size + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (xb.size + 2) * (yb.size + 2) > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Local grid strides exceed MPI datatype limits.\n");
        MPI_Abort(cart, 2);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\n", iterations);
        std::printf("MPI ranks: %d (%d x %d x %d)\n", ranks, dims[0], dims[1], dims[2]);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing grid...\n");
    }

    LocalGrid grid1(xb.size, yb.size, zb.size);
    LocalGrid grid2(xb.size, yb.size, zb.size);
    initializeGrid(grid1, xb, yb, zb, nx, ny);
    initializeGrid(grid2, xb, yb, zb, nx, ny); // preserves fixed physical boundaries
    FaceTypes faces(grid1);

    if (rank == 0) std::printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0)
            stencilIteration(grid1, grid2, faces, neighbors, cart, xb, yb, zb, nx, ny, nz);
        else
            stencilIteration(grid2, grid1, faces, neighbors, cart, xb, yb, zb, nx, ny, nz);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
        const double mcups = elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    LocalGrid& finalGrid = (iterations & 1) == 0 ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> global = gatherGrid(finalGrid, xb, yb, zb, nx, ny, nz,
                                              dims, cart, rank, ranks);
        if (rank == 0) print_results(global, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, cart, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? 0 : 1;
    }

    faces.release();
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return exitCode;
}
