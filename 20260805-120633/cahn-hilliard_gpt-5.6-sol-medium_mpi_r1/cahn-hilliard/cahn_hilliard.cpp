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

struct Domain {
    size_t nx = 0, ny = 0, nz = 0;
    size_t lx = 0, ly = 0, lz = 0;
    size_t ox = 0, oy = 0, oz = 0;
    size_t sx = 0, sy = 0, sz = 0;
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    MPI_Comm comm = MPI_COMM_NULL;
};

inline size_t index3(size_t x, size_t y, size_t z, const Domain& d) noexcept {
    return z * d.sz + y * d.sy + x;
}

static size_t localExtent(size_t n, int coordinate, int parts) {
    const size_t p = static_cast<size_t>(parts);
    return n / p + (static_cast<size_t>(coordinate) < n % p ? 1 : 0);
}

static size_t localOffset(size_t n, int coordinate, int parts) {
    const size_t p = static_cast<size_t>(parts);
    const size_t c = static_cast<size_t>(coordinate);
    return c * (n / p) + std::min(c, n % p);
}

// Choose the Cartesian grid with the lowest surface-to-volume ratio while
// ensuring that every rank owns at least one cell in each direction.
static bool chooseProcessGrid(int ranks, size_t nx, size_t ny, size_t nz,
                              std::array<int, 3>& best) {
    long double bestScore = std::numeric_limits<long double>::infinity();
    for (int px = 1; px <= ranks; ++px) {
        if (ranks % px != 0 || static_cast<size_t>(px) > nx) continue;
        const int yz = ranks / px;
        for (int py = 1; py <= yz; ++py) {
            if (yz % py != 0 || static_cast<size_t>(py) > ny) continue;
            const int pz = yz / py;
            if (static_cast<size_t>(pz) > nz) continue;

            // The first term estimates halo traffic; the second mildly
            // penalizes load imbalance caused by remainder blocks.
            const long double surface = static_cast<long double>(px) / nx
                                      + static_cast<long double>(py) / ny
                                      + static_cast<long double>(pz) / nz;
            const long double largest = static_cast<long double>((nx + px - 1) / px)
                                      * static_cast<long double>((ny + py - 1) / py)
                                      * static_cast<long double>((nz + pz - 1) / pz);
            const long double average = static_cast<long double>(nx) * ny * nz / ranks;
            const long double score = surface * (1.0L + 0.02L * (largest / average - 1.0L));
            if (score < bestScore) {
                bestScore = score;
                best = {px, py, pz};
            }
        }
    }
    return std::isfinite(bestScore);
}

class HaloExchange {
  public:
    explicit HaloExchange(const Domain& domain) : d(domain) {
        int sizes[3] = {static_cast<int>(d.lz + 2), static_cast<int>(d.ly + 2),
                        static_cast<int>(d.lx + 2)};
        int starts[3] = {0, 0, 0};
        int xFace[3] = {static_cast<int>(d.lz), static_cast<int>(d.ly), 1};
        int yFace[3] = {static_cast<int>(d.lz), 1, static_cast<int>(d.lx)};
        int zFace[3] = {1, static_cast<int>(d.ly), static_cast<int>(d.lx)};
        MPI_Type_create_subarray(3, sizes, xFace, starts, MPI_ORDER_C, MPI_DOUBLE, &face[0]);
        MPI_Type_create_subarray(3, sizes, yFace, starts, MPI_ORDER_C, MPI_DOUBLE, &face[1]);
        MPI_Type_create_subarray(3, sizes, zFace, starts, MPI_ORDER_C, MPI_DOUBLE, &face[2]);
        for (MPI_Datatype& type : face) MPI_Type_commit(&type);
        MPI_Cart_shift(d.comm, 0, 1, &neighbor[0], &neighbor[1]);
        MPI_Cart_shift(d.comm, 1, 1, &neighbor[2], &neighbor[3]);
        MPI_Cart_shift(d.comm, 2, 1, &neighbor[4], &neighbor[5]);
    }

    ~HaloExchange() { release(); }

    void release() {
        for (MPI_Datatype& type : face)
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
    }

    void begin(std::vector<double>& field) {
        requestCount = 0;
        double* a = field.data();
        // A low face is sent with the low tag and received by the neighbor as
        // its high halo; conversely for a high face.
        postPair(a + index3(0, 1, 1, d), a + index3(d.lx + 1, 1, 1, d),
                 a + index3(1, 1, 1, d), a + index3(d.lx, 1, 1, d),
                 face[0], neighbor[0], neighbor[1], 10);
        postPair(a + index3(1, 0, 1, d), a + index3(1, d.ly + 1, 1, d),
                 a + index3(1, 1, 1, d), a + index3(1, d.ly, 1, d),
                 face[1], neighbor[2], neighbor[3], 20);
        postPair(a + index3(1, 1, 0, d), a + index3(1, 1, d.lz + 1, d),
                 a + index3(1, 1, 1, d), a + index3(1, 1, d.lz, d),
                 face[2], neighbor[4], neighbor[5], 30);

        // Clamping at a physical boundary is equivalent to copying the
        // boundary cell into its missing-neighbor halo.
        if (neighbor[0] == MPI_PROC_NULL)
            for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
                a[index3(0, y, z, d)] = a[index3(1, y, z, d)];
        if (neighbor[1] == MPI_PROC_NULL)
            for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
                a[index3(d.lx + 1, y, z, d)] = a[index3(d.lx, y, z, d)];
        if (neighbor[2] == MPI_PROC_NULL)
            for (size_t z = 1; z <= d.lz; ++z) for (size_t x = 1; x <= d.lx; ++x)
                a[index3(x, 0, z, d)] = a[index3(x, 1, z, d)];
        if (neighbor[3] == MPI_PROC_NULL)
            for (size_t z = 1; z <= d.lz; ++z) for (size_t x = 1; x <= d.lx; ++x)
                a[index3(x, d.ly + 1, z, d)] = a[index3(x, d.ly, z, d)];
        if (neighbor[4] == MPI_PROC_NULL)
            for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x)
                a[index3(x, y, 0, d)] = a[index3(x, y, 1, d)];
        if (neighbor[5] == MPI_PROC_NULL)
            for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x)
                a[index3(x, y, d.lz + 1, d)] = a[index3(x, y, d.lz, d)];
    }

    void wait() { MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE); }

  private:
    const Domain& d;
    std::array<MPI_Datatype, 3> face{};
    std::array<int, 6> neighbor{};
    std::array<MPI_Request, 12> requests{};
    int requestCount = 0;

    void postPair(double* recvLow, double* recvHigh, double* sendLow, double* sendHigh,
                  MPI_Datatype type, int low, int high, int tag) {
        MPI_Irecv(recvLow, 1, type, low, tag + 1, d.comm, &requests[requestCount++]);
        MPI_Irecv(recvHigh, 1, type, high, tag, d.comm, &requests[requestCount++]);
        MPI_Isend(sendLow, 1, type, low, tag, d.comm, &requests[requestCount++]);
        MPI_Isend(sendHigh, 1, type, high, tag + 1, d.comm, &requests[requestCount++]);
    }
};

inline double laplacian(const std::vector<double>& a, size_t p, const Domain& d) noexcept {
    return (a[p + 1] + a[p - 1] - 2.0 * a[p])
         + (a[p + d.sy] + a[p - d.sy] - 2.0 * a[p])
         + (a[p + d.sz] + a[p - d.sz] - 2.0 * a[p]);
}

inline void chemicalCell(const std::vector<double>& c, std::vector<double>& mu,
                         size_t p, const Domain& d, double gamma,
                         double eAA, double eBB, double eAB) noexcept {
    const double cv = c[p];
    mu[p] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
          + 3.0 * cv + cv * cv * cv - gamma * laplacian(c, p, d);
}

static void computeChemicalRegion(const std::vector<double>& c, std::vector<double>& mu,
                                  const Domain& d, size_t xb, size_t xe,
                                  size_t yb, size_t ye, size_t zb, size_t ze,
                                  double gamma, double eAA, double eBB, double eAB) {
    for (size_t z = zb; z < ze; ++z)
        for (size_t y = yb; y < ye; ++y)
            for (size_t x = xb; x < xe; ++x) {
                const size_t p = index3(x, y, z, d);
                chemicalCell(c, mu, p, d, gamma, eAA, eBB, eAB);
            }
}

static void computeChemicalCore(const std::vector<double>& c, std::vector<double>& mu,
                                const Domain& d, double gamma,
                                double eAA, double eBB, double eAB) {
    const size_t xe = std::max<size_t>(3, d.lx > 1 ? d.lx - 1 : 3);
    const size_t ye = std::max<size_t>(3, d.ly > 1 ? d.ly - 1 : 3);
    const size_t ze = std::max<size_t>(3, d.lz > 1 ? d.lz - 1 : 3);
    if (d.lx >= 5 && d.ly >= 5 && d.lz >= 5)
        computeChemicalRegion(c, mu, d, 3, xe, 3, ye, 3, ze, gamma, eAA, eBB, eAB);
}

static void computeChemicalBoundary(const std::vector<double>& c, std::vector<double>& mu,
                                    const Domain& d, double gamma,
                                    double eAA, double eBB, double eAB) {
    const size_t xb = std::min<size_t>(3, d.lx + 1);
    const size_t yb = std::min<size_t>(3, d.ly + 1);
    const size_t zb = std::min<size_t>(3, d.lz + 1);
    const size_t xe = std::max(xb, d.lx > 1 ? d.lx - 1 : xb);
    const size_t ye = std::max(yb, d.ly > 1 ? d.ly - 1 : yb);
    const size_t ze = std::max(zb, d.lz > 1 ? d.lz - 1 : zb);
    const size_t xEnd = d.lx + 1, yEnd = d.ly + 1, zEnd = d.lz + 1;
    // Six disjoint rectangular regions cover the two-cell shell without a
    // branch over the much larger interior volume.
    computeChemicalRegion(c, mu, d, 1, xEnd, 1, yEnd, 1, zb, gamma, eAA, eBB, eAB);
    computeChemicalRegion(c, mu, d, 1, xEnd, 1, yEnd, ze, zEnd, gamma, eAA, eBB, eAB);
    computeChemicalRegion(c, mu, d, 1, xEnd, 1, yb, zb, ze, gamma, eAA, eBB, eAB);
    computeChemicalRegion(c, mu, d, 1, xEnd, ye, yEnd, zb, ze, gamma, eAA, eBB, eAB);
    computeChemicalRegion(c, mu, d, 1, xb, yb, ye, zb, ze, gamma, eAA, eBB, eAB);
    computeChemicalRegion(c, mu, d, xe, xEnd, yb, ye, zb, ze, gamma, eAA, eBB, eAB);
}

inline void updateCell(std::vector<double>& cnew, const std::vector<double>& cold,
                       const std::vector<double>& mu, size_t p, const Domain& d,
                       double dtD) noexcept {
    cnew[p] = cold[p] + dtD * laplacian(mu, p, d);
}

static void updateRegion(std::vector<double>& cnew, const std::vector<double>& cold,
                         const std::vector<double>& mu, const Domain& d,
                         size_t xb, size_t xe, size_t yb, size_t ye,
                         size_t zb, size_t ze, double dtD) {
    for (size_t z = zb; z < ze; ++z)
        for (size_t y = yb; y < ye; ++y)
            for (size_t x = xb; x < xe; ++x) {
                const size_t p = index3(x, y, z, d);
                updateCell(cnew, cold, mu, p, d, dtD);
            }
}

static void updateCore(std::vector<double>& cnew, const std::vector<double>& cold,
                       const std::vector<double>& mu, const Domain& d, double dtD) {
    if (d.lx >= 5 && d.ly >= 5 && d.lz >= 5)
        updateRegion(cnew, cold, mu, d, 3, d.lx - 1, 3, d.ly - 1, 3, d.lz - 1, dtD);
}

static void updateBoundary(std::vector<double>& cnew, const std::vector<double>& cold,
                           const std::vector<double>& mu, const Domain& d, double dtD) {
    const size_t xb = std::min<size_t>(3, d.lx + 1);
    const size_t yb = std::min<size_t>(3, d.ly + 1);
    const size_t zb = std::min<size_t>(3, d.lz + 1);
    const size_t xe = std::max(xb, d.lx > 1 ? d.lx - 1 : xb);
    const size_t ye = std::max(yb, d.ly > 1 ? d.ly - 1 : yb);
    const size_t ze = std::max(zb, d.lz > 1 ? d.lz - 1 : zb);
    const size_t xEnd = d.lx + 1, yEnd = d.ly + 1, zEnd = d.lz + 1;
    updateRegion(cnew, cold, mu, d, 1, xEnd, 1, yEnd, 1, zb, dtD);
    updateRegion(cnew, cold, mu, d, 1, xEnd, 1, yEnd, ze, zEnd, dtD);
    updateRegion(cnew, cold, mu, d, 1, xEnd, 1, yb, zb, ze, dtD);
    updateRegion(cnew, cold, mu, d, 1, xEnd, ye, yEnd, zb, ze, dtD);
    updateRegion(cnew, cold, mu, d, 1, xb, yb, ye, zb, ze, dtD);
    updateRegion(cnew, cold, mu, d, xe, xEnd, yb, ye, zb, ze, dtD);
}

static void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t volume = d.nx * d.ny * d.nz;
    for (size_t z = 1; z <= d.lz; ++z)
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t gx = d.ox + x - 1;
                const size_t gy = d.oy + y - 1;
                const size_t gz = d.oz + z - 1;
                const size_t linear = gz * (d.nx * d.ny) + gy * d.nx + gx;
                const double pseudo = (((linear + 1) * 1299709) % volume)
                                    / static_cast<double>(volume);
                c[index3(x, y, z, d)] = -1.0 + 2.0 * pseudo;
            }
}

static bool validateResult(const std::vector<double>& c, const Domain& d, int rank) {
    int bad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= d.lz; ++z)
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x) {
                const double value = c[index3(x, y, z, d)];
                bad |= !std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
    int anyBad = 0;
    double globalMin = 0.0, globalMax = 0.0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (rank == 0) {
        if (anyBad) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (!anyBad && (globalMax > 10.0 || globalMin < -10.0))
            std::printf("Validation failed: values out of expected range\n");
    }
    return !anyBad && globalMax <= 10.0 && globalMin >= -10.0;
}

static std::vector<double> gatherResult(const std::vector<double>& c, const Domain& d,
                                        int rank, int ranks) {
    const size_t localSize = d.lx * d.ly * d.lz;
    std::vector<double> packed(localSize);
    size_t q = 0;
    for (size_t z = 1; z <= d.lz; ++z)
        for (size_t y = 1; y <= d.ly; ++y)
            for (size_t x = 1; x <= d.lx; ++x)
                packed[q++] = c[index3(x, y, z, d)];

    const int localCount = static_cast<int>(localSize);
    std::vector<int> counts(rank == 0 ? ranks : 0), displacements(rank == 0 ? ranks : 0);
    MPI_Gather(&localCount, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, d.comm);
    std::vector<double> gathered;
    if (rank == 0) {
        int total = 0;
        for (int r = 0; r < ranks; ++r) {
            displacements[r] = total;
            total += counts[r];
        }
        gathered.resize(static_cast<size_t>(total));
    }
    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, d.comm);

    if (rank != 0) return {};
    std::vector<double> global(d.nx * d.ny * d.nz);
    for (int r = 0; r < ranks; ++r) {
        int coordinates[3];
        MPI_Cart_coords(d.comm, r, 3, coordinates);
        const size_t lx = localExtent(d.nx, coordinates[0], d.dims[0]);
        const size_t ly = localExtent(d.ny, coordinates[1], d.dims[1]);
        const size_t lz = localExtent(d.nz, coordinates[2], d.dims[2]);
        const size_t ox = localOffset(d.nx, coordinates[0], d.dims[0]);
        const size_t oy = localOffset(d.ny, coordinates[1], d.dims[1]);
        const size_t oz = localOffset(d.nz, coordinates[2], d.dims[2]);
        size_t source = static_cast<size_t>(displacements[r]);
        for (size_t z = 0; z < lz; ++z)
            for (size_t y = 0; y < ly; ++y)
                for (size_t x = 0; x < lx; ++x)
                    global[(oz + z) * d.nx * d.ny + (oy + y) * d.nx + ox + x] = gathered[source++];
    }
    return global;
}

static void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else { parseError = true; if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]); }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || parseError) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    const bool invalidSize = nx == 0 || ny == 0 || nz == 0 || iterations < 0
                          || nx > static_cast<size_t>(std::numeric_limits<int>::max() - 2)
                          || ny > static_cast<size_t>(std::numeric_limits<int>::max() - 2)
                          || nz > static_cast<size_t>(std::numeric_limits<int>::max() - 2)
                          || nx > std::numeric_limits<size_t>::max() / ny
                          || nx * ny > std::numeric_limits<size_t>::max() / nz;
    std::array<int, 3> dims{};
    if (invalidSize || !chooseProcessGrid(ranks, nx, ny, nz, dims)) {
        if (worldRank == 0)
            std::fprintf(stderr, "Invalid grid/iteration count, or no nonempty Cartesian decomposition for %d ranks.\n", ranks);
        MPI_Finalize();
        return 1;
    }

    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 0, &cart);
    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    Domain d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.dims = dims;
    d.comm = cart;
    MPI_Cart_coords(cart, rank, 3, d.coords.data());
    d.lx = localExtent(nx, d.coords[0], dims[0]);
    d.ly = localExtent(ny, d.coords[1], dims[1]);
    d.lz = localExtent(nz, d.coords[2], dims[2]);
    d.ox = localOffset(nx, d.coords[0], dims[0]);
    d.oy = localOffset(ny, d.coords[1], dims[1]);
    d.oz = localOffset(nz, d.coords[2], dims[2]);
    d.sy = d.lx + 2;
    d.sz = (d.ly + 2) * d.sy;
    const size_t allocated = (d.lz + 2) * d.sz;
    const size_t globalSize = nx * ny * nz;
    const size_t localCells = d.lx * d.ly * d.lz;
    const int localGatherLimit = localCells > static_cast<size_t>(std::numeric_limits<int>::max());
    int anyGatherLimit = 0;
    MPI_Allreduce(&localGatherLimit, &anyGatherLimit, 1, MPI_INT, MPI_MAX, cart);
    if (printResults && (anyGatherLimit
                        || globalSize > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        if (rank == 0) std::fprintf(stderr, "Result gathering exceeds MPI's count range.\n");
        MPI_Comm_free(&cart);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("MPI process grid: %d x %d x %d (%d ranks)\n", dims[0], dims[1], dims[2], ranks);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    std::vector<double> cold(allocated), cnew(allocated), mu(allocated);
    initializeConcentration(cold, d);
    HaloExchange halos(d);

    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB =  (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int step = 0; step < iterations; ++step) {
        halos.begin(cold);
        computeChemicalCore(cold, mu, d, gamma, eAA, eBB, eAB);
        halos.wait();
        computeChemicalBoundary(cold, mu, d, gamma, eAA, eBB, eAB);

        halos.begin(mu);
        updateCore(cnew, cold, mu, d, dt * diffusion);
        halos.wait();
        updateBoundary(cnew, cold, mu, d, dt * diffusion);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double mcups = maxElapsed > 0.0
                           ? static_cast<double>(globalSize) * iterations / maxElapsed / 1.0e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> global = gatherResult(cold, d, rank, ranks);
        if (rank == 0) print_results(global, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(cold, d, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    // Destroy MPI datatypes before their communicator and MPI itself.
    halos.release();
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return valid ? 0 : 1;
}
