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
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    // MPI Cartesian order is z, y, x.
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    std::array<int, 6> nbr{}; // x-, x+, y-, y+, z-, z+
    size_t nx = 0, ny = 0, nz = 0;
    size_t x0 = 0, y0 = 0, z0 = 0;
    size_t lx = 0, ly = 0, lz = 0;
    size_t sx = 0, sy = 0;
    MPI_Datatype send_lo[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype send_hi[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype recv_lo[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype recv_hi[3]{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Datatype interior = MPI_DATATYPE_NULL;

    size_t index(size_t x, size_t y, size_t z) const noexcept {
        return (z * sy + y) * sx + x;
    }
};

static void splitDimension(size_t n, int parts, int coord, size_t& begin, size_t& count) {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t extra = n % static_cast<size_t>(parts);
    count = base + (static_cast<size_t>(coord) < extra);
    begin = static_cast<size_t>(coord) * base + std::min(static_cast<size_t>(coord), extra);
}

// Choose an exact process grid constrained by the actual grid dimensions.  The
// score favors load balance first and, among similarly balanced layouts, small
// halo surfaces.
static bool chooseProcessGrid(int processes, size_t nx, size_t ny, size_t nz,
                              std::array<int, 3>& cartDims) {
    double best = std::numeric_limits<double>::infinity();
    int bestX = 0, bestY = 0, bestZ = 0;
    for (int px = 1; px <= processes && static_cast<size_t>(px) <= nx; ++px) {
        if (processes % px != 0) continue;
        const int rem = processes / px;
        for (int py = 1; py <= rem && static_cast<size_t>(py) <= ny; ++py) {
            if (rem % py != 0) continue;
            const int pz = rem / py;
            if (static_cast<size_t>(pz) > nz) continue;
            const double bx = std::ceil(static_cast<double>(nx) / px);
            const double by = std::ceil(static_cast<double>(ny) / py);
            const double bz = std::ceil(static_cast<double>(nz) / pz);
            const double volume = bx * by * bz;
            const double surface = bx * by + bx * bz + by * bz;
            const double score = volume + 0.05 * surface;
            if (score < best) {
                best = score;
                bestX = px;
                bestY = py;
                bestZ = pz;
            }
        }
    }
    if (bestX == 0) return false;
    cartDims = {bestZ, bestY, bestX};
    return true;
}

static MPI_Datatype subarrayType(const Domain& d, int z0, int y0, int x0,
                                 int nz, int ny, int nx) {
    const int sizes[3] = {static_cast<int>(d.lz + 2), static_cast<int>(d.ly + 2),
                          static_cast<int>(d.lx + 2)};
    const int subsizes[3] = {nz, ny, nx};
    const int starts[3] = {z0, y0, x0};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

static bool createDomain(Domain& d, size_t nx, size_t ny, size_t nz) {
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    if (!chooseProcessGrid(d.size, nx, ny, nz, d.dims)) return false;
    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims.data(), periods, 0, &d.comm);
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Cart_coords(d.comm, d.rank, 3, d.coords.data());

    d.nx = nx; d.ny = ny; d.nz = nz;
    splitDimension(nx, d.dims[2], d.coords[2], d.x0, d.lx);
    splitDimension(ny, d.dims[1], d.coords[1], d.y0, d.ly);
    splitDimension(nz, d.dims[0], d.coords[0], d.z0, d.lz);
    d.sx = d.lx + 2;
    d.sy = d.ly + 2;

    MPI_Cart_shift(d.comm, 2, 1, &d.nbr[0], &d.nbr[1]);
    MPI_Cart_shift(d.comm, 1, 1, &d.nbr[2], &d.nbr[3]);
    MPI_Cart_shift(d.comm, 0, 1, &d.nbr[4], &d.nbr[5]);

    // Direction order here is x, y, z.
    d.send_lo[0] = subarrayType(d, 1, 1, 1, static_cast<int>(d.lz), static_cast<int>(d.ly), 1);
    d.send_hi[0] = subarrayType(d, 1, 1, static_cast<int>(d.lx), static_cast<int>(d.lz), static_cast<int>(d.ly), 1);
    d.recv_lo[0] = subarrayType(d, 1, 1, 0, static_cast<int>(d.lz), static_cast<int>(d.ly), 1);
    d.recv_hi[0] = subarrayType(d, 1, 1, static_cast<int>(d.lx + 1), static_cast<int>(d.lz), static_cast<int>(d.ly), 1);

    d.send_lo[1] = subarrayType(d, 1, 1, 1, static_cast<int>(d.lz), 1, static_cast<int>(d.lx));
    d.send_hi[1] = subarrayType(d, 1, static_cast<int>(d.ly), 1, static_cast<int>(d.lz), 1, static_cast<int>(d.lx));
    d.recv_lo[1] = subarrayType(d, 1, 0, 1, static_cast<int>(d.lz), 1, static_cast<int>(d.lx));
    d.recv_hi[1] = subarrayType(d, 1, static_cast<int>(d.ly + 1), 1, static_cast<int>(d.lz), 1, static_cast<int>(d.lx));

    d.send_lo[2] = subarrayType(d, 1, 1, 1, 1, static_cast<int>(d.ly), static_cast<int>(d.lx));
    d.send_hi[2] = subarrayType(d, static_cast<int>(d.lz), 1, 1, 1, static_cast<int>(d.ly), static_cast<int>(d.lx));
    d.recv_lo[2] = subarrayType(d, 0, 1, 1, 1, static_cast<int>(d.ly), static_cast<int>(d.lx));
    d.recv_hi[2] = subarrayType(d, static_cast<int>(d.lz + 1), 1, 1, 1, static_cast<int>(d.ly), static_cast<int>(d.lx));
    d.interior = subarrayType(d, 1, 1, 1, static_cast<int>(d.lz),
                              static_cast<int>(d.ly), static_cast<int>(d.lx));
    return true;
}

static void destroyDomain(Domain& d) {
    for (int axis = 0; axis < 3; ++axis) {
        MPI_Type_free(&d.send_lo[axis]); MPI_Type_free(&d.send_hi[axis]);
        MPI_Type_free(&d.recv_lo[axis]); MPI_Type_free(&d.recv_hi[axis]);
    }
    MPI_Type_free(&d.interior);
    MPI_Comm_free(&d.comm);
}

static void copyPhysicalBoundaries(std::vector<double>& field, const Domain& d) {
    if (d.nbr[0] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
            field[d.index(0, y, z)] = field[d.index(1, y, z)];
    if (d.nbr[1] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
            field[d.index(d.lx + 1, y, z)] = field[d.index(d.lx, y, z)];
    if (d.nbr[2] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.lz; ++z) for (size_t x = 1; x <= d.lx; ++x)
            field[d.index(x, 0, z)] = field[d.index(x, 1, z)];
    if (d.nbr[3] == MPI_PROC_NULL)
        for (size_t z = 1; z <= d.lz; ++z) for (size_t x = 1; x <= d.lx; ++x)
            field[d.index(x, d.ly + 1, z)] = field[d.index(x, d.ly, z)];
    if (d.nbr[4] == MPI_PROC_NULL)
        for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x)
            field[d.index(x, y, 0)] = field[d.index(x, y, 1)];
    if (d.nbr[5] == MPI_PROC_NULL)
        for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x)
            field[d.index(x, y, d.lz + 1)] = field[d.index(x, y, d.lz)];
}

static int beginHaloExchange(std::vector<double>& field, const Domain& d,
                             std::array<MPI_Request, 12>& requests) {
    copyPhysicalBoundaries(field, d);
    int n = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const int lo = d.nbr[2 * axis], hi = d.nbr[2 * axis + 1];
        const int lowTag = 100 + 2 * axis;
        const int highTag = lowTag + 1;
        if (lo != MPI_PROC_NULL)
            MPI_Irecv(field.data(), 1, d.recv_lo[axis], lo, highTag, d.comm, &requests[n++]);
        if (hi != MPI_PROC_NULL)
            MPI_Irecv(field.data(), 1, d.recv_hi[axis], hi, lowTag, d.comm, &requests[n++]);
        if (lo != MPI_PROC_NULL)
            MPI_Isend(field.data(), 1, d.send_lo[axis], lo, lowTag, d.comm, &requests[n++]);
        if (hi != MPI_PROC_NULL)
            MPI_Isend(field.data(), 1, d.send_hi[axis], hi, highTag, d.comm, &requests[n++]);
    }
    return n;
}

template <class Function>
static inline void forCore(const Domain& d, Function&& fn) {
    for (size_t z = 2; z < d.lz; ++z)
        for (size_t y = 2; y < d.ly; ++y)
            for (size_t x = 2; x < d.lx; ++x) fn(x, y, z);
}

// Traverse the complement of the core exactly once, without a branch in the
// large interior loop.
template <class Function>
static inline void forShell(const Domain& d, Function&& fn) {
    for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x) fn(x, y, 1);
    if (d.lz > 1)
        for (size_t y = 1; y <= d.ly; ++y) for (size_t x = 1; x <= d.lx; ++x) fn(x, y, d.lz);
    for (size_t z = 2; z < d.lz; ++z) {
        for (size_t x = 1; x <= d.lx; ++x) fn(x, 1, z);
        if (d.ly > 1) for (size_t x = 1; x <= d.lx; ++x) fn(x, d.ly, z);
        for (size_t y = 2; y < d.ly; ++y) {
            fn(1, y, z);
            if (d.lx > 1) fn(d.lx, y, z);
        }
    }
}

static inline double laplacian(const double* field, const Domain& d,
                               size_t x, size_t y, size_t z) noexcept {
    const size_t i = d.index(x, y, z);
    // Keep the same arithmetic grouping as the original implementation so
    // decomposition does not introduce avoidable floating-point differences.
    const double cxx = field[i + 1] + field[i - 1] - 2.0 * field[i];
    const double cyy = field[i + d.sx] + field[i - d.sx] - 2.0 * field[i];
    const double czz = field[i + d.sx * d.sy] + field[i - d.sx * d.sy] - 2.0 * field[i];
    return cxx + cyy + czz;
}

static void computeChemicalPotential(std::vector<double>& c, std::vector<double>& mu,
                                     const Domain& d, double gamma,
                                     double eAA, double eBB, double eAB) {
    std::array<MPI_Request, 12> requests;
    const int nr = beginHaloExchange(c, d, requests);
    const double* cp = c.data();
    double* mp = mu.data();
    auto point = [&](size_t x, size_t y, size_t z) {
        const size_t i = d.index(x, y, z);
        const double cv = cp[i];
        mp[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
              + 3.0 * cv + cv * cv * cv - gamma * laplacian(cp, d, x, y, z);
    };
    forCore(d, point);
    if (nr) MPI_Waitall(nr, requests.data(), MPI_STATUSES_IGNORE);
    forShell(d, point);
}

static void updateConcentration(std::vector<double>& cnew, const std::vector<double>& cold,
                                std::vector<double>& mu, const Domain& d,
                                double dt, double diffusion) {
    std::array<MPI_Request, 12> requests;
    const int nr = beginHaloExchange(mu, d, requests);
    const double* oldp = cold.data();
    const double* mup = mu.data();
    double* newp = cnew.data();
    auto point = [&](size_t x, size_t y, size_t z) {
        const size_t i = d.index(x, y, z);
        newp[i] = oldp[i] + dt * diffusion * laplacian(mup, d, x, y, z);
    };
    forCore(d, point);
    if (nr) MPI_Waitall(nr, requests.data(), MPI_STATUSES_IGNORE);
    forShell(d, point);
}

static void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t volume = d.nx * d.ny * d.nz;
    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t gz = d.z0 + z - 1;
        for (size_t y = 1; y <= d.ly; ++y) {
            const size_t gy = d.y0 + y - 1;
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t gx = d.x0 + x - 1;
                const size_t linear = gz * (d.nx * d.ny) + gy * d.nx + gx;
                const double pseudo = (((linear + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                c[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c, const Domain& d) {
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    int localBad = 0;
    for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
        for (size_t x = 1; x <= d.lx; ++x) {
            const double value = c[d.index(x, y, z)];
            localBad |= !std::isfinite(value);
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
    double globalMin, globalMax;
    int globalBad;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_LOR, d.comm);
    if (d.rank == 0) {
        if (globalBad) std::printf("Validation failed: found NaN or Inf value\n");
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (!globalBad && (globalMax > 10.0 || globalMin < -10.0))
            std::printf("Validation failed: values out of expected range\n");
    }
    return !globalBad && globalMax <= 10.0 && globalMin >= -10.0;
}

static void unpackBlock(std::vector<double>& global, const std::vector<double>& block,
                        size_t nx, size_t ny, size_t x0, size_t y0, size_t z0,
                        size_t lx, size_t ly, size_t lz) {
    size_t p = 0;
    for (size_t z = 0; z < lz; ++z) for (size_t y = 0; y < ly; ++y)
        for (size_t x = 0; x < lx; ++x)
            global[((z0 + z) * ny + (y0 + y)) * nx + x0 + x] = block[p++];
}

static std::vector<double> packInterior(const std::vector<double>& field, const Domain& d) {
    std::vector<double> packed(d.lx * d.ly * d.lz);
    size_t p = 0;
    for (size_t z = 1; z <= d.lz; ++z) for (size_t y = 1; y <= d.ly; ++y)
        for (size_t x = 1; x <= d.lx; ++x) packed[p++] = field[d.index(x, y, z)];
    return packed;
}

static void printDistributedResults(const std::vector<double>& field, const Domain& d) {
    constexpr int tag = 901;
    if (d.rank != 0) {
        MPI_Send(field.data(), 1, d.interior, 0, tag, d.comm);
        return;
    }
    std::vector<double> global(d.nx * d.ny * d.nz);
    auto own = packInterior(field, d);
    unpackBlock(global, own, d.nx, d.ny, d.x0, d.y0, d.z0, d.lx, d.ly, d.lz);
    for (int source = 1; source < d.size; ++source) {
        int coords[3];
        MPI_Cart_coords(d.comm, source, 3, coords);
        size_t x0, y0, z0, lx, ly, lz;
        splitDimension(d.nx, d.dims[2], coords[2], x0, lx);
        splitDimension(d.ny, d.dims[1], coords[1], y0, ly);
        splitDimension(d.nz, d.dims[0], coords[0], z0, lz);
        std::vector<double> block(lx * ly * lz);
        MPI_Recv(block.data(), static_cast<int>(block.size()), MPI_DOUBLE, source, tag,
                 d.comm, MPI_STATUS_IGNORE);
        unpackBlock(global, block, d.nx, d.ny, x0, y0, z0, lx, ly, lz);
    }
    print_results(global, "Concentration");
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, help = false, parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]); parseError = true; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || parseError) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    const size_t intLimit = static_cast<size_t>(std::numeric_limits<int>::max() - 2);
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > intLimit || ny > intLimit || nz > intLimit ||
        nx > std::numeric_limits<size_t>::max() / ny || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (worldRank == 0) std::fprintf(stderr, "Grid dimensions and iterations must be valid positive ranges.\n");
        MPI_Finalize();
        return 1;
    }

    Domain domain;
    if (!createDomain(domain, nx, ny, nz)) {
        if (worldRank == 0)
            std::fprintf(stderr, "Cannot map %d MPI ranks onto a %zu x %zu x %zu grid without empty subdomains.\n",
                         worldSize, nx, ny, nz);
        MPI_Finalize();
        return 1;
    }
    if (domain.rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI ranks: %d (%d x %d x %d process grid)\n", domain.size,
                    domain.dims[2], domain.dims[1], domain.dims[0]);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    const size_t allocation = (domain.lx + 2) * (domain.ly + 2) * (domain.lz + 2);
    std::vector<double> cold(allocation), cnew(allocation), mu(allocation);
    initializeConcentration(cold, domain);

    if (domain.rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(domain.comm);
    const double start = MPI_Wtime();
    constexpr double dt = 0.01, eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0);
    constexpr double eAB = (2.0 / 9.0), gamma = 0.5, diffusion = 1.0;
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(cold, mu, domain, gamma, eAA, eBB, eAB);
        updateConcentration(cnew, cold, mu, domain, dt, diffusion);
        cold.swap(cnew);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (domain.rank == 0) {
        const double updates = static_cast<double>(nx) * static_cast<double>(ny) *
                               static_cast<double>(nz) * iterations;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
    }

    if (printResults) printDistributedResults(cold, domain);
    bool valid = true;
    if (validate) {
        if (domain.rank == 0) std::printf("Validating result...\n");
        valid = validateResult(cold, domain);
        if (domain.rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    destroyDomain(domain);
    MPI_Finalize();
    return valid ? 0 : 1;
}
