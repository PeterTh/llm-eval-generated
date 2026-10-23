#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// Split an axis of length n across `parts` ranks. Rank `rank` gets local points
// [offset, offset + local).
inline void splitAxis(const size_t n, const int parts, const int rank,
                      size_t& local, size_t& offset) noexcept {
    const size_t p = static_cast<size_t>(parts);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = n / p;
    const size_t rem = n % p;
    local = base + (r < rem ? 1 : 0);
    offset = r * base + std::min(r, rem);
}

// Domain decomposition description and per-rank working data.
struct Domain {
    int rank = 0;
    int nprocs = 1;
    int dims[3] = {1, 1, 1};      // decomposition along x, y, z
    int coords[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    int nbr[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                  MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    size_t nx = 0, ny = 0, nz = 0;     // global sizes
    size_t ox = 0, oy = 0, oz = 0;     // global offsets of this rank
    size_t lnx = 0, lny = 0, lnz = 0;  // owned/local sizes

    size_t ex = 0, ey = 0, ez = 0;     // extended sizes (one ghost layer each side)

    std::vector<Real> sendbuf[6];
    std::vector<Real> recvbuf[6];
    int faceCount[6] = {0, 0, 0, 0, 0, 0};
};

inline size_t extIdx(const Domain& d, const size_t i, const size_t j, const size_t k) noexcept {
    return (k * d.ey + j) * d.ex + i;
}

// Build a rank decomposition whose factors are mapped to the largest grid axes.
void chooseDecomposition(const size_t nx, const size_t ny, const size_t nz,
                         const int nprocs, int dims[3]) {
    int factors[3] = {0, 0, 0};
    MPI_Dims_create(nprocs, 3, factors);
    std::sort(factors, factors + 3, std::greater<int>());

    int order[3] = {0, 1, 2};
    const size_t sizes[3] = {nx, ny, nz};
    std::sort(order, order + 3, [&](const int a, const int b) {
        return sizes[a] > sizes[b];
    });

    for (int k = 0; k < 3; ++k) {
        dims[order[k]] = factors[k];
    }
}

void setupDomain(Domain& d, const size_t nx, const size_t ny, const size_t nz) {
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims, periods, 0, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords);

    splitAxis(nx, d.dims[0], d.coords[0], d.lnx, d.ox);
    splitAxis(ny, d.dims[1], d.coords[1], d.lny, d.oy);
    splitAxis(nz, d.dims[2], d.coords[2], d.lnz, d.oz);

    d.ex = d.lnx + 2;
    d.ey = d.lny + 2;
    d.ez = d.lnz + 2;

    // 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
    for (int a = 0; a < 3; ++a) {
        int src = MPI_PROC_NULL, dst = MPI_PROC_NULL;
        MPI_Cart_shift(d.cart, a, 1, &src, &dst);
        d.nbr[2 * a] = src;      // lower neighbour
        d.nbr[2 * a + 1] = dst;  // upper neighbour
    }

    d.faceCount[0] = d.faceCount[1] = static_cast<int>(d.lny * d.lnz);
    d.faceCount[2] = d.faceCount[3] = static_cast<int>(d.lnx * d.lnz);
    d.faceCount[4] = d.faceCount[5] = static_cast<int>(d.lnx * d.lny);

    for (int dir = 0; dir < 6; ++dir) {
        const size_t n = static_cast<size_t>(d.faceCount[dir]);
        d.sendbuf[dir].assign(n > 0 ? n : 1, Real(0));
        d.recvbuf[dir].assign(n > 0 ? n : 1, Real(0));
    }
}

void initializeGrid(std::vector<Real>& grid1, std::vector<Real>& grid2, const Domain& d) {
    for (size_t k = 1; k <= d.lnz; ++k) {
        const size_t gz = d.oz + k - 1;
        for (size_t j = 1; j <= d.lny; ++j) {
            const size_t gy = d.oy + j - 1;
            const size_t base = (gz * d.ny + gy) * d.nx + d.ox;
            for (size_t i = 1; i <= d.lnx; ++i) {
                const size_t gidx = base + (i - 1);
                const Real v = static_cast<Real>(gidx % 19) * 1.0;
                const size_t li = extIdx(d, i, j, k);
                grid1[li] = v;
                grid2[li] = v;
            }
        }
    }
}

void packFace(const int dir, const std::vector<Real>& a, const Domain& d, Real* buf) {
    size_t p = 0;
    if (dir < 2) {
        const size_t i = (dir == 0) ? 1 : d.lnx;
        for (size_t k = 1; k <= d.lnz; ++k) {
            for (size_t j = 1; j <= d.lny; ++j) {
                buf[p++] = a[extIdx(d, i, j, k)];
            }
        }
    } else if (dir < 4) {
        const size_t j = (dir == 2) ? 1 : d.lny;
        for (size_t k = 1; k <= d.lnz; ++k) {
            for (size_t i = 1; i <= d.lnx; ++i) {
                buf[p++] = a[extIdx(d, i, j, k)];
            }
        }
    } else {
        const size_t k = (dir == 4) ? 1 : d.lnz;
        for (size_t j = 1; j <= d.lny; ++j) {
            for (size_t i = 1; i <= d.lnx; ++i) {
                buf[p++] = a[extIdx(d, i, j, k)];
            }
        }
    }
}

void unpackFace(const int dir, const Real* buf, std::vector<Real>& a, const Domain& d) {
    size_t p = 0;
    if (dir < 2) {
        const size_t i = (dir == 0) ? 0 : d.ex - 1;
        for (size_t k = 1; k <= d.lnz; ++k) {
            for (size_t j = 1; j <= d.lny; ++j) {
                a[extIdx(d, i, j, k)] = buf[p++];
            }
        }
    } else if (dir < 4) {
        const size_t j = (dir == 2) ? 0 : d.ey - 1;
        for (size_t k = 1; k <= d.lnz; ++k) {
            for (size_t i = 1; i <= d.lnx; ++i) {
                a[extIdx(d, i, j, k)] = buf[p++];
            }
        }
    } else {
        const size_t k = (dir == 4) ? 0 : d.ez - 1;
        for (size_t j = 1; j <= d.lny; ++j) {
            for (size_t i = 1; i <= d.lnx; ++i) {
                a[extIdx(d, i, j, k)] = buf[p++];
            }
        }
    }
}

struct HaloExchange {
    MPI_Request reqs[12];
    int nreq = 0;
};

// Post the non-blocking exchange of the single ghost layer in all directions.
void postHaloExchange(std::vector<Real>& grid, Domain& d, HaloExchange& h) {
    h.nreq = 0;
    for (int dir = 0; dir < 6; ++dir) {
        if (d.nbr[dir] == MPI_PROC_NULL) continue;
        packFace(dir, grid, d, d.sendbuf[dir].data());
        const int cnt = d.faceCount[dir];
        MPI_Irecv(d.recvbuf[dir].data(), cnt, MPI_DOUBLE, d.nbr[dir],
                  100 + dir / 2, d.cart, &h.reqs[h.nreq++]);
        MPI_Isend(d.sendbuf[dir].data(), cnt, MPI_DOUBLE, d.nbr[dir],
                  100 + dir / 2, d.cart, &h.reqs[h.nreq++]);
    }
}

// Complete the exchange and install the received ghost layers.
void finishHaloExchange(std::vector<Real>& grid, Domain& d, HaloExchange& h) {
    if (h.nreq > 0) {
        MPI_Waitall(h.nreq, h.reqs, MPI_STATUSES_IGNORE);
    }
    for (int dir = 0; dir < 6; ++dir) {
        if (d.nbr[dir] == MPI_PROC_NULL) continue;
        unpackFace(dir, d.recvbuf[dir].data(), grid, d);
    }
    h.nreq = 0;
}

struct Bounds {
    size_t xs, xe, ys, ye, zs, ze;
};

inline bool boundsValid(const Bounds& b) noexcept {
    return b.xs <= b.xe && b.ys <= b.ye && b.zs <= b.ze;
}

// Local indices of global interior points along one axis (start > end when empty).
inline void axisInterior(const size_t n, const size_t off, const size_t len,
                         size_t& start, size_t& end) noexcept {
    if (len == 0) {
        start = 1;
        end = 0;
        return;
    }
    start = 1 + ((off == 0) ? size_t(1) : size_t(0));
    end = len - ((off + len == n) ? size_t(1) : size_t(0));
}

// Local indices of owned cells that are global interior points.
inline Bounds interiorBounds(const Domain& d) noexcept {
    Bounds b;
    axisInterior(d.nx, d.ox, d.lnx, b.xs, b.xe);
    axisInterior(d.ny, d.oy, d.lny, b.ys, b.ye);
    axisInterior(d.nz, d.oz, d.lnz, b.zs, b.ze);
    return b;
}

// 7-point stencil over a rectangular region of the extended array.
// Global boundary values are left untouched (the reference copies them unchanged).
void computeRegion(const std::vector<Real>& input, std::vector<Real>& output,
                   const Domain& d, const Bounds& b) {
    if (!boundsValid(b)) return;

    const size_t sy = d.ex;        // stride between y rows
    const size_t sz = d.ex * d.ey; // stride between z planes

    for (size_t k = b.zs; k <= b.ze; ++k) {
        for (size_t j = b.ys; j <= b.ye; ++j) {
            size_t idx = extIdx(d, b.xs, j, k);
            for (size_t i = b.xs; i <= b.xe; ++i, ++idx) {
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - sy];
                const Real back = input[idx + sy];
                const Real bottom = input[idx - sz];
                const Real top = input[idx + sz];
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Compute the part of the interior that does not touch a ghost layer. This can
// run while the non-blocking halo exchange is still in flight.
void computeOwnedInterior(const std::vector<Real>& input, std::vector<Real>& output,
                          const Domain& d, const Bounds& full, Bounds& safe) {
    safe = full;
    if (d.nbr[0] != MPI_PROC_NULL) safe.xs = std::max(safe.xs, size_t(2));
    if (d.nbr[1] != MPI_PROC_NULL) safe.xe = std::min(safe.xe, d.lnx >= 1 ? d.lnx - 1 : 0);
    if (d.nbr[2] != MPI_PROC_NULL) safe.ys = std::max(safe.ys, size_t(2));
    if (d.nbr[3] != MPI_PROC_NULL) safe.ye = std::min(safe.ye, d.lny >= 1 ? d.lny - 1 : 0);
    if (d.nbr[4] != MPI_PROC_NULL) safe.zs = std::max(safe.zs, size_t(2));
    if (d.nbr[5] != MPI_PROC_NULL) safe.ze = std::min(safe.ze, d.lnz >= 1 ? d.lnz - 1 : 0);

    if (boundsValid(safe)) {
        computeRegion(input, output, d, safe);
    }
}

// Compute the remainder of the interior (the shell adjacent to ghost layers).
void computeRemainingInterior(const std::vector<Real>& input, std::vector<Real>& output,
                              const Domain& d, const Bounds& full, const Bounds& safe) {
    if (!boundsValid(safe)) {
        computeRegion(input, output, d, full);
        return;
    }

    if (safe.zs > full.zs) {
        computeRegion(input, output, d, {full.xs, full.xe, full.ys, full.ye, full.zs, safe.zs - 1});
    }
    if (safe.ze < full.ze) {
        computeRegion(input, output, d, {full.xs, full.xe, full.ys, full.ye, safe.ze + 1, full.ze});
    }
    if (safe.ys > full.ys) {
        computeRegion(input, output, d, {full.xs, full.xe, full.ys, safe.ys - 1, safe.zs, safe.ze});
    }
    if (safe.ye < full.ye) {
        computeRegion(input, output, d, {full.xs, full.xe, safe.ye + 1, full.ye, safe.zs, safe.ze});
    }
    if (safe.xs > full.xs) {
        computeRegion(input, output, d, {full.xs, safe.xs - 1, safe.ys, safe.ye, safe.zs, safe.ze});
    }
    if (safe.xe < full.xe) {
        computeRegion(input, output, d, {safe.xe + 1, full.xe, safe.ys, safe.ye, safe.zs, safe.ze});
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output,
                      Domain& d, HaloExchange& h) {
    const Bounds full = interiorBounds(d);

    Bounds safe{};
    postHaloExchange(input, d, h);
    computeOwnedInterior(input, output, d, full, safe);
    finishHaloExchange(input, d, h);
    computeRemainingInterior(input, output, d, full, safe);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Collect the distributed grid into a full global grid on rank 0.
void gatherGlobalGrid(const std::vector<Real>& localGrid, Domain& d,
                      std::vector<Real>& globalGrid) {
    const size_t localCount = d.lnx * d.lny * d.lnz;
    std::vector<Real> packed(localCount > 0 ? localCount : 1);

    size_t p = 0;
    for (size_t k = 1; k <= d.lnz; ++k) {
        for (size_t j = 1; j <= d.lny; ++j) {
            for (size_t i = 1; i <= d.lnx; ++i) {
                packed[p++] = localGrid[extIdx(d, i, j, k)];
            }
        }
    }

    int myCount = static_cast<int>(localCount);
    std::vector<int> counts;
    std::vector<int> displs;
    if (d.rank == 0) {
        counts.resize(static_cast<size_t>(d.nprocs));
        displs.resize(static_cast<size_t>(d.nprocs));
    }
    MPI_Gather(&myCount, 1, MPI_INT,
               counts.data(), 1, MPI_INT, 0, d.cart);

    long long total = 0;
    if (d.rank == 0) {
        for (int r = 0; r < d.nprocs; ++r) {
            displs[r] = static_cast<int>(total);
            total += counts[r];
        }
        globalGrid.assign(static_cast<size_t>(total) > 0 ? static_cast<size_t>(total) : 1, Real(0));
    }

    std::vector<Real> gathered;
    if (d.rank == 0) gathered.resize(static_cast<size_t>(total) > 0 ? static_cast<size_t>(total) : 1);

    MPI_Gatherv(packed.data(), myCount, MPI_DOUBLE,
                gathered.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.cart);

    if (d.rank != 0) return;

    size_t pos = 0;
    for (int r = 0; r < d.nprocs; ++r) {
        int coords[3];
        MPI_Cart_coords(d.cart, r, 3, coords);
        size_t lnx, lny, lnz, ox, oy, oz;
        splitAxis(d.nx, d.dims[0], coords[0], lnx, ox);
        splitAxis(d.ny, d.dims[1], coords[1], lny, oy);
        splitAxis(d.nz, d.dims[2], coords[2], lnz, oz);

        for (size_t k = 0; k < lnz; ++k) {
            const size_t gz = oz + k;
            for (size_t j = 0; j < lny; ++j) {
                const size_t gy = oy + j;
                const size_t dst = (gz * d.ny + gy) * d.nx + ox;
                if (lnx > 0) {
                    std::memcpy(&globalGrid[dst], &gathered[pos], lnx * sizeof(Real));
                }
                pos += lnx;
            }
        }
    }

    globalGrid.resize(d.nx * d.ny * d.nz);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    Domain d;
    d.nprocs = worldSize;
    chooseDecomposition(nx, ny, nz, worldSize, d.dims);
    setupDomain(d, nx, ny, nz);

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const size_t gridSize = d.ex * d.ey * d.ez;
    std::vector<Real> grid1(gridSize, Real(0));
    std::vector<Real> grid2(gridSize, Real(0));

    initializeGrid(grid1, grid2, d);

    if (worldRank == 0) {
        printf("Running stencil computation...\n");
    }

    std::vector<Real>* cur = &grid1;
    std::vector<Real>* next = &grid2;
    HaloExchange exchange;

    MPI_Barrier(d.cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        stencilIteration(*cur, *next, d, exchange);
        std::swap(cur, next);
    }

    MPI_Barrier(d.cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    long long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, d.cart);

    std::vector<Real> globalGrid;
    if (printResults || validate) {
        gatherGlobalGrid(*cur, d, globalGrid);
    }

    if (worldRank == 0) {
        printf("Computation time: %lld ms\n", globalDuration);

        double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (static_cast<double>(globalDuration) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(globalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (validate) {
        int localCode = exitCode;
        int globalCode = 0;
        MPI_Allreduce(&localCode, &globalCode, 1, MPI_INT, MPI_MAX, d.cart);
        exitCode = globalCode;
    }

    MPI_Comm_free(&d.cart);
    MPI_Finalize();
    return exitCode;
}
