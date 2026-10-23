#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Decomposition {
    int rank = 0;
    int nprocs = 1;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};

    size_t gx = 0, gy = 0, gz = 0;
    size_t sx = 0, sy = 0, sz = 0;
    size_t lx = 0, ly = 0, lz = 0;
    size_t px = 0, py = 0, pz = 0;

    int neighborMinus[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int neighborPlus[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    MPI_Comm cart = MPI_COMM_NULL;
};

struct HaloBuffers {
    std::vector<Real> sendMinus[3];
    std::vector<Real> sendPlus[3];
    std::vector<Real> recvMinus[3];
    std::vector<Real> recvPlus[3];
};

// Partition [0, n) over 'parts' ranks.  When there are more ranks than cells,
// the ranks with zero cells are placed at the high end so that the ranks that
// do own cells always form a contiguous range.
inline size_t startFor(const size_t n, const size_t coord, const size_t parts) {
    if (parts >= n) {
        return std::min(coord, n);
    }
    return (n * coord) / parts;
}

inline size_t endFor(const size_t n, const size_t coord, const size_t parts) {
    if (parts >= n) {
        return std::min(coord + 1, n);
    }
    return (n * (coord + 1)) / parts;
}

// Local index (with one ghost layer on every side) of an owned cell.
inline size_t pidx(const Decomposition& d, const size_t i, const size_t j, const size_t k) {
    return ((k + 1) * d.py + (j + 1)) * d.px + (i + 1);
}

void chooseDims(const int nprocs, const size_t gx, const size_t gy, const size_t gz, int dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;
    if (nprocs == 1) {
        return;
    }

    bool found = false;
    size_t bestActive = 0;
    double bestMetric = 0.0;
    int bestDims[3] = {1, 1, 1};

    // Enumerate factorizations.  First maximize the number of ranks that own
    // at least one cell, then minimize the halo volume
    // (1/dx + 1/dy + 1/dz) among the remaining candidates.
    for (int d0 = 1; d0 <= nprocs; ++d0) {
        if (nprocs % d0 != 0) continue;
        const int rem0 = nprocs / d0;
        for (int d1 = 1; d1 <= rem0; ++d1) {
            if (rem0 % d1 != 0) continue;
            const int d2 = rem0 / d1;
            const int candidate[3] = {d0, d1, d2};
            const size_t e0 = std::min(static_cast<size_t>(d0), gx);
            const size_t e1 = std::min(static_cast<size_t>(d1), gy);
            const size_t e2 = std::min(static_cast<size_t>(d2), gz);
            const size_t active = e0 * e1 * e2;
            const double metric = 1.0 / static_cast<double>(e0) +
                                  1.0 / static_cast<double>(e1) +
                                  1.0 / static_cast<double>(e2);
            if (!found || active > bestActive ||
                (active == bestActive && metric < bestMetric - 1e-12)) {
                found = true;
                bestActive = active;
                bestMetric = metric;
                bestDims[0] = candidate[0];
                bestDims[1] = candidate[1];
                bestDims[2] = candidate[2];
            }
        }
    }

    dims[0] = bestDims[0];
    dims[1] = bestDims[1];
    dims[2] = bestDims[2];
}

void setupDecomposition(Decomposition& d, const size_t nx, const size_t ny, const size_t nz) {
    d.gx = nx;
    d.gy = ny;
    d.gz = nz;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    chooseDims(d.nprocs, nx, ny, nz, d.dims);

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, d.dims, periods, 0, &d.cart);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords);

    const size_t c0 = static_cast<size_t>(d.coords[0]);
    const size_t c1 = static_cast<size_t>(d.coords[1]);
    const size_t c2 = static_cast<size_t>(d.coords[2]);
    const size_t p0 = static_cast<size_t>(d.dims[0]);
    const size_t p1 = static_cast<size_t>(d.dims[1]);
    const size_t p2 = static_cast<size_t>(d.dims[2]);

    d.sx = startFor(nx, c0, p0);
    d.sy = startFor(ny, c1, p1);
    d.sz = startFor(nz, c2, p2);
    const size_t ex = endFor(nx, c0, p0);
    const size_t ey = endFor(ny, c1, p1);
    const size_t ez = endFor(nz, c2, p2);

    d.lx = ex - d.sx;
    d.ly = ey - d.sy;
    d.lz = ez - d.sz;
    d.px = d.lx + 2;
    d.py = d.ly + 2;
    d.pz = d.lz + 2;

    const size_t globalExtents[3] = {nx, ny, nz};

    if (d.lx == 0 || d.ly == 0 || d.lz == 0) {
        for (int dim = 0; dim < 3; ++dim) {
            d.neighborMinus[dim] = MPI_PROC_NULL;
            d.neighborPlus[dim] = MPI_PROC_NULL;
        }
        return;
    }

    for (int dim = 0; dim < 3; ++dim) {
        MPI_Cart_shift(d.cart, dim, 1, &d.neighborMinus[dim], &d.neighborPlus[dim]);
        int neighborCoords[3];
        const int neighbors[2] = {d.neighborMinus[dim], d.neighborPlus[dim]};
        for (int side = 0; side < 2; ++side) {
            if (neighbors[side] == MPI_PROC_NULL) continue;
            MPI_Cart_coords(d.cart, neighbors[side], 3, neighborCoords);
            const size_t nc = static_cast<size_t>(neighborCoords[dim]);
            const size_t np = static_cast<size_t>(d.dims[dim]);
            if (endFor(globalExtents[dim], nc, np) == startFor(globalExtents[dim], nc, np)) {
                if (side == 0) {
                    d.neighborMinus[dim] = MPI_PROC_NULL;
                } else {
                    d.neighborPlus[dim] = MPI_PROC_NULL;
                }
            }
        }
    }
}

void initializeGrid(std::vector<Real>& grid, const Decomposition& d) {
    const size_t gxy = d.gx * d.gy;
    for (size_t k = 0; k < d.lz; ++k) {
        const size_t gz = d.sz + k;
        for (size_t j = 0; j < d.ly; ++j) {
            const size_t gy = d.sy + j;
            size_t p = pidx(d, 0, j, k);
            size_t g = gz * gxy + gy * d.gx + d.sx;
            for (size_t i = 0; i < d.lx; ++i, ++p, ++g) {
                grid[p] = static_cast<Real>(g % 19);
            }
        }
    }
}

void allocateHaloBuffers(const Decomposition& d, HaloBuffers& h) {
    const size_t counts[3] = {
        d.ly * d.lz,
        d.lx * d.lz,
        d.lx * d.ly
    };
    for (int dim = 0; dim < 3; ++dim) {
        h.sendMinus[dim].resize(counts[dim]);
        h.sendPlus[dim].resize(counts[dim]);
        h.recvMinus[dim].resize(counts[dim]);
        h.recvPlus[dim].resize(counts[dim]);
    }
}

void packFace(const std::vector<Real>& a, const Decomposition& d,
              const int dim, const bool upper, Real* dst) {
    const size_t px = d.px;
    const size_t py = d.py;

    if (dim == 0) {
        const size_t pi = (upper ? d.lx - 1 : 0) + 1;
        for (size_t k = 0; k < d.lz; ++k) {
            size_t p = ((k + 1) * py + 1) * px + pi;
            Real* out = dst + k * d.ly;
            for (size_t j = 0; j < d.ly; ++j, p += px) {
                out[j] = a[p];
            }
        }
    } else if (dim == 1) {
        const size_t pj = (upper ? d.ly - 1 : 0) + 1;
        for (size_t k = 0; k < d.lz; ++k) {
            size_t p = ((k + 1) * py + pj) * px + 1;
            Real* out = dst + k * d.lx;
            for (size_t i = 0; i < d.lx; ++i, ++p) {
                out[i] = a[p];
            }
        }
    } else {
        const size_t pk = (upper ? d.lz - 1 : 0) + 1;
        for (size_t j = 0; j < d.ly; ++j) {
            size_t p = (pk * py + (j + 1)) * px + 1;
            Real* out = dst + j * d.lx;
            for (size_t i = 0; i < d.lx; ++i, ++p) {
                out[i] = a[p];
            }
        }
    }
}

void unpackGhost(std::vector<Real>& a, const Decomposition& d,
                 const int dim, const bool upper, const Real* src) {
    const size_t px = d.px;
    const size_t py = d.py;

    if (dim == 0) {
        const size_t pi = upper ? d.lx + 1 : 0;
        for (size_t k = 0; k < d.lz; ++k) {
            size_t p = ((k + 1) * py + 1) * px + pi;
            const Real* in = src + k * d.ly;
            for (size_t j = 0; j < d.ly; ++j, p += px) {
                a[p] = in[j];
            }
        }
    } else if (dim == 1) {
        const size_t pj = upper ? d.ly + 1 : 0;
        for (size_t k = 0; k < d.lz; ++k) {
            size_t p = ((k + 1) * py + pj) * px + 1;
            const Real* in = src + k * d.lx;
            for (size_t i = 0; i < d.lx; ++i, ++p) {
                a[p] = in[i];
            }
        }
    } else {
        const size_t pk = upper ? d.lz + 1 : 0;
        for (size_t j = 0; j < d.ly; ++j) {
            size_t p = (pk * py + (j + 1)) * px + 1;
            const Real* in = src + j * d.lx;
            for (size_t i = 0; i < d.lx; ++i, ++p) {
                a[p] = in[i];
            }
        }
    }
}

void exchangeHalos(std::vector<Real>& a, const Decomposition& d, HaloBuffers& h) {
    if (d.lx == 0 || d.ly == 0 || d.lz == 0) {
        return;
    }

    const int dim0Count = static_cast<int>(d.ly * d.lz);
    const int dim1Count = static_cast<int>(d.lx * d.lz);
    const int dim2Count = static_cast<int>(d.lx * d.ly);
    const int counts[3] = {dim0Count, dim1Count, dim2Count};

    MPI_Request requests[12];
    int nrequests = 0;

    for (int dim = 0; dim < 3; ++dim) {
        const int minus = d.neighborMinus[dim];
        const int plus = d.neighborPlus[dim];
        const int tagMinus = dim * 2;
        const int tagPlus = dim * 2 + 1;
        const int count = counts[dim];

        if (minus != MPI_PROC_NULL) {
            packFace(a, d, dim, false, h.sendMinus[dim].data());
            MPI_Isend(h.sendMinus[dim].data(), count, MPI_DOUBLE, minus,
                      tagMinus, d.cart, &requests[nrequests++]);
            MPI_Irecv(h.recvMinus[dim].data(), count, MPI_DOUBLE, minus,
                      tagPlus, d.cart, &requests[nrequests++]);
        }
        if (plus != MPI_PROC_NULL) {
            packFace(a, d, dim, true, h.sendPlus[dim].data());
            MPI_Isend(h.sendPlus[dim].data(), count, MPI_DOUBLE, plus,
                      tagPlus, d.cart, &requests[nrequests++]);
            MPI_Irecv(h.recvPlus[dim].data(), count, MPI_DOUBLE, plus,
                      tagMinus, d.cart, &requests[nrequests++]);
        }
    }

    MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);

    for (int dim = 0; dim < 3; ++dim) {
        if (d.neighborMinus[dim] != MPI_PROC_NULL) {
            unpackGhost(a, d, dim, false, h.recvMinus[dim].data());
        }
        if (d.neighborPlus[dim] != MPI_PROC_NULL) {
            unpackGhost(a, d, dim, true, h.recvPlus[dim].data());
        }
    }
}

// 7-point stencil computation on the locally owned subdomain.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Decomposition& d) {
    if (d.lx == 0 || d.ly == 0 || d.lz == 0) {
        return;
    }

    const size_t px = d.px;
    const size_t py = d.py;

    // Only locally owned cells that are interior in the global grid are updated.
    const size_t ilo = (d.sx == 0) ? 1 : 0;
    const size_t jlo = (d.sy == 0) ? 1 : 0;
    const size_t klo = (d.sz == 0) ? 1 : 0;
    const size_t ihi = (d.sx + d.lx == d.gx) ? d.lx - 1 : d.lx;
    const size_t jhi = (d.sy + d.ly == d.gy) ? d.ly - 1 : d.ly;
    const size_t khi = (d.sz + d.lz == d.gz) ? d.lz - 1 : d.lz;

    // Copy the physical boundary values, exactly as the original does.
    if (d.sx == 0) {
        for (size_t k = 0; k < d.lz; ++k) {
            for (size_t j = 0; j < d.ly; ++j) {
                const size_t p = pidx(d, 0, j, k);
                output[p] = input[p];
            }
        }
    }
    if (d.sx + d.lx == d.gx) {
        for (size_t k = 0; k < d.lz; ++k) {
            for (size_t j = 0; j < d.ly; ++j) {
                const size_t p = pidx(d, d.lx - 1, j, k);
                output[p] = input[p];
            }
        }
    }
    if (d.sy == 0) {
        for (size_t k = 0; k < d.lz; ++k) {
            for (size_t i = 0; i < d.lx; ++i) {
                const size_t p = pidx(d, i, 0, k);
                output[p] = input[p];
            }
        }
    }
    if (d.sy + d.ly == d.gy) {
        for (size_t k = 0; k < d.lz; ++k) {
            for (size_t i = 0; i < d.lx; ++i) {
                const size_t p = pidx(d, i, d.ly - 1, k);
                output[p] = input[p];
            }
        }
    }
    if (d.sz == 0) {
        for (size_t j = 0; j < d.ly; ++j) {
            for (size_t i = 0; i < d.lx; ++i) {
                const size_t p = pidx(d, i, j, 0);
                output[p] = input[p];
            }
        }
    }
    if (d.sz + d.lz == d.gz) {
        for (size_t j = 0; j < d.ly; ++j) {
            for (size_t i = 0; i < d.lx; ++i) {
                const size_t p = pidx(d, i, j, d.lz - 1);
                output[p] = input[p];
            }
        }
    }

    const size_t zStride = px * py;
    for (size_t k = klo; k < khi; ++k) {
        for (size_t j = jlo; j < jhi; ++j) {
            size_t p = ((k + 1) * py + (j + 1)) * px + 1 + ilo;
            for (size_t i = ilo; i < ihi; ++i, ++p) {
                const Real center = input[p];
                const Real left = input[p - 1];
                const Real right = input[p + 1];
                const Real front = input[p - px];
                const Real back = input[p + px];
                const Real bottom = input[p - zStride];
                const Real top = input[p + zStride];

                output[p] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void gatherGrid(const std::vector<Real>& local, const Decomposition& d,
                std::vector<Real>& global) {
    const int root = 0;
    const size_t lx = d.lx;
    const size_t ly = d.ly;
    const size_t lz = d.lz;

    std::vector<Real> sendbuf(lx * ly * lz);
    for (size_t k = 0; k < lz; ++k) {
        for (size_t j = 0; j < ly; ++j) {
            size_t p = pidx(d, 0, j, k);
            size_t q = (k * ly + j) * lx;
            for (size_t i = 0; i < lx; ++i, ++p, ++q) {
                sendbuf[q] = local[p];
            }
        }
    }

    const int myCount = static_cast<int>(sendbuf.size());
    int* counts = nullptr;
    int* displs = nullptr;

    if (d.rank == root) {
        counts = new int[d.nprocs];
        displs = new int[d.nprocs];
        int total = 0;
        for (int r = 0; r < d.nprocs; ++r) {
            int c[3];
            MPI_Cart_coords(d.cart, r, 3, c);
            const size_t rlx = endFor(d.gx, static_cast<size_t>(c[0]), static_cast<size_t>(d.dims[0])) -
                               startFor(d.gx, static_cast<size_t>(c[0]), static_cast<size_t>(d.dims[0]));
            const size_t rly = endFor(d.gy, static_cast<size_t>(c[1]), static_cast<size_t>(d.dims[1])) -
                               startFor(d.gy, static_cast<size_t>(c[1]), static_cast<size_t>(d.dims[1]));
            const size_t rlz = endFor(d.gz, static_cast<size_t>(c[2]), static_cast<size_t>(d.dims[2])) -
                               startFor(d.gz, static_cast<size_t>(c[2]), static_cast<size_t>(d.dims[2]));
            counts[r] = static_cast<int>(rlx * rly * rlz);
            displs[r] = total;
            total += counts[r];
        }

        std::vector<Real> recvbuf(static_cast<size_t>(total));
        MPI_Gatherv(sendbuf.data(), myCount, MPI_DOUBLE,
                    recvbuf.data(), counts, displs, MPI_DOUBLE, root, d.cart);

        global.assign(d.gx * d.gy * d.gz, 0.0);
        const size_t gxy = d.gx * d.gy;
        for (int r = 0; r < d.nprocs; ++r) {
            int c[3];
            MPI_Cart_coords(d.cart, r, 3, c);
            const size_t rsx = startFor(d.gx, static_cast<size_t>(c[0]), static_cast<size_t>(d.dims[0]));
            const size_t rsy = startFor(d.gy, static_cast<size_t>(c[1]), static_cast<size_t>(d.dims[1]));
            const size_t rsz = startFor(d.gz, static_cast<size_t>(c[2]), static_cast<size_t>(d.dims[2]));
            const size_t rex = endFor(d.gx, static_cast<size_t>(c[0]), static_cast<size_t>(d.dims[0]));
            const size_t rey = endFor(d.gy, static_cast<size_t>(c[1]), static_cast<size_t>(d.dims[1]));
            const size_t rez = endFor(d.gz, static_cast<size_t>(c[2]), static_cast<size_t>(d.dims[2]));
            const size_t rlx = rex - rsx;
            const size_t rly = rey - rsy;
            const size_t rlz = rez - rsz;

            const int offset = displs[r];
            for (size_t k = 0; k < rlz; ++k) {
                for (size_t j = 0; j < rly; ++j) {
                    size_t q = static_cast<size_t>(offset) + (k * rly + j) * rlx;
                    size_t g = (rsz + k) * gxy + (rsy + j) * d.gx + rsx;
                    for (size_t i = 0; i < rlx; ++i, ++q, ++g) {
                        global[g] = recvbuf[q];
                    }
                }
            }
        }
    } else {
        MPI_Gatherv(sendbuf.data(), myCount, MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE, root, d.cart);
    }

    delete[] counts;
    delete[] displs;
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

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    Decomposition d;
    setupDecomposition(d, nx, ny, nz);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate grids (double buffering), with one ghost layer per side.
    const size_t localSize = d.px * d.py * d.pz;
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize
    if (rank == 0) {
        printf("Initializing grid...\n");
    }
    initializeGrid(grid1, d);

    HaloBuffers halo;
    allocateHaloBuffers(d, halo);

    // Run stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(d.cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            exchangeHalos(grid1, d, halo);
            stencilIteration(grid1, grid2, d);
        } else {
            exchangeHalos(grid2, d, halo);
            stencilIteration(grid2, grid1, d);
        }
    }

    MPI_Barrier(d.cart);
    auto end = std::chrono::high_resolution_clock::now();
    long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, d.cart);
    auto duration = std::chrono::milliseconds(maxDuration);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;

    bool valid = true;
    if (printResults || validate) {
        std::vector<Real> finalGrid;
        gatherGrid(finalLocal, d, finalGrid);

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(finalGrid, "Grid");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(finalGrid, nx, ny, nz);
            }
        }
    }

    int validFlag = valid ? 0 : 1;
    MPI_Bcast(&validFlag, 1, MPI_INT, 0, d.cart);
    if (rank == 0) {
        if (validate) {
            if (validFlag == 0) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Comm_free(&d.cart);
    MPI_Finalize();

    return (validate && validFlag != 0) ? 1 : 0;
}
