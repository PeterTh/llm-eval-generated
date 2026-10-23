#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// A rectangular portion of the global grid. Ranges are half-open [start, end).
struct Block {
    size_t x0 = 0, x1 = 0;
    size_t y0 = 0, y1 = 0;
    size_t z0 = 0, z1 = 0;
};

struct Decomposition {
    int fx = 1;
    int fy = 1;
    int fz = 1;
};

// Distribute n elements over nblocks blocks as evenly as possible.
static void splitRange(const size_t n, const int coord, const int nblocks,
                       size_t& start, size_t& end) noexcept {
    const size_t blocks = static_cast<size_t>(nblocks);
    const size_t c = static_cast<size_t>(coord);
    const size_t base = n / blocks;
    const size_t rem = n % blocks;
    start = c * base + std::min(c, rem);
    end = start + base + (c < rem ? 1u : 0u);
}

static Block getBlock(const Decomposition& d, const int cx, const int cy, const int cz,
                      const size_t nx, const size_t ny, const size_t nz) noexcept {
    Block b;
    splitRange(nx, cx, d.fx, b.x0, b.x1);
    splitRange(ny, cy, d.fy, b.y0, b.y1);
    splitRange(nz, cz, d.fz, b.z0, b.z1);
    return b;
}

// Pick a factorization of the rank count over the three grid dimensions.
// The cost model minimizes the largest local volume plus the halo surface.
static Decomposition chooseDecomposition(const size_t nx, const size_t ny, const size_t nz,
                                         const int nranks) {
    Decomposition best;
    if (nranks <= 1) {
        return best;
    }

    std::vector<int> divisors;
    for (int i = 1; i <= nranks / i; ++i) {
        if (nranks % i == 0) {
            divisors.push_back(i);
            if (i != nranks / i) {
                divisors.push_back(nranks / i);
            }
        }
    }

    double bestScore = std::numeric_limits<double>::max();
    int bestActive = std::numeric_limits<int>::max();
    bool bestHasEmpty = true;

    for (const int fz : divisors) {
        const int rem = nranks / fz;
        for (const int fy : divisors) {
            if (fy > rem || rem % fy != 0) {
                continue;
            }
            const int fx = rem / fy;

            const size_t lx = (nx + static_cast<size_t>(fx) - 1) / static_cast<size_t>(fx);
            const size_t ly = (ny + static_cast<size_t>(fy) - 1) / static_cast<size_t>(fy);
            const size_t lz = (nz + static_cast<size_t>(fz) - 1) / static_cast<size_t>(fz);

            const double volume = static_cast<double>(lx) * static_cast<double>(ly) *
                                  static_cast<double>(lz);
            double surface = 0.0;
            if (fx > 1) surface += 2.0 * static_cast<double>(ly) * static_cast<double>(lz);
            if (fy > 1) surface += 2.0 * static_cast<double>(lx) * static_cast<double>(lz);
            if (fz > 1) surface += 2.0 * static_cast<double>(lx) * static_cast<double>(ly);

            const double score = volume + surface;
            const int active = (fx > 1 ? 1 : 0) + (fy > 1 ? 1 : 0) + (fz > 1 ? 1 : 0);
            const bool hasEmpty = static_cast<size_t>(fx) > nx ||
                                  static_cast<size_t>(fy) > ny ||
                                  static_cast<size_t>(fz) > nz;

            if (hasEmpty < bestHasEmpty ||
                (hasEmpty == bestHasEmpty &&
                 (score < bestScore - 1e-9 ||
                  (std::fabs(score - bestScore) <= 1e-9 && active < bestActive)))) {
                bestScore = score;
                bestActive = active;
                bestHasEmpty = hasEmpty;
                best.fx = fx;
                best.fy = fy;
                best.fz = fz;
            }
        }
    }

    if (bestActive == std::numeric_limits<int>::max()) {
        // Should not happen: (1, 1, nranks) is always a valid fallback.
        best.fx = 0;
        best.fy = 0;
        best.fz = 0;
    }
    return best;
}

static size_t blockVolume(const Block& b) noexcept {
    return (b.x1 - b.x0) * (b.y1 - b.y0) * (b.z1 - b.z0);
}

struct HaloDatatypes {
    MPI_Datatype sendX[2]{};
    MPI_Datatype recvX[2]{};
    MPI_Datatype sendY[2]{};
    MPI_Datatype recvY[2]{};
    MPI_Datatype sendZ[2]{};
    MPI_Datatype recvZ[2]{};
};

static MPI_Datatype makeSubarray(const int px, const int py, const int pz,
                                 const int sx, const int sy, const int sz,
                                 const int ax, const int ay, const int az) {
    // The local buffer is laid out with x contiguous (z is the slowest
    // dimension), so MPI's C-order dimensions are (z, y, x).
    const int sizes[3] = {pz, py, px};
    const int subsizes[3] = {sz, sy, sx};
    const int starts[3] = {az, ay, ax};
    MPI_Datatype type = MPI_DATATYPE_NULL;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

static HaloDatatypes createHaloDatatypes(const int lx, const int ly, const int lz) {
    const int px = lx + 2;
    const int py = ly + 2;
    const int pz = lz + 2;

    HaloDatatypes t;
    t.sendX[0] = makeSubarray(px, py, pz, 1, ly, lz, 1, 1, 1);
    t.sendX[1] = makeSubarray(px, py, pz, 1, ly, lz, lx, 1, 1);
    t.recvX[0] = makeSubarray(px, py, pz, 1, ly, lz, 0, 1, 1);
    t.recvX[1] = makeSubarray(px, py, pz, 1, ly, lz, lx + 1, 1, 1);

    t.sendY[0] = makeSubarray(px, py, pz, lx, 1, lz, 1, 1, 1);
    t.sendY[1] = makeSubarray(px, py, pz, lx, 1, lz, 1, ly, 1);
    t.recvY[0] = makeSubarray(px, py, pz, lx, 1, lz, 1, 0, 1);
    t.recvY[1] = makeSubarray(px, py, pz, lx, 1, lz, 1, ly + 1, 1);

    t.sendZ[0] = makeSubarray(px, py, pz, lx, ly, 1, 1, 1, 1);
    t.sendZ[1] = makeSubarray(px, py, pz, lx, ly, 1, 1, 1, lz);
    t.recvZ[0] = makeSubarray(px, py, pz, lx, ly, 1, 1, 1, 0);
    t.recvZ[1] = makeSubarray(px, py, pz, lx, ly, 1, 1, 1, lz + 1);

    return t;
}

static void freeHaloDatatypes(HaloDatatypes& t) {
    MPI_Type_free(&t.sendX[0]);
    MPI_Type_free(&t.sendX[1]);
    MPI_Type_free(&t.recvX[0]);
    MPI_Type_free(&t.recvX[1]);
    MPI_Type_free(&t.sendY[0]);
    MPI_Type_free(&t.sendY[1]);
    MPI_Type_free(&t.recvY[0]);
    MPI_Type_free(&t.recvY[1]);
    MPI_Type_free(&t.sendZ[0]);
    MPI_Type_free(&t.sendZ[1]);
    MPI_Type_free(&t.recvZ[0]);
    MPI_Type_free(&t.recvZ[1]);
}

static void exchangeHalos(Real* data, const HaloDatatypes& t,
                          const int nbrX[2], const int nbrY[2], const int nbrZ[2],
                          MPI_Comm cart) {
    MPI_Request reqs[12];
    int n = 0;

    MPI_Irecv(data, 1, t.recvX[0], nbrX[0], 100, cart, &reqs[n++]);
    MPI_Irecv(data, 1, t.recvX[1], nbrX[1], 100, cart, &reqs[n++]);
    MPI_Irecv(data, 1, t.recvY[0], nbrY[0], 101, cart, &reqs[n++]);
    MPI_Irecv(data, 1, t.recvY[1], nbrY[1], 101, cart, &reqs[n++]);
    MPI_Irecv(data, 1, t.recvZ[0], nbrZ[0], 102, cart, &reqs[n++]);
    MPI_Irecv(data, 1, t.recvZ[1], nbrZ[1], 102, cart, &reqs[n++]);

    MPI_Isend(data, 1, t.sendX[0], nbrX[0], 100, cart, &reqs[n++]);
    MPI_Isend(data, 1, t.sendX[1], nbrX[1], 100, cart, &reqs[n++]);
    MPI_Isend(data, 1, t.sendY[0], nbrY[0], 101, cart, &reqs[n++]);
    MPI_Isend(data, 1, t.sendY[1], nbrY[1], 101, cart, &reqs[n++]);
    MPI_Isend(data, 1, t.sendZ[0], nbrZ[0], 102, cart, &reqs[n++]);
    MPI_Isend(data, 1, t.sendZ[1], nbrZ[1], 102, cart, &reqs[n++]);

    MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE);
}

void initializeGrid(std::vector<Real>& grid, const Block& b,
                    const size_t nx, const size_t ny) {
    const size_t lx = b.x1 - b.x0;
    const size_t ly = b.y1 - b.y0;
    const size_t lz = b.z1 - b.z0;
    const size_t px = lx + 2;
    const size_t py = ly + 2;
    const auto idxL = [px, py](const size_t i, const size_t j, const size_t k) {
        return k * (px * py) + j * px + i;
    };

    for (size_t k = 0; k < lz; ++k) {
        for (size_t j = 0; j < ly; ++j) {
            for (size_t i = 0; i < lx; ++i) {
                const size_t gi = idx3(b.x0 + i, b.y0 + j, b.z0 + k, nx, ny);
                grid[idxL(i + 1, j + 1, k + 1)] = (gi % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation for the locally owned cells.
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Block& b, const size_t nx, const size_t ny, const size_t nz) {
    const int lx = static_cast<int>(b.x1 - b.x0);
    const int ly = static_cast<int>(b.y1 - b.y0);
    const int lz = static_cast<int>(b.z1 - b.z0);
    if (lx == 0 || ly == 0 || lz == 0) {
        return;
    }
    const int px = lx + 2;
    const int py = ly + 2;
    const size_t pxz = static_cast<size_t>(px);
    const size_t pyz = static_cast<size_t>(py);
    const auto idxL = [pxz, pyz](const int i, const int j, const int k) {
        return static_cast<size_t>(k) * (pxz * pyz) +
               static_cast<size_t>(j) * pxz + static_cast<size_t>(i);
    };

    // Copy the cells that lie on the global boundary (they never change).
    for (int k = 1; k <= lz; ++k) {
        const size_t gz = b.z0 + static_cast<size_t>(k - 1);
        const bool zBoundary = (gz == 0 || gz == nz - 1);
        if (zBoundary) {
            for (int j = 1; j <= ly; ++j) {
                const size_t row = idxL(1, j, k);
                std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(row),
                            static_cast<size_t>(lx),
                            output.begin() + static_cast<std::ptrdiff_t>(row));
            }
            continue;
        }

        for (int j = 1; j <= ly; ++j) {
            const size_t gy = b.y0 + static_cast<size_t>(j - 1);
            if (gy == 0 || gy == ny - 1) {
                const size_t row = idxL(1, j, k);
                std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(row),
                            static_cast<size_t>(lx),
                            output.begin() + static_cast<std::ptrdiff_t>(row));
                continue;
            }

            if (b.x0 == 0) {
                const size_t p = idxL(1, j, k);
                output[p] = input[p];
            }
            if (b.x0 + static_cast<size_t>(lx) - 1 == nx - 1) {
                const size_t p = idxL(lx, j, k);
                output[p] = input[p];
            }
        }
    }

    // Compute the global interior cells that this rank owns.
    const int x0 = static_cast<int>(b.x0);
    const int y0 = static_cast<int>(b.y0);
    const int z0 = static_cast<int>(b.z0);
    const int is = std::max(1, 2 - x0);
    const int ie = std::min(lx, static_cast<int>(nx - 1) - x0);
    const int js = std::max(1, 2 - y0);
    const int je = std::min(ly, static_cast<int>(ny - 1) - y0);
    const int ks = std::max(1, 2 - z0);
    const int ke = std::min(lz, static_cast<int>(nz - 1) - z0);

    for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
                const size_t idx = idxL(i, j, k);

                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - pxz];
                const Real back = input[idx + pxz];
                const Real bottom = input[idx - pxz * pyz];
                const Real top = input[idx + pxz * pyz];

                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Gather the distributed owned cells into a global, rank-0 vector.
static std::vector<Real> gatherGrid(const Real* data, const Block& b,
                                    const int px, const int py,
                                    const Decomposition& d,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const int rank, const int nranks, MPI_Comm cart) {
    const size_t lx = b.x1 - b.x0;
    const size_t ly = b.y1 - b.y0;
    const size_t lz = b.z1 - b.z0;
    const size_t localN = lx * ly * lz;

    std::vector<Real> sendbuf(localN > 0 ? localN : 1);
    for (size_t k = 0; k < lz; ++k) {
        for (size_t j = 0; j < ly; ++j) {
            const size_t src = (k + 1) * (static_cast<size_t>(px) * static_cast<size_t>(py)) +
                               (j + 1) * static_cast<size_t>(px) + 1;
            const size_t dst = (k * ly + j) * lx;
            std::copy_n(data + src, lx, sendbuf.begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }

    std::vector<int> counts;
    std::vector<int> displs;
    int total = 0;

    if (rank == 0) {
        counts.resize(static_cast<size_t>(nranks));
        displs.resize(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) {
            int c[3];
            MPI_Cart_coords(cart, r, 3, c);
            const Block rb = getBlock(d, c[0], c[1], c[2], nx, ny, nz);
            const size_t rn = (rb.x1 - rb.x0) * (rb.y1 - rb.y0) * (rb.z1 - rb.z0);
            counts[static_cast<size_t>(r)] = static_cast<int>(rn);
            displs[static_cast<size_t>(r)] = total;
            total += static_cast<int>(rn);
        }
    }

    std::vector<Real> packed;
    if (rank == 0) {
        packed.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(sendbuf.data(), static_cast<int>(localN), MPI_DOUBLE,
                rank == 0 ? packed.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, cart);

    std::vector<Real> global;
    if (rank == 0) {
        global.resize(nx * ny * nz);
        for (int r = 0; r < nranks; ++r) {
            int c[3];
            MPI_Cart_coords(cart, r, 3, c);
            const Block rb = getBlock(d, c[0], c[1], c[2], nx, ny, nz);
            const size_t rlx = rb.x1 - rb.x0;
            const size_t rly = rb.y1 - rb.y0;
            const size_t rlz = rb.z1 - rb.z0;
            const Real* src = packed.data() + displs[static_cast<size_t>(r)];
            for (size_t k = 0; k < rlz; ++k) {
                for (size_t j = 0; j < rly; ++j) {
                    for (size_t i = 0; i < rlx; ++i) {
                        global[idx3(rb.x0 + i, rb.y0 + j, rb.z0 + k, nx, ny)] =
                            src[(k * rly + j) * rlx + i];
                    }
                }
            }
        }
    }

    return global;
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
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    const Decomposition decomp = chooseDecomposition(nx, ny, nz, nranks);
    if (decomp.fx == 0) {
        if (rank == 0) {
            printf("Unable to decompose grid %zu x %zu x %zu over %d MPI ranks\n",
                   nx, ny, nz, nranks);
        }
        MPI_Finalize();
        return 1;
    }

    int dims[3] = {decomp.fx, decomp.fy, decomp.fz};
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    const Block block = getBlock(decomp, coords[0], coords[1], coords[2], nx, ny, nz);

    const size_t lx = block.x1 - block.x0;
    const size_t ly = block.y1 - block.y0;
    const size_t lz = block.z1 - block.z0;
    const size_t localVolume = lx * ly * lz;
    const size_t localSize = (lx + 2) * (ly + 2) * (lz + 2);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    // Allocate local grids (double buffering, including one halo layer per side)
    std::vector<Real> grid1(localSize, 0.0);
    std::vector<Real> grid2(localSize, 0.0);

    initializeGrid(grid1, block, nx, ny);

    int nbrX[2];
    int nbrY[2];
    int nbrZ[2];
    MPI_Cart_shift(cart, 0, 1, &nbrX[0], &nbrX[1]);
    MPI_Cart_shift(cart, 1, 1, &nbrY[0], &nbrY[1]);
    MPI_Cart_shift(cart, 2, 1, &nbrZ[0], &nbrZ[1]);

    // Ranks whose block is empty (possible when there are more ranks than
    // cells along a dimension) do not participate in halo exchange.
    const auto neighborHasCells = [&](const int nbr) {
        if (nbr == MPI_PROC_NULL) {
            return false;
        }
        int nc[3];
        MPI_Cart_coords(cart, nbr, 3, nc);
        return blockVolume(getBlock(decomp, nc[0], nc[1], nc[2], nx, ny, nz)) > 0;
    };

    int* neighborArrays[3] = {nbrX, nbrY, nbrZ};
    for (int* nbrs : neighborArrays) {
        for (int d = 0; d < 2; ++d) {
            if (localVolume == 0 || !neighborHasCells(nbrs[d])) {
                nbrs[d] = MPI_PROC_NULL;
            }
        }
    }

    HaloDatatypes haloTypes{};
    if (localVolume > 0) {
        haloTypes = createHaloDatatypes(static_cast<int>(lx),
                                        static_cast<int>(ly),
                                        static_cast<int>(lz));
    }

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Real>* in = &grid1;
    std::vector<Real>* out = &grid2;

    for (int iter = 0; iter < iterations; ++iter) {
        if (localVolume > 0) {
            exchangeHalos(in->data(), haloTypes, nbrX, nbrY, nbrZ, cart);
            stencilIteration(*in, *out, block, nx, ny, nz);
        }
        std::swap(in, out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (localVolume > 0) {
        freeHaloDatatypes(haloTypes);
    }

    const std::vector<Real>& finalGridLocal = *in;

    long localDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDurationMs);

        // Calculate performance metrics
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (maxDurationMs / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final grid on rank 0 for output and validation.
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid = gatherGrid(finalGridLocal.data(), block,
                               static_cast<int>(lx + 2), static_cast<int>(ly + 2),
                               decomp, nx, ny, nz, rank, nranks, cart);
    }

    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        int localValid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            localValid = validateResult(finalGrid, nx, ny, nz) ? 1 : 0;
            printf("Validation: %s\n", localValid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&localValid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Comm_free(&cart);
        MPI_Finalize();
        return localValid ? 0 : 1;
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
