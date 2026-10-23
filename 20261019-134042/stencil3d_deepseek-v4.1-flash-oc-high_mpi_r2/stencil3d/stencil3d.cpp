#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation for the global (row-major) grid
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 3D index calculation for a local block including one ghost layer on each side
inline constexpr size_t sidx(const size_t i, const size_t j, const size_t k, const size_t gx, const size_t gy) noexcept {
    return (k * gy + j) * gx + i;
}

struct Partition {
    size_t offset;
    size_t size;
};

// Split n elements over p ranks (contiguous blocks, as balanced as possible).
inline Partition partition(const size_t n, const int p, const int coord) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    const size_t c = static_cast<size_t>(coord);
    const size_t sz = base + (c < rem ? 1 : 0);
    const size_t off = c * base + std::min(c, rem);
    return {off, sz};
}

// Choose a balanced 3D process grid that factors P, preferring non-empty blocks.
static void chooseDims(const int P, const size_t nx, const size_t ny, const size_t nz, int out[3]) {
    auto scoreDim = [](const int d, const size_t n) {
        const size_t nd = std::max<size_t>(1, n);
        return 1.0 / static_cast<double>((nd + static_cast<size_t>(d) - 1) / static_cast<size_t>(d));
    };

    double best = 1e300;
    int ba = 1, bb = 1, bc = P;
    double bestFeasible = 1e300;
    int fa = 0, fb = 0, fc = 0;

    for (int a = 1; a <= P; ++a) {
        if (P % a != 0) continue;
        const int remA = P / a;
        for (int b = 1; b <= remA; ++b) {
            if (remA % b != 0) continue;
            const int c = remA / b;
            const double s = scoreDim(a, nx) + scoreDim(b, ny) + scoreDim(c, nz);
            if (s < best) {
                best = s;
                ba = a;
                bb = b;
                bc = c;
            }
            if (a <= static_cast<int>(nx) && b <= static_cast<int>(ny) &&
                c <= static_cast<int>(nz) && s < bestFeasible) {
                bestFeasible = s;
                fa = a;
                fb = b;
                fc = c;
            }
        }
    }

    if (fa != 0) {
        out[0] = fa;
        out[1] = fb;
        out[2] = fc;
    } else {
        out[0] = ba;
        out[1] = bb;
        out[2] = bc;
    }
}

static void initializeLocal(Real* grid,
                            const size_t gx, const size_t gy,
                            const size_t lx, const size_t ly, const size_t lz,
                            const size_t x0, const size_t y0, const size_t z0,
                            const size_t nx, const size_t ny) {
    const size_t nxny = nx * ny;
    for (size_t k = 0; k < lz; ++k) {
        const size_t gz = z0 + k;
        for (size_t j = 0; j < ly; ++j) {
            const size_t gy_ = y0 + j;
            for (size_t i = 0; i < lx; ++i) {
                const size_t gx_ = x0 + i;
                const size_t gidx = gz * nxny + gy_ * nx + gx_;
                grid[sidx(i + 1, j + 1, k + 1, gx, gy)] = static_cast<Real>(gidx % 19) * 1.0;
            }
        }
    }
}

// Exchange the one-cell ghost halos of a local block with the six neighbours.
static void exchangeHalos(Real* grid,
                          const MPI_Comm cart,
                          const size_t lx, const size_t ly, const size_t lz,
                          Real* buf, const size_t maxFace) {
    const size_t gx = lx + 2;
    const size_t gy = ly + 2;

    Real* lo = buf;
    Real* hi = buf + maxFace;
    Real* rlo = buf + 2 * maxFace;
    Real* rhi = buf + 3 * maxFace;

    for (int d = 0; d < 3; ++d) {
        int src = MPI_PROC_NULL;
        int dst = MPI_PROC_NULL;
        MPI_Cart_shift(cart, d, 1, &src, &dst);

        // No neighbours in this dimension: both sides are global boundaries.
        if (src == MPI_PROC_NULL && dst == MPI_PROC_NULL) continue;

        size_t face = 0;
        if (d == 0) {
            face = ly * lz;
            for (size_t k = 0; k < lz; ++k) {
                for (size_t j = 0; j < ly; ++j) {
                    const size_t q = k * ly + j;
                    lo[q] = grid[sidx(1, j + 1, k + 1, gx, gy)];
                    hi[q] = grid[sidx(lx, j + 1, k + 1, gx, gy)];
                }
            }
        } else if (d == 1) {
            face = lx * lz;
            for (size_t k = 0; k < lz; ++k) {
                for (size_t i = 0; i < lx; ++i) {
                    const size_t q = k * lx + i;
                    lo[q] = grid[sidx(i + 1, 1, k + 1, gx, gy)];
                    hi[q] = grid[sidx(i + 1, ly, k + 1, gx, gy)];
                }
            }
        } else {
            face = lx * ly;
            for (size_t j = 0; j < ly; ++j) {
                for (size_t i = 0; i < lx; ++i) {
                    const size_t q = j * lx + i;
                    lo[q] = grid[sidx(i + 1, j + 1, 1, gx, gy)];
                    hi[q] = grid[sidx(i + 1, j + 1, lz, gx, gy)];
                }
            }
        }

        const int count = static_cast<int>(face);
        const int tagLow = 10 + d;
        const int tagHigh = 20 + d;

        // Send our high face to the upper neighbour, receive the lower ghost.
        MPI_Sendrecv(hi, count, MPI_DOUBLE, dst, tagLow,
                     rlo, count, MPI_DOUBLE, src, tagLow,
                     cart, MPI_STATUS_IGNORE);

        // Send our low face to the lower neighbour, receive the upper ghost.
        MPI_Sendrecv(lo, count, MPI_DOUBLE, src, tagHigh,
                     rhi, count, MPI_DOUBLE, dst, tagHigh,
                     cart, MPI_STATUS_IGNORE);

        if (src != MPI_PROC_NULL) {
            if (d == 0) {
                for (size_t k = 0; k < lz; ++k) {
                    for (size_t j = 0; j < ly; ++j) {
                        grid[sidx(0, j + 1, k + 1, gx, gy)] = rlo[k * ly + j];
                    }
                }
            } else if (d == 1) {
                for (size_t k = 0; k < lz; ++k) {
                    for (size_t i = 0; i < lx; ++i) {
                        grid[sidx(i + 1, 0, k + 1, gx, gy)] = rlo[k * lx + i];
                    }
                }
            } else {
                for (size_t j = 0; j < ly; ++j) {
                    for (size_t i = 0; i < lx; ++i) {
                        grid[sidx(i + 1, j + 1, 0, gx, gy)] = rlo[j * lx + i];
                    }
                }
            }
        }

        if (dst != MPI_PROC_NULL) {
            if (d == 0) {
                for (size_t k = 0; k < lz; ++k) {
                    for (size_t j = 0; j < ly; ++j) {
                        grid[sidx(lx + 1, j + 1, k + 1, gx, gy)] = rhi[k * ly + j];
                    }
                }
            } else if (d == 1) {
                for (size_t k = 0; k < lz; ++k) {
                    for (size_t i = 0; i < lx; ++i) {
                        grid[sidx(i + 1, ly + 1, k + 1, gx, gy)] = rhi[k * lx + i];
                    }
                }
            } else {
                for (size_t j = 0; j < ly; ++j) {
                    for (size_t i = 0; i < lx; ++i) {
                        grid[sidx(i + 1, j + 1, lz + 1, gx, gy)] = rhi[j * lx + i];
                    }
                }
            }
        }
    }
}

// 7-point stencil computation on the local block (including ghost-aware edges).
static void stencilLocal(const Real* __restrict__ input,
                         Real* __restrict__ output,
                         const size_t gx, const size_t gy,
                         const size_t lx, const size_t ly, const size_t lz,
                         const size_t x0, const size_t y0, const size_t z0,
                         const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = gx * gy;
    const size_t gxmax = nx - 1;
    const size_t gymax = ny - 1;
    const size_t gzmax = nz - 1;

    for (size_t k = 0; k < lz; ++k) {
        const size_t gz = z0 + k;
        const bool zBoundary = (gz == 0 || gz == gzmax);
        for (size_t j = 0; j < ly; ++j) {
            const size_t gy_ = y0 + j;
            const bool yBoundary = (gy_ == 0 || gy_ == gymax);
            const size_t rowBase = sidx(1, j + 1, k + 1, gx, gy);
            for (size_t i = 0; i < lx; ++i) {
                const size_t gx_ = x0 + i;
                const size_t idx = rowBase + i;
                if (zBoundary || yBoundary || gx_ == 0 || gx_ == gxmax) {
                    output[idx] = input[idx];
                } else {
                    const Real center = input[idx];
                    const Real left = input[idx - 1];
                    const Real right = input[idx + 1];
                    const Real front = input[idx - gx];
                    const Real back = input[idx + gx];
                    const Real bottom = input[idx - plane];
                    const Real top = input[idx + plane];

                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
}

// Gather the distributed blocks into a full global grid on the root rank.
static void gatherGlobal(const Real* grid,
                         const size_t gx, const size_t gy,
                         const size_t lx, const size_t ly, const size_t lz,
                         const int dims[3],
                         const MPI_Comm cart,
                         const int rank, const int nranks,
                         const size_t nx, const size_t ny, const size_t nz,
                         std::vector<Real>& global) {
    std::vector<Real> sendbuf(lx * ly * lz);
    for (size_t k = 0; k < lz; ++k) {
        for (size_t j = 0; j < ly; ++j) {
            for (size_t i = 0; i < lx; ++i) {
                sendbuf[(k * ly + j) * lx + i] = grid[sidx(i + 1, j + 1, k + 1, gx, gy)];
            }
        }
    }

    std::vector<int> counts(static_cast<size_t>(nranks));
    std::vector<int> displs(static_cast<size_t>(nranks));
    size_t total = 0;
    for (int r = 0; r < nranks; ++r) {
        int coords[3];
        MPI_Cart_coords(cart, r, 3, coords);
        const Partition px = partition(nx, dims[0], coords[0]);
        const Partition py = partition(ny, dims[1], coords[1]);
        const Partition pz = partition(nz, dims[2], coords[2]);
        const size_t c = px.size * py.size * pz.size;
        counts[static_cast<size_t>(r)] = static_cast<int>(c);
        displs[static_cast<size_t>(r)] = static_cast<int>(total);
        total += c;
    }

    std::vector<Real> recvbuf;
    if (rank == 0) recvbuf.resize(total);

    MPI_Gatherv(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                counts.data(), displs.data(), MPI_DOUBLE, 0, cart);

    if (rank != 0) return;

    global.assign(nx * ny * nz, static_cast<Real>(0));
    for (int r = 0; r < nranks; ++r) {
        int coords[3];
        MPI_Cart_coords(cart, r, 3, coords);
        const Partition px = partition(nx, dims[0], coords[0]);
        const Partition py = partition(ny, dims[1], coords[1]);
        const Partition pz = partition(nz, dims[2], coords[2]);
        const Real* block = recvbuf.data() + displs[static_cast<size_t>(r)];
        for (size_t k = 0; k < pz.size; ++k) {
            const size_t z = pz.offset + k;
            for (size_t j = 0; j < py.size; ++j) {
                const size_t y = py.offset + j;
                for (size_t i = 0; i < px.size; ++i) {
                    const size_t x = px.offset + i;
                    global[idx3(x, y, z, nx, ny)] = block[(k * py.size + j) * px.size + i];
                }
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

    int dims[3];
    chooseDims(worldSize, nx, ny, nz, dims);

    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &nranks);

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    const Partition px = partition(nx, dims[0], coords[0]);
    const Partition py = partition(ny, dims[1], coords[1]);
    const Partition pz = partition(nz, dims[2], coords[2]);

    const size_t x0 = px.offset;
    const size_t y0 = py.offset;
    const size_t z0 = pz.offset;
    const size_t lx = px.size;
    const size_t ly = py.size;
    const size_t lz = pz.size;
    const size_t gx = lx + 2;
    const size_t gy = ly + 2;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localSize = gx * gy * (lz + 2);
    std::vector<Real> grid1(localSize);
    std::vector<Real> grid2(localSize);

    // Initialize the first grid directly from the global index formula.
    if (rank == 0) printf("Initializing grid...\n");
    initializeLocal(grid1.data(), gx, gy, lx, ly, lz, x0, y0, z0, nx, ny);

    const size_t maxFace = std::max({lx * ly, lx * lz, ly * lz});
    std::vector<Real> haloBuf(4 * maxFace + 4);

    if (rank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? grid1.data() : grid2.data();
        Real* output = (iter % 2 == 0) ? grid2.data() : grid1.data();

        exchangeHalos(input, cart, lx, ly, lz, haloBuf.data(), maxFace);
        stencilLocal(input, output, gx, gy, lx, ly, lz, x0, y0, z0, nx, ny, nz);
    }

    MPI_Barrier(cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance metrics
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const Real* finalLocal = (iterations % 2 == 0) ? grid1.data() : grid2.data();
    std::vector<Real> finalGrid;

    if (printResults || validate) {
        gatherGlobal(finalLocal, gx, gy, lx, ly, lz, dims, cart, rank, nranks, nx, ny, nz, finalGrid);
    }

    if (printResults && rank == 0) {
        print_results(finalGrid, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, cart);
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return exitCode;
}
