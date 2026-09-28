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

// Physical parameters (compile-time constants so the stencil can be fully optimized)
static constexpr double dx = 1.0;
static constexpr double dy = 1.0;
static constexpr double dz = 1.0;
static constexpr double dt = 0.01;
static constexpr double e_AA = -(2.0 / 9.0);
static constexpr double e_BB = -(2.0 / 9.0);
static constexpr double e_AB = (2.0 / 9.0);
static constexpr double gamma_ = 0.5;
static constexpr double D = 1.0;

// MPI state
static int mpiRank = 0;
static int mpiSize = 1;
static MPI_Comm cartComm = MPI_COMM_NULL;

// Local subdomain geometry (interior extents and extents including the 1-cell halo)
struct Domain {
    size_t lnx = 0, lny = 0, lnz = 0;    // interior sizes
    size_t gnx = 0, gny = 0, gnz = 0;    // sizes including halo (lnx + 2, ...)
    size_t x0 = 0, y0 = 0, z0 = 0;       // global offset of the interior origin
    size_t sx = 0, sy = 0, sz = 0;       // linear strides of the padded array
    size_t paddedSize = 0;
    int neighbor[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                       MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
};

// Neighbor slots: 2*dim + side, dim 0 = x, 1 = y, 2 = z; side 0 = low, 1 = high
enum { XLO = 0, XHI = 1, YLO = 2, YHI = 3, ZLO = 4, ZHI = 5 };

static Domain dom;
static MPI_Datatype faceType[3];     // one face datatype per dimension
static size_t sendOffset[6];
static size_t recvOffset[6];

// Index inside the padded local array
static inline size_t lidx(const size_t x, const size_t y, const size_t z) noexcept {
    return z * dom.sz + y * dom.sy + x * dom.sx;
}

// Block distribution of n cells over p ranks
static inline size_t blockStart(const size_t n, const int p, const int i) noexcept {
    const size_t q = n / static_cast<size_t>(p);
    const size_t r = n % static_cast<size_t>(p);
    const size_t iu = static_cast<size_t>(i);
    return iu * q + std::min(iu, r);
}
static inline size_t blockCount(const size_t n, const int p, const int i) noexcept {
    return blockStart(n, p, i + 1) - blockStart(n, p, i);
}

// Choose a process grid that uses as many ranks as possible and, among those,
// minimizes the per-rank halo surface. Every rank keeps at least one cell; if the
// grid is too small for all ranks, the surplus ranks stay idle.
static void chooseDims(const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    int bestUsed = 0;
    double bestSurface = 0.0;
    dims[0] = dims[1] = dims[2] = 1;

    for (int px = 1; px <= mpiSize; ++px) {
        if (mpiSize % px != 0 || static_cast<size_t>(px) > nx) continue;
        const int remX = mpiSize / px;
        for (int py = 1; py <= remX; ++py) {
            if (remX % py != 0 || static_cast<size_t>(py) > ny) continue;
            const int remY = remX / py;
            for (int pz = 1; pz <= remY; ++pz) {
                if (remY % pz != 0 || static_cast<size_t>(pz) > nz) continue;

                const int used = px * py * pz;
                const double lx = static_cast<double>(nx) / px;
                const double ly = static_cast<double>(ny) / py;
                const double lz = static_cast<double>(nz) / pz;
                // Communicated surface, weighting the strided x-faces slightly higher
                const double surface = 1.25 * ly * lz + ly * lx + lx * lz;

                if (used > bestUsed || (used == bestUsed && surface < bestSurface - 1e-12)) {
                    bestUsed = used;
                    bestSurface = surface;
                    dims[0] = pz;
                    dims[1] = py;
                    dims[2] = px;
                }
            }
        }
    }
}

// Returns false for ranks that are not part of the process grid (grid too small).
static bool setupDomain(const size_t nx, const size_t ny, const size_t nz) {
    int dims[3];
    chooseDims(nx, ny, nz, dims);

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cartComm);
    if (cartComm == MPI_COMM_NULL) return false;

    MPI_Comm_rank(cartComm, &mpiRank);
    MPI_Comm_size(cartComm, &mpiSize);

    int coords[3];
    MPI_Cart_coords(cartComm, mpiRank, 3, coords);

    // Cartesian dimension 0 maps to z, 1 to y, 2 to x
    dom.lnx = blockCount(nx, dims[2], coords[2]);
    dom.lny = blockCount(ny, dims[1], coords[1]);
    dom.lnz = blockCount(nz, dims[0], coords[0]);
    dom.x0 = blockStart(nx, dims[2], coords[2]);
    dom.y0 = blockStart(ny, dims[1], coords[1]);
    dom.z0 = blockStart(nz, dims[0], coords[0]);

    dom.gnx = dom.lnx + 2;
    dom.gny = dom.lny + 2;
    dom.gnz = dom.lnz + 2;
    dom.sx = 1;
    dom.sy = dom.gnx;
    dom.sz = dom.gnx * dom.gny;
    dom.paddedSize = dom.gnx * dom.gny * dom.gnz;

    MPI_Cart_shift(cartComm, 2, 1, &dom.neighbor[XLO], &dom.neighbor[XHI]);
    MPI_Cart_shift(cartComm, 1, 1, &dom.neighbor[YLO], &dom.neighbor[YHI]);
    MPI_Cart_shift(cartComm, 0, 1, &dom.neighbor[ZLO], &dom.neighbor[ZHI]);

    // Face datatypes describing an interior-only face of the padded array
    const int sizes[3] = {static_cast<int>(dom.gnz), static_cast<int>(dom.gny), static_cast<int>(dom.gnx)};
    const int starts[3] = {0, 0, 0};
    const int subsizes[3][3] = {
        {static_cast<int>(dom.lnz), static_cast<int>(dom.lny), 1},                        // x-face
        {static_cast<int>(dom.lnz), 1, static_cast<int>(dom.lnx)},                        // y-face
        {1, static_cast<int>(dom.lny), static_cast<int>(dom.lnx)}                         // z-face
    };
    for (int d = 0; d < 3; ++d) {
        MPI_Type_create_subarray(3, sizes, subsizes[d], starts, MPI_ORDER_C, MPI_DOUBLE, &faceType[d]);
        MPI_Type_commit(&faceType[d]);
    }

    sendOffset[XLO] = lidx(1, 1, 1);
    recvOffset[XLO] = lidx(0, 1, 1);
    sendOffset[XHI] = lidx(dom.lnx, 1, 1);
    recvOffset[XHI] = lidx(dom.lnx + 1, 1, 1);

    sendOffset[YLO] = lidx(1, 1, 1);
    recvOffset[YLO] = lidx(1, 0, 1);
    sendOffset[YHI] = lidx(1, dom.lny, 1);
    recvOffset[YHI] = lidx(1, dom.lny + 1, 1);

    sendOffset[ZLO] = lidx(1, 1, 1);
    recvOffset[ZLO] = lidx(1, 1, 0);
    sendOffset[ZHI] = lidx(1, 1, dom.lnz);
    recvOffset[ZHI] = lidx(1, 1, dom.lnz + 1);
    return true;
}

// Post all halo sends/receives. The 7-point stencil never reads ghost edges/corners,
// so all six directions can be exchanged concurrently.
static inline void startHaloExchange(std::vector<double>& f, MPI_Request* req) {
    for (int s = 0; s < 6; ++s) {
        const int d = s / 2;
        MPI_Irecv(f.data() + recvOffset[s], 1, faceType[d], dom.neighbor[s], s, cartComm, &req[2 * s]);
        MPI_Isend(f.data() + sendOffset[s], 1, faceType[d], dom.neighbor[s], s ^ 1, cartComm, &req[2 * s + 1]);
    }
}

// Clamped (zero-gradient) boundary conditions: mirror the outermost interior plane
// into the ghost plane wherever there is no neighbor rank. This reproduces the
// index-clamping of the serial version exactly.
static void fillPhysicalBoundaries(std::vector<double>& f) {
    double* p = f.data();
    const size_t lnx = dom.lnx, lny = dom.lny, lnz = dom.lnz;

    if (dom.neighbor[XLO] == MPI_PROC_NULL) {
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y) p[lidx(0, y, z)] = p[lidx(1, y, z)];
    }
    if (dom.neighbor[XHI] == MPI_PROC_NULL) {
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y) p[lidx(lnx + 1, y, z)] = p[lidx(lnx, y, z)];
    }
    if (dom.neighbor[YLO] == MPI_PROC_NULL) {
        for (size_t z = 1; z <= lnz; ++z)
            std::memcpy(p + lidx(1, 0, z), p + lidx(1, 1, z), lnx * sizeof(double));
    }
    if (dom.neighbor[YHI] == MPI_PROC_NULL) {
        for (size_t z = 1; z <= lnz; ++z)
            std::memcpy(p + lidx(1, lny + 1, z), p + lidx(1, lny, z), lnx * sizeof(double));
    }
    if (dom.neighbor[ZLO] == MPI_PROC_NULL) {
        for (size_t y = 1; y <= lny; ++y)
            std::memcpy(p + lidx(1, y, 0), p + lidx(1, y, 1), lnx * sizeof(double));
    }
    if (dom.neighbor[ZHI] == MPI_PROC_NULL) {
        for (size_t y = 1; y <= lny; ++y)
            std::memcpy(p + lidx(1, y, lnz + 1), p + lidx(1, y, lnz), lnx * sizeof(double));
    }
}

// Compute Laplacian on the padded array (ghost cells already hold the correct values)
static inline double laplacian(const double* __restrict c, const size_t idx,
                               const size_t sx, const size_t sy, const size_t sz) noexcept {
    const double cc = c[idx];
    const double cxx = (c[idx + sx] + c[idx - sx] - 2.0 * cc) / (dx * dx);
    const double cyy = (c[idx + sy] + c[idx - sy] - 2.0 * cc) / (dy * dy);
    const double czz = (c[idx + sz] + c[idx - sz] - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Chemical potential over the box [x0,x1] x [y0,y1] x [z0,z1] (inclusive, local indices)
static void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                                     const size_t x0, const size_t x1, const size_t y0, const size_t y1,
                                     const size_t z0, const size_t z1) {
    if (x0 > x1 || y0 > y1 || z0 > z1) return;
    const double* __restrict cp = c.data();
    double* __restrict mup = mu.data();
    const size_t sx = dom.sx, sy = dom.sy, sz = dom.sz;

    for (size_t z = z0; z <= z1; ++z) {
        for (size_t y = y0; y <= y1; ++y) {
            const size_t base = z * sz + y * sy;
            for (size_t x = x0; x <= x1; ++x) {
                const size_t idx = base + x;
                const double cv = cp[idx];
                mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                           + 3.0 * cv + cv * cv * cv
                           - gamma_ * laplacian(cp, idx, sx, sy, sz);
            }
        }
    }
}

// Cahn-Hilliard update over the box [x0,x1] x [y0,y1] x [z0,z1] (inclusive, local indices)
static void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu,
                               const size_t x0, const size_t x1, const size_t y0, const size_t y1,
                               const size_t z0, const size_t z1) {
    if (x0 > x1 || y0 > y1 || z0 > z1) return;
    const double* __restrict coldp = cold.data();
    const double* __restrict mup = mu.data();
    double* __restrict cnewp = cnew.data();
    const size_t sx = dom.sx, sy = dom.sy, sz = dom.sz;

    for (size_t z = z0; z <= z1; ++z) {
        for (size_t y = y0; y <= y1; ++y) {
            const size_t base = z * sz + y * sy;
            for (size_t x = x0; x <= x1; ++x) {
                const size_t idx = base + x;
                cnewp[idx] = coldp[idx] + dt * D * laplacian(mup, idx, sx, sy, sz);
            }
        }
    }
}

// Apply a kernel to the halo-dependent shell, i.e. the interior box minus the inner box
template <typename Kernel>
static void applyShell(Kernel&& k) {
    const size_t lnx = dom.lnx, lny = dom.lny, lnz = dom.lnz;

    k(1, lnx, 1, lny, 1, 1);                                            // z low plane
    if (lnz > 1) k(1, lnx, 1, lny, lnz, lnz);                           // z high plane
    if (lnz > 2) {
        k(1, lnx, 1, 1, 2, lnz - 1);                                    // y low
        if (lny > 1) k(1, lnx, lny, lny, 2, lnz - 1);                   // y high
        if (lny > 2) {
            k(1, 1, 2, lny - 1, 2, lnz - 1);                            // x low
            if (lnx > 1) k(lnx, lnx, 2, lny - 1, 2, lnz - 1);           // x high
        }
    }
}

// Initialize concentration field (identical pseudo-random sequence as the serial code)
static void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = 0; z < dom.lnz; ++z) {
        for (size_t y = 0; y < dom.lny; ++y) {
            for (size_t x = 0; x < dom.lnx; ++x) {
                const size_t linear_id = (z + dom.z0) * (nx * ny) + (y + dom.y0) * nx + (x + dom.x0);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[lidx(x + 1, y + 1, z + 1)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    int localBad = 0;
    double minVal = 0.0;
    double maxVal = 0.0;
    bool first = true;

    for (size_t z = 1; z <= dom.lnz; ++z) {
        for (size_t y = 1; y <= dom.lny; ++y) {
            const double* row = c.data() + lidx(1, y, z);
            for (size_t x = 0; x < dom.lnx; ++x) {
                const double val = row[x];
                if (std::isnan(val) || std::isinf(val)) {
                    localBad = 1;
                    continue;
                }
                if (first) {
                    minVal = maxVal = val;
                    first = false;
                } else {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }
    }
    if (first) {
        minVal = std::numeric_limits<double>::infinity();
        maxVal = -std::numeric_limits<double>::infinity();
    }

    int anyBad = 0;
    MPI_Allreduce(&localBad, &anyBad, 1, MPI_INT, MPI_MAX, cartComm);
    if (anyBad) {
        if (mpiRank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double globalMin = 0.0, globalMax = 0.0;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, cartComm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, cartComm);

    if (mpiRank == 0) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);

    if (globalMax > 10.0 || globalMin < -10.0) {
        if (mpiRank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Collect the distributed field into a single globally-ordered array on rank 0
static void gatherField(const std::vector<double>& c, std::vector<double>& global,
                        const size_t nx, const size_t ny, const size_t nz) {
    std::vector<double> packed(dom.lnx * dom.lny * dom.lnz);
    for (size_t z = 0; z < dom.lnz; ++z) {
        for (size_t y = 0; y < dom.lny; ++y) {
            std::memcpy(packed.data() + (z * dom.lny + y) * dom.lnx,
                        c.data() + lidx(1, y + 1, z + 1), dom.lnx * sizeof(double));
        }
    }

    if (mpiRank != 0) {
        const size_t meta[6] = {dom.x0, dom.y0, dom.z0, dom.lnx, dom.lny, dom.lnz};
        MPI_Send(meta, 6, MPI_UNSIGNED_LONG, 0, 100, cartComm);
        MPI_Send(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE, 0, 101, cartComm);
        return;
    }

    global.assign(nx * ny * nz, 0.0);
    auto place = [&](const size_t x0, const size_t y0, const size_t z0,
                     const size_t lx, const size_t ly, const size_t lz, const double* src) {
        for (size_t z = 0; z < lz; ++z) {
            for (size_t y = 0; y < ly; ++y) {
                std::memcpy(global.data() + (z + z0) * (nx * ny) + (y + y0) * nx + x0,
                            src + (z * ly + y) * lx, lx * sizeof(double));
            }
        }
    };
    place(dom.x0, dom.y0, dom.z0, dom.lnx, dom.lny, dom.lnz, packed.data());

    std::vector<double> buf;
    for (int r = 1; r < mpiSize; ++r) {
        size_t meta[6];
        MPI_Recv(meta, 6, MPI_UNSIGNED_LONG, r, 100, cartComm, MPI_STATUS_IGNORE);
        const size_t n = meta[3] * meta[4] * meta[5];
        buf.resize(n);
        MPI_Recv(buf.data(), static_cast<int>(n), MPI_DOUBLE, r, 101, cartComm, MPI_STATUS_IGNORE);
        place(meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], buf.data());
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (mpiRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;

    if (!setupDomain(nx, ny, nz)) {
        // More ranks than the grid can accommodate: this rank stays idle.
        MPI_Finalize();
        return 0;
    }

    // Allocate arrays (padded by one ghost cell in each direction)
    std::vector<double> cold(dom.paddedSize, 0.0);
    std::vector<double> cnew(dom.paddedSize, 0.0);
    std::vector<double> mu(dom.paddedSize, 0.0);

    // Initialize concentration field
    if (mpiRank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);

    // Run simulation
    if (mpiRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cartComm);
    auto start = std::chrono::high_resolution_clock::now();

    MPI_Request requests[12];
    for (int t = 0; t < iterations; ++t) {
        // Chemical potential: exchange c, overlap with the halo-independent interior
        startHaloExchange(cold, requests);
        fillPhysicalBoundaries(cold);
        computeChemicalPotential(cold, mu, 2, dom.lnx - 1, 2, dom.lny - 1, 2, dom.lnz - 1);
        MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
        applyShell([&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            computeChemicalPotential(cold, mu, x0, x1, y0, y1, z0, z1);
        });

        // Concentration update: exchange mu, overlap the same way
        startHaloExchange(mu, requests);
        fillPhysicalBoundaries(mu);
        cahnHilliardUpdate(cnew, cold, mu, 2, dom.lnx - 1, 2, dom.lny - 1, 2, dom.lnz - 1);
        MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
        applyShell([&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            cahnHilliardUpdate(cnew, cold, mu, x0, x1, y0, y1, z0, z1);
        });

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = static_cast<long>(duration.count());
    long elapsedMs = localMs;
    MPI_Allreduce(&localMs, &elapsedMs, 1, MPI_LONG, MPI_MAX, cartComm);

    if (mpiRank == 0) {
        printf("Computation time: %ld ms\n", elapsedMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> global;
        gatherField(cold, global, nx, ny, nz);
        if (mpiRank == 0) print_results(global, "Concentration");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (mpiRank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold);

        if (valid) {
            if (mpiRank == 0) printf("Validation: PASSED\n");
        } else {
            if (mpiRank == 0) printf("Validation: FAILED\n");
            status = 1;
        }
    }

    for (int d = 0; d < 3; ++d) MPI_Type_free(&faceType[d]);
    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return status;
}
