#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// MPI domain decomposition
//
// The global nx x ny x nz grid is split into a 3D Cartesian process grid.
// Each rank owns a block of lnx x lny x lnz cells and stores it with a
// one-cell ghost layer on every side (local array is (lnx+2)(lny+2)(lnz+2)).
// Local cell (x,y,z) with 0<=x<lnx lives at local index (x+1,y+1,z+1).
//
// The original code uses clamped boundary conditions (neighbor index is
// clamped to the grid), i.e. the out-of-range neighbor value equals the
// boundary value itself.  At global boundaries we therefore fill the ghost
// layer with a copy of the adjacent boundary face, which makes the stencil
// kernel branch-free and reproduces the serial arithmetic exactly.
// ---------------------------------------------------------------------------

struct Decomp {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1};     // process grid: [0]=x, [1]=y, [2]=z
    int coords[3] = {0, 0, 0};
    int nbr_lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbr_hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t gstart[3] = {0, 0, 0}; // global start index of local block
    size_t ln[3] = {0, 0, 0};     // local block sizes (lnx, lny, lnz)
    size_t lg[3] = {2, 2, 2};     // local sizes including ghosts
    MPI_Datatype face[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
};

inline size_t lidx(const Decomp& d, const size_t x, const size_t y, const size_t z) noexcept {
    return z * (d.lg[0] * d.lg[1]) + y * d.lg[0] + x;
}

// Block range [start, end) for coordinate k of p parts over n cells
static inline void blockRange(const size_t n, const int p, const int k, size_t& start, size_t& end) {
    start = (n * static_cast<size_t>(k)) / static_cast<size_t>(p);
    end = (n * static_cast<size_t>(k + 1)) / static_cast<size_t>(p);
}

static void setupDecomp(Decomp& d, const size_t nx, const size_t ny, const size_t nz) {
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);

    // Let MPI pick a balanced factorization, then assign the largest process
    // counts to the largest grid dimensions (ties prefer z, then y, then x so
    // the contiguous x direction stays as long as possible for vectorization).
    int pd[3] = {0, 0, 0};
    MPI_Dims_create(d.size, 3, pd);               // non-increasing order
    std::sort(pd, pd + 3, [](int a, int b) { return a > b; });

    const size_t n[3] = {nx, ny, nz};
    int order[3] = {2, 1, 0};                       // candidate dims: z, y, x
    std::stable_sort(order, order + 3, [&](int a, int b) { return n[a] > n[b]; });

    int dims[3];
    for (int i = 0; i < 3; ++i) dims[order[i]] = pd[i];

    // MPI_Cart_create wants dims in the order we index them: we use x,y,z.
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Cart_coords(d.cart, d.rank, 3, d.coords);
    for (int i = 0; i < 3; ++i) {
        d.dims[i] = dims[i];
        size_t s, e;
        blockRange(n[i], d.dims[i], d.coords[i], s, e);
        d.gstart[i] = s;
        d.ln[i] = e - s;
        d.lg[i] = d.ln[i] + 2;
    }

    // Neighbor lookup.  If there are more processes than cells along a
    // dimension, some blocks are empty.  Empty ranks do not take part in the
    // exchange at all; non-empty ranks skip over empty blocks to the nearest
    // non-empty one so that the clamped boundary is applied exactly where the
    // serial code applies it.
    const bool empty = (d.ln[0] == 0 || d.ln[1] == 0 || d.ln[2] == 0);
    for (int i = 0; i < 3; ++i) {
        d.nbr_lo[i] = MPI_PROC_NULL;
        d.nbr_hi[i] = MPI_PROC_NULL;
        if (empty) continue;
        for (int dir = -1; dir <= 1; dir += 2) {
            for (int k = d.coords[i] + dir; k >= 0 && k < d.dims[i]; k += dir) {
                size_t s, e;
                blockRange(n[i], d.dims[i], k, s, e);
                if (e > s) {
                    int cc[3] = {d.coords[0], d.coords[1], d.coords[2]};
                    cc[i] = k;
                    MPI_Cart_rank(d.cart, cc, dir < 0 ? &d.nbr_lo[i] : &d.nbr_hi[i]);
                    break;
                }
            }
        }
    }

    // Face datatypes (describe one layer of the interior block, excluding ghosts)
    const int lnx = static_cast<int>(d.ln[0]);
    const int lny = static_cast<int>(d.ln[1]);
    const int lnz = static_cast<int>(d.ln[2]);
    const int sx = static_cast<int>(d.lg[0]);
    const MPI_Aint sxy = static_cast<MPI_Aint>(d.lg[0] * d.lg[1]) * static_cast<MPI_Aint>(sizeof(double));

    // x-face: lny elements strided by sx, repeated lnz times strided by sxy
    MPI_Datatype col;
    MPI_Type_vector(lny, 1, sx, MPI_DOUBLE, &col);
    MPI_Type_create_hvector(lnz, 1, sxy, col, &d.face[0]);
    MPI_Type_free(&col);
    // y-face: lnz rows of lnx contiguous doubles strided by sx*sy
    MPI_Type_vector(lnz, lnx, sx * static_cast<int>(d.lg[1]), MPI_DOUBLE, &d.face[1]);
    // z-face: lny rows of lnx contiguous doubles strided by sx
    MPI_Type_vector(lny, lnx, sx, MPI_DOUBLE, &d.face[2]);
    for (int i = 0; i < 3; ++i) MPI_Type_commit(&d.face[i]);
}

static void freeDecomp(Decomp& d) {
    for (int i = 0; i < 3; ++i) MPI_Type_free(&d.face[i]);
    MPI_Comm_free(&d.cart);
}

// Copy boundary faces into ghost cells at global domain boundaries (clamped BC)
static void applyClampedBoundaries(const Decomp& d, double* a) {
    const size_t lnx = d.ln[0], lny = d.ln[1], lnz = d.ln[2];
    if (d.nbr_lo[0] == MPI_PROC_NULL)
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y) a[lidx(d, 0, y, z)] = a[lidx(d, 1, y, z)];
    if (d.nbr_hi[0] == MPI_PROC_NULL)
        for (size_t z = 1; z <= lnz; ++z)
            for (size_t y = 1; y <= lny; ++y) a[lidx(d, lnx + 1, y, z)] = a[lidx(d, lnx, y, z)];
    if (d.nbr_lo[1] == MPI_PROC_NULL)
        for (size_t z = 1; z <= lnz; ++z)
            std::memcpy(&a[lidx(d, 1, 0, z)], &a[lidx(d, 1, 1, z)], lnx * sizeof(double));
    if (d.nbr_hi[1] == MPI_PROC_NULL)
        for (size_t z = 1; z <= lnz; ++z)
            std::memcpy(&a[lidx(d, 1, lny + 1, z)], &a[lidx(d, 1, lny, z)], lnx * sizeof(double));
    if (d.nbr_lo[2] == MPI_PROC_NULL)
        for (size_t y = 1; y <= lny; ++y)
            std::memcpy(&a[lidx(d, 1, y, 0)], &a[lidx(d, 1, y, 1)], lnx * sizeof(double));
    if (d.nbr_hi[2] == MPI_PROC_NULL)
        for (size_t y = 1; y <= lny; ++y)
            std::memcpy(&a[lidx(d, 1, y, lnz + 1)], &a[lidx(d, 1, y, lnz)], lnx * sizeof(double));
}

// Start nonblocking halo exchange of the interior faces of `a` (12 requests)
static void startHaloExchange(const Decomp& d, double* a, MPI_Request* req) {
    const size_t lnx = d.ln[0], lny = d.ln[1], lnz = d.ln[2];
    // Send/recv addresses: [dim][lo/hi]
    const size_t send_lo[3] = {lidx(d, 1, 1, 1), lidx(d, 1, 1, 1), lidx(d, 1, 1, 1)};
    const size_t send_hi[3] = {lidx(d, lnx, 1, 1), lidx(d, 1, lny, 1), lidx(d, 1, 1, lnz)};
    const size_t recv_lo[3] = {lidx(d, 0, 1, 1), lidx(d, 1, 0, 1), lidx(d, 1, 1, 0)};
    const size_t recv_hi[3] = {lidx(d, lnx + 1, 1, 1), lidx(d, 1, lny + 1, 1), lidx(d, 1, 1, lnz + 1)};
    int r = 0;
    for (int i = 0; i < 3; ++i) {
        MPI_Irecv(a + recv_lo[i], 1, d.face[i], d.nbr_lo[i], 2 * i + 1, d.cart, &req[r++]);
        MPI_Irecv(a + recv_hi[i], 1, d.face[i], d.nbr_hi[i], 2 * i, d.cart, &req[r++]);
    }
    for (int i = 0; i < 3; ++i) {
        MPI_Isend(a + send_lo[i], 1, d.face[i], d.nbr_lo[i], 2 * i, d.cart, &req[r++]);
        MPI_Isend(a + send_hi[i], 1, d.face[i], d.nbr_hi[i], 2 * i + 1, d.cart, &req[r++]);
    }
}

// Compute 7-point Laplacian on the ghosted local array at local index i
template <typename F>
static inline void forBox(const Decomp& d, size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1, F&& f) {
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    for (size_t z = z0; z < z1; ++z)
        for (size_t y = y0; y < y1; ++y)
            f(lidx(d, x0, y, z), x1 - x0);
}

// Chemical potential over a local box (ghosted coordinates)
static void computeChemicalPotentialBox(const Decomp& d, const double* __restrict c, double* __restrict mu,
                                        size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t sx = 1, sy = d.lg[0], sz = d.lg[0] * d.lg[1];
    forBox(d, x0, x1, y0, y1, z0, z1, [&](const size_t base, const size_t len) {
        for (size_t i = base; i < base + len; ++i) {
            const double cv = c[i];
            const double cxx = (c[i + sx] + c[i - sx] - 2.0 * cv) / (dx * dx);
            const double cyy = (c[i + sy] + c[i - sy] - 2.0 * cv) / (dy * dy);
            const double czz = (c[i + sz] + c[i - sz] - 2.0 * cv) / (dz * dz);
            mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv
                    - gamma * (cxx + cyy + czz);
        }
    });
}

// Cahn-Hilliard update over a local box (ghosted coordinates)
static void cahnHilliardUpdateBox(const Decomp& d, double* __restrict cnew, const double* __restrict cold,
                                  const double* __restrict mu,
                                  size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1,
                                  const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t sx = 1, sy = d.lg[0], sz = d.lg[0] * d.lg[1];
    forBox(d, x0, x1, y0, y1, z0, z1, [&](const size_t base, const size_t len) {
        for (size_t i = base; i < base + len; ++i) {
            const double mv = mu[i];
            const double mxx = (mu[i + sx] + mu[i - sx] - 2.0 * mv) / (dx * dx);
            const double myy = (mu[i + sy] + mu[i - sy] - 2.0 * mv) / (dy * dy);
            const double mzz = (mu[i + sz] + mu[i - sz] - 2.0 * mv) / (dz * dz);
            cnew[i] = cold[i] + dt * D * (mxx + myy + mzz);
        }
    });
}

// Apply a box kernel to the inner region (cells not touching any ghost), or
// to the outer shell of the local block (cells adjacent to a ghost layer).
template <typename Kernel>
static void applyInner(const Decomp& d, Kernel&& k) {
    const size_t lnx = d.ln[0], lny = d.ln[1], lnz = d.ln[2];
    k(2, lnx, 2, lny, 2, lnz);
}

template <typename Kernel>
static void applyShell(const Decomp& d, Kernel&& k) {
    const size_t lnx = d.ln[0], lny = d.ln[1], lnz = d.ln[2];
    if (lnx == 0 || lny == 0 || lnz == 0) return;
    // x faces (full y,z extent)
    k(1, 2, 1, lny + 1, 1, lnz + 1);
    k(std::max<size_t>(2, lnx), lnx + 1, 1, lny + 1, 1, lnz + 1);
    // y faces (x in inner range)
    k(2, lnx, 1, 2, 1, lnz + 1);
    k(2, lnx, std::max<size_t>(2, lny), lny + 1, 1, lnz + 1);
    // z faces (x,y in inner range)
    k(2, lnx, 2, lny, 1, 2);
    k(2, lnx, 2, lny, std::max<size_t>(2, lnz), lnz + 1);
}

// Initialize concentration field (local block, same values as the serial code)
static void initializeConcentration(const Decomp& d, std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < d.ln[2]; ++z) {
        const size_t gz = d.gstart[2] + z;
        for (size_t y = 0; y < d.ln[1]; ++y) {
            const size_t gy = d.gstart[1] + y;
            for (size_t x = 0; x < d.ln[0]; ++x) {
                const size_t gx = d.gstart[0] + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[lidx(d, x + 1, y + 1, z + 1)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather the distributed field into a global (nx*ny*nz) array on rank 0
static void gatherField(const Decomp& d, const std::vector<double>& local, std::vector<double>& global,
                        const size_t nx, const size_t ny, const size_t nz) {
    const size_t lcount = d.ln[0] * d.ln[1] * d.ln[2];
    std::vector<double> packed(lcount);
    for (size_t z = 0; z < d.ln[2]; ++z)
        for (size_t y = 0; y < d.ln[1]; ++y)
            std::memcpy(&packed[(z * d.ln[1] + y) * d.ln[0]], &local[lidx(d, 1, y + 1, z + 1)], d.ln[0] * sizeof(double));

    std::vector<int> counts, displs;
    std::vector<double> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        size_t off = 0;
        for (int r = 0; r < d.size; ++r) {
            int cc[3];
            MPI_Cart_coords(d.cart, r, 3, cc);
            size_t cnt = 1;
            const size_t n[3] = {nx, ny, nz};
            for (int i = 0; i < 3; ++i) {
                size_t s, e;
                blockRange(n[i], d.dims[i], cc[i], s, e);
                cnt *= (e - s);
            }
            counts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(off);
            off += cnt;
        }
        recv.resize(off);
    }
    MPI_Gatherv(packed.data(), static_cast<int>(lcount), MPI_DOUBLE,
                recv.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, d.cart);

    if (d.rank == 0) {
        global.resize(nx * ny * nz);
        const size_t n[3] = {nx, ny, nz};
        for (int r = 0; r < d.size; ++r) {
            int cc[3];
            MPI_Cart_coords(d.cart, r, 3, cc);
            size_t s[3], e[3];
            for (int i = 0; i < 3; ++i) blockRange(n[i], d.dims[i], cc[i], s[i], e[i]);
            const size_t bx = e[0] - s[0], by = e[1] - s[1];
            const double* src = recv.data() + displs[r];
            for (size_t z = s[2]; z < e[2]; ++z)
                for (size_t y = s[1]; y < e[1]; ++y)
                    std::memcpy(&global[z * (nx * ny) + y * nx + s[0]],
                                src + ((z - s[2]) * by + (y - s[1])) * bx, bx * sizeof(double));
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
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
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    const bool isRoot = (worldRank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    Decomp d;
    setupDecomp(d, nx, ny, nz);

    if (isRoot) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid %d x %d x %d)\n", d.size, d.dims[0], d.dims[1], d.dims[2]);
    }
    
    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    size_t gridSize = nx * ny * nz;
    const size_t localSize = d.lg[0] * d.lg[1] * d.lg[2];
    
    // Allocate arrays (local block with ghost layers)
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);
    
    // Initialize concentration field
    if (isRoot) printf("Initializing concentration field...\n");
    initializeConcentration(d, cold, nx, ny, nz);
    
    // Run simulation
    if (isRoot) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.cart);
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Request req[12];
    for (int t = 0; t < iterations; ++t) {
        // Exchange halos of c, overlap with interior chemical potential
        startHaloExchange(d, cold.data(), req);
        applyClampedBoundaries(d, cold.data());
        applyInner(d, [&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            computeChemicalPotentialBox(d, cold.data(), mu.data(), x0, x1, y0, y1, z0, z1,
                                        dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        });
        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
        applyShell(d, [&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            computeChemicalPotentialBox(d, cold.data(), mu.data(), x0, x1, y0, y1, z0, z1,
                                        dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        });

        // Exchange halos of mu, overlap with interior concentration update
        startHaloExchange(d, mu.data(), req);
        applyClampedBoundaries(d, mu.data());
        applyInner(d, [&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            cahnHilliardUpdateBox(d, cnew.data(), cold.data(), mu.data(), x0, x1, y0, y1, z0, z1,
                                  D, dt, dx, dy, dz);
        });
        MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
        applyShell(d, [&](size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
            cahnHilliardUpdateBox(d, cnew.data(), cold.data(), mu.data(), x0, x1, y0, y1, z0, z1,
                                  D, dt, dx, dy, dz);
        });
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(d.cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (isRoot) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<double> global;
        gatherField(d, cold, global, nx, ny, nz);

        if (isRoot) {
            // Print results for external validation
            if (printResults) {
                print_results(global, "Concentration");
            }
            
            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global, nx, ny, nz);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, d.cart);
    }

    freeDecomp(d);
    MPI_Finalize();
    return exitCode;
}
