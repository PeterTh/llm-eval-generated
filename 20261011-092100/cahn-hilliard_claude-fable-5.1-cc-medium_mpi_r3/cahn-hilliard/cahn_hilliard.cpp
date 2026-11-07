#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Distributed-memory (MPI) Cahn-Hilliard solver.
//
// The global grid nx x ny x nz is split into a 3D Cartesian block
// decomposition. Each rank owns a local block of size nxl x nyl x nzl and
// stores it with a one-cell halo in every direction. The clamped boundary
// condition of the original code (neighbour index saturates at the domain
// edge) is reproduced exactly by copying the rank's own boundary face into
// the halo on global domain boundaries, so the 7-point stencil is applied
// uniformly everywhere and produces bitwise identical results.
//
// Per time step two halo exchanges are performed (concentration c, then
// chemical potential mu). Each exchange is overlapped with the stencil
// computation on the interior of the local block.
// ---------------------------------------------------------------------------

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;   // Cartesian communicator
    int rank = 0, size = 1;
    int dims[3] = {1, 1, 1};         // process grid (x, y, z)
    int coords[3] = {0, 0, 0};
    int nbr_lo[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    int nbr_hi[3] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    size_t n[3] = {0, 0, 0};         // global sizes
    size_t off[3] = {0, 0, 0};       // global offset of local block
    size_t nl[3] = {0, 0, 0};        // local block extents
    size_t sy = 0, sz = 0;           // strides of halo'd local array
    size_t local_size = 0;           // halo'd local array size
};

// Block range [begin, end) of dimension of length n split into p parts, part i
static inline void blockRange(size_t n, int p, int i, size_t& begin, size_t& len) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    const size_t ii = static_cast<size_t>(i);
    begin = ii * base + std::min(ii, rem);
    len = base + (ii < rem ? 1 : 0);
}

// Index into the halo'd local array (x, y, z are local coordinates, halo at -1 and nl)
static inline size_t lidx(const Domain& d, long x, long y, long z) noexcept {
    return static_cast<size_t>(z + 1) * d.sz + static_cast<size_t>(y + 1) * d.sy + static_cast<size_t>(x + 1);
}

// Choose the process grid: minimise communicated halo surface while keeping
// every dimension of the process grid no larger than the grid itself.
// Ties are broken towards fewer x-splits (keeps the contiguous x-rows long).
static void chooseDims(int P, const size_t n[3], int dims[3]) {
    double best = -1.0;
    dims[0] = dims[1] = dims[2] = 0;
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0 || static_cast<size_t>(px) > n[0]) continue;
        const int Pyz = P / px;
        for (int py = 1; py <= Pyz; ++py) {
            if (Pyz % py != 0 || static_cast<size_t>(py) > n[1]) continue;
            const int pz = Pyz / py;
            if (static_cast<size_t>(pz) > n[2]) continue;
            const double lx = static_cast<double>(n[0]) / px;
            const double ly = static_cast<double>(n[1]) / py;
            const double lz = static_cast<double>(n[2]) / pz;
            // halo faces actually exchanged (none in a dimension with a single process)
            const double surf = (px > 1 ? 2.0 * ly * lz : 0.0) + (py > 1 ? 2.0 * lx * lz : 0.0) +
                                (pz > 1 ? 2.0 * lx * ly : 0.0);
            // small penalty for x-splits (strided packing, shorter inner loops)
            const double cost = surf + 1e-3 * px * ly * lz;
            if (best < 0.0 || cost < best) {
                best = cost;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Halo exchange with communication/computation overlap
// ---------------------------------------------------------------------------
struct HaloExchange {
    const Domain* d = nullptr;
    std::vector<double> sendbuf[3][2], recvbuf[3][2];
    MPI_Request reqs[12];
    int nreq = 0;

    void init(const Domain& dom) {
        d = &dom;
        const size_t faceSize[3] = {dom.nl[1] * dom.nl[2], dom.nl[0] * dom.nl[2], dom.nl[0] * dom.nl[1]};
        for (int dim = 0; dim < 3; ++dim) {
            for (int s = 0; s < 2; ++s) {
                sendbuf[dim][s].assign(faceSize[dim], 0.0);
                recvbuf[dim][s].assign(faceSize[dim], 0.0);
            }
        }
    }

    // Copy the face at local coordinate `pos` in dimension `dim` to/from a buffer
    template <bool Pack>
    static void copyFace(const Domain& d, double* a, double* buf, int dim, long pos) {
        const long nx = static_cast<long>(d.nl[0]), ny = static_cast<long>(d.nl[1]), nz = static_cast<long>(d.nl[2]);
        size_t k = 0;
        if (dim == 0) {
            for (long z = 0; z < nz; ++z)
                for (long y = 0; y < ny; ++y) {
                    double& cell = a[lidx(d, pos, y, z)];
                    if constexpr (Pack) buf[k++] = cell; else cell = buf[k++];
                }
        } else if (dim == 1) {
            for (long z = 0; z < nz; ++z) {
                double* row = &a[lidx(d, 0, pos, z)];
                if constexpr (Pack) std::memcpy(buf + k, row, nx * sizeof(double));
                else std::memcpy(row, buf + k, nx * sizeof(double));
                k += nx;
            }
        } else {
            for (long y = 0; y < ny; ++y) {
                double* row = &a[lidx(d, 0, y, pos)];
                if constexpr (Pack) std::memcpy(buf + k, row, nx * sizeof(double));
                else std::memcpy(row, buf + k, nx * sizeof(double));
                k += nx;
            }
        }
    }

    // Copy own boundary face into the halo (clamped boundary condition)
    static void clampFace(const Domain& d, double* a, int dim, bool hi) {
        const long nx = static_cast<long>(d.nl[0]), ny = static_cast<long>(d.nl[1]), nz = static_cast<long>(d.nl[2]);
        const long src = hi ? static_cast<long>(d.nl[dim]) - 1 : 0;
        const long dst = hi ? static_cast<long>(d.nl[dim]) : -1;
        if (dim == 0) {
            for (long z = 0; z < nz; ++z)
                for (long y = 0; y < ny; ++y) a[lidx(d, dst, y, z)] = a[lidx(d, src, y, z)];
        } else if (dim == 1) {
            for (long z = 0; z < nz; ++z)
                std::memcpy(&a[lidx(d, 0, dst, z)], &a[lidx(d, 0, src, z)], nx * sizeof(double));
        } else {
            for (long y = 0; y < ny; ++y)
                std::memcpy(&a[lidx(d, 0, y, dst)], &a[lidx(d, 0, y, src)], nx * sizeof(double));
        }
    }

    // Start exchange: post receives, pack & send, fill clamped halos
    void start(double* a) {
        const Domain& dom = *d;
        nreq = 0;
        for (int dim = 0; dim < 3; ++dim) {
            const int count = static_cast<int>(sendbuf[dim][0].size());
            // receive from low neighbour into halo at -1, from high into halo at nl
            if (dom.nbr_lo[dim] != MPI_PROC_NULL)
                MPI_Irecv(recvbuf[dim][0].data(), count, MPI_DOUBLE, dom.nbr_lo[dim], 2 * dim + 1, dom.comm, &reqs[nreq++]);
            if (dom.nbr_hi[dim] != MPI_PROC_NULL)
                MPI_Irecv(recvbuf[dim][1].data(), count, MPI_DOUBLE, dom.nbr_hi[dim], 2 * dim, dom.comm, &reqs[nreq++]);
        }
        for (int dim = 0; dim < 3; ++dim) {
            const int count = static_cast<int>(sendbuf[dim][0].size());
            if (dom.nbr_lo[dim] != MPI_PROC_NULL) {
                copyFace<true>(dom, a, sendbuf[dim][0].data(), dim, 0);
                MPI_Isend(sendbuf[dim][0].data(), count, MPI_DOUBLE, dom.nbr_lo[dim], 2 * dim, dom.comm, &reqs[nreq++]);
            } else {
                clampFace(dom, a, dim, false);
            }
            if (dom.nbr_hi[dim] != MPI_PROC_NULL) {
                copyFace<true>(dom, a, sendbuf[dim][1].data(), dim, static_cast<long>(dom.nl[dim]) - 1);
                MPI_Isend(sendbuf[dim][1].data(), count, MPI_DOUBLE, dom.nbr_hi[dim], 2 * dim + 1, dom.comm, &reqs[nreq++]);
            } else {
                clampFace(dom, a, dim, true);
            }
        }
    }

    // Finish exchange: wait and unpack received halos
    void finish(double* a) {
        const Domain& dom = *d;
        if (nreq > 0) MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        for (int dim = 0; dim < 3; ++dim) {
            if (dom.nbr_lo[dim] != MPI_PROC_NULL) copyFace<false>(dom, a, recvbuf[dim][0].data(), dim, -1);
            if (dom.nbr_hi[dim] != MPI_PROC_NULL) copyFace<false>(dom, a, recvbuf[dim][1].data(), dim, static_cast<long>(dom.nl[dim]));
        }
    }
};

// ---------------------------------------------------------------------------
// Stencil kernels on a sub-box [x0,x1) x [y0,y1) x [z0,z1) of the local block
// (same arithmetic and evaluation order as the original code)
// ---------------------------------------------------------------------------
struct Params {
    double dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB;
};

// Compute chemical potential on a sub-box
static inline void computeChemicalPotentialBox(const Domain& d, const double* __restrict c, double* __restrict mu,
                                               const Params& p, long x0, long x1, long y0, long y1, long z0, long z1) {
    const double dx2 = p.dx * p.dx, dy2 = p.dy * p.dy, dz2 = p.dz * p.dz;
    const size_t sy = d.sy, sz = d.sz;
    for (long z = z0; z < z1; ++z) {
        for (long y = y0; y < y1; ++y) {
            const size_t base = lidx(d, 0, y, z);
            for (long x = x0; x < x1; ++x) {
                const size_t idx = base + static_cast<size_t>(x);
                const double cv = c[idx];
                const double cxx = (c[idx + 1] + c[idx - 1] - 2.0 * cv) / dx2;
                const double cyy = (c[idx + sy] + c[idx - sy] - 2.0 * cv) / dy2;
                const double czz = (c[idx + sz] + c[idx - sz] - 2.0 * cv) / dz2;
                mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - p.gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update step on a sub-box
static inline void cahnHilliardUpdateBox(const Domain& d, double* __restrict cnew, const double* __restrict cold,
                                         const double* __restrict mu, const Params& p,
                                         long x0, long x1, long y0, long y1, long z0, long z1) {
    const double dx2 = p.dx * p.dx, dy2 = p.dy * p.dy, dz2 = p.dz * p.dz;
    const size_t sy = d.sy, sz = d.sz;
    for (long z = z0; z < z1; ++z) {
        for (long y = y0; y < y1; ++y) {
            const size_t base = lidx(d, 0, y, z);
            for (long x = x0; x < x1; ++x) {
                const size_t idx = base + static_cast<size_t>(x);
                const double mv = mu[idx];
                const double mxx = (mu[idx + 1] + mu[idx - 1] - 2.0 * mv) / dx2;
                const double myy = (mu[idx + sy] + mu[idx - sy] - 2.0 * mv) / dy2;
                const double mzz = (mu[idx + sz] + mu[idx - sz] - 2.0 * mv) / dz2;
                cnew[idx] = cold[idx] + p.dt * p.D * (mxx + myy + mzz);
            }
        }
    }
}

// Apply a kernel to the whole local block, split into the interior (which does
// not depend on halo data) and the boundary shell (which does).
template <typename F>
static void forInterior(const Domain& d, F&& f) {
    const long nx = static_cast<long>(d.nl[0]), ny = static_cast<long>(d.nl[1]), nz = static_cast<long>(d.nl[2]);
    if (nx > 2 && ny > 2 && nz > 2) f(1, nx - 1, 1, ny - 1, 1, nz - 1);
}

template <typename F>
static void forShell(const Domain& d, F&& f) {
    const long nx = static_cast<long>(d.nl[0]), ny = static_cast<long>(d.nl[1]), nz = static_cast<long>(d.nl[2]);
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    if (!(nx > 2 && ny > 2 && nz > 2)) {  // no interior: everything is shell
        f(0, nx, 0, ny, 0, nz);
        return;
    }
    f(0, nx, 0, ny, 0, 1);            // low z plane
    f(0, nx, 0, ny, nz - 1, nz);      // high z plane
    f(0, nx, 0, 1, 1, nz - 1);        // low y plane
    f(0, nx, ny - 1, ny, 1, nz - 1);  // high y plane
    f(0, 1, 1, ny - 1, 1, nz - 1);    // low x plane
    f(nx - 1, nx, 1, ny - 1, 1, nz - 1);  // high x plane
}

// Initialize concentration field (same pseudo-random sequence as the original)
static void initializeConcentration(const Domain& d, std::vector<double>& c) {
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    const size_t vol = nx * ny * nz;
    for (size_t z = 0; z < d.nl[2]; ++z) {
        const size_t gz = d.off[2] + z;
        for (size_t y = 0; y < d.nl[1]; ++y) {
            const size_t gy = d.off[1] + y;
            double* row = &c[lidx(d, 0, static_cast<long>(y), static_cast<long>(z))];
            for (size_t x = 0; x < d.nl[0]; ++x) {
                const size_t gx = d.off[0] + x;
                const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather the distributed field into a global array (in original z,y,x order) on rank 0
static void gatherGlobal(const Domain& d, const std::vector<double>& c, std::vector<double>& global) {
    const size_t nloc = d.nl[0] * d.nl[1] * d.nl[2];
    std::vector<double> packed(nloc);
    size_t k = 0;
    for (size_t z = 0; z < d.nl[2]; ++z)
        for (size_t y = 0; y < d.nl[1]; ++y) {
            std::memcpy(&packed[k], &c[lidx(d, 0, static_cast<long>(y), static_cast<long>(z))], d.nl[0] * sizeof(double));
            k += d.nl[0];
        }

    std::vector<int> counts, displs;
    std::vector<double> recv;
    if (d.rank == 0) {
        counts.resize(d.size);
        displs.resize(d.size);
        int total = 0;
        for (int r = 0; r < d.size; ++r) {
            int crd[3];
            MPI_Cart_coords(d.comm, r, 3, crd);
            size_t b, l[3];
            for (int i = 0; i < 3; ++i) blockRange(d.n[i], d.dims[i], crd[i], b, l[i]);
            counts[r] = static_cast<int>(l[0] * l[1] * l[2]);
            displs[r] = total;
            total += counts[r];
        }
        recv.resize(static_cast<size_t>(total));
    }
    MPI_Gatherv(packed.data(), static_cast<int>(nloc), MPI_DOUBLE, recv.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, d.comm);

    if (d.rank == 0) {
        global.resize(d.n[0] * d.n[1] * d.n[2]);
        for (int r = 0; r < d.size; ++r) {
            int crd[3];
            MPI_Cart_coords(d.comm, r, 3, crd);
            size_t b[3], l[3];
            for (int i = 0; i < 3; ++i) blockRange(d.n[i], d.dims[i], crd[i], b[i], l[i]);
            const double* src = &recv[static_cast<size_t>(displs[r])];
            for (size_t z = 0; z < l[2]; ++z)
                for (size_t y = 0; y < l[1]; ++y) {
                    const size_t gidx = (b[2] + z) * (d.n[0] * d.n[1]) + (b[1] + y) * d.n[0] + b[0];
                    std::memcpy(&global[gidx], src, l[0] * sizeof(double));
                    src += l[0];
                }
        }
    }
}

// Distributed validation (identical output to the original on rank 0)
static bool validateResult(const Domain& d, const std::vector<double>& c) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    for (size_t z = 0; z < d.nl[2]; ++z)
        for (size_t y = 0; y < d.nl[1]; ++y) {
            const double* row = &c[lidx(d, 0, static_cast<long>(y), static_cast<long>(z))];
            for (size_t x = 0; x < d.nl[0]; ++x) {
                const double val = row[x];
                if (std::isnan(val) || std::isinf(val)) bad = 1;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    int anyBad = 0;
    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, d.comm);

    if (anyBad) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (d.rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
    if (gmax > 10.0 || gmin < -10.0) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

static void printUsage(const char* progName) {
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
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks receive identical argv)
    int exitCode = -1;
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
            exitCode = 0;
            break;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", worldSize);
    }

    // Physical parameters
    const Params p{1.0, 1.0, 1.0, 0.01, 1.0, 0.5, -(2.0 / 9.0), -(2.0 / 9.0), (2.0 / 9.0)};

    const size_t gridSize = nx * ny * nz;

    // Use at most one rank per grid cell; surplus ranks sit idle.
    const int activeRanks = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), std::max<size_t>(gridSize, 1)));
    const bool active = worldRank < activeRanks;
    MPI_Comm activeComm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (!active) {
        MPI_Finalize();
        return 0;
    }

    // Build the Cartesian process grid
    Domain d;
    d.n[0] = nx; d.n[1] = ny; d.n[2] = nz;
    chooseDims(activeRanks, d.n, d.dims);
    {
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(activeComm, 3, d.dims, periods, 1, &d.comm);
        MPI_Comm_free(&activeComm);
    }
    MPI_Comm_rank(d.comm, &d.rank);
    MPI_Comm_size(d.comm, &d.size);
    MPI_Cart_coords(d.comm, d.rank, 3, d.coords);
    for (int i = 0; i < 3; ++i) {
        MPI_Cart_shift(d.comm, i, 1, &d.nbr_lo[i], &d.nbr_hi[i]);
        blockRange(d.n[i], d.dims[i], d.coords[i], d.off[i], d.nl[i]);
    }
    d.sy = d.nl[0] + 2;
    d.sz = (d.nl[0] + 2) * (d.nl[1] + 2);
    d.local_size = d.sz * (d.nl[2] + 2);
    if (d.rank == 0) printf("Process grid: %d x %d x %d\n", d.dims[0], d.dims[1], d.dims[2]);

    // Allocate arrays (with halo)
    std::vector<double> cold(d.local_size, 0.0);
    std::vector<double> cnew(d.local_size, 0.0);
    std::vector<double> mu(d.local_size, 0.0);

    HaloExchange halo;
    halo.init(d);

    // Initialize concentration field
    if (d.rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(d, cold);

    // Run simulation
    if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential: exchange c halo, overlap with interior
        halo.start(cold.data());
        forInterior(d, [&](long x0, long x1, long y0, long y1, long z0, long z1) {
            computeChemicalPotentialBox(d, cold.data(), mu.data(), p, x0, x1, y0, y1, z0, z1);
        });
        halo.finish(cold.data());
        forShell(d, [&](long x0, long x1, long y0, long y1, long z0, long z1) {
            computeChemicalPotentialBox(d, cold.data(), mu.data(), p, x0, x1, y0, y1, z0, z1);
        });

        // Update concentration: exchange mu halo, overlap with interior
        halo.start(mu.data());
        forInterior(d, [&](long x0, long x1, long y0, long y1, long z0, long z1) {
            cahnHilliardUpdateBox(d, cnew.data(), cold.data(), mu.data(), p, x0, x1, y0, y1, z0, z1);
        });
        halo.finish(mu.data());
        forShell(d, [&](long x0, long x1, long y0, long y1, long z0, long z1) {
            cahnHilliardUpdateBox(d, cnew.data(), cold.data(), mu.data(), p, x0, x1, y0, y1, z0, z1);
        });

        // Swap buffers
        std::swap(cold, cnew);
    }

    MPI_Barrier(d.comm);
    const double tEnd = MPI_Wtime();
    const long localDurationMs = static_cast<long>((tEnd - tStart) * 1000.0);
    long durationMs = 0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_LONG, MPI_MAX, 0, d.comm);

    if (d.rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> global;
        gatherGlobal(d, cold, global);
        if (d.rank == 0) print_results(global, "Concentration");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (d.rank == 0) printf("Validating result...\n");
        bool valid = validateResult(d, cold);
        if (d.rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        ret = valid ? 0 : 1;
    }

    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return ret;
}
