#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Distributed memory parallelization (MPI)
//
// The 3D grid is decomposed into a 3D Cartesian block distribution. Every rank
// stores its own block surrounded by a one cell deep halo. Since the stencil is
// a 7 point Laplacian with clamped (zero gradient) boundary conditions, the
// halo cells at the physical domain boundary are simply mirrored copies of the
// adjacent interior plane, which makes the clamped stencil identical to the
// interior stencil everywhere.
// ---------------------------------------------------------------------------

// Local grid geometry (padded by one halo layer in every direction)
struct Grid {
    size_t lnx = 0, lny = 0, lnz = 0;    // local interior extents
    size_t ox = 0, oy = 0, oz = 0;       // global offset of the local block
    size_t pnx = 0, pny = 0, pnz = 0;    // padded extents
    size_t pnxy = 0;                     // pnx * pny
    size_t bytes = 0;                    // padded element count
};

// Inclusive index range (in padded/local coordinates)
struct Box {
    int x0, x1, y0, y1, z0, z1;
};

// Neighbor directions: 0:-x 1:+x 2:-y 3:+y 4:-z 5:+z
struct Halo {
    int nbr[6];
    MPI_Datatype send[6];
    MPI_Datatype recv[6];
};

// Chemical potential kernel on a sub-box
static void computeChemicalPotentialBox(const double* __restrict__ c, double* __restrict__ mu,
                                        const Grid& g, const Box& b,
                                        const double idx2, const double idy2, const double idz2,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    if(b.x1 < b.x0 || b.y1 < b.y0 || b.z1 < b.z0) return;

    const size_t pnx = g.pnx;
    const size_t pnxy = g.pnxy;

    for(int z = b.z0; z <= b.z1; ++z) {
        for(int y = b.y0; y <= b.y1; ++y) {
            const size_t base = static_cast<size_t>(z) * pnxy + static_cast<size_t>(y) * pnx;
            const double* __restrict__ p = c + base;
            double* __restrict__ o = mu + base;
            for(int x = b.x0; x <= b.x1; ++x) {
                const double cv = p[x];
                const double cxx = (p[x + 1] + p[x - 1] - 2.0 * cv) * idx2;
                const double cyy = (p[x + pnx] + p[x - pnx] - 2.0 * cv) * idy2;
                const double czz = (p[x + pnxy] + p[x - pnxy] - 2.0 * cv) * idz2;
                o[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update kernel on a sub-box
static void cahnHilliardUpdateBox(double* __restrict__ cnew, const double* __restrict__ cold,
                                  const double* __restrict__ mu, const Grid& g, const Box& b,
                                  const double D, const double dt,
                                  const double idx2, const double idy2, const double idz2) {
    if(b.x1 < b.x0 || b.y1 < b.y0 || b.z1 < b.z0) return;

    const size_t pnx = g.pnx;
    const size_t pnxy = g.pnxy;
    const double dtD = dt * D;

    for(int z = b.z0; z <= b.z1; ++z) {
        for(int y = b.y0; y <= b.y1; ++y) {
            const size_t base = static_cast<size_t>(z) * pnxy + static_cast<size_t>(y) * pnx;
            const double* __restrict__ p = mu + base;
            const double* __restrict__ co = cold + base;
            double* __restrict__ o = cnew + base;
            for(int x = b.x0; x <= b.x1; ++x) {
                const double mv = p[x];
                const double cxx = (p[x + 1] + p[x - 1] - 2.0 * mv) * idx2;
                const double cyy = (p[x + pnx] + p[x - pnx] - 2.0 * mv) * idy2;
                const double czz = (p[x + pnxy] + p[x - pnxy] - 2.0 * mv) * idz2;
                o[x] = co[x] + dtD * (cxx + cyy + czz);
            }
        }
    }
}

// Apply the kernel to all cells that require halo data (the outer shell of the
// local block). The inner part is handled separately so it can overlap with the
// halo exchange.
template <typename Kernel>
static void forEachShellBox(const Grid& g, Kernel&& k) {
    const int lx = static_cast<int>(g.lnx);
    const int ly = static_cast<int>(g.lny);
    const int lz = static_cast<int>(g.lnz);

    k(Box{1, lx, 1, ly, 1, 1});
    if(lz >= 2) k(Box{1, lx, 1, ly, lz, lz});
    k(Box{1, lx, 1, 1, 2, lz - 1});
    if(ly >= 2) k(Box{1, lx, ly, ly, 2, lz - 1});
    k(Box{1, 1, 2, ly - 1, 2, lz - 1});
    if(lx >= 2) k(Box{lx, lx, 2, ly - 1, 2, lz - 1});
}

static Box innerBox(const Grid& g) {
    return Box{2, static_cast<int>(g.lnx) - 1, 2, static_cast<int>(g.lny) - 1, 2, static_cast<int>(g.lnz) - 1};
}

// Mirror the interior boundary plane into the halo at physical domain borders,
// reproducing the clamped boundary condition of the serial code.
static void fillPhysicalHalos(double* f, const Grid& g, const Halo& h) {
    const size_t pnx = g.pnx;
    const size_t pnxy = g.pnxy;

    if(h.nbr[0] == MPI_PROC_NULL) {
        for(size_t z = 1; z <= g.lnz; ++z)
            for(size_t y = 1; y <= g.lny; ++y) {
                const size_t base = z * pnxy + y * pnx;
                f[base] = f[base + 1];
            }
    }
    if(h.nbr[1] == MPI_PROC_NULL) {
        for(size_t z = 1; z <= g.lnz; ++z)
            for(size_t y = 1; y <= g.lny; ++y) {
                const size_t base = z * pnxy + y * pnx;
                f[base + g.lnx + 1] = f[base + g.lnx];
            }
    }
    if(h.nbr[2] == MPI_PROC_NULL) {
        for(size_t z = 1; z <= g.lnz; ++z) {
            const size_t base = z * pnxy;
            memcpy(f + base + 1, f + base + pnx + 1, g.lnx * sizeof(double));
        }
    }
    if(h.nbr[3] == MPI_PROC_NULL) {
        for(size_t z = 1; z <= g.lnz; ++z) {
            const size_t base = z * pnxy + (g.lny + 1) * pnx;
            memcpy(f + base + 1, f + base - pnx + 1, g.lnx * sizeof(double));
        }
    }
    if(h.nbr[4] == MPI_PROC_NULL) {
        for(size_t y = 1; y <= g.lny; ++y)
            memcpy(f + y * pnx + 1, f + pnxy + y * pnx + 1, g.lnx * sizeof(double));
    }
    if(h.nbr[5] == MPI_PROC_NULL) {
        for(size_t y = 1; y <= g.lny; ++y)
            memcpy(f + (g.lnz + 1) * pnxy + y * pnx + 1, f + g.lnz * pnxy + y * pnx + 1, g.lnx * sizeof(double));
    }
}

static void startHaloExchange(double* f, const Halo& h, MPI_Comm comm, MPI_Request* reqs) {
    int n = 0;
    for(int d = 0; d < 6; ++d) {
        if(h.nbr[d] == MPI_PROC_NULL) continue;
        MPI_Irecv(f, 1, h.recv[d], h.nbr[d], d ^ 1, comm, &reqs[n++]);
        MPI_Isend(f, 1, h.send[d], h.nbr[d], d, comm, &reqs[n++]);
    }
    for(; n < 12; ++n) reqs[n] = MPI_REQUEST_NULL;
}

// Build the derived datatypes describing the six send/receive halo faces.
static void createHaloTypes(Halo& h, const Grid& g) {
    const int sizes[3] = {static_cast<int>(g.pnz), static_cast<int>(g.pny), static_cast<int>(g.pnx)};
    const int lx = static_cast<int>(g.lnx);
    const int ly = static_cast<int>(g.lny);
    const int lz = static_cast<int>(g.lnz);

    const int subs[6][3] = {{lz, ly, 1}, {lz, ly, 1}, {lz, 1, lx}, {lz, 1, lx}, {1, ly, lx}, {1, ly, lx}};
    const int sstart[6][3] = {{1, 1, 1}, {1, 1, lx}, {1, 1, 1}, {1, ly, 1}, {1, 1, 1}, {lz, 1, 1}};
    const int rstart[6][3] = {{1, 1, 0}, {1, 1, lx + 1}, {1, 0, 1}, {1, ly + 1, 1}, {0, 1, 1}, {lz + 1, 1, 1}};

    for(int d = 0; d < 6; ++d) {
        if(h.nbr[d] == MPI_PROC_NULL) {
            h.send[d] = MPI_DATATYPE_NULL;
            h.recv[d] = MPI_DATATYPE_NULL;
            continue;
        }
        MPI_Type_create_subarray(3, sizes, subs[d], sstart[d], MPI_ORDER_C, MPI_DOUBLE, &h.send[d]);
        MPI_Type_commit(&h.send[d]);
        MPI_Type_create_subarray(3, sizes, subs[d], rstart[d], MPI_ORDER_C, MPI_DOUBLE, &h.recv[d]);
        MPI_Type_commit(&h.recv[d]);
    }
}

// Initialize concentration field (global indexing, identical to the serial code)
static void initializeConcentration(std::vector<double>& c, const Grid& g,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for(size_t z = 0; z < g.lnz; ++z) {
        const size_t gz = g.oz + z;
        for(size_t y = 0; y < g.lny; ++y) {
            const size_t gy = g.oy + y;
            double* row = c.data() + (z + 1) * g.pnxy + (y + 1) * g.pnx + 1;
            const size_t rowbase = gz * (nx * ny) + gy * nx + g.ox;
            for(size_t x = 0; x < g.lnx; ++x) {
                const size_t linear_id = rowbase + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static bool validateResult(const std::vector<double>& c, const Grid& g, const int rank) {
    // Check for NaN or Inf
    int bad = 0;
    double minVal = 0.0, maxVal = 0.0;
    bool first = true;
    for(size_t z = 1; z <= g.lnz; ++z)
        for(size_t y = 1; y <= g.lny; ++y) {
            const double* row = c.data() + z * g.pnxy + y * g.pnx + 1;
            for(size_t x = 0; x < g.lnx; ++x) {
                const double val = row[x];
                if(std::isnan(val) || std::isinf(val)) bad = 1;
                if(first) {
                    minVal = maxVal = val;
                    first = false;
                } else {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                }
            }
        }
    if(first) {
        minVal = std::numeric_limits<double>::infinity();
        maxVal = -std::numeric_limits<double>::infinity();
    }

    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if(anyBad) {
        if(rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if(rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);

    // Values should generally stay within reasonable bounds
    if(gmax > 10.0 || gmin < -10.0) {
        if(rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Collect the distributed field on rank 0 (in global linear order) and print it
static void printDistributedResults(const std::vector<double>& c, const Grid& g,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const int rank, const int nranks) {
    std::vector<double> local(g.lnx * g.lny * g.lnz);
    for(size_t z = 0; z < g.lnz; ++z)
        for(size_t y = 0; y < g.lny; ++y)
            memcpy(local.data() + (z * g.lny + y) * g.lnx,
                   c.data() + (z + 1) * g.pnxy + (y + 1) * g.pnx + 1, g.lnx * sizeof(double));

    long long myinfo[6] = {static_cast<long long>(g.ox), static_cast<long long>(g.oy), static_cast<long long>(g.oz),
                           static_cast<long long>(g.lnx), static_cast<long long>(g.lny), static_cast<long long>(g.lnz)};
    std::vector<long long> info(rank == 0 ? 6 * nranks : 0);
    MPI_Gather(myinfo, 6, MPI_LONG_LONG, info.data(), 6, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        MPI_Send(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, 0, 77, MPI_COMM_WORLD);
        return;
    }

    std::vector<double> global(nx * ny * nz);
    std::vector<double> buf;
    for(int r = 0; r < nranks; ++r) {
        const size_t ox = static_cast<size_t>(info[6 * r + 0]);
        const size_t oy = static_cast<size_t>(info[6 * r + 1]);
        const size_t oz = static_cast<size_t>(info[6 * r + 2]);
        const size_t bx = static_cast<size_t>(info[6 * r + 3]);
        const size_t by = static_cast<size_t>(info[6 * r + 4]);
        const size_t bz = static_cast<size_t>(info[6 * r + 5]);

        const double* src;
        if(r == 0) {
            src = local.data();
        } else {
            buf.resize(bx * by * bz);
            MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, r, 77, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            src = buf.data();
        }

        for(size_t z = 0; z < bz; ++z)
            for(size_t y = 0; y < by; ++y)
                memcpy(global.data() + (oz + z) * (nx * ny) + (oy + y) * nx + ox,
                       src + (z * by + y) * bx, bx * sizeof(double));
    }

    print_results(global, "Concentration");
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

// Pick the process grid (pz, py, px) minimizing the halo surface while keeping
// at least one cell per rank in each dimension.
static void chooseDims(const int nranks, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    double best = -1.0;
    dims[0] = dims[1] = dims[2] = 0;

    for(int pz = 1; pz <= nranks; ++pz) {
        if(nranks % pz != 0 || static_cast<size_t>(pz) > nz) continue;
        const int rem = nranks / pz;
        for(int py = 1; py <= rem; ++py) {
            if(rem % py != 0 || static_cast<size_t>(py) > ny) continue;
            const int px = rem / py;
            if(static_cast<size_t>(px) > nx) continue;

            const double bx = static_cast<double>(nx) / px;
            const double by = static_cast<double>(ny) / py;
            const double bz = static_cast<double>(nz) / pz;
            // Communication surface; slightly penalize cuts along x to keep the
            // contiguous (vectorized) dimension long.
            const double cost = (px > 1 ? 2.0 * by * bz * 1.25 : 0.0) + (py > 1 ? 2.0 * bx * bz : 0.0) +
                                (pz > 1 ? 2.0 * bx * by : 0.0);
            if(best < 0.0 || cost < best) {
                best = cost;
                dims[0] = pz;
                dims[1] = py;
                dims[2] = px;
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for(int i = 1; i < argc; ++i) {
        if(strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if(strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if(strcmp(argv[i], "-h") == 0) {
            if(rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if(rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if(ny == 0) ny = nx;
    if(nz == 0) nz = nx;

    if(rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
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

    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);

    const size_t gridSize = nx * ny * nz;

    // Set up the process grid. If the rank count admits no decomposition that
    // leaves every rank at least one cell per dimension (e.g. a prime number of
    // ranks on a small grid), the largest usable subset of ranks is used.
    int dims[3] = {0, 0, 0};
    int nactive = 0;
    for(int na = nranks; na >= 1; --na) {
        chooseDims(na, nx, ny, nz, dims);
        if(dims[0] != 0) {
            nactive = na;
            break;
        }
    }
    if(rank == 0 && nactive != nranks) {
        printf("Note: grid too small, using %d of %d ranks\n", nactive, nranks);
    }

    const bool active = rank < nactive;
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm split = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &split);

    Grid g;
    Halo halo;
    for(int d = 0; d < 6; ++d) {
        halo.nbr[d] = MPI_PROC_NULL;
        halo.send[d] = MPI_DATATYPE_NULL;
        halo.recv[d] = MPI_DATATYPE_NULL;
    }

    if(active) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(split, 3, dims, periods, 1, &cart);
        int crank = 0;
        MPI_Comm_rank(cart, &crank);
        int coords[3] = {0, 0, 0};
        MPI_Cart_coords(cart, crank, 3, coords);

        const size_t n[3] = {nz, ny, nx};
        size_t ln[3], off[3];
        for(int d = 0; d < 3; ++d) {
            const size_t base = n[d] / dims[d];
            const size_t rem = n[d] % dims[d];
            ln[d] = base + (static_cast<size_t>(coords[d]) < rem ? 1 : 0);
            off[d] = static_cast<size_t>(coords[d]) * base + std::min(static_cast<size_t>(coords[d]), rem);
        }
        g.lnz = ln[0]; g.lny = ln[1]; g.lnx = ln[2];
        g.oz = off[0]; g.oy = off[1]; g.ox = off[2];

        MPI_Cart_shift(cart, 2, 1, &halo.nbr[0], &halo.nbr[1]);
        MPI_Cart_shift(cart, 1, 1, &halo.nbr[2], &halo.nbr[3]);
        MPI_Cart_shift(cart, 0, 1, &halo.nbr[4], &halo.nbr[5]);
    }

    g.pnx = g.lnx + 2;
    g.pny = g.lny + 2;
    g.pnz = g.lnz + 2;
    g.pnxy = g.pnx * g.pny;
    g.bytes = g.pnxy * g.pnz;

    if(active) createHaloTypes(halo, g);

    // Allocate arrays (halo padded)
    std::vector<double> cold(g.bytes, 0.0);
    std::vector<double> cnew(g.bytes, 0.0);
    std::vector<double> mu(g.bytes, 0.0);

    // Initialize concentration field
    if(rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, g, nx, ny, nz);

    // Run simulation
    if(rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    MPI_Request reqs[12];
    const Box inner = innerBox(g);

    for(int t = 0; active && t < iterations; ++t) {
        // Chemical potential: needs one halo layer of c
        fillPhysicalHalos(cold.data(), g, halo);
        startHaloExchange(cold.data(), halo, cart, reqs);
        computeChemicalPotentialBox(cold.data(), mu.data(), g, inner, idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB);
        MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);
        forEachShellBox(g, [&](const Box& b) {
            computeChemicalPotentialBox(cold.data(), mu.data(), g, b, idx2, idy2, idz2, gamma, e_AA, e_BB, e_AB);
        });

        // Concentration update: needs one halo layer of mu
        fillPhysicalHalos(mu.data(), g, halo);
        startHaloExchange(mu.data(), halo, cart, reqs);
        cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), g, inner, D, dt, idx2, idy2, idz2);
        MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);
        forEachShellBox(g, [&](const Box& b) {
            cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), g, b, D, dt, idx2, idy2, idz2);
        });

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double end = MPI_Wtime();
    double elapsed = end - start;
    double maxElapsed = elapsed;
    MPI_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const long durationMs = static_cast<long>(maxElapsed * 1000.0);
    if(rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / maxElapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    if(printResults) {
        printDistributedResults(cold, g, nx, ny, nz, rank, nranks);
    }

    int ret = 0;

    // Validation
    if(validate) {
        if(rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold, g, rank);

        if(valid) {
            if(rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if(rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    for(int d = 0; d < 6; ++d) {
        if(halo.send[d] != MPI_DATATYPE_NULL) MPI_Type_free(&halo.send[d]);
        if(halo.recv[d] != MPI_DATATYPE_NULL) MPI_Type_free(&halo.recv[d]);
    }
    if(cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    if(split != MPI_COMM_NULL) MPI_Comm_free(&split);
    MPI_Finalize();
    return ret;
}
