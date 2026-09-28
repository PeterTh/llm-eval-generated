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
// MPI parallelization: 3D Cartesian domain decomposition with a one-cell halo.
//
// Each rank owns a block of the global grid and stores it in a local array
// padded by one ghost layer in every direction. The clamped ("copy") boundary
// condition of the original serial code is reproduced by mirroring the
// outermost owned plane into the ghost layer at physical domain boundaries,
// which makes the stencil evaluation bitwise identical to the serial version.
//
// Per time step two halo exchanges are required (one for c before computing
// the chemical potential, one for mu before the concentration update). Both
// are overlapped with the computation of the block interior.
// ---------------------------------------------------------------------------

// Half-open iteration box in local (ghosted) index space.
struct Box {
    size_t i0, i1, j0, j1, k0, k1;
};

namespace {

// Local (ghosted) array geometry, set up once in main.
size_t g_lx = 0;   // padded extent in x
size_t g_lxy = 0;  // padded extent in x * y

inline size_t lidx(const size_t i, const size_t j, const size_t k) noexcept {
    return k * g_lxy + j * g_lx + i;
}

// Chemical potential over a sub-box of the owned region.
void computeChemicalPotentialBox(const double* __restrict__ c, double* __restrict__ mu, const Box b,
                                 const double dx, const double dy, const double dz,
                                 const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t lx = g_lx;
    const size_t lxy = g_lxy;
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);

    for (size_t k = b.k0; k < b.k1; ++k) {
        for (size_t j = b.j0; j < b.j1; ++j) {
            const size_t row = k * lxy + j * lx;
            for (size_t i = b.i0; i < b.i1; ++i) {
                const size_t idx = row + i;
                const double cv = c[idx];

                const double cxx = (c[idx + 1] + c[idx - 1] - 2.0 * cv) * idx2;
                const double cyy = (c[idx + lx] + c[idx - lx] - 2.0 * cv) * idy2;
                const double czz = (c[idx + lxy] + c[idx - lxy] - 2.0 * cv) * idz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update over a sub-box of the owned region.
void cahnHilliardUpdateBox(double* __restrict__ cnew, const double* __restrict__ cold,
                           const double* __restrict__ mu, const Box b,
                           const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t lx = g_lx;
    const size_t lxy = g_lxy;
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double idz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    for (size_t k = b.k0; k < b.k1; ++k) {
        for (size_t j = b.j0; j < b.j1; ++j) {
            const size_t row = k * lxy + j * lx;
            for (size_t i = b.i0; i < b.i1; ++i) {
                const size_t idx = row + i;
                const double mv = mu[idx];

                const double cxx = (mu[idx + 1] + mu[idx - 1] - 2.0 * mv) * idx2;
                const double cyy = (mu[idx + lx] + mu[idx - lx] - 2.0 * mv) * idy2;
                const double czz = (mu[idx + lxy] + mu[idx - lxy] - 2.0 * mv) * idz2;

                cnew[idx] = cold[idx] + dtD * (cxx + cyy + czz);
            }
        }
    }
}

// Initialize the locally owned part of the concentration field. Identical
// values to the serial initialization, computed from global coordinates.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t x0, const size_t y0, const size_t z0,
                             const size_t nxl, const size_t nyl, const size_t nzl) {
    const size_t vol = nx * ny * nz;

    for (size_t k = 0; k < nzl; ++k) {
        for (size_t j = 0; j < nyl; ++j) {
            const size_t base = (z0 + k) * (nx * ny) + (y0 + j) * nx + x0;
            double* __restrict__ row = &c[lidx(1, j + 1, k + 1)];
            for (size_t i = 0; i < nxl; ++i) {
                const size_t linear_id = base + i;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[i] = -1.0 + 2.0 * pseudo;
            }
        }
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

// Pick a 3D process grid (px, py, pz) for `procs` ranks such that no dimension
// is over-decomposed, minimizing the halo surface. Splitting x is penalized
// slightly because those faces are strided.
bool chooseDims(const int procs, const size_t nx, const size_t ny, const size_t nz, int dims[3]) {
    bool found = false;
    double best = 0.0;
    for (int px = 1; px <= procs; ++px) {
        if (procs % px != 0) continue;
        if (static_cast<size_t>(px) > nx) continue;
        const int rest = procs / px;
        for (int py = 1; py <= rest; ++py) {
            if (rest % py != 0) continue;
            if (static_cast<size_t>(py) > ny) continue;
            const int pz = rest / py;
            if (static_cast<size_t>(pz) > nz) continue;

            const double bx = static_cast<double>(nx) / px;
            const double by = static_cast<double>(ny) / py;
            const double bz = static_cast<double>(nz) / pz;
            const double cost = 1.3 * by * bz + bx * bz + bx * by;
            if (!found || cost < best) {
                best = cost;
                found = true;
                dims[0] = pz;  // cartesian dim 0 -> z
                dims[1] = py;
                dims[2] = px;
            }
        }
    }
    return found;
}

// Block distribution of n cells over p parts.
void blockRange(const size_t n, const int p, const int coord, size_t& start, size_t& count) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem = n % static_cast<size_t>(p);
    const size_t c = static_cast<size_t>(coord);
    count = base + (c < rem ? 1 : 0);
    start = c * base + std::min(c, rem);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (world_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
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

    const size_t gridSize = nx * ny * nz;

    // ---- Determine process grid; ranks that cannot be given work stay idle.
    int dims[3] = {1, 1, 1};
    int active_procs = world_size;
    while (active_procs > 1 && !chooseDims(active_procs, nx, ny, nz, dims)) {
        --active_procs;
    }
    if (active_procs == 1) {
        dims[0] = dims[1] = dims[2] = 1;
    }

    const int active = (world_rank < active_procs) ? 1 : 0;
    MPI_Comm comm_active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active, world_rank, &comm_active);

    int exit_code = 0;

    if (active) {
        // ---- Cartesian topology (non-periodic; boundaries are clamped).
        int periods[3] = {0, 0, 0};
        MPI_Comm cart = MPI_COMM_NULL;
        MPI_Cart_create(comm_active, 3, dims, periods, 0, &cart);

        int cart_rank = 0;
        MPI_Comm_rank(cart, &cart_rank);
        int coords[3] = {0, 0, 0};
        MPI_Cart_coords(cart, cart_rank, 3, coords);

        // Neighbors: index [dim][0] = lower, [dim][1] = upper.
        int nbr[3][2];
        for (int d = 0; d < 3; ++d) {
            MPI_Cart_shift(cart, d, 1, &nbr[d][0], &nbr[d][1]);
        }

        // ---- Local block geometry.
        size_t x0, y0, z0, nxl, nyl, nzl;
        blockRange(nx, dims[2], coords[2], x0, nxl);
        blockRange(ny, dims[1], coords[1], y0, nyl);
        blockRange(nz, dims[0], coords[0], z0, nzl);

        g_lx = nxl + 2;
        g_lxy = (nxl + 2) * (nyl + 2);
        const size_t lx = g_lx;
        const size_t lxy = g_lxy;
        const size_t localAlloc = lxy * (nzl + 2);

        std::vector<double> cold(localAlloc, 0.0);
        std::vector<double> cnew(localAlloc, 0.0);
        std::vector<double> mu(localAlloc, 0.0);

        // ---- Halo datatypes: [dim][side][send=0/recv=1]
        const int sizes[3] = {static_cast<int>(nzl + 2), static_cast<int>(nyl + 2), static_cast<int>(nxl + 2)};
        MPI_Datatype faceType[3][2][2];
        for (int d = 0; d < 3; ++d) {
            const int n_d = static_cast<int>(d == 0 ? nzl : (d == 1 ? nyl : nxl));
            int subsizes[3] = {static_cast<int>(nzl), static_cast<int>(nyl), static_cast<int>(nxl)};
            subsizes[d] = 1;
            // start offsets along dimension d: inner planes 1 / n_d, ghost planes 0 / n_d+1
            const int planes[2][2] = {{1, 0}, {n_d, n_d + 1}};
            for (int side = 0; side < 2; ++side) {
                for (int which = 0; which < 2; ++which) {
                    int starts[3] = {1, 1, 1};
                    starts[d] = planes[side][which];
                    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE,
                                             &faceType[d][side][which]);
                    MPI_Type_commit(&faceType[d][side][which]);
                }
            }
        }

        // ---- Mirror owned boundary planes into ghost layers where the block
        //      touches the physical domain boundary (reproduces clamping).
        auto fillPhysicalBoundaries = [&](double* a) {
            if (nbr[2][0] == MPI_PROC_NULL) {  // x low
                for (size_t k = 1; k <= nzl; ++k)
                    for (size_t j = 1; j <= nyl; ++j) a[k * lxy + j * lx] = a[k * lxy + j * lx + 1];
            }
            if (nbr[2][1] == MPI_PROC_NULL) {  // x high
                for (size_t k = 1; k <= nzl; ++k)
                    for (size_t j = 1; j <= nyl; ++j) a[k * lxy + j * lx + nxl + 1] = a[k * lxy + j * lx + nxl];
            }
            if (nbr[1][0] == MPI_PROC_NULL) {  // y low
                for (size_t k = 1; k <= nzl; ++k)
                    memcpy(&a[k * lxy + 1], &a[k * lxy + lx + 1], nxl * sizeof(double));
            }
            if (nbr[1][1] == MPI_PROC_NULL) {  // y high
                for (size_t k = 1; k <= nzl; ++k)
                    memcpy(&a[k * lxy + (nyl + 1) * lx + 1], &a[k * lxy + nyl * lx + 1], nxl * sizeof(double));
            }
            if (nbr[0][0] == MPI_PROC_NULL) {  // z low
                for (size_t j = 1; j <= nyl; ++j) memcpy(&a[j * lx + 1], &a[lxy + j * lx + 1], nxl * sizeof(double));
            }
            if (nbr[0][1] == MPI_PROC_NULL) {  // z high
                for (size_t j = 1; j <= nyl; ++j)
                    memcpy(&a[(nzl + 1) * lxy + j * lx + 1], &a[nzl * lxy + j * lx + 1], nxl * sizeof(double));
            }
        };

        auto startExchange = [&](double* a, MPI_Request* req) {
            int n = 0;
            for (int d = 0; d < 3; ++d) {
                for (int side = 0; side < 2; ++side) {
                    const int tag = 2 * d + side;
                    MPI_Irecv(a, 1, faceType[d][side][1], nbr[d][side], tag, cart, &req[n++]);
                    MPI_Isend(a, 1, faceType[d][side][0], nbr[d][side], 2 * d + (1 - side), cart, &req[n++]);
                }
            }
        };

        // ---- Iteration boxes: interior core (halo independent) + 6 shells.
        const Box core = {2, nxl, 2, nyl, 2, nzl};
        Box shells[6];
        int nshells = 0;
        shells[nshells++] = {1, nxl + 1, 1, nyl + 1, 1, 2};
        if (nzl >= 2) shells[nshells++] = {1, nxl + 1, 1, nyl + 1, nzl, nzl + 1};
        shells[nshells++] = {1, nxl + 1, 1, 2, 2, nzl};
        if (nyl >= 2) shells[nshells++] = {1, nxl + 1, nyl, nyl + 1, 2, nzl};
        shells[nshells++] = {1, 2, 2, nyl, 2, nzl};
        if (nxl >= 2) shells[nshells++] = {nxl, nxl + 1, 2, nyl, 2, nzl};

        if (world_rank == 0) printf("Initializing concentration field...\n");
        initializeConcentration(cold, nx, ny, nz, x0, y0, z0, nxl, nyl, nzl);

        if (world_rank == 0) printf("Running Cahn-Hilliard simulation...\n");

        MPI_Barrier(cart);
        const double tstart = MPI_Wtime();

        MPI_Request req[12];
        for (int t = 0; t < iterations; ++t) {
            // Chemical potential: exchange c, overlap with interior work.
            fillPhysicalBoundaries(cold.data());
            startExchange(cold.data(), req);
            computeChemicalPotentialBox(cold.data(), mu.data(), core, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
            for (int s = 0; s < nshells; ++s) {
                computeChemicalPotentialBox(cold.data(), mu.data(), shells[s], dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // Concentration update: exchange mu, overlap with interior work.
            fillPhysicalBoundaries(mu.data());
            startExchange(mu.data(), req);
            cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), core, D, dt, dx, dy, dz);
            MPI_Waitall(12, req, MPI_STATUSES_IGNORE);
            for (int s = 0; s < nshells; ++s) {
                cahnHilliardUpdateBox(cnew.data(), cold.data(), mu.data(), shells[s], D, dt, dx, dy, dz);
            }

            // Swap buffers
            std::swap(cold, cnew);
        }

        const double tlocal = MPI_Wtime() - tstart;
        double tmax = tlocal;
        MPI_Allreduce(&tlocal, &tmax, 1, MPI_DOUBLE, MPI_MAX, cart);

        if (world_rank == 0) {
            const long ms = static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::duration<double>(tmax))
                                                  .count());
            printf("Computation time: %ld ms\n", ms);

            const double cellUpdates = static_cast<double>(gridSize) * iterations;
            const double mcups = cellUpdates / (ms / 1000.0) / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // ---- Result output: gather the global field on rank 0 so that the
        //      reported statistics/hash match the serial reference exactly.
        if (printResults) {
            std::vector<double> sendbuf(nxl * nyl * nzl);
            for (size_t k = 0; k < nzl; ++k) {
                for (size_t j = 0; j < nyl; ++j) {
                    memcpy(&sendbuf[(k * nyl + j) * nxl], &cold[lidx(1, j + 1, k + 1)], nxl * sizeof(double));
                }
            }

            long long geom[6] = {static_cast<long long>(x0), static_cast<long long>(y0), static_cast<long long>(z0),
                                 static_cast<long long>(nxl), static_cast<long long>(nyl),
                                 static_cast<long long>(nzl)};
            std::vector<long long> allGeom(cart_rank == 0 ? 6 * static_cast<size_t>(active_procs) : size_t{0});
            MPI_Gather(geom, 6, MPI_LONG_LONG, allGeom.data(), 6, MPI_LONG_LONG, 0, cart);

            if (cart_rank == 0) {
                std::vector<double> full(gridSize);
                std::vector<double> recvbuf;
                for (int r = 0; r < active_procs; ++r) {
                    const size_t rx0 = static_cast<size_t>(allGeom[6 * r + 0]);
                    const size_t ry0 = static_cast<size_t>(allGeom[6 * r + 1]);
                    const size_t rz0 = static_cast<size_t>(allGeom[6 * r + 2]);
                    const size_t rnx = static_cast<size_t>(allGeom[6 * r + 3]);
                    const size_t rny = static_cast<size_t>(allGeom[6 * r + 4]);
                    const size_t rnz = static_cast<size_t>(allGeom[6 * r + 5]);
                    const double* src = nullptr;
                    if (r == 0) {
                        src = sendbuf.data();
                    } else {
                        recvbuf.resize(rnx * rny * rnz);
                        MPI_Recv(recvbuf.data(), static_cast<int>(rnx * rny * rnz), MPI_DOUBLE, r, 99, cart,
                                 MPI_STATUS_IGNORE);
                        src = recvbuf.data();
                    }
                    for (size_t k = 0; k < rnz; ++k) {
                        for (size_t j = 0; j < rny; ++j) {
                            memcpy(&full[(rz0 + k) * (nx * ny) + (ry0 + j) * nx + rx0], &src[(k * rny + j) * rnx],
                                   rnx * sizeof(double));
                        }
                    }
                }
                print_results(full, "Concentration");
            } else {
                MPI_Send(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE, 0, 99, cart);
            }
        }

        // ---- Validation (distributed reductions over the owned cells).
        if (validate) {
            if (world_rank == 0) printf("Validating result...\n");

            int localBad = 0;
            double localMin = 0.0;
            double localMax = 0.0;
            bool first = true;
            for (size_t k = 1; k <= nzl; ++k) {
                for (size_t j = 1; j <= nyl; ++j) {
                    const double* __restrict__ row = &cold[k * lxy + j * lx + 1];
                    for (size_t i = 0; i < nxl; ++i) {
                        const double v = row[i];
                        if (std::isnan(v) || std::isinf(v)) localBad = 1;
                        if (first) {
                            localMin = v;
                            localMax = v;
                            first = false;
                        } else {
                            localMin = std::min(localMin, v);
                            localMax = std::max(localMax, v);
                        }
                    }
                }
            }
            if (first) {  // no owned cells (cannot happen, but keep reductions neutral)
                localMin = HUGE_VAL;
                localMax = -HUGE_VAL;
            }

            int globalBad = 0;
            MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MAX, cart);

            bool valid = true;
            if (globalBad) {
                if (world_rank == 0) printf("Validation failed: found NaN or Inf value\n");
                valid = false;
            } else {
                double minVal = 0.0;
                double maxVal = 0.0;
                MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, cart);
                MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, cart);

                if (world_rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

                if (maxVal > 10.0 || minVal < -10.0) {
                    if (world_rank == 0) printf("Validation failed: values out of expected range\n");
                    valid = false;
                }
            }

            if (world_rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }

        for (int d = 0; d < 3; ++d) {
            for (int side = 0; side < 2; ++side) {
                for (int which = 0; which < 2; ++which) MPI_Type_free(&faceType[d][side][which]);
            }
        }
        MPI_Comm_free(&cart);
    }

    MPI_Comm_free(&comm_active);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
