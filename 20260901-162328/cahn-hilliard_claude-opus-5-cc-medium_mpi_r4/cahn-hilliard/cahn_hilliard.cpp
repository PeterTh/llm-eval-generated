#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Distributed memory parallelization (MPI)
//
// The 3D grid is decomposed into pencils along the Y and Z dimensions; the X
// dimension is kept contiguous and undivided so that the innermost loop stays
// long and vectorizable.  Every rank stores its local block surrounded by a
// one-cell halo.  The clamped ("copy") boundary condition of the original code
// is reproduced by filling the halo cells at the global domain boundary with a
// copy of the adjacent interior cell, so that the very same 7-point stencil can
// be applied uniformly to all local cells.
// ---------------------------------------------------------------------------

// 3D index calculation (local, padded array)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t mx, const size_t my) noexcept {
    return (z * my + y) * mx + x;
}

// Local domain description
struct Domain {
    // local interior extents
    size_t lnx = 0, lny = 0, lnz = 0;
    // padded extents (interior + 2 halo cells per dimension)
    size_t mx = 0, my = 0, mz = 0;
    // global offsets of the local interior block
    size_t oy = 0, oz = 0;
    // neighbours (MPI_PROC_NULL if at the global boundary)
    int yn = MPI_PROC_NULL, yp = MPI_PROC_NULL;
    int zn = MPI_PROC_NULL, zp = MPI_PROC_NULL;
    // derived datatype describing a Y-plane (lnz rows of mx doubles)
    MPI_Datatype yplane = MPI_DATATYPE_NULL;
    MPI_Comm comm = MPI_COMM_NULL;
};

// Fill the X halo cells with a copy of the neighbouring interior cell
// (clamped boundary condition; X is never decomposed).
static void fillXHalos(std::vector<double>& f, const Domain& d) {
    const size_t mx = d.mx, my = d.my;
    for (size_t z = 1; z <= d.lnz; ++z) {
        for (size_t y = 1; y <= d.lny; ++y) {
            double* __restrict__ row = &f[idx3(0, y, z, mx, my)];
            row[0] = row[1];
            row[mx - 1] = row[mx - 2];
        }
    }
}

// Post all halo exchanges for field f
static void exchangeBegin(std::vector<double>& f, const Domain& d, MPI_Request* req) {
    const size_t mx = d.mx, my = d.my;
    const int planeCount = static_cast<int>(mx * my);

    // Z direction: whole XY planes are contiguous in memory
    MPI_Irecv(&f[idx3(0, 0, 0, mx, my)], planeCount, MPI_DOUBLE, d.zn, 0, d.comm, &req[0]);
    MPI_Irecv(&f[idx3(0, 0, d.lnz + 1, mx, my)], planeCount, MPI_DOUBLE, d.zp, 1, d.comm, &req[1]);
    MPI_Isend(&f[idx3(0, 0, 1, mx, my)], planeCount, MPI_DOUBLE, d.zn, 1, d.comm, &req[2]);
    MPI_Isend(&f[idx3(0, 0, d.lnz, mx, my)], planeCount, MPI_DOUBLE, d.zp, 0, d.comm, &req[3]);

    // Y direction: strided rows, described by a derived datatype
    MPI_Irecv(&f[idx3(0, 0, 1, mx, my)], 1, d.yplane, d.yn, 2, d.comm, &req[4]);
    MPI_Irecv(&f[idx3(0, d.lny + 1, 1, mx, my)], 1, d.yplane, d.yp, 3, d.comm, &req[5]);
    MPI_Isend(&f[idx3(0, 1, 1, mx, my)], 1, d.yplane, d.yn, 3, d.comm, &req[6]);
    MPI_Isend(&f[idx3(0, d.lny, 1, mx, my)], 1, d.yplane, d.yp, 2, d.comm, &req[7]);
}

// Wait for the halo exchange and apply the clamped boundary condition at the
// global domain boundaries (where no neighbour exists).
static void exchangeEnd(std::vector<double>& f, const Domain& d, MPI_Request* req) {
    MPI_Waitall(8, req, MPI_STATUSES_IGNORE);

    const size_t mx = d.mx, my = d.my;
    if (d.zn == MPI_PROC_NULL) {
        std::memcpy(&f[idx3(0, 0, 0, mx, my)], &f[idx3(0, 0, 1, mx, my)], mx * my * sizeof(double));
    }
    if (d.zp == MPI_PROC_NULL) {
        std::memcpy(&f[idx3(0, 0, d.lnz + 1, mx, my)], &f[idx3(0, 0, d.lnz, mx, my)], mx * my * sizeof(double));
    }
    if (d.yn == MPI_PROC_NULL) {
        for (size_t z = 1; z <= d.lnz; ++z) {
            std::memcpy(&f[idx3(0, 0, z, mx, my)], &f[idx3(0, 1, z, mx, my)], mx * sizeof(double));
        }
    }
    if (d.yp == MPI_PROC_NULL) {
        for (size_t z = 1; z <= d.lnz; ++z) {
            std::memcpy(&f[idx3(0, d.lny + 1, z, mx, my)], &f[idx3(0, d.lny, z, mx, my)], mx * sizeof(double));
        }
    }
}

// Compute chemical potential on the given (inclusive) local index range
static void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu, const Domain& d,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                     const size_t z0, const size_t z1, const size_t y0, const size_t y1) {
    const size_t mx = d.mx, my = d.my;
    const double ixx = 1.0 / (dx * dx);
    const double iyy = 1.0 / (dy * dy);
    const double izz = 1.0 / (dz * dz);

    for (size_t z = z0; z <= z1; ++z) {
        for (size_t y = y0; y <= y1; ++y) {
            const double* __restrict__ cc = &c[idx3(0, y, z, mx, my)];
            const double* __restrict__ cxm = cc - 1;
            const double* __restrict__ cxp = cc + 1;
            const double* __restrict__ cym = &c[idx3(0, y - 1, z, mx, my)];
            const double* __restrict__ cyp = &c[idx3(0, y + 1, z, mx, my)];
            const double* __restrict__ czm = &c[idx3(0, y, z - 1, mx, my)];
            const double* __restrict__ czp = &c[idx3(0, y, z + 1, mx, my)];
            double* __restrict__ out = &mu[idx3(0, y, z, mx, my)];

            for (size_t x = 1; x <= d.lnx; ++x) {
                const double cv = cc[x];
                const double cxx = (cxp[x] + cxm[x] - 2.0 * cv) * ixx;
                const double cyy = (cyp[x] + cym[x] - 2.0 * cv) * iyy;
                const double czz = (czp[x] + czm[x] - 2.0 * cv) * izz;
                out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update step on the given (inclusive) local index range
static void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                               const std::vector<double>& mu, const Domain& d,
                               const double D, const double dt, const double dx, const double dy, const double dz,
                               const size_t z0, const size_t z1, const size_t y0, const size_t y1) {
    const size_t mx = d.mx, my = d.my;
    const double ixx = 1.0 / (dx * dx);
    const double iyy = 1.0 / (dy * dy);
    const double izz = 1.0 / (dz * dz);
    const double dtD = dt * D;

    for (size_t z = z0; z <= z1; ++z) {
        for (size_t y = y0; y <= y1; ++y) {
            const double* __restrict__ mc = &mu[idx3(0, y, z, mx, my)];
            const double* __restrict__ mxm = mc - 1;
            const double* __restrict__ mxp = mc + 1;
            const double* __restrict__ mym = &mu[idx3(0, y - 1, z, mx, my)];
            const double* __restrict__ myp = &mu[idx3(0, y + 1, z, mx, my)];
            const double* __restrict__ mzm = &mu[idx3(0, y, z - 1, mx, my)];
            const double* __restrict__ mzp = &mu[idx3(0, y, z + 1, mx, my)];
            const double* __restrict__ old = &cold[idx3(0, y, z, mx, my)];
            double* __restrict__ out = &cnew[idx3(0, y, z, mx, my)];

            for (size_t x = 1; x <= d.lnx; ++x) {
                const double mv = mc[x];
                const double mxx = (mxp[x] + mxm[x] - 2.0 * mv) * ixx;
                const double myy = (myp[x] + mym[x] - 2.0 * mv) * iyy;
                const double mzz = (mzp[x] + mzm[x] - 2.0 * mv) * izz;
                out[x] = old[x] + dtD * (mxx + myy + mzz);
            }
        }
    }
}

// Initialize the local part of the concentration field
static void initializeConcentration(std::vector<double>& c, const Domain& d,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t mx = d.mx, my = d.my;

    for (size_t z = 1; z <= d.lnz; ++z) {
        const size_t gz = d.oz + z - 1;
        for (size_t y = 1; y <= d.lny; ++y) {
            const size_t gy = d.oy + y - 1;
            const size_t base = gz * (nx * ny) + gy * nx;
            double* __restrict__ out = &c[idx3(0, y, z, mx, my)];
            for (size_t x = 1; x <= d.lnx; ++x) {
                const size_t linear_id = base + (x - 1);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                out[x] = -1.0 + 2.0 * pseudo;
            }
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

// Gather the distributed field into the global array on rank 0
static void gatherField(const std::vector<double>& f, const Domain& d, std::vector<double>& global,
                        const size_t nx, const size_t ny, const size_t nz, const int rank, const int nprocs) {
    const size_t mx = d.mx, my = d.my;

    // pack the local interior block
    std::vector<double> sendbuf(d.lnx * d.lny * d.lnz);
    for (size_t z = 1; z <= d.lnz; ++z) {
        for (size_t y = 1; y <= d.lny; ++y) {
            std::memcpy(&sendbuf[((z - 1) * d.lny + (y - 1)) * d.lnx], &f[idx3(1, y, z, mx, my)], d.lnx * sizeof(double));
        }
    }

    // collect the block descriptions (oy, oz, lny, lnz) of all ranks
    long long desc[4] = {static_cast<long long>(d.oy), static_cast<long long>(d.oz),
                         static_cast<long long>(d.lny), static_cast<long long>(d.lnz)};
    std::vector<long long> alldesc(rank == 0 ? 4 * static_cast<size_t>(nprocs) : 0);
    MPI_Gather(desc, 4, MPI_LONG_LONG, rank == 0 ? alldesc.data() : nullptr, 4, MPI_LONG_LONG, 0, d.comm);

    if (rank != 0) {
        MPI_Send(sendbuf.data(), static_cast<int>(sendbuf.size()), MPI_DOUBLE, 0, 7, d.comm);
        return;
    }

    global.assign(nx * ny * nz, 0.0);
    std::vector<double> recvbuf;
    for (int r = 0; r < nprocs; ++r) {
        const size_t roy = static_cast<size_t>(alldesc[4 * r + 0]);
        const size_t roz = static_cast<size_t>(alldesc[4 * r + 1]);
        const size_t rly = static_cast<size_t>(alldesc[4 * r + 2]);
        const size_t rlz = static_cast<size_t>(alldesc[4 * r + 3]);
        const double* src;
        if (r == 0) {
            src = sendbuf.data();
        } else {
            recvbuf.resize(nx * rly * rlz);
            MPI_Recv(recvbuf.data(), static_cast<int>(recvbuf.size()), MPI_DOUBLE, r, 7, d.comm, MPI_STATUS_IGNORE);
            src = recvbuf.data();
        }
        for (size_t z = 0; z < rlz; ++z) {
            for (size_t y = 0; y < rly; ++y) {
                std::memcpy(&global[(roz + z) * (nx * ny) + (roy + y) * nx], &src[(z * rly + y) * nx], nx * sizeof(double));
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) printUsage(argv[0]);
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

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nprocs);
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

    // ---- domain decomposition: pencils over (Y, Z), X stays contiguous ----
    // Pick the factorization py * pz = nactive (nactive <= nprocs) that minimizes
    // the per-rank halo surface; every rank must own at least one plane in Y and Z.
    int pz = 1, py = 1, nactive = 0;
    for (int n = nprocs; n >= 1 && nactive == 0; --n) {
        double best = -1.0;
        for (int cz = 1; cz <= n; ++cz) {
            if (n % cz != 0) continue;
            const int cy = n / cz;
            if (static_cast<size_t>(cz) > nz || static_cast<size_t>(cy) > ny) continue;
            // communication volume per rank ~ ny/py + nz/pz (in units of nx cells)
            const double cost = static_cast<double>(ny) / cy + static_cast<double>(nz) / cz;
            if (best < 0.0 || cost < best) {
                best = cost;
                pz = cz;
                py = cy;
                nactive = n;
            }
        }
    }

    // Ranks beyond the usable count stay idle (only possible for tiny grids)
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < nactive ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    nprocs = nactive;

    Domain d;
    int dims[2] = {pz, py};
    int periods[2] = {0, 0};
    MPI_Cart_create(active, 2, dims, periods, 1, &d.comm);
    MPI_Comm_free(&active);
    MPI_Comm_rank(d.comm, &rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(d.comm, rank, 2, coords);
    MPI_Cart_shift(d.comm, 0, 1, &d.zn, &d.zp);
    MPI_Cart_shift(d.comm, 1, 1, &d.yn, &d.yp);

    const size_t rz = static_cast<size_t>(coords[0]);
    const size_t ry = static_cast<size_t>(coords[1]);
    d.lnx = nx;
    d.lnz = nz / pz + (rz < nz % static_cast<size_t>(pz) ? 1 : 0);
    d.lny = ny / py + (ry < ny % static_cast<size_t>(py) ? 1 : 0);
    d.oz = rz * (nz / pz) + std::min(rz, nz % static_cast<size_t>(pz));
    d.oy = ry * (ny / py) + std::min(ry, ny % static_cast<size_t>(py));
    d.mx = d.lnx + 2;
    d.my = d.lny + 2;
    d.mz = d.lnz + 2;

    MPI_Type_vector(static_cast<int>(d.lnz), static_cast<int>(d.mx), static_cast<int>(d.mx * d.my),
                    MPI_DOUBLE, &d.yplane);
    MPI_Type_commit(&d.yplane);

    if (rank == 0) {
        printf("Decomposition: %d (z) x %d (y)\n", pz, py);
    }

    // Allocate arrays (local block + halo)
    const size_t localSize = d.mx * d.my * d.mz;
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, d, nx, ny, nz);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    // Interior region that does not depend on halo data (overlap region)
    const bool hasInner = (d.lnz >= 3 && d.lny >= 3);
    const size_t iz0 = 2, iz1 = d.lnz - 1, iy0 = 2, iy1 = d.lny - 1;

    MPI_Request req[8];

    MPI_Barrier(d.comm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (overlapping halo exchange of c)
        fillXHalos(cold, d);
        exchangeBegin(cold, d, req);
        if (hasInner) {
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, iz0, iz1, iy0, iy1);
        }
        exchangeEnd(cold, d, req);
        if (hasInner) {
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 1, 1, 1, d.lny);
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, d.lnz, d.lnz, 1, d.lny);
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, iz0, iz1, 1, 1);
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, iz0, iz1, d.lny, d.lny);
        } else {
            computeChemicalPotential(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 1, d.lnz, 1, d.lny);
        }

        // Update concentration (overlapping halo exchange of mu)
        fillXHalos(mu, d);
        exchangeBegin(mu, d, req);
        if (hasInner) {
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, iz0, iz1, iy0, iy1);
        }
        exchangeEnd(mu, d, req);
        if (hasInner) {
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, 1, 1, 1, d.lny);
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, d.lnz, d.lnz, 1, d.lny);
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, iz0, iz1, 1, 1);
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, iz0, iz1, d.lny, d.lny);
        } else {
            cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz, 1, d.lnz, 1, d.lny);
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = elapsed;
    MPI_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / maxElapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<double> global;
        gatherField(cold, d, global, nx, ny, nz, rank, nprocs);

        // Print results for external validation
        if (printResults && rank == 0) {
            print_results(global, "Concentration");
        }

        // Validation
        if (validate) {
            if (rank == 0) {
                printf("Validating result...\n");
                bool valid = validateResult(global, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, d.comm);
        }
    }

    MPI_Type_free(&d.yplane);
    MPI_Comm_free(&d.comm);
    MPI_Finalize();
    return exitCode;
}
