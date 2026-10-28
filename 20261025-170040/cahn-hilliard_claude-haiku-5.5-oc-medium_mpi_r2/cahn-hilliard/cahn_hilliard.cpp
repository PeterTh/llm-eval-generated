#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Physics {
    double dx, dy, dz;
    double dt;
    double e_AA, e_BB, e_AB;
    double gamma;
    double D;
};

// Cartesian decomposition of the global grid. Each rank owns a block of
// len[0] x len[1] x len[2] cells starting at start[] in global coordinates.
// Local storage is padded with one ghost layer on every side: owned cells
// have local coordinates in [1, len], ghosts live at 0 and len+1.
struct Decomp {
    int dims[3];      // process grid extents (x, y, z)
    int coords[3];    // this rank's position in the process grid
    int nbrLo[3];     // lower neighbour per dimension (MPI_PROC_NULL at domain edge)
    int nbrHi[3];     // upper neighbour per dimension (MPI_PROC_NULL at domain edge)
    size_t gn[3];     // global extents
    size_t start[3];  // global offset of the local block
    size_t len[3];    // local block extents
    size_t pext[3];   // padded extents (len + 2)
};

// Balanced block partition of n items over p parts.
inline size_t blockStart(size_t coord, size_t n, size_t p) { return coord * n / p; }
inline size_t blockLen(size_t coord, size_t n, size_t p) { return (coord + 1) * n / p - coord * n / p; }
inline size_t ceilDiv(size_t a, size_t b) { return (a + b - 1) / b; }

// Linear index into a padded local array (x fastest, then y, then z).
inline size_t lidx(const Decomp& dec, const size_t p[3]) {
    return (p[2] * dec.pext[1] + p[1]) * dec.pext[0] + p[0];
}

// Choose a process grid dims with product <= P and every extent <= the global
// extent, so that every active rank owns at least one cell. The product is
// maximized first (ranks left over become idle), then the largest block is
// minimized (load balance), then the halo surface (communication).
// Returns the number of active ranks (the product of dims).
int chooseDims(int P, const size_t gn[3], int dims[3]) {
    int bestCount = 0;
    size_t bestVol = 0, bestSurf = 0;
    const size_t maxPx = std::min(static_cast<size_t>(P), gn[0]);
    for (size_t px = 1; px <= maxPx; ++px) {
        const size_t maxPy = std::min(static_cast<size_t>(P) / px, gn[1]);
        for (size_t py = 1; py <= maxPy; ++py) {
            const size_t pz = std::min(static_cast<size_t>(P) / (px * py), gn[2]);
            const int count = static_cast<int>(px * py * pz);
            const size_t bx = ceilDiv(gn[0], px), by = ceilDiv(gn[1], py), bz = ceilDiv(gn[2], pz);
            const size_t vol = bx * by * bz;
            const size_t surf = bx * by + by * bz + bz * bx;
            const bool better = count > bestCount ||
                                (count == bestCount && (vol < bestVol || (vol == bestVol && surf < bestSurf)));
            if (better) {
                bestCount = count;
                bestVol = vol;
                bestSurf = surf;
                dims[0] = static_cast<int>(px);
                dims[1] = static_cast<int>(py);
                dims[2] = static_cast<int>(pz);
            }
        }
    }
    return bestCount;
}

// Visit every owned cell of the face plane `plane` (local coordinate along d),
// iterating over the owned extents of the two other dimensions.
template <class F>
inline void forEachFaceCell(const Decomp& dec, int d, size_t plane, F&& fn) {
    const int a = (d + 1) % 3, b = (d + 2) % 3;
    size_t p[3];
    p[d] = plane;
    for (size_t jb = 1; jb <= dec.len[b]; ++jb) {
        p[b] = jb;
        for (size_t ia = 1; ia <= dec.len[a]; ++ia) {
            p[a] = ia;
            fn(lidx(dec, p));
        }
    }
}

// Copy plane srcPlane onto plane dstPlane along dimension d (owned extents only).
void copyFacePlane(double* f, const Decomp& dec, int d, size_t dstPlane, size_t srcPlane) {
    const int a = (d + 1) % 3, b = (d + 2) % 3;
    size_t p[3], q[3];
    p[d] = dstPlane;
    q[d] = srcPlane;
    for (size_t jb = 1; jb <= dec.len[b]; ++jb) {
        p[b] = q[b] = jb;
        for (size_t ia = 1; ia <= dec.len[a]; ++ia) {
            p[a] = q[a] = ia;
            f[lidx(dec, p)] = f[lidx(dec, q)];
        }
    }
}

inline size_t faceSize(const Decomp& dec, int d) {
    return dec.len[(d + 1) % 3] * dec.len[(d + 2) % 3];
}

// Nonblocking halo exchange of one padded field. Ghost cells on a global
// domain edge are filled from the adjacent owned plane, which reproduces the
// clamped boundary condition of the serial code.
struct Halo {
    std::array<std::vector<double>, 6> sbuf;  // index 2*d (lo face) / 2*d+1 (hi face)
    std::array<std::vector<double>, 6> rbuf;
    std::array<MPI_Request, 12> req;
};

void haloInit(const Decomp& dec, Halo& h) {
    for (int d = 0; d < 3; ++d) {
        const size_t n = faceSize(dec, d);
        for (int s = 0; s < 2; ++s) {
            h.sbuf[2 * d + s].assign(n, 0.0);
            h.rbuf[2 * d + s].assign(n, 0.0);
        }
    }
}

void haloStart(const Decomp& dec, const double* f, Halo& h, MPI_Comm comm) {
    int n = 0;
    for (int d = 0; d < 3; ++d) {
        const int count = static_cast<int>(faceSize(dec, d));
        const int lo = dec.nbrLo[d], hi = dec.nbrHi[d];
        // Messages travelling towards +d use tag 2d, towards -d use tag 2d+1.
        MPI_Irecv(h.rbuf[2 * d].data(), count, MPI_DOUBLE, lo, 2 * d, comm, &h.req[n++]);
        MPI_Irecv(h.rbuf[2 * d + 1].data(), count, MPI_DOUBLE, hi, 2 * d + 1, comm, &h.req[n++]);

        double* sHi = h.sbuf[2 * d + 1].data();
        size_t pos = 0;
        forEachFaceCell(dec, d, dec.len[d], [&](size_t idx) { sHi[pos++] = f[idx]; });
        MPI_Isend(sHi, count, MPI_DOUBLE, hi, 2 * d, comm, &h.req[n++]);

        double* sLo = h.sbuf[2 * d].data();
        pos = 0;
        forEachFaceCell(dec, d, 1, [&](size_t idx) { sLo[pos++] = f[idx]; });
        MPI_Isend(sLo, count, MPI_DOUBLE, lo, 2 * d + 1, comm, &h.req[n++]);
    }
}

void haloFinish(const Decomp& dec, double* f, Halo& h) {
    MPI_Waitall(12, h.req.data(), MPI_STATUSES_IGNORE);
    for (int d = 0; d < 3; ++d) {
        const size_t hiGhost = dec.len[d] + 1;
        if (dec.nbrLo[d] != MPI_PROC_NULL) {
            const double* r = h.rbuf[2 * d].data();
            size_t pos = 0;
            forEachFaceCell(dec, d, 0, [&](size_t idx) { f[idx] = r[pos++]; });
        } else {
            copyFacePlane(f, dec, d, 0, 1);
        }
        if (dec.nbrHi[d] != MPI_PROC_NULL) {
            const double* r = h.rbuf[2 * d + 1].data();
            size_t pos = 0;
            forEachFaceCell(dec, d, hiGhost, [&](size_t idx) { f[idx] = r[pos++]; });
        } else {
            copyFacePlane(f, dec, d, hiGhost, dec.len[d]);
        }
    }
}

// Inclusive box of padded local coordinates (signed to allow empty boxes).
struct Box {
    long i0, i1, j0, j1, k0, k1;
};

// Owned cells whose 7-point stencil touches no ghost cell.
Box innerBox(const Decomp& dec) {
    return Box{2, static_cast<long>(dec.len[0]) - 1,
               2, static_cast<long>(dec.len[1]) - 1,
               2, static_cast<long>(dec.len[2]) - 1};
}

// Owned cells on the surface of the block. Together with innerBox these
// partition the owned cells exactly.
std::vector<Box> shellBoxes(const Decomp& dec) {
    const long L0 = static_cast<long>(dec.len[0]);
    const long L1 = static_cast<long>(dec.len[1]);
    const long L2 = static_cast<long>(dec.len[2]);
    std::vector<Box> boxes;
    // z faces (full x, y)
    boxes.push_back({1, L0, 1, L1, 1, 1});
    if (L2 > 1) boxes.push_back({1, L0, 1, L1, L2, L2});
    // y faces (full x, z interior)
    boxes.push_back({1, L0, 1, 1, 2, L2 - 1});
    if (L1 > 1) boxes.push_back({1, L0, L1, L1, 2, L2 - 1});
    // x faces (y, z interior)
    boxes.push_back({1, 1, 2, L1 - 1, 2, L2 - 1});
    if (L0 > 1) boxes.push_back({L0, L0, 2, L1 - 1, 2, L2 - 1});
    return boxes;
}

// 7-point Laplacian with unit-stride x, stride sx for y and sxy for z.
inline double laplacian7(const double* f, size_t idx, size_t sx, size_t sxy, const Physics& ph) {
    const double cxx = (f[idx + 1] + f[idx - 1] - 2.0 * f[idx]) / (ph.dx * ph.dx);
    const double cyy = (f[idx + sx] + f[idx - sx] - 2.0 * f[idx]) / (ph.dy * ph.dy);
    const double czz = (f[idx + sxy] + f[idx - sxy] - 2.0 * f[idx]) / (ph.dz * ph.dz);
    return cxx + cyy + czz;
}

void computeChemicalPotentialBox(const double* c, double* mu, const Decomp& dec, const Box& b, const Physics& ph) {
    const size_t sx = dec.pext[0];
    const size_t sxy = dec.pext[0] * dec.pext[1];
    for (long k = b.k0; k <= b.k1; ++k) {
        for (long j = b.j0; j <= b.j1; ++j) {
            const size_t row = static_cast<size_t>(k) * sxy + static_cast<size_t>(j) * sx;
            for (long i = b.i0; i <= b.i1; ++i) {
                const size_t idx = row + static_cast<size_t>(i);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * ph.e_AA + (cv - 1.0) * ph.e_BB - 2.0 * cv * ph.e_AB)
                          + 3.0 * cv + cv * cv * cv
                          - ph.gamma * laplacian7(c, idx, sx, sxy, ph);
            }
        }
    }
}

void cahnHilliardUpdateBox(const double* cold, double* cnew, const double* mu, const Decomp& dec,
                           const Box& b, const Physics& ph) {
    const size_t sx = dec.pext[0];
    const size_t sxy = dec.pext[0] * dec.pext[1];
    for (long k = b.k0; k <= b.k1; ++k) {
        for (long j = b.j0; j <= b.j1; ++j) {
            const size_t row = static_cast<size_t>(k) * sxy + static_cast<size_t>(j) * sx;
            for (long i = b.i0; i <= b.i1; ++i) {
                const size_t idx = row + static_cast<size_t>(i);
                cnew[idx] = cold[idx] + ph.dt * ph.D * laplacian7(mu, idx, sx, sxy, ph);
            }
        }
    }
}

// Advance the field by `iterations` steps. Communication is overlapped with
// computation on the block interior.
void runSimulation(const Decomp& dec, std::vector<double>& cold, std::vector<double>& cnew,
                   std::vector<double>& mu, int iterations, const Physics& ph, MPI_Comm comm) {
    Halo hc, hm;
    haloInit(dec, hc);
    haloInit(dec, hm);
    const Box inner = innerBox(dec);
    const std::vector<Box> shell = shellBoxes(dec);

    for (int t = 0; t < iterations; ++t) {
        // Chemical potential: needs ghost cells of c.
        haloStart(dec, cold.data(), hc, comm);
        computeChemicalPotentialBox(cold.data(), mu.data(), dec, inner, ph);
        haloFinish(dec, cold.data(), hc);
        for (const Box& b : shell) computeChemicalPotentialBox(cold.data(), mu.data(), dec, b, ph);

        // Concentration update: needs ghost cells of mu. The interior only
        // reads owned mu cells, so it overlaps the mu halo exchange.
        haloStart(dec, mu.data(), hm, comm);
        cahnHilliardUpdateBox(cold.data(), cnew.data(), mu.data(), dec, inner, ph);
        haloFinish(dec, mu.data(), hm);
        for (const Box& b : shell) cahnHilliardUpdateBox(cold.data(), cnew.data(), mu.data(), dec, b, ph);

        std::swap(cold, cnew);
    }
}

// Gather the full field into `full` in global x-fastest order on rank 0 of
// `comm` (the active Cartesian communicator).
void gatherField(const Decomp& dec, const std::vector<double>& local, std::vector<double>& full, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const size_t count = dec.len[0] * dec.len[1] * dec.len[2];
    std::vector<double> packed(count);
    size_t pos = 0;
    for (size_t k = 1; k <= dec.len[2]; ++k)
        for (size_t j = 1; j <= dec.len[1]; ++j)
            for (size_t i = 1; i <= dec.len[0]; ++i) {
                const size_t p[3] = {i, j, k};
                packed[pos++] = local[lidx(dec, p)];
            }

    std::vector<int> counts(size), displs(size);
    std::vector<std::array<size_t, 3>> starts(size), lens(size);
    int off = 0;
    for (int r = 0; r < size; ++r) {
        int c[3];
        MPI_Cart_coords(comm, r, 3, c);
        for (int d = 0; d < 3; ++d) {
            starts[r][d] = blockStart(c[d], dec.gn[d], dec.dims[d]);
            lens[r][d] = blockLen(c[d], dec.gn[d], dec.dims[d]);
        }
        counts[r] = static_cast<int>(lens[r][0] * lens[r][1] * lens[r][2]);
        displs[r] = off;
        off += counts[r];
    }

    std::vector<double> gathered;
    if (rank == 0) gathered.resize(off);
    MPI_Gatherv(packed.data(), static_cast<int>(count), MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        const size_t nx = dec.gn[0], ny = dec.gn[1];
        full.assign(dec.gn[0] * dec.gn[1] * dec.gn[2], 0.0);
        for (int r = 0; r < size; ++r) {
            const size_t bx = lens[r][0], by = lens[r][1], bz = lens[r][2];
            const double* src = gathered.data() + displs[r];
            size_t q = 0;
            for (size_t k = 0; k < bz; ++k)
                for (size_t j = 0; j < by; ++j)
                    for (size_t i = 0; i < bx; ++i) {
                        const size_t g = (starts[r][2] + k) * nx * ny + (starts[r][1] + j) * nx + (starts[r][0] + i);
                        full[g] = src[q++];
                    }
        }
    }
}

// Distributed equivalent of validateResult(): global NaN/Inf check and range check.
// Called on every world rank; idle ranks own no cells and contribute neutral values.
bool validateDistributed(const Decomp& dec, const std::vector<double>& c, MPI_Comm comm, int rank) {
    int localBad = 0;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (size_t k = 1; k <= dec.len[2]; ++k)
        for (size_t j = 1; j <= dec.len[1]; ++j)
            for (size_t i = 1; i <= dec.len[0]; ++i) {
                const size_t p[3] = {i, j, k};
                const double v = c[lidx(dec, p)];
                if (std::isnan(v) || std::isinf(v)) localBad = 1;
                localMin = std::min(localMin, v);
                localMax = std::max(localMax, v);
            }

    int anyBad = 0;
    MPI_Allreduce(&localBad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = 0.0, maxVal = 0.0;
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    int status = -1;  // >= 0 means exit now with this status
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
            if (root) printUsage(argv[0]);
            status = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }
    if (status >= 0) {
        MPI_Finalize();
        return status;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const Physics ph{1.0, 1.0, 1.0, 0.01, -(2.0 / 9.0), -(2.0 / 9.0), (2.0 / 9.0), 0.5, 1.0};

    const size_t gn[3] = {nx, ny, nz};
    const size_t gridSize = nx * ny * nz;

    // Ranks beyond the process-grid size are idle: they hold no cells and do
    // not join the Cartesian communicator, but still take part in the
    // world-level reductions below.
    Decomp dec{};
    std::vector<double> cold, cnew, mu;
    MPI_Comm work = MPI_COMM_NULL;
    double maxSecs = 0.0;

    if (gridSize > 0) {
        const int active = chooseDims(size, gn, dec.dims);
        const bool isActive = rank < active;
        MPI_Comm_split(MPI_COMM_WORLD, isActive ? 0 : MPI_UNDEFINED, rank, &work);

        if (isActive) {
            int periods[3] = {0, 0, 0};
            MPI_Comm cart = MPI_COMM_NULL;
            MPI_Cart_create(work, 3, dec.dims, periods, 0, &cart);
            MPI_Comm_free(&work);
            work = cart;

            MPI_Cart_coords(work, rank, 3, dec.coords);
            for (int d = 0; d < 3; ++d) {
                MPI_Cart_shift(work, d, 1, &dec.nbrLo[d], &dec.nbrHi[d]);
                dec.gn[d] = gn[d];
                dec.start[d] = blockStart(dec.coords[d], gn[d], dec.dims[d]);
                dec.len[d] = blockLen(dec.coords[d], gn[d], dec.dims[d]);
                dec.pext[d] = dec.len[d] + 2;
            }

            const size_t padded = dec.pext[0] * dec.pext[1] * dec.pext[2];
            cold.assign(padded, 0.0);
            cnew.assign(padded, 0.0);
            mu.assign(padded, 0.0);

            // Initialize concentration field (pseudo-random values in [-1, 1]
            // derived from the global linear index, independent of decomposition).
            const size_t vol = gridSize;
            for (size_t k = 1; k <= dec.len[2]; ++k)
                for (size_t j = 1; j <= dec.len[1]; ++j)
                    for (size_t i = 1; i <= dec.len[0]; ++i) {
                        const size_t p[3] = {i, j, k};
                        const size_t gz = dec.start[2] + k - 1;
                        const size_t gy = dec.start[1] + j - 1;
                        const size_t gx = dec.start[0] + i - 1;
                        const size_t linear_id = gz * (nx * ny) + gy * nx + gx;
                        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                        cold[lidx(dec, p)] = -1.0 + 2.0 * pseudo;
                    }
        }
    }

    if (root) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    if (work != MPI_COMM_NULL) {
        runSimulation(dec, cold, cnew, mu, iterations, ph, work);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double secs = MPI_Wtime() - t0;
    MPI_Reduce(&secs, &maxSecs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long long ms = static_cast<long long>(maxSecs * 1000.0);
        printf("Computation time: %lld ms\n", ms);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / maxSecs / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Full field on rank 0 for printing.
    if (printResults) {
        std::vector<double> full;
        if (work != MPI_COMM_NULL) gatherField(dec, cold, full, work);
        if (root) print_results(full, "Concentration");
    }

    int exitCode = 0;
    if (validate) {
        if (root) printf("Validating result...\n");
        // An empty grid has no values to validate.
        const bool valid = (gridSize == 0) || validateDistributed(dec, cold, MPI_COMM_WORLD, rank);
        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        exitCode = valid ? 0 : 1;
    }

    if (work != MPI_COMM_NULL) MPI_Comm_free(&work);
    MPI_Finalize();
    return exitCode;
}
