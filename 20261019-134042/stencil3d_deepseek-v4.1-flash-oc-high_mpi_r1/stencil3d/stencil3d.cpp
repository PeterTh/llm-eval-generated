#include <algorithm>
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

inline size_t partSize(const size_t n, const int p, const int c) noexcept {
    if (c < 0 || c >= p) return 0;
    const size_t pc = static_cast<size_t>(p);
    const size_t cc = static_cast<size_t>(c);
    return n / pc + (cc < (n % pc) ? 1 : 0);
}

inline size_t partStart(const size_t n, const int p, const int c) noexcept {
    const size_t pc = static_cast<size_t>(p);
    const size_t cc = static_cast<size_t>(c);
    const size_t base = n / pc;
    const size_t rem = n % pc;
    return cc * base + std::min(cc, rem);
}

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype ytype = MPI_DATATYPE_NULL;

    int rank = 0;
    int size = 1;
    int py = 1, pz = 1;
    int cy = 0, cz = 0;

    int ym = MPI_PROC_NULL, yp = MPI_PROC_NULL;
    int zm = MPI_PROC_NULL, zp = MPI_PROC_NULL;
    int yme = MPI_PROC_NULL, ype = MPI_PROC_NULL;
    int zme = MPI_PROC_NULL, zpe = MPI_PROC_NULL;

    size_t nx = 0, ny = 0, nz = 0;
    size_t y0 = 0, z0 = 0;
    size_t lny = 0, lnz = 0;
    size_t nyh = 0, nzh = 0;
    size_t plane = 0;
    size_t localSize = 0;
};

void initializeGrid(Domain& d, std::vector<Real>& grid) {
    std::fill(grid.begin(), grid.end(), Real(0));

    for (size_t lz = 0; lz < d.lnz; ++lz) {
        const size_t gz = d.z0 + lz;
        const size_t zbase = (lz + 1) * d.plane;
        for (size_t ly = 0; ly < d.lny; ++ly) {
            const size_t gy = d.y0 + ly;
            const size_t base = zbase + (ly + 1) * d.nx;
            const size_t gbase = (gz * d.ny + gy) * d.nx;
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t idx = gbase + x;
                grid[base + x] = (idx % 19) * 1.0;
            }
        }
    }
}

void exchangeHalos(Domain& d, std::vector<Real>& grid) {
    Real* const p = grid.data();

    Real* const ySendMin = p + ((1 * d.nyh) + 1) * d.nx;
    Real* const yRecvMax = p + ((1 * d.nyh) + (d.nyh - 1)) * d.nx;
    Real* const ySendMax = p + ((1 * d.nyh) + d.lny) * d.nx;
    Real* const yRecvMin = p + ((1 * d.nyh) + 0) * d.nx;

    Real* const zSendMin = p + ((1 * d.nyh) + 1) * d.nx;
    Real* const zRecvMax = p + (((d.nzh - 1) * d.nyh) + 1) * d.nx;
    Real* const zSendMax = p + ((d.lnz * d.nyh) + 1) * d.nx;
    Real* const zRecvMin = p + ((0 * d.nyh) + 1) * d.nx;

    const int zcount = static_cast<int>(d.lny * d.nx);

    MPI_Request req[8];
    int nreq = 0;

    if (d.ype != MPI_PROC_NULL)
        MPI_Irecv(yRecvMax, 1, d.ytype, d.ype, 10, d.comm, &req[nreq++]);
    if (d.yme != MPI_PROC_NULL)
        MPI_Isend(ySendMin, 1, d.ytype, d.yme, 10, d.comm, &req[nreq++]);
    if (d.yme != MPI_PROC_NULL)
        MPI_Irecv(yRecvMin, 1, d.ytype, d.yme, 11, d.comm, &req[nreq++]);
    if (d.ype != MPI_PROC_NULL)
        MPI_Isend(ySendMax, 1, d.ytype, d.ype, 11, d.comm, &req[nreq++]);

    if (d.zpe != MPI_PROC_NULL)
        MPI_Irecv(zRecvMax, zcount, MPI_DOUBLE, d.zpe, 20, d.comm, &req[nreq++]);
    if (d.zme != MPI_PROC_NULL)
        MPI_Isend(zSendMin, zcount, MPI_DOUBLE, d.zme, 20, d.comm, &req[nreq++]);
    if (d.zme != MPI_PROC_NULL)
        MPI_Irecv(zRecvMin, zcount, MPI_DOUBLE, d.zme, 21, d.comm, &req[nreq++]);
    if (d.zpe != MPI_PROC_NULL)
        MPI_Isend(zSendMax, zcount, MPI_DOUBLE, d.zpe, 21, d.comm, &req[nreq++]);

    if (nreq > 0) {
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
    }
}

// 7-point stencil computation
void stencilIteration(Domain& d,
                      std::vector<Real>& input,
                      std::vector<Real>& output) {
    exchangeHalos(d, input);

    const Real* const in = input.data();
    Real* const out = output.data();

    const size_t nx = d.nx;
    const size_t ny = d.ny;
    const size_t nz = d.nz;
    const size_t lny = d.lny;
    const size_t lnz = d.lnz;
    const size_t plane = d.plane;

    for (size_t lz = 1; lz <= lnz; ++lz) {
        const size_t gz = d.z0 + (lz - 1);
        const bool zBoundary = (gz == 0 || gz == nz - 1);
        const size_t zbase = lz * plane;

        for (size_t ly = 1; ly <= lny; ++ly) {
            const size_t gy = d.y0 + (ly - 1);
            const size_t base = zbase + ly * nx;

            if (zBoundary || gy == 0 || gy == ny - 1) {
                std::memcpy(&out[base], &in[base], nx * sizeof(Real));
                continue;
            }

            out[base] = in[base];
            out[base + nx - 1] = in[base + nx - 1];

            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t i = base + x;
                out[i] = (in[i] +
                          in[i - 1] +
                          in[i + 1] +
                          in[i - nx] +
                          in[i + nx] +
                          in[i - plane] +
                          in[i + plane]) / 7.0;
            }
        }
    }
}

bool validateResult(Domain& d, const std::vector<Real>& grid) {
    bool localBad = false;
    Real localMin = std::numeric_limits<Real>::max();
    Real localMax = std::numeric_limits<Real>::lowest();

    for (size_t lz = 1; lz <= d.lnz; ++lz) {
        const size_t zbase = lz * d.plane;
        for (size_t ly = 1; ly <= d.lny; ++ly) {
            const size_t base = zbase + ly * d.nx;
            for (size_t x = 0; x < d.nx; ++x) {
                const Real val = grid[base + x];
                if (std::isnan(val) || std::isinf(val)) {
                    localBad = true;
                }
                localMin = std::min(localMin, val);
                localMax = std::max(localMax, val);
            }
        }
    }

    int bad = localBad ? 1 : 0;
    int globalBad = 0;
    MPI_Allreduce(&bad, &globalBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (globalBad) {
        if (d.rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (d.rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    }

    if (globalMax > 1e6 || globalMin < -1e6) {
        if (d.rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
        return false;
    }

    return true;
}

void gatherGrid(Domain& d, const std::vector<Real>& grid,
                std::vector<Real>& global, int root) {
    const int localCount = static_cast<int>(d.lny * d.lnz * d.nx);

    std::vector<int> counts(d.size);
    std::vector<int> displs(d.size);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total = 0;
    for (int r = 0; r < d.size; ++r) {
        displs[r] = total;
        total += counts[r];
    }

    std::vector<Real> packed;
    packed.reserve(static_cast<size_t>(localCount));
    for (size_t lz = 1; lz <= d.lnz; ++lz) {
        const size_t zbase = lz * d.plane;
        for (size_t ly = 1; ly <= d.lny; ++ly) {
            const Real* const src = &grid[zbase + ly * d.nx];
            packed.insert(packed.end(), src, src + d.nx);
        }
    }

    std::vector<Real> recv;
    Real* recvPtr = nullptr;
    if (d.rank == root) {
        recv.resize(static_cast<size_t>(total));
        recvPtr = recv.data();
    }

    MPI_Gatherv(packed.empty() ? nullptr : packed.data(), localCount, MPI_DOUBLE,
                recvPtr, counts.data(), displs.data(), MPI_DOUBLE,
                root, MPI_COMM_WORLD);

    const int meta[4] = {static_cast<int>(d.y0),
                         static_cast<int>(d.z0),
                         static_cast<int>(d.lny),
                         static_cast<int>(d.lnz)};

    std::vector<int> allmeta;
    int* allmetaPtr = nullptr;
    if (d.rank == root) {
        allmeta.resize(static_cast<size_t>(4 * d.size));
        allmetaPtr = allmeta.data();
    }

    MPI_Gather(meta, 4, MPI_INT, allmetaPtr, 4, MPI_INT, root, MPI_COMM_WORLD);

    if (d.rank != root) return;

    global.assign(d.nx * d.ny * d.nz, Real(0));
    for (int r = 0; r < d.size; ++r) {
        const int y0 = allmeta[4 * r + 0];
        const int z0 = allmeta[4 * r + 1];
        const int lny = allmeta[4 * r + 2];
        const int lnz = allmeta[4 * r + 3];
        const Real* src = recv.data() + displs[r];
        for (int z = 0; z < lnz; ++z) {
            for (int y = 0; y < lny; ++y) {
                const size_t dst = (static_cast<size_t>(z0 + z) * d.ny +
                                    static_cast<size_t>(y0 + y)) * d.nx;
                std::memcpy(&global[dst], src, d.nx * sizeof(Real));
                src += d.nx;
            }
        }
    }
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

    if (worldRank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dims);
    const int periods[2] = {0, 0};

    Domain d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &d.comm);

    int cartRank = 0;
    int coords[2] = {0, 0};
    MPI_Comm_rank(d.comm, &cartRank);
    MPI_Cart_coords(d.comm, cartRank, 2, coords);

    d.rank = worldRank;
    d.size = worldSize;
    d.py = dims[0];
    d.pz = dims[1];
    d.cy = coords[0];
    d.cz = coords[1];
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;

    d.lny = partSize(ny, d.py, d.cy);
    d.lnz = partSize(nz, d.pz, d.cz);
    d.y0 = partStart(ny, d.py, d.cy);
    d.z0 = partStart(nz, d.pz, d.cz);
    d.nyh = d.lny + 2;
    d.nzh = d.lnz + 2;
    d.plane = d.nx * d.nyh;
    d.localSize = d.plane * d.nzh;

    MPI_Cart_shift(d.comm, 0, 1, &d.ym, &d.yp);
    MPI_Cart_shift(d.comm, 1, 1, &d.zm, &d.zp);

    const bool haveData = (d.lny > 0 && d.lnz > 0);
    const bool yMinHasData = (partSize(ny, d.py, d.cy - 1) > 0);
    const bool yMaxHasData = (partSize(ny, d.py, d.cy + 1) > 0);
    const bool zMinHasData = (partSize(nz, d.pz, d.cz - 1) > 0);
    const bool zMaxHasData = (partSize(nz, d.pz, d.cz + 1) > 0);

    d.yme = (haveData && d.ym != MPI_PROC_NULL && yMinHasData) ? d.ym : MPI_PROC_NULL;
    d.ype = (haveData && d.yp != MPI_PROC_NULL && yMaxHasData) ? d.yp : MPI_PROC_NULL;
    d.zme = (haveData && d.zm != MPI_PROC_NULL && zMinHasData) ? d.zm : MPI_PROC_NULL;
    d.zpe = (haveData && d.zp != MPI_PROC_NULL && zMaxHasData) ? d.zp : MPI_PROC_NULL;

    if (d.lnz > 0) {
        MPI_Type_vector(static_cast<int>(d.lnz),
                        static_cast<int>(d.nx),
                        static_cast<int>(d.plane),
                        MPI_DOUBLE, &d.ytype);
        MPI_Type_commit(&d.ytype);
    }

    // Allocate grids (double buffering)
    std::vector<Real> grid1(d.localSize);
    std::vector<Real> grid2(d.localSize);

    // Initialize
    if (worldRank == 0) printf("Initializing grid...\n");
    initializeGrid(d, grid1);

    // Run stencil iterations
    if (worldRank == 0) printf("Running stencil computation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d, grid1, grid2);
        } else {
            stencilIteration(d, grid2, grid1);
        }
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Allreduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const long duration = static_cast<long>(elapsed * 1000.0);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration);

        // Calculate performance metrics
        const double cellUpdates =
            static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups =
            cellUpdates / (static_cast<double>(duration) / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        std::vector<Real> globalGrid;
        gatherGrid(d, finalGrid, globalGrid, 0);
        if (worldRank == 0) {
            print_results(globalGrid, "Grid");
        }
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        const bool valid = validateResult(d, finalGrid);

        if (worldRank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitCode = valid ? 0 : 1;
    }

    if (d.ytype != MPI_DATATYPE_NULL) {
        MPI_Type_free(&d.ytype);
    }
    if (d.comm != MPI_COMM_NULL) {
        MPI_Comm_free(&d.comm);
    }

    MPI_Finalize();
    return exitCode;
}
