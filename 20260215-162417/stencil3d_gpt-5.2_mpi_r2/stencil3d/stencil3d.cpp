#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

// Avoid pulling deprecated MPI C++ bindings from some MPI implementations.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation (row-major, contiguous X then Y then Z)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static void printUsage(const char* progName) {
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

static void computeZDecomp(const size_t nz, const int rank, const int size, size_t& zStart, size_t& localNz) {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);

    const size_t base = (p == 0) ? 0 : (nz / p);
    const size_t rem = (p == 0) ? 0 : (nz % p);

    localNz = base + (r < rem ? 1 : 0);
    zStart = r * base + std::min(r, rem);
}

static void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t localNz, const size_t zStartGlobal) {
    const size_t plane = nx * ny;
    for (size_t lz = 0; lz < localNz; ++lz) {
        const size_t gz = zStartGlobal + lz;
        const size_t zOff = (lz + 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowOff = zOff + y * nx;
            const size_t gRowBase = gz * plane + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t gIdx = gRowBase + x;
                grid[rowOff + x] = static_cast<Real>(gIdx % 19);
            }
        }
    }
}

static bool validateDistributed(const std::vector<Real>& localFinal,
                               const size_t nx,
                               const size_t ny,
                               const size_t nz,
                               const size_t localNz,
                               const size_t zStart,
                               MPI_Comm comm,
                               int rank) {
    if (nz == 0 || nx == 0 || ny == 0) {
        if (rank == 0) {
            printf("Value range: [0.000000, 0.000000]\n");
        }
        return true;
    }

    const size_t plane = nx * ny;

    int localOk = 1;
    Real localMin = 0.0;
    Real localMax = 0.0;
    bool haveAny = false;

    for (size_t lz = 0; lz < localNz; ++lz) {
        const size_t gz = zStart + lz;
        (void)gz;
        const Real* p = localFinal.data() + (lz + 1) * plane;
        for (size_t i = 0; i < plane; ++i) {
            const Real v = p[i];
            if (std::isnan(v) || std::isinf(v)) {
                localOk = 0;
                break;
            }
            if (!haveAny) {
                localMin = v;
                localMax = v;
                haveAny = true;
            } else {
                localMin = std::min(localMin, v);
                localMax = std::max(localMax, v);
            }
        }
        if (!localOk) break;
    }

    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_LAND, comm);

    // If some ranks have no planes, make them neutral for min/max reductions.
    Real minIn = haveAny ? localMin : std::numeric_limits<Real>::infinity();
    Real maxIn = haveAny ? localMax : -std::numeric_limits<Real>::infinity();

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Allreduce(&minIn, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxIn, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (!globalOk) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);

        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    // Keep all ranks consistent with rank 0 decision.
    int rootDecision = (rank == 0) ? 1 : 0;
    if (rank == 0) {
        rootDecision = (globalOk && (globalMax <= 1e6) && (globalMin >= -1e6)) ? 1 : 0;
    }
    MPI_Bcast(&rootDecision, 1, MPI_INT, 0, comm);
    return rootDecision == 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;

    int exitCode = -1;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
        if (exitCode == -1) {
            if (ny == 0) ny = nx;
            if (nz == 0) nz = nx;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    if (exitCode != -1) {
        MPI_Finalize();
        return exitCode;
    }

    // Broadcast parameters (portable size_t handling)
    uint64_t nx64 = static_cast<uint64_t>(nx);
    uint64_t ny64 = static_cast<uint64_t>(ny);
    uint64_t nz64 = static_cast<uint64_t>(nz);
    MPI_Bcast(&nx64, 1, MPI_UINT64_T, 0, comm);
    MPI_Bcast(&ny64, 1, MPI_UINT64_T, 0, comm);
    MPI_Bcast(&nz64, 1, MPI_UINT64_T, 0, comm);
    nx = static_cast<size_t>(nx64);
    ny = static_cast<size_t>(ny64);
    nz = static_cast<size_t>(nz64);

    MPI_Bcast(&iterations, 1, MPI_INT, 0, comm);
    MPI_Bcast(&validate, 1, MPI_INT, 0, comm);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, comm);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", size);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t zStart = 0;
    size_t localNz = 0;
    computeZDecomp(nz, rank, size, zStart, localNz);

    const size_t plane = nx * ny;
    const size_t localElems = (localNz > 0) ? (localNz + 2) * plane : 0;

    std::vector<Real> grid1(localElems);
    std::vector<Real> grid2(localElems);

    if (rank == 0) {
        printf("Initializing grid...\n");
    }

    if (localNz > 0) {
        initializeLocalGrid(grid1, nx, ny, localNz, zStart);
    }

    // Stencil iterations
    if (rank == 0) {
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    const Real inv7 = static_cast<Real>(1.0 / 7.0);
    const size_t innerX0 = 1;
    const size_t innerX1 = (nx >= 2) ? nx - 1 : 0;
    const size_t innerY0 = 1;
    const size_t innerY1 = (ny >= 2) ? ny - 1 : 0;

    for (int iter = 0; iter < iterations; ++iter) {
        const std::vector<Real>& in = (iter % 2 == 0) ? grid1 : grid2;
        std::vector<Real>& out = (iter % 2 == 0) ? grid2 : grid1;

        if (localNz == 0) {
            continue;
        }

        // Copy whole planes first (preserves boundary semantics); compute overwrites interior.
        const size_t planeBytes = plane * sizeof(Real);
        for (size_t lz = 1; lz <= localNz; ++lz) {
            std::memcpy(out.data() + lz * plane, in.data() + lz * plane, planeBytes);
        }

        const size_t globalZ0 = zStart;
        const size_t globalZ1 = zStart + localNz; // exclusive

        const int hasPrev = (globalZ0 > 0) ? 1 : 0;
        const int hasNext = (globalZ1 < nz) ? 1 : 0;

        const int prevRank = hasPrev ? (rank - 1) : MPI_PROC_NULL;
        const int nextRank = hasNext ? (rank + 1) : MPI_PROC_NULL;

        MPI_Request reqs[4];
        int nreq = 0;

        // Receive halos into in's ghost planes.
        if (prevRank != MPI_PROC_NULL) {
            MPI_Irecv(const_cast<Real*>(in.data()) + 0 * plane, static_cast<int>(plane), MPI_DOUBLE, prevRank, 101, comm, &reqs[nreq++]);
        }
        if (nextRank != MPI_PROC_NULL) {
            MPI_Irecv(const_cast<Real*>(in.data()) + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, nextRank, 100, comm, &reqs[nreq++]);
        }
        // Send boundary planes.
        if (prevRank != MPI_PROC_NULL) {
            MPI_Isend(const_cast<Real*>(in.data()) + 1 * plane, static_cast<int>(plane), MPI_DOUBLE, prevRank, 100, comm, &reqs[nreq++]);
        }
        if (nextRank != MPI_PROC_NULL) {
            MPI_Isend(const_cast<Real*>(in.data()) + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, nextRank, 101, comm, &reqs[nreq++]);
        }

        // Compute planes that don't depend on halos.
        if (localNz >= 3 && nx >= 3 && ny >= 3 && nz >= 3) {
            for (size_t lz = 2; lz <= localNz - 1; ++lz) {
                const size_t gz = zStart + (lz - 1);
                if (gz == 0 || gz + 1 >= nz) continue;

                const Real* inZ = in.data() + lz * plane;
                Real* outZ = out.data() + lz * plane;

                const Real* inZm = inZ - plane;
                const Real* inZp = inZ + plane;

                for (size_t y = innerY0; y < innerY1; ++y) {
                    const size_t off = y * nx;
                    for (size_t x = innerX0; x < innerX1; ++x) {
                        const size_t i = off + x;
                        outZ[i] = (inZ[i] + inZ[i - 1] + inZ[i + 1] + inZ[i - nx] + inZ[i + nx] + inZm[i] + inZp[i]) * inv7;
                    }
                }
            }
        }

        if (nreq) {
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        }

        // Compute halo-adjacent planes (if they are not global boundary planes).
        if (nx >= 3 && ny >= 3 && nz >= 3) {
            // First local plane
            {
                const size_t lz = 1;
                const size_t gz = zStart;
                if (gz != 0 && gz + 1 < nz) {
                    const Real* inZ = in.data() + lz * plane;
                    Real* outZ = out.data() + lz * plane;
                    const Real* inZm = inZ - plane; // halo or valid local
                    const Real* inZp = inZ + plane;

                    for (size_t y = innerY0; y < innerY1; ++y) {
                        const size_t off = y * nx;
                        for (size_t x = innerX0; x < innerX1; ++x) {
                            const size_t i = off + x;
                            outZ[i] = (inZ[i] + inZ[i - 1] + inZ[i + 1] + inZ[i - nx] + inZ[i + nx] + inZm[i] + inZp[i]) * inv7;
                        }
                    }
                }
            }

            // Last local plane
            if (localNz >= 2) {
                const size_t lz = localNz;
                const size_t gz = zStart + (lz - 1);
                if (gz != 0 && gz + 1 < nz) {
                    const Real* inZ = in.data() + lz * plane;
                    Real* outZ = out.data() + lz * plane;
                    const Real* inZm = inZ - plane;
                    const Real* inZp = inZ + plane; // halo or valid local

                    for (size_t y = innerY0; y < innerY1; ++y) {
                        const size_t off = y * nx;
                        for (size_t x = innerX0; x < innerX1; ++x) {
                            const size_t i = off + x;
                            outZ[i] = (inZ[i] + inZ[i - 1] + inZ[i + 1] + inZ[i - nx] + inZ[i + nx] + inZm[i] + inZp[i]) * inv7;
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(comm);
    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double cellUpdates = (nx >= 2 && ny >= 2 && nz >= 2)
                                       ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * static_cast<double>(iterations)
                                       : 0.0;
        const double mcups = (maxTime > 0.0) ? (cellUpdates / maxTime / 1e6) : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? grid1 : grid2;

    if (printResults) {
        std::vector<Real> globalFinal;
        if (rank == 0) {
            globalFinal.resize(nx * ny * nz);
        }

        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            size_t zs = 0, lnz = 0;
            computeZDecomp(nz, r, size, zs, lnz);
            const size_t cnt = lnz * plane;
            counts[r] = static_cast<int>(cnt);
            displs[r] = static_cast<int>(zs * plane);
        }

        const Real* sendPtr = (localNz > 0) ? (localFinal.data() + 1 * plane) : nullptr;
        MPI_Gatherv(sendPtr,
                    static_cast<int>(localNz * plane),
                    MPI_DOUBLE,
                    (rank == 0) ? globalFinal.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    comm);

        if (rank == 0) {
            print_results(globalFinal, "Grid");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        const bool ok = validateDistributed(localFinal, nx, ny, nz, localNz, zStart, comm, rank);

        if (rank == 0) {
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }

        MPI_Finalize();
        return ok ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
