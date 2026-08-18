#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Slab {
    size_t localNz;
    size_t zOffset;
    int lowerRank;
    int upperRank;
};

Slab decomposeZ(const size_t nz, const int rank, const int ranks) {
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t zOffset = static_cast<size_t>(rank) * base +
                           std::min(static_cast<size_t>(rank), remainder);
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));

    return {localNz,
            zOffset,
            localNz != 0 && rank > 0 ? rank - 1 : MPI_PROC_NULL,
            localNz != 0 && rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL};
}

inline size_t index3(const size_t x, const size_t y, const size_t z,
                     const size_t nx, const size_t planeSize) noexcept {
    return z * planeSize + y * nx + x;
}

// Begin a halo exchange. Physical z boundaries are filled by copying the
// adjacent plane, which exactly implements the original clamped boundary.
void beginHaloExchange(std::vector<double>& field, const Slab& slab,
                       const size_t planeSize, MPI_Comm comm,
                       MPI_Request requests[4]) {
    int count = 0;
    if (slab.localNz == 0) {
        requests[0] = MPI_REQUEST_NULL;
        requests[1] = MPI_REQUEST_NULL;
        requests[2] = MPI_REQUEST_NULL;
        requests[3] = MPI_REQUEST_NULL;
        return;
    }

    if (slab.lowerRank == MPI_PROC_NULL) {
        std::copy_n(field.data() + planeSize, planeSize, field.data());
    } else {
        MPI_Irecv(field.data(), static_cast<int>(planeSize), MPI_DOUBLE,
                  slab.lowerRank, 101, comm, &requests[count++]);
        MPI_Isend(field.data() + planeSize, static_cast<int>(planeSize), MPI_DOUBLE,
                  slab.lowerRank, 102, comm, &requests[count++]);
    }

    if (slab.upperRank == MPI_PROC_NULL) {
        std::copy_n(field.data() + slab.localNz * planeSize, planeSize,
                    field.data() + (slab.localNz + 1) * planeSize);
    } else {
        MPI_Irecv(field.data() + (slab.localNz + 1) * planeSize,
                  static_cast<int>(planeSize), MPI_DOUBLE,
                  slab.upperRank, 102, comm, &requests[count++]);
        MPI_Isend(field.data() + slab.localNz * planeSize,
                  static_cast<int>(planeSize), MPI_DOUBLE,
                  slab.upperRank, 101, comm, &requests[count++]);
    }

    while (count < 4) requests[count++] = MPI_REQUEST_NULL;
}

inline double laplacianAt(const double* const field, const size_t x,
                          const size_t y, const size_t z, const size_t nx,
                          const size_t ny, const size_t planeSize,
                          const double invDx2, const double invDy2,
                          const double invDz2) noexcept {
    const size_t center = index3(x, y, z, nx, planeSize);
    const size_t left = x == 0 ? center : center - 1;
    const size_t right = x + 1 == nx ? center : center + 1;
    const size_t down = y == 0 ? center : center - nx;
    const size_t up = y + 1 == ny ? center : center + nx;

    const double cxx = (field[right] + field[left] - 2.0 * field[center]) * invDx2;
    const double cyy = (field[up] + field[down] - 2.0 * field[center]) * invDy2;
    const double czz = (field[center + planeSize] + field[center - planeSize] -
                        2.0 * field[center]) * invDz2;
    return cxx + cyy + czz;
}

void computeChemicalPotentialRange(const std::vector<double>& c,
                                   std::vector<double>& mu,
                                   const size_t nx, const size_t ny,
                                   const size_t planeSize,
                                   const size_t zBegin, const size_t zEnd,
                                   const double invDx2, const double invDy2,
                                   const double invDz2, const double gamma,
                                   const double eAA, const double eBB,
                                   const double eAB) {
    const double* const cData = c.data();
    double* const muData = mu.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = index3(x, y, z, nx, planeSize);
                const double cv = cData[index];
                muData[index] =
                    4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                    3.0 * cv + cv * cv * cv -
                    gamma * laplacianAt(cData, x, y, z, nx, ny, planeSize,
                                        invDx2, invDy2, invDz2);
            }
        }
    }
}

void updateRange(std::vector<double>& cnew, const std::vector<double>& cold,
                 const std::vector<double>& mu, const size_t nx,
                 const size_t ny, const size_t planeSize,
                 const size_t zBegin, const size_t zEnd, const double dtD,
                 const double invDx2, const double invDy2,
                 const double invDz2) {
    double* const newData = cnew.data();
    const double* const oldData = cold.data();
    const double* const muData = mu.data();
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = index3(x, y, z, nx, planeSize);
                newData[index] = oldData[index] +
                    dtD * laplacianAt(muData, x, y, z, nx, ny, planeSize,
                                      invDx2, invDy2, invDz2);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx,
                             const size_t ny, const size_t nz,
                             const Slab& slab, const size_t planeSize) {
    const size_t volume = nx * ny * nz;
    for (size_t localZ = 1; localZ <= slab.localNz; ++localZ) {
        const size_t globalZ = slab.zOffset + localZ - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = globalZ * planeSize + y * nx + x;
                const double pseudo =
                    (((linearId + 1) * 1299709) % volume) /
                    static_cast<double>(volume);
                c[index3(x, y, localZ, nx, planeSize)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const Slab& slab,
                    const size_t planeSize, MPI_Comm comm, const int rank) {
    bool localFinite = true;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    const size_t begin = planeSize;
    const size_t end = (slab.localNz + 1) * planeSize;
    for (size_t i = begin; i < end; ++i) {
        localFinite = localFinite && std::isfinite(c[i]);
        localMin = std::min(localMin, c[i]);
        localMax = std::max(localMax, c[i]);
    }

    int finite = localFinite ? 1 : 0;
    int globalFinite = 0;
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&finite, &globalFinite, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    int valid = 1;
    if (rank == 0) {
        if (!globalFinite) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        }
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 10.0 || globalMin < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    return valid != 0;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    const bool dimensionsValid = nx > 0 && ny > 0 && nz > 0 && iterations >= 0 &&
        nx <= std::numeric_limits<size_t>::max() / ny &&
        nx * ny <= std::numeric_limits<size_t>::max() / nz &&
        nx * ny <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (!dimensionsValid) {
        if (rank == 0) {
            std::fprintf(stderr, "Grid dimensions and iteration count must be positive and fit MPI message limits.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t planeSize = nx * ny;
    const size_t gridSize = planeSize * nz;
    const Slab slab = decomposeZ(nz, rank, ranks);

    if (rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI processes: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    // One ghost plane on each side. Empty ranks allocate no field storage.
    const size_t localAllocation = slab.localNz == 0 ? 0 : (slab.localNz + 2) * planeSize;
    std::vector<double> cold(localAllocation);
    std::vector<double> cnew(localAllocation);
    std::vector<double> mu(localAllocation);
    initializeConcentration(cold, nx, ny, nz, slab, planeSize);

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;
    constexpr double invDx2 = 1.0 / (dx * dx);
    constexpr double invDy2 = 1.0 / (dy * dy);
    constexpr double invDz2 = 1.0 / (dz * dz);

    if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int step = 0; step < iterations; ++step) {
        MPI_Request requests[4];
        beginHaloExchange(cold, slab, planeSize, comm, requests);

        // Planes that do not require a ghost can be computed in flight.
        if (slab.localNz > 2) {
            computeChemicalPotentialRange(cold, mu, nx, ny, planeSize, 2,
                                          slab.localNz, invDx2, invDy2, invDz2,
                                          gamma, eAA, eBB, eAB);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (slab.localNz != 0) {
            computeChemicalPotentialRange(cold, mu, nx, ny, planeSize, 1, 2,
                                          invDx2, invDy2, invDz2, gamma,
                                          eAA, eBB, eAB);
            if (slab.localNz > 1) {
                computeChemicalPotentialRange(cold, mu, nx, ny, planeSize,
                                              slab.localNz, slab.localNz + 1,
                                              invDx2, invDy2, invDz2, gamma,
                                              eAA, eBB, eAB);
            }
        }

        beginHaloExchange(mu, slab, planeSize, comm, requests);
        if (slab.localNz > 2) {
            updateRange(cnew, cold, mu, nx, ny, planeSize, 2, slab.localNz,
                        dt * diffusion, invDx2, invDy2, invDz2);
        }
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        if (slab.localNz != 0) {
            updateRange(cnew, cold, mu, nx, ny, planeSize, 1, 2,
                        dt * diffusion, invDx2, invDy2, invDz2);
            if (slab.localNz > 1) {
                updateRange(cnew, cold, mu, nx, ny, planeSize,
                            slab.localNz, slab.localNz + 1, dt * diffusion,
                            invDx2, invDy2, invDz2);
            }
        }
        cold.swap(cnew);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %ld ms\n", milliseconds);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        bool gatherSupported = gridSize <= static_cast<size_t>(std::numeric_limits<int>::max());
        int gatherOk = gatherSupported ? 1 : 0;
        MPI_Bcast(&gatherOk, 1, MPI_INT, 0, comm);
        if (!gatherOk) {
            if (rank == 0) std::fprintf(stderr, "Result output exceeds MPI_Gatherv limits.\n");
            MPI_Finalize();
            return 1;
        }

        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const Slab other = decomposeZ(nz, r, ranks);
                counts[r] = static_cast<int>(other.localNz * planeSize);
                displacements[r] = static_cast<int>(other.zOffset * planeSize);
            }
            global.resize(gridSize);
        }
        const int localCount = static_cast<int>(slab.localNz * planeSize);
        const double* sendBuffer = slab.localNz == 0 ? nullptr : cold.data() + planeSize;
        MPI_Gatherv(sendBuffer, localCount, MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(global, "Concentration");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        valid = validateResult(cold, slab, planeSize, comm, rank);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
