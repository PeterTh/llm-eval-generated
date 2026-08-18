#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

namespace {

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

// A Cartesian rank owns [x0, x0 + nx) x [y0, y0 + ny) x [z0, z0 + nz).
// Every local array has a one-cell halo on all sides.
struct Domain {
    size_t globalNx = 0;
    size_t globalNy = 0;
    size_t globalNz = 0;
    size_t x0 = 0;
    size_t y0 = 0;
    size_t z0 = 0;
    size_t nx = 0;
    size_t ny = 0;
    size_t nz = 0;
    size_t pitchX = 0;
    size_t pitchY = 0;
    MPI_Comm comm = MPI_COMM_NULL;
    int xMinus = MPI_PROC_NULL;
    int xPlus = MPI_PROC_NULL;
    int yMinus = MPI_PROC_NULL;
    int yPlus = MPI_PROC_NULL;
    int zMinus = MPI_PROC_NULL;
    int zPlus = MPI_PROC_NULL;
    MPI_Datatype xFace = MPI_DATATYPE_NULL;
    MPI_Datatype yFace = MPI_DATATYPE_NULL;
    MPI_Datatype zFace = MPI_DATATYPE_NULL;

    [[nodiscard]] size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * pitchY + y * pitchX + x;
    }

    [[nodiscard]] size_t storageSize() const noexcept {
        return (nz + 2) * pitchY;
    }
};

struct ProcessGrid {
    int size = 1;
    // MPI Cartesian dimension order is z, y, x so the last dimension is contiguous in memory.
    std::array<int, 3> dims{1, 1, 1};
};

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

bool parseOptions(const int argc, char** argv, Options& options, bool& showHelp) {
    showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.nx = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.ny = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.nz = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
            return true;
        } else {
            return false;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return options.nx >= 2 && options.ny >= 2 && options.nz >= 2 && options.iterations >= 0;
}

std::vector<int> divisors(const int value) {
    std::vector<int> result;
    for (int divisor = 1; divisor <= value / divisor; ++divisor) {
        if (value % divisor != 0) continue;
        result.push_back(divisor);
        if (divisor != value / divisor) result.push_back(value / divisor);
    }
    return result;
}

// Pick the largest usable rectangular process grid.  Reducing the active
// communicator only matters for unusually small grids or prime rank counts
// that cannot be represented without empty Cartesian blocks.
ProcessGrid chooseProcessGrid(const int worldSize, const size_t nx, const size_t ny, const size_t nz) {
    const size_t cellCount = nx * ny * nz;
    const int requested = static_cast<int>(std::min(cellCount, static_cast<size_t>(worldSize)));

    for (int processCount = requested; processCount >= 1; --processCount) {
        const std::vector<int> zDivisors = divisors(processCount);
        double bestSurface = std::numeric_limits<double>::infinity();
        ProcessGrid best{};
        bool found = false;

        for (const int zParts : zDivisors) {
            if (static_cast<size_t>(zParts) > nz) continue;
            const int remaining = processCount / zParts;
            for (const int yParts : divisors(remaining)) {
                const int xParts = remaining / yParts;
                if (static_cast<size_t>(yParts) > ny || static_cast<size_t>(xParts) > nx) continue;

                const double localX = static_cast<double>(nx) / xParts;
                const double localY = static_cast<double>(ny) / yParts;
                const double localZ = static_cast<double>(nz) / zParts;
                const double surface = localX * localY + localX * localZ + localY * localZ;
                if (surface < bestSurface) {
                    bestSurface = surface;
                    best.size = processCount;
                    best.dims = {zParts, yParts, xParts};
                    found = true;
                }
            }
        }
        if (found) return best;
    }

    return {};
}

void decomposeDimension(const size_t globalSize, const int parts, const int coordinate,
                        size_t& offset, size_t& localSize) {
    const size_t base = globalSize / static_cast<size_t>(parts);
    const size_t remainder = globalSize % static_cast<size_t>(parts);
    localSize = base + (coordinate < static_cast<int>(remainder) ? 1 : 0);
    offset = static_cast<size_t>(coordinate) * base +
             std::min(static_cast<size_t>(coordinate), remainder);
}

bool fitsMpiCount(const Domain& domain) {
    const size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    return domain.nx <= maxCount && domain.ny <= maxCount && domain.nz <= maxCount &&
           domain.pitchX <= maxCount && domain.pitchY <= maxCount;
}

bool createFaceTypes(Domain& domain) {
    if (!fitsMpiCount(domain)) return false;

    const int localX = static_cast<int>(domain.nx);
    const int localY = static_cast<int>(domain.ny);
    const int localZ = static_cast<int>(domain.nz);
    const int pitchX = static_cast<int>(domain.pitchX);
    const int pitchY = static_cast<int>(domain.pitchY);

    MPI_Datatype xRow = MPI_DATATYPE_NULL;
    MPI_Type_vector(localY, 1, pitchX, MPI_DOUBLE, &xRow);
    MPI_Type_commit(&xRow);
    MPI_Type_create_hvector(localZ, 1, static_cast<MPI_Aint>(pitchY * sizeof(Real)), xRow, &domain.xFace);
    MPI_Type_commit(&domain.xFace);
    MPI_Type_free(&xRow);

    MPI_Type_vector(localZ, localX, pitchY, MPI_DOUBLE, &domain.yFace);
    MPI_Type_commit(&domain.yFace);

    MPI_Type_vector(localY, localX, pitchX, MPI_DOUBLE, &domain.zFace);
    MPI_Type_commit(&domain.zFace);
    return true;
}

void destroyFaceTypes(Domain& domain) {
    if (domain.xFace != MPI_DATATYPE_NULL) MPI_Type_free(&domain.xFace);
    if (domain.yFace != MPI_DATATYPE_NULL) MPI_Type_free(&domain.yFace);
    if (domain.zFace != MPI_DATATYPE_NULL) MPI_Type_free(&domain.zFace);
}

void initializeGrid(std::vector<Real>& grid, const Domain& domain) {
    for (size_t localZ = 1; localZ <= domain.nz; ++localZ) {
        const size_t globalZ = domain.z0 + localZ - 1;
        for (size_t localY = 1; localY <= domain.ny; ++localY) {
            const size_t globalY = domain.y0 + localY - 1;
            const size_t globalBase = (globalZ * domain.globalNy + globalY) * domain.globalNx + domain.x0;
            const size_t localBase = domain.index(1, localY, localZ);
            for (size_t localX = 0; localX < domain.nx; ++localX) {
                grid[localBase + localX] = static_cast<Real>((globalBase + localX) % 19);
            }
        }
    }
}

// Tags distinguish the direction of each face.  A rank receives its minus
// ghost layer using the plus tag sent by its minus neighbour, and vice versa.
int beginHaloExchange(const Domain& domain, std::vector<Real>& grid,
                      std::array<MPI_Request, 12>& requests) {
    int requestCount = 0;
    const auto postFace = [&](const int minusRank, const int plusRank, const MPI_Datatype face,
                              const size_t receiveMinus, const size_t sendMinus,
                              const size_t receivePlus, const size_t sendPlus, const int minusTag) {
        const int plusTag = minusTag + 1;
        if (minusRank != MPI_PROC_NULL) {
            MPI_Irecv(grid.data() + receiveMinus, 1, face, minusRank, plusTag, domain.comm,
                      &requests[requestCount++]);
            MPI_Isend(grid.data() + sendMinus, 1, face, minusRank, minusTag, domain.comm,
                      &requests[requestCount++]);
        }
        if (plusRank != MPI_PROC_NULL) {
            MPI_Irecv(grid.data() + receivePlus, 1, face, plusRank, minusTag, domain.comm,
                      &requests[requestCount++]);
            MPI_Isend(grid.data() + sendPlus, 1, face, plusRank, plusTag, domain.comm,
                      &requests[requestCount++]);
        }
    };

    // In the x direction adjacent face bases differ by exactly one element.
    postFace(domain.xMinus, domain.xPlus, domain.xFace,
             domain.index(0, 1, 1), domain.index(1, 1, 1),
             domain.index(domain.nx + 1, 1, 1), domain.index(domain.nx, 1, 1), 100);
    // In the y direction they differ by one full row.
    postFace(domain.yMinus, domain.yPlus, domain.yFace,
             domain.index(1, 0, 1), domain.index(1, 1, 1),
             domain.index(1, domain.ny + 1, 1), domain.index(1, domain.ny, 1), 110);
    // In the z direction they differ by one local plane.
    postFace(domain.zMinus, domain.zPlus, domain.zFace,
             domain.index(1, 1, 0), domain.index(1, 1, 1),
             domain.index(1, 1, domain.nz + 1), domain.index(1, 1, domain.nz), 120);
    return requestCount;
}

void computeDeepInterior(const std::vector<Real>& input, std::vector<Real>& output, const Domain& domain) {
    if (domain.nx <= 2 || domain.ny <= 2 || domain.nz <= 2) return;

    const Real* const in = input.data();
    Real* const out = output.data();
    const size_t xStride = domain.pitchX;
    const size_t zStride = domain.pitchY;
    for (size_t z = 2; z < domain.nz; ++z) {
        for (size_t y = 2; y < domain.ny; ++y) {
            size_t index = domain.index(2, y, z);
            const size_t end = domain.index(domain.nx, y, z);
            for (; index < end; ++index) {
                out[index] = (in[index] + in[index - 1] + in[index + 1] +
                              in[index - xStride] + in[index + xStride] +
                              in[index - zStride] + in[index + zStride]) / 7.0;
            }
        }
    }
}

void computeHaloShell(const std::vector<Real>& input, std::vector<Real>& output, const Domain& domain) {
    const Real* const in = input.data();
    Real* const out = output.data();
    const size_t xStride = domain.pitchX;
    const size_t zStride = domain.pitchY;

    const auto globalYBoundary = [&](const size_t y) {
        return (y == 1 && domain.y0 == 0) ||
               (y == domain.ny && domain.y0 + domain.ny == domain.globalNy);
    };
    const auto globalZBoundary = [&](const size_t z) {
        return (z == 1 && domain.z0 == 0) ||
               (z == domain.nz && domain.z0 + domain.nz == domain.globalNz);
    };
    const auto update = [&](const size_t x, const size_t y, const size_t z, const bool isGlobalBoundary) {
        const size_t index = domain.index(x, y, z);
        if (isGlobalBoundary) {
            out[index] = in[index];
        } else {
            out[index] = (in[index] + in[index - 1] + in[index + 1] +
                          in[index - xStride] + in[index + xStride] +
                          in[index - zStride] + in[index + zStride]) / 7.0;
        }
    };

    // Cover each of the six local faces exactly once.  This avoids a second
    // volume-sized pass after the deep interior was computed during communication.
    const auto updateXFace = [&](const size_t x, const bool globalXBoundary) {
        for (size_t z = 1; z <= domain.nz; ++z) {
            const bool zBoundary = globalZBoundary(z);
            for (size_t y = 1; y <= domain.ny; ++y) {
                update(x, y, z, globalXBoundary || globalYBoundary(y) || zBoundary);
            }
        }
    };
    updateXFace(1, domain.x0 == 0 ||
                       (domain.nx == 1 && domain.x0 + domain.nx == domain.globalNx));
    if (domain.nx > 1) {
        updateXFace(domain.nx, domain.x0 + domain.nx == domain.globalNx);
    }

    const auto updateYFace = [&](const size_t y, const bool yBoundary) {
        for (size_t z = 1; z <= domain.nz; ++z) {
            const bool zBoundary = globalZBoundary(z);
            for (size_t x = 2; x < domain.nx; ++x) {
                update(x, y, z, yBoundary || zBoundary);
            }
        }
    };
    updateYFace(1, domain.y0 == 0 ||
                       (domain.ny == 1 && domain.y0 + domain.ny == domain.globalNy));
    if (domain.ny > 1) {
        updateYFace(domain.ny, domain.y0 + domain.ny == domain.globalNy);
    }

    const auto updateZFace = [&](const size_t z, const bool zBoundary) {
        for (size_t y = 2; y < domain.ny; ++y) {
            for (size_t x = 2; x < domain.nx; ++x) {
                update(x, y, z, zBoundary);
            }
        }
    };
    updateZFace(1, domain.z0 == 0 ||
                       (domain.nz == 1 && domain.z0 + domain.nz == domain.globalNz));
    if (domain.nz > 1) {
        updateZFace(domain.nz, domain.z0 + domain.nz == domain.globalNz);
    }
}

void stencilIteration(std::vector<Real>& input, std::vector<Real>& output, const Domain& domain) {
    std::array<MPI_Request, 12> requests{};
    const int requestCount = beginHaloExchange(domain, input, requests);
    computeDeepInterior(input, output, domain);
    MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
    computeHaloShell(input, output, domain);
}

bool validateResult(const std::vector<Real>& grid, const Domain& domain, const int rank) {
    int localInvalid = 0;
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t base = domain.index(1, y, z);
            for (size_t x = 0; x < domain.nx; ++x) {
                const Real value = grid[base + x];
                localInvalid |= !std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int invalid = 0;
    MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_LOR, domain.comm);
    if (invalid != 0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, domain.comm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);

    int valid = 1;
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, domain.comm);
    return valid != 0;
}

std::vector<Real> gatherGlobalGrid(const std::vector<Real>& grid, const Domain& domain, const int rank,
                                   const int processCount) {
    const size_t localCountSize = domain.nx * domain.ny * domain.nz;
    const int localCount = static_cast<int>(localCountSize);
    std::vector<Real> packed(localCountSize);
    size_t packedOffset = 0;
    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const Real* const source = grid.data() + domain.index(1, y, z);
            std::copy_n(source, domain.nx, packed.data() + packedOffset);
            packedOffset += domain.nx;
        }
    }

    std::array<unsigned long long, 6> localMetadata{
        static_cast<unsigned long long>(domain.x0), static_cast<unsigned long long>(domain.y0),
        static_cast<unsigned long long>(domain.z0), static_cast<unsigned long long>(domain.nx),
        static_cast<unsigned long long>(domain.ny), static_cast<unsigned long long>(domain.nz)};
    std::vector<unsigned long long> metadata;
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<Real> gathered;
    if (rank == 0) {
        metadata.resize(static_cast<size_t>(processCount) * localMetadata.size());
        counts.resize(processCount);
        displacements.resize(processCount);
    }

    MPI_Gather(localMetadata.data(), static_cast<int>(localMetadata.size()), MPI_UNSIGNED_LONG_LONG,
               metadata.data(), static_cast<int>(localMetadata.size()), MPI_UNSIGNED_LONG_LONG, 0, domain.comm);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, domain.comm);

    if (rank == 0) {
        int totalCount = 0;
        for (int process = 0; process < processCount; ++process) {
            displacements[process] = totalCount;
            totalCount += counts[process];
        }
        gathered.resize(domain.globalNx * domain.globalNy * domain.globalNz);
    }

    std::vector<Real> rankOrderedData;
    if (rank == 0) {
        const size_t totalCount = static_cast<size_t>(displacements.back()) + static_cast<size_t>(counts.back());
        rankOrderedData.resize(totalCount);
    }
    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, rankOrderedData.data(), counts.data(), displacements.data(),
                MPI_DOUBLE, 0, domain.comm);

    if (rank == 0) {
        for (int process = 0; process < processCount; ++process) {
            const auto* const block = metadata.data() + static_cast<size_t>(process) * localMetadata.size();
            const size_t x0 = static_cast<size_t>(block[0]);
            const size_t y0 = static_cast<size_t>(block[1]);
            const size_t z0 = static_cast<size_t>(block[2]);
            const size_t blockNx = static_cast<size_t>(block[3]);
            const size_t blockNy = static_cast<size_t>(block[4]);
            const size_t blockNz = static_cast<size_t>(block[5]);
            size_t sourceOffset = static_cast<size_t>(displacements[process]);
            for (size_t z = 0; z < blockNz; ++z) {
                for (size_t y = 0; y < blockNy; ++y) {
                    const size_t destination = ((z0 + z) * domain.globalNy + (y0 + y)) * domain.globalNx + x0;
                    std::copy_n(rankOrderedData.data() + sourceOffset, blockNx, gathered.data() + destination);
                    sourceOffset += blockNx;
                }
            }
        }
    }
    return gathered;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    Options options;
    bool showHelp = false;
    if (!parseOptions(argc, argv, options, showHelp)) {
        if (worldRank == 0) {
            printf("Invalid option or grid size\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if (showHelp) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    const ProcessGrid processGrid = chooseProcessGrid(worldSize, options.nx, options.ny, options.nz);
    MPI_Comm activeComm = MPI_COMM_NULL;
    const int color = worldRank < processGrid.size ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, worldRank, &activeComm);
    if (activeComm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(activeComm, &rank);
    MPI_Comm_size(activeComm, &processCount);

    int dimensions[3] = {processGrid.dims[0], processGrid.dims[1], processGrid.dims[2]};
    int periods[3] = {0, 0, 0};
    MPI_Comm cartComm = MPI_COMM_NULL;
    MPI_Cart_create(activeComm, 3, dimensions, periods, 0, &cartComm);
    MPI_Comm_free(&activeComm);

    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(cartComm, rank, 3, coordinates);
    Domain domain;
    domain.globalNx = options.nx;
    domain.globalNy = options.ny;
    domain.globalNz = options.nz;
    domain.comm = cartComm;
    decomposeDimension(options.nz, dimensions[0], coordinates[0], domain.z0, domain.nz);
    decomposeDimension(options.ny, dimensions[1], coordinates[1], domain.y0, domain.ny);
    decomposeDimension(options.nx, dimensions[2], coordinates[2], domain.x0, domain.nx);
    domain.pitchX = domain.nx + 2;
    domain.pitchY = domain.pitchX * (domain.ny + 2);
    MPI_Cart_shift(cartComm, 2, 1, &domain.xMinus, &domain.xPlus);
    MPI_Cart_shift(cartComm, 1, 1, &domain.yMinus, &domain.yPlus);
    MPI_Cart_shift(cartComm, 0, 1, &domain.zMinus, &domain.zPlus);

    int localCanRun = createFaceTypes(domain) ? 1 : 0;
    int canRun = 0;
    MPI_Allreduce(&localCanRun, &canRun, 1, MPI_INT, MPI_LAND, cartComm);
    if (canRun == 0) {
        if (rank == 0) printf("Grid decomposition exceeds MPI count limits\n");
        destroyFaceTypes(domain);
        MPI_Comm_free(&cartComm);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        printf("Iterations: %d\n", options.iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", processCount);
        printf("Process grid: %d x %d x %d (Z x Y x X)\n", dimensions[0], dimensions[1], dimensions[2]);
        if (processCount != worldSize) {
            printf("Using %d of %d MPI processes to avoid empty Cartesian blocks\n", processCount, worldSize);
        }
        printf("Initializing grid...\n");
    }

    std::vector<Real> grid1(domain.storageSize());
    std::vector<Real> grid2(domain.storageSize());
    initializeGrid(grid1, domain);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cartComm);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        if ((iteration & 1) == 0) {
            stencilIteration(grid1, grid2, domain);
        } else {
            stencilIteration(grid2, grid1, domain);
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const double localMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    double elapsedMilliseconds = 0.0;
    MPI_Reduce(&localMilliseconds, &elapsedMilliseconds, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
        printf("Computation time: %ld ms\n", durationMilliseconds);
        const double cellUpdates = static_cast<double>((options.nx - 2) * (options.ny - 2) * (options.nz - 2)) *
                                   static_cast<double>(options.iterations);
        const double mcups = cellUpdates / (elapsedMilliseconds / 1000.0) / 1.0e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (options.iterations % 2 == 0) ? grid1 : grid2;
    if (options.printResults) {
        std::vector<Real> globalGrid = gatherGlobalGrid(finalGrid, domain, rank, processCount);
        if (rank == 0) print_results(globalGrid, "Grid");
    }

    int exitCode = 0;
    if (options.validate) {
        if (rank == 0) printf("Validating result...\n");
        if (validateResult(finalGrid, domain, rank)) {
            if (rank == 0) printf("Validation: PASSED\n");
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    destroyFaceTypes(domain);
    MPI_Comm_free(&cartComm);
    MPI_Finalize();
    return exitCode;
}
