#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#if defined(__GNUC__) || defined(__clang__)
#define STENCIL_RESTRICT __restrict__
#else
#define STENCIL_RESTRICT
#endif

struct Options {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

struct Partition {
    size_t begin = 0; // First global grid coordinate owned by this rank.
    size_t count = 0;
};

struct ProcessGrid {
    int activeRanks = 1;
    std::array<int, 3> dims{1, 1, 1}; // x, y, z
};

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseSize(const char* text, size_t& value) {
    if (text[0] == '-') return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Returns 0 on success, 1 on an error, and 2 after printing help.
int parseArguments(const int argc, char** argv, Options& options, const bool report) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nx)) {
                if (report) std::fprintf(stderr, "Invalid X dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.ny)) {
                if (report) std::fprintf(stderr, "Invalid Y dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            if (!parseSize(argv[++i], options.nz)) {
                if (report) std::fprintf(stderr, "Invalid Z dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            if (!parseIterations(argv[++i], options.iterations)) {
                if (report) std::fprintf(stderr, "Invalid iteration count: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (report) printUsage(argv[0]);
            return 2;
        } else {
            if (report) {
                std::fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    return 0;
}

bool checkedProduct(const size_t a, const size_t b, size_t& product) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    product = a * b;
    return true;
}

size_t ceilDivide(const size_t value, const int parts) {
    return value / static_cast<size_t>(parts) +
           (value % static_cast<size_t>(parts) != 0 ? 1 : 0);
}

// Pick the largest usable Cartesian grid, then favor balanced work and small
// halo surfaces. Unlike MPI_Dims_create, this also respects anisotropic grids
// and never creates an empty subdomain.
ProcessGrid chooseProcessGrid(const int worldSize, const size_t ix,
                              const size_t iy, const size_t iz) {
    const size_t interiorCells = ix * iy * iz;
    const int maximumRanks = interiorCells < static_cast<size_t>(worldSize)
                                 ? static_cast<int>(interiorCells)
                                 : worldSize;
    for (int ranks = maximumRanks; ranks >= 1; --ranks) {
        bool found = false;
        std::array<int, 3> best{1, 1, 1};
        size_t bestMaxCells = std::numeric_limits<size_t>::max();
        long double bestSurface = std::numeric_limits<long double>::infinity();
        long double bestTotalSurface = std::numeric_limits<long double>::infinity();

        for (int px = 1; px <= ranks; ++px) {
            if (ranks % px != 0 || static_cast<size_t>(px) > ix) continue;
            const int yz = ranks / px;
            for (int py = 1; py <= yz; ++py) {
                if (yz % py != 0 || static_cast<size_t>(py) > iy) continue;
                const int pz = yz / py;
                if (static_cast<size_t>(pz) > iz) continue;

                const size_t maxX = ceilDivide(ix, px);
                const size_t maxY = ceilDivide(iy, py);
                const size_t maxZ = ceilDivide(iz, pz);
                const size_t maxCells = maxX * maxY * maxZ;
                const long double surface =
                    (px > 1 ? 2.0L * maxY * maxZ : 0.0L) +
                    (py > 1 ? 2.0L * maxX * maxZ : 0.0L) +
                    (pz > 1 ? 2.0L * maxX * maxY : 0.0L);
                const long double totalSurface =
                    static_cast<long double>(px - 1) * iy * iz +
                    static_cast<long double>(py - 1) * ix * iz +
                    static_cast<long double>(pz - 1) * ix * iy;

                if (!found || maxCells < bestMaxCells ||
                    (maxCells == bestMaxCells && surface < bestSurface) ||
                    (maxCells == bestMaxCells && surface == bestSurface &&
                     totalSurface < bestTotalSurface)) {
                    found = true;
                    best = {px, py, pz};
                    bestMaxCells = maxCells;
                    bestSurface = surface;
                    bestTotalSurface = totalSurface;
                }
            }
        }
        if (found) return {ranks, best};
    }
    return {};
}

Partition makePartition(const size_t interiorExtent, const int coordinate,
                        const int processes) {
    const size_t base = interiorExtent / static_cast<size_t>(processes);
    const size_t remainder = interiorExtent % static_cast<size_t>(processes);
    const size_t coord = static_cast<size_t>(coordinate);
    const size_t count = base + (coord < remainder ? 1 : 0);
    const size_t offset = coord * base + std::min(coord, remainder);
    return {offset + 1, count};
}

void initializeGlobalGrid(std::vector<Real>& grid) {
    for (size_t i = 0; i < grid.size(); ++i) grid[i] = static_cast<Real>(i % 19);
}

void initializeLocalGrid(std::vector<Real>& grid, const Partition& xPart,
                         const Partition& yPart, const Partition& zPart,
                         const size_t nx, const size_t ny) {
    const size_t sx = xPart.count + 2;
    const size_t sy = yPart.count + 2;
    const size_t sz = zPart.count + 2;
    const size_t globalPlane = nx * ny;
    for (size_t z = 0; z < sz; ++z) {
        const size_t gz = zPart.begin + z - 1;
        for (size_t y = 0; y < sy; ++y) {
            const size_t gy = yPart.begin + y - 1;
            for (size_t x = 0; x < sx; ++x) {
                const size_t gx = xPart.begin + x - 1;
                const size_t globalIndex = gz * globalPlane + gy * nx + gx;
                grid[idx3(x, y, z, sx, sy)] = static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

class FaceTypes {
  public:
    FaceTypes(const size_t lx, const size_t ly, const size_t lz) {
        const std::array<int, 3> sizes{static_cast<int>(lz + 2),
                                       static_cast<int>(ly + 2),
                                       static_cast<int>(lx + 2)};
        const std::array<int, 3> starts{0, 0, 0};
        const std::array<int, 3> xSub{static_cast<int>(lz), static_cast<int>(ly), 1};
        const std::array<int, 3> ySub{static_cast<int>(lz), 1, static_cast<int>(lx)};
        const std::array<int, 3> zSub{1, static_cast<int>(ly), static_cast<int>(lx)};
        MPI_Type_create_subarray(3, sizes.data(), xSub.data(), starts.data(),
                                 MPI_ORDER_C, MPI_DOUBLE, &xFace);
        MPI_Type_create_subarray(3, sizes.data(), ySub.data(), starts.data(),
                                 MPI_ORDER_C, MPI_DOUBLE, &yFace);
        MPI_Type_create_subarray(3, sizes.data(), zSub.data(), starts.data(),
                                 MPI_ORDER_C, MPI_DOUBLE, &zFace);
        MPI_Type_commit(&xFace);
        MPI_Type_commit(&yFace);
        MPI_Type_commit(&zFace);
    }

    FaceTypes(const FaceTypes&) = delete;
    FaceTypes& operator=(const FaceTypes&) = delete;

    ~FaceTypes() {
        if (xFace != MPI_DATATYPE_NULL) MPI_Type_free(&xFace);
        if (yFace != MPI_DATATYPE_NULL) MPI_Type_free(&yFace);
        if (zFace != MPI_DATATYPE_NULL) MPI_Type_free(&zFace);
    }

    MPI_Datatype xFace = MPI_DATATYPE_NULL;
    MPI_Datatype yFace = MPI_DATATYPE_NULL;
    MPI_Datatype zFace = MPI_DATATYPE_NULL;
};

class PersistentHaloExchange {
  public:
    PersistentHaloExchange(Real* data, const size_t lx, const size_t ly,
                           const size_t lz, const FaceTypes& faces,
                           const std::array<int, 6>& neighbors, MPI_Comm comm) {
        const size_t sx = lx + 2;
        const size_t sy = ly + 2;
        const auto at = [sx, sy](const size_t x, const size_t y, const size_t z) {
            return idx3(x, y, z, sx, sy);
        };
        const auto receive = [&](Real* buffer, const MPI_Datatype type,
                                 const int neighbor, const int tag) {
            if (neighbor != MPI_PROC_NULL) {
                MPI_Recv_init(buffer, 1, type, neighbor, tag, comm,
                              &receives_[receiveCount_++]);
            }
        };
        const auto send = [&](Real* buffer, const MPI_Datatype type,
                              const int neighbor, const int tag) {
            if (neighbor != MPI_PROC_NULL) {
                MPI_Send_init(buffer, 1, type, neighbor, tag, comm,
                              &sends_[sendCount_++]);
            }
        };

        // Receive requests are kept separate so computation can resume as soon
        // as incoming halos arrive, without waiting for outgoing sends.
        receive(data + at(0, 1, 1), faces.xFace, neighbors[0], X_TO_PLUS);
        receive(data + at(lx + 1, 1, 1), faces.xFace, neighbors[1], X_TO_MINUS);
        receive(data + at(1, 0, 1), faces.yFace, neighbors[2], Y_TO_PLUS);
        receive(data + at(1, ly + 1, 1), faces.yFace, neighbors[3], Y_TO_MINUS);
        receive(data + at(1, 1, 0), faces.zFace, neighbors[4], Z_TO_PLUS);
        receive(data + at(1, 1, lz + 1), faces.zFace, neighbors[5], Z_TO_MINUS);

        send(data + at(1, 1, 1), faces.xFace, neighbors[0], X_TO_MINUS);
        send(data + at(lx, 1, 1), faces.xFace, neighbors[1], X_TO_PLUS);
        send(data + at(1, 1, 1), faces.yFace, neighbors[2], Y_TO_MINUS);
        send(data + at(1, ly, 1), faces.yFace, neighbors[3], Y_TO_PLUS);
        send(data + at(1, 1, 1), faces.zFace, neighbors[4], Z_TO_MINUS);
        send(data + at(1, 1, lz), faces.zFace, neighbors[5], Z_TO_PLUS);
    }

    PersistentHaloExchange(const PersistentHaloExchange&) = delete;
    PersistentHaloExchange& operator=(const PersistentHaloExchange&) = delete;

    ~PersistentHaloExchange() {
        for (int i = 0; i < receiveCount_; ++i) {
            MPI_Request_free(&receives_[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < sendCount_; ++i) {
            MPI_Request_free(&sends_[static_cast<size_t>(i)]);
        }
    }

    void start() {
        if (receiveCount_ != 0) MPI_Startall(receiveCount_, receives_.data());
        if (sendCount_ != 0) MPI_Startall(sendCount_, sends_.data());
    }

    void waitForReceives() {
        if (receiveCount_ != 0) {
            MPI_Waitall(receiveCount_, receives_.data(), MPI_STATUSES_IGNORE);
        }
    }

    void waitForSends() {
        if (sendCount_ != 0) MPI_Waitall(sendCount_, sends_.data(), MPI_STATUSES_IGNORE);
    }

  private:
    static constexpr int X_TO_MINUS = 10;
    static constexpr int X_TO_PLUS = 11;
    static constexpr int Y_TO_MINUS = 12;
    static constexpr int Y_TO_PLUS = 13;
    static constexpr int Z_TO_MINUS = 14;
    static constexpr int Z_TO_PLUS = 15;

    std::array<MPI_Request, 6> receives_{};
    std::array<MPI_Request, 6> sends_{};
    int receiveCount_ = 0;
    int sendCount_ = 0;
};

inline void computeBlock(const Real* STENCIL_RESTRICT input,
                         Real* STENCIL_RESTRICT output, const size_t sx,
                         const size_t sy, const size_t xBegin, const size_t xEnd,
                         const size_t yBegin, const size_t yEnd,
                         const size_t zBegin, const size_t zEnd) {
    if (xBegin >= xEnd || yBegin >= yEnd || zBegin >= zEnd) return;
    const size_t plane = sx * sy;
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            const size_t rowStart = idx3(xBegin, y, z, sx, sy);
            const size_t rowEnd = rowStart + (xEnd - xBegin);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t i = rowStart; i < rowEnd; ++i) {
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - sx] + input[i + sx] + input[i - plane] +
                             input[i + plane]) /
                            7.0;
            }
        }
    }
}

// Compute the complement of an already-computed rectangular core. The six
// blocks are disjoint. On ranks at a physical boundary, the core extends to
// that boundary because its fixed halo is already available.
void computeCoreComplement(const Real* STENCIL_RESTRICT input,
                           Real* STENCIL_RESTRICT output, const size_t lx,
                           const size_t ly, const size_t lz, const size_t xBegin,
                           const size_t xEnd, const size_t yBegin,
                           const size_t yEnd, const size_t zBegin,
                           const size_t zEnd) {
    const size_t sx = lx + 2;
    const size_t sy = ly + 2;
    if (xBegin >= xEnd || yBegin >= yEnd || zBegin >= zEnd) {
        computeBlock(input, output, sx, sy, 1, lx + 1, 1, ly + 1, 1, lz + 1);
        return;
    }

    computeBlock(input, output, sx, sy, 1, lx + 1, 1, ly + 1, 1, zBegin);
    computeBlock(input, output, sx, sy, 1, lx + 1, 1, ly + 1, zEnd, lz + 1);
    computeBlock(input, output, sx, sy, 1, lx + 1, 1, yBegin, zBegin, zEnd);
    computeBlock(input, output, sx, sy, 1, lx + 1, yEnd, ly + 1, zBegin, zEnd);
    computeBlock(input, output, sx, sy, 1, xBegin, yBegin, yEnd, zBegin, zEnd);
    computeBlock(input, output, sx, sy, xEnd, lx + 1, yBegin, yEnd, zBegin, zEnd);
}

void exchangeAndCompute(PersistentHaloExchange& exchange, const Real* input,
                        Real* output, const size_t lx, const size_t ly,
                        const size_t lz, const std::array<int, 6>& neighbors) {
    const size_t xBegin = neighbors[0] == MPI_PROC_NULL ? 1 : 2;
    const size_t xEnd = neighbors[1] == MPI_PROC_NULL ? lx + 1 : lx;
    const size_t yBegin = neighbors[2] == MPI_PROC_NULL ? 1 : 2;
    const size_t yEnd = neighbors[3] == MPI_PROC_NULL ? ly + 1 : ly;
    const size_t zBegin = neighbors[4] == MPI_PROC_NULL ? 1 : 2;
    const size_t zEnd = neighbors[5] == MPI_PROC_NULL ? lz + 1 : lz;

    exchange.start();
    // This region has no halo dependency and overlaps network progress.
    computeBlock(input, output, lx + 2, ly + 2, xBegin, xEnd, yBegin, yEnd,
                 zBegin, zEnd);
    exchange.waitForReceives();
    computeCoreComplement(input, output, lx, ly, lz, xBegin, xEnd, yBegin, yEnd,
                          zBegin, zEnd);
    exchange.waitForSends();
}

void sendLarge(const Real* data, size_t count, const int destination, MPI_Comm comm) {
    constexpr size_t chunkSize = static_cast<size_t>(1) << 30;
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, chunkSize));
        MPI_Send(data, chunk, MPI_DOUBLE, destination, 90, comm);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveLarge(Real* data, size_t count, const int source, MPI_Comm comm) {
    constexpr size_t chunkSize = static_cast<size_t>(1) << 30;
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, chunkSize));
        MPI_Recv(data, chunk, MPI_DOUBLE, source, 90, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

std::vector<Real> packInterior(const Real* grid, const size_t lx, const size_t ly,
                               const size_t lz) {
    std::vector<Real> packed(lx * ly * lz);
    const size_t sx = lx + 2;
    const size_t sy = ly + 2;
    size_t target = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            const Real* row = grid + idx3(1, y, z, sx, sy);
            std::copy_n(row, lx, packed.data() + target);
            target += lx;
        }
    }
    return packed;
}

void unpackInterior(const Real* packed, std::vector<Real>& global,
                    const Partition& xp, const Partition& yp,
                    const Partition& zp, const size_t nx, const size_t ny) {
    size_t source = 0;
    for (size_t z = 0; z < zp.count; ++z) {
        for (size_t y = 0; y < yp.count; ++y) {
            Real* row = global.data() +
                        idx3(xp.begin, yp.begin + y, zp.begin + z, nx, ny);
            std::copy_n(packed + source, xp.count, row);
            source += xp.count;
        }
    }
}

std::vector<Real> gatherGlobalGrid(const Real* localGrid, const Partition& xp,
                                   const Partition& yp, const Partition& zp,
                                   const size_t nx, const size_t ny,
                                   const size_t nz, const std::array<int, 3>& dims,
                                   const int cartRank, const int cartSize,
                                   MPI_Comm cartComm) {
    std::vector<Real> packed = packInterior(localGrid, xp.count, yp.count, zp.count);
    if (cartRank != 0) {
        sendLarge(packed.data(), packed.size(), 0, cartComm);
        return {};
    }

    std::vector<Real> global(nx * ny * nz);
    initializeGlobalGrid(global);
    for (int source = 0; source < cartSize; ++source) {
        std::array<int, 3> coords{};
        MPI_Cart_coords(cartComm, source, 3, coords.data());
        const Partition sourceX = makePartition(nx - 2, coords[0], dims[0]);
        const Partition sourceY = makePartition(ny - 2, coords[1], dims[1]);
        const Partition sourceZ = makePartition(nz - 2, coords[2], dims[2]);
        const size_t sourceCount = sourceX.count * sourceY.count * sourceZ.count;

        if (source == 0) {
            unpackInterior(packed.data(), global, sourceX, sourceY, sourceZ, nx, ny);
        } else {
            std::vector<Real> received(sourceCount);
            receiveLarge(received.data(), received.size(), source, cartComm);
            unpackInterior(received.data(), global, sourceX, sourceY, sourceZ, nx, ny);
        }
    }
    return global;
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto bounds = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 1e6 || *bounds.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void includeInitialValue(const size_t index, Real& minimum, Real& maximum) {
    const Real value = static_cast<Real>(index % 19);
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
}

// Validation normally remains distributed. Rank zero adds the immutable outer
// boundary to the reductions without allocating the global grid.
bool validateDistributed(const Real* grid, const Partition& xp,
                         const Partition& yp, const Partition& zp,
                         const size_t nx, const size_t ny, const size_t nz,
                         const int cartRank, MPI_Comm cartComm) {
    const size_t sx = xp.count + 2;
    const size_t sy = yp.count + 2;
    int finite = 1;
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -std::numeric_limits<Real>::infinity();
    for (size_t z = 1; z <= zp.count; ++z) {
        for (size_t y = 1; y <= yp.count; ++y) {
            for (size_t x = 1; x <= xp.count; ++x) {
                const Real value = grid[idx3(x, y, z, sx, sy)];
                if (!std::isfinite(value)) finite = 0;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
        }
    }

    if (cartRank == 0) {
        const size_t plane = nx * ny;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                includeInitialValue(y * nx + x, minimum, maximum);
                includeInitialValue((nz - 1) * plane + y * nx + x, minimum, maximum);
            }
        }
        for (size_t z = 1; z + 1 < nz; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                includeInitialValue(z * plane + x, minimum, maximum);
                includeInitialValue(z * plane + (ny - 1) * nx + x, minimum, maximum);
            }
            for (size_t y = 1; y + 1 < ny; ++y) {
                includeInitialValue(z * plane + y * nx, minimum, maximum);
                includeInitialValue(z * plane + y * nx + nx - 1, minimum, maximum);
            }
        }
    }

    int allFinite = 0;
    Real globalMinimum = 0.0;
    Real globalMaximum = 0.0;
    MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, 0, cartComm);
    MPI_Reduce(&minimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, 0, cartComm);
    MPI_Reduce(&maximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

    int valid = 1;
    if (cartRank == 0) {
        if (!allFinite) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            std::printf("Value range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            if (globalMaximum > 1e6 || globalMinimum < -1e6) {
                std::printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, cartComm);
    return valid != 0;
}

int runActiveBenchmark(const Options& options, const ProcessGrid& processGrid,
                       MPI_Comm activeComm) {
    const std::array<int, 3> periods{0, 0, 0};
    MPI_Comm cartComm = MPI_COMM_NULL;
    MPI_Cart_create(activeComm, 3, processGrid.dims.data(), periods.data(), 0,
                    &cartComm);

    int cartRank = 0;
    int cartSize = 1;
    MPI_Comm_rank(cartComm, &cartRank);
    MPI_Comm_size(cartComm, &cartSize);
    std::array<int, 3> coords{};
    MPI_Cart_coords(cartComm, cartRank, 3, coords.data());

    const Partition xp = makePartition(options.nx - 2, coords[0], processGrid.dims[0]);
    const Partition yp = makePartition(options.ny - 2, coords[1], processGrid.dims[1]);
    const Partition zp = makePartition(options.nz - 2, coords[2], processGrid.dims[2]);
    const size_t lx = xp.count;
    const size_t ly = yp.count;
    const size_t lz = zp.count;

    std::array<int, 6> neighbors{};
    MPI_Cart_shift(cartComm, 0, 1, &neighbors[0], &neighbors[1]);
    MPI_Cart_shift(cartComm, 1, 1, &neighbors[2], &neighbors[3]);
    MPI_Cart_shift(cartComm, 2, 1, &neighbors[4], &neighbors[5]);

    if (cartRank == 0) {
        std::printf("3D Stencil Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
        std::printf("Iterations: %d\n", options.iterations);
        std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d (Cartesian grid: %d x %d x %d)\n", cartSize,
                    processGrid.dims[0], processGrid.dims[1], processGrid.dims[2]);
        std::printf("Initializing grid...\n");
    }

    size_t localXY = 0;
    size_t localSize = 0;
    if (!checkedProduct(lx + 2, ly + 2, localXY) ||
        !checkedProduct(localXY, lz + 2, localSize)) {
        if (cartRank == 0) std::fprintf(stderr, "Local grid size overflow\n");
        MPI_Abort(cartComm, 1);
    }

    int exitCode = 0;
    {
        std::vector<Real> grid1(localSize);
        std::vector<Real> grid2(localSize);
        initializeLocalGrid(grid1, xp, yp, zp, options.nx, options.ny);
        initializeLocalGrid(grid2, xp, yp, zp, options.nx, options.ny);

        FaceTypes faceTypes(lx, ly, lz);
        PersistentHaloExchange exchange1(grid1.data(), lx, ly, lz, faceTypes,
                                         neighbors, cartComm);
        PersistentHaloExchange exchange2(grid2.data(), lx, ly, lz, faceTypes,
                                         neighbors, cartComm);

        Real* current = grid1.data();
        Real* next = grid2.data();
        PersistentHaloExchange* currentExchange = &exchange1;
        PersistentHaloExchange* nextExchange = &exchange2;

        if (cartRank == 0) std::printf("Running stencil computation...\n");
        MPI_Barrier(cartComm);
        const double start = MPI_Wtime();
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            exchangeAndCompute(*currentExchange, current, next, lx, ly, lz,
                               neighbors);
            std::swap(current, next);
            std::swap(currentExchange, nextExchange);
        }
        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

        if (cartRank == 0) {
            const auto milliseconds = static_cast<long long>(elapsed * 1000.0);
            const double updates = static_cast<double>(options.nx - 2) *
                                   static_cast<double>(options.ny - 2) *
                                   static_cast<double>(options.nz - 2) *
                                   static_cast<double>(options.iterations);
            const double mcups = updates / elapsed / 1.0e6;
            std::printf("Computation time: %lld ms\n", milliseconds);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        std::vector<Real> globalResult;
        if (options.printResults) {
            globalResult = gatherGlobalGrid(current, xp, yp, zp, options.nx,
                                            options.ny, options.nz, processGrid.dims,
                                            cartRank, cartSize, cartComm);
            if (cartRank == 0) print_results(globalResult, "Grid");
        }

        if (options.validate) {
            if (cartRank == 0) std::printf("Validating result...\n");
            bool valid = true;
            if (options.printResults) {
                int validInteger = 1;
                if (cartRank == 0) validInteger = validateResult(globalResult) ? 1 : 0;
                MPI_Bcast(&validInteger, 1, MPI_INT, 0, cartComm);
                valid = validInteger != 0;
            } else {
                valid = validateDistributed(current, xp, yp, zp, options.nx,
                                            options.ny, options.nz, cartRank, cartComm);
            }
            if (cartRank == 0) {
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Comm_free(&cartComm);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    Options options;
    const int parseStatus = parseArguments(argc, argv, options, worldRank == 0);
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    size_t xy = 0;
    size_t gridSize = 0;
    const bool validDimensions = options.nx >= 3 && options.ny >= 3 && options.nz >= 3 &&
        options.nx <= static_cast<size_t>(INT_MAX - 2) &&
        options.ny <= static_cast<size_t>(INT_MAX - 2) &&
        options.nz <= static_cast<size_t>(INT_MAX - 2) &&
        checkedProduct(options.nx, options.ny, xy) &&
        checkedProduct(xy, options.nz, gridSize);
    if (!validDimensions) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "Grid dimensions must each be at least 3, fit MPI datatype "
                         "limits, and have a representable total size.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const ProcessGrid processGrid =
        chooseProcessGrid(worldSize, options.nx - 2, options.ny - 2, options.nz - 2);
    if (worldRank == 0 && processGrid.activeRanks != worldSize) {
        std::fprintf(stderr,
                     "Warning: using %d of %d MPI ranks because the interior grid "
                     "cannot be partitioned into more nonempty Cartesian blocks.\n",
                     processGrid.activeRanks, worldSize);
    }

    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD,
                   worldRank < processGrid.activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &activeComm);

    int exitCode = 0;
    if (activeComm != MPI_COMM_NULL) {
        exitCode = runActiveBenchmark(options, processGrid, activeComm);
        MPI_Comm_free(&activeComm);
    }

    // Inactive excess ranks wait here and receive the root's final status.
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
