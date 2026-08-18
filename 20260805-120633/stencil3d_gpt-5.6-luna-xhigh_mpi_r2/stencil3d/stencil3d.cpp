#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// The local arrays contain one ghost cell on every side.  X is the unit
// stride dimension, followed by Y and then Z.
inline constexpr std::size_t idx3(const std::size_t x,
                                  const std::size_t y,
                                  const std::size_t z,
                                  const std::size_t nx,
                                  const std::size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

enum Face : int {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5,
    FaceCount = 6
};

struct Block {
    std::size_t globalNx = 0;
    std::size_t globalNy = 0;
    std::size_t globalNz = 0;

    std::size_t xStart = 0;
    std::size_t yStart = 0;
    std::size_t zStart = 0;
    std::size_t xCount = 0;
    std::size_t yCount = 0;
    std::size_t zCount = 0;

    std::array<int, 3> cartDims{};
    std::array<int, FaceCount> neighbors{};

    std::size_t localNx() const noexcept { return xCount + 2; }
    std::size_t localNy() const noexcept { return yCount + 2; }
    std::size_t localNz() const noexcept { return zCount + 2; }
    std::size_t planeSize() const noexcept { return localNx() * localNy(); }
};

struct FaceBuffers {
    std::array<std::vector<Real>, FaceCount> send;
    std::array<std::vector<Real>, FaceCount> receive;

    explicit FaceBuffers(const Block& block) {
        const std::size_t xFace = block.yCount * block.zCount;
        const std::size_t yFace = block.xCount * block.zCount;
        const std::size_t zFace = block.xCount * block.yCount;

        send[XMinus].resize(xFace);
        send[XPlus].resize(xFace);
        receive[XMinus].resize(xFace);
        receive[XPlus].resize(xFace);

        send[YMinus].resize(yFace);
        send[YPlus].resize(yFace);
        receive[YMinus].resize(yFace);
        receive[YPlus].resize(yFace);

        send[ZMinus].resize(zFace);
        send[ZPlus].resize(zFace);
        receive[ZMinus].resize(zFace);
        receive[ZPlus].resize(zFace);
    }
};

struct Options {
    std::size_t nx = 128;
    std::size_t ny = 0;
    std::size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    bool help = false;
    bool valid = true;
};

bool parseSize(const char* text, std::size_t& value) {
    if (text == nullptr || text[0] == '-' || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }

    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parseIterations(const char* text, int& value) {
    if (text == nullptr || text[0] == '-' || text[0] == '\0') {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
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

Options parseOptions(int argc, char** argv) {
    Options options;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.nx) && options.valid;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.ny) && options.valid;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.valid = parseSize(argv[++i], options.nz) && options.valid;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.valid = parseIterations(argv[++i], options.iterations) && options.valid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            options.help = true;
        } else {
            options.valid = false;
        }
    }

    if (options.ny == 0) {
        options.ny = options.nx;
    }
    if (options.nz == 0) {
        options.nz = options.nx;
    }

    if (options.nx == 0 || options.ny == 0 || options.nz == 0) {
        options.valid = false;
    }

    return options;
}

bool checkedProduct(const std::size_t a, const std::size_t b, const std::size_t c,
                    std::size_t& result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    const std::size_t ab = a * b;
    if (c != 0 && ab > std::numeric_limits<std::size_t>::max() / c) {
        return false;
    }
    result = ab * c;
    return true;
}

void splitDimension(const std::size_t globalSize, const int parts, const int coordinate,
                    std::size_t& start, std::size_t& count) {
    const std::size_t partCount = static_cast<std::size_t>(parts);
    const std::size_t base = globalSize / partCount;
    const std::size_t remainder = globalSize % partCount;
    const std::size_t coordinateValue = static_cast<std::size_t>(coordinate);

    count = base + (coordinateValue < remainder ? 1 : 0);
    start = coordinateValue * base + std::min(coordinateValue, remainder);
}

// MPI_Dims_create does not accept upper bounds for dimensions.  Try every
// permutation of its factorization so a thin global dimension does not force
// otherwise usable ranks to become inactive.
int chooseActiveRanks(const int worldSize,
                      const std::size_t globalNx,
                      const std::size_t globalNy,
                      const std::size_t globalNz,
                      std::array<int, 3>& dimensions) {
    constexpr int permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
    };

    for (int candidate = worldSize; candidate >= 1; --candidate) {
        int raw[3] = {0, 0, 0};
        MPI_Dims_create(candidate, 3, raw);

        double bestSurface = std::numeric_limits<double>::infinity();
        std::array<int, 3> best{};
        bool found = false;
        for (const auto& permutation : permutations) {
            const std::array<int, 3> trial = {
                raw[permutation[0]], raw[permutation[1]], raw[permutation[2]]
            };
            if (static_cast<std::size_t>(trial[0]) > globalNx ||
                static_cast<std::size_t>(trial[1]) > globalNy ||
                static_cast<std::size_t>(trial[2]) > globalNz) {
                continue;
            }

            const double localX = static_cast<double>(globalNx) / trial[0];
            const double localY = static_cast<double>(globalNy) / trial[1];
            const double localZ = static_cast<double>(globalNz) / trial[2];
            const double surface = localY * localZ + localX * localZ + localX * localY;
            if (!found || surface < bestSurface) {
                found = true;
                bestSurface = surface;
                best = trial;
            }
        }

        if (found) {
            dimensions = best;
            return candidate;
        }
    }

    // The dimensions are positive, so one rank is always a valid fallback.
    dimensions = {1, 1, 1};
    return 1;
}

Block makeBlock(const std::size_t globalNx,
                const std::size_t globalNy,
                const std::size_t globalNz,
                const std::array<int, 3>& cartDims,
                const int coordinates[3]) {
    Block block;
    block.globalNx = globalNx;
    block.globalNy = globalNy;
    block.globalNz = globalNz;
    block.cartDims = cartDims;

    splitDimension(globalNx, cartDims[0], coordinates[0], block.xStart, block.xCount);
    splitDimension(globalNy, cartDims[1], coordinates[1], block.yStart, block.yCount);
    splitDimension(globalNz, cartDims[2], coordinates[2], block.zStart, block.zCount);
    return block;
}

int mpiCount(const std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "MPI message is too large for the configured MPI interface\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

void initializeGrid(std::vector<Real>& grid, const Block& block) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();

    for (std::size_t z = 1; z <= block.zCount; ++z) {
        const std::size_t globalZ = block.zStart + z - 1;
        for (std::size_t y = 1; y <= block.yCount; ++y) {
            const std::size_t globalY = block.yStart + y - 1;
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const std::size_t globalX = block.xStart + x - 1;
                const std::size_t globalIndex =
                    (globalZ * block.globalNy + globalY) * block.globalNx + globalX;
                grid[idx3(x, y, z, localNx, localNy)] =
                    static_cast<Real>(globalIndex % 19);
            }
        }
    }
}

void packFaces(const std::vector<Real>& input, const Block& block, FaceBuffers& buffers) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();

    if (block.neighbors[XMinus] != MPI_PROC_NULL || block.neighbors[XPlus] != MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= block.zCount; ++z) {
            for (std::size_t y = 1; y <= block.yCount; ++y) {
                const std::size_t offset = (z - 1) * block.yCount + (y - 1);
                const std::size_t minusIndex = idx3(1, y, z, localNx, localNy);
                const std::size_t plusIndex = idx3(block.xCount, y, z, localNx, localNy);
                if (block.neighbors[XMinus] != MPI_PROC_NULL) {
                    buffers.send[XMinus][offset] = input[minusIndex];
                }
                if (block.neighbors[XPlus] != MPI_PROC_NULL) {
                    buffers.send[XPlus][offset] = input[plusIndex];
                }
            }
        }
    }

    if (block.neighbors[YMinus] != MPI_PROC_NULL || block.neighbors[YPlus] != MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= block.zCount; ++z) {
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const std::size_t offset = (z - 1) * block.xCount + (x - 1);
                const std::size_t minusIndex = idx3(x, 1, z, localNx, localNy);
                const std::size_t plusIndex = idx3(x, block.yCount, z, localNx, localNy);
                if (block.neighbors[YMinus] != MPI_PROC_NULL) {
                    buffers.send[YMinus][offset] = input[minusIndex];
                }
                if (block.neighbors[YPlus] != MPI_PROC_NULL) {
                    buffers.send[YPlus][offset] = input[plusIndex];
                }
            }
        }
    }

    if (block.neighbors[ZMinus] != MPI_PROC_NULL || block.neighbors[ZPlus] != MPI_PROC_NULL) {
        for (std::size_t y = 1; y <= block.yCount; ++y) {
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const std::size_t offset = (y - 1) * block.xCount + (x - 1);
                const std::size_t minusIndex = idx3(x, y, 1, localNx, localNy);
                const std::size_t plusIndex = idx3(x, y, block.zCount, localNx, localNy);
                if (block.neighbors[ZMinus] != MPI_PROC_NULL) {
                    buffers.send[ZMinus][offset] = input[minusIndex];
                }
                if (block.neighbors[ZPlus] != MPI_PROC_NULL) {
                    buffers.send[ZPlus][offset] = input[plusIndex];
                }
            }
        }
    }
}

void beginHaloExchange(std::vector<Real>& input,
                       const Block& block,
                       FaceBuffers& buffers,
                       MPI_Comm cartComm,
                       std::array<MPI_Request, 12>& requests,
                       int& requestCount) {
    packFaces(input, block, buffers);
    requestCount = 0;

    const auto postReceive = [&](const Face face, const int source, const int tag) {
        if (source != MPI_PROC_NULL) {
            MPI_Irecv(buffers.receive[face].data(), mpiCount(buffers.receive[face].size()),
                      MPI_DOUBLE, source, tag, cartComm, &requests[requestCount++]);
        }
    };
    const auto postSend = [&](const Face face, const int destination, const int tag) {
        if (destination != MPI_PROC_NULL) {
            MPI_Isend(buffers.send[face].data(), mpiCount(buffers.send[face].size()),
                      MPI_DOUBLE, destination, tag, cartComm, &requests[requestCount++]);
        }
    };

    // Post all receives before any sends.  Opposite faces use opposite tags,
    // allowing every rank to issue the same six-neighbor exchange pattern.
    postReceive(XMinus, block.neighbors[XMinus], 101);
    postReceive(XPlus, block.neighbors[XPlus], 100);
    postReceive(YMinus, block.neighbors[YMinus], 201);
    postReceive(YPlus, block.neighbors[YPlus], 200);
    postReceive(ZMinus, block.neighbors[ZMinus], 301);
    postReceive(ZPlus, block.neighbors[ZPlus], 300);

    postSend(XMinus, block.neighbors[XMinus], 100);
    postSend(XPlus, block.neighbors[XPlus], 101);
    postSend(YMinus, block.neighbors[YMinus], 200);
    postSend(YPlus, block.neighbors[YPlus], 201);
    postSend(ZMinus, block.neighbors[ZMinus], 300);
    postSend(ZPlus, block.neighbors[ZPlus], 301);
}

void unpackFaces(std::vector<Real>& input, const Block& block, const FaceBuffers& buffers) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();

    if (block.neighbors[XMinus] != MPI_PROC_NULL || block.neighbors[XPlus] != MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= block.zCount; ++z) {
            for (std::size_t y = 1; y <= block.yCount; ++y) {
                const std::size_t offset = (z - 1) * block.yCount + (y - 1);
                if (block.neighbors[XMinus] != MPI_PROC_NULL) {
                    input[idx3(0, y, z, localNx, localNy)] = buffers.receive[XMinus][offset];
                }
                if (block.neighbors[XPlus] != MPI_PROC_NULL) {
                    input[idx3(block.xCount + 1, y, z, localNx, localNy)] =
                        buffers.receive[XPlus][offset];
                }
            }
        }
    }

    if (block.neighbors[YMinus] != MPI_PROC_NULL || block.neighbors[YPlus] != MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= block.zCount; ++z) {
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const std::size_t offset = (z - 1) * block.xCount + (x - 1);
                if (block.neighbors[YMinus] != MPI_PROC_NULL) {
                    input[idx3(x, 0, z, localNx, localNy)] = buffers.receive[YMinus][offset];
                }
                if (block.neighbors[YPlus] != MPI_PROC_NULL) {
                    input[idx3(x, block.yCount + 1, z, localNx, localNy)] =
                        buffers.receive[YPlus][offset];
                }
            }
        }
    }

    if (block.neighbors[ZMinus] != MPI_PROC_NULL || block.neighbors[ZPlus] != MPI_PROC_NULL) {
        for (std::size_t y = 1; y <= block.yCount; ++y) {
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const std::size_t offset = (y - 1) * block.xCount + (x - 1);
                if (block.neighbors[ZMinus] != MPI_PROC_NULL) {
                    input[idx3(x, y, 0, localNx, localNy)] = buffers.receive[ZMinus][offset];
                }
                if (block.neighbors[ZPlus] != MPI_PROC_NULL) {
                    input[idx3(x, y, block.zCount + 1, localNx, localNy)] =
                        buffers.receive[ZPlus][offset];
                }
            }
        }
    }
}

struct Range {
    std::size_t begin = 0;
    std::size_t end = 0;
};

Range interiorRange(const std::size_t start,
                    const std::size_t count,
                    const std::size_t globalSize) {
    Range range;
    range.begin = start == 0 ? 2 : 1;
    range.end = start + count == globalSize ? count : count + 1;
    return range;
}

void copyOwnedRow(const std::vector<Real>& input,
                  std::vector<Real>& output,
                  const std::size_t xCount,
                  const std::size_t localNx,
                  const std::size_t localNy,
                  const std::size_t y,
                  const std::size_t z) {
    const std::size_t source = idx3(1, y, z, localNx, localNy);
    const std::size_t destination = idx3(1, y, z, localNx, localNy);
    std::memcpy(output.data() + destination, input.data() + source, xCount * sizeof(Real));
}

void copyGlobalBoundaries(const std::vector<Real>& input,
                          std::vector<Real>& output,
                          const Block& block) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();
    const Range interiorY = interiorRange(block.yStart, block.yCount, block.globalNy);

    for (std::size_t z = 1; z <= block.zCount; ++z) {
        const std::size_t globalZ = block.zStart + z - 1;
        if (globalZ == 0 || globalZ + 1 == block.globalNz) {
            for (std::size_t y = 1; y <= block.yCount; ++y) {
                copyOwnedRow(input, output, block.xCount, localNx, localNy, y, z);
            }
            continue;
        }

        if (block.yStart == 0) {
            copyOwnedRow(input, output, block.xCount, localNx, localNy, 1, z);
        }
        if (block.yStart + block.yCount == block.globalNy) {
            copyOwnedRow(input, output, block.xCount, localNx, localNy, block.yCount, z);
        }

        for (std::size_t y = interiorY.begin; y < interiorY.end; ++y) {
            if (block.xStart == 0) {
                output[idx3(1, y, z, localNx, localNy)] =
                    input[idx3(1, y, z, localNx, localNy)];
            }
            if (block.xStart + block.xCount == block.globalNx) {
                output[idx3(block.xCount, y, z, localNx, localNy)] =
                    input[idx3(block.xCount, y, z, localNx, localNy)];
            }
        }
    }
}

void computeStencilBox(const std::vector<Real>& input,
                       std::vector<Real>& output,
                       const Block& block,
                       const Range xRange,
                       const Range yRange,
                       const Range zRange) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();
    const std::size_t planeStride = localNx * localNy;
    const std::size_t rowStride = localNx;

    for (std::size_t z = zRange.begin; z < zRange.end; ++z) {
        for (std::size_t y = yRange.begin; y < yRange.end; ++y) {
            const std::size_t row = z * planeStride + y * rowStride;
            for (std::size_t x = xRange.begin; x < xRange.end; ++x) {
                const std::size_t center = row + x;
                output[center] = (input[center] + input[center - 1] + input[center + 1] +
                                  input[center - rowStride] + input[center + rowStride] +
                                  input[center - planeStride] + input[center + planeStride]) / 7.0;
            }
        }
    }
}

void computeCore(const std::vector<Real>& input,
                 std::vector<Real>& output,
                 const Block& block,
                 const Range xRange,
                 const Range yRange,
                 const Range zRange,
                 Range& coreX,
                 Range& coreY,
                 Range& coreZ) {
    coreX = xRange;
    coreY = yRange;
    coreZ = zRange;

    if (block.neighbors[XMinus] != MPI_PROC_NULL) {
        ++coreX.begin;
    }
    if (block.neighbors[XPlus] != MPI_PROC_NULL) {
        --coreX.end;
    }
    if (block.neighbors[YMinus] != MPI_PROC_NULL) {
        ++coreY.begin;
    }
    if (block.neighbors[YPlus] != MPI_PROC_NULL) {
        --coreY.end;
    }
    if (block.neighbors[ZMinus] != MPI_PROC_NULL) {
        ++coreZ.begin;
    }
    if (block.neighbors[ZPlus] != MPI_PROC_NULL) {
        --coreZ.end;
    }

    if (coreX.begin < coreX.end && coreY.begin < coreY.end && coreZ.begin < coreZ.end) {
        computeStencilBox(input, output, block, coreX, coreY, coreZ);
    }
}

void computeShell(const std::vector<Real>& input,
                  std::vector<Real>& output,
                  const Block& block,
                  const Range xRange,
                  const Range yRange,
                  const Range zRange,
                  const Range coreX,
                  const Range coreY,
                  const Range coreZ) {
    const bool hasCore = coreX.begin < coreX.end && coreY.begin < coreY.end &&
                         coreZ.begin < coreZ.end;
    if (!hasCore) {
        computeStencilBox(input, output, block, xRange, yRange, zRange);
        return;
    }

    // The shell is split into disjoint boxes, avoiding a per-cell branch and
    // avoiding recomputation of the interior already evaluated during MPI.
    computeStencilBox(input, output, block,
                      xRange, yRange,
                      {zRange.begin, std::min(zRange.end, coreZ.begin)});
    computeStencilBox(input, output, block,
                      xRange, yRange,
                      {std::max(zRange.begin, coreZ.end), zRange.end});

    const Range coreZClipped = {
        std::max(zRange.begin, coreZ.begin), std::min(zRange.end, coreZ.end)
    };
    computeStencilBox(input, output, block,
                      xRange,
                      {yRange.begin, std::min(yRange.end, coreY.begin)},
                      coreZClipped);
    computeStencilBox(input, output, block,
                      xRange,
                      {std::max(yRange.begin, coreY.end), yRange.end},
                      coreZClipped);

    const Range coreYClipped = {
        std::max(yRange.begin, coreY.begin), std::min(yRange.end, coreY.end)
    };
    computeStencilBox(input, output, block,
                      {xRange.begin, std::min(xRange.end, coreX.begin)},
                      coreYClipped,
                      coreZClipped);
    computeStencilBox(input, output, block,
                      {std::max(xRange.begin, coreX.end), xRange.end},
                      coreYClipped,
                      coreZClipped);
}

void stencilIteration(std::vector<Real>& input,
                      std::vector<Real>& output,
                      const Block& block,
                      MPI_Comm cartComm,
                      FaceBuffers& buffers) {
    std::array<MPI_Request, 12> requests{};
    int requestCount = 0;
    beginHaloExchange(input, block, buffers, cartComm, requests, requestCount);

    const Range xInterior = interiorRange(block.xStart, block.xCount, block.globalNx);
    const Range yInterior = interiorRange(block.yStart, block.yCount, block.globalNy);
    const Range zInterior = interiorRange(block.zStart, block.zCount, block.globalNz);

    copyGlobalBoundaries(input, output, block);

    Range coreX;
    Range coreY;
    Range coreZ;
    computeCore(input, output, block, xInterior, yInterior, zInterior,
                coreX, coreY, coreZ);

    if (requestCount != 0) {
        MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
    }
    unpackFaces(input, block, buffers);

    computeShell(input, output, block, xInterior, yInterior, zInterior,
                 coreX, coreY, coreZ);
}

void packOwned(const std::vector<Real>& local,
               const Block& block,
               std::vector<Real>& packed) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();
    packed.resize(block.xCount * block.yCount * block.zCount);

    std::size_t offset = 0;
    for (std::size_t z = 1; z <= block.zCount; ++z) {
        for (std::size_t y = 1; y <= block.yCount; ++y) {
            const std::size_t source = idx3(1, y, z, localNx, localNy);
            std::memcpy(packed.data() + offset, local.data() + source,
                        block.xCount * sizeof(Real));
            offset += block.xCount;
        }
    }
}

void placePackedBlock(const Real* packed,
                      const Block& block,
                      std::vector<Real>& globalGrid) {
    std::size_t offset = 0;
    for (std::size_t z = 0; z < block.zCount; ++z) {
        for (std::size_t y = 0; y < block.yCount; ++y) {
            const std::size_t globalIndex =
                idx3(block.xStart, block.yStart + y, block.zStart + z,
                     block.globalNx, block.globalNy);
            std::memcpy(globalGrid.data() + globalIndex, packed + offset,
                        block.xCount * sizeof(Real));
            offset += block.xCount;
        }
    }
}

void placePackedBlock(const std::vector<Real>& packed,
                      const Block& block,
                      std::vector<Real>& globalGrid) {
    placePackedBlock(packed.data(), block, globalGrid);
}

Block blockForCartRank(const std::size_t globalNx,
                       const std::size_t globalNy,
                       const std::size_t globalNz,
                       const std::array<int, 3>& cartDims,
                       MPI_Comm cartComm,
                       const int cartRank) {
    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(cartComm, cartRank, 3, coordinates);
    return makeBlock(globalNx, globalNy, globalNz, cartDims, coordinates);
}

void gatherGlobalGrid(const std::vector<Real>& localGrid,
                      const Block& localBlock,
                      MPI_Comm cartComm,
                      const int cartRank,
                      const int activeRanks,
                      std::vector<Real>& globalGrid) {
    std::vector<Real> packed;
    packOwned(localGrid, localBlock, packed);

    std::size_t globalSize = 0;
    checkedProduct(localBlock.globalNx, localBlock.globalNy, localBlock.globalNz, globalSize);

    if (globalSize <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<Real> packedAll;
        if (cartRank == 0) {
            counts.resize(activeRanks);
            displacements.resize(activeRanks);
            packedAll.resize(globalSize);
            std::size_t displacement = 0;
            for (int rank = 0; rank < activeRanks; ++rank) {
                const Block block = blockForCartRank(localBlock.globalNx, localBlock.globalNy,
                                                     localBlock.globalNz, localBlock.cartDims,
                                                     cartComm, rank);
                const std::size_t count = block.xCount * block.yCount * block.zCount;
                counts[rank] = mpiCount(count);
                displacements[rank] = mpiCount(displacement);
                displacement += count;
            }
        }

        MPI_Gatherv(packed.data(), mpiCount(packed.size()), MPI_DOUBLE,
                    cartRank == 0 ? packedAll.data() : nullptr,
                    cartRank == 0 ? counts.data() : nullptr,
                    cartRank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, cartComm);

        if (cartRank == 0) {
            globalGrid.resize(globalSize);
            for (int rank = 0; rank < activeRanks; ++rank) {
                const Block block = blockForCartRank(localBlock.globalNx, localBlock.globalNy,
                                                     localBlock.globalNz, localBlock.cartDims,
                                                     cartComm, rank);
                placePackedBlock(packedAll.data() + displacements[rank], block, globalGrid);
            }
        }
        return;
    }

    // MPI_Gatherv uses int counts and displacements.  Keep -r functional for
    // very large grids by exchanging each block in bounded chunks instead.
    if (cartRank == 0) {
        globalGrid.resize(globalSize);
        placePackedBlock(packed, localBlock, globalGrid);
        for (int rank = 1; rank < activeRanks; ++rank) {
            const Block block = blockForCartRank(localBlock.globalNx, localBlock.globalNy,
                                                 localBlock.globalNz, localBlock.cartDims,
                                                 cartComm, rank);
            const std::size_t blockSize = block.xCount * block.yCount * block.zCount;
            std::vector<Real> received(blockSize);
            for (std::size_t offset = 0; offset < blockSize;) {
                const std::size_t chunk = std::min<std::size_t>(
                    blockSize - offset, static_cast<std::size_t>(std::numeric_limits<int>::max()));
                MPI_Recv(received.data() + offset, mpiCount(chunk), MPI_DOUBLE,
                         rank, 400, cartComm, MPI_STATUS_IGNORE);
                offset += chunk;
            }
            placePackedBlock(received, block, globalGrid);
        }
    } else {
        for (std::size_t offset = 0; offset < packed.size();) {
            const std::size_t chunk = std::min<std::size_t>(
                packed.size() - offset, static_cast<std::size_t>(std::numeric_limits<int>::max()));
            MPI_Send(packed.data() + offset, mpiCount(chunk), MPI_DOUBLE,
                     0, 400, cartComm);
            offset += chunk;
        }
    }
}

bool validateDistributed(const std::vector<Real>& localGrid,
                         const Block& block,
                         MPI_Comm cartComm,
                         const int cartRank) {
    const std::size_t localNx = block.localNx();
    const std::size_t localNy = block.localNy();
    int localInvalid = 0;
    Real localMin = 0.0;
    Real localMax = 0.0;
    bool first = true;

    for (std::size_t z = 1; z <= block.zCount; ++z) {
        for (std::size_t y = 1; y <= block.yCount; ++y) {
            for (std::size_t x = 1; x <= block.xCount; ++x) {
                const Real value = localGrid[idx3(x, y, z, localNx, localNy)];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = 1;
                }
                if (first) {
                    localMin = value;
                    localMax = value;
                    first = false;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    int globalInvalid = 0;
    MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_MAX, cartComm);
    if (globalInvalid != 0) {
        if (cartRank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    Real globalMin = 0.0;
    Real globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, cartComm);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

    bool valid = true;
    if (cartRank == 0) {
        std::printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
        if (globalMax > 1e6 || globalMin < -1e6) {
            std::printf("Validation failed: values out of expected range\n");
            valid = false;
        }
    }

    int validInt = valid ? 1 : 0;
    MPI_Bcast(&validInt, 1, MPI_INT, 0, cartComm);
    return validInt != 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    const Options options = parseOptions(argc, argv);
    if (options.help || !options.valid) {
        if (worldRank == 0) {
            if (!options.valid) {
                std::printf("Invalid command line arguments.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return options.valid ? 0 : 1;
    }

    std::size_t globalSize = 0;
    if (!checkedProduct(options.nx, options.ny, options.nz, globalSize) ||
        options.nx > std::numeric_limits<std::size_t>::max() - 2 ||
        options.ny > std::numeric_limits<std::size_t>::max() - 2 ||
        options.nz > std::numeric_limits<std::size_t>::max() - 2) {
        if (worldRank == 0) {
            std::printf("Grid dimensions are too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    std::array<int, 3> cartDims{};
    const int activeRanks = chooseActiveRanks(worldSize, options.nx, options.ny, options.nz,
                                              cartDims);

    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &activeComm);

    int exitCode = 0;
    if (worldRank < activeRanks) {
        MPI_Comm cartComm = MPI_COMM_NULL;
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(activeComm, 3, cartDims.data(), periods, 0, &cartComm);

        int cartRank = 0;
        MPI_Comm_rank(cartComm, &cartRank);
        int coordinates[3] = {0, 0, 0};
        MPI_Cart_coords(cartComm, cartRank, 3, coordinates);

        Block block = makeBlock(options.nx, options.ny, options.nz, cartDims, coordinates);
        MPI_Cart_shift(cartComm, 0, 1, &block.neighbors[XMinus], &block.neighbors[XPlus]);
        MPI_Cart_shift(cartComm, 1, 1, &block.neighbors[YMinus], &block.neighbors[YPlus]);
        MPI_Cart_shift(cartComm, 2, 1, &block.neighbors[ZMinus], &block.neighbors[ZPlus]);

        std::size_t localStorage = 0;
        if (!checkedProduct(block.localNx(), block.localNy(), block.localNz(), localStorage)) {
            std::fprintf(stderr, "Local grid dimensions are too large.\n");
            MPI_Abort(cartComm, 1);
        }

        if (cartRank == 0) {
            std::printf("3D Stencil Benchmark\n");
            std::printf("Grid size: %zu x %zu x %zu\n", options.nx, options.ny, options.nz);
            std::printf("Iterations: %d\n", options.iterations);
            std::printf("MPI ranks: %d (Cartesian decomposition: %d x %d x %d)\n",
                        activeRanks, cartDims[0], cartDims[1], cartDims[2]);
            std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
            std::printf("Initializing grid...\n");
        }

        std::vector<Real> grid1(localStorage);
        std::vector<Real> grid2(localStorage);
        initializeGrid(grid1, block);

        if (cartRank == 0) {
            std::printf("Running stencil computation...\n");
        }

        FaceBuffers buffers(block);
        MPI_Barrier(cartComm);
        const double start = MPI_Wtime();

        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            if ((iteration & 1) == 0) {
                stencilIteration(grid1, grid2, block, cartComm, buffers);
            } else {
                stencilIteration(grid2, grid1, block, cartComm, buffers);
            }
        }

        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

        if (cartRank == 0) {
            const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
            const double interiorX = options.nx > 2 ? static_cast<double>(options.nx - 2) : 0.0;
            const double interiorY = options.ny > 2 ? static_cast<double>(options.ny - 2) : 0.0;
            const double interiorZ = options.nz > 2 ? static_cast<double>(options.nz - 2) : 0.0;
            const double cellUpdates = interiorX * interiorY * interiorZ * options.iterations;
            const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
            std::printf("Computation time: %ld ms\n", durationMilliseconds);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        const std::vector<Real>& finalGrid =
            (options.iterations % 2 == 0) ? grid1 : grid2;
        if (options.printResults) {
            std::vector<Real> globalGrid;
            gatherGlobalGrid(finalGrid, block, cartComm, cartRank, activeRanks, globalGrid);
            if (cartRank == 0) {
                print_results(globalGrid, "Grid");
            }
        }

        if (options.validate) {
            if (cartRank == 0) {
                std::printf("Validating result...\n");
            }
            if (!validateDistributed(finalGrid, block, cartComm, cartRank)) {
                exitCode = 1;
            } else if (cartRank == 0) {
                std::printf("Validation: PASSED\n");
            }
        }

        MPI_Comm_free(&cartComm);
        MPI_Comm_free(&activeComm);
    }

    MPI_Finalize();
    return exitCode;
}
