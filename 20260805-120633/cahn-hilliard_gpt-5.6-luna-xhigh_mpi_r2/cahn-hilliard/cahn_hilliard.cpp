#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Interior cells occupy [1, local_n + 1] in every direction.  The extra
// layer on each side contains the values needed by the seven-point stencil.
inline constexpr std::size_t idx3(const std::size_t x, const std::size_t y, const std::size_t z,
                                  const std::size_t nx, const std::size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

enum Face : int {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5
};

struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[3] = {1, 1, 1};
    int coords[3] = {0, 0, 0};
    int neighbors[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL,
                        MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    std::size_t globalX = 0;
    std::size_t globalY = 0;
    std::size_t globalZ = 0;
    std::size_t localX = 0;
    std::size_t localY = 0;
    std::size_t localZ = 0;
    std::size_t offsetX = 0;
    std::size_t offsetY = 0;
    std::size_t offsetZ = 0;
};

struct HaloBuffers {
    std::array<std::vector<double>, 6> send;
    std::array<std::vector<double>, 6> receive;
    std::array<MPI_Request, 12> requests{};
    int activeRequestCount = 0;

    HaloBuffers(const std::size_t nx, const std::size_t ny, const std::size_t nz) {
        const std::size_t xFace = ny * nz;
        const std::size_t yFace = nx * nz;
        const std::size_t zFace = nx * ny;
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

inline double computeLaplacian(const std::vector<double>& field,
                               const std::size_t nxWithHalo, const std::size_t nyWithHalo,
                               const std::size_t x, const std::size_t y, const std::size_t z,
                               const double invDx2, const double invDy2, const double invDz2) noexcept {
    const std::size_t center = idx3(x, y, z, nxWithHalo, nyWithHalo);
    const double cxx = (field[center + 1] + field[center - 1] - 2.0 * field[center]) * invDx2;
    const double cyy = (field[idx3(x, y + 1, z, nxWithHalo, nyWithHalo)] +
                        field[idx3(x, y - 1, z, nxWithHalo, nyWithHalo)] -
                        2.0 * field[center]) * invDy2;
    const double czz = (field[idx3(x, y, z + 1, nxWithHalo, nyWithHalo)] +
                        field[idx3(x, y, z - 1, nxWithHalo, nyWithHalo)] -
                        2.0 * field[center]) * invDz2;
    return cxx + cyy + czz;
}

void chooseProcessGrid(const std::size_t globalX, const std::size_t globalY, const std::size_t globalZ,
                       const int processCount, int dims[3]) {
    // Pick the factorization with the smallest total face area per process.
    // This keeps communication low and also makes the decomposition follow
    // the physical aspect ratio of non-cubic grids.
    long double bestScore = std::numeric_limits<long double>::infinity();
    dims[0] = dims[1] = dims[2] = 0;

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<std::size_t>(px) > globalX) {
            continue;
        }
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<std::size_t>(py) > globalY) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<std::size_t>(pz) > globalZ) {
                continue;
            }

            const long double score =
                (static_cast<long double>(globalY) * globalZ) / px +
                (static_cast<long double>(globalX) * globalZ) / py +
                (static_cast<long double>(globalX) * globalY) / pz;
            if (score < bestScore) {
                bestScore = score;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
}

void localExtent(const std::size_t globalExtent, const int parts, const int coordinate,
                 std::size_t& offset, std::size_t& extent) {
    const std::size_t base = globalExtent / static_cast<std::size_t>(parts);
    const std::size_t remainder = globalExtent % static_cast<std::size_t>(parts);
    extent = base + (static_cast<std::size_t>(coordinate) < remainder ? 1 : 0);
    offset = static_cast<std::size_t>(coordinate) * base +
             std::min(static_cast<std::size_t>(coordinate), remainder);
}

Domain makeDomain(const std::size_t globalX, const std::size_t globalY, const std::size_t globalZ,
                  const int worldSize, const int worldRank) {
    Domain domain;
    domain.globalX = globalX;
    domain.globalY = globalY;
    domain.globalZ = globalZ;
    domain.size = worldSize;
    domain.rank = worldRank;

    chooseProcessGrid(globalX, globalY, globalZ, worldSize, domain.dims);
    if (domain.dims[0] == 0) {
        return domain;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, domain.dims, periods, 0, &domain.cart);
    MPI_Cart_coords(domain.cart, worldRank, 3, domain.coords);

    MPI_Cart_shift(domain.cart, 0, 1, &domain.neighbors[XMinus], &domain.neighbors[XPlus]);
    MPI_Cart_shift(domain.cart, 1, 1, &domain.neighbors[YMinus], &domain.neighbors[YPlus]);
    MPI_Cart_shift(domain.cart, 2, 1, &domain.neighbors[ZMinus], &domain.neighbors[ZPlus]);

    localExtent(globalX, domain.dims[0], domain.coords[0], domain.offsetX, domain.localX);
    localExtent(globalY, domain.dims[1], domain.coords[1], domain.offsetY, domain.localY);
    localExtent(globalZ, domain.dims[2], domain.coords[2], domain.offsetZ, domain.localZ);
    return domain;
}

void packFaces(const std::vector<double>& field, const Domain& domain, HaloBuffers& buffers) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;

    std::size_t n = 0;
    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            buffers.send[XMinus][n] = field[idx3(1, y, z, nx, ny)];
            buffers.send[XPlus][n] = field[idx3(domain.localX, y, z, nx, ny)];
            ++n;
        }
    }

    n = 0;
    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        for (std::size_t x = 1; x <= domain.localX; ++x) {
            buffers.send[YMinus][n] = field[idx3(x, 1, z, nx, ny)];
            buffers.send[YPlus][n] = field[idx3(x, domain.localY, z, nx, ny)];
            ++n;
        }
    }

    n = 0;
    for (std::size_t y = 1; y <= domain.localY; ++y) {
        for (std::size_t x = 1; x <= domain.localX; ++x) {
            buffers.send[ZMinus][n] = field[idx3(x, y, 1, nx, ny)];
            buffers.send[ZPlus][n] = field[idx3(x, y, domain.localZ, nx, ny)];
            ++n;
        }
    }
}

void fillPhysicalBoundaryHalos(std::vector<double>& field, const Domain& domain) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;

    if (domain.neighbors[XMinus] == MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t y = 1; y <= domain.localY; ++y) {
                field[idx3(0, y, z, nx, ny)] = field[idx3(1, y, z, nx, ny)];
            }
        }
    }
    if (domain.neighbors[XPlus] == MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t y = 1; y <= domain.localY; ++y) {
                field[idx3(domain.localX + 1, y, z, nx, ny)] =
                    field[idx3(domain.localX, y, z, nx, ny)];
            }
        }
    }
    if (domain.neighbors[YMinus] == MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                field[idx3(x, 0, z, nx, ny)] = field[idx3(x, 1, z, nx, ny)];
            }
        }
    }
    if (domain.neighbors[YPlus] == MPI_PROC_NULL) {
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                field[idx3(x, domain.localY + 1, z, nx, ny)] =
                    field[idx3(x, domain.localY, z, nx, ny)];
            }
        }
    }
    if (domain.neighbors[ZMinus] == MPI_PROC_NULL) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                field[idx3(x, y, 0, nx, ny)] = field[idx3(x, y, 1, nx, ny)];
            }
        }
    }
    if (domain.neighbors[ZPlus] == MPI_PROC_NULL) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                field[idx3(x, y, domain.localZ + 1, nx, ny)] =
                    field[idx3(x, y, domain.localZ, nx, ny)];
            }
        }
    }
}

void unpackFaces(std::vector<double>& field, const Domain& domain, const HaloBuffers& buffers) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;

    if (domain.neighbors[XMinus] != MPI_PROC_NULL || domain.neighbors[XPlus] != MPI_PROC_NULL) {
        std::size_t n = 0;
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t y = 1; y <= domain.localY; ++y) {
                if (domain.neighbors[XMinus] != MPI_PROC_NULL) {
                    field[idx3(0, y, z, nx, ny)] = buffers.receive[XMinus][n];
                }
                if (domain.neighbors[XPlus] != MPI_PROC_NULL) {
                    field[idx3(domain.localX + 1, y, z, nx, ny)] = buffers.receive[XPlus][n];
                }
                ++n;
            }
        }
    }

    if (domain.neighbors[YMinus] != MPI_PROC_NULL || domain.neighbors[YPlus] != MPI_PROC_NULL) {
        std::size_t n = 0;
        for (std::size_t z = 1; z <= domain.localZ; ++z) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                if (domain.neighbors[YMinus] != MPI_PROC_NULL) {
                    field[idx3(x, 0, z, nx, ny)] = buffers.receive[YMinus][n];
                }
                if (domain.neighbors[YPlus] != MPI_PROC_NULL) {
                    field[idx3(x, domain.localY + 1, z, nx, ny)] = buffers.receive[YPlus][n];
                }
                ++n;
            }
        }
    }

    if (domain.neighbors[ZMinus] != MPI_PROC_NULL || domain.neighbors[ZPlus] != MPI_PROC_NULL) {
        std::size_t n = 0;
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                if (domain.neighbors[ZMinus] != MPI_PROC_NULL) {
                    field[idx3(x, y, 0, nx, ny)] = buffers.receive[ZMinus][n];
                }
                if (domain.neighbors[ZPlus] != MPI_PROC_NULL) {
                    field[idx3(x, y, domain.localZ + 1, nx, ny)] = buffers.receive[ZPlus][n];
                }
                ++n;
            }
        }
    }
}

void exchangeHalos(std::vector<double>& field, const Domain& domain, HaloBuffers& buffers) {
    packFaces(field, domain, buffers);
    fillPhysicalBoundaryHalos(field, domain);

    if (buffers.activeRequestCount != 0) {
        MPI_Startall(buffers.activeRequestCount, buffers.requests.data());
        MPI_Waitall(buffers.activeRequestCount, buffers.requests.data(), MPI_STATUSES_IGNORE);
    }

    unpackFaces(field, domain, buffers);
}

void initializeHaloRequests(HaloBuffers& buffers, const Domain& domain) {
    // A receive for a minus face matches the plus-face send of the neighbor,
    // and vice versa. Persistent requests avoid rebuilding the MPI request
    // graph on every timestep.
    const int receiveFace[6] = {XPlus, XMinus, YPlus, YMinus, ZPlus, ZMinus};
    for (int face = 0; face < 6; ++face) {
        if (domain.neighbors[face] != MPI_PROC_NULL) {
            MPI_Recv_init(buffers.receive[face].data(), static_cast<int>(buffers.receive[face].size()),
                          MPI_DOUBLE, domain.neighbors[face], receiveFace[face], domain.cart,
                          &buffers.requests[buffers.activeRequestCount++]);
        }
    }
    for (int face = 0; face < 6; ++face) {
        if (domain.neighbors[face] != MPI_PROC_NULL) {
            MPI_Send_init(buffers.send[face].data(), static_cast<int>(buffers.send[face].size()),
                          MPI_DOUBLE, domain.neighbors[face], face, domain.cart,
                          &buffers.requests[buffers.activeRequestCount++]);
        }
    }
}

void destroyHaloRequests(HaloBuffers& buffers) {
    for (int request = 0; request < buffers.activeRequestCount; ++request) {
        MPI_Request_free(&buffers.requests[request]);
    }
    buffers.activeRequestCount = 0;
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const Domain& domain, const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                const std::size_t index = idx3(x, y, z, nx, ny);
                const double cv = c[index];
                mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                            3.0 * cv + cv * cv * cv -
                            gamma * computeLaplacian(c, nx, ny, x, y, z,
                                                     invDx2, invDy2, invDz2);
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, const Domain& domain,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                const std::size_t index = idx3(x, y, z, nx, ny);
                cnew[index] = cold[index] + dt * D *
                              computeLaplacian(mu, nx, ny, x, y, z,
                                               invDx2, invDy2, invDz2);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const Domain& domain) {
    const std::size_t volume = domain.globalX * domain.globalY * domain.globalZ;
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;

    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        const std::size_t globalZ = domain.offsetZ + z - 1;
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            const std::size_t globalY = domain.offsetY + y - 1;
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                const std::size_t globalX = domain.offsetX + x - 1;
                const std::size_t linearId =
                    globalZ * (domain.globalX * domain.globalY) + globalY * domain.globalX + globalX;
                const std::size_t pseudoNumerator = ((linearId + 1) * 1299709) % volume;
                const double pseudo = pseudoNumerator / static_cast<double>(volume);
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const std::size_t nx,
                    [[maybe_unused]] const std::size_t ny, [[maybe_unused]] const std::size_t nz) {
    if (c.empty()) {
        printf("Validation failed: empty concentration field\n");
        return false;
    }

    // Check for NaN or Inf.
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minVal = c[0];
    double maxVal = c[0];
    for (const double value : c) {
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void packInterior(const std::vector<double>& local, const Domain& domain, std::vector<double>& packed) {
    const std::size_t nx = domain.localX + 2;
    const std::size_t ny = domain.localY + 2;
    std::size_t n = 0;
    for (std::size_t z = 1; z <= domain.localZ; ++z) {
        for (std::size_t y = 1; y <= domain.localY; ++y) {
            for (std::size_t x = 1; x <= domain.localX; ++x) {
                packed[n++] = local[idx3(x, y, z, nx, ny)];
            }
        }
    }
}

void unpackGlobalBlock(std::vector<double>& global, const std::vector<double>& packed,
                       const std::size_t globalX, const std::size_t globalY,
                       const unsigned long long* metadata) {
    const std::size_t offsetX = static_cast<std::size_t>(metadata[0]);
    const std::size_t offsetY = static_cast<std::size_t>(metadata[1]);
    const std::size_t offsetZ = static_cast<std::size_t>(metadata[2]);
    const std::size_t localX = static_cast<std::size_t>(metadata[3]);
    const std::size_t localY = static_cast<std::size_t>(metadata[4]);
    const std::size_t localZ = static_cast<std::size_t>(metadata[5]);

    std::size_t n = 0;
    for (std::size_t z = 0; z < localZ; ++z) {
        for (std::size_t y = 0; y < localY; ++y) {
            for (std::size_t x = 0; x < localX; ++x) {
                global[idx3(offsetX + x, offsetY + y, offsetZ + z, globalX, globalY)] = packed[n++];
            }
        }
    }
}

// Gather only for result reporting/validation.  The time-stepping path never
// materializes the global field on one rank.
std::vector<double> gatherGlobalField(const std::vector<double>& local, const Domain& domain) {
    const std::size_t localVolume = domain.localX * domain.localY * domain.localZ;
    std::vector<double> packed(localVolume);
    packInterior(local, domain, packed);

    const unsigned long long localMetadata[6] = {
        static_cast<unsigned long long>(domain.offsetX),
        static_cast<unsigned long long>(domain.offsetY),
        static_cast<unsigned long long>(domain.offsetZ),
        static_cast<unsigned long long>(domain.localX),
        static_cast<unsigned long long>(domain.localY),
        static_cast<unsigned long long>(domain.localZ)};
    std::vector<unsigned long long> allMetadata;
    if (domain.rank == 0) {
        allMetadata.resize(static_cast<std::size_t>(domain.size) * 6);
    }
    MPI_Gather(localMetadata, 6, MPI_UNSIGNED_LONG_LONG,
               domain.rank == 0 ? allMetadata.data() : nullptr, 6, MPI_UNSIGNED_LONG_LONG,
               0, domain.cart);

    std::vector<double> global;
    if (domain.rank == 0) {
        global.resize(domain.globalX * domain.globalY * domain.globalZ);
        unpackGlobalBlock(global, packed, domain.globalX, domain.globalY, allMetadata.data());
        std::vector<double> receiveBuffer;
        for (int rank = 1; rank < domain.size; ++rank) {
            const unsigned long long* metadata = allMetadata.data() + static_cast<std::size_t>(rank) * 6;
            const std::size_t rankVolume = static_cast<std::size_t>(metadata[3]) *
                                            static_cast<std::size_t>(metadata[4]) *
                                            static_cast<std::size_t>(metadata[5]);
            receiveBuffer.resize(rankVolume);
            MPI_Recv(receiveBuffer.data(), static_cast<int>(rankVolume), MPI_DOUBLE,
                     rank, 9100, domain.cart, MPI_STATUS_IGNORE);
            unpackGlobalBlock(global, receiveBuffer, domain.globalX, domain.globalY, metadata);
        }
    } else {
        MPI_Send(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE,
                 0, 9100, domain.cart);
    }
    return global;
}

bool parseSizeOption(const char* text, std::size_t& value) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool parseIterationsOption(const char* text, int& value) {
    char* end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::size_t globalX = 64;
    std::size_t globalY = 0;
    std::size_t globalZ = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool parseOk = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseOk = parseSizeOption(argv[++i], globalX) && parseOk;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseOk = parseSizeOption(argv[++i], globalY) && parseOk;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseOk = parseSizeOption(argv[++i], globalZ) && parseOk;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            parseOk = parseIterationsOption(argv[++i], iterations) && parseOk;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseOk = false;
            if (worldRank == 0) {
                printf("Unknown option or missing option value: %s\n", argv[i]);
            }
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }
    if (globalY == 0) {
        globalY = globalX;
    }
    if (globalZ == 0) {
        globalZ = globalX;
    }
    if (!parseOk) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const bool multiplicationOverflow =
        globalX != 0 && globalY > std::numeric_limits<std::size_t>::max() / globalX;
    const std::size_t xy = multiplicationOverflow ? 0 : globalX * globalY;
    const bool volumeOverflow = xy != 0 && globalZ > std::numeric_limits<std::size_t>::max() / xy;
    const std::size_t globalVolume = volumeOverflow ? 0 : xy * globalZ;
    if (globalX > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        globalY > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        globalZ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        multiplicationOverflow || volumeOverflow || globalVolume == 0) {
        if (worldRank == 0) {
            printf("Invalid grid size\n");
        }
        MPI_Finalize();
        return 1;
    }

    Domain domain = makeDomain(globalX, globalY, globalZ, worldSize, worldRank);
    if (domain.dims[0] == 0 || domain.cart == MPI_COMM_NULL ||
        domain.localX == 0 || domain.localY == 0 || domain.localZ == 0) {
        if (worldRank == 0) {
            printf("Unable to distribute %zu x %zu x %zu over %d MPI ranks; "
                   "use no more ranks than a valid 3D process-grid factorization.\n",
                   globalX, globalY, globalZ, worldSize);
        }
        if (domain.cart != MPI_COMM_NULL) {
            MPI_Comm_free(&domain.cart);
        }
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("MPI ranks: %d (process grid: %d x %d x %d)\n",
               worldSize, domain.dims[0], domain.dims[1], domain.dims[2]);
        printf("Grid size: %zu x %zu x %zu\n", globalX, globalY, globalZ);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters.
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const std::size_t localVolume = domain.localX * domain.localY * domain.localZ;
    const std::size_t localArraySize =
        (domain.localX + 2) * (domain.localY + 2) * (domain.localZ + 2);
    if (localVolume > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            printf("Each MPI rank must own fewer than INT_MAX cells for MPI result transfers\n");
        }
        MPI_Comm_free(&domain.cart);
        MPI_Finalize();
        return 1;
    }

    std::vector<double> cold(localArraySize);
    std::vector<double> cnew(localArraySize);
    std::vector<double> mu(localArraySize);
    HaloBuffers halos(domain.localX, domain.localY, domain.localZ);
    initializeHaloRequests(halos, domain);

    if (worldRank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, domain);

    if (worldRank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(domain.cart);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, domain, halos);
        computeChemicalPotential(cold, mu, domain, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, domain, halos);
        cahnHilliardUpdate(cnew, cold, mu, domain, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.cart);

    if (worldRank == 0) {
        const long long durationMs = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
        const double cellUpdates = static_cast<double>(globalVolume) * iterations;
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> globalResult;
    if (printResults || validate) {
        globalResult = gatherGlobalField(cold, domain);
    }

    if (worldRank == 0 && printResults) {
        print_results(globalResult, "Concentration");
    }

    int validationStatus = 1;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            validationStatus = validateResult(globalResult, globalX, globalY, globalZ) ? 0 : 1;
            printf("Validation: %s\n", validationStatus == 0 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, domain.cart);
    }

    destroyHaloRequests(halos);
    MPI_Comm_free(&domain.cart);
    MPI_Finalize();
    return validate ? validationStatus : 0;
}
