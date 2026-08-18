#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// The global field is split into rectangular blocks.  Each block has one
// layer of ghost cells in every direction.  The x coordinate is contiguous,
// which makes the local stencil and the z-face messages cache friendly.
struct Decomposition {
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0;
    int size = 0;
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    std::array<int, 6> neighbors{};

    size_t nx = 0;
    size_t ny = 0;
    size_t nz = 0;
    size_t x0 = 0;
    size_t y0 = 0;
    size_t z0 = 0;
    size_t lx = 0;
    size_t ly = 0;
    size_t lz = 0;
    size_t ex = 0;
    size_t ey = 0;
    size_t ez = 0;

    inline size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return (z * ey + y) * ex + x;
    }

    inline size_t localCellCount() const noexcept {
        return lx * ly * lz;
    }
};

enum Face : int {
    XMinus = 0,
    XPlus = 1,
    YMinus = 2,
    YPlus = 3,
    ZMinus = 4,
    ZPlus = 5
};

size_t checkedProduct(const size_t a, const size_t b, const size_t c) {
    if (a == 0 || b == 0 || c == 0 || a > std::numeric_limits<size_t>::max() / b ||
        a * b > std::numeric_limits<size_t>::max() / c) {
        return 0;
    }
    return a * b * c;
}

// Return a process grid that respects the physical grid dimensions.  A
// plain MPI_Dims_create can assign more ranks to an axis than there are cells
// on small/anisotropic inputs, so the factorization is constrained here.  The
// selected grid minimizes surface/volume ratio, i.e. communication per cell.
std::array<int, 3> chooseProcessGrid(const int processCount, const size_t nx, const size_t ny,
                                     const size_t nz) {
    std::array<int, 3> best{0, 0, 0};
    long double bestCost = std::numeric_limits<long double>::max();

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<size_t>(px) > nx) {
            continue;
        }
        const int remaining = processCount / px;
        for (int py = 1; py <= remaining; ++py) {
            if (remaining % py != 0 || static_cast<size_t>(py) > ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > nz) {
                continue;
            }

            const size_t blockX = (nx + static_cast<size_t>(px) - 1) / static_cast<size_t>(px);
            const size_t blockY = (ny + static_cast<size_t>(py) - 1) / static_cast<size_t>(py);
            const size_t blockZ = (nz + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz);
            const long double volume = static_cast<long double>(blockX) * blockY * blockZ;
            const long double surface = 2.0L *
                (static_cast<long double>(blockY) * blockZ +
                 static_cast<long double>(blockX) * blockZ +
                 static_cast<long double>(blockX) * blockY);
            const long double cost = surface / volume;

            if (cost < bestCost) {
                bestCost = cost;
                best = {px, py, pz};
            }
        }
    }

    return best;
}

void splitDimension(const size_t globalSize, const int parts, const int coordinate,
                    size_t& start, size_t& count) {
    const size_t base = globalSize / static_cast<size_t>(parts);
    const size_t remainder = globalSize % static_cast<size_t>(parts);
    count = base + (static_cast<size_t>(coordinate) < remainder ? 1 : 0);
    start = static_cast<size_t>(coordinate) * base +
            std::min(static_cast<size_t>(coordinate), remainder);
}

Decomposition makeDecomposition(MPI_Comm activeComm, const size_t nx, const size_t ny,
                                const size_t nz, const int activeSize) {
    Decomposition d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.dims = chooseProcessGrid(activeSize, nx, ny, nz);

    const int periods[3] = {0, 0, 0};
    MPI_Cart_create(activeComm, 3, d.dims.data(), periods, 0, &d.cart);
    MPI_Comm_rank(d.cart, &d.rank);
    MPI_Comm_size(d.cart, &d.size);

    int coordinates[3] = {0, 0, 0};
    MPI_Cart_coords(d.cart, d.rank, 3, coordinates);
    d.coords = {coordinates[0], coordinates[1], coordinates[2]};

    splitDimension(nx, d.dims[0], d.coords[0], d.x0, d.lx);
    splitDimension(ny, d.dims[1], d.coords[1], d.y0, d.ly);
    splitDimension(nz, d.dims[2], d.coords[2], d.z0, d.lz);
    d.ex = d.lx + 2;
    d.ey = d.ly + 2;
    d.ez = d.lz + 2;

    int xMinus = MPI_PROC_NULL;
    int xPlus = MPI_PROC_NULL;
    int yMinus = MPI_PROC_NULL;
    int yPlus = MPI_PROC_NULL;
    int zMinus = MPI_PROC_NULL;
    int zPlus = MPI_PROC_NULL;
    MPI_Cart_shift(d.cart, 0, 1, &xMinus, &xPlus);
    MPI_Cart_shift(d.cart, 1, 1, &yMinus, &yPlus);
    MPI_Cart_shift(d.cart, 2, 1, &zMinus, &zPlus);
    d.neighbors = {xMinus, xPlus, yMinus, yPlus, zMinus, zPlus};
    return d;
}

inline double computeLaplacian(const std::vector<double>& field, const Decomposition& d,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) {
    const size_t center = d.index(x, y, z);
    const double cxx = (field[center + 1] + field[center - 1] - 2.0 * field[center]) /
                       (dx * dx);
    const double cyy = (field[center + d.ex] + field[center - d.ex] - 2.0 * field[center]) /
                       (dy * dy);
    const double czz = (field[center + d.ex * d.ey] + field[center - d.ex * d.ey] -
                        2.0 * field[center]) / (dz * dz);
    return cxx + cyy + czz;
}

inline void computeChemicalPotentialCell(const std::vector<double>& c, std::vector<double>& mu,
                                         const Decomposition& d, const double dx, const double dy,
                                         const double dz, const double gamma, const double e_AA,
                                         const double e_BB, const double e_AB, const size_t x,
                                         const size_t y, const size_t z) {
    const size_t index = d.index(x, y, z);
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                3.0 * cv + cv * cv * cv -
                gamma * computeLaplacian(c, d, dx, dy, dz, x, y, z);
}

void computeChemicalPotentialRegion(const std::vector<double>& c, std::vector<double>& mu,
                                    const Decomposition& d, const double dx, const double dy,
                                    const double dz, const double gamma, const double e_AA,
                                    const double e_BB, const double e_AB, const bool interior) {
    if (interior) {
        for (size_t z = 2; z < d.lz; ++z) {
            for (size_t y = 2; y < d.ly; ++y) {
                for (size_t x = 2; x < d.lx; ++x) {
                    computeChemicalPotentialCell(c, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                                 x, y, z);
                }
            }
        }
        return;
    }

    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            if (z == 1 || z == d.lz || y == 1 || y == d.ly) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    computeChemicalPotentialCell(c, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                                 x, y, z);
                }
            } else {
                computeChemicalPotentialCell(c, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                             1, y, z);
                if (d.lx > 1) {
                    computeChemicalPotentialCell(c, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                                 d.lx, y, z);
                }
            }
        }
    }
}

inline void cahnHilliardUpdateCell(std::vector<double>& cnew, const std::vector<double>& cold,
                                   const std::vector<double>& mu, const Decomposition& d,
                                   const double D, const double dt, const double dx,
                                   const double dy, const double dz, const size_t x,
                                   const size_t y, const size_t z) {
    const size_t index = d.index(x, y, z);
    cnew[index] = cold[index] + dt * D * computeLaplacian(mu, d, dx, dy, dz, x, y, z);
}

void cahnHilliardUpdateRegion(std::vector<double>& cnew, const std::vector<double>& cold,
                              const std::vector<double>& mu, const Decomposition& d,
                              const double D, const double dt, const double dx, const double dy,
                              const double dz, const bool interior) {
    if (interior) {
        for (size_t z = 2; z < d.lz; ++z) {
            for (size_t y = 2; y < d.ly; ++y) {
                for (size_t x = 2; x < d.lx; ++x) {
                    cahnHilliardUpdateCell(cnew, cold, mu, d, D, dt, dx, dy, dz, x, y, z);
                }
            }
        }
        return;
    }

    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            if (z == 1 || z == d.lz || y == 1 || y == d.ly) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    cahnHilliardUpdateCell(cnew, cold, mu, d, D, dt, dx, dy, dz, x, y, z);
                }
            } else {
                cahnHilliardUpdateCell(cnew, cold, mu, d, D, dt, dx, dy, dz, 1, y, z);
                if (d.lx > 1) {
                    cahnHilliardUpdateCell(cnew, cold, mu, d, D, dt, dx, dy, dz, d.lx, y, z);
                }
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const Decomposition& d) {
    const size_t xy = d.nx * d.ny;
    const size_t volume = d.nx * d.ny * d.nz;

    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t globalZ = d.z0 + z - 1;
        for (size_t y = 1; y <= d.ly; ++y) {
            const size_t globalY = d.y0 + y - 1;
            for (size_t x = 1; x <= d.lx; ++x) {
                const size_t globalX = d.x0 + x - 1;
                const size_t index = d.index(x, y, z);
                const size_t linearId = globalZ * xy + globalY * d.nx + globalX;
                const double pseudo = ((((linearId + 1) * 1299709) % volume) /
                                      static_cast<double>(volume));
                c[index] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Pack/unpack makes each MPI message contiguous.  This avoids repeatedly
// interpreting strided derived datatypes and gives the network a compact face
// for every exchange.  The packing is also done before the nonblocking sends,
// so the field arrays can be read by the overlapped stencil safely.
class HaloExchange {
public:
    explicit HaloExchange(const Decomposition& decomposition) : d(decomposition) {
        const std::array<size_t, 6> faceSizes{
            d.ly * d.lz, d.ly * d.lz, d.lx * d.lz, d.lx * d.lz, d.lx * d.ly, d.lx * d.ly};
        for (int face = 0; face < 6; ++face) {
            sendBuffers[face].resize(faceSizes[face]);
            receiveBuffers[face].resize(faceSizes[face]);
        }
    }

    void start(std::vector<double>& field) {
        requestCount = 0;
        for (int face = 0; face < 6; ++face) {
            if (d.neighbors[face] == MPI_PROC_NULL) {
                copyPhysicalBoundary(field, face);
                continue;
            }

            packFace(field, face);
            const int count = mpiCount(sendBuffers[face].size());
            const int tag = 100 + face / 2;
            MPI_Irecv(receiveBuffers[face].data(), count, MPI_DOUBLE, d.neighbors[face], tag,
                      d.cart, &requests[requestCount++]);
            MPI_Isend(sendBuffers[face].data(), count, MPI_DOUBLE, d.neighbors[face], tag,
                      d.cart, &requests[requestCount++]);
        }
    }

    void finish(std::vector<double>& field) {
        if (requestCount != 0) {
            MPI_Waitall(requestCount, requests.data(), MPI_STATUSES_IGNORE);
        }
        for (int face = 0; face < 6; ++face) {
            if (d.neighbors[face] != MPI_PROC_NULL) {
                unpackFace(field, face);
            }
        }
        requestCount = 0;
    }

private:
    const Decomposition& d;
    std::array<std::vector<double>, 6> sendBuffers;
    std::array<std::vector<double>, 6> receiveBuffers;
    std::array<MPI_Request, 12> requests{};
    int requestCount = 0;

    static int mpiCount(const size_t count) {
        if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            std::fprintf(stderr, "MPI face is too large for the MPI count type\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return static_cast<int>(count);
    }

    void packFace(const std::vector<double>& field, const int face) {
        size_t position = 0;
        if (face == XMinus || face == XPlus) {
            const size_t x = face == XMinus ? 1 : d.lx;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t y = 1; y <= d.ly; ++y) {
                    sendBuffers[face][position++] = field[d.index(x, y, z)];
                }
            }
        } else if (face == YMinus || face == YPlus) {
            const size_t y = face == YMinus ? 1 : d.ly;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    sendBuffers[face][position++] = field[d.index(x, y, z)];
                }
            }
        } else {
            const size_t z = face == ZMinus ? 1 : d.lz;
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    sendBuffers[face][position++] = field[d.index(x, y, z)];
                }
            }
        }
    }

    void unpackFace(std::vector<double>& field, const int face) {
        size_t position = 0;
        if (face == XMinus || face == XPlus) {
            const size_t x = face == XMinus ? 0 : d.lx + 1;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t y = 1; y <= d.ly; ++y) {
                    field[d.index(x, y, z)] = receiveBuffers[face][position++];
                }
            }
        } else if (face == YMinus || face == YPlus) {
            const size_t y = face == YMinus ? 0 : d.ly + 1;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    field[d.index(x, y, z)] = receiveBuffers[face][position++];
                }
            }
        } else {
            const size_t z = face == ZMinus ? 0 : d.lz + 1;
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    field[d.index(x, y, z)] = receiveBuffers[face][position++];
                }
            }
        }
    }

    void copyPhysicalBoundary(std::vector<double>& field, const int face) {
        if (face == XMinus || face == XPlus) {
            const size_t sourceX = face == XMinus ? 1 : d.lx;
            const size_t ghostX = face == XMinus ? 0 : d.lx + 1;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t y = 1; y <= d.ly; ++y) {
                    field[d.index(ghostX, y, z)] = field[d.index(sourceX, y, z)];
                }
            }
        } else if (face == YMinus || face == YPlus) {
            const size_t sourceY = face == YMinus ? 1 : d.ly;
            const size_t ghostY = face == YMinus ? 0 : d.ly + 1;
            for (size_t z = 1; z <= d.lz; ++z) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    field[d.index(x, ghostY, z)] = field[d.index(x, sourceY, z)];
                }
            }
        } else {
            const size_t sourceZ = face == ZMinus ? 1 : d.lz;
            const size_t ghostZ = face == ZMinus ? 0 : d.lz + 1;
            for (size_t y = 1; y <= d.ly; ++y) {
                for (size_t x = 1; x <= d.lx; ++x) {
                    field[d.index(x, y, ghostZ)] = field[d.index(x, y, sourceZ)];
                }
            }
        }
    }
};

bool validateResult(const std::vector<double>& c, const Decomposition& d) {
    int localInvalid = 0;
    double localMin = std::numeric_limits<double>::max();
    double localMax = std::numeric_limits<double>::lowest();
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            for (size_t x = 1; x <= d.lx; ++x) {
                const double value = c[d.index(x, y, z)];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = 1;
                }
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int invalid = 0;
    MPI_Allreduce(&localInvalid, &invalid, 1, MPI_INT, MPI_MAX, d.cart);
    if (invalid != 0) {
        if (d.rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double minValue = 0.0;
    double maxValue = 0.0;
    MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, d.cart);
    MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, d.cart);

    int valid = 1;
    if (d.rank == 0) {
        std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
        if (maxValue > 10.0 || minValue < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, d.cart);
    return valid != 0;
}

void packInterior(const std::vector<double>& field, const Decomposition& d,
                  std::vector<double>& packed) {
    packed.resize(d.localCellCount());
    size_t position = 0;
    for (size_t z = 1; z <= d.lz; ++z) {
        for (size_t y = 1; y <= d.ly; ++y) {
            for (size_t x = 1; x <= d.lx; ++x) {
                packed[position++] = field[d.index(x, y, z)];
            }
        }
    }
}

void gatherAndPrintResults(const std::vector<double>& field, const Decomposition& d,
                           const char* name) {
    std::vector<double> packed;
    packInterior(field, d, packed);
    const int localCount = static_cast<int>(packed.size());

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<unsigned long long> metadata;
    if (d.rank == 0) {
        counts.resize(static_cast<size_t>(d.size));
        displacements.resize(static_cast<size_t>(d.size));
        metadata.resize(static_cast<size_t>(d.size) * 6);
    }

    MPI_Gather(&localCount, 1, MPI_INT, d.rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0,
               d.cart);

    const unsigned long long localMetadata[6] = {
        static_cast<unsigned long long>(d.x0), static_cast<unsigned long long>(d.y0),
        static_cast<unsigned long long>(d.z0), static_cast<unsigned long long>(d.lx),
        static_cast<unsigned long long>(d.ly), static_cast<unsigned long long>(d.lz)};
    MPI_Gather(localMetadata, 6, MPI_UNSIGNED_LONG_LONG,
               d.rank == 0 ? metadata.data() : nullptr, 6, MPI_UNSIGNED_LONG_LONG, 0, d.cart);

    std::vector<double> received;
    if (d.rank == 0) {
        int displacement = 0;
        for (int rank = 0; rank < d.size; ++rank) {
            displacements[rank] = displacement;
            if (counts[rank] > std::numeric_limits<int>::max() - displacement) {
                std::fprintf(stderr, "Result gather is too large for MPI_Gatherv\n");
                MPI_Abort(d.cart, 1);
            }
            displacement += counts[rank];
        }
        received.resize(static_cast<size_t>(displacement));
    }

    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, d.rank == 0 ? received.data() : nullptr,
                d.rank == 0 ? counts.data() : nullptr, d.rank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, d.cart);

    if (d.rank == 0) {
        std::vector<double> global(d.nx * d.ny * d.nz);
        for (int rank = 0; rank < d.size; ++rank) {
            const auto* meta = metadata.data() + static_cast<size_t>(rank) * 6;
            const size_t x0 = static_cast<size_t>(meta[0]);
            const size_t y0 = static_cast<size_t>(meta[1]);
            const size_t z0 = static_cast<size_t>(meta[2]);
            const size_t lx = static_cast<size_t>(meta[3]);
            const size_t ly = static_cast<size_t>(meta[4]);
            const size_t lz = static_cast<size_t>(meta[5]);
            const double* source = received.data() + displacements[rank];
            size_t position = 0;
            for (size_t z = 0; z < lz; ++z) {
                for (size_t y = 0; y < ly; ++y) {
                    for (size_t x = 0; x < lx; ++x) {
                        const size_t globalIndex = (z0 + z) * (d.nx * d.ny) +
                                                   (y0 + y) * d.nx + x0 + x;
                        global[globalIndex] = source[position++];
                    }
                }
            }
        }
        print_results(global, name);
    }
}

void printUsage(const char* progName, const int rank) {
    if (rank != 0) {
        return;
    }
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseError = true;
        }
    }

    if (showHelp) {
        printUsage(argv[0], worldRank);
        MPI_Finalize();
        return 0;
    }
    if (parseError || nx == 0) {
        if (worldRank == 0 && !parseError) {
            std::printf("Grid dimensions must be positive\n");
            printUsage(argv[0], worldRank);
        }
        MPI_Finalize();
        return 1;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (ny == 0 || nz == 0) {
        if (worldRank == 0) {
            std::printf("Grid dimensions must be positive\n");
            printUsage(argv[0], worldRank);
        }
        MPI_Finalize();
        return 1;
    }

    const size_t gridSize = checkedProduct(nx, ny, nz);
    if (gridSize == 0) {
        if (worldRank == 0) {
            std::printf("Grid dimensions are too large\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Keep all ranks in MPI, including the one-process case.  If more ranks
    // than cells are launched, excess ranks are idle rather than creating
    // zero-sized subdomains or changing the numerical update.
    int activeCount = static_cast<int>(
        std::min(static_cast<size_t>(worldSize), gridSize));
    // A Cartesian process grid must factor the active rank count.  For
    // example, five ranks cannot tile a 2 x 3 x 1 domain with nonempty
    // rectangular blocks.  In that unusual case, use the largest valid
    // one-dimensional split and leave the remaining ranks idle.
    if (chooseProcessGrid(activeCount, nx, ny, nz)[0] == 0) {
        const size_t largestDimension = std::max(nx, std::max(ny, nz));
        activeCount = static_cast<int>(std::min(static_cast<size_t>(activeCount), largestDimension));
    }
    const bool active = worldRank < activeCount;
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);

    int returnCode = 0;
    if (active) {
        Decomposition d = makeDecomposition(activeComm, nx, ny, nz, activeCount);
        MPI_Comm_free(&activeComm);

        const size_t localStorageSize = d.ex * d.ey * d.ez;
        std::vector<double> cold(localStorageSize, 0.0);
        std::vector<double> cnew(localStorageSize, 0.0);
        std::vector<double> mu(localStorageSize, 0.0);

        if (d.rank == 0) {
            std::printf("Initializing concentration field...\n");
        }
        initializeConcentration(cold, d);

        if (d.rank == 0) {
            std::printf("Running Cahn-Hilliard simulation...\n");
        }

        const double dx = 1.0;
        const double dy = 1.0;
        const double dz = 1.0;
        const double dt = 0.01;
        const double e_AA = -(2.0 / 9.0);
        const double e_BB = -(2.0 / 9.0);
        const double e_AB = (2.0 / 9.0);
        const double gamma = 0.5;
        const double D = 1.0;

        HaloExchange halo(d);
        MPI_Barrier(d.cart);
        const double start = MPI_Wtime();

        for (int t = 0; t < iterations; ++t) {
            // The strict local interior does not depend on a remote face, so
            // it can run while the six c faces are transferred.
            halo.start(cold);
            computeChemicalPotentialRegion(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                           true);
            halo.finish(cold);
            computeChemicalPotentialRegion(cold, mu, d, dx, dy, dz, gamma, e_AA, e_BB, e_AB,
                                           false);

            // The same overlap pattern applies to the second stencil using
            // the freshly computed chemical-potential field.
            halo.start(mu);
            cahnHilliardUpdateRegion(cnew, cold, mu, d, D, dt, dx, dy, dz, true);
            halo.finish(mu);
            cahnHilliardUpdateRegion(cnew, cold, mu, d, D, dt, dx, dy, dz, false);
            std::swap(cold, cnew);
        }

        const double elapsed = MPI_Wtime() - start;
        double maximumElapsed = 0.0;
        MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, d.cart);

        if (d.rank == 0) {
            const long durationMs = static_cast<long>(maximumElapsed * 1000.0);
            std::printf("Computation time: %ld ms\n", durationMs);
            const double safeElapsed = maximumElapsed > 0.0 ? maximumElapsed :
                                       std::numeric_limits<double>::min();
            const double cellUpdates = static_cast<double>(gridSize) * iterations;
            const double mcups = cellUpdates / safeElapsed / 1e6;
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            gatherAndPrintResults(cold, d, "Concentration");
        }

        if (validate) {
            if (d.rank == 0) {
                std::printf("Validating result...\n");
            }
            if (!validateResult(cold, d)) {
                returnCode = 1;
            } else if (d.rank == 0) {
                std::printf("Validation: PASSED\n");
            }
        }

        MPI_Comm_free(&d.cart);
    }

    // Idle ranks do not participate in the Cartesian communicator, but all
    // MPI processes still rendezvous before finalization.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
