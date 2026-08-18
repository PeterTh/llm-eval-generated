#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Block {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t x0;
    size_t y0;
    size_t z0;
    size_t sx;
    size_t plane;

    [[nodiscard]] size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * plane + y * sx + x;
    }

};

struct Interval {
    size_t begin;
    size_t size;
};

[[nodiscard]] Interval splitInterval(const size_t globalSize, const int coordinate, const int parts) noexcept {
    const size_t partCount = static_cast<size_t>(parts);
    const size_t coord = static_cast<size_t>(coordinate);
    const size_t base = globalSize / partCount;
    const size_t remainder = globalSize % partCount;
    return {coord * base + std::min(coord, remainder), base + (coord < remainder ? 1U : 0U)};
}

[[nodiscard]] bool checkedMultiply(const size_t a, const size_t b, size_t& result) noexcept {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

[[nodiscard]] bool parseSize(const char* text, size_t& value) noexcept {
    if (text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

[[nodiscard]] bool parseIterations(const char* text, int& value) noexcept {
    if (text[0] == '\0' || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

// Select the Cartesian process grid with the fewest total inter-rank face
// cells. This accounts for non-cubic domains, unlike MPI_Dims_create.
[[nodiscard]] bool chooseProcessGrid(const int processCount, const size_t nx, const size_t ny,
                                     const size_t nz, std::array<int, 3>& best) noexcept {
    long double bestCommunication = std::numeric_limits<long double>::infinity();
    size_t bestLargestBlock = std::numeric_limits<size_t>::max();
    bool found = false;

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

            const long double communication =
                static_cast<long double>(px - 1) * static_cast<long double>(ny) * static_cast<long double>(nz) +
                static_cast<long double>(py - 1) * static_cast<long double>(nx) * static_cast<long double>(nz) +
                static_cast<long double>(pz - 1) * static_cast<long double>(nx) * static_cast<long double>(ny);
            const size_t largestBlock = ((nx + static_cast<size_t>(px) - 1) / static_cast<size_t>(px)) *
                                        ((ny + static_cast<size_t>(py) - 1) / static_cast<size_t>(py)) *
                                        ((nz + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz));

            if (!found || communication < bestCommunication ||
                (communication == bestCommunication && largestBlock < bestLargestBlock)) {
                best = {px, py, pz};
                bestCommunication = communication;
                bestLargestBlock = largestBlock;
                found = true;
            }
        }
    }
    return found;
}

class HaloExchange {
public:
    HaloExchange(const Block& block, const std::array<int, 3>& coordinates,
                 const std::array<int, 3>& processGrid, MPI_Comm communicator)
        : block_(block), coordinates_(coordinates), processGrid_(processGrid), communicator_(communicator) {
        const int sizes[3] = {static_cast<int>(block.nz + 2), static_cast<int>(block.ny + 2),
                              static_cast<int>(block.nx + 2)};
        const int starts[3] = {0, 0, 0};
        const int xFace[3] = {static_cast<int>(block.nz), static_cast<int>(block.ny), 1};
        const int yFace[3] = {static_cast<int>(block.nz), 1, static_cast<int>(block.nx)};
        const int zFace[3] = {1, static_cast<int>(block.ny), static_cast<int>(block.nx)};

        MPI_Type_create_subarray(3, sizes, xFace, starts, MPI_ORDER_C, MPI_DOUBLE, &faceTypes_[0]);
        MPI_Type_create_subarray(3, sizes, yFace, starts, MPI_ORDER_C, MPI_DOUBLE, &faceTypes_[1]);
        MPI_Type_create_subarray(3, sizes, zFace, starts, MPI_ORDER_C, MPI_DOUBLE, &faceTypes_[2]);
        for (MPI_Datatype& datatype : faceTypes_) {
            MPI_Type_commit(&datatype);
        }

        sendCounts_.fill(1);
        receiveCounts_.fill(1);
        sendTypes_ = {faceTypes_[0], faceTypes_[0], faceTypes_[1], faceTypes_[1], faceTypes_[2], faceTypes_[2]};
        receiveTypes_ = sendTypes_;

        setDisplacement(sendDisplacements_[0], block.index(1, 1, 1));
        setDisplacement(sendDisplacements_[1], block.index(block.nx, 1, 1));
        setDisplacement(sendDisplacements_[2], block.index(1, 1, 1));
        setDisplacement(sendDisplacements_[3], block.index(1, block.ny, 1));
        setDisplacement(sendDisplacements_[4], block.index(1, 1, 1));
        setDisplacement(sendDisplacements_[5], block.index(1, 1, block.nz));

        setDisplacement(receiveDisplacements_[0], block.index(0, 1, 1));
        setDisplacement(receiveDisplacements_[1], block.index(block.nx + 1, 1, 1));
        setDisplacement(receiveDisplacements_[2], block.index(1, 0, 1));
        setDisplacement(receiveDisplacements_[3], block.index(1, block.ny + 1, 1));
        setDisplacement(receiveDisplacements_[4], block.index(1, 1, 0));
        setDisplacement(receiveDisplacements_[5], block.index(1, 1, block.nz + 1));
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        for (MPI_Datatype& datatype : faceTypes_) {
            MPI_Type_free(&datatype);
        }
    }

    void start(double* field, MPI_Request& request) const noexcept {
        fillPhysicalHalos(field);
        MPI_Ineighbor_alltoallw(field, sendCounts_.data(), sendDisplacements_.data(), sendTypes_.data(), field,
                                receiveCounts_.data(), receiveDisplacements_.data(), receiveTypes_.data(),
                                communicator_, &request);
    }

private:
    static void setDisplacement(MPI_Aint& displacement, const size_t element) noexcept {
        displacement = static_cast<MPI_Aint>(element * sizeof(double));
    }

    void fillPhysicalHalos(double* field) const noexcept {
        if (coordinates_[0] == 0) {
            for (size_t z = 1; z <= block_.nz; ++z) {
                for (size_t y = 1; y <= block_.ny; ++y) {
                    field[block_.index(0, y, z)] = field[block_.index(1, y, z)];
                }
            }
        }
        if (coordinates_[0] + 1 == processGrid_[0]) {
            for (size_t z = 1; z <= block_.nz; ++z) {
                for (size_t y = 1; y <= block_.ny; ++y) {
                    field[block_.index(block_.nx + 1, y, z)] = field[block_.index(block_.nx, y, z)];
                }
            }
        }
        if (coordinates_[1] == 0) {
            for (size_t z = 1; z <= block_.nz; ++z) {
                for (size_t x = 1; x <= block_.nx; ++x) {
                    field[block_.index(x, 0, z)] = field[block_.index(x, 1, z)];
                }
            }
        }
        if (coordinates_[1] + 1 == processGrid_[1]) {
            for (size_t z = 1; z <= block_.nz; ++z) {
                for (size_t x = 1; x <= block_.nx; ++x) {
                    field[block_.index(x, block_.ny + 1, z)] = field[block_.index(x, block_.ny, z)];
                }
            }
        }
        if (coordinates_[2] == 0) {
            for (size_t y = 1; y <= block_.ny; ++y) {
                for (size_t x = 1; x <= block_.nx; ++x) {
                    field[block_.index(x, y, 0)] = field[block_.index(x, y, 1)];
                }
            }
        }
        if (coordinates_[2] + 1 == processGrid_[2]) {
            for (size_t y = 1; y <= block_.ny; ++y) {
                for (size_t x = 1; x <= block_.nx; ++x) {
                    field[block_.index(x, y, block_.nz + 1)] = field[block_.index(x, y, block_.nz)];
                }
            }
        }
    }

    const Block& block_;
    const std::array<int, 3>& coordinates_;
    const std::array<int, 3>& processGrid_;
    MPI_Comm communicator_;
    std::array<MPI_Datatype, 3> faceTypes_{};
    std::array<int, 6> sendCounts_{};
    std::array<int, 6> receiveCounts_{};
    std::array<MPI_Aint, 6> sendDisplacements_{};
    std::array<MPI_Aint, 6> receiveDisplacements_{};
    std::array<MPI_Datatype, 6> sendTypes_{};
    std::array<MPI_Datatype, 6> receiveTypes_{};
};

inline double laplacianAt(const double* field, const size_t index, const size_t rowStride,
                          const size_t planeStride, const double dx, const double dy, const double dz) noexcept {
    const double centerTwice = 2.0 * field[index];
    const double cxx = (field[index + 1] + field[index - 1] - centerTwice) / (dx * dx);
    const double cyy = (field[index + rowStride] + field[index - rowStride] - centerTwice) / (dy * dy);
    const double czz = (field[index + planeStride] + field[index - planeStride] - centerTwice) / (dz * dz);
    return cxx + cyy + czz;
}

void computeChemicalPotentialBox(const double* __restrict cold, double* __restrict mu, const Block& block,
                                 const size_t xBegin,
                                 const size_t xEnd, const size_t yBegin, const size_t yEnd,
                                 const size_t zBegin, const size_t zEnd, const double dx, const double dy,
                                 const double dz, const double gamma, const double eAA, const double eBB,
                                 const double eAB) noexcept {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            size_t index = block.index(xBegin, y, z);
            for (size_t x = xBegin; x < xEnd; ++x, ++index) {
                const double cv = cold[index];
                mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) + 3.0 * cv +
                            cv * cv * cv - gamma * laplacianAt(cold, index, block.sx, block.plane, dx, dy, dz);
            }
        }
    }
}

void updateBox(double* __restrict cnew, const double* __restrict cold, const double* __restrict mu,
               const Block& block, const size_t xBegin,
               const size_t xEnd, const size_t yBegin, const size_t yEnd, const size_t zBegin,
               const size_t zEnd, const double D, const double dt, const double dx, const double dy,
               const double dz) noexcept {
    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            size_t index = block.index(xBegin, y, z);
            for (size_t x = xBegin; x < xEnd; ++x, ++index) {
                cnew[index] = cold[index] + dt * D * laplacianAt(mu, index, block.sx, block.plane, dx, dy, dz);
            }
        }
    }
}

void computeChemicalPotentialCore(const double* cold, double* mu, const Block& block, const double dx,
                                  const double dy, const double dz, const double gamma, const double eAA,
                                  const double eBB, const double eAB) noexcept {
    if (block.nx > 2 && block.ny > 2 && block.nz > 2) {
        computeChemicalPotentialBox(cold, mu, block, 2, block.nx, 2, block.ny, 2, block.nz, dx, dy, dz,
                                    gamma, eAA, eBB, eAB);
    }
}

void updateCore(double* cnew, const double* cold, const double* mu, const Block& block, const double D,
                const double dt, const double dx, const double dy, const double dz) noexcept {
    if (block.nx > 2 && block.ny > 2 && block.nz > 2) {
        updateBox(cnew, cold, mu, block, 2, block.nx, 2, block.ny, 2, block.nz, D, dt, dx, dy, dz);
    }
}

void computeChemicalPotentialBoundary(const double* cold, double* mu, const Block& block, const double dx,
                                      const double dy, const double dz, const double gamma, const double eAA,
                                      const double eBB, const double eAB) noexcept {
    computeChemicalPotentialBox(cold, mu, block, 1, block.nx + 1, 1, block.ny + 1, 1, 2, dx, dy, dz,
                                gamma, eAA, eBB, eAB);
    if (block.nz > 1) {
        computeChemicalPotentialBox(cold, mu, block, 1, block.nx + 1, 1, block.ny + 1, block.nz,
                                    block.nz + 1, dx, dy, dz, gamma, eAA, eBB, eAB);
    }
    if (block.nz > 2) {
        computeChemicalPotentialBox(cold, mu, block, 1, block.nx + 1, 1, 2, 2, block.nz, dx, dy, dz,
                                    gamma, eAA, eBB, eAB);
        if (block.ny > 1) {
            computeChemicalPotentialBox(cold, mu, block, 1, block.nx + 1, block.ny, block.ny + 1, 2,
                                        block.nz, dx, dy, dz, gamma, eAA, eBB, eAB);
        }
        if (block.ny > 2) {
            computeChemicalPotentialBox(cold, mu, block, 1, 2, 2, block.ny, 2, block.nz, dx, dy, dz,
                                        gamma, eAA, eBB, eAB);
            if (block.nx > 1) {
                computeChemicalPotentialBox(cold, mu, block, block.nx, block.nx + 1, 2, block.ny, 2,
                                            block.nz, dx, dy, dz, gamma, eAA, eBB, eAB);
            }
        }
    }
}

void updateBoundary(double* cnew, const double* cold, const double* mu, const Block& block, const double D,
                    const double dt, const double dx, const double dy, const double dz) noexcept {
    updateBox(cnew, cold, mu, block, 1, block.nx + 1, 1, block.ny + 1, 1, 2, D, dt, dx, dy, dz);
    if (block.nz > 1) {
        updateBox(cnew, cold, mu, block, 1, block.nx + 1, 1, block.ny + 1, block.nz, block.nz + 1, D, dt,
                  dx, dy, dz);
    }
    if (block.nz > 2) {
        updateBox(cnew, cold, mu, block, 1, block.nx + 1, 1, 2, 2, block.nz, D, dt, dx, dy, dz);
        if (block.ny > 1) {
            updateBox(cnew, cold, mu, block, 1, block.nx + 1, block.ny, block.ny + 1, 2, block.nz, D, dt,
                      dx, dy, dz);
        }
        if (block.ny > 2) {
            updateBox(cnew, cold, mu, block, 1, 2, 2, block.ny, 2, block.nz, D, dt, dx, dy, dz);
            if (block.nx > 1) {
                updateBox(cnew, cold, mu, block, block.nx, block.nx + 1, 2, block.ny, 2, block.nz, D, dt,
                          dx, dy, dz);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& concentration, const Block& block, const size_t globalNx,
                             const size_t globalNy, const size_t globalVolume) noexcept {
    for (size_t z = 1; z <= block.nz; ++z) {
        const size_t globalZ = block.z0 + z - 1;
        for (size_t y = 1; y <= block.ny; ++y) {
            const size_t globalY = block.y0 + y - 1;
            for (size_t x = 1; x <= block.nx; ++x) {
                const size_t globalX = block.x0 + x - 1;
                const size_t linearId = globalZ * (globalNx * globalNy) + globalY * globalNx + globalX;
                const double pseudo = (((linearId + 1) * 1299709) % globalVolume) /
                                      static_cast<double>(globalVolume);
                concentration[block.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

[[nodiscard]] int validateResult(const std::vector<double>& concentration, const Block& block,
                                 const int cartRank, MPI_Comm communicator) noexcept {
    int localFinite = 1;
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= block.nz; ++z) {
        for (size_t y = 1; y <= block.ny; ++y) {
            for (size_t x = 1; x <= block.nx; ++x) {
                const double value = concentration[block.index(x, y, z)];
                if (!std::isfinite(value)) {
                    localFinite = 0;
                } else {
                    localMinimum = std::min(localMinimum, value);
                    localMaximum = std::max(localMaximum, value);
                }
            }
        }
    }

    int globalFinite = 0;
    double globalMinimum = 0.0;
    double globalMaximum = 0.0;
    MPI_Reduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_MIN, 0, communicator);
    MPI_Reduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN, 0, communicator);
    MPI_Reduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);

    int valid = 1;
    if (cartRank == 0) {
        if (!globalFinite) {
            std::printf("Validation failed: found NaN or Inf value\n");
            valid = 0;
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
            if (globalMaximum > 10.0 || globalMinimum < -10.0) {
                std::printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, communicator);
    return valid;
}

[[nodiscard]] MPI_Datatype makeOwnedDatatype(const Block& block) {
    const int sizes[3] = {static_cast<int>(block.nz + 2), static_cast<int>(block.ny + 2),
                          static_cast<int>(block.nx + 2)};
    const int subsizes[3] = {static_cast<int>(block.nz), static_cast<int>(block.ny),
                             static_cast<int>(block.nx)};
    const int starts[3] = {1, 1, 1};
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &datatype);
    MPI_Type_commit(&datatype);
    return datatype;
}

[[nodiscard]] std::vector<double> gatherGlobal(const std::vector<double>& local, const Block& block,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const std::array<int, 3>& processGrid, const int cartRank,
                                               const int processCount, MPI_Comm communicator) {
    constexpr int gatherTag = 101;
    MPI_Datatype ownedDatatype = makeOwnedDatatype(block);
    std::vector<double> global;

    if (cartRank != 0) {
        MPI_Send(local.data(), 1, ownedDatatype, 0, gatherTag, communicator);
        MPI_Type_free(&ownedDatatype);
        return global;
    }

    global.resize(nx * ny * nz);
    for (size_t z = 0; z < block.nz; ++z) {
        for (size_t y = 0; y < block.ny; ++y) {
            const size_t localOffset = block.index(1, y + 1, z + 1);
            const size_t globalOffset = (block.z0 + z) * (nx * ny) + (block.y0 + y) * nx + block.x0;
            std::copy_n(local.data() + localOffset, block.nx, global.data() + globalOffset);
        }
    }

    std::vector<MPI_Request> requests(static_cast<size_t>(processCount - 1), MPI_REQUEST_NULL);
    std::vector<MPI_Datatype> receiveTypes(static_cast<size_t>(processCount - 1), MPI_DATATYPE_NULL);
    const int globalSizes[3] = {static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx)};
    for (int source = 1; source < processCount; ++source) {
        std::array<int, 3> coordinates{};
        MPI_Cart_coords(communicator, source, 3, coordinates.data());
        const Interval x = splitInterval(nx, coordinates[0], processGrid[0]);
        const Interval y = splitInterval(ny, coordinates[1], processGrid[1]);
        const Interval z = splitInterval(nz, coordinates[2], processGrid[2]);
        const int subsizes[3] = {static_cast<int>(z.size), static_cast<int>(y.size), static_cast<int>(x.size)};
        const int starts[3] = {static_cast<int>(z.begin), static_cast<int>(y.begin), static_cast<int>(x.begin)};
        MPI_Type_create_subarray(3, globalSizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE,
                                 &receiveTypes[static_cast<size_t>(source - 1)]);
        MPI_Type_commit(&receiveTypes[static_cast<size_t>(source - 1)]);
        MPI_Irecv(global.data(), 1, receiveTypes[static_cast<size_t>(source - 1)], source, gatherTag, communicator,
                  &requests[static_cast<size_t>(source - 1)]);
    }
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
    for (MPI_Datatype& datatype : receiveTypes) {
        MPI_Type_free(&datatype);
    }
    MPI_Type_free(&ownedDatatype);
    return global;
}

void printUsage(const char* programName) noexcept {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

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
            argumentsValid = parseSize(argv[++i], nx) && argumentsValid;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], ny) && argumentsValid;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            argumentsValid = parseSize(argv[++i], nz) && argumentsValid;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            argumentsValid = parseIterations(argv[++i], iterations) && argumentsValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t xy = 0;
    size_t gridSize = 0;
    if (!checkedMultiply(nx, ny, xy) || !checkedMultiply(xy, nz, gridSize)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid dimensions overflow the addressable size.\n");
        }
        MPI_Finalize();
        return 1;
    }

    std::array<int, 3> processGrid{};
    int processGridFound = 0;
    if (worldRank == 0) {
        processGridFound = chooseProcessGrid(processCount, nx, ny, nz, processGrid) ? 1 : 0;
    }
    MPI_Bcast(&processGridFound, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(processGrid.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (!processGridFound) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "Cannot map %d MPI processes onto a %zu x %zu x %zu nonempty Cartesian grid.\n",
                         processCount, nx, ny, nz);
        }
        MPI_Finalize();
        return 1;
    }

    const int periods[3] = {0, 0, 0};
    MPI_Comm cartesian = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, processGrid.data(), periods, 1, &cartesian);
    int cartRank = 0;
    std::array<int, 3> coordinates{};
    MPI_Comm_rank(cartesian, &cartRank);
    MPI_Cart_coords(cartesian, cartRank, 3, coordinates.data());

    const Interval x = splitInterval(nx, coordinates[0], processGrid[0]);
    const Interval y = splitInterval(ny, coordinates[1], processGrid[1]);
    const Interval z = splitInterval(nz, coordinates[2], processGrid[2]);
    if (x.size + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        y.size + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        z.size + 2 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (printResults &&
         (nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
          ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
          nz > static_cast<size_t>(std::numeric_limits<int>::max())))) {
        if (cartRank == 0) {
            std::fprintf(stderr, "Grid dimensions exceed MPI datatype limits.\n");
        }
        MPI_Comm_free(&cartesian);
        MPI_Finalize();
        return 1;
    }

    const size_t localSx = x.size + 2;
    size_t localPlane = 0;
    size_t localStorage = 0;
    if (!checkedMultiply(localSx, y.size + 2, localPlane) ||
        !checkedMultiply(localPlane, z.size + 2, localStorage)) {
        if (cartRank == 0) {
            std::fprintf(stderr, "Local grid dimensions overflow the addressable size.\n");
        }
        MPI_Comm_free(&cartesian);
        MPI_Finalize();
        return 1;
    }
    const Block block{x.size, y.size, z.size, x.begin, y.begin, z.begin, localSx, localPlane};

    if (cartRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI processes: %d (%d x %d x %d)\n", processCount, processGrid[0], processGrid[1],
                    processGrid[2]);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = 2.0 / 9.0;
    const double gamma = 0.5;
    const double D = 1.0;

    std::vector<double> cold(localStorage);
    std::vector<double> cnew(localStorage);
    std::vector<double> mu(localStorage);

    if (cartRank == 0) {
        std::printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, block, nx, ny, gridSize);

    int exitCode = 0;
    {
        HaloExchange halo(block, coordinates, processGrid, cartesian);
        if (cartRank == 0) {
            std::printf("Running Cahn-Hilliard simulation...\n");
        }
        MPI_Barrier(cartesian);
        const double start = MPI_Wtime();

        for (int timeStep = 0; timeStep < iterations; ++timeStep) {
            MPI_Request request = MPI_REQUEST_NULL;
            halo.start(cold.data(), request);
            computeChemicalPotentialCore(cold.data(), mu.data(), block, dx, dy, dz, gamma, eAA, eBB, eAB);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            computeChemicalPotentialBoundary(cold.data(), mu.data(), block, dx, dy, dz, gamma, eAA, eBB, eAB);

            halo.start(mu.data(), request);
            updateCore(cnew.data(), cold.data(), mu.data(), block, D, dt, dx, dy, dz);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            updateBoundary(cnew.data(), cold.data(), mu.data(), block, D, dt, dx, dy, dz);
            std::swap(cold, cnew);
        }

        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian);
        if (cartRank == 0) {
            const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
            const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
            const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
            std::printf("Computation time: %lld ms\n", milliseconds);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            std::vector<double> global =
                gatherGlobal(cold, block, nx, ny, nz, processGrid, cartRank, processCount, cartesian);
            if (cartRank == 0) {
                print_results(global, "Concentration");
            }
        }

        if (validate) {
            if (cartRank == 0) {
                std::printf("Validating result...\n");
            }
            const int valid = validateResult(cold, block, cartRank, cartesian);
            if (cartRank == 0) {
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Comm_free(&cartesian);
    MPI_Finalize();
    return exitCode;
}
