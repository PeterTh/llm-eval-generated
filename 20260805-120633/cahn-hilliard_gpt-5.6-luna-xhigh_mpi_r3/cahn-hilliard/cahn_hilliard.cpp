#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation. Local fields include one ghost cell on every side.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct LocalDomain {
    size_t globalNx;
    size_t globalNy;
    size_t globalNz;
    size_t nx;
    size_t ny;
    size_t nz;
    size_t x0;
    size_t y0;
    size_t z0;
    size_t pitchX;
    size_t pitchY;
    size_t plane;
};

size_t localExtent(const size_t globalExtent, const int parts, const int coordinate) {
    const size_t base = globalExtent / static_cast<size_t>(parts);
    const size_t remainder = globalExtent % static_cast<size_t>(parts);
    return base + (static_cast<size_t>(coordinate) < remainder ? 1U : 0U);
}

size_t localOffset(const size_t globalExtent, const int parts, const int coordinate) {
    const size_t base = globalExtent / static_cast<size_t>(parts);
    const size_t remainder = globalExtent % static_cast<size_t>(parts);
    return static_cast<size_t>(coordinate) * base +
           std::min(static_cast<size_t>(coordinate), remainder);
}

// Pick a process grid that minimizes the communication surface/volume ratio.
// Each process owns at least one cell in every decomposed direction.
bool chooseProcessGrid(const int processCount, const size_t nx, const size_t ny,
                       const size_t nz, int dims[3]) {
    std::vector<int> divisors;
    for (int divisor = 1; divisor <= processCount / divisor; ++divisor) {
        if (processCount % divisor != 0) {
            continue;
        }
        divisors.push_back(divisor);
        if (divisor != processCount / divisor) {
            divisors.push_back(processCount / divisor);
        }
    }

    bool found = false;
    long double bestScore = std::numeric_limits<long double>::max();
    for (const int px : divisors) {
        if (static_cast<size_t>(px) > nx || processCount % px != 0) {
            continue;
        }
        const int remaining = processCount / px;
        for (const int py : divisors) {
            if (py <= 0 || remaining % py != 0 || static_cast<size_t>(py) > ny) {
                continue;
            }
            const int pz = remaining / py;
            if (static_cast<size_t>(pz) > nz) {
                continue;
            }

            // For a regular decomposition, this is proportional to the
            // communicated face area divided by the owned volume.
            const long double score =
                static_cast<long double>(px) / static_cast<long double>(nx) +
                static_cast<long double>(py) / static_cast<long double>(ny) +
                static_cast<long double>(pz) / static_cast<long double>(nz);
            if (!found || score < bestScore) {
                found = true;
                bestScore = score;
                dims[0] = px;
                dims[1] = py;
                dims[2] = pz;
            }
        }
    }
    return found;
}

// Find the largest usable active communicator. This also handles small grids
// where an arbitrary MPI process count cannot be factored into nonempty slabs.
bool chooseActiveProcessGrid(const int worldSize, const size_t nx, const size_t ny,
                             const size_t nz, int& activeSize, int dims[3]) {
    const size_t cellCount = nx * ny * nz;
    const size_t requested = std::min(static_cast<size_t>(worldSize), cellCount);
    for (size_t candidate = requested; candidate >= 1; --candidate) {
        if (candidate > static_cast<size_t>(INT_MAX)) {
            continue;
        }
        int candidateDims[3] = {0, 0, 0};
        if (chooseProcessGrid(static_cast<int>(candidate), nx, ny, nz, candidateDims)) {
            activeSize = static_cast<int>(candidate);
            dims[0] = candidateDims[0];
            dims[1] = candidateDims[1];
            dims[2] = candidateDims[2];
            return true;
        }
        if (candidate == 1) {
            break;
        }
    }
    return false;
}

inline double laplacianAt(const double* const values, const size_t pitchX,
                          const size_t pitchY, const size_t x, const size_t y,
                          const size_t z, const double dx, const double dy,
                          const double dz) noexcept {
    const size_t plane = pitchX * pitchY;
    const size_t center = idx3(x, y, z, pitchX, pitchY);
    const double cv = values[center];

    // Keep the same stencil and operation grouping as the original serial
    // implementation. Ghost cells provide the clamped boundary values.
    const double cxx = (values[center + 1] + values[center - 1] -
                        2.0 * cv) / (dx * dx);
    const double cyy = (values[center + pitchX] + values[center - pitchX] -
                        2.0 * cv) / (dy * dy);
    const double czz = (values[center + plane] + values[center - plane] -
                        2.0 * cv) / (dz * dz);
    return cxx + cyy + czz;
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const LocalDomain& domain, const double dx,
                              const double dy, const double dz, const double gamma,
                              const double e_AA, const double e_BB,
                              const double e_AB) {
    const double* const concentration = c.data();
    double* const chemicalPotential = mu.data();
    const size_t pitchX = domain.pitchX;
    const size_t pitchY = domain.pitchY;

    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t row = idx3(0, y, z, pitchX, pitchY);
            for (size_t x = 1; x <= domain.nx; ++x) {
                const size_t index = row + x;
                const double cv = concentration[index];
                const double laplacian = laplacianAt(concentration, pitchX, pitchY,
                                                     x, y, z, dx, dy, dz);
                chemicalPotential[index] =
                    4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                           2.0 * cv * e_AB) +
                    3.0 * cv + cv * cv * cv - gamma * laplacian;
            }
        }
    }
}

void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu, const LocalDomain& domain,
                        const double D, const double dt, const double dx,
                        const double dy, const double dz) {
    const double* const oldConcentration = cold.data();
    const double* const chemicalPotential = mu.data();
    double* const newConcentration = cnew.data();
    const size_t pitchX = domain.pitchX;
    const size_t pitchY = domain.pitchY;

    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t row = idx3(0, y, z, pitchX, pitchY);
            for (size_t x = 1; x <= domain.nx; ++x) {
                const size_t index = row + x;
                newConcentration[index] =
                    oldConcentration[index] +
                    dt * D * laplacianAt(chemicalPotential, pitchX, pitchY,
                                         x, y, z, dx, dy, dz);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const LocalDomain& domain) {
    const size_t globalPlane = domain.globalNx * domain.globalNy;
    const size_t volume = domain.globalNx * globalPlane * domain.globalNz;
    double* const concentration = c.data();

    for (size_t z = 1; z <= domain.nz; ++z) {
        const size_t globalZ = domain.z0 + z - 1;
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t globalY = domain.y0 + y - 1;
            const size_t row = idx3(0, y, z, domain.pitchX, domain.pitchY);
            for (size_t x = 1; x <= domain.nx; ++x) {
                const size_t globalX = domain.x0 + x - 1;
                const size_t linearId = globalZ * globalPlane + globalY * domain.globalNx +
                                        globalX;
                // Generate the same deterministic pseudo-random value in [-1, 1]
                // as the serial benchmark, using the global cell id.
                const size_t pseudoId = ((linearId + 1) * 1299709) % volume;
                const double pseudo = pseudoId / static_cast<double>(volume);
                concentration[row + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

class HaloExchange {
public:
    HaloExchange(const MPI_Comm communicator, const LocalDomain& domain)
        : comm_(communicator), domain_(domain), xMinus_(MPI_PROC_NULL),
          xPlus_(MPI_PROC_NULL), yMinus_(MPI_PROC_NULL), yPlus_(MPI_PROC_NULL),
          zMinus_(MPI_PROC_NULL), zPlus_(MPI_PROC_NULL), xFace_(MPI_DATATYPE_NULL),
          yFace_(MPI_DATATYPE_NULL), zFace_(MPI_DATATYPE_NULL) {
        MPI_Cart_shift(comm_, 0, 1, &xMinus_, &xPlus_);
        MPI_Cart_shift(comm_, 1, 1, &yMinus_, &yPlus_);
        MPI_Cart_shift(comm_, 2, 1, &zMinus_, &zPlus_);

        const int xRowCount = mpiCount(domain_.ny, "x halo row");
        const int xPlaneCount = mpiCount(domain_.nz, "x halo plane");
        const int yFaceCount = mpiCount(domain_.nz, "y halo");
        const int yFaceWidth = mpiCount(domain_.nx, "y halo row");
        const int zFaceRows = mpiCount(domain_.ny, "z halo row");
        const int zFaceWidth = mpiCount(domain_.nx, "z halo row width");
        const MPI_Aint rowStride = static_cast<MPI_Aint>(domain_.pitchX) *
                                   static_cast<MPI_Aint>(sizeof(double));
        const MPI_Aint planeStride = static_cast<MPI_Aint>(domain_.plane) *
                                     static_cast<MPI_Aint>(sizeof(double));

        // X faces are strided in Y and have an additional gap between Z
        // planes because each local plane contains two ghost rows.
        mpiCheck(MPI_Type_create_hvector(xRowCount, 1, rowStride, MPI_DOUBLE,
                                         &xRowFace_),
                 "MPI_Type_create_hvector(x row)");
        mpiCheck(MPI_Type_commit(&xRowFace_), "MPI_Type_commit(x row)");
        mpiCheck(MPI_Type_create_hvector(xPlaneCount, 1, planeStride, xRowFace_,
                                         &xFace_),
                 "MPI_Type_create_hvector(x)");
        mpiCheck(MPI_Type_commit(&xFace_), "MPI_Type_commit(x)");
        mpiCheck(MPI_Type_create_hvector(yFaceCount, yFaceWidth, planeStride,
                                         MPI_DOUBLE, &yFace_),
                 "MPI_Type_create_hvector(y)");
        mpiCheck(MPI_Type_commit(&yFace_), "MPI_Type_commit(y)");
        // Z faces are also noncontiguous because every owned Y row is
        // followed by an X-direction ghost row.
        mpiCheck(MPI_Type_create_hvector(zFaceRows, zFaceWidth, rowStride,
                                         MPI_DOUBLE, &zFace_),
                 "MPI_Type_create_hvector(z)");
        mpiCheck(MPI_Type_commit(&zFace_), "MPI_Type_commit(z)");
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        if (xFace_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&xFace_);
        }
        if (yFace_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&yFace_);
        }
        if (zFace_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&zFace_);
        }
        if (xRowFace_ != MPI_DATATYPE_NULL) {
            MPI_Type_free(&xRowFace_);
        }
    }

    void exchange(std::vector<double>& field) const {
        double* const values = field.data();
        fillPhysicalBoundaries(values);

        MPI_Request requests[12];
        int requestCount = 0;

        // Post receives first so all six directions can make progress without
        // depending on eager-send buffering.
        if (xMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(0, 1, 1, domain_.pitchX, domain_.pitchY),
                               1, xFace_, xMinus_, 701, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(x-)");
        }
        if (xPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(domain_.nx + 1, 1, 1, domain_.pitchX,
                                             domain_.pitchY),
                               1, xFace_, xPlus_, 701, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(x+)");
        }
        if (yMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(1, 0, 1, domain_.pitchX, domain_.pitchY),
                               1, yFace_, yMinus_, 702, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(y-)");
        }
        if (yPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(1, domain_.ny + 1, 1, domain_.pitchX,
                                             domain_.pitchY),
                               1, yFace_, yPlus_, 702, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(y+)");
        }
        if (zMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(1, 1, 0, domain_.pitchX, domain_.pitchY),
                               1, zFace_, zMinus_, 703, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(z-)");
        }
        if (zPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Irecv(values + idx3(1, 1, domain_.nz + 1, domain_.pitchX,
                                             domain_.pitchY),
                               1, zFace_, zPlus_, 703, comm_,
                               &requests[requestCount++]),
                     "MPI_Irecv(z+)");
        }

        if (xMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(1, 1, 1, domain_.pitchX, domain_.pitchY),
                               1, xFace_, xMinus_, 701, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(x-)");
        }
        if (xPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(domain_.nx, 1, 1, domain_.pitchX,
                                             domain_.pitchY),
                               1, xFace_, xPlus_, 701, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(x+)");
        }
        if (yMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(1, 1, 1, domain_.pitchX, domain_.pitchY),
                               1, yFace_, yMinus_, 702, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(y-)");
        }
        if (yPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(1, domain_.ny, 1, domain_.pitchX,
                                             domain_.pitchY),
                               1, yFace_, yPlus_, 702, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(y+)");
        }
        if (zMinus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(1, 1, 1, domain_.pitchX, domain_.pitchY),
                               1, zFace_, zMinus_, 703, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(z-)");
        }
        if (zPlus_ != MPI_PROC_NULL) {
            mpiCheck(MPI_Isend(values + idx3(1, 1, domain_.nz, domain_.pitchX,
                                             domain_.pitchY),
                               1, zFace_, zPlus_, 703, comm_,
                               &requests[requestCount++]),
                     "MPI_Isend(z+)");
        }

        if (requestCount != 0) {
            mpiCheck(MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE),
                     "MPI_Waitall");
        }
    }

private:
    static int mpiCount(const size_t value, const char* const description) {
        if (value > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "MPI datatype count too large for %s\n", description);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return static_cast<int>(value);
    }

    static void mpiCheck(const int error, const char* const operation) {
        if (error != MPI_SUCCESS) {
            std::fprintf(stderr, "%s failed with MPI error %d\n", operation, error);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    void fillPhysicalBoundaries(double* const values) const {
        if (xMinus_ == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.nz; ++z) {
                for (size_t y = 1; y <= domain_.ny; ++y) {
                    values[idx3(0, y, z, domain_.pitchX, domain_.pitchY)] =
                        values[idx3(1, y, z, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
        if (xPlus_ == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.nz; ++z) {
                for (size_t y = 1; y <= domain_.ny; ++y) {
                    values[idx3(domain_.nx + 1, y, z, domain_.pitchX, domain_.pitchY)] =
                        values[idx3(domain_.nx, y, z, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
        if (yMinus_ == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.nz; ++z) {
                for (size_t x = 1; x <= domain_.nx; ++x) {
                    values[idx3(x, 0, z, domain_.pitchX, domain_.pitchY)] =
                        values[idx3(x, 1, z, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
        if (yPlus_ == MPI_PROC_NULL) {
            for (size_t z = 1; z <= domain_.nz; ++z) {
                for (size_t x = 1; x <= domain_.nx; ++x) {
                    values[idx3(x, domain_.ny + 1, z, domain_.pitchX, domain_.pitchY)] =
                        values[idx3(x, domain_.ny, z, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
        if (zMinus_ == MPI_PROC_NULL) {
            for (size_t y = 1; y <= domain_.ny; ++y) {
                for (size_t x = 1; x <= domain_.nx; ++x) {
                    values[idx3(x, y, 0, domain_.pitchX, domain_.pitchY)] =
                        values[idx3(x, y, 1, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
        if (zPlus_ == MPI_PROC_NULL) {
            for (size_t y = 1; y <= domain_.ny; ++y) {
                for (size_t x = 1; x <= domain_.nx; ++x) {
                    values[idx3(x, y, domain_.nz + 1, domain_.pitchX,
                                 domain_.pitchY)] =
                        values[idx3(x, y, domain_.nz, domain_.pitchX, domain_.pitchY)];
                }
            }
        }
    }

    MPI_Comm comm_;
    const LocalDomain& domain_;
    int xMinus_;
    int xPlus_;
    int yMinus_;
    int yPlus_;
    int zMinus_;
    int zPlus_;
    MPI_Datatype xFace_;
    MPI_Datatype xRowFace_ = MPI_DATATYPE_NULL;
    MPI_Datatype yFace_;
    MPI_Datatype zFace_;
};

void packOwned(const std::vector<double>& field, const LocalDomain& domain,
               std::vector<double>& packed) {
    packed.resize(domain.nx * domain.ny * domain.nz);
    size_t destination = 0;
    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t source = idx3(1, y, z, domain.pitchX, domain.pitchY);
            std::copy_n(field.data() + source, domain.nx, packed.data() + destination);
            destination += domain.nx;
        }
    }
}

// Gather only on request, reconstructing the global z-y-x layout used by the
// serial benchmark so the existing result hash and samples remain meaningful.
std::vector<double> gatherGlobalField(const std::vector<double>& field,
                                      const LocalDomain& domain, const MPI_Comm cartComm,
                                      const int cartRank, const int cartSize) {
    std::vector<double> packed;
    packOwned(field, domain, packed);

    if (packed.size() > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "Local result is too large for MPI_Gatherv\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int localCount = static_cast<int>(packed.size());
    std::vector<int> counts(cartRank == 0 ? static_cast<size_t>(cartSize) : 0);
    MPI_Gather(&localCount, 1, MPI_INT,
               cartRank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, cartComm);

    std::vector<int> displacements(cartRank == 0 ? static_cast<size_t>(cartSize) : 0);
    std::vector<double> gathered;
    if (cartRank == 0) {
        long long totalCount = 0;
        for (int rank = 0; rank < cartSize; ++rank) {
            if (totalCount > static_cast<long long>(INT_MAX) - counts[rank]) {
                std::fprintf(stderr, "Global result is too large for MPI_Gatherv\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            displacements[rank] = static_cast<int>(totalCount);
            totalCount += counts[rank];
        }
        gathered.resize(static_cast<size_t>(totalCount));
    }

    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE,
                cartRank == 0 ? gathered.data() : nullptr,
                cartRank == 0 ? counts.data() : nullptr,
                cartRank == 0 ? displacements.data() : nullptr,
                MPI_DOUBLE, 0, cartComm);

    std::vector<double> global;
    if (cartRank != 0) {
        return global;
    }

    global.resize(domain.globalNx * domain.globalNy * domain.globalNz);
    int processGrid[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    int topologyCoordinates[3] = {0, 0, 0};
    MPI_Cart_get(cartComm, 3, processGrid, periods, topologyCoordinates);
    for (int rank = 0; rank < cartSize; ++rank) {
        int coordinates[3] = {0, 0, 0};
        MPI_Cart_coords(cartComm, rank, 3, coordinates);

        // The process-grid dimensions are encoded by the Cartesian topology;
        // use them to recover each rank's block bounds.
        const size_t blockX = localExtent(domain.globalNx, processGrid[0], coordinates[0]);
        const size_t blockY = localExtent(domain.globalNy, processGrid[1], coordinates[1]);
        const size_t blockZ = localExtent(domain.globalNz, processGrid[2], coordinates[2]);
        const size_t startX = localOffset(domain.globalNx, processGrid[0], coordinates[0]);
        const size_t startY = localOffset(domain.globalNy, processGrid[1], coordinates[1]);
        const size_t startZ = localOffset(domain.globalNz, processGrid[2], coordinates[2]);
        const double* source = gathered.data() + displacements[rank];
        size_t sourceOffset = 0;
        for (size_t z = 0; z < blockZ; ++z) {
            for (size_t y = 0; y < blockY; ++y) {
                const size_t destination = idx3(startX, startY + y, startZ + z,
                                                domain.globalNx, domain.globalNy);
                std::copy_n(source + sourceOffset, blockX, global.data() + destination);
                sourceOffset += blockX;
            }
        }
    }
    return global;
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minValue = c[0];
    double maxValue = c[0];
    for (const double value : c) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

bool validateDistributed(const std::vector<double>& field, const LocalDomain& domain,
                         const MPI_Comm comm, const int rank) {
    bool localInvalid = false;
    double localMin = std::numeric_limits<double>::max();
    double localMax = std::numeric_limits<double>::lowest();
    for (size_t z = 1; z <= domain.nz; ++z) {
        for (size_t y = 1; y <= domain.ny; ++y) {
            const size_t row = idx3(0, y, z, domain.pitchX, domain.pitchY);
            for (size_t x = 1; x <= domain.nx; ++x) {
                const double value = field[row + x];
                if (std::isnan(value) || std::isinf(value)) {
                    localInvalid = true;
                } else {
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
            }
        }
    }

    const int localInvalidInt = localInvalid ? 1 : 0;
    int globalInvalid = 0;
    MPI_Allreduce(&localInvalidInt, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
    if (globalInvalid != 0) {
        if (rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    }
    return globalMax <= 10.0 && globalMin >= -10.0;
}

bool parseSize(const char* const text, size_t& value) {
    if (text == nullptr || text[0] == '-') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

void printUsage(const char* progName) {
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
    bool showHelp = false;
    bool parseOk = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], nx) && parseOk;
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], ny) && parseOk;
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            parseOk = parseSize(argv[++i], nz) && parseOk;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseOk = false;
        }
    }

    if (worldRank == 0 && showHelp) {
        printUsage(argv[0]);
    }
    if (showHelp || !parseOk) {
        if (worldRank == 0 && !showHelp) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp ? 0 : 1;
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid size is too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t gridSize = nx * ny * nz;

    int selection[4] = {0, 0, 0, 0};
    if (worldRank == 0) {
        if (!chooseActiveProcessGrid(worldSize, nx, ny, nz, selection[0], &selection[1])) {
            std::fprintf(stderr, "Unable to construct a nonempty MPI domain decomposition\n");
        }
    }
    MPI_Bcast(selection, 4, MPI_INT, 0, MPI_COMM_WORLD);
    const int activeSize = selection[0];
    int dims[3] = {selection[1], selection[2], selection[3]};
    if (activeSize <= 0 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &activeComm);
    if (worldRank >= activeSize) {
        MPI_Finalize();
        return 0;
    }

    MPI_Comm cartComm = MPI_COMM_NULL;
    const int periods[3] = {0, 0, 0};
    // Let the MPI implementation map Cartesian neighbors onto the physical
    // cluster topology when it can improve locality.
    MPI_Cart_create(activeComm, 3, dims, periods, 1, &cartComm);
    if (cartComm == MPI_COMM_NULL) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int cartRank = 0;
    int cartSize = 1;
    int coordinates[3] = {0, 0, 0};
    MPI_Comm_rank(cartComm, &cartRank);
    MPI_Comm_size(cartComm, &cartSize);
    MPI_Cart_coords(cartComm, cartRank, 3, coordinates);

    LocalDomain domain{nx,
                       ny,
                       nz,
                       localExtent(nx, dims[0], coordinates[0]),
                       localExtent(ny, dims[1], coordinates[1]),
                       localExtent(nz, dims[2], coordinates[2]),
                       localOffset(nx, dims[0], coordinates[0]),
                       localOffset(ny, dims[1], coordinates[1]),
                       localOffset(nz, dims[2], coordinates[2]),
                       0,
                       0,
                       0};
    domain.pitchX = domain.nx + 2;
    domain.pitchY = domain.ny + 2;
    domain.plane = domain.pitchX * domain.pitchY;
    const size_t localSize = domain.plane * (domain.nz + 2);

    if (cartRank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
        std::fflush(stdout);
    }

    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    initializeConcentration(cold, domain);

    {
        HaloExchange halos(cartComm, domain);
        if (cartRank == 0) {
            std::printf("Running Cahn-Hilliard simulation...\n");
            std::fflush(stdout);
        }
        MPI_Barrier(cartComm);
        const double start = MPI_Wtime();

        // Each iteration has two nearest-neighbor exchanges. No global
        // synchronization or reduction is on the timestep critical path.
        for (int t = 0; t < iterations; ++t) {
            halos.exchange(cold);
            computeChemicalPotential(cold, mu, domain, 1.0, 1.0, 1.0, 0.5,
                                     -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
            halos.exchange(mu);
            cahnHilliardUpdate(cnew, cold, mu, domain, 1.0, 0.01, 1.0, 1.0, 1.0);
            std::swap(cold, cnew);
        }

        MPI_Barrier(cartComm);
        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);

        if (cartRank == 0) {
            const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
            std::printf("Computation time: %lld ms\n", milliseconds);
            const double cellUpdates = static_cast<double>(gridSize) * iterations;
            const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0;
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }
    }

    std::vector<double> globalResult;
    if (printResults) {
        globalResult = gatherGlobalField(cold, domain, cartComm, cartRank, cartSize);
        if (cartRank == 0) {
            print_results(globalResult, "Concentration");
        }
    }

    bool valid = true;
    if (validate) {
        if (cartRank == 0) {
            std::printf("Validating result...\n");
        }
        if (printResults) {
            int validInt = 1;
            if (cartRank == 0) {
                validInt = validateResult(globalResult, nx, ny, nz) ? 1 : 0;
            }
            MPI_Bcast(&validInt, 1, MPI_INT, 0, cartComm);
            valid = validInt != 0;
        } else {
            valid = validateDistributed(cold, domain, cartComm, cartRank);
        }

        if (cartRank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Comm_free(&cartComm);
    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
