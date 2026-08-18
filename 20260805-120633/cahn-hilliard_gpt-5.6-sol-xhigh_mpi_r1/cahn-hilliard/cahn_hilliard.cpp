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

#if defined(__GNUC__) || defined(__clang__)
#define CH_RESTRICT __restrict__
#else
#define CH_RESTRICT
#endif

namespace {

[[noreturn]] void mpiFailure(const int error, const char* operation) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char message[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(call)                                    \
    do {                                                   \
        const int mpi_error_ = (call);                     \
        if (mpi_error_ != MPI_SUCCESS) {                   \
            mpiFailure(mpi_error_, #call);                 \
        }                                                  \
    } while (false)

bool checkedProduct(const size_t a, const size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

// Choose a Cartesian process grid which minimizes the total number of grid
// values crossing internal subdomain faces. Ties favor Z and then Y cuts,
// whose faces are more contiguous in the X-major memory layout.
bool chooseProcessGrid(const int processCount, const std::array<size_t, 3>& global,
                       std::array<int, 3>& dimensions) {
    long double bestInterface = std::numeric_limits<long double>::infinity();
    size_t bestMaxVolume = std::numeric_limits<size_t>::max();
    bool found = false;

    for (int px = 1; px <= processCount; ++px) {
        if (processCount % px != 0 || static_cast<size_t>(px) > global[0]) {
            continue;
        }
        const int remainder = processCount / px;
        for (int py = 1; py <= remainder; ++py) {
            if (remainder % py != 0 || static_cast<size_t>(py) > global[1]) {
                continue;
            }
            const int pz = remainder / py;
            if (static_cast<size_t>(pz) > global[2]) {
                continue;
            }

            const long double interfaces =
                static_cast<long double>(px - 1) * global[1] * global[2] +
                static_cast<long double>(py - 1) * global[0] * global[2] +
                static_cast<long double>(pz - 1) * global[0] * global[1];
            const size_t maxX = (global[0] + static_cast<size_t>(px) - 1) / static_cast<size_t>(px);
            const size_t maxY = (global[1] + static_cast<size_t>(py) - 1) / static_cast<size_t>(py);
            const size_t maxZ = (global[2] + static_cast<size_t>(pz) - 1) / static_cast<size_t>(pz);
            size_t maxXY = 0;
            size_t maxVolume = std::numeric_limits<size_t>::max();
            if (checkedProduct(maxX, maxY, maxXY)) {
                checkedProduct(maxXY, maxZ, maxVolume);
            }

            const bool betterTie = interfaces == bestInterface &&
                (maxVolume < bestMaxVolume ||
                 (maxVolume == bestMaxVolume &&
                  (px < dimensions[0] || (px == dimensions[0] && py < dimensions[1]))));
            if (!found || interfaces < bestInterface || betterTie) {
                dimensions = {px, py, pz};
                bestInterface = interfaces;
                bestMaxVolume = maxVolume;
                found = true;
            }
        }
    }
    return found;
}

struct Decomposition {
    MPI_Comm communicator = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    std::array<int, 3> dimensions{};
    std::array<int, 3> coordinates{};
    std::array<int, 3> lowerNeighbor{};
    std::array<int, 3> upperNeighbor{};
    std::array<size_t, 3> global{};
    std::array<size_t, 3> local{};
    std::array<size_t, 3> start{};

    Decomposition(const std::array<size_t, 3>& globalSize, const std::array<int, 3>& processGrid)
        : dimensions(processGrid), global(globalSize) {
        const int periods[3] = {0, 0, 0};
        MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 3, dimensions.data(), periods, 1, &communicator));
        MPI_CHECK(MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Comm_rank(communicator, &rank));
        MPI_CHECK(MPI_Comm_size(communicator, &size));
        MPI_CHECK(MPI_Cart_coords(communicator, rank, 3, coordinates.data()));

        for (int axis = 0; axis < 3; ++axis) {
            MPI_CHECK(MPI_Cart_shift(communicator, axis, 1, &lowerNeighbor[axis], &upperNeighbor[axis]));
            const size_t processes = static_cast<size_t>(dimensions[axis]);
            const size_t coordinate = static_cast<size_t>(coordinates[axis]);
            const size_t base = global[axis] / processes;
            const size_t remainder = global[axis] % processes;
            local[axis] = base + (coordinate < remainder ? 1 : 0);
            start[axis] = coordinate * base + std::min(coordinate, remainder);
        }
    }

    Decomposition(const Decomposition&) = delete;
    Decomposition& operator=(const Decomposition&) = delete;

    ~Decomposition() {
        if (communicator != MPI_COMM_NULL) {
            MPI_Comm_free(&communicator);
        }
    }
};

struct Field {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t rowStride;
    size_t planeStride;
    std::vector<double> values;

    explicit Field(const std::array<size_t, 3>& local)
        : nx(local[0]), ny(local[1]), nz(local[2]), rowStride(nx + 2),
          planeStride(rowStride * (ny + 2)), values(planeStride * (nz + 2)) {}

    size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * planeStride + y * rowStride + x;
    }

    double* pointer(const size_t x, const size_t y, const size_t z) noexcept {
        return values.data() + index(x, y, z);
    }

    const double* pointer(const size_t x, const size_t y, const size_t z) const noexcept {
        return values.data() + index(x, y, z);
    }
};

struct HaloTypes {
    std::array<MPI_Datatype, 3> face{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};

    explicit HaloTypes(const Field& field) {
        const int sizes[3] = {static_cast<int>(field.nz + 2), static_cast<int>(field.ny + 2),
                              static_cast<int>(field.nx + 2)};
        const int starts[3] = {0, 0, 0};
        const int xFace[3] = {static_cast<int>(field.nz), static_cast<int>(field.ny), 1};
        const int yFace[3] = {static_cast<int>(field.nz), 1, static_cast<int>(field.nx)};
        const int zFace[3] = {1, static_cast<int>(field.ny), static_cast<int>(field.nx)};
        const int* shapes[3] = {xFace, yFace, zFace};

        for (int axis = 0; axis < 3; ++axis) {
            MPI_CHECK(MPI_Type_create_subarray(3, sizes, shapes[axis], starts, MPI_ORDER_C,
                                                MPI_DOUBLE, &face[axis]));
            MPI_CHECK(MPI_Type_commit(&face[axis]));
        }
    }

    HaloTypes(const HaloTypes&) = delete;
    HaloTypes& operator=(const HaloTypes&) = delete;

    ~HaloTypes() {
        for (MPI_Datatype& datatype : face) {
            if (datatype != MPI_DATATYPE_NULL) {
                MPI_Type_free(&datatype);
            }
        }
    }
};

class HaloPlan {
public:
    HaloPlan(Field& field, const Decomposition& decomposition, const HaloTypes& types)
        : field_(field), decomposition_(decomposition) {
        requests_.reserve(12);
        for (int axis = 0; axis < 3; ++axis) {
            const int lowerTag = 2 * axis;
            const int upperTag = lowerTag + 1;
            if (decomposition_.lowerNeighbor[axis] != MPI_PROC_NULL) {
                addReceive(facePointer(axis, false, true), types.face[axis],
                           decomposition_.lowerNeighbor[axis], upperTag);
                addSend(facePointer(axis, false, false), types.face[axis],
                        decomposition_.lowerNeighbor[axis], lowerTag);
            }
            if (decomposition_.upperNeighbor[axis] != MPI_PROC_NULL) {
                addReceive(facePointer(axis, true, true), types.face[axis],
                           decomposition_.upperNeighbor[axis], lowerTag);
                addSend(facePointer(axis, true, false), types.face[axis],
                        decomposition_.upperNeighbor[axis], upperTag);
            }
        }
    }

    HaloPlan(const HaloPlan&) = delete;
    HaloPlan& operator=(const HaloPlan&) = delete;

    ~HaloPlan() {
        for (MPI_Request& request : requests_) {
            if (request != MPI_REQUEST_NULL) {
                MPI_Request_free(&request);
            }
        }
    }

    void start() {
        fillPhysicalBoundaries();
        if (!requests_.empty()) {
            MPI_CHECK(MPI_Startall(static_cast<int>(requests_.size()), requests_.data()));
        }
    }

    void wait() {
        if (!requests_.empty()) {
            MPI_CHECK(MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE));
        }
    }

private:
    Field& field_;
    const Decomposition& decomposition_;
    std::vector<MPI_Request> requests_;

    double* facePointer(const int axis, const bool upper, const bool ghost) {
        size_t x = 1;
        size_t y = 1;
        size_t z = 1;
        const size_t position = upper
            ? (ghost ? decomposition_.local[axis] + 1 : decomposition_.local[axis])
            : (ghost ? 0 : 1);
        if (axis == 0) x = position;
        if (axis == 1) y = position;
        if (axis == 2) z = position;
        return field_.pointer(x, y, z);
    }

    void addReceive(double* buffer, const MPI_Datatype datatype, const int source, const int tag) {
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_CHECK(MPI_Recv_init(buffer, 1, datatype, source, tag, decomposition_.communicator, &request));
        requests_.push_back(request);
    }

    void addSend(double* buffer, const MPI_Datatype datatype, const int destination, const int tag) {
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_CHECK(MPI_Send_init(buffer, 1, datatype, destination, tag, decomposition_.communicator, &request));
        requests_.push_back(request);
    }

    void fillPhysicalBoundaries() {
        if (decomposition_.lowerNeighbor[0] == MPI_PROC_NULL) {
            for (size_t z = 1; z <= field_.nz; ++z)
                for (size_t y = 1; y <= field_.ny; ++y)
                    *field_.pointer(0, y, z) = *field_.pointer(1, y, z);
        }
        if (decomposition_.upperNeighbor[0] == MPI_PROC_NULL) {
            for (size_t z = 1; z <= field_.nz; ++z)
                for (size_t y = 1; y <= field_.ny; ++y)
                    *field_.pointer(field_.nx + 1, y, z) = *field_.pointer(field_.nx, y, z);
        }
        if (decomposition_.lowerNeighbor[1] == MPI_PROC_NULL) {
            for (size_t z = 1; z <= field_.nz; ++z)
                std::copy_n(field_.pointer(1, 1, z), field_.nx, field_.pointer(1, 0, z));
        }
        if (decomposition_.upperNeighbor[1] == MPI_PROC_NULL) {
            for (size_t z = 1; z <= field_.nz; ++z)
                std::copy_n(field_.pointer(1, field_.ny, z), field_.nx,
                            field_.pointer(1, field_.ny + 1, z));
        }
        if (decomposition_.lowerNeighbor[2] == MPI_PROC_NULL) {
            for (size_t y = 1; y <= field_.ny; ++y)
                std::copy_n(field_.pointer(1, y, 1), field_.nx, field_.pointer(1, y, 0));
        }
        if (decomposition_.upperNeighbor[2] == MPI_PROC_NULL) {
            for (size_t y = 1; y <= field_.ny; ++y)
                std::copy_n(field_.pointer(1, y, field_.nz), field_.nx,
                            field_.pointer(1, y, field_.nz + 1));
        }
    }
};

struct Range {
    size_t begin;
    size_t end;
};

struct Box {
    Range x;
    Range y;
    Range z;
};

Box communicationIndependentBox(const Decomposition& decomposition) {
    Box box{};
    Range* ranges[3] = {&box.x, &box.y, &box.z};
    for (int axis = 0; axis < 3; ++axis) {
        ranges[axis]->begin = 1 + (decomposition.lowerNeighbor[axis] != MPI_PROC_NULL ? 1 : 0);
        ranges[axis]->end = decomposition.local[axis] + 1 -
                            (decomposition.upperNeighbor[axis] != MPI_PROC_NULL ? 1 : 0);
    }
    return box;
}

bool nonempty(const Box& box) {
    return box.x.begin < box.x.end && box.y.begin < box.y.end && box.z.begin < box.z.end;
}

template <typename Kernel>
void applyStencilWithHalo(HaloPlan& halo, const Decomposition& decomposition, Kernel&& kernel) {
    const Box core = communicationIndependentBox(decomposition);
    halo.start();

    if (nonempty(core)) {
        kernel(core);
    }
    halo.wait();

    const Box whole{{1, decomposition.local[0] + 1},
                    {1, decomposition.local[1] + 1},
                    {1, decomposition.local[2] + 1}};
    if (!nonempty(core)) {
        kernel(whole);
        return;
    }

    // Seven disjoint boxes cover the halo-dependent shell around the core.
    kernel({whole.x, whole.y, {whole.z.begin, core.z.begin}});
    kernel({whole.x, whole.y, {core.z.end, whole.z.end}});
    kernel({whole.x, {whole.y.begin, core.y.begin}, core.z});
    kernel({whole.x, {core.y.end, whole.y.end}, core.z});
    kernel({{whole.x.begin, core.x.begin}, core.y, core.z});
    kernel({{core.x.end, whole.x.end}, core.y, core.z});
}

void computeChemicalPotential(const Field& concentration, Field& potential, const Box& box,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double eAA, const double eBB, const double eAB) {
    const double* CH_RESTRICT c = concentration.values.data();
    double* CH_RESTRICT mu = potential.values.data();
    const size_t row = concentration.rowStride;
    const size_t plane = concentration.planeStride;

    for (size_t z = box.z.begin; z < box.z.end; ++z) {
        for (size_t y = box.y.begin; y < box.y.end; ++y) {
            const size_t base = z * plane + y * row;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t x = box.x.begin; x < box.x.end; ++x) {
                const size_t index = base + x;
                const double value = c[index];
                const double cxx = (c[index + 1] + c[index - 1] - 2.0 * value) / (dx * dx);
                const double cyy = (c[index + row] + c[index - row] - 2.0 * value) / (dy * dy);
                const double czz = (c[index + plane] + c[index - plane] - 2.0 * value) / (dz * dz);

                mu[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB)
                            + 3.0 * value + value * value * value
                            - gamma * (cxx + cyy + czz);
            }
        }
    }
}

void cahnHilliardUpdate(Field& next, const Field& current, const Field& potential, const Box& box,
                        const double diffusion, const double dt,
                        const double dx, const double dy, const double dz) {
    double* CH_RESTRICT output = next.values.data();
    const double* CH_RESTRICT input = current.values.data();
    const double* CH_RESTRICT mu = potential.values.data();
    const size_t row = current.rowStride;
    const size_t plane = current.planeStride;

    for (size_t z = box.z.begin; z < box.z.end; ++z) {
        for (size_t y = box.y.begin; y < box.y.end; ++y) {
            const size_t base = z * plane + y * row;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t x = box.x.begin; x < box.x.end; ++x) {
                const size_t index = base + x;
                const double value = mu[index];
                const double muxx = (mu[index + 1] + mu[index - 1] - 2.0 * value) / (dx * dx);
                const double muyy = (mu[index + row] + mu[index - row] - 2.0 * value) / (dy * dy);
                const double muzz = (mu[index + plane] + mu[index - plane] - 2.0 * value) / (dz * dz);
                output[index] = input[index] + dt * diffusion * (muxx + muyy + muzz);
            }
        }
    }
}

void initializeConcentration(Field& concentration, const Decomposition& decomposition, const size_t volume) {
    const size_t globalXY = decomposition.global[0] * decomposition.global[1];
    for (size_t z = 1; z <= concentration.nz; ++z) {
        const size_t globalZ = decomposition.start[2] + z - 1;
        for (size_t y = 1; y <= concentration.ny; ++y) {
            const size_t globalY = decomposition.start[1] + y - 1;
            for (size_t x = 1; x <= concentration.nx; ++x) {
                const size_t globalX = decomposition.start[0] + x - 1;
                const size_t linearId = globalZ * globalXY + globalY * decomposition.global[0] + globalX;
                const double pseudo = (((linearId + 1) * size_t{1299709}) % volume) /
                                      static_cast<double>(volume);
                *concentration.pointer(x, y, z) = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const Field& concentration, const Decomposition& decomposition) {
    int localFinite = 1;
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= concentration.nz; ++z) {
        for (size_t y = 1; y <= concentration.ny; ++y) {
            const double* row = concentration.pointer(1, y, z);
            for (size_t x = 0; x < concentration.nx; ++x) {
                const double value = row[x];
                if (std::isnan(value) || std::isinf(value)) {
                    localFinite = 0;
                } else {
                    localMinimum = std::min(localMinimum, value);
                    localMaximum = std::max(localMaximum, value);
                }
            }
        }
    }

    int globallyFinite = 0;
    MPI_CHECK(MPI_Allreduce(&localFinite, &globallyFinite, 1, MPI_INT, MPI_LAND,
                            decomposition.communicator));
    if (!globallyFinite) {
        if (decomposition.rank == 0) {
            std::printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    double globalMinimum = 0.0;
    double globalMaximum = 0.0;
    MPI_CHECK(MPI_Allreduce(&localMinimum, &globalMinimum, 1, MPI_DOUBLE, MPI_MIN,
                            decomposition.communicator));
    MPI_CHECK(MPI_Allreduce(&localMaximum, &globalMaximum, 1, MPI_DOUBLE, MPI_MAX,
                            decomposition.communicator));

    if (decomposition.rank == 0) {
        std::printf("Concentration range: [%.6f, %.6f]\n", globalMinimum, globalMaximum);
        if (globalMaximum > 10.0 || globalMinimum < -10.0) {
            std::printf("Validation failed: values out of expected range\n");
        }
    }
    return globalMaximum <= 10.0 && globalMinimum >= -10.0;
}

std::vector<double> gatherResult(const Field& concentration, const Decomposition& decomposition,
                                 const size_t globalVolume) {
    const size_t localCountSize = concentration.nx * concentration.ny * concentration.nz;
    const int localCount = static_cast<int>(localCountSize);
    std::vector<double> packed(localCountSize);
    size_t offset = 0;
    for (size_t z = 1; z <= concentration.nz; ++z) {
        for (size_t y = 1; y <= concentration.ny; ++y) {
            std::copy_n(concentration.pointer(1, y, z), concentration.nx, packed.data() + offset);
            offset += concentration.nx;
        }
    }

    std::array<unsigned long long, 6> metadata{
        decomposition.start[0], decomposition.start[1], decomposition.start[2],
        decomposition.local[0], decomposition.local[1], decomposition.local[2]};
    std::vector<unsigned long long> allMetadata;
    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<double> gathered;
    if (decomposition.rank == 0) {
        allMetadata.resize(static_cast<size_t>(decomposition.size) * metadata.size());
        counts.resize(decomposition.size);
        displacements.resize(decomposition.size);
    }

    MPI_CHECK(MPI_Gather(metadata.data(), static_cast<int>(metadata.size()), MPI_UNSIGNED_LONG_LONG,
                         allMetadata.data(), static_cast<int>(metadata.size()), MPI_UNSIGNED_LONG_LONG,
                         0, decomposition.communicator));
    MPI_CHECK(MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                         0, decomposition.communicator));

    if (decomposition.rank == 0) {
        int displacement = 0;
        for (int process = 0; process < decomposition.size; ++process) {
            displacements[process] = displacement;
            displacement += counts[process];
        }
        gathered.resize(globalVolume);
    }
    MPI_CHECK(MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, gathered.data(), counts.data(),
                          displacements.data(), MPI_DOUBLE, 0, decomposition.communicator));

    if (decomposition.rank != 0) {
        return {};
    }

    std::vector<double> global(globalVolume);
    const size_t globalX = decomposition.global[0];
    const size_t globalY = decomposition.global[1];
    for (int process = 0; process < decomposition.size; ++process) {
        const unsigned long long* info = allMetadata.data() + static_cast<size_t>(process) * 6;
        const size_t startX = static_cast<size_t>(info[0]);
        const size_t startY = static_cast<size_t>(info[1]);
        const size_t startZ = static_cast<size_t>(info[2]);
        const size_t localX = static_cast<size_t>(info[3]);
        const size_t localY = static_cast<size_t>(info[4]);
        const size_t localZ = static_cast<size_t>(info[5]);
        size_t source = static_cast<size_t>(displacements[process]);
        for (size_t z = 0; z < localZ; ++z) {
            for (size_t y = 0; y < localY; ++y) {
                const size_t destination = (startZ + z) * globalX * globalY +
                                           (startY + y) * globalX + startX;
                std::copy_n(gathered.data() + source, localX, global.data() + destination);
                source += localX;
            }
        }
    }
    return global;
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

int runBenchmark(const int argc, char** argv) {
    int worldRank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Grid dimensions must be positive and the iteration count non-negative.\n");
        }
        return 1;
    }

    size_t globalXY = 0;
    size_t gridSize = 0;
    if (!checkedProduct(nx, ny, globalXY) || !checkedProduct(globalXY, nz, gridSize)) {
        if (worldRank == 0) std::fprintf(stderr, "The requested grid size overflows size_t.\n");
        return 1;
    }
    if (printResults && gridSize > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::fprintf(stderr, "-r requires at most INT_MAX grid values for MPI_Gatherv.\n");
        }
        return 1;
    }

    const std::array<size_t, 3> global{nx, ny, nz};
    std::array<int, 3> processGrid{};
    if (!chooseProcessGrid(worldSize, global, processGrid)) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "Cannot form a nonempty 3D Cartesian decomposition of %zu x %zu x %zu over %d ranks.\n",
                         nx, ny, nz, worldSize);
        }
        return 1;
    }

    Decomposition decomposition(global, processGrid);
    for (size_t extent : decomposition.local) {
        if (extent > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
            if (decomposition.rank == 0) std::fprintf(stderr, "A local grid extent exceeds MPI datatype limits.\n");
            return 1;
        }
    }

    if (decomposition.rank == 0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
        std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Time steps: %d\n", iterations);
        std::printf("MPI ranks: %d (%d x %d x %d Cartesian grid)\n", decomposition.size,
                    decomposition.dimensions[0], decomposition.dimensions[1], decomposition.dimensions[2]);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing concentration field...\n");
    }

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double eAA = -(2.0 / 9.0);
    const double eBB = -(2.0 / 9.0);
    const double eAB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double diffusion = 1.0;

    Field concentration[2] = {Field(decomposition.local), Field(decomposition.local)};
    Field potential(decomposition.local);
    initializeConcentration(concentration[0], decomposition, gridSize);

    HaloTypes haloTypes(concentration[0]);
    HaloPlan concentrationHalo[2] = {
        HaloPlan(concentration[0], decomposition, haloTypes),
        HaloPlan(concentration[1], decomposition, haloTypes)};
    HaloPlan potentialHalo(potential, decomposition, haloTypes);

    if (decomposition.rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(decomposition.communicator));
    const double startTime = MPI_Wtime();

    int current = 0;
    for (int step = 0; step < iterations; ++step) {
        applyStencilWithHalo(concentrationHalo[current], decomposition, [&](const Box& box) {
            computeChemicalPotential(concentration[current], potential, box, dx, dy, dz,
                                     gamma, eAA, eBB, eAB);
        });

        const int next = 1 - current;
        applyStencilWithHalo(potentialHalo, decomposition, [&](const Box& box) {
            cahnHilliardUpdate(concentration[next], concentration[current], potential, box,
                               diffusion, dt, dx, dy, dz);
        });
        current = next;
    }

    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         decomposition.communicator));

    if (decomposition.rank == 0) {
        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = elapsed > 0.0 ? cellUpdates / elapsed / 1.0e6 : 0.0;
        std::printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> globalResult = gatherResult(concentration[current], decomposition, gridSize);
        if (decomposition.rank == 0) {
            print_results(globalResult, "Concentration");
        }
    }

    bool valid = true;
    if (validate) {
        if (decomposition.rank == 0) std::printf("Validating result...\n");
        valid = validateResult(concentration[current], decomposition);
        if (decomposition.rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    return valid ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }
    (void)provided;

    const int result = runBenchmark(argc, argv);
    MPI_CHECK(MPI_Finalize());
    return result;
}
