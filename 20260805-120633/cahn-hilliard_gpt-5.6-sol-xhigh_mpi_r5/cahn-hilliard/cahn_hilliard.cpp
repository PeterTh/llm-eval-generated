#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Block {
    int size;
    int begin;
};

Block splitDimension(const int globalSize, const int parts, const int coordinate) {
    const int base = globalSize / parts;
    const int remainder = globalSize % parts;
    return {base + (coordinate < remainder ? 1 : 0),
            coordinate * base + std::min(coordinate, remainder)};
}

// Choose a factorization which minimizes the total area of inter-process
// interfaces.  Unlike MPI_Dims_create, this also guarantees a nonempty block
// on every rank for highly anisotropic grids.
std::array<int, 3> chooseProcessGrid(const int processes, const std::array<int, 3>& global) {
    std::array<int, 3> best{0, 0, 0};
    long double bestSurface = std::numeric_limits<long double>::infinity();
    long long bestMaxVolume = std::numeric_limits<long long>::max();

    for (int px = 1; px <= std::min(processes, global[0]); ++px) {
        if (processes % px != 0) continue;
        const int remaining = processes / px;
        for (int py = 1; py <= std::min(remaining, global[1]); ++py) {
            if (remaining % py != 0) continue;
            const int pz = remaining / py;
            if (pz > global[2]) continue;

            const long double surface =
                static_cast<long double>(px - 1) * global[1] * global[2] +
                static_cast<long double>(py - 1) * global[0] * global[2] +
                static_cast<long double>(pz - 1) * global[0] * global[1];
            const long long maxVolume =
                static_cast<long long>((global[0] + px - 1) / px) *
                ((global[1] + py - 1) / py) * ((global[2] + pz - 1) / pz);

            // Communication volume is the primary scalability constraint;
            // maximum rank load breaks ties between equivalent factorizations.
            if (surface < bestSurface ||
                (surface == bestSurface && maxVolume < bestMaxVolume)) {
                best = {px, py, pz};
                bestSurface = surface;
                bestMaxVolume = maxVolume;
            }
        }
    }
    return best;
}

struct Domain {
    MPI_Comm cart = MPI_COMM_NULL;
    std::array<int, 3> global{};   // x, y, z
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    std::array<int, 3> local{};
    std::array<int, 3> begin{};
    std::array<int, 6> neighbor{}; // x-, x+, y-, y+, z-, z+
    std::array<MPI_Datatype, 3> face{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                     MPI_DATATYPE_NULL};
    std::size_t rowStride = 0;
    std::size_t planeStride = 0;

    Domain(MPI_Comm world, const std::array<int, 3>& globalSizes,
           const std::array<int, 3>& processGrid)
        : global(globalSizes), dims(processGrid) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(world, 3, dims.data(), periods, 0, &cart);

        int rank = 0;
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 3, coords.data());
        for (int d = 0; d < 3; ++d) {
            const Block block = splitDimension(global[d], dims[d], coords[d]);
            local[d] = block.size;
            begin[d] = block.begin;
        }
        MPI_Cart_shift(cart, 0, 1, &neighbor[0], &neighbor[1]);
        MPI_Cart_shift(cart, 1, 1, &neighbor[2], &neighbor[3]);
        MPI_Cart_shift(cart, 2, 1, &neighbor[4], &neighbor[5]);

        rowStride = static_cast<std::size_t>(local[0] + 2);
        planeStride = rowStride * static_cast<std::size_t>(local[1] + 2);
        createFaceTypes();
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    ~Domain() {
        for (MPI_Datatype& datatype : face) {
            if (datatype != MPI_DATATYPE_NULL) MPI_Type_free(&datatype);
        }
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    }

    std::size_t index(const int x, const int y, const int z) const noexcept {
        return static_cast<std::size_t>(z) * planeStride +
               static_cast<std::size_t>(y) * rowStride + static_cast<std::size_t>(x);
    }

    std::size_t allocatedCells() const noexcept {
        return static_cast<std::size_t>(local[2] + 2) * planeStride;
    }

    std::size_t interiorCells() const noexcept {
        return static_cast<std::size_t>(local[0]) * local[1] * local[2];
    }

  private:
    void createFaceTypes() {
        // X face: one value from every interior row of every interior plane.
        MPI_Datatype xRows = MPI_DATATYPE_NULL;
        MPI_Type_vector(local[1], 1, static_cast<int>(rowStride), MPI_DOUBLE, &xRows);
        MPI_Type_create_hvector(local[2], 1,
                                static_cast<MPI_Aint>(planeStride * sizeof(double)),
                                xRows, &face[0]);
        MPI_Type_commit(&face[0]);
        MPI_Type_free(&xRows);

        // Y face: one contiguous X row from every interior plane.
        MPI_Datatype yRow = MPI_DATATYPE_NULL;
        MPI_Type_contiguous(local[0], MPI_DOUBLE, &yRow);
        MPI_Type_create_hvector(local[2], 1,
                                static_cast<MPI_Aint>(planeStride * sizeof(double)),
                                yRow, &face[1]);
        MPI_Type_commit(&face[1]);
        MPI_Type_free(&yRow);

        // Z face: all interior X rows in one plane.
        MPI_Type_vector(local[1], local[0], static_cast<int>(rowStride), MPI_DOUBLE,
                        &face[2]);
        MPI_Type_commit(&face[2]);
    }
};

class PersistentHaloExchange {
  public:
    PersistentHaloExchange(double* field, const Domain& domain) : domain_(domain), field_(field) {
        const int lx = domain_.local[0];
        const int ly = domain_.local[1];
        const int lz = domain_.local[2];

        // A low ghost receives the neighbor's high face and vice versa.
        addPair(0, domain_.index(0, 1, 1), domain_.index(lx + 1, 1, 1),
                domain_.index(1, 1, 1), domain_.index(lx, 1, 1), 100);
        addPair(1, domain_.index(1, 0, 1), domain_.index(1, ly + 1, 1),
                domain_.index(1, 1, 1), domain_.index(1, ly, 1), 102);
        addPair(2, domain_.index(1, 1, 0), domain_.index(1, 1, lz + 1),
                domain_.index(1, 1, 1), domain_.index(1, 1, lz), 104);
    }

    PersistentHaloExchange(const PersistentHaloExchange&) = delete;
    PersistentHaloExchange& operator=(const PersistentHaloExchange&) = delete;

    ~PersistentHaloExchange() {
        for (MPI_Request& request : requests_) {
            if (request != MPI_REQUEST_NULL) MPI_Request_free(&request);
        }
    }

    void start() {
        fillPhysicalBoundaries();
        MPI_Startall(static_cast<int>(requests_.size()), requests_.data());
    }

    void wait() { MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE); }

  private:
    const Domain& domain_;
    double* field_;
    std::array<MPI_Request, 12> requests_{};
    int nextRequest_ = 0;

    void addPair(const int dimension, const std::size_t lowGhost,
                 const std::size_t highGhost, const std::size_t lowInterior,
                 const std::size_t highInterior, const int lowTag) {
        const int lowNeighbor = domain_.neighbor[2 * dimension];
        const int highNeighbor = domain_.neighbor[2 * dimension + 1];
        const MPI_Datatype datatype = domain_.face[dimension];

        MPI_Recv_init(field_ + lowGhost, 1, datatype, lowNeighbor, lowTag + 1,
                      domain_.cart, &requests_[nextRequest_++]);
        MPI_Recv_init(field_ + highGhost, 1, datatype, highNeighbor, lowTag,
                      domain_.cart, &requests_[nextRequest_++]);
        MPI_Send_init(field_ + lowInterior, 1, datatype, lowNeighbor, lowTag,
                      domain_.cart, &requests_[nextRequest_++]);
        MPI_Send_init(field_ + highInterior, 1, datatype, highNeighbor, lowTag + 1,
                      domain_.cart, &requests_[nextRequest_++]);
    }

    void fillPhysicalBoundaries() {
        const int lx = domain_.local[0];
        const int ly = domain_.local[1];
        const int lz = domain_.local[2];

        if (domain_.neighbor[0] == MPI_PROC_NULL) {
            for (int z = 1; z <= lz; ++z)
                for (int y = 1; y <= ly; ++y)
                    field_[domain_.index(0, y, z)] = field_[domain_.index(1, y, z)];
        }
        if (domain_.neighbor[1] == MPI_PROC_NULL) {
            for (int z = 1; z <= lz; ++z)
                for (int y = 1; y <= ly; ++y)
                    field_[domain_.index(lx + 1, y, z)] = field_[domain_.index(lx, y, z)];
        }
        if (domain_.neighbor[2] == MPI_PROC_NULL) {
            for (int z = 1; z <= lz; ++z)
                for (int x = 1; x <= lx; ++x)
                    field_[domain_.index(x, 0, z)] = field_[domain_.index(x, 1, z)];
        }
        if (domain_.neighbor[3] == MPI_PROC_NULL) {
            for (int z = 1; z <= lz; ++z)
                for (int x = 1; x <= lx; ++x)
                    field_[domain_.index(x, ly + 1, z)] = field_[domain_.index(x, ly, z)];
        }
        if (domain_.neighbor[4] == MPI_PROC_NULL) {
            for (int y = 1; y <= ly; ++y)
                for (int x = 1; x <= lx; ++x)
                    field_[domain_.index(x, y, 0)] = field_[domain_.index(x, y, 1)];
        }
        if (domain_.neighbor[5] == MPI_PROC_NULL) {
            for (int y = 1; y <= ly; ++y)
                for (int x = 1; x <= lx; ++x)
                    field_[domain_.index(x, y, lz + 1)] = field_[domain_.index(x, y, lz)];
        }
    }
};

inline double laplacian(const double* field, const std::size_t i,
                        const std::size_t rowStride, const std::size_t planeStride,
                        const double dx, const double dy, const double dz) noexcept {
    const double center = field[i];
    const double cxx = (field[i + 1] + field[i - 1] - 2.0 * center) / (dx * dx);
    const double cyy = (field[i + rowStride] + field[i - rowStride] - 2.0 * center) /
                       (dy * dy);
    const double czz =
        (field[i + planeStride] + field[i - planeStride] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

inline void chemicalCell(const double* __restrict concentration, double* __restrict mu,
                         const std::size_t i, const Domain& domain, const double dx,
                         const double dy, const double dz, const double gamma,
                         const double eAA, const double eBB, const double eAB) noexcept {
    const double cv = concentration[i];
    mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
            3.0 * cv + cv * cv * cv -
            gamma * laplacian(concentration, i, domain.rowStride, domain.planeStride,
                              dx, dy, dz);
}

inline void updateCell(double* __restrict next, const double* __restrict current,
                       const double* __restrict mu, const std::size_t i,
                       const Domain& domain, const double scale, const double dx,
                       const double dy, const double dz) noexcept {
    next[i] = current[i] +
              scale * laplacian(mu, i, domain.rowStride, domain.planeStride, dx, dy, dz);
}

template <typename CellOperation>
void applyCore(const Domain& domain, CellOperation&& operation) {
    const int lx = domain.local[0];
    const int ly = domain.local[1];
    const int lz = domain.local[2];
    if (lx < 3 || ly < 3 || lz < 3) return;

    for (int z = 2; z < lz; ++z)
        for (int y = 2; y < ly; ++y)
            for (int x = 2; x < lx; ++x) operation(domain.index(x, y, z));
}

template <typename CellOperation>
void applyBoundaryShell(const Domain& domain, CellOperation&& operation) {
    const int lx = domain.local[0];
    const int ly = domain.local[1];
    const int lz = domain.local[2];

    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            if (z == 1 || z == lz || y == 1 || y == ly) {
                for (int x = 1; x <= lx; ++x) operation(domain.index(x, y, z));
            } else {
                operation(domain.index(1, y, z));
                if (lx > 1) operation(domain.index(lx, y, z));
            }
        }
    }
}

void initializeConcentration(std::vector<double>& concentration, const Domain& domain) {
    const std::size_t globalVolume = static_cast<std::size_t>(domain.global[0]) *
                                     domain.global[1] * domain.global[2];
    for (int z = 1; z <= domain.local[2]; ++z) {
        const std::size_t globalZ = static_cast<std::size_t>(domain.begin[2] + z - 1);
        for (int y = 1; y <= domain.local[1]; ++y) {
            const std::size_t globalY = static_cast<std::size_t>(domain.begin[1] + y - 1);
            for (int x = 1; x <= domain.local[0]; ++x) {
                const std::size_t globalX = static_cast<std::size_t>(domain.begin[0] + x - 1);
                const std::size_t linearId =
                    globalZ * static_cast<std::size_t>(domain.global[0]) * domain.global[1] +
                    globalY * domain.global[0] + globalX;
                const double pseudo =
                    (((linearId + 1) * 1299709) % globalVolume) /
                    static_cast<double>(globalVolume);
                concentration[domain.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

std::vector<double> gatherResult(const std::vector<double>& localResult,
                                 const Domain& domain, const int rank) {
    MPI_Datatype localInterior = MPI_DATATYPE_NULL;
    const int localSizes[3] = {domain.local[2] + 2, domain.local[1] + 2,
                               domain.local[0] + 2};
    const int localSubsizes[3] = {domain.local[2], domain.local[1], domain.local[0]};
    const int localStarts[3] = {1, 1, 1};
    MPI_Type_create_subarray(3, localSizes, localSubsizes, localStarts, MPI_ORDER_C,
                             MPI_DOUBLE, &localInterior);
    MPI_Type_commit(&localInterior);

    std::vector<double> result;
    if (rank == 0) {
        result.resize(static_cast<std::size_t>(domain.global[0]) * domain.global[1] *
                      domain.global[2]);
    }

    if (rank == 0) {
        int processes = 1;
        MPI_Comm_size(domain.cart, &processes);
        for (int source = 0; source < processes; ++source) {
            std::array<int, 3> coords{};
            MPI_Cart_coords(domain.cart, source, 3, coords.data());
            std::array<int, 3> sourceLocal{};
            std::array<int, 3> sourceBegin{};
            for (int d = 0; d < 3; ++d) {
                const Block block = splitDimension(domain.global[d], domain.dims[d], coords[d]);
                sourceLocal[d] = block.size;
                sourceBegin[d] = block.begin;
            }

            const int globalSizes[3] = {domain.global[2], domain.global[1], domain.global[0]};
            const int globalSubsizes[3] = {sourceLocal[2], sourceLocal[1], sourceLocal[0]};
            const int globalStarts[3] = {sourceBegin[2], sourceBegin[1], sourceBegin[0]};
            MPI_Datatype globalBlock = MPI_DATATYPE_NULL;
            MPI_Type_create_subarray(3, globalSizes, globalSubsizes, globalStarts, MPI_ORDER_C,
                                     MPI_DOUBLE, &globalBlock);
            MPI_Type_commit(&globalBlock);

            if (source == 0) {
                MPI_Sendrecv(localResult.data(), 1, localInterior, 0, 200, result.data(), 1,
                             globalBlock, 0, 200, domain.cart, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(result.data(), 1, globalBlock, source, 200, domain.cart,
                         MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&globalBlock);
        }
    } else {
        MPI_Send(localResult.data(), 1, localInterior, 0, 200, domain.cart);
    }

    MPI_Type_free(&localInterior);
    return result;
}

bool validateResult(const std::vector<double>& concentration, const Domain& domain,
                    const int rank) {
    bool localFinite = true;
    double localMin = std::numeric_limits<double>::infinity();
    double localMax = -std::numeric_limits<double>::infinity();
    for (int z = 1; z <= domain.local[2]; ++z) {
        for (int y = 1; y <= domain.local[1]; ++y) {
            for (int x = 1; x <= domain.local[0]; ++x) {
                const double value = concentration[domain.index(x, y, z)];
                localFinite = localFinite && std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }

    int finite = localFinite ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_MIN, domain.cart);
    double globalMin = 0.0;
    double globalMax = 0.0;
    MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, domain.cart);
    MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, domain.cart);

    int valid = finite && localFinite;
    if (rank == 0) {
        if (!finite) {
            std::printf("Validation failed: found NaN or Inf value\n");
        } else {
            std::printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
            if (globalMax > 10.0 || globalMin < -10.0) {
                std::printf("Validation failed: values out of expected range\n");
                valid = 0;
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, domain.cart);
    return valid != 0;
}

void printUsage(const char* programName) {
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

struct Options {
    int nx = 64;
    int ny = 0;
    int nz = 0;
    int iterations = 20;
    int validate = 0;
    int printResults = 0;
};

int parseOptions(const int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.nz = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            options.validate = 1;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            options.printResults = 1;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            return 1;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    if (options.nx <= 0 || options.ny <= 0 || options.nz <= 0 || options.iterations < 0) {
        std::printf("Grid dimensions must be positive and time steps must be nonnegative.\n");
        return -1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    Options options;
    int parseStatus = 0;
    if (worldRank == 0) parseStatus = parseOptions(argc, argv, options);
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (parseStatus != 0) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus < 0 ? 1 : 0;
    }
    MPI_Bcast(&options, static_cast<int>(sizeof(options)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const std::size_t gridSize = static_cast<std::size_t>(options.nx) * options.ny * options.nz;
    if (static_cast<std::size_t>(processes) > gridSize) {
        if (worldRank == 0)
            std::fprintf(stderr, "The number of MPI processes cannot exceed the number of cells.\n");
        MPI_Finalize();
        return 1;
    }

    int exitCode = 0;
    const std::array<int, 3> global{options.nx, options.ny, options.nz};
    const std::array<int, 3> processGrid = chooseProcessGrid(processes, global);
    if (processGrid[0] == 0) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "This grid cannot be divided into %d nonempty rectangular MPI blocks.\n",
                         processes);
        }
        MPI_Finalize();
        return 1;
    }

    {
        Domain domain(MPI_COMM_WORLD, global, processGrid);
        int rank = 0;
        MPI_Comm_rank(domain.cart, &rank);

        if (rank == 0) {
            std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
            std::printf("Grid size: %d x %d x %d\n", options.nx, options.ny, options.nz);
            std::printf("Time steps: %d\n", options.iterations);
            std::printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
            std::printf("MPI processes: %d (%d x %d x %d)\n", processes, domain.dims[0],
                        domain.dims[1], domain.dims[2]);
        }

        // Physical parameters.
        constexpr double dx = 1.0;
        constexpr double dy = 1.0;
        constexpr double dz = 1.0;
        constexpr double dt = 0.01;
        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = (2.0 / 9.0);
        constexpr double gamma = 0.5;
        constexpr double diffusion = 1.0;

        std::vector<double> fieldA(domain.allocatedCells());
        std::vector<double> fieldB(domain.allocatedCells());
        std::vector<double> mu(domain.allocatedCells());
        std::vector<double>* current = &fieldA;
        std::vector<double>* next = &fieldB;

        if (rank == 0) std::printf("Initializing concentration field...\n");
        initializeConcentration(*current, domain);

        PersistentHaloExchange fieldAHalo(fieldA.data(), domain);
        PersistentHaloExchange fieldBHalo(fieldB.data(), domain);
        PersistentHaloExchange muHalo(mu.data(), domain);

        if (rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(domain.cart);
        const double start = MPI_Wtime();

        for (int step = 0; step < options.iterations; ++step) {
            PersistentHaloExchange& currentHalo =
                current == &fieldA ? fieldAHalo : fieldBHalo;
            currentHalo.start();

            auto chemical = [&](const std::size_t i) {
                chemicalCell(current->data(), mu.data(), i, domain, dx, dy, dz, gamma,
                             eAA, eBB, eAB);
            };
            applyCore(domain, chemical);
            currentHalo.wait();
            applyBoundaryShell(domain, chemical);

            muHalo.start();
            auto update = [&](const std::size_t i) {
                updateCell(next->data(), current->data(), mu.data(), i, domain,
                           dt * diffusion, dx, dy, dz);
            };
            applyCore(domain, update);
            muHalo.wait();
            applyBoundaryShell(domain, update);
            std::swap(current, next);
        }

        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.cart);

        if (rank == 0) {
            const long long milliseconds = static_cast<long long>(elapsed * 1000.0);
            std::printf("Computation time: %lld ms\n", milliseconds);
            const double cellUpdates = static_cast<double>(gridSize) * options.iterations;
            const double mcups = cellUpdates / elapsed / 1.0e6;
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (options.printResults) {
            std::vector<double> globalResult = gatherResult(*current, domain, rank);
            if (rank == 0) print_results(globalResult, "Concentration");
        }

        if (options.validate) {
            if (rank == 0) std::printf("Validating result...\n");
            const bool valid = validateResult(*current, domain, rank);
            if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
