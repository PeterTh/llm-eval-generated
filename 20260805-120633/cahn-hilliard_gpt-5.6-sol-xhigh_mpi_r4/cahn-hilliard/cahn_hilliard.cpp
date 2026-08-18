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

namespace {

enum Face : int { X_MINUS = 0, X_PLUS = 1, Y_MINUS = 2, Y_PLUS = 3, Z_MINUS = 4, Z_PLUS = 5 };

constexpr int oppositeFace(const int face) noexcept { return face ^ 1; }

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Extent {
    size_t size;
    size_t offset;
};

Extent splitExtent(const size_t globalSize, const int partitions, const int coordinate) {
    const size_t count = globalSize / static_cast<size_t>(partitions);
    const size_t remainder = globalSize % static_cast<size_t>(partitions);
    const size_t extra = static_cast<size_t>(coordinate) < remainder ? 1 : 0;
    return {count + extra,
            static_cast<size_t>(coordinate) * count +
                std::min(static_cast<size_t>(coordinate), remainder)};
}

// Choose a process grid that minimizes the slowest rank's work and then the
// total inter-rank interface area.  The final tie-break favors Z/Y cuts because
// their faces are cheaper to pack from the X-major storage layout.
std::array<int, 3> chooseProcessGrid(const int processCount, const size_t nx,
                                     const size_t ny, const size_t nz) {
    std::array<int, 3> best = {0, 0, 0};
    long double bestScore = std::numeric_limits<long double>::infinity();

    for (int px = 1; px <= processCount && static_cast<size_t>(px) <= nx; ++px) {
        if (processCount % px != 0) continue;
        const int yz = processCount / px;
        for (int py = 1; py <= yz && static_cast<size_t>(py) <= ny; ++py) {
            if (yz % py != 0) continue;
            const int pz = yz / py;
            if (static_cast<size_t>(pz) > nz) continue;

            const long double maxVolume =
                static_cast<long double>((nx + px - 1) / px) *
                static_cast<long double>((ny + py - 1) / py) *
                static_cast<long double>((nz + pz - 1) / pz);
            const long double interfaces =
                static_cast<long double>(px - 1) * ny * nz +
                static_cast<long double>(py - 1) * nx * nz +
                static_cast<long double>(pz - 1) * nx * ny;
            const long double score = maxVolume + interfaces;

            const bool betterTie = score == bestScore &&
                (pz > best[2] || (pz == best[2] && py > best[1]));
            if (score < bestScore || betterTie) {
                bestScore = score;
                best = {px, py, pz};
            }
        }
    }
    return best;
}

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int processCount = 1;
    std::array<int, 3> dims{};
    std::array<int, 3> coords{};
    std::array<int, 6> neighbor{};
    std::array<size_t, 3> global{};
    std::array<size_t, 3> local{};
    std::array<size_t, 3> offset{};
    size_t rowStride = 0;
    size_t planeStride = 0;

    Domain(const size_t nx, const size_t ny, const size_t nz) : global{nx, ny, nz} {
        MPI_Comm_size(MPI_COMM_WORLD, &processCount);
        dims = chooseProcessGrid(processCount, nx, ny, nz);
        if (dims[0] == 0) {
            if (processCount > 0) {
                int worldRank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
                if (worldRank == 0) {
                    std::fprintf(stderr,
                                 "Error: the %zu x %zu x %zu grid cannot be split among %d nonempty MPI ranks\n",
                                 nx, ny, nz, processCount);
                }
            }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords.data());
        MPI_Cart_shift(comm, 0, 1, &neighbor[X_MINUS], &neighbor[X_PLUS]);
        MPI_Cart_shift(comm, 1, 1, &neighbor[Y_MINUS], &neighbor[Y_PLUS]);
        MPI_Cart_shift(comm, 2, 1, &neighbor[Z_MINUS], &neighbor[Z_PLUS]);

        for (int d = 0; d < 3; ++d) {
            const Extent extent = splitExtent(global[d], dims[d], coords[d]);
            local[d] = extent.size;
            offset[d] = extent.offset;
        }
        rowStride = local[0] + 2;
        planeStride = rowStride * (local[1] + 2);
    }

    ~Domain() {
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    size_t index(const size_t x, const size_t y, const size_t z) const noexcept {
        return z * planeStride + y * rowStride + x;
    }

    size_t allocatedSize() const noexcept { return planeStride * (local[2] + 2); }
    size_t ownedSize() const noexcept { return local[0] * local[1] * local[2]; }
};

class HaloExchange {
  public:
    explicit HaloExchange(const Domain& domain) : domain_(domain) {
        const size_t xFace = domain_.local[1] * domain_.local[2];
        const size_t yFace = domain_.local[0] * domain_.local[2];
        const size_t zFace = domain_.local[0] * domain_.local[1];
        const std::array<size_t, 6> sizes = {xFace, xFace, yFace, yFace, zFace, zFace};

        for (int face = 0; face < 6; ++face) {
            if (sizes[face] > static_cast<size_t>(std::numeric_limits<int>::max())) {
                if (domain_.rank == 0) {
                    std::fprintf(stderr, "Error: an MPI halo exceeds the implementation's count limit\n");
                }
                MPI_Abort(domain_.comm, 2);
            }
            send_[face].resize(sizes[face]);
            receive_[face].resize(sizes[face]);
            if (domain_.neighbor[face] == MPI_PROC_NULL) continue;

            MPI_Request receiveRequest;
            MPI_Request sendRequest;
            MPI_Recv_init(receive_[face].data(), static_cast<int>(sizes[face]), MPI_DOUBLE,
                          domain_.neighbor[face], 100 + oppositeFace(face), domain_.comm,
                          &receiveRequest);
            MPI_Send_init(send_[face].data(), static_cast<int>(sizes[face]), MPI_DOUBLE,
                          domain_.neighbor[face], 100 + face, domain_.comm, &sendRequest);
            requests_.push_back(receiveRequest);
            requests_.push_back(sendRequest);
        }
    }

    ~HaloExchange() {
        for (MPI_Request& request : requests_) MPI_Request_free(&request);
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void begin(std::vector<double>& field) {
        packOrClampX(field);
        packOrClampY(field);
        packOrClampZ(field);
        if (!requests_.empty()) {
            MPI_Startall(static_cast<int>(requests_.size()), requests_.data());
        }
    }

    void end(std::vector<double>& field) {
        if (!requests_.empty()) {
            MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
        }
        unpackX(field);
        unpackY(field);
        unpackZ(field);
    }

  private:
    const Domain& domain_;
    std::array<std::vector<double>, 6> send_;
    std::array<std::vector<double>, 6> receive_;
    std::vector<MPI_Request> requests_;

    void packOrClampX(std::vector<double>& field) {
        size_t p = 0;
        for (size_t z = 1; z <= domain_.local[2]; ++z) {
            for (size_t y = 1; y <= domain_.local[1]; ++y) {
                const size_t base = domain_.index(0, y, z);
                if (domain_.neighbor[X_MINUS] == MPI_PROC_NULL) field[base] = field[base + 1];
                else send_[X_MINUS][p] = field[base + 1];
                if (domain_.neighbor[X_PLUS] == MPI_PROC_NULL)
                    field[base + domain_.local[0] + 1] = field[base + domain_.local[0]];
                else
                    send_[X_PLUS][p] = field[base + domain_.local[0]];
                ++p;
            }
        }
    }

    void packOrClampY(std::vector<double>& field) {
        size_t p = 0;
        for (size_t z = 1; z <= domain_.local[2]; ++z) {
            const size_t lower = domain_.index(1, 0, z);
            const size_t upper = domain_.index(1, domain_.local[1], z);
            if (domain_.neighbor[Y_MINUS] == MPI_PROC_NULL)
                std::copy_n(field.data() + lower + domain_.rowStride, domain_.local[0],
                            field.data() + lower);
            else {
                std::copy_n(field.data() + lower + domain_.rowStride, domain_.local[0],
                            send_[Y_MINUS].data() + p);
            }
            if (domain_.neighbor[Y_PLUS] == MPI_PROC_NULL)
                std::copy_n(field.data() + upper, domain_.local[0],
                            field.data() + upper + domain_.rowStride);
            else
                std::copy_n(field.data() + upper, domain_.local[0], send_[Y_PLUS].data() + p);
            p += domain_.local[0];
        }
    }

    void packOrClampZ(std::vector<double>& field) {
        size_t p = 0;
        for (size_t y = 1; y <= domain_.local[1]; ++y) {
            const size_t lower = domain_.index(1, y, 0);
            const size_t upper = domain_.index(1, y, domain_.local[2]);
            if (domain_.neighbor[Z_MINUS] == MPI_PROC_NULL)
                std::copy_n(field.data() + lower + domain_.planeStride, domain_.local[0],
                            field.data() + lower);
            else
                std::copy_n(field.data() + lower + domain_.planeStride, domain_.local[0],
                            send_[Z_MINUS].data() + p);
            if (domain_.neighbor[Z_PLUS] == MPI_PROC_NULL)
                std::copy_n(field.data() + upper, domain_.local[0],
                            field.data() + upper + domain_.planeStride);
            else
                std::copy_n(field.data() + upper, domain_.local[0], send_[Z_PLUS].data() + p);
            p += domain_.local[0];
        }
    }

    void unpackX(std::vector<double>& field) {
        size_t p = 0;
        for (size_t z = 1; z <= domain_.local[2]; ++z) {
            for (size_t y = 1; y <= domain_.local[1]; ++y) {
                const size_t base = domain_.index(0, y, z);
                if (domain_.neighbor[X_MINUS] != MPI_PROC_NULL) field[base] = receive_[X_MINUS][p];
                if (domain_.neighbor[X_PLUS] != MPI_PROC_NULL)
                    field[base + domain_.local[0] + 1] = receive_[X_PLUS][p];
                ++p;
            }
        }
    }

    void unpackY(std::vector<double>& field) {
        size_t p = 0;
        for (size_t z = 1; z <= domain_.local[2]; ++z) {
            const size_t lower = domain_.index(1, 0, z);
            const size_t upper = domain_.index(1, domain_.local[1] + 1, z);
            if (domain_.neighbor[Y_MINUS] != MPI_PROC_NULL)
                std::copy_n(receive_[Y_MINUS].data() + p, domain_.local[0], field.data() + lower);
            if (domain_.neighbor[Y_PLUS] != MPI_PROC_NULL)
                std::copy_n(receive_[Y_PLUS].data() + p, domain_.local[0], field.data() + upper);
            p += domain_.local[0];
        }
    }

    void unpackZ(std::vector<double>& field) {
        size_t p = 0;
        for (size_t y = 1; y <= domain_.local[1]; ++y) {
            const size_t lower = domain_.index(1, y, 0);
            const size_t upper = domain_.index(1, y, domain_.local[2] + 1);
            if (domain_.neighbor[Z_MINUS] != MPI_PROC_NULL)
                std::copy_n(receive_[Z_MINUS].data() + p, domain_.local[0], field.data() + lower);
            if (domain_.neighbor[Z_PLUS] != MPI_PROC_NULL)
                std::copy_n(receive_[Z_PLUS].data() + p, domain_.local[0], field.data() + upper);
            p += domain_.local[0];
        }
    }
};

template <class Operation>
inline void computeInterior(const Domain& domain, Operation&& operation) {
    for (size_t z = 2; z < domain.local[2]; ++z)
        for (size_t y = 2; y < domain.local[1]; ++y)
            for (size_t x = 2; x < domain.local[0]; ++x) operation(x, y, z);
}

template <class Operation>
inline void computeBoundary(const Domain& domain, Operation&& operation) {
    const size_t nx = domain.local[0];
    const size_t ny = domain.local[1];
    const size_t nz = domain.local[2];

    // X faces include their edges and corners.
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t y = 1; y <= ny; ++y) {
            operation(1, y, z);
            if (nx > 1) operation(nx, y, z);
        }
    }
    // Y faces omit the X-face points already handled above.
    for (size_t z = 1; z <= nz; ++z) {
        for (size_t x = 2; x < nx; ++x) {
            operation(x, 1, z);
            if (ny > 1) operation(x, ny, z);
        }
    }
    // Z faces omit both sets of points already handled above.
    for (size_t y = 2; y < ny; ++y) {
        for (size_t x = 2; x < nx; ++x) {
            operation(x, y, 1);
            if (nz > 1) operation(x, y, nz);
        }
    }
}

inline double laplacianAt(const std::vector<double>& field, const Domain& domain,
                          const size_t x, const size_t y, const size_t z,
                          const double dx, const double dy, const double dz) noexcept {
    const size_t i = domain.index(x, y, z);
    const double center = field[i];
    const double cxx = (field[i + 1] + field[i - 1] - 2.0 * center) / (dx * dx);
    const double cyy =
        (field[i + domain.rowStride] + field[i - domain.rowStride] - 2.0 * center) /
        (dy * dy);
    const double czz =
        (field[i + domain.planeStride] + field[i - domain.planeStride] - 2.0 * center) /
        (dz * dz);
    return cxx + cyy + czz;
}

void computeChemicalPotential(std::vector<double>& concentration, std::vector<double>& mu,
                              HaloExchange& exchange, const Domain& domain, const double dx,
                              const double dy, const double dz, const double gamma,
                              const double eAA, const double eBB, const double eAB) {
    const auto point = [&](const size_t x, const size_t y, const size_t z) {
        const size_t i = domain.index(x, y, z);
        const double c = concentration[i];
        mu[i] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) +
                3.0 * c + c * c * c -
                gamma * laplacianAt(concentration, domain, x, y, z, dx, dy, dz);
    };

    exchange.begin(concentration);
    computeInterior(domain, point);
    exchange.end(concentration);
    computeBoundary(domain, point);
}

void cahnHilliardUpdate(std::vector<double>& next, const std::vector<double>& current,
                        std::vector<double>& mu, HaloExchange& exchange, const Domain& domain,
                        const double diffusion, const double dt, const double dx,
                        const double dy, const double dz) {
    const auto point = [&](const size_t x, const size_t y, const size_t z) {
        const size_t i = domain.index(x, y, z);
        next[i] = current[i] + dt * diffusion * laplacianAt(mu, domain, x, y, z, dx, dy, dz);
    };

    exchange.begin(mu);
    computeInterior(domain, point);
    exchange.end(mu);
    computeBoundary(domain, point);
}

void initializeConcentration(std::vector<double>& concentration, const Domain& domain) {
    const size_t volume = domain.global[0] * domain.global[1] * domain.global[2];
    for (size_t z = 1; z <= domain.local[2]; ++z) {
        const size_t globalZ = domain.offset[2] + z - 1;
        for (size_t y = 1; y <= domain.local[1]; ++y) {
            const size_t globalY = domain.offset[1] + y - 1;
            for (size_t x = 1; x <= domain.local[0]; ++x) {
                const size_t globalX = domain.offset[0] + x - 1;
                const size_t linearId =
                    globalZ * (domain.global[0] * domain.global[1]) +
                    globalY * domain.global[0] + globalX;
                const double pseudo =
                    (((linearId + 1) * 1299709) % volume) / static_cast<double>(volume);
                concentration[domain.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration, const Domain& domain) {
    int localFinite = 1;
    double localMinimum = std::numeric_limits<double>::infinity();
    double localMaximum = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= domain.local[2]; ++z) {
        for (size_t y = 1; y <= domain.local[1]; ++y) {
            for (size_t x = 1; x <= domain.local[0]; ++x) {
                const double value = concentration[domain.index(x, y, z)];
                if (!std::isfinite(value)) localFinite = 0;
                localMinimum = std::min(localMinimum, value);
                localMaximum = std::max(localMaximum, value);
            }
        }
    }

    int finite = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_LAND, domain.comm);
    MPI_Allreduce(&localMinimum, &minimum, 1, MPI_DOUBLE, MPI_MIN, domain.comm);
    MPI_Allreduce(&localMaximum, &maximum, 1, MPI_DOUBLE, MPI_MAX, domain.comm);

    if (!finite) {
        if (domain.rank == 0) std::printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (domain.rank == 0) std::printf("Concentration range: [%.6f, %.6f]\n", minimum, maximum);
    if (maximum > 10.0 || minimum < -10.0) {
        if (domain.rank == 0) std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

std::vector<double> gatherGlobalResult(const std::vector<double>& localField,
                                       const Domain& domain) {
    if (domain.ownedSize() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (domain.rank == 0) std::fprintf(stderr, "Error: local result exceeds MPI_Gatherv count limit\n");
        MPI_Abort(domain.comm, 2);
    }
    const int localCount = static_cast<int>(domain.ownedSize());
    std::vector<double> packed(domain.ownedSize());
    size_t p = 0;
    for (size_t z = 1; z <= domain.local[2]; ++z)
        for (size_t y = 1; y <= domain.local[1]; ++y) {
            std::copy_n(localField.data() + domain.index(1, y, z), domain.local[0],
                        packed.data() + p);
            p += domain.local[0];
        }

    std::vector<int> counts;
    std::vector<int> displacements;
    if (domain.rank == 0) {
        counts.resize(domain.processCount);
        displacements.resize(domain.processCount);
    }
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, domain.comm);

    std::vector<double> rankOrdered;
    if (domain.rank == 0) {
        long long total = 0;
        for (int r = 0; r < domain.processCount; ++r) {
            if (total > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "Error: global result exceeds MPI_Gatherv displacement limit\n");
                MPI_Abort(domain.comm, 2);
            }
            displacements[r] = static_cast<int>(total);
            total += counts[r];
        }
        if (total > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "Error: global result exceeds MPI_Gatherv count limit\n");
            MPI_Abort(domain.comm, 2);
        }
        rankOrdered.resize(static_cast<size_t>(total));
    }

    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE, rankOrdered.data(), counts.data(),
                displacements.data(), MPI_DOUBLE, 0, domain.comm);

    std::vector<double> globalField;
    if (domain.rank == 0) {
        globalField.resize(domain.global[0] * domain.global[1] * domain.global[2]);
        for (int r = 0; r < domain.processCount; ++r) {
            std::array<int, 3> coordinates{};
            MPI_Cart_coords(domain.comm, r, 3, coordinates.data());
            const Extent ex = splitExtent(domain.global[0], domain.dims[0], coordinates[0]);
            const Extent ey = splitExtent(domain.global[1], domain.dims[1], coordinates[1]);
            const Extent ez = splitExtent(domain.global[2], domain.dims[2], coordinates[2]);
            size_t source = static_cast<size_t>(displacements[r]);
            for (size_t z = 0; z < ez.size; ++z) {
                for (size_t y = 0; y < ey.size; ++y) {
                    const size_t destination =
                        idx3(ex.offset, ey.offset + y, ez.offset + z, domain.global[0],
                             domain.global[1]);
                    std::copy_n(rankOrdered.data() + source, ex.size,
                                globalField.data() + destination);
                    source += ex.size;
                }
            }
        }
    }
    return globalField;
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool help = false;
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
            help = true;
        } else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            argumentsValid = false;
            break;
        }
    }

    if (help || !argumentsValid) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (worldRank == 0) std::fprintf(stderr, "Error: grid dimensions and iterations must be valid\n");
        MPI_Finalize();
        return 1;
    }

    int returnCode = 0;
    {
        Domain domain(nx, ny, nz);
        if (domain.rank == 0) {
            std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
            std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            std::printf("Time steps: %d\n", iterations);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("MPI processes: %d (%d x %d x %d process grid)\n", domain.processCount,
                        domain.dims[0], domain.dims[1], domain.dims[2]);
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
        const size_t globalSize = nx * ny * nz;

        std::vector<double> current(domain.allocatedSize());
        std::vector<double> next(domain.allocatedSize());
        std::vector<double> mu(domain.allocatedSize());

        if (domain.rank == 0) std::printf("Initializing concentration field...\n");
        initializeConcentration(current, domain);
        if (domain.rank == 0) std::printf("Running Cahn-Hilliard simulation...\n");

        double localSeconds = 0.0;
        {
            HaloExchange exchange(domain);
            MPI_Barrier(domain.comm);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                computeChemicalPotential(current, mu, exchange, domain, dx, dy, dz, gamma, eAA,
                                         eBB, eAB);
                cahnHilliardUpdate(next, current, mu, exchange, domain, diffusion, dt, dx, dy, dz);
                std::swap(current, next);
            }
            localSeconds = MPI_Wtime() - start;
        }

        double seconds = 0.0;
        MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
        if (domain.rank == 0) {
            const long milliseconds = static_cast<long>(seconds * 1000.0);
            const double cellUpdates = static_cast<double>(globalSize) * iterations;
            const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1.0e6 : 0.0;
            std::printf("Computation time: %ld ms\n", milliseconds);
            std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            std::vector<double> globalResult = gatherGlobalResult(current, domain);
            if (domain.rank == 0) print_results(globalResult, "Concentration");
        }

        if (validate) {
            if (domain.rank == 0) std::printf("Validating result...\n");
            const bool valid = validateResult(current, domain);
            if (domain.rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) returnCode = 1;
        }
    }

    MPI_Finalize();
    return returnCode;
}
