#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {

struct Box {
    size_t xBegin, xEnd, yBegin, yEnd, zBegin, zEnd;
};

// Balance the largest block and halo traffic. Equal-cost grids favor keeping
// the contiguous X direction intact.
int chooseProcessGrid(const std::array<size_t, 3>& global, int ranks,
                      std::array<int, 3>& dimensions) {
    const size_t volume = global[0] * global[1] * global[2];
    int active = static_cast<int>(std::min(volume, static_cast<size_t>(ranks)));
    for (; active > 0; --active) {
        std::vector<int> divisors;
        for (int d = 1; d <= active / d; ++d) {
            if (active % d == 0) {
                divisors.push_back(d);
                if (d != active / d) divisors.push_back(active / d);
            }
        }
        std::sort(divisors.begin(), divisors.end());
        long double best = std::numeric_limits<long double>::infinity();
        for (int px : divisors) {
            if (static_cast<size_t>(px) > global[0]) continue;
            const int remaining = active / px;
            for (int py : divisors) {
                if (remaining % py != 0 || static_cast<size_t>(py) > global[1]) continue;
                const int pz = remaining / py;
                if (static_cast<size_t>(pz) > global[2]) continue;
                const long double largestBlock =
                    static_cast<long double>(global[0] / px + (global[0] % px != 0)) *
                    (global[1] / py + (global[1] % py != 0)) *
                    (global[2] / pz + (global[2] % pz != 0));
                const long double interfaces =
                    static_cast<long double>(px - 1) * global[1] * global[2] +
                    static_cast<long double>(py - 1) * global[0] * global[2] +
                    static_cast<long double>(pz - 1) * global[0] * global[1];
                const long double cost = largestBlock + 4.0L * interfaces / active;
                if (cost < best) {
                    best = cost;
                    dimensions = {px, py, pz};
                }
            }
        }
        if (std::isfinite(best)) return active;
    }
    return 0;
}

struct Domain {
    std::array<size_t, 3> size{}, origin{};
    std::array<int, 6> neighbors{};
    size_t rowStride, planeStride, storage;
    Box interior;
    std::array<Box, 6> boundary;

    Domain(const std::array<size_t, 3>& global, const std::array<int, 3>& dimensions,
           const std::array<int, 3>& coordinates) {
        for (int axis = 0; axis < 3; ++axis) {
            const size_t base = global[axis] / dimensions[axis];
            const size_t remainder = global[axis] % dimensions[axis];
            size[axis] = base + (static_cast<size_t>(coordinates[axis]) < remainder);
            origin[axis] = static_cast<size_t>(coordinates[axis]) * base +
                           std::min(static_cast<size_t>(coordinates[axis]), remainder);
            if (size[axis] > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
                throw std::runtime_error("Local grid dimensions exceed MPI datatype limits");
            }
        }
        rowStride = size[0] + 2;
        const size_t maxElements = std::vector<double>().max_size();
        if (rowStride > maxElements / (size[1] + 2)) {
            throw std::runtime_error("Local grid size is too large");
        }
        planeStride = rowStride * (size[1] + 2);
        if (planeStride > maxElements / (size[2] + 2)) {
            throw std::runtime_error("Local grid size is too large");
        }
        storage = planeStride * (size[2] + 2);
    }

    size_t index(size_t x, size_t y, size_t z) const {
        return z * planeStride + y * rowStride + x;
    }

    void setNeighbors(MPI_Comm communicator) {
        std::array<size_t, 3> begin{}, end{};
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(communicator, axis, 1, &neighbors[2 * axis], &neighbors[2 * axis + 1]);
            begin[axis] = 1 + (neighbors[2 * axis] != MPI_PROC_NULL);
            end[axis] = std::max(begin[axis], size[axis] + 1 -
                                (neighbors[2 * axis + 1] != MPI_PROC_NULL));
        }
        interior = {begin[0], end[0], begin[1], end[1], begin[2], end[2]};
        // Disjoint slabs cover the complement, including blocks only one cell
        // thick. No cell is computed twice at edges or corners.
        boundary = {{
            {1, size[0] + 1, 1, size[1] + 1, 1, begin[2]},
            {1, size[0] + 1, 1, size[1] + 1, end[2], size[2] + 1},
            {1, size[0] + 1, 1, begin[1], begin[2], end[2]},
            {1, size[0] + 1, end[1], size[1] + 1, begin[2], end[2]},
            {1, begin[0], begin[1], end[1], begin[2], end[2]},
            {end[0], size[0] + 1, begin[1], end[1], begin[2], end[2]}
        }};
    }
};

// A strided block type avoids packing entire local fields and supports block
// element counts larger than INT_MAX; only individual dimensions are int.
MPI_Datatype makeBlockType(size_t nx, size_t ny, size_t nz,
                           size_t rowStride, size_t planeStride) {
    MPI_Datatype row, plane, block;
    MPI_Type_contiguous(static_cast<int>(nx), MPI_DOUBLE, &row);
    MPI_Type_create_hvector(static_cast<int>(ny), 1,
                            static_cast<MPI_Aint>(rowStride * sizeof(double)), row, &plane);
    MPI_Type_create_hvector(static_cast<int>(nz), 1,
                            static_cast<MPI_Aint>(planeStride * sizeof(double)), plane, &block);
    MPI_Type_commit(&block);
    MPI_Type_free(&plane);
    MPI_Type_free(&row);
    return block;
}

class HaloExchange {
    double* field;
    const Domain& domain;
    std::array<MPI_Datatype, 3> types{};
    std::array<MPI_Request, 12> requests{};
    int count = 0;
    bool active = false;

    void clampBoundaries() {
        const auto& n = domain.size;
        if (domain.neighbors[0] == MPI_PROC_NULL || domain.neighbors[1] == MPI_PROC_NULL) {
            for (size_t z = 1; z <= n[2]; ++z) {
                for (size_t y = 1; y <= n[1]; ++y) {
                    const size_t row = domain.index(0, y, z);
                    if (domain.neighbors[0] == MPI_PROC_NULL) field[row] = field[row + 1];
                    if (domain.neighbors[1] == MPI_PROC_NULL) field[row + n[0] + 1] = field[row + n[0]];
                }
            }
        }
        for (int side = 0; side < 2; ++side) {
            if (domain.neighbors[2 + side] == MPI_PROC_NULL) {
                const size_t sourceY = side == 0 ? 1 : n[1];
                const size_t haloY = side == 0 ? 0 : n[1] + 1;
                for (size_t z = 1; z <= n[2]; ++z) {
                    std::copy_n(field + domain.index(1, sourceY, z), n[0],
                                field + domain.index(1, haloY, z));
                }
            }
            if (domain.neighbors[4 + side] == MPI_PROC_NULL) {
                const size_t sourceZ = side == 0 ? 1 : n[2];
                const size_t haloZ = side == 0 ? 0 : n[2] + 1;
                for (size_t y = 1; y <= n[1]; ++y) {
                    std::copy_n(field + domain.index(1, y, sourceZ), n[0],
                                field + domain.index(1, y, haloZ));
                }
            }
        }
    }

public:
    HaloExchange(std::vector<double>& values, const Domain& layout,
                 MPI_Comm communicator, int tagBase) : field(values.data()), domain(layout) {
        for (int axis = 0; axis < 3; ++axis) {
            if (domain.neighbors[2 * axis] == MPI_PROC_NULL &&
                domain.neighbors[2 * axis + 1] == MPI_PROC_NULL) {
                types[axis] = MPI_DATATYPE_NULL;
                continue;
            }
            auto face = domain.size;
            face[axis] = 1;
            types[axis] = makeBlockType(face[0], face[1], face[2],
                                       domain.rowStride, domain.planeStride);
        }
        // Persistent requests stay bound to the same allocations across
        // ping-pong steps; register all receives ahead of sends.
        for (int sending = 0; sending < 2; ++sending) {
            for (int direction = 0; direction < 6; ++direction) {
                const int neighbor = domain.neighbors[direction];
                if (neighbor == MPI_PROC_NULL) continue;
                const int axis = direction / 2;
                const int side = direction % 2;
                std::array<size_t, 3> position{1, 1, 1};
                position[axis] = sending ? (side == 0 ? 1 : domain.size[axis]) :
                                          (side == 0 ? 0 : domain.size[axis] + 1);
                double* address = field + domain.index(position[0], position[1], position[2]);
                if (sending) {
                    MPI_Send_init(address, 1, types[axis], neighbor, tagBase + direction,
                                  communicator, &requests[count++]);
                } else {
                    MPI_Recv_init(address, 1, types[axis], neighbor, tagBase + (direction ^ 1),
                                  communicator, &requests[count++]);
                }
            }
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        finish();
        for (int i = 0; i < count; ++i) MPI_Request_free(&requests[i]);
        for (auto& type : types) {
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
        }
    }

    void begin() {
        clampBoundaries();
        if (count != 0) {
            MPI_Startall(count, requests.data());
            active = true;
        }
    }

    void progress() {
        if (active) {
            int complete;
            MPI_Testall(count, requests.data(), &complete, MPI_STATUSES_IGNORE);
            if (complete) active = false;
        }
    }

    void finish() {
        if (active) {
            MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
            active = false;
        }
    }
};

// Padded halos remove per-cell boundary branches and make the innermost loop
// contiguous and vectorizable. Preserve the original floating-point expression
// order, including the three separate second differences.
template <bool ChemicalPotential>
void computeBox(double* __restrict__ output, const double* __restrict__ input,
                const double* __restrict__ concentration, const Domain& domain,
                const Box& box, HaloExchange* exchange = nullptr) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double dt = 0.01;
    constexpr double D = 1.0;
    const size_t rowStride = domain.rowStride;
    const size_t planeStride = domain.planeStride;
    if (box.xBegin >= box.xEnd || box.yBegin >= box.yEnd || box.zBegin >= box.zEnd) return;
    for (size_t z = box.zBegin; z < box.zEnd; ++z) {
        for (size_t y = box.yBegin; y < box.yEnd; ++y) {
            const size_t row = z * planeStride + y * rowStride;
            for (size_t x = box.xBegin; x < box.xEnd; ++x) {
                const size_t i = row + x;
                const double cv = input[i];
                const double cxx = input[i + 1] + input[i - 1] - 2.0 * cv;
                const double cyy = input[i + rowStride] + input[i - rowStride] - 2.0 * cv;
                const double czz = input[i + planeStride] + input[i - planeStride] - 2.0 * cv;
                const double laplacian = cxx + cyy + czz;
                if constexpr (ChemicalPotential) {
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                } else {
                    output[i] = concentration[i] + dt * D * laplacian;
                }
            }
        }
        // Drive progress on MPI implementations without background progress.
        if (exchange != nullptr && ((z - box.zBegin) & 7) == 7) exchange->progress();
    }
}

template <bool ChemicalPotential>
void computeBoundary(double* output, const double* input, const double* concentration,
                     const Domain& domain) {
    for (const auto& box : domain.boundary) {
        computeBox<ChemicalPotential>(output, input, concentration, domain, box);
    }
}

void initializeConcentration(std::vector<double>& values, const Domain& domain,
                             const std::array<size_t, 3>& global) {
    const size_t volume = global[0] * global[1] * global[2];
    for (size_t z = 1; z <= domain.size[2]; ++z) {
        for (size_t y = 1; y <= domain.size[1]; ++y) {
            const size_t globalRow = (domain.origin[2] + z - 1) * global[0] * global[1] +
                                    (domain.origin[1] + y - 1) * global[0] + domain.origin[0];
            const size_t localRow = domain.index(1, y, z);
            for (size_t x = 0; x < domain.size[0]; ++x) {
                const size_t linearId = globalRow + x;
                const double pseudo = (((linearId + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                values[localRow + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather only for -r, directly into the original global ordering so the existing
// Kahan sum, samples and byte hash are preserved.
void printDistributedResults(const std::vector<double>& values, const Domain& domain,
                             const std::array<size_t, 3>& global,
                             const std::array<int, 3>& dimensions, MPI_Comm communicator) {
    int rank, ranks;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &ranks);
    constexpr int resultTag = 12;
    if (rank != 0) {
        MPI_Datatype block = makeBlockType(domain.size[0], domain.size[1], domain.size[2],
                                           domain.rowStride, domain.planeStride);
        MPI_Send(values.data() + domain.index(1, 1, 1), 1, block, 0, resultTag, communicator);
        MPI_Type_free(&block);
        return;
    }
    std::vector<double> result(global[0] * global[1] * global[2]);
    const size_t globalPlane = global[0] * global[1];
    for (size_t z = 0; z < domain.size[2]; ++z) {
        for (size_t y = 0; y < domain.size[1]; ++y) {
            const size_t destination = (domain.origin[2] + z) * globalPlane +
                                       (domain.origin[1] + y) * global[0] + domain.origin[0];
            std::copy_n(values.data() + domain.index(1, y + 1, z + 1), domain.size[0],
                        result.data() + destination);
        }
    }
    for (int source = 1; source < ranks; ++source) {
        std::array<int, 3> coordinates;
        MPI_Cart_coords(communicator, source, 3, coordinates.data());
        const Domain blockDomain(global, dimensions, coordinates);
        MPI_Datatype block = makeBlockType(blockDomain.size[0], blockDomain.size[1], blockDomain.size[2],
                                           global[0], globalPlane);
        const size_t destination = blockDomain.origin[2] * globalPlane +
                                   blockDomain.origin[1] * global[0] + blockDomain.origin[0];
        MPI_Recv(result.data() + destination, 1, block, source, resultTag,
                 communicator, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(result, "Concentration");
}

bool validateResult(const std::vector<double>& values, const Domain& domain, MPI_Comm communicator) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    int nonfinite = 0;
    for (size_t z = 1; z <= domain.size[2]; ++z) {
        for (size_t y = 1; y <= domain.size[1]; ++y) {
            const size_t row = domain.index(1, y, z);
            for (size_t x = 0; x < domain.size[0]; ++x) {
                const double value = values[row + x];
                if (!std::isfinite(value)) {
                    nonfinite = 1;
                } else {
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
            }
        }
    }
    // Negate the maximum to combine all validation statistics in one reduction.
    // Halos are excluded from every statistic.
    const double local[3] = {minimum, -maximum, nonfinite ? -1.0 : 0.0};
    double summary[3];
    MPI_Reduce(local, summary, 3, MPI_DOUBLE, MPI_MIN, 0, communicator);
    int rank;
    MPI_Comm_rank(communicator, &rank);
    if (rank != 0) return true;
    if (summary[2] < 0.0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Concentration range: [%.6f, %.6f]\n", summary[0], -summary[1]);
    if (-summary[1] > 10.0 || summary[0] < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int runSimulation(const std::array<size_t, 3>& global, const std::array<int, 3>& dimensions,
                  int iterations, bool validate, bool printResults, MPI_Comm communicator) {
    int rank;
    MPI_Comm_rank(communicator, &rank);
    std::array<int, 3> coordinates;
    MPI_Cart_coords(communicator, rank, 3, coordinates.data());
    Domain domain(global, dimensions, coordinates);
    domain.setNeighbors(communicator);
    std::array<std::vector<double>, 2> concentration{
        std::vector<double>(domain.storage), std::vector<double>(domain.storage)};
    std::vector<double> mu(domain.storage);
    HaloExchange concentrationHalo0(concentration[0], domain, communicator, 0);
    HaloExchange concentrationHalo1(concentration[1], domain, communicator, 0);
    HaloExchange muHalo(mu, domain, communicator, 6);
    std::array<HaloExchange*, 2> concentrationHalos{&concentrationHalo0, &concentrationHalo1};

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(concentration[0], domain, global);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(communicator);
    const double start = MPI_Wtime();
    int current = 0;
    for (int t = 0; t < iterations; ++t) {
        auto& cold = concentration[current];
        auto& cnew = concentration[1 - current];
        auto& coldHalo = *concentrationHalos[current];
        coldHalo.begin();
        computeBox<true>(mu.data(), cold.data(), nullptr, domain, domain.interior, &coldHalo);
        coldHalo.finish();
        computeBoundary<true>(mu.data(), cold.data(), nullptr, domain);

        muHalo.begin();
        computeBox<false>(cnew.data(), mu.data(), cold.data(), domain, domain.interior, &muHalo);
        muHalo.finish();
        computeBoundary<false>(cnew.data(), mu.data(), cold.data(), domain);
        current = 1 - current;
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, communicator);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double cellUpdates = static_cast<double>(global[0] * global[1] * global[2]) * iterations;
        const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    const auto& result = concentration[current];
    if (printResults) printDistributedResults(result, domain, global, dimensions, communicator);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(result, domain, communicator);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
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

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive and the volume must fit in size_t\n");
        MPI_Finalize();
        return 1;
    }

    int exitCode = 0;
    try {
        const std::array<size_t, 3> global{nx, ny, nz};
        std::array<int, 3> dimensions;
        const int activeRanks = chooseProcessGrid(global, ranks, dimensions);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI process grid: %d x %d x %d (%d ranks)\n",
                   dimensions[0], dimensions[1], dimensions[2], activeRanks);
        }
        MPI_Comm activeCommunicator;
        MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED,
                       rank, &activeCommunicator);
        if (activeCommunicator != MPI_COMM_NULL) {
            MPI_Comm cartesian;
            const int periodic[3] = {0, 0, 0};
            MPI_Cart_create(activeCommunicator, 3, dimensions.data(), periodic, 0, &cartesian);
            exitCode = runSimulation(global, dimensions, iterations, validate, printResults, cartesian);
            MPI_Comm_free(&cartesian);
            MPI_Comm_free(&activeCommunicator);
        }
        // Idle ranks (when the grid is too small) receive the same validation
        // status and participate in finalization.
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
