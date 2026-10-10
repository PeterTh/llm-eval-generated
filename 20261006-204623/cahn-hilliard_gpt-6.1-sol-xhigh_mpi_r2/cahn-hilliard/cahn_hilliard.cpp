#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

struct Box {
    size_t x0, x1, y0, y1, z0, z1;
};

// Prefer small interface areas and balanced blocks, including for thin grids.
// If the process count cannot form nonempty Cartesian blocks, use the largest
// feasible subset; surplus ranks still participate in MPI initialization/finalization.
int chooseProcessGrid(const std::array<size_t, 3>& global, int processes,
                      std::array<int, 3>& dims) {
    const size_t volume = global[0] * global[1] * global[2];
    for (int count = static_cast<int>(std::min(volume, static_cast<size_t>(processes)));
         count > 0; --count) {
        std::vector<int> divisors;
        for (int d = 1; d <= count / d; ++d) {
            if (count % d == 0) {
                divisors.push_back(d);
                if (d != count / d) divisors.push_back(count / d);
            }
        }
        std::sort(divisors.begin(), divisors.end());
        double best = std::numeric_limits<double>::infinity();
        for (int px : divisors) {
            if (static_cast<size_t>(px) > global[0]) continue;
            const int rest = count / px;
            for (int py : divisors) {
                if (rest % py != 0 || static_cast<size_t>(py) > global[1]) continue;
                const int pz = rest / py;
                if (static_cast<size_t>(pz) > global[2]) continue;
                const double surface = (px - 1) / static_cast<double>(global[0])
                                     + (py - 1) / static_cast<double>(global[1])
                                     + (pz - 1) / static_cast<double>(global[2]);
                const double largest = static_cast<double>((global[0] + px - 1) / px)
                                     * ((global[1] + py - 1) / py)
                                     * ((global[2] + pz - 1) / pz);
                const double imbalance = largest / (static_cast<double>(volume) / count);
                const double score = surface * imbalance;
                // Stable ties favor longer contiguous X rows.
                if (score < best) {
                    best = score;
                    dims = {px, py, pz};
                }
            }
        }
        if (std::isfinite(best)) return count;
    }
    return 1;
}

void blockExtent(size_t global, int parts, int coordinate,
                 size_t& length, size_t& offset) {
    const size_t base = global / parts;
    const size_t remainder = global % parts;
    length = base + (static_cast<size_t>(coordinate) < remainder);
    offset = static_cast<size_t>(coordinate) * base
           + std::min(static_cast<size_t>(coordinate), remainder);
}

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, ranks = 1;
    std::array<int, 3> dims{}, coordinates{};
    std::array<size_t, 3> global{}, n{}, offset{};
    std::array<int, 6> neighbors{};
    std::array<MPI_Datatype, 3> faces{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL,
                                      MPI_DATATYPE_NULL};
    std::array<size_t, 6> sendOffsets{}, recvOffsets{};
    size_t pitch = 0, plane = 0, storage = 0;
    Box interior{};
    std::vector<Box> boundary;

    Domain(MPI_Comm active, const std::array<size_t, 3>& grid,
           const std::array<int, 3>& processGrid) : dims(processGrid), global(grid) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims.data(), periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        MPI_Cart_coords(comm, rank, 3, coordinates.data());
        for (int axis = 0; axis < 3; ++axis) {
            blockExtent(global[axis], dims[axis], coordinates[axis], n[axis], offset[axis]);
            MPI_Cart_shift(comm, axis, 1, &neighbors[2 * axis], &neighbors[2 * axis + 1]);
        }
        pitch = n[0] + 2;
        plane = pitch * (n[1] + 2);
        if (plane > std::vector<double>().max_size() / (n[2] + 2))
            throw std::length_error("Local grid is too large");
        storage = plane * (n[2] + 2);

        const int sizes[3] = {static_cast<int>(n[2] + 2),
                              static_cast<int>(n[1] + 2), static_cast<int>(n[0] + 2)};
        const int starts[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            if (dims[axis] == 1) continue;
            int subsizes[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]),
                               static_cast<int>(n[0])};
            subsizes[2 - axis] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &faces[axis]);
            MPI_Type_commit(&faces[axis]);
        }
        sendOffsets = {index(1, 1, 1), index(n[0], 1, 1),
                       index(1, 1, 1), index(1, n[1], 1),
                       index(1, 1, 1), index(1, 1, n[2])};
        recvOffsets = {index(0, 1, 1), index(n[0] + 1, 1, 1),
                       index(1, 0, 1), index(1, n[1] + 1, 1),
                       index(1, 1, 0), index(1, 1, n[2] + 1)};

        std::array<size_t, 3> low{}, high{};
        for (int axis = 0; axis < 3; ++axis) {
            low[axis] = neighbors[2 * axis] == MPI_PROC_NULL ? 1 : 2;
            high[axis] = neighbors[2 * axis + 1] == MPI_PROC_NULL ? n[axis] + 1 : n[axis];
            high[axis] = std::max(low[axis], high[axis]);
        }
        interior = {low[0], high[0], low[1], high[1], low[2], high[2]};
        // Disjoint shell boxes also work for blocks only one cell wide.
        addBoundary({1, n[0] + 1, 1, n[1] + 1, 1, low[2]});
        addBoundary({1, n[0] + 1, 1, n[1] + 1, high[2], n[2] + 1});
        addBoundary({1, n[0] + 1, 1, low[1], low[2], high[2]});
        addBoundary({1, n[0] + 1, high[1], n[1] + 1, low[2], high[2]});
        addBoundary({1, low[0], low[1], high[1], low[2], high[2]});
        addBoundary({high[0], n[0] + 1, low[1], high[1], low[2], high[2]});
    }

    ~Domain() {
        for (auto& face : faces)
            if (face != MPI_DATATYPE_NULL) MPI_Type_free(&face);
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
    }

    size_t index(size_t x, size_t y, size_t z) const noexcept {
        return z * plane + y * pitch + x;
    }

    void addBoundary(const Box& box) {
        if (box.x0 < box.x1 && box.y0 < box.y1 && box.z0 < box.z1)
            boundary.push_back(box);
    }

    int beginExchange(std::vector<double>& field, MPI_Request* requests) const {
        int count = 0;
        // Receive directly into ghost faces; no packing buffers or per-step allocation.
        for (int side = 0; side < 6; ++side) {
            if (neighbors[side] != MPI_PROC_NULL)
                MPI_Irecv(field.data() + recvOffsets[side], 1, faces[side / 2],
                          neighbors[side], side ^ 1, comm, &requests[count++]);
        }
        for (int side = 0; side < 6; ++side) {
            if (neighbors[side] != MPI_PROC_NULL)
                MPI_Isend(field.data() + sendOffsets[side], 1, faces[side / 2],
                          neighbors[side], side, comm, &requests[count++]);
        }
        // Replicate physical edge values for the original clamped Laplacian.
        // Only axial faces are needed; the stencil never reads ghost edges/corners.
        for (int side = 0; side < 2; ++side) {
            if (neighbors[side] == MPI_PROC_NULL) {
                const size_t from = side == 0 ? 1 : n[0];
                const size_t to = side == 0 ? 0 : n[0] + 1;
                for (size_t z = 1; z <= n[2]; ++z)
                    for (size_t y = 1; y <= n[1]; ++y)
                        field[index(to, y, z)] = field[index(from, y, z)];
            }
            if (neighbors[2 + side] == MPI_PROC_NULL) {
                const size_t from = side == 0 ? 1 : n[1];
                const size_t to = side == 0 ? 0 : n[1] + 1;
                for (size_t z = 1; z <= n[2]; ++z)
                    std::copy_n(field.data() + index(1, from, z), n[0],
                                field.data() + index(1, to, z));
            }
            if (neighbors[4 + side] == MPI_PROC_NULL) {
                const size_t from = side == 0 ? 1 : n[2];
                const size_t to = side == 0 ? 0 : n[2] + 1;
                for (size_t y = 1; y <= n[1]; ++y)
                    std::copy_n(field.data() + index(1, y, from), n[0],
                                field.data() + index(1, y, to));
            }
        }
        return count;
    }
};

// The benchmark's physical parameters are fixed. Keep the original operation
// order, without fast-math, while removing boundary branches from the hot loops.
// Distinct output/input buffers permit SIMD across contiguous X rows.
template <bool Chemical>
void computeBox(const double* __restrict__ source, double* __restrict__ output,
                const double* __restrict__ cold, const Domain& domain, const Box& box) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5, dt = 0.01, D = 1.0;
    const size_t pitch = domain.pitch, plane = domain.plane;
    for (size_t z = box.z0; z < box.z1; ++z) {
        for (size_t y = box.y0; y < box.y1; ++y) {
            const size_t row = z * plane + y * pitch;
            for (size_t x = box.x0; x < box.x1; ++x) {
                const size_t i = row + x;
                const double cv = source[i];
                const double cxx = source[i + 1] + source[i - 1] - 2.0 * cv;
                const double cyy = source[i + pitch] + source[i - pitch] - 2.0 * cv;
                const double czz = source[i + plane] + source[i - plane] - 2.0 * cv;
                const double laplacian = cxx + cyy + czz; // dx = dy = dz = 1
                if constexpr (Chemical) {
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB
                                      - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                } else {
                    output[i] = cold[i] + dt * D * laplacian;
                }
            }
        }
    }
}

template <bool Chemical>
void stencilStep(std::vector<double>& source, std::vector<double>& output,
                 const std::vector<double>& cold, const Domain& domain) {
    MPI_Request requests[12];
    const int count = domain.beginExchange(source, requests);
    computeBox<Chemical>(source.data(), output.data(), cold.data(), domain, domain.interior);
    if (count != 0) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    for (const Box& box : domain.boundary)
        computeBox<Chemical>(source.data(), output.data(), cold.data(), domain, box);
}

void initializeConcentration(std::vector<double>& field, const Domain& domain) {
    const size_t nx = domain.global[0], ny = domain.global[1];
    const size_t volume = nx * ny * domain.global[2];
    for (size_t z = 1; z <= domain.n[2]; ++z) {
        for (size_t y = 1; y <= domain.n[1]; ++y) {
            const size_t globalRow = (domain.offset[2] + z - 1) * (nx * ny)
                                  + (domain.offset[1] + y - 1) * nx + domain.offset[0];
            const size_t localRow = domain.index(1, y, z);
            for (size_t x = 0; x < domain.n[0]; ++x) {
                const size_t linear_id = globalRow + x;
                const double pseudo = (((linear_id + 1) * 1299709) % volume)
                                    / static_cast<double>(volume);
                field[localRow + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gathering is solely for -r: ordinary simulation/validation remains distributed.
// Subarray types place each block directly in original X-fastest global order,
// preserving the existing Kahan sum, sample indices and bytewise hash.
void printDistributedResults(const std::vector<double>& field, const Domain& domain) {
    const int localSizes[3] = {static_cast<int>(domain.n[2] + 2),
                               static_cast<int>(domain.n[1] + 2),
                               static_cast<int>(domain.n[0] + 2)};
    const int localSubsizes[3] = {static_cast<int>(domain.n[2]),
                                  static_cast<int>(domain.n[1]), static_cast<int>(domain.n[0])};
    const int localStarts[3] = {1, 1, 1};
    MPI_Datatype localType;
    MPI_Type_create_subarray(3, localSizes, localSubsizes, localStarts, MPI_ORDER_C,
                             MPI_DOUBLE, &localType);
    MPI_Type_commit(&localType);
    if (domain.rank != 0) {
        MPI_Send(field.data(), 1, localType, 0, 6, domain.comm);
    } else {
        std::vector<double> global(domain.global[0] * domain.global[1] * domain.global[2]);
        for (size_t z = 1; z <= domain.n[2]; ++z)
            for (size_t y = 1; y <= domain.n[1]; ++y) {
                const size_t dest = (domain.offset[2] + z - 1) * domain.global[0] * domain.global[1]
                                  + (domain.offset[1] + y - 1) * domain.global[0] + domain.offset[0];
                std::copy_n(field.data() + domain.index(1, y, z), domain.n[0], global.data() + dest);
            }
        const int sizes[3] = {static_cast<int>(domain.global[2]),
                              static_cast<int>(domain.global[1]), static_cast<int>(domain.global[0])};
        for (int rank = 1; rank < domain.ranks; ++rank) {
            int coordinates[3], subsizes[3], starts[3];
            MPI_Cart_coords(domain.comm, rank, 3, coordinates);
            for (int axis = 0; axis < 3; ++axis) {
                size_t length, offset;
                blockExtent(domain.global[axis], domain.dims[axis], coordinates[axis], length, offset);
                subsizes[2 - axis] = static_cast<int>(length);
                starts[2 - axis] = static_cast<int>(offset);
            }
            MPI_Datatype globalType;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &globalType);
            MPI_Type_commit(&globalType);
            MPI_Recv(global.data(), 1, globalType, rank, 6, domain.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&globalType);
        }
        print_results(global, "Concentration");
    }
    MPI_Type_free(&localType);
}

bool validateResult(const std::vector<double>& field, const Domain& domain) {
    int nonfinite = 0, anyNonfinite = 0;
    double bounds[2] = {std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity()};
    for (size_t z = 1; z <= domain.n[2]; ++z)
        for (size_t y = 1; y <= domain.n[1]; ++y)
            for (size_t x = 1; x <= domain.n[0]; ++x) {
                const double value = field[domain.index(x, y, z)];
                nonfinite |= !std::isfinite(value);
                bounds[0] = std::min(bounds[0], value);
                bounds[1] = std::min(bounds[1], -value);
            }
    double globalBounds[2];
    MPI_Reduce(&nonfinite, &anyNonfinite, 1, MPI_INT, MPI_MAX, 0, domain.comm);
    MPI_Reduce(bounds, globalBounds, 2, MPI_DOUBLE, MPI_MIN, 0, domain.comm);
    if (domain.rank != 0) return true;
    if (anyNonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const double minVal = globalBounds[0], maxVal = -globalBounds[1];
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
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

int run(int argc, char** argv, int worldRank, int worldSize) {
    std::array<size_t, 3> grid{64, 0, 0};
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        const bool dimension = strcmp(option, "-x") == 0 || strcmp(option, "-y") == 0
                            || strcmp(option, "-z") == 0;
        if ((dimension || strcmp(option, "-i") == 0) && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long long value = std::strtoll(argument, &end, 10);
            if (errno != 0 || end == argument || *end != '\0' || value < 0
                || value > (dimension ? INT_MAX - 2LL : INT_MAX)) {
                if (worldRank == 0) fprintf(stderr, "Invalid value for %s: %s\n", option, argument);
                return 1;
            }
            if (dimension) {
                const int axis = strcmp(option, "-x") == 0 ? 0 : (strcmp(option, "-y") == 0 ? 1 : 2);
                grid[axis] = static_cast<size_t>(value);
            } else {
                iterations = static_cast<int>(value);
            }
        } else if (strcmp(option, "-v") == 0) {
            validate = true;
        } else if (strcmp(option, "-r") == 0) {
            printResults = true;
        } else if (strcmp(option, "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", option);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (grid[1] == 0) grid[1] = grid[0];
    if (grid[2] == 0) grid[2] = grid[0];
    const size_t maxElements = std::vector<double>().max_size();
    if (grid[0] == 0 || grid[0] > maxElements / grid[1]
        || grid[0] * grid[1] > maxElements / grid[2]) {
        if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive and fit in addressable memory\n");
        return 1;
    }
    const size_t gridSize = grid[0] * grid[1] * grid[2];
    std::array<int, 3> dims{};
    const int activeRanks = chooseProcessGrid(grid, worldSize, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &active);
    int result = 0;
    if (active != MPI_COMM_NULL) {
        {
            Domain domain(active, grid, dims);
            if (domain.rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", grid[0], grid[1], grid[2]);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("MPI processes: %d (Cartesian grid: %d x %d x %d)\n",
                       activeRanks, dims[0], dims[1], dims[2]);
                printf("Initializing concentration field...\n");
            }
            std::vector<double> cold(domain.storage), cnew(domain.storage), mu(domain.storage);
            initializeConcentration(cold, domain);
            if (domain.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(domain.comm);
            const double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                stencilStep<true>(cold, mu, cold, domain);
                stencilStep<false>(mu, cnew, cold, domain);
                cold.swap(cnew);
            }
            const double elapsed = MPI_Wtime() - start;
            double duration = 0.0;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
            if (domain.rank == 0) {
                printf("Computation time: %.3f ms\n", duration * 1000.0);
                const double mcups = duration > 0.0
                    ? static_cast<double>(gridSize) * iterations / duration / 1e6 : 0.0;
                printf("Performance: %.3f MCellUpdates/s\n", mcups);
            }
            if (printResults) printDistributedResults(cold, domain);
            if (validate) {
                if (domain.rank == 0) printf("Validating result...\n");
                const bool valid = validateResult(cold, domain);
                if (domain.rank == 0) {
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    result = valid ? 0 : 1;
                }
            }
        }
        MPI_Comm_free(&active);
    }
    // Include surplus ranks in the exit status without any timestep collectives.
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 1;
    try {
        result = run(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
