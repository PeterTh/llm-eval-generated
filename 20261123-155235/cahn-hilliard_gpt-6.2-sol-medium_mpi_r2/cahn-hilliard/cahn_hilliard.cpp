#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Index = std::array<int, 3>;

struct Box {
    Index begin;
    Index end;

    Index extent() const {
        return {end[0] - begin[0], end[1] - begin[1], end[2] - begin[2]};
    }
};

struct Strides {
    size_t row;
    size_t plane;

    size_t offset(const Index& point) const {
        return static_cast<size_t>(point[2]) * plane + point[1] * row + point[0];
    }
};

enum MessageTag { ConcentrationHalo, PotentialHalo, FinalBlock };

size_t multiply(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw std::length_error("grid size exceeds addressable memory");
    }
    return a * b;
}

// Try the largest Cartesian grid that fits, minimizing the largest block's
// volume and the amount of data exchanged across its interfaces.
Index processGrid(const Index& grid, size_t volume, int ranks) {
    for (int count = static_cast<int>(std::min(volume, static_cast<size_t>(ranks)));
         count >= 1; --count) {
        Index best{};
        double bestCost = std::numeric_limits<double>::infinity();
        for (int px = 1; px <= std::min(grid[0], count); ++px) {
            if (count % px != 0) continue;
            const int yz = count / px;
            for (int py = 1; py <= std::min(grid[1], yz); ++py) {
                if (yz % py != 0) continue;
                const int pz = yz / py;
                if (pz > grid[2]) continue;
                const Index parts{px, py, pz};
                double largestBlock = 1.0;
                double interfaceArea = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    largestBlock *= grid[axis] / parts[axis] + (grid[axis] % parts[axis] != 0);
                    interfaceArea += static_cast<double>(parts[axis] - 1) / grid[axis];
                }
                const double cost = largestBlock + 4.0 * (volume / static_cast<double>(count)) * interfaceArea;
                if (cost < bestCost) {
                    best = parts;
                    bestCost = cost;
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

Box partition(const Index& grid, const Index& parts, const Index& coords) {
    Box block;
    for (int axis = 0; axis < 3; ++axis) {
        const int width = grid[axis] / parts[axis];
        const int remainder = grid[axis] % parts[axis];
        block.begin[axis] = coords[axis] * width + std::min(coords[axis], remainder);
        block.end[axis] = block.begin[axis] + width + (coords[axis] < remainder);
    }
    return block;
}

// Describe a rectangular region starting at the supplied buffer address.
// Strides in bytes keep the offsets valid for grids larger than INT_MAX cells.
MPI_Datatype boxType(const Index& extent, const Strides& strides) {
    MPI_Datatype row, plane, box;
    MPI_Type_contiguous(extent[0], MPI_DOUBLE, &row);
    MPI_Type_create_hvector(extent[1], 1, strides.row * sizeof(double), row, &plane);
    MPI_Type_create_hvector(extent[2], 1, strides.plane * sizeof(double), plane, &box);
    MPI_Type_commit(&box);
    MPI_Type_free(&plane);
    MPI_Type_free(&row);
    return box;
}

class HaloExchange {
public:
    HaloExchange(MPI_Comm comm, const Box& owned, Strides strides,
                 const std::array<double*, 3>& fields)
        : extent(owned.extent()), strides(strides), interior(owned) {
        const std::array<size_t, 3> steps{1, strides.row, strides.plane};
        for (int axis = 0; axis < 3; ++axis) {
            int neighbors[2];
            MPI_Cart_shift(comm, axis, 1, &neighbors[0], &neighbors[1]);
            if (neighbors[0] == MPI_PROC_NULL && neighbors[1] == MPI_PROC_NULL) continue;

            Index faceExtent = extent;
            faceExtent[axis] = 1;
            MPI_Datatype type = boxType(faceExtent, strides);
            for (int side = 0; side < 2; ++side) {
                const int face = 2 * axis + side;
                if (neighbors[side] == MPI_PROC_NULL) continue;
                remote[face] = true;
                Index send = owned.begin;
                send[axis] = side == 0 ? owned.begin[axis] : owned.end[axis] - 1;
                const size_t step = steps[axis];
                for (size_t field = 0; field < fields.size(); ++field) {
                    double* faceData = fields[field] + strides.offset(send);
                    double* halo = side == 0 ? faceData - step : faceData + step;
                    const int tag = field == 2 ? PotentialHalo : ConcentrationHalo;
                    MPI_Recv_init(halo, 1, type, neighbors[side], tag, comm,
                                  &requests[field][2 * count]);
                    MPI_Send_init(faceData, 1, type, neighbors[side], tag, comm,
                                  &requests[field][2 * count + 1]);
                }
                ++count;
            }
            MPI_Type_free(&type);

            if (remote[2 * axis]) ++interior.begin[axis];
            if (remote[2 * axis + 1]) {
                interior.end[axis] = std::max(interior.begin[axis], interior.end[axis] - 1);
            }
        }

        // Peel disjoint face regions off the owned block. Only these depend
        // on incoming halos. An axis one cell wide can have an empty interior.
        Box remaining = owned;
        for (int axis = 0; axis < 3; ++axis) {
            if (remaining.begin[axis] < interior.begin[axis]) {
                Box face = remaining;
                face.end[axis] = interior.begin[axis];
                boundary[boundaryCount++] = face;
                remaining.begin[axis] = interior.begin[axis];
            }
            if (remaining.end[axis] > interior.end[axis]) {
                Box face = remaining;
                face.begin[axis] = interior.end[axis];
                boundary[boundaryCount++] = face;
                remaining.end[axis] = interior.end[axis];
            }
        }
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    ~HaloExchange() {
        for (auto& field : requests) {
            for (int i = 0; i < 2 * count; ++i) MPI_Request_free(&field[i]);
        }
    }

    template <typename Compute>
    void apply(int field, double* data, Compute compute) {
        if (count != 0) MPI_Startall(2 * count, requests[field].data());
        clampFaces(data);
        compute(interior);
        if (count != 0) MPI_Waitall(2 * count, requests[field].data(), MPI_STATUSES_IGNORE);
        for (int i = 0; i < boundaryCount; ++i) compute(boundary[i]);
    }

private:
    void clampFaces(double* data) const {
        const auto [nx, ny, nz] = extent;
        const auto [row, plane] = strides;
        // Only face interiors are used by the seven-point stencil.
        if (!remote[0] || !remote[1]) {
            for (int z = 0; z < nz; ++z) {
                for (int y = 0; y < ny; ++y) {
                    double* line = data + z * plane + y * row;
                    if (!remote[0]) line[-1] = line[0];
                    if (!remote[1]) line[nx] = line[nx - 1];
                }
            }
        }
        if (!remote[2] || !remote[3]) {
            for (int z = 0; z < nz; ++z) {
                double* first = data + z * plane;
                if (!remote[2]) std::copy_n(first, nx, first - row);
                double* last = first + (ny - 1) * row;
                if (!remote[3]) std::copy_n(last, nx, last + row);
            }
        }
        if (!remote[4] || !remote[5]) {
            for (int y = 0; y < ny; ++y) {
                double* first = data + y * row;
                if (!remote[4]) std::copy_n(first, nx, first - plane);
                double* last = first + (nz - 1) * plane;
                if (!remote[5]) std::copy_n(last, nx, last + plane);
            }
        }
    }

    Index extent;
    Strides strides;
    Box interior;
    std::array<Box, 6> boundary;
    std::array<bool, 6> remote{};
    std::array<std::array<MPI_Request, 12>, 3> requests;
    int count = 0;
    int boundaryCount = 0;
};

// The benchmark uses unit spacing along all three axes.
inline double laplacian(const double* cell, Strides strides) {
    const double center = *cell;
    const double cxx = cell[1] + cell[-1] - 2.0 * center;
    const double cyy = *(cell + strides.row) + *(cell - strides.row) - 2.0 * center;
    const double czz = *(cell + strides.plane) + *(cell - strides.plane) - 2.0 * center;
    return cxx + cyy + czz;
}

template <bool ChemicalPotential>
void stencil(const Box& box, Strides strides, const double* input,
             const double* concentration, double* __restrict__ output) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double dt = 0.01;
    constexpr double D = 1.0;

    const int width = box.end[0] - box.begin[0];
    if (width == 0) return;
    for (int z = box.begin[2]; z < box.end[2]; ++z) {
        for (int y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t begin = strides.offset({box.begin[0], y, z});
            for (size_t i = begin; i < begin + width; ++i) {
                const double cv = concentration[i];
                const double lap = laplacian(input + i, strides);
                if constexpr (ChemicalPotential) {
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * lap;
                } else {
                    output[i] = cv + dt * D * lap;
                }
            }
        }
    }
}

void initializeConcentration(double* data, const Box& block, Strides local, Strides global,
                             size_t volume) {
    const auto [nx, ny, nz] = block.extent();
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const size_t start = global.offset({block.begin[0], block.begin[1] + y, block.begin[2] + z});
            double* row = data + local.offset({0, y, z});
            for (int x = 0; x < nx; ++x) {
                const double pseudo = ((start + x + 1) * 1299709) % volume / static_cast<double>(volume);
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void printDistributedResults(MPI_Comm comm, const double* data, const Box& block,
                             Strides local, const Index& grid, const Index& parts, Strides global) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto [nx, ny, nz] = block.extent();
    if (rank != 0) {
        MPI_Datatype type = boxType({nx, ny, nz}, local);
        MPI_Send(data, 1, type, 0, FinalBlock, comm);
        MPI_Type_free(&type);
        return;
    }

    // Receive directly into global row-major order for the original Kahan sum
    // and order-dependent FNV-1a hash. Only -r allocates the full grid on root.
    std::vector<double> result(global.plane * grid[2]);
    std::vector<MPI_Request> requests(size - 1);
    for (int source = 1; source < size; ++source) {
        Index coords;
        MPI_Cart_coords(comm, source, 3, coords.data());
        const Box remote = partition(grid, parts, coords);
        MPI_Datatype type = boxType(remote.extent(), global);
        MPI_Irecv(result.data() + global.offset(remote.begin), 1, type, source, FinalBlock, comm,
                  &requests[source - 1]);
        MPI_Type_free(&type);
    }
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const size_t offset = global.offset({block.begin[0], block.begin[1] + y, block.begin[2] + z});
            std::copy_n(data + z * local.plane + y * local.row, nx, result.data() + offset);
        }
    }
    MPI_Waitall(size - 1, requests.data(), MPI_STATUSES_IGNORE);
    print_results(result, "Concentration");
}

bool validateResult(MPI_Comm comm, int rank, const double* data, const Index& extent, Strides strides) {
    const auto [nx, ny, nz] = extent;
    const double inf = std::numeric_limits<double>::infinity();
    double bounds[3] = {-inf, -inf, 0.0};
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            const double* row = data + strides.offset({0, y, z});
            for (int x = 0; x < nx; ++x) {
                const double value = row[x];
                if (std::isfinite(value)) {
                    bounds[0] = std::max(bounds[0], -value);
                    bounds[1] = std::max(bounds[1], value);
                } else {
                    bounds[2] = 1.0;
                }
            }
        }
    }

    double global[3];
    MPI_Reduce(bounds, global, 3, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank != 0) return true;
    if (global[2] != 0.0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Concentration range: [%.6f, %.6f]\n", -global[0], global[1]);
    if (global[1] > 10.0 || global[0] > 10.0) {
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

int runBenchmark(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::array<int, 7> options{64, 0, 0, 20, 0, 0, -1};
    auto& [nx, ny, nz, iterations, validate, printResults, exitStatus] = options;
    if (rank == 0) {
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
                printUsage(argv[0]);
                exitStatus = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitStatus = 1;
                break;
            }
        }
        if (exitStatus == -1 && (nx < 0 || ny < 0 || nz < 0)) {
            printf("Grid dimensions must be nonnegative\n");
            exitStatus = 1;
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }
    MPI_Bcast(options.data(), options.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if (exitStatus != -1) return exitStatus;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Index grid{nx, ny, nz};
    const Strides global{static_cast<size_t>(nx), multiply(nx, ny)};
    const size_t volume = multiply(global.plane, nz);
    if (volume == 0) {
        if (rank != 0) return 0;
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
        printf("Computation time: 0 ms\n");
        printf("Performance: 0.000 MCellUpdates/s\n");
        if (printResults) print_results(std::vector<double>{}, "Concentration");
        if (validate) {
            printf("Validating result...\n");
            printf("Validation failed: empty concentration field\n");
            printf("Validation: FAILED\n");
        }
        return validate ? 1 : 0;
    }

    const Index parts = processGrid(grid, volume, ranks);
    const Index periods{};
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods.data(), false, &comm);
    if (comm == MPI_COMM_NULL) return 0;

    Index coords;
    MPI_Cart_coords(comm, rank, 3, coords.data());
    const Box block = partition(grid, parts, coords);
    const Index extent = block.extent();
    const Box owned{{0, 0, 0}, extent};
    const size_t paddedX = static_cast<size_t>(extent[0]) + 2;
    const Strides local{paddedX, multiply(paddedX, static_cast<size_t>(extent[1]) + 2)};
    const size_t elements = multiply(local.plane, static_cast<size_t>(extent[2]) + 2);
    std::array<std::unique_ptr<double[]>, 3> storage;
    for (auto& buffer : storage) buffer = std::make_unique_for_overwrite<double[]>(elements);
    // Leave one ghost layer around each field, with pointers to the first owned cell.
    const size_t first = local.offset({1, 1, 1});
    const std::array<double*, 3> fields{
        storage[0].get() + first, storage[1].get() + first, storage[2].get() + first
    };

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(fields[0], block, local, global, volume);
    int current = 0;
    double elapsed;
    {
        HaloExchange exchange(comm, owned, local, fields);
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        double* mu = fields[2];
        for (int step = 0; step < iterations; ++step) {
            double* cold = fields[current];
            double* cnew = fields[1 - current];
            exchange.apply(current, cold, [&](const Box& region) {
                stencil<true>(region, local, cold, cold, mu);
            });
            exchange.apply(2, mu, [&](const Box& region) {
                stencil<false>(region, local, mu, cold, cnew);
            });
            current = 1 - current;
        }
        elapsed = MPI_Wtime() - start;
    }

    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double cellUpdates = static_cast<double>(volume) * iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) printDistributedResults(comm, fields[current], block, local, grid, parts, global);

    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateResult(comm, rank, fields[current], extent, local);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, status);
    }
    MPI_Finalize();
    return status;
}
