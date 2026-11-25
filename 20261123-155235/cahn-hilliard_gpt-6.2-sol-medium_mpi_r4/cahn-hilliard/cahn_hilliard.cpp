#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

using Position = std::array<size_t, 3>;
using ProcessGrid = std::array<int, 3>;

size_t volume(const Position& extent) {
    return extent[0] * extent[1] * extent[2];
}

struct Box {
    Position begin;
    Position end;
};

struct Block {
    Position origin;
    Position size;
};

// Prefer using all available ranks, then minimize the largest block and the
// total area of internal faces. Ties favor splitting slower-varying axes.
ProcessGrid chooseProcessGrid(const Position& grid, int ranks) {
    ProcessGrid best{};
    int bestRanks = 0;
    size_t bestVolume = 0;
    long double bestArea = 0;
    for (int px = 1; px <= std::min(static_cast<int>(grid[0]), ranks); ++px) {
        for (int py = 1; py <= std::min(static_cast<int>(grid[1]), ranks / px); ++py) {
            const int pz = std::min(static_cast<int>(grid[2]), ranks / (px * py));
            const int usedRanks = px * py * pz;
            if (usedRanks < bestRanks) continue;

            const size_t largestVolume = ((grid[0] + px - 1) / px)
                                       * ((grid[1] + py - 1) / py)
                                       * ((grid[2] + pz - 1) / pz);
            const long double area = static_cast<long double>(px - 1) * grid[1] * grid[2]
                                   + static_cast<long double>(py - 1) * grid[0] * grid[2]
                                   + static_cast<long double>(pz - 1) * grid[0] * grid[1];
            if (usedRanks > bestRanks || largestVolume < bestVolume ||
                (largestVolume == bestVolume && area < bestArea)) {
                best = {px, py, pz};
                bestRanks = usedRanks;
                bestVolume = largestVolume;
                bestArea = area;
            }
        }
    }
    return best;
}

Block rankBlock(const Position& grid, const ProcessGrid& processes, MPI_Comm comm, int rank) {
    ProcessGrid coords;
    MPI_Cart_coords(comm, rank, 3, coords.data());
    Block block{};
    for (int axis = 0; axis < 3; ++axis) {
        const size_t base = grid[axis] / processes[axis];
        const size_t extra = grid[axis] % processes[axis];
        const size_t coord = coords[axis];
        block.origin[axis] = coord * base + std::min(coord, extra);
        block.size[axis] = base + (coord < extra);
    }
    return block;
}

size_t offset(const Position& pos, const Position& strides) {
    return pos[0] + pos[1] * strides[1] + pos[2] * strides[2];
}

// The caller passes a pointer to the first cell. Strides are in doubles.
MPI_Datatype blockType(const Position& size, const Position& strides) {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(size[0]), MPI_DOUBLE, &type);
    for (int axis = 1; axis < 3; ++axis) {
        if (size[axis] == 1) continue;
        MPI_Datatype next;
        MPI_Type_create_hvector(static_cast<int>(size[axis]), 1,
                                static_cast<MPI_Aint>(strides[axis] * sizeof(double)), type, &next);
        MPI_Type_free(&type);
        type = next;
    }
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    Block block;
    Position strides;
    size_t storageSize;
    size_t firstCell;
    Box interior;
    std::array<Box, 6> boundary;
    int boundaryCount = 0;
    std::array<std::array<int, 2>, 3> neighbors;

    Domain(const Position& grid, const ProcessGrid& processes, MPI_Comm comm, int rank)
        : block(rankBlock(grid, processes, comm, rank)), interior{{}, block.size} {
        Position halo{};
        Position padded = block.size;
        for (int axis = 0; axis < 3; ++axis) {
            MPI_Cart_shift(comm, axis, 1, &neighbors[axis][0], &neighbors[axis][1]);
            halo[axis] = processes[axis] > 1;
            padded[axis] += 2 * halo[axis];
        }
        strides = {1, padded[0], padded[0] * padded[1]};
        storageSize = volume(padded);
        firstCell = offset(halo, strides);

        // Peel off disjoint faces that depend on a receive. A one-cell block
        // can touch both neighbors on an axis, but its cells are only updated once.
        for (int axis = 2; axis >= 0; --axis) {
            if (neighbors[axis][0] != MPI_PROC_NULL) {
                boundary[boundaryCount] = interior;
                boundary[boundaryCount++].end[axis] = ++interior.begin[axis];
            }
            if (interior.begin[axis] == interior.end[axis]) break;
            if (neighbors[axis][1] != MPI_PROC_NULL) {
                boundary[boundaryCount] = interior;
                boundary[boundaryCount++].begin[axis] = --interior.end[axis];
            }
            if (interior.begin[axis] == interior.end[axis]) break;
        }
    }
};

class HaloExchange {
public:
    HaloExchange(const Domain& domain, MPI_Comm comm) : comm(comm) {
        for (int axis = 0; axis < 3; ++axis) {
            if (domain.neighbors[axis][0] == MPI_PROC_NULL &&
                domain.neighbors[axis][1] == MPI_PROC_NULL) continue;

            Position faceSize = domain.block.size;
            faceSize[axis] = 1;
            faceTypes[axis] = blockType(faceSize, domain.strides);
            const size_t stride = domain.strides[axis];
            if (domain.neighbors[axis][0] != MPI_PROC_NULL) {
                faces[count++] = {domain.neighbors[axis][0], faceTypes[axis],
                                  domain.firstCell, domain.firstCell - stride};
            }
            if (domain.neighbors[axis][1] != MPI_PROC_NULL) {
                const size_t lastCell = domain.firstCell + (domain.block.size[axis] - 1) * stride;
                faces[count++] = {domain.neighbors[axis][1], faceTypes[axis], lastCell, lastCell + stride};
            }
        }
    }

    ~HaloExchange() {
        for (auto& type : faceTypes) {
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
        }
    }

    void start(double* storage, MPI_Request* requests) const {
        for (int i = 0; i < count; ++i) {
            const auto& face = faces[i];
            MPI_Irecv(storage + face.receive, 1, face.type, face.rank, 0, comm, &requests[i]);
        }
        for (int i = 0; i < count; ++i) {
            const auto& face = faces[i];
            MPI_Isend(storage + face.send, 1, face.type, face.rank, 0, comm, &requests[count + i]);
        }
    }

    void finish(MPI_Request* requests) const {
        if (count != 0) MPI_Waitall(2 * count, requests, MPI_STATUSES_IGNORE);
    }

private:
    struct Face {
        int rank;
        MPI_Datatype type;
        size_t send;
        size_t receive;
    };

    MPI_Comm comm;
    std::array<Face, 6> faces;
    std::array<MPI_Datatype, 3> faceTypes{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    int count = 0;
};

// Both passes use the same seven-point Laplacian with unit grid spacing.
// Clamp physical boundaries and use the exchanged halo at rank boundaries.
template<bool ChemicalPotential>
void applyStencil(const Domain& domain, const Box& box, const double* __restrict__ field,
                  const double* __restrict__ cold, double* __restrict__ output) {
    if (box.begin[0] == box.end[0] || box.begin[1] == box.end[1] || box.begin[2] == box.end[2]) return;

    const auto& size = domain.block.size;
    const auto& neighbors = domain.neighbors;
    const size_t rowStride = domain.strides[1];
    const size_t planeStride = domain.strides[2];
    const bool clampLeft = box.begin[0] == 0 && neighbors[0][0] == MPI_PROC_NULL;
    const bool clampRight = box.end[0] == size[0] && neighbors[0][1] == MPI_PROC_NULL;
    const size_t first = box.begin[0] + clampLeft;
    const size_t last = box.end[0] - clampRight;

    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        const size_t zm = z == 0 && neighbors[2][0] == MPI_PROC_NULL ? 0 : planeStride;
        const size_t zp = z + 1 == size[2] && neighbors[2][1] == MPI_PROC_NULL ? 0 : planeStride;
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t ym = y == 0 && neighbors[1][0] == MPI_PROC_NULL ? 0 : rowStride;
            const size_t yp = y + 1 == size[1] && neighbors[1][1] == MPI_PROC_NULL ? 0 : rowStride;
            const size_t rowOffset = z * planeStride + y * rowStride;
            const double* row = field + rowOffset;
            const double* yPrev = row - ym;
            const double* yNext = row + yp;
            const double* zPrev = row - zm;
            const double* zNext = row + zp;
            double* out = output + rowOffset;
            const auto update = [&](size_t x, int xm, int xp) {
                const double* cell = row + x;
                const double center = *cell;
                const double cxx = cell[xp] + cell[xm] - 2.0 * center;
                const double cyy = yNext[x] + yPrev[x] - 2.0 * center;
                const double czz = zNext[x] + zPrev[x] - 2.0 * center;
                const double laplacian = cxx + cyy + czz;
                if constexpr (ChemicalPotential) {
                    constexpr double eAA = -(2.0 / 9.0);
                    constexpr double eBB = -(2.0 / 9.0);
                    constexpr double eAB = 2.0 / 9.0;
                    out[x] = 4.5 * ((center + 1.0) * eAA + (center - 1.0) * eBB - 2.0 * center * eAB)
                           + 3.0 * center + center * center * center - 0.5 * laplacian;
                } else {
                    out[x] = cold[rowOffset + x] + 0.01 * laplacian;
                }
            };

            if (clampLeft) update(0, 0, size[0] == 1 && clampRight ? 0 : 1);
            for (size_t x = first; x < last; ++x) update(x, -1, 1);
            if (clampRight && first <= last) update(last, -1, 0);
        }
    }
}

template<bool ChemicalPotential>
void stencilPass(const Domain& domain, const HaloExchange& halos, double* field,
                 const double* cold, double* output) {
    MPI_Request requests[12];
    halos.start(field, requests);
    applyStencil<ChemicalPotential>(domain, domain.interior, field + domain.firstCell, cold,
                                    output + domain.firstCell);
    halos.finish(requests);
    for (int i = 0; i < domain.boundaryCount; ++i) {
        applyStencil<ChemicalPotential>(domain, domain.boundary[i], field + domain.firstCell, cold,
                                        output + domain.firstCell);
    }
}

void initializeConcentration(double* c, const Position& grid, const Domain& domain) {
    const auto& [nx, ny, nz] = domain.block.size;
    const auto& origin = domain.block.origin;
    const size_t gridSize = volume(grid);
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t globalRow = ((origin[2] + z) * grid[1] + origin[1] + y) * grid[0] + origin[0];
            double* row = c + z * domain.strides[2] + y * domain.strides[1];
            for (size_t x = 0; x < nx; ++x) {
                const double pseudo = ((globalRow + x + 1) * 1299709 % gridSize) / static_cast<double>(gridSize);
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void reportResults(const double* c, const Position& grid, const Domain& domain,
                   const ProcessGrid& processes, MPI_Comm comm, int rank) {
    if (rank != 0) {
        MPI_Datatype type = blockType(domain.block.size, domain.strides);
        MPI_Send(c, 1, type, 0, 0, comm);
        MPI_Type_free(&type);
        return;
    }

    int ranks;
    MPI_Comm_size(comm, &ranks);
    const Position strides{1, grid[0], grid[0] * grid[1]};
    std::vector<double> global(volume(grid));
    std::vector<MPI_Request> requests(ranks - 1);
    for (int peer = 1; peer < ranks; ++peer) {
        const auto block = rankBlock(grid, processes, comm, peer);
        MPI_Datatype type = blockType(block.size, strides);
        MPI_Irecv(global.data() + offset(block.origin, strides), 1, type, peer, 0, comm, &requests[peer - 1]);
        MPI_Type_free(&type);
    }
    const auto& [nx, ny, nz] = domain.block.size;
    double* local = global.data() + offset(domain.block.origin, strides);
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            std::copy_n(c + z * domain.strides[2] + y * domain.strides[1], nx,
                        local + z * strides[2] + y * strides[1]);
        }
    }
    if (ranks > 1) MPI_Waitall(ranks - 1, requests.data(), MPI_STATUSES_IGNORE);
    print_results(global, "Concentration");
}

bool validateResult(const double* c, const Domain& domain, MPI_Comm comm, int rank) {
    // Validate owned cells only. Reducing the negated minimum and the maximum
    // together gives every rank the same status in a single collective.
    const auto& [nx, ny, nz] = domain.block.size;
    double local[3]{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), 0.0};
    for (size_t z = 0; z < nz && local[2] == 0.0; ++z) {
        for (size_t y = 0; y < ny && local[2] == 0.0; ++y) {
            const double* row = c + z * domain.strides[2] + y * domain.strides[1];
            for (size_t x = 0; x < nx; ++x) {
                const double value = row[x];
                if (!std::isfinite(value)) {
                    local[2] = 1.0;
                    break;
                }
                local[0] = std::max(local[0], -value);
                local[1] = std::max(local[1], value);
            }
        }
    }
    double global[3];
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX, comm);
    const bool valid = global[2] == 0.0 && global[0] <= 10.0 && global[1] <= 10.0;
    if (rank == 0) {
        if (global[2] != 0.0) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", -global[0], global[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

int runSimulation(const Position& grid, const ProcessGrid& processes, int iterations,
                  bool validate, bool printResults, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", grid[0], grid[1], grid[2]);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    const Domain domain(grid, processes, comm, rank);
    const HaloExchange halos(domain, comm);
    auto cold = std::make_unique_for_overwrite<double[]>(domain.storageSize);
    auto cnew = std::make_unique<double[]>(domain.storageSize);
    auto mu = std::make_unique<double[]>(domain.storageSize);
    initializeConcentration(cold.get() + domain.firstCell, grid, domain);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        stencilPass<true>(domain, halos, cold.get(), nullptr, mu.get());
        stencilPass<false>(domain, halos, mu.get(), cold.get() + domain.firstCell, cnew.get());
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", std::floor(duration * 1000.0));
        const double cellUpdates = static_cast<double>(volume(grid)) * iterations;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const double* result = cold.get() + domain.firstCell;
    if (printResults) reportResults(result, grid, domain, processes, comm, rank);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        return validateResult(result, domain, comm, rank) ? 0 : 1;
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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int nx = 64;
    int ny = 0;
    int nz = 0;
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
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
    const Position grid{static_cast<size_t>(nx), static_cast<size_t>(ny), static_cast<size_t>(nz)};
    const size_t maxElements = static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(double);
    if (maxElements / grid[0] / grid[1] < grid[2]) {
        if (rank == 0) fprintf(stderr, "Grid size exceeds the addressable range\n");
        MPI_Finalize();
        return 1;
    }

    ProcessGrid processes;
    if (rank == 0) processes = chooseProcessGrid(grid, ranks);
    MPI_Bcast(processes.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    int periods[3]{};
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, processes.data(), periods, 0, &comm);
    int status = 0;
    if (comm != MPI_COMM_NULL) {
        status = runSimulation(grid, processes, iterations, validate, printResults, comm);
        MPI_Comm_free(&comm);
    }
    MPI_Finalize();
    return status;
}
