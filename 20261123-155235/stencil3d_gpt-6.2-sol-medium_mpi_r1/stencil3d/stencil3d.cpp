#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

using Real = double;
using Index = std::array<size_t, 3>;
using Coordinates = std::array<int, 3>;

struct Box {
    Index begin;
    Index end;

    bool empty() const {
        return begin[0] == end[0] || begin[1] == end[1] || begin[2] == end[2];
    }

    Index shape() const {
        return {end[0] - begin[0], end[1] - begin[1], end[2] - begin[2]};
    }
};

size_t volume(const Index& shape) {
    return shape[0] * shape[1] * shape[2];
}

size_t offset(const Index& point, const Index& shape) {
    return (point[2] * shape[1] + point[1]) * shape[0] + point[0];
}

size_t interiorExtent(size_t n) {
    return n > 2 ? n - 2 : 0;
}

// Balance the largest block and minimize its face traffic among feasible grids.
Coordinates processGrid(const Index& global, int ranks) {
    Coordinates limit;
    for (int axis = 0; axis < 3; ++axis) {
        const size_t extent = interiorExtent(global[axis]);
        limit[axis] = static_cast<int>(std::min<size_t>(ranks, extent ? extent : global[axis]));
    }

    Coordinates best{1, 1, 1};
    auto bestScore = std::make_tuple(0, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::infinity());
    for (int x = 1; x <= limit[0]; ++x) {
        const int maxY = std::min(limit[1], ranks / x);
        for (int y = 1; y <= maxY; ++y) {
            const int z = std::min(limit[2], ranks / (x * y));
            const Coordinates dims{x, y, z};
            std::array<double, 3> longest;
            for (int axis = 0; axis < 3; ++axis) {
                const size_t interior = interiorExtent(global[axis]);
                const size_t n = interior ? interior : global[axis];
                longest[axis] = (n + dims[axis] - 1) / dims[axis];
            }
            const double cells = longest[0] * longest[1] * longest[2];
            double faces = 0;
            for (int axis = 0; axis < 3; ++axis) {
                faces += std::min(2, dims[axis] - 1) *
                         longest[(axis + 1) % 3] * longest[(axis + 2) % 3];
            }
            const auto score = std::make_tuple(-(x * y * z), cells, faces);
            if (score < bestScore) {
                bestScore = score;
                best = dims;
            }
        }
    }
    return best;
}

struct Block {
    Index globalBegin;
    Index owned;
    Index padded;
    Box work;
};

Block localBlock(const Index& global, const Coordinates& dims, const Coordinates& coords) {
    Block block{};
    for (int axis = 0; axis < 3; ++axis) {
        const size_t interior = interiorExtent(global[axis]);
        const size_t n = interior ? interior : global[axis];
        const size_t coord = coords[axis];
        const size_t base = n / dims[axis];
        const size_t remainder = n % dims[axis];
        const size_t count = base + (coord < remainder);
        const size_t begin = coord * base + std::min(coord, remainder);
        const size_t lowBoundary = interior && coords[axis] == 0;
        const size_t highBoundary = interior && coords[axis] == dims[axis] - 1;

        block.globalBegin[axis] = begin + (interior != 0) - lowBoundary;
        block.owned[axis] = count + lowBoundary + highBoundary;
        block.padded[axis] = block.owned[axis] + 2;
        block.work.begin[axis] = 1 + lowBoundary;
        block.work.end[axis] = block.work.begin[axis] + (interior ? count : 0);
    }
    return block;
}

void initializeGrid(Real* first, Real* second, const Block& block, const Index& global) {
    // Initialize both time buffers so physical boundaries need no per-step copying.
    for (size_t z = 1; z <= block.owned[2]; ++z) {
        for (size_t y = 1; y <= block.owned[1]; ++y) {
            const size_t local = offset({1, y, z}, block.padded);
            const size_t start = offset({block.globalBegin[0], block.globalBegin[1] + y - 1,
                                         block.globalBegin[2] + z - 1}, global);
            for (size_t x = 0; x < block.owned[0]; ++x) {
                const Real value = static_cast<Real>((start + x) % 19);
                first[local + x] = value;
                second[local + x] = value;
            }
        }
    }
}

void stencil(const Real* input, Real* __restrict output, const Index& padded, const Box& box) {
    const size_t pitch = padded[0];
    const size_t plane = pitch * padded[1];
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t row = z * plane + y * pitch;
            const Real* center = input + row;
            const Real* front = input + (row - pitch);
            const Real* back = input + (row + pitch);
            const Real* bottom = input + (row - plane);
            const Real* top = input + (row + plane);
            Real* dest = output + row;
            for (size_t x = box.begin[0]; x < box.end[0]; ++x) {
                dest[x] = (center[x] + center[x - 1] + center[x + 1] +
                           front[x] + back[x] + bottom[x] + top[x]) / 7.0;
            }
        }
    }
}

// One MPI element describes a brick in a potentially padded array.
MPI_Datatype brickType(const Index& layout, const Index& shape) {
    MPI_Datatype rows, brick;
    MPI_Type_create_hvector(static_cast<int>(shape[1]), static_cast<int>(shape[0]),
                            layout[0] * sizeof(Real), MPI_DOUBLE, &rows);
    if (shape[2] == 1) {
        brick = rows;
    } else {
        MPI_Type_create_hvector(static_cast<int>(shape[2]), 1,
                                layout[0] * layout[1] * sizeof(Real), rows, &brick);
        MPI_Type_free(&rows);
    }
    MPI_Type_commit(&brick);
    return brick;
}

struct Halos {
    Box core;
    std::array<Box, 6> shell;
    int shellCount = 0;
    std::array<std::array<MPI_Request, 12>, 2> requests;
    int requestCount = 0;
    std::array<MPI_Datatype, 3> types{MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};

    Halos(const Block& block, Real* first, Real* second, MPI_Comm comm) : core(block.work) {
        if (core.empty()) return;

        // Peel disjoint face regions from the core, keeping x contiguous wherever possible.
        for (int axis = 2; axis >= 0; --axis) {
            int neighbors[2];
            MPI_Cart_shift(comm, axis, 1, &neighbors[0], &neighbors[1]);
            if (neighbors[0] == MPI_PROC_NULL && neighbors[1] == MPI_PROC_NULL) continue;

            Index shape = block.work.shape();
            shape[axis] = 1;
            MPI_Datatype& type = types[axis];
            type = brickType(block.padded, shape);

            for (int side = 0; side < 2; ++side) {
                const int peer = neighbors[side];
                if (peer == MPI_PROC_NULL) continue;

                Index send = block.work.begin;
                send[axis] = side == 0 ? block.work.begin[axis] : block.work.end[axis] - 1;
                Index recv = send;
                recv[axis] = side == 0 ? send[axis] - 1 : send[axis] + 1;
                const size_t sendOffset = offset(send, block.padded);
                const size_t recvOffset = offset(recv, block.padded);
                for (int buffer = 0; buffer < 2; ++buffer) {
                    Real* grid = buffer == 0 ? first : second;
                    MPI_Recv_init(grid + recvOffset, 1, type, peer, axis, comm,
                                  &requests[buffer][requestCount]);
                    MPI_Send_init(grid + sendOffset, 1, type, peer, axis, comm,
                                  &requests[buffer][requestCount + 1]);
                }
                requestCount += 2;

                if (!core.empty()) {
                    Box face = core;
                    if (side == 0) {
                        face.end[axis] = ++core.begin[axis];
                    } else {
                        face.begin[axis] = --core.end[axis];
                    }
                    shell[shellCount++] = face;
                }
            }
        }
    }

    ~Halos() {
        for (auto& buffer : requests) {
            for (int i = 0; i < requestCount; ++i) MPI_Request_free(&buffer[i]);
        }
        for (auto& type : types) {
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
        }
    }
};

void reportGrid(const Real* grid, const Block& block, const Index& global,
                const Coordinates& dims, int rank, MPI_Comm comm) {
    const size_t first = offset({1, 1, 1}, block.padded);
    constexpr int tag = 3;
    if (rank != 0) {
        MPI_Datatype type = brickType(block.padded, block.owned);
        MPI_Send(grid + first, 1, type, 0, tag, comm);
        MPI_Type_free(&type);
        return;
    }

    int ranks;
    MPI_Comm_size(comm, &ranks);
    // The serial sum and hash both depend on the global x-fastest ordering.
    std::vector<Real> result(volume(global));
    std::vector<MPI_Request> receives(ranks - 1);
    for (int peer = 1; peer < ranks; ++peer) {
        Coordinates coords;
        MPI_Cart_coords(comm, peer, 3, coords.data());
        const Block remote = localBlock(global, dims, coords);
        MPI_Datatype type = brickType(global, remote.owned);
        MPI_Irecv(result.data() + offset(remote.globalBegin, global), 1, type, peer, tag,
                  comm, &receives[peer - 1]);
        MPI_Type_free(&type);
    }

    for (size_t z = 0; z < block.owned[2]; ++z) {
        for (size_t y = 0; y < block.owned[1]; ++y) {
            const Index start{block.globalBegin[0], block.globalBegin[1] + y,
                              block.globalBegin[2] + z};
            std::copy_n(grid + offset({1, y + 1, z + 1}, block.padded), block.owned[0],
                        result.data() + offset(start, global));
        }
    }
    MPI_Waitall(static_cast<int>(receives.size()), receives.data(), MPI_STATUSES_IGNORE);
    print_results(result, "Grid");
}

bool validateGrid(const Real* grid, const Block& block, int rank, MPI_Comm comm) {
    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -minVal;
    Real nonfinite = 0.0;
    for (size_t z = 1; z <= block.owned[2]; ++z) {
        for (size_t y = 1; y <= block.owned[1]; ++y) {
            const Real* row = grid + offset({1, y, z}, block.padded);
            for (size_t x = 0; x < block.owned[0]; ++x) {
                const Real value = row[x];
                nonfinite += !std::isfinite(value);
                minVal = std::min(minVal, value);
                maxVal = std::max(maxVal, value);
            }
        }
    }

    const Real local[3] = {-minVal, maxVal, nonfinite};
    Real global[3];
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX, comm);
    const bool finite = global[2] == 0.0;
    const bool bounded = global[0] <= 1e6 && global[1] <= 1e6;
    if (rank == 0) {
        if (!finite) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Value range: [%.6f, %.6f]\n", -global[0], global[1]);
            if (!bounded) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", finite && bounded ? "PASSED" : "FAILED");
    }
    return finite && bounded;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseNumber(const char* text, int& value) {
    char* end;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool fitsGrid(const Index& shape) {
    const size_t maxElements = std::vector<Real>().max_size();
    return shape[0] <= maxElements / shape[1] / shape[2];
}

int runBenchmark(const Index& global, int iterations, bool validate, bool printResults,
                 const Coordinates& dims, MPI_Comm comm) {
    int rank;
    Coordinates coords;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords.data());
    const Block block = localBlock(global, dims, coords);
    const int localFits = fitsGrid(block.padded);
    int allFit;
    MPI_Allreduce(&localFits, &allFit, 1, MPI_INT, MPI_MIN, comm);
    if (!allFit) {
        if (rank == 0) std::fprintf(stderr, "Local grid dimensions are too large\n");
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    auto first = std::make_unique_for_overwrite<Real[]>(volume(block.padded));
    auto second = std::make_unique_for_overwrite<Real[]>(volume(block.padded));
    initializeGrid(first.get(), second.get(), block, global);
    Halos halos(block, first.get(), second.get(), comm);
    Real* grids[2] = {first.get(), second.get()};
    int current = 0;

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    if (!block.work.empty()) {
        for (int iter = 0; iter < iterations; ++iter) {
            Real* input = grids[current];
            Real* output = grids[1 - current];
            auto& requests = halos.requests[current];
            if (halos.requestCount) MPI_Startall(halos.requestCount, requests.data());
            if (!halos.core.empty()) stencil(input, output, block.padded, halos.core);
            if (halos.requestCount) {
                MPI_Waitall(halos.requestCount, requests.data(), MPI_STATUSES_IGNORE);
                for (int i = 0; i < halos.shellCount; ++i) {
                    stencil(input, output, block.padded, halos.shell[i]);
                }
            }
            current = 1 - current;
        }
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = static_cast<double>(interiorExtent(global[0])) *
                               interiorExtent(global[1]) * interiorExtent(global[2]) * iterations;
        const double mcups = elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0;
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalGrid = grids[current];
    if (printResults) reportGrid(finalGrid, block, global, dims, rank, comm);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        return validateGrid(finalGrid, block, rank, comm) ? 0 : 1;
    }
    return 0;
}

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        int* number = nullptr;
        if (strcmp(arg, "-x") == 0 && i + 1 < argc) {
            number = &nx;
        } else if (strcmp(arg, "-y") == 0 && i + 1 < argc) {
            number = &ny;
        } else if (strcmp(arg, "-z") == 0 && i + 1 < argc) {
            number = &nz;
        } else if (strcmp(arg, "-i") == 0 && i + 1 < argc) {
            number = &iterations;
        } else if (strcmp(arg, "-v") == 0) {
            validate = true;
        } else if (strcmp(arg, "-r") == 0) {
            printResults = true;
        } else if (strcmp(arg, "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", arg);
                printUsage(argv[0]);
            }
            return 1;
        }

        if (number && !parseNumber(argv[++i], *number)) {
            if (rank == 0) std::fprintf(stderr, "Invalid value for %s: %s\n", arg, argv[i]);
            return 1;
        }
    }

    if (nx == 0) {
        if (rank == 0) std::fprintf(stderr, "Grid size in X must be positive\n");
        return 1;
    }
    const Index global{static_cast<size_t>(nx), static_cast<size_t>(ny ? ny : nx),
                       static_cast<size_t>(nz ? nz : nx)};
    if (!fitsGrid(global)) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions are too large\n");
        return 1;
    }

    Coordinates dims;
    if (rank == 0) dims = processGrid(global, ranks);
    MPI_Bcast(dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    const Coordinates periods{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), 1, &cart);
    if (cart == MPI_COMM_NULL) return 0;
    const int result = runBenchmark(global, iterations, validate, printResults, dims, cart);
    MPI_Comm_free(&cart);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 0;
    try {
        result = run(argc, argv);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Unable to allocate the stencil grid\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        result = 1;
    }
    MPI_Finalize();
    return result;
}
