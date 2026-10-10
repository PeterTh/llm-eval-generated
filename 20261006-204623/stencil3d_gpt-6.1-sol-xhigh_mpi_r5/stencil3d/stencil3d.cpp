#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Extent = std::array<size_t, 3>;

struct Options {
    Extent size{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

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

// Select the largest usable process grid and minimize its total internal face
// area. Ties favor an unsplit X dimension, keeping the contiguous loops long.
std::array<int, 3> processGrid(const Extent& size, int processes) {
    std::array<int, 3> best{1, 1, 1};
    int bestCount = 0;
    long double bestArea = std::numeric_limits<long double>::infinity();
    for (int x = 1; x <= processes && static_cast<size_t>(x) <= size[0]; ++x) {
        for (int y = 1; y <= processes / x && static_cast<size_t>(y) <= size[1]; ++y) {
            const int z = static_cast<int>(std::min(size[2], static_cast<size_t>(processes / x / y)));
            const int count = x * y * z;
            const long double area = (x - 1) * static_cast<long double>(size[1]) * size[2]
                                   + (y - 1) * static_cast<long double>(size[0]) * size[2]
                                   + (z - 1) * static_cast<long double>(size[0]) * size[1];
            if (count > bestCount || (count == bestCount && area < bestArea)) {
                best = {x, y, z};
                bestCount = count;
                bestArea = area;
            }
        }
    }
    return best;
}

void partition(const Extent& global, const std::array<int, 3>& dims,
               const std::array<int, 3>& coords, Extent& size, Extent& begin) {
    for (int a = 0; a < 3; ++a) {
        const size_t quotient = global[a] / dims[a];
        const size_t remainder = global[a] % dims[a];
        const size_t coordinate = static_cast<size_t>(coords[a]);
        size[a] = quotient + (coordinate < remainder);
        begin[a] = coordinate * quotient + std::min(coordinate, remainder);
    }
}

struct Box {
    Extent begin;
    Extent end; // Exclusive.

    bool empty() const {
        return begin[0] >= end[0] || begin[1] >= end[1] || begin[2] >= end[2];
    }
};

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int processes = 0;
    Extent global, size{}, begin{};
    std::array<int, 3> dims;
    std::array<int, 6> neighbors{};
    size_t row = 0, plane = 0, cells = 0;
    Box updates{}, core{};
    std::array<Box, 6> edges{};

    Domain(const Extent& globalSize, const std::array<int, 3>& dimensions, MPI_Comm active)
        : global(globalSize), dims(dimensions) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims.data(), periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &processes);
        std::array<int, 3> coords{};
        MPI_Cart_coords(comm, rank, 3, coords.data());
        partition(global, dims, coords, size, begin);
        row = size[0] + 2;
        if (row > std::numeric_limits<size_t>::max() / (size[1] + 2))
            throw std::runtime_error("Local grid size overflows size_t");
        plane = row * (size[1] + 2);
        if (plane > static_cast<size_t>(std::numeric_limits<MPI_Aint>::max())
                        / sizeof(Real) / (size[2] + 2))
            throw std::runtime_error("Local grid is too large");
        cells = plane * (size[2] + 2);
        for (int a = 0; a < 3; ++a) {
            MPI_Cart_shift(comm, a, 1, &neighbors[2 * a], &neighbors[2 * a + 1]);
            // Global boundary cells stay fixed in both buffers.
            updates.begin[a] = 1 + (begin[a] == 0);
            updates.end[a] = size[a] + 1 - (begin[a] + size[a] == global[a]);
            core.begin[a] = std::max(updates.begin[a],
                neighbors[2 * a] == MPI_PROC_NULL ? size_t{1} : size_t{2});
            core.end[a] = std::min(updates.end[a],
                size[a] + (neighbors[2 * a + 1] == MPI_PROC_NULL));
        }
        // Six disjoint boxes cover the cells outside the communication-free
        // core. An empty core is handled by computing the entire update box.
        Box remaining = updates;
        for (int a = 2; a >= 0; --a) {
            edges[2 * a] = remaining;
            edges[2 * a].end[a] = core.begin[a];
            edges[2 * a + 1] = remaining;
            edges[2 * a + 1].begin[a] = core.end[a];
            remaining.begin[a] = core.begin[a];
            remaining.end[a] = core.end[a];
        }
    }

    ~Domain() { MPI_Comm_free(&comm); }

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }
};

void initializeGrid(std::array<std::vector<Real>, 2>& grids, const Domain& domain) {
    for (size_t z = 1; z <= domain.size[2]; ++z) {
        for (size_t y = 1; y <= domain.size[1]; ++y) {
            size_t globalIndex = ((domain.begin[2] + z - 1) * domain.global[1]
                               + domain.begin[1] + y - 1) * domain.global[0] + domain.begin[0];
            size_t localIndex = domain.index(1, y, z);
            for (size_t x = 0; x < domain.size[0]; ++x, ++globalIndex, ++localIndex) {
                const Real value = static_cast<Real>(globalIndex % 19);
                grids[0][localIndex] = value;
                grids[1][localIndex] = value;
            }
        }
    }
}

// Persistent requests amortize setup across iterations. Y/Z faces use derived
// datatypes directly in the grids. Pack the strided X faces explicitly to avoid
// MPI traversing a datatype with one segment for every individual double.
struct HaloExchange {
    const Domain& domain;
    std::array<MPI_Datatype, 3> types{};
    std::array<std::array<MPI_Request, 12>, 2> requests{};
    std::array<std::vector<Real>, 2> sendX, receiveX;
    int count = 0;

    HaloExchange(const Domain& d, std::array<std::vector<Real>, 2>& grids) : domain(d) {
        MPI_Datatype rowType;
        const int nx = static_cast<int>(d.size[0]);
        const int ny = static_cast<int>(d.size[1]);
        const int nz = static_cast<int>(d.size[2]);
        MPI_Type_contiguous(ny, MPI_DOUBLE, &rowType);
        MPI_Type_contiguous(nz, rowType, &types[0]);
        MPI_Type_free(&rowType);
        MPI_Type_contiguous(nx, MPI_DOUBLE, &rowType);
        MPI_Type_create_hvector(nz, 1, static_cast<MPI_Aint>(d.plane * sizeof(Real)), rowType, &types[1]);
        MPI_Type_free(&rowType);
        MPI_Type_vector(ny, nx, static_cast<int>(d.row), MPI_DOUBLE, &types[2]);
        for (auto& type : types) MPI_Type_commit(&type);
        for (int side = 0; side < 2; ++side) {
            if (d.neighbors[side] != MPI_PROC_NULL) {
                sendX[side].resize(d.size[1] * d.size[2]);
                receiveX[side].resize(d.size[1] * d.size[2]);
            }
        }
        for (int buffer = 0; buffer < 2; ++buffer) {
            int n = 0;
            // Post receives before sends; each face has a distinct tag.
            for (int sending = 0; sending < 2; ++sending) {
                for (int face = 0; face < 6; ++face) {
                    const int peer = d.neighbors[face];
                    if (peer == MPI_PROC_NULL) continue;
                    const int axis = face / 2, side = face % 2;
                    Real* address;
                    if (axis == 0) {
                        address = sending ? sendX[side].data() : receiveX[side].data();
                    } else {
                        Extent position{1, 1, 1};
                        position[axis] = sending ? (side ? d.size[axis] : 1)
                                                 : (side ? d.size[axis] + 1 : 0);
                        address = grids[buffer].data() + d.index(position[0], position[1], position[2]);
                    }
                    if (sending)
                        MPI_Send_init(address, 1, types[axis], peer, face, d.comm, &requests[buffer][n++]);
                    else
                        MPI_Recv_init(address, 1, types[axis], peer, face ^ 1, d.comm, &requests[buffer][n++]);
                }
            }
            count = n;
        }
    }

    ~HaloExchange() {
        for (auto& buffer : requests)
            for (int i = 0; i < count; ++i) MPI_Request_free(&buffer[i]);
        for (auto& type : types) MPI_Type_free(&type);
    }

    void start(int buffer, const Real* grid) {
        for (int side = 0; side < 2; ++side) {
            if (domain.neighbors[side] == MPI_PROC_NULL) continue;
            const size_t x = side ? domain.size[0] : 1;
            size_t i = 0;
            for (size_t z = 1; z <= domain.size[2]; ++z)
                for (size_t y = 1; y <= domain.size[1]; ++y)
                    sendX[side][i++] = grid[domain.index(x, y, z)];
        }
        if (count) MPI_Startall(count, requests[buffer].data());
    }

    void finish(int buffer, Real* grid) {
        if (count) MPI_Waitall(count, requests[buffer].data(), MPI_STATUSES_IGNORE);
        for (int side = 0; side < 2; ++side) {
            if (domain.neighbors[side] == MPI_PROC_NULL) continue;
            const size_t x = side ? domain.size[0] + 1 : 0;
            size_t i = 0;
            for (size_t z = 1; z <= domain.size[2]; ++z)
                for (size_t y = 1; y <= domain.size[1]; ++y)
                    grid[domain.index(x, y, z)] = receiveX[side][i++];
        }
    }
};

void stencilBox(const Real* __restrict__ input, Real* __restrict__ output,
                const Domain& domain, const Box& box,
                MPI_Request* requests = nullptr, int requestCount = 0) {
    if (box.empty()) return;
    const size_t row = domain.row, plane = domain.plane;
    bool pending = requestCount != 0;
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            size_t i = z * plane + y * row + box.begin[0];
            for (size_t x = box.begin[0]; x < box.end[0]; ++x, ++i) {
                // Keep the original arithmetic order and division, including
                // its floating-point rounding; do not enable fast-math.
                output[i] = (input[i] + input[i - 1] + input[i + 1]
                           + input[i - row] + input[i + row]
                           + input[i - plane] + input[i + plane]) / 7.0;
            }
        }
        // MPI implementations without asynchronous progress still get an
        // opportunity to advance transfers during long interior computations.
        if (pending && ((z - box.begin[2]) % 8 == 7)) {
            int complete = 0;
            MPI_Testall(requestCount, requests, &complete, MPI_STATUSES_IGNORE);
            pending = !complete;
        }
    }
}

void printDistributedResults(const std::vector<Real>& grid, const Domain& domain) {
    if (domain.rank != 0) {
        const int sizes[3] = {static_cast<int>(domain.size[2] + 2),
                             static_cast<int>(domain.size[1] + 2), static_cast<int>(domain.size[0] + 2)};
        const int subsizes[3] = {static_cast<int>(domain.size[2]),
                                static_cast<int>(domain.size[1]), static_cast<int>(domain.size[0])};
        const int starts[3] = {1, 1, 1};
        MPI_Datatype block;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(grid.data(), 1, block, 0, 6, domain.comm);
        MPI_Type_free(&block);
        return;
    }

    // Only -r allocates a global grid. Receiving directly into subarrays retains
    // the serial order used by the Kahan sum, samples, and bytewise result hash.
    std::vector<Real> globalGrid(domain.global[0] * domain.global[1] * domain.global[2]);
    for (size_t z = 1; z <= domain.size[2]; ++z)
        for (size_t y = 1; y <= domain.size[1]; ++y) {
            const size_t offset = ((domain.begin[2] + z - 1) * domain.global[1]
                                + domain.begin[1] + y - 1) * domain.global[0] + domain.begin[0];
            std::copy_n(grid.data() + domain.index(1, y, z), domain.size[0], globalGrid.data() + offset);
        }
    const int sizes[3] = {static_cast<int>(domain.global[2]),
                         static_cast<int>(domain.global[1]), static_cast<int>(domain.global[0])};
    for (int peer = 1; peer < domain.processes; ++peer) {
        std::array<int, 3> coords{};
        Extent size{}, begin{};
        MPI_Cart_coords(domain.comm, peer, 3, coords.data());
        partition(domain.global, domain.dims, coords, size, begin);
        const int subsizes[3] = {static_cast<int>(size[2]), static_cast<int>(size[1]), static_cast<int>(size[0])};
        const int starts[3] = {static_cast<int>(begin[2]), static_cast<int>(begin[1]), static_cast<int>(begin[0])};
        MPI_Datatype block;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(globalGrid.data(), 1, block, peer, 6, domain.comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(globalGrid, "Grid");
}

bool validateResult(const std::vector<Real>& grid, const Domain& domain) {
    Real minValue = std::numeric_limits<Real>::infinity();
    Real maxValue = -std::numeric_limits<Real>::infinity();
    bool invalid = false;
    for (size_t z = 1; z <= domain.size[2]; ++z)
        for (size_t y = 1; y <= domain.size[1]; ++y)
            for (size_t x = 1; x <= domain.size[0]; ++x) {
                const Real value = grid[domain.index(x, y, z)];
                invalid |= !std::isfinite(value);
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
    const Real local[3] = {-minValue, maxValue, invalid ? 1.0 : 0.0};
    Real global[3]{};
    MPI_Reduce(local, global, 3, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (domain.rank != 0) return true;
    if (global[2] != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", -global[0], global[1]);
    if (global[1] > 1e6 || -global[0] < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

int runDistributed(const Options& options, const std::array<int, 3>& dims, MPI_Comm active) {
    Domain domain(options.size, dims, active);
    std::array<std::vector<Real>, 2> grids{std::vector<Real>(domain.cells), std::vector<Real>(domain.cells)};
    if (domain.rank == 0) printf("Initializing grid...\n");
    initializeGrid(grids, domain);
    int result = 0;
    {
        HaloExchange halos(domain, grids);
        if (domain.rank == 0) printf("Running stencil computation...\n");
        MPI_Barrier(domain.comm);
        const double start = MPI_Wtime();
        // When the entire grid consists of fixed boundaries, no exchange is
        // needed. Otherwise even ranks owning only boundary cells participate.
        if (options.size[0] > 2 && options.size[1] > 2 && options.size[2] > 2) {
            for (int iter = 0; iter < options.iterations; ++iter) {
                const int buffer = iter % 2;
                Real* input = grids[buffer].data();
                Real* output = grids[1 - buffer].data();
                halos.start(buffer, input);
                if (!domain.core.empty())
                    stencilBox(input, output, domain, domain.core, halos.requests[buffer].data(), halos.count);
                halos.finish(buffer, input);
                if (domain.core.empty()) {
                    stencilBox(input, output, domain, domain.updates);
                } else {
                    for (const auto& edge : domain.edges)
                        stencilBox(input, output, domain, edge);
                }
            }
        }
        const double localTime = MPI_Wtime() - start;
        double elapsed = 0;
        MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
        if (domain.rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
            double cellUpdates = static_cast<double>(options.iterations);
            for (size_t n : options.size) cellUpdates *= static_cast<double>(n > 2 ? n - 2 : 0);
            printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? cellUpdates / elapsed / 1e6 : 0.0);
        }
    }
    const auto& finalGrid = grids[options.iterations % 2];
    if (options.printResults) printDistributedResults(finalGrid, domain);
    if (options.validate) {
        if (domain.rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, domain);
        if (domain.rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    return result;
}

int runBenchmark(int argc, char** argv, int rank) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if ((strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0
                 || strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            const char* value = argv[++i];
            size_t number = 0;
            const auto parsed = std::from_chars(value, value + strlen(value), number);
            // Leave room for ghost cells in MPI's int-sized datatype dimensions.
            const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max()) - 2;
            if (parsed.ec != std::errc{} || *parsed.ptr != '\0' || number > limit) {
                if (rank == 0) fprintf(stderr, "Invalid value for -%c: %s\n", option, value);
                return 1;
            }
            if (option == 'i') options.iterations = static_cast<int>(number);
            else options.size[option == 'x' ? 0 : option == 'y' ? 1 : 2] = number;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (options.size[1] == 0) options.size[1] = options.size[0];
    if (options.size[2] == 0) options.size[2] = options.size[0];
    size_t cells = 1;
    for (size_t n : options.size) {
        if (n == 0 || cells > std::numeric_limits<size_t>::max() / n) {
            if (rank == 0) fprintf(stderr, "Grid dimensions must be positive and their product must fit size_t\n");
            return 1;
        }
        cells *= n;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", options.size[0], options.size[1], options.size[2]);
        printf("Iterations: %d\n", options.iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
    }
    int processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    const auto dims = processGrid(options.size, processes);
    const int activeCount = dims[0] * dims[1] * dims[2];
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeCount ? 0 : MPI_UNDEFINED, rank, &active);
    int result = 0;
    if (active != MPI_COMM_NULL) {
        result = runDistributed(options, dims, active);
        MPI_Comm_free(&active);
    }
    // Idle ranks (when a grid is too small for all ranks) share the exit status.
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
