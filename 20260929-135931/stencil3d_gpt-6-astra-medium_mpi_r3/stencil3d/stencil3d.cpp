#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Arrays describing the decomposition use MPI's C ordering: z, y, x.
struct Block {
    std::array<int, 3> size, start;
};

Block blockFor(const std::array<int, 3>& global, const int* dims, const int* coords) {
    Block b;
    for (int a = 0; a < 3; ++a) {
        const int base = global[a] / dims[a], extra = global[a] % dims[a];
        b.size[a] = base + (coords[a] < extra);
        b.start[a] = coords[a] * base + std::min(coords[a], extra);
    }
    return b;
}

// Minimize exchanged surface area, accounting for the actual grid aspect ratio.
// Slightly favor contiguous z faces over strided y/x faces when areas are similar.
// On very small grids, leave surplus ranks idle rather than allocate empty blocks.
int chooseTopology(const std::array<int, 3>& n, int ranks, int* dims) {
    for (int used = ranks; used > 0; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int z = 1; z <= std::min(n[0], used); ++z) {
            if (used % z) continue;
            const int rest = used / z;
            for (int y = 1; y <= std::min(n[1], rest); ++y) {
                if (rest % y) continue;
                const int x = rest / y;
                if (x > n[2]) continue;
                const double area = double(z - 1) * n[1] * n[2]
                                  + 1.02 * double(y - 1) * n[0] * n[2]
                                  + 1.10 * double(x - 1) * n[0] * n[1];
                if (area < best) {
                    best = area;
                    dims[0] = z; dims[1] = y; dims[2] = x;
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
}

MPI_Datatype subarray(const int* sizes, const int* subsizes, const int* starts) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
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

int run(int argc, char** argv, int rank, int ranks) {
    std::array<int, 3> global = {0, 0, 128};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "-x") || !strcmp(arg, "-y") ||
             !strcmp(arg, "-z") || !strcmp(arg, "-i")) && i + 1 < argc) {
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            const bool isIterations = !strcmp(arg, "-i");
            if (!*argv[i] || *end || value < 0 || value > INT_MAX - 2 ||
                (!isIterations && !strcmp(arg, "-x") && value == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid value for %s\n", arg);
                return 1;
            }
            if (isIterations) iterations = static_cast<int>(value);
            else global[!strcmp(arg, "-z") ? 0 : !strcmp(arg, "-y") ? 1 : 2] = static_cast<int>(value);
        } else if (!strcmp(arg, "-v")) validate = true;
        else if (!strcmp(arg, "-r")) printResults = true;
        else if (!strcmp(arg, "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", arg); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!global[0]) global[0] = global[2];
    if (!global[1]) global[1] = global[2];
    size_t gridSize = 1;
    for (int n : global) {
        if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(n))
            throw std::runtime_error("Grid is too large");
        gridSize *= n;
    }

    int dims[3];
    const int used = chooseTopology(global, static_cast<int>(std::min(size_t(ranks), gridSize)), dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= used) return 0;
    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 3, dims, periods, 0, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    const Block b = blockFor(global, dims, coords);
    int storage[3] = {b.size[0] + 2, b.size[1] + 2, b.size[2] + 2};
    const size_t row = storage[2], plane = row * storage[1];
    if (plane > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(storage[0]))
        throw std::runtime_error("Local grid is too large");
    std::vector<Real> input(plane * storage[0]), output(input.size());
    for (int z = 1; z <= b.size[0]; ++z)
        for (int y = 1; y <= b.size[1]; ++y)
            for (int x = 1; x <= b.size[2]; ++x) {
                const size_t index = (size_t(b.start[0] + z - 1) * global[1]
                                      + b.start[1] + y - 1) * global[2] + b.start[2] + x - 1;
                input[size_t(z) * plane + size_t(y) * row + x] = Real(index % 19);
            }
    // Physical boundary cells are never updated, so initialize both buffers once.
    output = input;
    int neighbor[6];
    MPI_Datatype sendType[6], recvType[6];
    for (int a = 0; a < 3; ++a) {
        MPI_Cart_shift(cart, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        for (int side = 0; side < 2; ++side) {
            const int f = 2*a + side;
            if (neighbor[f] == MPI_PROC_NULL) continue;
            auto face = b.size;
            face[a] = 1;
            int start[3] = {1, 1, 1};
            start[a] = side ? b.size[a] : 1;
            sendType[f] = subarray(storage, face.data(), start);
            start[a] = side ? b.size[a] + 1 : 0;
            recvType[f] = subarray(storage, face.data(), start);
        }
    }
    int lo[3], hi[3], coreLo[3], coreHi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = 1 + (b.start[a] == 0);
        hi[a] = b.size[a] + 1 - (b.start[a] + b.size[a] == global[a]);
        coreLo[a] = std::max(lo[a], 2);
        coreHi[a] = std::max(coreLo[a], std::min(hi[a], b.size[a]));
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\nValidation: %s\n",
               global[2], global[1], global[0], iterations, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid: %d x %d x %d)\n", used, dims[2], dims[1], dims[0]);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(cart);
    const double startTime = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (int f = 0; f < 6; ++f)
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Irecv(input.data(), 1, recvType[f], neighbor[f], f ^ 1, cart, &requests[count++]);
        for (int f = 0; f < 6; ++f)
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Isend(input.data(), 1, sendType[f], neighbor[f], f, cart, &requests[count++]);
        // Keep x contiguous for SIMD and preserve the original arithmetic order.
        const Real* src = input.data();
        Real* dst = output.data();
        auto updateRow = [&](int z, int y, int begin, int end) {
            const size_t base = size_t(z) * plane + size_t(y) * row;
            for (int x = begin; x < end; ++x) {
                const size_t i = base + x;
                dst[i] = (src[i] + src[i-1] + src[i+1] + src[i-row] + src[i+row]
                          + src[i-plane] + src[i+plane]) / 7.0;
            }
        };
        for (int z = coreLo[0]; z < coreHi[0]; ++z)
            for (int y = coreLo[1]; y < coreHi[1]; ++y)
                updateRow(z, y, coreLo[2], coreHi[2]);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Update the shell exactly once, including blocks only one cell wide.
        for (int z = lo[0]; z < hi[0]; ++z)
            for (int y = lo[1]; y < hi[1]; ++y) {
                if (z >= coreLo[0] && z < coreHi[0] && y >= coreLo[1] && y < coreHi[1]) {
                    updateRow(z, y, lo[2], std::min(hi[2], coreLo[2]));
                    updateRow(z, y, coreHi[2], hi[2]);
                } else updateRow(z, y, lo[2], hi[2]);
            }
        input.swap(output);
    }
    const double elapsed = MPI_Wtime() - startTime;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const double updates = double(std::max(0, global[0]-2)) * std::max(0, global[1]-2)
                               * std::max(0, global[2]-2) * iterations;
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (printResults) {
        // Derived datatypes restore global x-fastest order without int-sized
        // gather displacements or a full-grid allocation on every rank.
        int one[3] = {1, 1, 1};
        MPI_Datatype owned = subarray(storage, b.size.data(), one);
        if (rank == 0) {
            std::vector<Real> full(gridSize);
            for (int r = 0; r < used; ++r) {
                int rc[3];
                MPI_Cart_coords(cart, r, 3, rc);
                const Block rb = blockFor(global, dims, rc);
                MPI_Datatype destination = subarray(global.data(), rb.size.data(), rb.start.data());
                if (r == 0)
                    MPI_Sendrecv(input.data(), 1, owned, 0, 6, full.data(), 1, destination,
                                 0, 6, cart, MPI_STATUS_IGNORE);
                else MPI_Recv(full.data(), 1, destination, r, 6, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&destination);
            }
            print_results(full, "Grid");
        } else MPI_Send(input.data(), 1, owned, 0, 6, cart);
        MPI_Type_free(&owned);
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity(), localMax = -localMin;
        int localValid = 1;
        for (int z = 1; z <= b.size[0]; ++z)
            for (int y = 1; y <= b.size[1]; ++y)
                for (int x = 1; x <= b.size[2]; ++x) {
                    const Real v = input[size_t(z)*plane + size_t(y)*row + x];
                    localValid &= std::isfinite(v);
                    localMin = std::min(localMin, v);
                    localMax = std::max(localMax, v);
                }
        Real minimum, maximum;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&localMin, &minimum, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&localMax, &maximum, 1, MPI_DOUBLE, MPI_MAX, cart);
        valid &= minimum >= -1e6 && maximum <= 1e6;
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", minimum, maximum);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (int f = 0; f < 6; ++f) if (neighbor[f] != MPI_PROC_NULL) {
        MPI_Type_free(&sendType[f]);
        MPI_Type_free(&recvType[f]);
    }
    MPI_Comm_free(&cart);
    MPI_Comm_free(&active);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        status = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
