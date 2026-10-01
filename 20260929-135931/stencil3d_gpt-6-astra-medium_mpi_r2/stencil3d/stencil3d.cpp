#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <new>
#include <exception>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Choose a process grid that fits even thin domains. Prefer all ranks, then
// minimize face traffic, with a small penalty for strided X faces.
std::array<int, 3> processGrid(const int* n, int ranks) {
    std::array<int, 3> best{1, 1, 1};
    int used = 0;
    double cost = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= std::min(n[0], ranks); ++x) {
        for (int y = 1; y <= std::min(n[1], ranks / x); ++y) {
            int z = std::min(n[2], ranks / x / y);
            int count = x * y * z;
            double traffic = 1.1 * (x - 1) * double(n[1]) * n[2]
                           + (y - 1) * double(n[0]) * n[2]
                           + (z - 1) * double(n[0]) * n[1];
            if (count > used || (count == used && traffic < cost)) {
                best = {x, y, z};
                used = count;
                cost = traffic;
            }
        }
    }
    return best;
}

struct Block {
    int n[3], start[3];
    size_t pitch, plane;
    Block(const int* global, const int* dims, const int* coords) {
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coords[a] < global[a] % dims[a]);
            start[a] = coords[a] * (global[a] / dims[a])
                     + std::min(coords[a], global[a] % dims[a]);
        }
        pitch = size_t(n[0]) + 2;
        plane = pitch * (size_t(n[1]) + 2);
    }
    size_t index(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * pitch + x;
    }
};

// Disjoint rectangular regions keep the innermost loop contiguous and free of
// boundary tests. Preserve the serial expression's floating-point order.
void update(const Real* __restrict__ in, Real* __restrict__ out,
            const Block& b, const int* lo, const int* hi) {
    for (int z = lo[2]; z < hi[2]; ++z)
        for (int y = lo[1]; y < hi[1]; ++y) {
            size_t row = b.index(0, y, z);
            for (int x = lo[0]; x < hi[0]; ++x) {
                size_t i = row + x;
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-b.pitch]
                        + in[i+b.pitch] + in[i-b.plane] + in[i+b.plane]) / 7.0;
            }
        }
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
    int global[3] = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            char option = argv[i][1];
            char* end = nullptr;
            long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX - 2 ||
                (option == 'x' && value == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid value for -%c\n", option);
                return 1;
            }
            if (option == 'i') iterations = int(value);
            else global[option - 'x'] = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    size_t total = 1;
    for (int a = 0; a < 3; ++a) {
        if (total > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(global[a])) {
            if (rank == 0) fprintf(stderr, "Grid is too large\n");
            return 1;
        }
        total *= global[a];
    }
    auto dims = processGrid(global, ranks);
    int active = dims[0] * dims[1] * dims[2];
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    // Excess ranks participate in final status propagation in main.
    if (rank >= active) return 0;
    int periodic[3] = {0, 0, 0}, coords[3];
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims.data(), periodic, 0, &cart);
    MPI_Comm_free(&workers);
    MPI_Cart_coords(cart, rank, 3, coords);
    Block b(global, dims.data(), coords);
    if (b.plane > std::numeric_limits<size_t>::max() / sizeof(Real) / (size_t(b.n[2]) + 2))
        throw std::bad_alloc();
    std::vector<Real> grid1(b.plane * (size_t(b.n[2]) + 2));
    for (int z = 1; z <= b.n[2]; ++z)
        for (int y = 1; y <= b.n[1]; ++y)
            for (int x = 1; x <= b.n[0]; ++x) {
                size_t i = (size_t(b.start[2] + z - 1) * global[1]
                         + b.start[1] + y - 1) * global[0] + b.start[0] + x - 1;
                grid1[b.index(x, y, z)] = Real(i % 19);
            }
    // Physical boundaries are immutable in both buffers.
    std::vector<Real> grid2 = grid1;
    int neighbors[6];
    MPI_Datatype faces[3];
    size_t sendOffset[6], recvOffset[6];
    int sizes[3] = {b.n[2] + 2, b.n[1] + 2, b.n[0] + 2};
    int lo[3], hi[3], coreLo[3], coreHi[3];
    for (int a = 0; a < 3; ++a) {
        MPI_Cart_shift(cart, a, 1, &neighbors[2*a], &neighbors[2*a+1]);
        int subsizes[3] = {b.n[2], b.n[1], b.n[0]};
        int starts[3] = {0, 0, 0};
        subsizes[2-a] = 1;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &faces[a]);
        MPI_Type_commit(&faces[a]);
        int pos[3] = {1, 1, 1};
        sendOffset[2*a] = b.index(pos[0], pos[1], pos[2]);
        pos[a] = 0;
        recvOffset[2*a] = b.index(pos[0], pos[1], pos[2]);
        pos[a] = b.n[a];
        sendOffset[2*a+1] = b.index(pos[0], pos[1], pos[2]);
        pos[a]++;
        recvOffset[2*a+1] = b.index(pos[0], pos[1], pos[2]);
        lo[a] = 1 + (b.start[a] == 0);
        hi[a] = std::max(lo[a], b.n[a] + 1 - (b.start[a] + b.n[a] == global[a]));
        coreLo[a] = std::min(hi[a], std::max(lo[a], 2));
        coreHi[a] = std::max(coreLo[a], std::min(hi[a], b.n[a]));
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI process grid: %d x %d x %d\n", dims[0], dims[1], dims[2]);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(cart);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (int f = 0; f < 6; ++f)
            if (neighbors[f] != MPI_PROC_NULL)
                MPI_Irecv(grid1.data() + recvOffset[f], 1, faces[f/2], neighbors[f],
                          f ^ 1, cart, &requests[count++]);
        for (int f = 0; f < 6; ++f)
            if (neighbors[f] != MPI_PROC_NULL)
                MPI_Isend(grid1.data() + sendOffset[f], 1, faces[f/2], neighbors[f],
                          f, cart, &requests[count++]);
        update(grid1.data(), grid2.data(), b, coreLo, coreHi);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Peel six disjoint slabs around the already computed core.
        int remLo[3], remHi[3];
        std::copy(lo, lo + 3, remLo);
        std::copy(hi, hi + 3, remHi);
        for (int a = 0; a < 3; ++a) {
            int end = remHi[a];
            remHi[a] = coreLo[a];
            update(grid1.data(), grid2.data(), b, remLo, remHi);
            remHi[a] = end;
            remLo[a] = coreHi[a];
            update(grid1.data(), grid2.data(), b, remLo, remHi);
            remLo[a] = coreLo[a];
            remHi[a] = coreHi[a];
        }
        grid1.swap(grid2);
    }
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        double updates = double(std::max(0, global[0]-2)) * std::max(0, global[1]-2)
                       * std::max(0, global[2]-2) * iterations;
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }
    if (printResults) {
        // Receive each block directly into its global position; no packing or
        // global-sized storage is needed on any non-root rank.
        int localStarts[3] = {1, 1, 1};
        int localSizes[3] = {b.n[2], b.n[1], b.n[0]};
        MPI_Datatype owned;
        MPI_Type_create_subarray(3, sizes, localSizes, localStarts, MPI_ORDER_C, MPI_DOUBLE, &owned);
        MPI_Type_commit(&owned);
        if (rank != 0) MPI_Send(grid1.data(), 1, owned, 0, 10, cart);
        else {
            std::vector<Real> result(total);
            for (int z = 1; z <= b.n[2]; ++z)
                for (int y = 1; y <= b.n[1]; ++y)
                    std::copy_n(grid1.data() + b.index(1, y, z), b.n[0],
                        result.data() + (size_t(b.start[2]+z-1)*global[1]+b.start[1]+y-1)*global[0]+b.start[0]);
            int globalSizes[3] = {global[2], global[1], global[0]};
            for (int r = 1; r < active; ++r) {
                MPI_Cart_coords(cart, r, 3, coords);
                Block other(global, dims.data(), coords);
                int extents[3] = {other.n[2], other.n[1], other.n[0]};
                int offsets[3] = {other.start[2], other.start[1], other.start[0]};
                MPI_Datatype target;
                MPI_Type_create_subarray(3, globalSizes, extents, offsets, MPI_ORDER_C, MPI_DOUBLE, &target);
                MPI_Type_commit(&target);
                MPI_Recv(result.data(), 1, target, r, 10, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
            print_results(result, "Grid");
        }
        MPI_Type_free(&owned);
    }
    int valid = 1;
    if (validate) {
        Real minVal = std::numeric_limits<Real>::infinity(), maxVal = -minVal;
        for (int z = 1; z <= b.n[2]; ++z)
            for (int y = 1; y <= b.n[1]; ++y)
                for (int x = 1; x <= b.n[0]; ++x) {
                    Real v = grid1[b.index(x, y, z)];
                    if (!std::isfinite(v)) valid = 0;
                    minVal = std::min(minVal, v);
                    maxVal = std::max(maxVal, v);
                }
        Real globalMin, globalMax;
        MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, cart);
        valid = valid && globalMin >= -1e6 && globalMax <= 1e6;
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (auto& face : faces) MPI_Type_free(&face);
    MPI_Comm_free(&cart);
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
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
