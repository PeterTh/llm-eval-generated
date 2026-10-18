#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;
using Int3 = std::array<int, 3>;

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// Minimize exchanged face area, respecting the shape of the global domain.
// Surplus ranks are idle only when no nonempty decomposition can use them all.
Int3 processGrid(const Int3& global, int worldSize, int& active) {
    for (active = worldSize; active > 0; --active) {
        Int3 best{0, 0, 0};
        double bestCost = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(global[0], active); ++x) {
            if (active % x) continue;
            int rest = active / x;
            for (int y = 1; y <= std::min(global[1], rest); ++y) {
                if (rest % y) continue;
                int z = rest / y;
                if (z > global[2]) continue;
                double cost = double(x - 1) * global[1] * global[2]
                            + double(y - 1) * global[0] * global[2]
                            + double(z - 1) * global[0] * global[1];
                // Break ties in favor of balanced local workloads.
                double volume = double((global[0] + x - 1) / x)
                              * ((global[1] + y - 1) / y) * ((global[2] + z - 1) / z);
                cost += volume / (double(global[0]) * global[1] * global[2] + 1.0);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = {x, y, z};
                }
            }
        }
        if (best[0]) return best;
    }
    return {1, 1, 1};
}

void block(const Int3& global, const Int3& dims, const Int3& coords,
           Int3& n, Int3& offset) {
    for (int d = 0; d < 3; ++d) {
        int base = global[d] / dims[d], extra = global[d] % dims[d];
        n[d] = base + (coords[d] < extra);
        offset[d] = coords[d] * base + std::min(coords[d], extra);
    }
}

// Coordinates are x,y,z; MPI's C-order subarrays use z,y,x.
MPI_Datatype subarray(const Int3& sizes, const Int3& counts, const Int3& starts) {
    int s[3] = {sizes[2], sizes[1], sizes[0]};
    int c[3] = {counts[2], counts[1], counts[0]};
    int b[3] = {starts[2], starts[1], starts[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, s, c, b, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Unit-stride inner loop retains the original floating-point evaluation order.
void stencilBox(const Real* __restrict__ input, Real* __restrict__ output,
                size_t nx, size_t ny, const Int3& begin, const Int3& end) {
    const size_t plane = nx * ny;
    for (int z = begin[2]; z < end[2]; ++z) {
        for (int y = begin[1]; y < end[1]; ++y) {
            size_t row = idx3(0, y, z, nx, ny);
            for (int x = begin[0]; x < end[0]; ++x) {
                size_t i = row + x;
                output[i] = (input[i] + input[i-1] + input[i+1]
                           + input[i-nx] + input[i+nx]
                           + input[i-plane] + input[i+plane]) / 7.0;
            }
        }
    }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int worldRank, int worldSize) {
    Int3 global{128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "-r")) {
            printResults = true;
        } else if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
                    !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            char option = argv[i][1];
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || *end || end == argv[i] || value < 0 || value > INT_MAX - 2 ||
                (option == 'x' && value == 0)) {
                if (worldRank == 0) fprintf(stderr, "Invalid value for -%c: %s\n", option, argv[i]);
                return 1;
            }
            if (option == 'i') iterations = static_cast<int>(value);
            else global[option - 'x'] = static_cast<int>(value);
        } else {
            if (worldRank == 0) {
                printf("Unknown or incomplete option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    size_t total = 1;
    for (int d : global) {
        if (total > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(d)) {
            if (worldRank == 0) fprintf(stderr, "Grid size exceeds addressable memory\n");
            return 1;
        }
        total *= d;
    }

    int active;
    Int3 dims = processGrid(global, worldSize, active);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    if (workers == MPI_COMM_NULL) return 0;
    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims.data(), periods, 0, &cart);
    MPI_Comm_free(&workers);
    int rank;
    MPI_Comm_rank(cart, &rank);
    Int3 coords, n, offset, padded;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    block(global, dims, coords, n, offset);
    for (int d = 0; d < 3; ++d) padded[d] = n[d] + 2;
    size_t localSize = 1;
    for (int d : padded) {
        if (localSize > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(d))
            MPI_Abort(MPI_COMM_WORLD, 1);
        localSize *= d;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("MPI process grid: %d x %d x %d (%d ranks)\n", dims[0], dims[1], dims[2], active);
        printf("Initializing grid...\n");
    }
    std::vector<Real> input(localSize), output(localSize);
    for (int z = 1; z <= n[2]; ++z)
        for (int y = 1; y <= n[1]; ++y)
            for (int x = 1; x <= n[0]; ++x) {
                size_t local = idx3(x, y, z, padded[0], padded[1]);
                size_t index = idx3(offset[0] + x - 1, offset[1] + y - 1,
                                    offset[2] + z - 1, global[0], global[1]);
                input[local] = output[local] = (index % 19) * 1.0;
            }

    int neighbors[6];
    MPI_Datatype sendTypes[6], recvTypes[6];
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbors[2*d], &neighbors[2*d+1]);
        Int3 counts = n;
        counts[d] = 1;
        for (int side = 0; side < 2; ++side) {
            Int3 start{1, 1, 1};
            start[d] = side ? n[d] : 1;
            sendTypes[2*d+side] = subarray(padded, counts, start);
            start[d] = side ? n[d] + 1 : 0;
            recvTypes[2*d+side] = subarray(padded, counts, start);
        }
    }
    Int3 begin, end, coreBegin, coreEnd;
    for (int d = 0; d < 3; ++d) {
        begin[d] = 1 + (offset[d] == 0);
        end[d] = std::max(begin[d], n[d] + 1 - (offset[d] + n[d] == global[d]));
        coreBegin[d] = std::min(end[d], std::max(begin[d], 2));
        coreEnd[d] = std::max(coreBegin[d], std::min(end[d], n[d]));
    }
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    double startTime = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (int face = 0; face < 6; ++face)
            if (neighbors[face] != MPI_PROC_NULL)
                MPI_Irecv(input.data(), 1, recvTypes[face], neighbors[face], face ^ 1,
                          cart, &requests[count++]);
        for (int face = 0; face < 6; ++face)
            if (neighbors[face] != MPI_PROC_NULL)
                MPI_Isend(input.data(), 1, sendTypes[face], neighbors[face], face,
                          cart, &requests[count++]);
        stencilBox(input.data(), output.data(), padded[0], padded[1], coreBegin, coreEnd);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Six disjoint slabs cover the shell without revisiting the core.
        Int3 lo = begin, hi = end;
        for (int d = 2; d >= 0; --d) {
            Int3 slabEnd = hi;
            slabEnd[d] = coreBegin[d];
            stencilBox(input.data(), output.data(), padded[0], padded[1], lo, slabEnd);
            Int3 slabBegin = lo;
            slabBegin[d] = coreEnd[d];
            stencilBox(input.data(), output.data(), padded[0], padded[1], slabBegin, hi);
            lo[d] = coreBegin[d];
            hi[d] = coreEnd[d];
        }
        // Physical boundaries were initialized in both buffers and stay fixed.
        input.swap(output);
    }
    double elapsed = MPI_Wtime() - startTime, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        double updates = double(std::max(0, global[0]-2)) * std::max(0, global[1]-2)
                       * std::max(0, global[2]-2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    if (printResults) {
        MPI_Datatype owned = subarray(padded, n, {1, 1, 1});
        if (rank == 0) {
            std::vector<Real> full(total);
            for (int z = 0; z < n[2]; ++z)
                for (int y = 0; y < n[1]; ++y)
                    std::copy_n(input.data() + idx3(1, y+1, z+1, padded[0], padded[1]), n[0],
                                full.data() + idx3(offset[0], offset[1]+y, offset[2]+z, global[0], global[1]));
            for (int source = 1; source < active; ++source) {
                Int3 c, count, origin;
                MPI_Cart_coords(cart, source, 3, c.data());
                block(global, dims, c, count, origin);
                MPI_Datatype destination = subarray(global, count, origin);
                MPI_Recv(full.data(), 1, destination, source, 0, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&destination);
            }
            print_results(full, "Grid");
        } else {
            MPI_Send(input.data(), 1, owned, 0, 0, cart);
        }
        MPI_Type_free(&owned);
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity(), localMax = -localMin;
        int finite = 1;
        for (int z = 1; z <= n[2]; ++z)
            for (int y = 1; y <= n[1]; ++y)
                for (int x = 1; x <= n[0]; ++x) {
                    Real value = input[idx3(x, y, z, padded[0], padded[1])];
                    finite &= std::isfinite(value);
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
        Real minValue, maxValue;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, cart);
        valid = valid && minValue >= -1e6 && maxValue <= 1e6;
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", minValue, maxValue);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (int face = 0; face < 6; ++face) {
        MPI_Type_free(&sendTypes[face]);
        MPI_Type_free(&recvTypes[face]);
    }
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
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
