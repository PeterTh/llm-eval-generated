#include <mpi.h>

#include <algorithm>
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

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// Choose a compact process grid, including for highly anisotropic domains.
// Never create empty blocks; surplus ranks wait in MPI_Finalize.
int processGrid(int ranks, const int global[3], int dims[3]) {
    const size_t cells = size_t(global[0]) * global[1] * global[2];
    for (int n = int(std::min(size_t(ranks), cells)); n > 0; --n) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(n, global[0]); ++x) {
            if (n % x) continue;
            const int yz = n / x;
            for (int y = 1; y <= std::min(yz, global[1]); ++y) {
                if (yz % y) continue;
                const int z = yz / y;
                if (z > global[2]) continue;
                // Total internal face area, normalized by domain volume.
                const double cost = double(x - 1) / global[0]
                                  + double(y - 1) / global[1]
                                  + double(z - 1) / global[2];
                if (cost < best) {
                    best = cost;
                    dims[0] = x;
                    dims[1] = y;
                    dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return n;
    }
    return 1;
}

void block(const int global[3], const int dims[3], const int coords[3],
           int local[3], int begin[3]) {
    for (int d = 0; d < 3; ++d) {
        const int base = global[d] / dims[d];
        const int extra = global[d] % dims[d];
        local[d] = base + (coords[d] < extra);
        begin[d] = coords[d] * base + std::min(coords[d], extra);
    }
}

MPI_Datatype subarray(const int sizes[3], const int extents[3], const int starts[3]) {
    // Storage is x-contiguous, so MPI's C-order dimensions are z,y,x.
    const int s[3] = {sizes[2], sizes[1], sizes[0]};
    const int e[3] = {extents[2], extents[1], extents[0]};
    const int b[3] = {starts[2], starts[1], starts[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, s, e, b, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Bounds are half-open and exclude the fixed global boundary cells.
void stencilBox(const Real* __restrict__ input, Real* __restrict__ output,
                size_t nx, size_t ny, const int lo[3], const int hi[3]) {
    const size_t plane = nx * ny;
    for (int z = lo[2]; z < hi[2]; ++z) {
        for (int y = lo[1]; y < hi[1]; ++y) {
            const size_t row = idx3(0, y, z, nx, ny);
            for (int x = lo[0]; x < hi[0]; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1]
                           + input[i - nx] + input[i + nx]
                           + input[i - plane] + input[i + plane]) / 7.0;
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

int run(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int global[3] = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        int* value = nullptr;
        if (!strcmp(argv[i], "-x")) value = &global[0];
        else if (!strcmp(argv[i], "-y")) value = &global[1];
        else if (!strcmp(argv[i], "-z")) value = &global[2];
        else if (!strcmp(argv[i], "-i")) value = &iterations;
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
        if (value) {
            if (++i == argc) {
                if (worldRank == 0) fprintf(stderr, "Missing option value\n");
                return 1;
            }
            char* end;
            errno = 0;
            const long parsed = strtol(argv[i], &end, 10);
            if (errno || end == argv[i] || *end || parsed < 0 || parsed > INT_MAX - 2) {
                if (worldRank == 0) fprintf(stderr, "Invalid option value: %s\n", argv[i]);
                return 1;
            }
            *value = int(parsed);
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    if (!global[0]) {
        if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }
    size_t cells = 1;
    for (int d = 0; d < 3; ++d) {
        if (cells > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(global[d]))
            throw std::overflow_error("Grid size is too large");
        cells *= global[d];
    }

    int dims[3] = {};
    const int active = processGrid(worldSize, global, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED,
                   worldRank, &workers);
    if (worldRank >= active) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&workers);
    int rank, coords[3], local[3], begin[3], padded[3];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 3, coords);
    block(global, dims, coords, local, begin);
    size_t storage = 1;
    for (int d = 0; d < 3; ++d) {
        padded[d] = local[d] + 2;
        if (storage > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(padded[d]))
            throw std::overflow_error("Local grid size is too large");
        storage *= padded[d];
    }
    std::vector<Real> grid1(storage), grid2;
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    for (int z = 1; z <= local[2]; ++z)
        for (int y = 1; y <= local[1]; ++y)
            for (int x = 1; x <= local[0]; ++x) {
                const size_t g = idx3(begin[0] + x - 1, begin[1] + y - 1,
                                      begin[2] + z - 1, global[0], global[1]);
                grid1[idx3(x, y, z, padded[0], padded[1])] = Real(g % 19);
            }
    // Both buffers retain the fixed boundaries for the entire run.
    grid2 = grid1;

    MPI_Datatype faces[3];
    int neighbors[3][2], lo[3], hi[3], innerLo[3], innerHi[3];
    size_t sendOffset[3][2], recvOffset[3][2];
    const int zero[3] = {0, 0, 0};
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbors[d][0], &neighbors[d][1]);
        int extents[3] = {local[0], local[1], local[2]};
        extents[d] = 1;
        faces[d] = subarray(padded, extents, zero);
        for (int side = 0; side < 2; ++side) {
            int s[3] = {1, 1, 1}, r[3] = {1, 1, 1};
            s[d] = side ? local[d] : 1;
            r[d] = side ? local[d] + 1 : 0;
            sendOffset[d][side] = idx3(s[0], s[1], s[2], padded[0], padded[1]);
            recvOffset[d][side] = idx3(r[0], r[1], r[2], padded[0], padded[1]);
        }
        lo[d] = 1 + (begin[d] == 0);
        hi[d] = local[d] + 1 - (begin[d] + local[d] == global[d]);
        hi[d] = std::max(lo[d], hi[d]);
        innerLo[d] = std::min(hi[d], std::max(lo[d], 2));
        innerHi[d] = std::max(innerLo[d], std::min(hi[d], local[d]));
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (int d = 0; d < 3; ++d)
            for (int side = 0; side < 2; ++side)
                if (neighbors[d][side] != MPI_PROC_NULL)
                    MPI_Irecv(grid1.data() + recvOffset[d][side], 1, faces[d],
                              neighbors[d][side], 2 * d + (1 - side), cart, &requests[count++]);
        for (int d = 0; d < 3; ++d)
            for (int side = 0; side < 2; ++side)
                if (neighbors[d][side] != MPI_PROC_NULL)
                    MPI_Isend(grid1.data() + sendOffset[d][side], 1, faces[d],
                              neighbors[d][side], 2 * d + side, cart, &requests[count++]);

        stencilBox(grid1.data(), grid2.data(), padded[0], padded[1], innerLo, innerHi);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        // Partition the remaining shell into six disjoint boxes. This also
        // handles blocks just one or two cells wide without duplicate updates.
        int shellLo[3] = {lo[0], lo[1], lo[2]};
        int shellHi[3] = {hi[0], hi[1], hi[2]};
        for (int d = 0; d < 3; ++d) {
            shellHi[d] = innerLo[d];
            stencilBox(grid1.data(), grid2.data(), padded[0], padded[1], shellLo, shellHi);
            shellLo[d] = innerHi[d];
            shellHi[d] = hi[d];
            stencilBox(grid1.data(), grid2.data(), padded[0], padded[1], shellLo, shellHi);
            shellLo[d] = innerLo[d];
            shellHi[d] = innerHi[d];
        }
        grid1.swap(grid2);
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", seconds * 1000.0);
        const double updates = double(std::max(0, global[0] - 2))
                             * std::max(0, global[1] - 2)
                             * std::max(0, global[2] - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
    }

    if (printResults) {
        // Only this optional output path requires a global array. Receive each
        // block directly into its global position, preserving the serial hash
        // and Kahan summation order, with no MPI int-sized element counts.
        const int one[3] = {1, 1, 1};
        MPI_Datatype owned = subarray(padded, local, one);
        if (rank == 0) {
            std::vector<Real> full(cells);
            for (int z = 0; z < local[2]; ++z)
                for (int y = 0; y < local[1]; ++y)
                    std::copy_n(grid1.data() + idx3(1, y + 1, z + 1, padded[0], padded[1]),
                                local[0], full.data() + idx3(begin[0], begin[1] + y,
                                                           begin[2] + z, global[0], global[1]));
            for (int source = 1; source < active; ++source) {
                int c[3], n[3], b[3];
                MPI_Cart_coords(cart, source, 3, c);
                block(global, dims, c, n, b);
                MPI_Datatype destination = subarray(global, n, b);
                MPI_Recv(full.data(), 1, destination, source, 10, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&destination);
            }
            print_results(full, "Grid");
        } else {
            MPI_Send(grid1.data(), 1, owned, 0, 10, cart);
        }
        MPI_Type_free(&owned);
    }

    int valid = 1;
    if (validate) {
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -minVal;
        int finite = 1;
        for (int z = 1; z <= local[2]; ++z)
            for (int y = 1; y <= local[1]; ++y)
                for (int x = 1; x <= local[0]; ++x) {
                    const Real value = grid1[idx3(x, y, z, padded[0], padded[1])];
                    finite &= std::isfinite(value);
                    minVal = std::min(minVal, value);
                    maxVal = std::max(maxVal, value);
                }
        double globalMin, globalMax;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, cart);
        valid = valid && globalMin >= -1e6 && globalMax <= 1e6;
        if (rank == 0) {
            printf("Validating result...\n");
            printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (auto& face : faces) MPI_Type_free(&face);
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& e) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
