#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Choose a balanced decomposition subject to nonempty local domains. Prefer
// contiguous faces when two decompositions have the same surface area.
static bool chooseDimensions(int ranks, const size_t global[3], int dims[3]) {
    double best = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= ranks && size_t(x) <= global[0]; ++x) {
        if (ranks % x) continue;
        const int remainder = ranks / x;
        for (int y = 1; y <= remainder && size_t(y) <= global[1]; ++y) {
            if (remainder % y) continue;
            const int z = remainder / y;
            if (size_t(z) > global[2]) continue;
            const double cost = double(x - 1) / global[0] * 1.001
                              + double(y - 1) / global[1] * 1.0001
                              + double(z - 1) / global[2];
            if (cost < best) {
                best = cost;
                dims[0] = x; dims[1] = y; dims[2] = z;
            }
        }
    }
    return std::isfinite(best);
}

static void block(const size_t global[3], const int dims[3], const int coords[3],
                  int local[3], size_t offset[3]) {
    for (int d = 0; d < 3; ++d) {
        const size_t base = global[d] / dims[d];
        const size_t extra = global[d] % dims[d];
        local[d] = int(base + (size_t(coords[d]) < extra));
        offset[d] = base * coords[d] + std::min(size_t(coords[d]), extra);
    }
}

static MPI_Datatype subarray(const int sizes[3], const int count[3], const int start[3]) {
    // MPI array order is z, y, x; x remains the contiguous/vectorized dimension.
    int s[3] = {sizes[2], sizes[1], sizes[0]};
    int c[3] = {count[2], count[1], count[0]};
    int o[3] = {start[2], start[1], start[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, s, c, o, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Half-open local ranges. Physical boundaries are initialized in both buffers
// and never rewritten. Preserve the original floating-point addition order.
static void update(const Real* __restrict input, Real* __restrict output,
                   size_t row, size_t plane, const int lo[3], const int hi[3]) {
    for (int z = lo[2]; z < hi[2]; ++z) {
        for (int y = lo[1]; y < hi[1]; ++y) {
            const size_t begin = size_t(z) * plane + size_t(y) * row;
            for (int x = lo[0]; x < hi[0]; ++x) {
                const size_t i = begin + x;
                output[i] = (input[i] + input[i-1] + input[i+1]
                           + input[i-row] + input[i+row]
                           + input[i-plane] + input[i+plane]) / 7.0;
            }
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

static int run(int argc, char** argv) {
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > INT_MAX - 2 || ny > INT_MAX - 2 ||
        nz > INT_MAX - 2 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz / sizeof(Real)) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t global[3] = {nx, ny, nz};
    int dims[3];
    int active = int(std::min(size_t(worldSize), nx * ny * nz));
    while (!chooseDimensions(active, global, dims)) --active;
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    if (rank >= active) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&workers);
    int coords[3], local[3];
    size_t offset[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    block(global, dims, coords, local, offset);
    const int sizes[3] = {local[0]+2, local[1]+2, local[2]+2};
    const size_t row = sizes[0], plane = row * sizes[1];
    if (plane > std::numeric_limits<size_t>::max() / size_t(sizes[2]) / sizeof(Real)) {
        if (rank == 0) fprintf(stderr, "Local grid allocation is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Real> grid1(plane * sizes[2]);
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    for (int z = 1; z <= local[2]; ++z)
        for (int y = 1; y <= local[1]; ++y)
            for (int x = 1; x <= local[0]; ++x) {
                const size_t index = (offset[2]+z-1)*nx*ny + (offset[1]+y-1)*nx + offset[0]+x-1;
                grid1[size_t(z)*plane + size_t(y)*row + x] = Real(index % 19);
            }
    std::vector<Real> grid2 = grid1;
    std::array<int, 6> neighbors;
    std::array<MPI_Datatype, 6> sendTypes, recvTypes;
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbors[2*d], &neighbors[2*d+1]);
        for (int side = 0; side < 2; ++side) {
            const int face = 2*d+side;
            sendTypes[face] = recvTypes[face] = MPI_DATATYPE_NULL;
            if (neighbors[face] == MPI_PROC_NULL) continue;
            int count[3] = {local[0], local[1], local[2]};
            int start[3] = {1, 1, 1};
            count[d] = 1;
            start[d] = side ? local[d] : 1;
            sendTypes[face] = subarray(sizes, count, start);
            start[d] = side ? local[d]+1 : 0;
            recvTypes[face] = subarray(sizes, count, start);
        }
    }
    int lo[3], hi[3], innerLo[3], innerHi[3];
    bool hasInner = true;
    for (int d = 0; d < 3; ++d) {
        lo[d] = 1 + (offset[d] == 0);
        hi[d] = local[d] + 1 - (offset[d]+local[d] == global[d]);
        innerLo[d] = std::max(lo[d], 2);
        innerHi[d] = std::min(hi[d], local[d]);
        hasInner = hasInner && innerLo[d] < innerHi[d];
    }
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double startTime = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int nrequests = 0;
        for (int face = 0; face < 6; ++face) {
            if (neighbors[face] == MPI_PROC_NULL) continue;
            MPI_Irecv(grid1.data(), 1, recvTypes[face], neighbors[face], face ^ 1,
                      cart, &requests[nrequests++]);
        }
        for (int face = 0; face < 6; ++face) {
            if (neighbors[face] == MPI_PROC_NULL) continue;
            MPI_Isend(grid1.data(), 1, sendTypes[face], neighbors[face], face,
                      cart, &requests[nrequests++]);
        }
        if (hasInner) {
            // Periodic polling also permits overlap on MPI implementations that
            // require calls into MPI to progress outstanding transfers.
            int tileLo[3] = {innerLo[0], innerLo[1], innerLo[2]};
            int tileHi[3] = {innerHi[0], innerHi[1], innerHi[2]};
            for (int z = innerLo[2]; z < innerHi[2]; z += 8) {
                tileLo[2] = z;
                tileHi[2] = std::min(z + 8, innerHi[2]);
                update(grid1.data(), grid2.data(), row, plane, tileLo, tileHi);
                if (nrequests) {
                    int complete;
                    MPI_Testall(nrequests, requests, &complete, MPI_STATUSES_IGNORE);
                }
            }
        }
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        if (!hasInner) {
            update(grid1.data(), grid2.data(), row, plane, lo, hi);
        } else {
            // Six disjoint slabs cover the remaining shell without a per-cell branch.
            int shellLo[3] = {lo[0], lo[1], lo[2]};
            int shellHi[3] = {hi[0], hi[1], hi[2]};
            for (int d = 0; d < 3; ++d) {
                shellHi[d] = innerLo[d];
                update(grid1.data(), grid2.data(), row, plane, shellLo, shellHi);
                shellHi[d] = hi[d];
                shellLo[d] = innerHi[d];
                update(grid1.data(), grid2.data(), row, plane, shellLo, shellHi);
                shellLo[d] = innerLo[d];
                shellHi[d] = innerHi[d];
            }
        }
        grid1.swap(grid2);
    }
    const double localTime = MPI_Wtime() - startTime;
    double elapsed = 0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", long(elapsed * 1000));
        const double updates = double(nx > 2 ? nx-2 : 0) * double(ny > 2 ? ny-2 : 0)
                             * double(nz > 2 ? nz-2 : 0) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0);
    }
    if (printResults) {
        const int begin[3] = {1, 1, 1};
        MPI_Datatype owned = subarray(sizes, local, begin);
        if (rank == 0) {
            std::vector<Real> full(nx * ny * nz);
            const int fullSizes[3] = {int(nx), int(ny), int(nz)};
            for (int source = 0; source < active; ++source) {
                int sourceCoords[3], sourceLocal[3];
                size_t sourceOffset[3];
                MPI_Cart_coords(cart, source, 3, sourceCoords);
                block(global, dims, sourceCoords, sourceLocal, sourceOffset);
                if (source == 0) {
                    for (int z = 0; z < local[2]; ++z)
                        for (int y = 0; y < local[1]; ++y)
                            std::copy_n(grid1.data() + size_t(z+1)*plane + size_t(y+1)*row + 1,
                                        local[0], full.data() + (offset[2]+z)*nx*ny + (offset[1]+y)*nx + offset[0]);
                } else {
                    const int sourceStart[3] = {int(sourceOffset[0]), int(sourceOffset[1]), int(sourceOffset[2])};
                    MPI_Datatype destination = subarray(fullSizes, sourceLocal, sourceStart);
                    MPI_Recv(full.data(), 1, destination, source, 0, cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&destination);
                }
            }
            print_results(full, "Grid");
        } else {
            MPI_Send(grid1.data(), 1, owned, 0, 0, cart);
        }
        MPI_Type_free(&owned);
    }
    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int finite = 1;
        for (int z = 1; z <= local[2]; ++z)
            for (int y = 1; y <= local[1]; ++y)
                for (int x = 1; x <= local[0]; ++x) {
                    const Real value = grid1[size_t(z)*plane + size_t(y)*row + x];
                    finite = finite && std::isfinite(value);
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
        Real minValue, maxValue;
        MPI_Allreduce(&finite, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, cart);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) printf("Validation failed: found NaN or Inf value\n");
            else printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
        }
        if (maxValue > 1e6 || minValue < -1e6) {
            valid = 0;
            if (rank == 0) printf("Validation failed: values out of expected range\n");
        }
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    for (int face = 0; face < 6; ++face) {
        if (sendTypes[face] != MPI_DATATYPE_NULL) MPI_Type_free(&sendTypes[face]);
        if (recvTypes[face] != MPI_DATATYPE_NULL) MPI_Type_free(&recvTypes[face]);
    }
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Stencil failed: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
