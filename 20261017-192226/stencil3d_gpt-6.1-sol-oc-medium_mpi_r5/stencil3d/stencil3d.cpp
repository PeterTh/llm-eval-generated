#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Coordinates and MPI subarrays use z,y,x order so x remains contiguous.
using Shape = std::array<int, 3>;

size_t index3(int z, int y, int x, const Shape& size) {
    return (size_t(z) * size[1] + y) * size[2] + x;
}

void partition(const Shape& global, const Shape& dims, const Shape& coords,
               Shape& local, Shape& offset) {
    for (int d = 0; d < 3; ++d) {
        local[d] = global[d] / dims[d] + (coords[d] < global[d] % dims[d]);
        offset[d] = coords[d] * (global[d] / dims[d])
                  + std::min(coords[d], global[d] % dims[d]);
    }
}

// Minimize total exchanged face area, with a load-imbalance penalty. Unlike
// MPI_Dims_create this also accounts for elongated grids and short dimensions.
int processGrid(int ranks, const Shape& global, Shape& dims) {
    for (int used = ranks; used > 0; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int z = 1; z <= std::min(used, global[0]); ++z) {
            if (used % z) continue;
            const int rest = used / z;
            for (int y = 1; y <= std::min(rest, global[1]); ++y) {
                if (rest % y) continue;
                const int x = rest / y;
                if (x > global[2]) continue;
                const double volume = double(global[0]) * global[1] * global[2];
                const double largest = double((global[0] - 1) / z + 1)
                    * ((global[1] - 1) / y + 1) * ((global[2] - 1) / x + 1);
                const double area = double(z - 1) * global[1] * global[2]
                    + double(y - 1) * global[0] * global[2]
                    + double(x - 1) * global[0] * global[1];
                const double score = (area + 1) * (largest * used / volume);
                // Break geometric ties in favor of z, then y decomposition:
                // those faces have contiguous x runs and cheaper MPI packing.
                if (score <= best) {
                    best = score;
                    dims = {z, y, x};
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
}

MPI_Datatype subarray(const Shape& size, const Shape& count, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, size.data(), count.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Half-open boxes, with the original evaluation order preserved for exact
// results. The contiguous inner loop is automatically vectorized at -O3.
void updateBox(const Real* __restrict__ input, Real* __restrict__ output,
               const Shape& size, const Shape& lo, const Shape& hi) {
    const size_t row = size[2];
    const size_t plane = size_t(size[1]) * row;
    for (int z = lo[0]; z < hi[0]; ++z) {
        for (int y = lo[1]; y < hi[1]; ++y) {
            const size_t base = index3(z, y, 0, size);
            for (int x = lo[2]; x < hi[2]; ++x) {
                const size_t i = base + x;
                output[i] = (input[i] + input[i-1] + input[i+1]
                    + input[i-row] + input[i+row] + input[i-plane]
                    + input[i+plane]) / 7.0;
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

int run(int argc, char** argv, int worldRank, int worldSize) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > INT_MAX - 2 || ny > INT_MAX - 2 || nz > INT_MAX - 2
        || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny / nz / sizeof(Real)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        return 1;
    }

    const Shape global = {int(nz), int(ny), int(nx)};
    Shape dims;
    const int used = processGrid(worldSize, global, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < used ? 0 : MPI_UNDEFINED, worldRank, &active);
    // Excess ranks participate in MPI startup/finalization, but own no cells.
    if (active == MPI_COMM_NULL) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 3, dims.data(), periods, 0, &cart);
    MPI_Comm_free(&active);
    int rank;
    MPI_Comm_rank(cart, &rank);
    Shape coords, local, offset;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    partition(global, dims, coords, local, offset);
    Shape size = {local[0] + 2, local[1] + 2, local[2] + 2};
    const size_t cells = size_t(size[0]) * size[1] * size[2];
    std::vector<Real> grid1(cells), grid2(cells);
    for (int z = 1; z <= local[0]; ++z)
        for (int y = 1; y <= local[1]; ++y)
            for (int x = 1; x <= local[2]; ++x) {
                const size_t i = index3(z, y, x, size);
                const size_t g = index3(offset[0] + z - 1, offset[1] + y - 1,
                                        offset[2] + x - 1, global);
                // Both buffers contain the immutable global boundary values.
                grid1[i] = grid2[i] = Real(g % 19);
            }

    std::array<int, 6> neighbor;
    std::array<MPI_Datatype, 6> sendType, recvType;
    sendType.fill(MPI_DATATYPE_NULL);
    recvType.fill(MPI_DATATYPE_NULL);
    Shape lo, hi, coreLo, coreHi;
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        lo[d] = 1 + (offset[d] == 0);
        hi[d] = local[d] + 1 - (offset[d] + local[d] == global[d]);
        hi[d] = std::max(lo[d], hi[d]);
        coreLo[d] = std::max(lo[d], neighbor[2*d] == MPI_PROC_NULL ? 1 : 2);
        coreHi[d] = std::max(coreLo[d], std::min(hi[d],
            neighbor[2*d+1] == MPI_PROC_NULL ? local[d] + 1 : local[d]));
        for (int side = 0; side < 2; ++side) {
            const int face = 2*d + side;
            if (neighbor[face] == MPI_PROC_NULL) continue;
            Shape count = local, start = {1, 1, 1};
            count[d] = 1;
            start[d] = side ? local[d] : 1;
            sendType[face] = subarray(size, count, start);
            start[d] = side ? local[d] + 1 : 0;
            recvType[face] = subarray(size, count, start);
        }
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d x %d in X/Y/Z)\n", used, dims[2], dims[1], dims[0]);
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    Real* input = grid1.data();
    Real* output = grid2.data();
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int nrequests = 0;
        for (int f = 0; f < 6; ++f)
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Irecv(input, 1, recvType[f], neighbor[f], f ^ 1, cart, &requests[nrequests++]);
        for (int f = 0; f < 6; ++f)
            if (neighbor[f] != MPI_PROC_NULL)
                MPI_Isend(input, 1, sendType[f], neighbor[f], f, cart, &requests[nrequests++]);
        updateBox(input, output, size, coreLo, coreHi);
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);
        // Peel off six disjoint boxes around the already computed core.
        Shape remainingLo = lo, remainingHi = hi;
        for (int d = 0; d < 3; ++d) {
            Shape end = remainingHi;
            end[d] = std::min(remainingHi[d], coreLo[d]);
            updateBox(input, output, size, remainingLo, end);
            remainingLo[d] = end[d];
            Shape begin = remainingLo;
            begin[d] = std::max(remainingLo[d], std::min(remainingHi[d], coreHi[d]));
            updateBox(input, output, size, begin, remainingHi);
            remainingHi[d] = begin[d];
        }
        std::swap(input, output);
    }
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double updates = double(std::max(0, global[0] - 2))
            * std::max(0, global[1] - 2) * std::max(0, global[2] - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
    }

    if (printResults) {
        // Only the optional external-validation path needs a global allocation.
        // Derived types preserve global ordering without packing or int-sized
        // MPI_Gatherv displacements, including unevenly partitioned blocks.
        MPI_Datatype owned = subarray(size, local, {1, 1, 1});
        if (rank == 0) {
            std::vector<Real> result(nx * ny * nz);
            for (int r = 0; r < used; ++r) {
                Shape c, block, origin;
                MPI_Cart_coords(cart, r, 3, c.data());
                partition(global, dims, c, block, origin);
                if (r == 0) {
                    for (int z = 0; z < local[0]; ++z)
                        for (int y = 0; y < local[1]; ++y)
                            std::copy_n(input + index3(z+1, y+1, 1, size), local[2],
                                result.data() + index3(z+offset[0], y+offset[1], offset[2], global));
                } else {
                    MPI_Datatype target = subarray(global, block, origin);
                    MPI_Recv(result.data(), 1, target, r, 6, cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&target);
                }
            }
            print_results(result, "Grid");
        } else {
            MPI_Send(input, 1, owned, 0, 6, cart);
        }
        MPI_Type_free(&owned);
    }

    int valid = 1;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -localMin;
        int localValid = 1;
        for (int z = 1; z <= local[0]; ++z)
            for (int y = 1; y <= local[1]; ++y)
                for (int x = 1; x <= local[2]; ++x) {
                    const Real value = input[index3(z, y, x, size)];
                    localValid &= std::isfinite(value);
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
        Real minVal, maxVal;
        MPI_Allreduce(&localValid, &valid, 1, MPI_INT, MPI_MIN, cart);
        MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, cart);
        MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, cart);
        valid &= minVal >= -1e6 && maxVal <= 1e6;
        if (rank == 0) {
            printf("Validating result...\nValue range: [%.6f, %.6f]\n", minVal, maxVal);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    for (int f = 0; f < 6; ++f) {
        if (sendType[f] != MPI_DATATYPE_NULL) MPI_Type_free(&sendType[f]);
        if (recvType[f] != MPI_DATATYPE_NULL) MPI_Type_free(&recvType[f]);
    }
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int globalResult;
    MPI_Allreduce(&result, &globalResult, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalResult;
}
