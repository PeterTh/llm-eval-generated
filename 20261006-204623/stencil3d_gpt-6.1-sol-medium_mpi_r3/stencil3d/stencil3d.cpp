#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// Choose a process grid that minimizes halo traffic, accounting for the shape
// of the problem. Never give a rank an empty block. Excess ranks remain idle.
int chooseProcessGrid(const int global[3], int ranks, int dims[3]) {
    for (int count = ranks; count >= 1; --count) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(count, global[0]); ++x) {
            if (count % x) continue;
            const int rest = count / x;
            for (int y = 1; y <= std::min(rest, global[1]); ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (z > global[2]) continue;
                // Total area of internal faces (each exchanged in both directions).
                const double traffic = double(x - 1) * global[1] * global[2]
                                     + double(y - 1) * global[0] * global[2]
                                     + double(z - 1) * global[0] * global[1];
                if (traffic < best) {
                    best = traffic;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return count;
    }
    return 1;
}

void blockShape(const int global[3], const int dims[3], const int coords[3],
                int local[3], int offset[3]) {
    for (int d = 0; d < 3; ++d) {
        const int base = global[d] / dims[d];
        const int extra = global[d] % dims[d];
        local[d] = base + (coords[d] < extra);
        offset[d] = coords[d] * base + std::min(coords[d], extra);
    }
}

MPI_Datatype subarray(const int shape[3], const int extent[3], const int start[3]) {
    // MPI subarrays use slowest-to-fastest dimensions in C order.
    const int sizes[3] = {shape[2], shape[1], shape[0]};
    const int subsizes[3] = {extent[2], extent[1], extent[0]};
    const int starts[3] = {start[2], start[1], start[0]};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

// Contiguous x loops allow SIMD without changing the order of the seven additions.
void updateBox(const Real* __restrict__ input, Real* __restrict__ output,
               size_t sx, size_t sy, int x0, int x1, int y0, int y1, int z0, int z1) {
    const size_t plane = sx * sy;
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            const size_t row = idx3(0, y, z, sx, sy);
            for (int x = x0; x < x1; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1]
                           + input[i - sx] + input[i + sx]
                           + input[i - plane] + input[i + plane]) / 7.0;
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

// MPI must be finalized on every normal exit, including help and parse errors.
struct MPISession {
    MPISession(int& argc, char**& argv) { MPI_Init(&argc, &argv); }
    ~MPISession() { MPI_Finalize(); }
};

int run(int argc, char** argv, int rank, int worldSize) {
    int global[3] = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        int dimension = -1;
        if (!strcmp(argv[i], "-x")) dimension = 0;
        if (!strcmp(argv[i], "-y")) dimension = 1;
        if (!strcmp(argv[i], "-z")) dimension = 2;
        if ((dimension >= 0 || !strcmp(argv[i], "-i")) && i + 1 < argc) {
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX - 2 ||
                (dimension == 0 && value == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                return 1;
            }
            if (dimension >= 0) global[dimension] = static_cast<int>(value);
            else iterations = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "-r")) {
            printResults = true;
        } else if (!strcmp(argv[i], "-h")) {
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
    size_t gridSize = 1;
    for (int d = 0; d < 3; ++d) {
        if (gridSize > std::numeric_limits<size_t>::max() / sizeof(Real) / global[d]) {
            if (rank == 0) fprintf(stderr, "Grid size is too large\n");
            return 1;
        }
        gridSize *= global[d];
    }
    int dims[3];
    const int activeRanks = chooseProcessGrid(global, worldSize, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) return 0;
    int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(active, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&active);
    int coords[3], local[3], offset[3], shape[3], neighbor[3][2];
    MPI_Cart_coords(cart, rank, 3, coords);
    blockShape(global, dims, coords, local, offset);
    size_t localSize = 1;
    for (int d = 0; d < 3; ++d) {
        shape[d] = local[d] + 2;
        if (localSize > std::numeric_limits<size_t>::max() / sizeof(Real) / shape[d]) {
            fprintf(stderr, "Local grid size is too large\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        localSize *= shape[d];
        MPI_Cart_shift(cart, d, 1, &neighbor[d][0], &neighbor[d][1]);
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d; process grid: %d x %d x %d\n",
               activeRanks, dims[0], dims[1], dims[2]);
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(localSize), grid2(localSize);
    // Initialize both buffers so physical boundaries remain fixed without a
    // boundary-copy pass. Initialization is based on the original global index.
    for (int z = 1; z <= local[2]; ++z)
        for (int y = 1; y <= local[1]; ++y)
            for (int x = 1; x <= local[0]; ++x) {
                const size_t globalIndex = idx3(offset[0] + x - 1, offset[1] + y - 1,
                                               offset[2] + z - 1, global[0], global[1]);
                const size_t i = idx3(x, y, z, shape[0], shape[1]);
                grid1[i] = grid2[i] = Real(globalIndex % 19);
            }

    MPI_Datatype sendFace[3][2], recvFace[3][2];
    for (int d = 0; d < 3; ++d) {
        int extent[3] = {local[0], local[1], local[2]};
        extent[d] = 1;
        for (int side = 0; side < 2; ++side) {
            sendFace[d][side] = recvFace[d][side] = MPI_DATATYPE_NULL;
            if (neighbor[d][side] == MPI_PROC_NULL) continue;
            int start[3] = {1, 1, 1};
            start[d] = side ? local[d] : 1;
            sendFace[d][side] = subarray(shape, extent, start);
            start[d] = side ? local[d] + 1 : 0;
            recvFace[d][side] = subarray(shape, extent, start);
        }
    }
    // Both buffer addresses and the topology are fixed. Persistent requests
    // amortize communication setup over all iterations.
    MPI_Request requests[2][12];
    int nrequests = 0;
    for (int buffer = 0; buffer < 2; ++buffer) {
        Real* input = (buffer == 0 ? grid1 : grid2).data();
        nrequests = 0;
        for (int d = 0; d < 3; ++d)
            for (int side = 0; side < 2; ++side)
                if (neighbor[d][side] != MPI_PROC_NULL)
                    MPI_Recv_init(input, 1, recvFace[d][side], neighbor[d][side],
                                  2 * d + (1 - side), cart, &requests[buffer][nrequests++]);
        for (int d = 0; d < 3; ++d)
            for (int side = 0; side < 2; ++side)
                if (neighbor[d][side] != MPI_PROC_NULL)
                    MPI_Send_init(input, 1, sendFace[d][side], neighbor[d][side],
                                  2 * d + side, cart, &requests[buffer][nrequests++]);
    }
    // [lo, hi) excludes physical boundaries; [innerLo, innerHi) needs no halo.
    int lo[3], hi[3], innerLo[3], innerHi[3];
    for (int d = 0; d < 3; ++d) {
        lo[d] = offset[d] == 0 ? 2 : 1;
        hi[d] = local[d] + 1 - (offset[d] + local[d] == global[d]);
        lo[d] = std::min(lo[d], hi[d]);
        innerLo[d] = std::min(hi[d], std::max(lo[d], 2));
        innerHi[d] = std::max(innerLo[d], std::min(hi[d], local[d]));
    }
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    const double startTime = MPI_Wtime();
    // Degenerate grids have no interior points, hence no communication is needed.
    const bool hasInterior = global[0] > 2 && global[1] > 2 && global[2] > 2;
    for (int iter = 0; iter < iterations && hasInterior; ++iter) {
        Real* input = (iter % 2 == 0 ? grid1 : grid2).data();
        Real* output = (iter % 2 == 0 ? grid2 : grid1).data();
        MPI_Startall(nrequests, requests[iter % 2]);
        updateBox(input, output, shape[0], shape[1], innerLo[0], innerHi[0],
                  innerLo[1], innerHi[1], innerLo[2], innerHi[2]);
        MPI_Waitall(nrequests, requests[iter % 2], MPI_STATUSES_IGNORE);
        // Six disjoint boxes cover the remaining cells, including blocks only
        // one cell thick. No edge/corner halos are needed for a 7-point stencil.
        updateBox(input, output, shape[0], shape[1], lo[0], hi[0], lo[1], hi[1], lo[2], innerLo[2]);
        updateBox(input, output, shape[0], shape[1], lo[0], hi[0], lo[1], hi[1], innerHi[2], hi[2]);
        updateBox(input, output, shape[0], shape[1], lo[0], hi[0], lo[1], innerLo[1], innerLo[2], innerHi[2]);
        updateBox(input, output, shape[0], shape[1], lo[0], hi[0], innerHi[1], hi[1], innerLo[2], innerHi[2]);
        updateBox(input, output, shape[0], shape[1], lo[0], innerLo[0], innerLo[1], innerHi[1], innerLo[2], innerHi[2]);
        updateBox(input, output, shape[0], shape[1], innerHi[0], hi[0], innerLo[1], innerHi[1], innerLo[2], innerHi[2]);
    }
    const double elapsed = MPI_Wtime() - startTime;
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        const double updates = double(std::max(0, global[0] - 2))
                             * std::max(0, global[1] - 2) * std::max(0, global[2] - 2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0.0);
    }
    const std::vector<Real>& finalGrid = iterations % 2 == 0 ? grid1 : grid2;
    if (printResults) {
        // Gathering is outside the timed region and only requested with -r.
        // Derived types place blocks directly in their original global order,
        // preserving the original Kahan sum, samples, and bytewise hash.
        const int ownedStart[3] = {1, 1, 1};
        MPI_Datatype owned = subarray(shape, local, ownedStart);
        if (rank == 0) {
            std::vector<Real> gathered(gridSize);
            for (int z = 1; z <= local[2]; ++z)
                for (int y = 1; y <= local[1]; ++y) {
                    const size_t source = idx3(1, y, z, shape[0], shape[1]);
                    const size_t dest = idx3(offset[0], offset[1] + y - 1,
                                            offset[2] + z - 1, global[0], global[1]);
                    std::copy_n(finalGrid.data() + source, local[0], gathered.data() + dest);
                }
            for (int r = 1; r < activeRanks; ++r) {
                int peerCoords[3], peerLocal[3], peerOffset[3];
                MPI_Cart_coords(cart, r, 3, peerCoords);
                blockShape(global, dims, peerCoords, peerLocal, peerOffset);
                MPI_Datatype target = subarray(global, peerLocal, peerOffset);
                MPI_Recv(gathered.data(), 1, target, r, 6, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
            print_results(gathered, "Grid");
        } else {
            MPI_Send(finalGrid.data(), 1, owned, 0, 6, cart);
        }
        MPI_Type_free(&owned);
    }
    int status = 0;
    if (validate) {
        Real localMin = std::numeric_limits<Real>::infinity();
        Real localMax = -std::numeric_limits<Real>::infinity();
        int localBad = 0, bad = 0;
        for (int z = 1; z <= local[2]; ++z)
            for (int y = 1; y <= local[1]; ++y)
                for (int x = 1; x <= local[0]; ++x) {
                    const Real value = finalGrid[idx3(x, y, z, shape[0], shape[1])];
                    localBad |= !std::isfinite(value);
                    localMin = std::min(localMin, value);
                    localMax = std::max(localMax, value);
                }
        Real minValue, maxValue;
        MPI_Reduce(&localMin, &minValue, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&localMax, &maxValue, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        MPI_Reduce(&localBad, &bad, 1, MPI_INT, MPI_MAX, 0, cart);
        if (rank == 0) {
            printf("Validating result...\n");
            if (bad) {
                printf("Validation failed: found NaN or Inf value\n");
                status = 1;
            } else {
                printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
                if (maxValue > 1e6 || minValue < -1e6) {
                    printf("Validation failed: values out of expected range\n");
                    status = 1;
                }
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, cart);
    }
    for (int buffer = 0; buffer < 2; ++buffer)
        for (int r = 0; r < nrequests; ++r)
            MPI_Request_free(&requests[buffer][r]);
    for (int d = 0; d < 3; ++d)
        for (int side = 0; side < 2; ++side)
            if (sendFace[d][side] != MPI_DATATYPE_NULL) {
                MPI_Type_free(&sendFace[d][side]);
                MPI_Type_free(&recvFace[d][side]);
            }
    MPI_Comm_free(&cart);
    return status;
}

int main(int argc, char** argv) {
    MPISession mpi(argc, argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    try {
        return run(argc, argv, rank, worldSize);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
