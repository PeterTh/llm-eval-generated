#include <algorithm>
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

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

// Select compact blocks, including for anisotropic grids. If the grid has
// fewer usable blocks than ranks, unused ranks wait outside the work communicator.
int processGrid(int worldSize, const int global[3], int dims[3]) {
    for (int count = worldSize; count > 0; --count) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(global[0], count); ++x) {
            if (count % x) continue;
            const int rest = count / x;
            for (int y = 1; y <= std::min(global[1], rest); ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (z > global[2]) continue;
                // Total internal interface area (two directions per interface).
                const double cost = double(x - 1) * global[1] * global[2]
                                  + double(y - 1) * global[0] * global[2]
                                  + double(z - 1) * global[0] * global[1];
                if (cost < best) {
                    best = cost;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return count;
    }
    return 1;
}

MPI_Datatype subarray(const int sizes[3], const int subsizes[3], const int starts[3]) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                            MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Domain {
    // Coordinates are x,y,z; MPI subarray sizes are z,y,x (C storage order).
    int n[3], start[3], neighbor[6];
    size_t sx, plane;
    MPI_Datatype sendFace[6], recvFace[6], owned;

    Domain(MPI_Comm comm, const int global[3], const int dims[3]) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int d = 0; d < 3; ++d) {
            const int base = global[d] / dims[d], extra = global[d] % dims[d];
            n[d] = base + (coords[d] < extra);
            start[d] = coords[d] * base + std::min(coords[d], extra);
            MPI_Cart_shift(comm, d, 1, &neighbor[2*d], &neighbor[2*d+1]);
        }
        sx = size_t(n[0]) + 2;
        plane = sx * (size_t(n[1]) + 2);
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        int counts[3] = {n[2], n[1], n[0]}, offsets[3] = {1, 1, 1};
        owned = subarray(sizes, counts, offsets);
        for (int d = 0; d < 3; ++d) {
            int face[3] = {n[2], n[1], n[0]};
            face[2-d] = 1;
            for (int side = 0; side < 2; ++side) {
                int at[3] = {1, 1, 1};
                at[2-d] = side ? n[d] : 1;
                sendFace[2*d+side] = subarray(sizes, face, at);
                at[2-d] = side ? n[d]+1 : 0;
                recvFace[2*d+side] = subarray(sizes, face, at);
            }
        }
    }

    void release() {
        MPI_Type_free(&owned);
        for (int f = 0; f < 6; ++f) {
            MPI_Type_free(&sendFace[f]);
            MPI_Type_free(&recvFace[f]);
        }
    }
};

// Keep exactly the serial addition order. Both buffers already contain the
// immutable physical boundaries, so boundary copying is unnecessary.
void stencilIteration(const Real* input, Real* output, const Domain& d,
                      const int global[3], bool shell) {
    int lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = d.start[a] == 0 ? 2 : 1;
        hi[a] = d.n[a] - (d.start[a] + d.n[a] == global[a]);
    }
    for (int z = shell ? lo[2] : std::max(lo[2], 2);
         z <= (shell ? hi[2] : std::min(hi[2], d.n[2]-1)); ++z) {
        for (int y = shell ? lo[1] : std::max(lo[1], 2);
             y <= (shell ? hi[1] : std::min(hi[1], d.n[1]-1)); ++y) {
            const bool coreRow = z > 1 && z < d.n[2] && y > 1 && y < d.n[1];
            const int begin = shell ? lo[0] : std::max(lo[0], 2);
            const int end = shell ? hi[0] : std::min(hi[0], d.n[0]-1);
            // On core rows only the two x faces remain after the overlap work.
            for (int part = 0; part < (shell && coreRow ? 2 : 1); ++part) {
                const int first = shell && coreRow ? (part == 0 ? 1 : d.n[0]) : begin;
                const int last = shell && coreRow ? first : end;
                if (first < begin || last > end || (part == 1 && d.n[0] == 1)) continue;
                size_t i = idx3(first, y, z, d.sx, d.n[1]+2);
                for (int x = first; x <= last; ++x, ++i) {
                    output[i] = (input[i] + input[i-1] + input[i+1]
                               + input[i-d.sx] + input[i+d.sx]
                               + input[i-d.plane] + input[i+d.plane]) / 7.0;
                }
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    int global[3] = {128, 0, 0}, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        int axis = -1;
        if (strcmp(argv[i], "-x") == 0) axis = 0;
        else if (strcmp(argv[i], "-y") == 0) axis = 1;
        else if (strcmp(argv[i], "-z") == 0) axis = 2;
        if ((axis >= 0 || strcmp(argv[i], "-i") == 0) && i+1 < argc) {
            char* end;
            const long value = strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < (axis >= 0 ? 1 : 0) || value > INT_MAX-2) {
                if (rank == 0) fprintf(stderr, "Invalid numeric argument: %s\n", argv[i]);
                MPI_Finalize();
                return 1;
            }
            if (axis >= 0) global[axis] = int(value);
            else iterations = int(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    size_t padded = 1;
    for (int a = 0; a < 3; ++a) {
        if (padded > std::numeric_limits<size_t>::max() / sizeof(Real) / (size_t(global[a])+2)) {
            if (rank == 0) fprintf(stderr, "Grid is too large\n");
            MPI_Finalize();
            return 1;
        }
        padded *= size_t(global[a])+2;
    }

    int dims[3];
    const int active = processGrid(worldSize, global, dims);
    MPI_Comm work;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
    int status = 0;
    if (rank < active) {
        int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(work, 3, dims, periods, 0, &cart);
        Domain d(cart, global, dims);
        std::vector<Real> grid1(d.plane * (size_t(d.n[2])+2));
        for (int z = 1; z <= d.n[2]; ++z)
            for (int y = 1; y <= d.n[1]; ++y)
                for (int x = 1; x <= d.n[0]; ++x) {
                    const size_t index = idx3(size_t(d.start[0])+x-1,
                        size_t(d.start[1])+y-1, size_t(d.start[2])+z-1, global[0], global[1]);
                    grid1[idx3(x, y, z, d.sx, d.n[1]+2)] = Real(index % 19);
                }
        std::vector<Real> grid2 = grid1;
        if (rank == 0) {
            printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\n",
                   global[0], global[1], global[2], iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d, process grid: %d x %d x %d\n", active, dims[0], dims[1], dims[2]);
            printf("Initializing grid...\nRunning stencil computation...\n");
        }
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Request requests[12];
            int count = 0;
            for (int f = 0; f < 6; ++f)
                if (d.neighbor[f] != MPI_PROC_NULL)
                    MPI_Irecv(grid1.data(), 1, d.recvFace[f], d.neighbor[f], f^1,
                              cart, &requests[count++]);
            for (int f = 0; f < 6; ++f)
                if (d.neighbor[f] != MPI_PROC_NULL)
                    MPI_Isend(grid1.data(), 1, d.sendFace[f], d.neighbor[f], f,
                              cart, &requests[count++]);
            stencilIteration(grid1.data(), grid2.data(), d, global, false);
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            stencilIteration(grid1.data(), grid2.data(), d, global, true);
            grid1.swap(grid2);
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            const double updates = double(std::max(global[0]-2, 0))
                * std::max(global[1]-2, 0) * std::max(global[2]-2, 0) * iterations;
            printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0.0);
        }

        if (printResults) {
            // Gather only on request, preserving serial ordering for Kahan sum
            // and the order-dependent hash. Derived types avoid packing copies.
            if (rank == 0) {
                const int sizes[3] = {global[2], global[1], global[0]};
                std::vector<Real> full(size_t(global[0]) * global[1] * global[2]);
                for (int r = 0; r < active; ++r) {
                    int coords[3], counts[3], offsets[3];
                    MPI_Cart_coords(cart, r, 3, coords);
                    for (int a = 0; a < 3; ++a) {
                        const int base = global[a] / dims[a], extra = global[a] % dims[a];
                        counts[2-a] = base + (coords[a] < extra);
                        offsets[2-a] = coords[a] * base + std::min(coords[a], extra);
                    }
                    MPI_Datatype target = subarray(sizes, counts, offsets);
                    if (r == 0)
                        MPI_Sendrecv(grid1.data(), 1, d.owned, 0, 6, full.data(), 1,
                                     target, 0, 6, cart, MPI_STATUS_IGNORE);
                    else
                        MPI_Recv(full.data(), 1, target, r, 6, cart, MPI_STATUS_IGNORE);
                    MPI_Type_free(&target);
                }
                print_results(full, "Grid");
            } else {
                MPI_Send(grid1.data(), 1, d.owned, 0, 6, cart);
            }
        }
        if (validate) {
            double localMin = std::numeric_limits<double>::infinity();
            double localMax = -localMin;
            int invalid = 0;
            for (int z = 1; z <= d.n[2]; ++z)
                for (int y = 1; y <= d.n[1]; ++y)
                    for (int x = 1; x <= d.n[0]; ++x) {
                        const Real value = grid1[idx3(x, y, z, d.sx, d.n[1]+2)];
                        invalid |= !std::isfinite(value);
                        localMin = std::min(localMin, value);
                        localMax = std::max(localMax, value);
                    }
            double minVal, maxVal;
            MPI_Allreduce(&invalid, &status, 1, MPI_INT, MPI_MAX, cart);
            MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
            MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (rank == 0) {
                printf("Validating result...\nValue range: [%.6f, %.6f]\n", minVal, maxVal);
                status |= minVal < -1e6 || maxVal > 1e6;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        d.release();
        MPI_Comm_free(&cart);
        MPI_Comm_free(&work);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
