#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
// MPI subarrays use C storage order: z, y, x.
using Shape = std::array<int, 3>;

struct Block {
    Shape begin;
    Shape size;
};

Block partition(const Shape& global, const Shape& dims, const Shape& coords) {
    Block block;
    for (int d = 0; d < 3; ++d) {
        int width = global[d] / dims[d];
        int extra = global[d] % dims[d];
        block.begin[d] = coords[d] * width + std::min(coords[d], extra);
        block.size[d] = width + (coords[d] < extra);
    }
    return block;
}

// Minimize internal face area. Break ties in favor of contiguous xy faces.
Shape processGrid(const Shape& global, int ranks) {
    for (int count = ranks; count > 0; --count) {
        double best = std::numeric_limits<double>::infinity();
        Shape dims{};
        for (int x = 1; x <= std::min(count, global[2]); ++x) {
            if (count % x) continue;
            int rest = count / x;
            for (int y = 1; y <= std::min(rest, global[1]); ++y) {
                if (rest % y) continue;
                int z = rest / y;
                if (z > global[0]) continue;
                double area = double(global[1]) * global[2] * (z - 1)
                            + double(global[0]) * global[2] * (y - 1)
                            + double(global[0]) * global[1] * (x - 1);
                if (area < best) {
                    best = area;
                    dims = {z, y, x};
                }
            }
        }
        if (dims[0]) return dims;
    }
    return {1, 1, 1};
}

MPI_Datatype subarray(const Shape& full, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, full.data(), size.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Face {
    int peer, tag;
    MPI_Datatype send, receive;
};

// Half-open box in local coordinates.
struct Box { Shape lo, hi; };

void update(const Real* input, Real* output, const Box& box, size_t row, size_t plane) {
    if (box.lo[2] == box.hi[2]) return;
    for (int z = box.lo[0]; z < box.hi[0]; ++z) {
        for (int y = box.lo[1]; y < box.hi[1]; ++y) {
            size_t base = z * plane + y * row;
            const Real* p = input + base;
            Real* q = output + base;
            for (int x = box.lo[2]; x < box.hi[2]; ++x) {
                q[x] = (p[x] + p[x-1] + p[x+1] + (p-row)[x] + (p+row)[x]
                        + (p-plane)[x] + (p+plane)[x]) / 7.0;
            }
        }
    }
}

void report(const std::vector<Real>& grid, const Shape& full, const Shape& global,
            const Shape& dims, const Block& block, const Shape& origin, MPI_Comm cart, int rank) {
    MPI_Datatype owned = subarray(full, block.size, origin);
    if (rank == 0) {
        std::vector<Real> result(size_t(global[0]) * global[1] * global[2]);
        int ranks;
        MPI_Comm_size(cart, &ranks);
        for (int src = 0; src < ranks; ++src) {
            Shape coords;
            MPI_Cart_coords(cart, src, 3, coords.data());
            Block part = partition(global, dims, coords);
            MPI_Datatype destination = subarray(global, part.size, part.begin);
            if (src == 0) {
                MPI_Sendrecv(grid.data(), 1, owned, 0, 0, result.data(), 1, destination,
                             0, 0, MPI_COMM_SELF, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(result.data(), 1, destination, src, 0, cart, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&destination);
        }
        print_results(result, "Grid");
    } else {
        MPI_Send(grid.data(), 1, owned, 0, 0, cart);
    }
    MPI_Type_free(&owned);
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
    
    if (!nx || !ny || !nz || std::max({nx, ny, nz}) > INT_MAX - 2 ||
        nx > std::numeric_limits<size_t>::max() / sizeof(Real) / ny / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid size\n");
        return 1;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const Shape global{int(nz), int(ny), int(nx)};
    const Shape dims = processGrid(global, int(std::min<size_t>(ranks, nx * ny * nz)));
    const int periodic[3]{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periodic, 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    Shape coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    const Block block = partition(global, dims, coords);
    Shape full = block.size;
    // Unsplit dimensions need no halo. Keeping them contiguous also makes
    // whole xy planes a single transfer for a z-slab decomposition.
    Shape origin;
    for (int d = 0; d < 3; ++d) {
        origin[d] = dims[d] > 1;
        full[d] += 2 * origin[d];
    }
    const size_t row = full[2], plane = row * full[1];
    std::vector<Real> input(plane * full[0]);
    for (int z = 0; z < block.size[0]; ++z) {
        for (int y = 0; y < block.size[1]; ++y) {
            size_t globalBase = ((block.begin[0] + z) * ny + block.begin[1] + y) * nx
                                + block.begin[2];
            size_t localBase = (z + origin[0]) * plane + (y + origin[1]) * row + origin[2];
            for (int x = 0; x < block.size[2]; ++x)
                input[localBase + x] = Real((globalBase + x) % 19);
        }
    }
    // Both buffers retain the fixed physical boundary throughout the run.
    std::vector<Real> output(input);
    Box interior, core;
    std::vector<Face> faces;
    for (int d = 0; d < 3; ++d) {
        int peer[2];
        MPI_Cart_shift(cart, d, 1, &peer[0], &peer[1]);
        interior.lo[d] = origin[d] + (peer[0] == MPI_PROC_NULL);
        interior.hi[d] = std::max(interior.lo[d], origin[d] + block.size[d] - (peer[1] == MPI_PROC_NULL));
        core.lo[d] = std::min(interior.hi[d], interior.lo[d] + (peer[0] != MPI_PROC_NULL));
        core.hi[d] = std::max(core.lo[d], interior.hi[d] - (peer[1] != MPI_PROC_NULL));
        for (int side = 0; side < 2; ++side) {
            if (peer[side] == MPI_PROC_NULL) continue;
            Shape extent = block.size, send = origin, recv = origin;
            extent[d] = 1;
            send[d] = side ? block.size[d] : 1;
            recv[d] = side ? block.size[d] + 1 : 0;
            faces.push_back({peer[side], 2*d + side, subarray(full, extent, send),
                             subarray(full, extent, recv)});
        }
    }

    // Peel disjoint faces off the interior, leaving the communication-independent core.
    std::vector<Box> edges;
    Box remaining = interior;
    for (int d = 0; d < 3; ++d) {
        if (remaining.lo[d] < core.lo[d]) {
            Box edge = remaining;
            edge.hi[d] = core.lo[d];
            edges.push_back(edge);
            remaining.lo[d] = core.lo[d];
        }
        if (remaining.hi[d] > core.hi[d]) {
            Box edge = remaining;
            edge.lo[d] = core.hi[d];
            edges.push_back(edge);
            remaining.hi[d] = core.hi[d];
        }
    }

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request requests[12];
        int count = 0;
        for (const Face& f : faces)
            MPI_Irecv(input.data(), 1, f.receive, f.peer, f.tag ^ 1, cart, &requests[count++]);
        for (const Face& f : faces)
            MPI_Isend(input.data(), 1, f.send, f.peer, f.tag, cart, &requests[count++]);
        update(input.data(), output.data(), core, row, plane);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        for (const Box& edge : edges)
            update(input.data(), output.data(), edge, row, plane);
        input.swap(output);
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        double updates = double(std::max(0, global[0]-2)) * std::max(0, global[1]-2)
                         * std::max(0, global[2]-2) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates / duration / 1e6);
    }

    // The original selects its zero-filled second buffer for negative odd counts.
    if (iterations < 0 && iterations % 2) std::fill(input.begin(), input.end(), 0.0);
    if (printResults) report(input, full, global, dims, block, origin, cart, rank);
    int failed = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        Real low = std::numeric_limits<Real>::infinity(), high = -low;
        int nonfinite = 0;
        for (int z = 0; z < block.size[0]; ++z) {
            for (int y = 0; y < block.size[1]; ++y) {
                const Real* p = input.data() + (z + origin[0]) * plane + (y + origin[1]) * row + origin[2];
                for (int x = 0; x < block.size[2]; ++x) {
                    nonfinite |= !std::isfinite(p[x]);
                    low = std::min(low, p[x]);
                    high = std::max(high, p[x]);
                }
            }
        }
        Real minVal, maxVal;
        int invalid;
        MPI_Reduce(&nonfinite, &invalid, 1, MPI_INT, MPI_MAX, 0, cart);
        MPI_Reduce(&low, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
        MPI_Reduce(&high, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            if (invalid) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 1e6 || minVal < -1e6)
                    printf("Validation failed: values out of expected range\n");
            }
            failed = invalid || maxVal > 1e6 || minVal < -1e6;
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    for (Face& f : faces) {
        MPI_Type_free(&f.send);
        MPI_Type_free(&f.receive);
    }
    MPI_Comm_free(&cart);
    return failed;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
