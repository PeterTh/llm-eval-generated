#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
// MPI subarray coordinates follow C storage order (z, y, x).
using Shape = std::array<int, 3>;

Shape processGrid(const Shape& global, int ranks) {
    Shape best{1, 1, 1};
    int used = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= std::min(global[0], ranks); ++z) {
        for (int y = 1; y <= std::min(global[1], ranks / z); ++y) {
            int x = std::min(global[2], ranks / z / y);
            int count = z * y * x;
            if (count < used) continue;
            // Total internal face area, weighted for strided packing in x.
            double cost = double(z-1) * global[1] * global[2]
                        + double(y-1) * global[0] * global[2]
                        + 2.0 * (x-1) * global[0] * global[1];
            // Prefer contiguous z faces when two grids exchange equal areas.
            if (count > used || cost < bestCost || (cost == bestCost && z > best[0])) {
                best = {z, y, x};
                used = count;
                bestCost = cost;
            }
        }
    }
    return best;
}

struct Block {
    Shape extent, origin;
};

Block partition(const Shape& global, const Shape& dims, MPI_Comm cart, int rank) {
    Shape coord;
    MPI_Cart_coords(cart, rank, 3, coord.data());
    Block block;
    for (int d = 0; d < 3; ++d) {
        int width = global[d] / dims[d], extra = global[d] % dims[d];
        block.extent[d] = width + (coord[d] < extra);
        block.origin[d] = coord[d] * width + std::min(coord[d], extra);
    }
    return block;
}

size_t volume(const Shape& s) {
    return size_t(s[0]) * s[1] * s[2];
}

size_t offset(int z, int y, int x, const Shape& s) {
    return (size_t(z) * s[1] + y) * s[2] + x;
}

MPI_Datatype subarray(const Shape& layout, const Shape& extent, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, layout.data(), extent.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Face {
    int peer, side;
    MPI_Datatype sendType, recvType;
};

struct Box {
    Shape lo, hi; // Half-open local coordinates
};

void update(const Real* __restrict__ input, Real* __restrict__ output,
            const Shape& layout, const Box& box) {
    const size_t row = layout[2], plane = row * layout[1];
    for (int z = box.lo[0]; z < box.hi[0]; ++z) {
        for (int y = box.lo[1]; y < box.hi[1]; ++y) {
            size_t base = offset(z, y, 0, layout);
            for (int x = box.lo[2]; x < box.hi[2]; ++x) {
                size_t i = base + x;
                output[i] = (input[i] + input[i-1] + input[i+1] + input[i-row]
                           + input[i+row] + input[i-plane] + input[i+plane]) / 7.0;
            }
        }
    }
}

void reportGrid(const std::vector<Real>& local, const Shape& layout, const Shape& start,
                const Shape& global, const Shape& dims, int rank, MPI_Comm cart) {
    if (dims == Shape{1, 1, 1}) {
        print_results(local, "Grid");
        return;
    }
    Block own = partition(global, dims, cart, rank);
    MPI_Datatype owned = subarray(layout, own.extent, start);
    if (rank == 0) {
        std::vector<Real> result(volume(global));
        for (int z = 0; z < own.extent[0]; ++z) {
            for (int y = 0; y < own.extent[1]; ++y) {
                std::copy_n(local.data() + offset(z+start[0], y+start[1], start[2], layout),
                            own.extent[2], result.data() + offset(z, y, 0, global));
            }
        }
        int count;
        MPI_Comm_size(cart, &count);
        for (int peer = 1; peer < count; ++peer) {
            Block block = partition(global, dims, cart, peer);
            MPI_Datatype dest = subarray(global, block.extent, block.origin);
            MPI_Recv(result.data(), 1, dest, peer, 0, cart, MPI_STATUS_IGNORE);
            MPI_Type_free(&dest);
        }
        print_results(result, "Grid");
    } else {
        MPI_Send(local.data(), 1, owned, 0, 0, cart);
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

int benchmark(int argc, char** argv, int rank, int ranks) {
    int nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, printResults = false;

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
    
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }
    Shape global{nz, ny, nx}, dims = processGrid(global, ranks);
    Shape periodic{};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periodic.data(), 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", nx, ny, nz);
        printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    Block block = partition(global, dims, cart, rank);
    Shape layout, start;
    Box interior, core;
    std::array<int, 6> peers;
    for (int d = 0; d < 3; ++d) {
        MPI_Cart_shift(cart, d, 1, &peers[2*d], &peers[2*d+1]);
        bool low = peers[2*d] != MPI_PROC_NULL, high = peers[2*d+1] != MPI_PROC_NULL;
        start[d] = low;
        layout[d] = block.extent[d] + low + high;
        interior.lo[d] = start[d] + !low;
        interior.hi[d] = std::max(interior.lo[d], start[d] + block.extent[d] - !high);
        core.lo[d] = std::min(interior.hi[d], interior.lo[d] + low);
        core.hi[d] = std::max(core.lo[d], interior.hi[d] - high);
    }
    std::vector<Face> faces;
    for (int side = 0; side < 6; ++side) {
        if (peers[side] == MPI_PROC_NULL) continue;
        int d = side / 2;
        Shape extent = block.extent, send = start, recv;
        extent[d] = 1;
        if (side % 2) send[d] += block.extent[d] - 1;
        recv = send;
        recv[d] += side % 2 ? 1 : -1;
        faces.push_back({peers[side], side, subarray(layout, extent, send),
                         subarray(layout, extent, recv)});
    }
    // Partition the halo-dependent shell into disjoint rectangular regions.
    std::vector<Box> shell;
    Box remaining = interior;
    for (int d = 0; d < 3; ++d) {
        if (remaining.lo[d] < core.lo[d]) {
            Box edge = remaining;
            edge.hi[d] = core.lo[d];
            remaining.lo[d] = core.lo[d];
            shell.push_back(edge);
        }
        if (remaining.hi[d] > core.hi[d]) {
            Box edge = remaining;
            edge.lo[d] = core.hi[d];
            remaining.hi[d] = core.hi[d];
            shell.push_back(edge);
        }
    }
    std::vector<Real> input(volume(layout));
    for (int z = 0; z < block.extent[0]; ++z) {
        for (int y = 0; y < block.extent[1]; ++y) {
            size_t g = offset(z + block.origin[0], y + block.origin[1], block.origin[2], global);
            size_t i = offset(z + start[0], y + start[1], start[2], layout);
            for (int x = 0; x < block.extent[2]; ++x) input[i+x] = (g+x) % 19;
        }
    }
    // Both buffers retain the fixed physical boundaries across swaps.
    std::vector<Real> output = input;
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(cart);
    double begin = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::array<MPI_Request, 12> requests;
        int count = 0;
        for (const Face& face : faces)
            MPI_Irecv(input.data(), 1, face.recvType, face.peer, face.side ^ 1,
                      cart, &requests[count++]);
        for (const Face& face : faces)
            MPI_Isend(input.data(), 1, face.sendType, face.peer, face.side,
                      cart, &requests[count++]);
        update(input.data(), output.data(), layout, core);
        if (count) MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
        for (const Box& edge : shell) update(input.data(), output.data(), layout, edge);
        input.swap(output);
    }
    double elapsed = MPI_Wtime() - begin, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        double updates = double(std::max(nx-2, 0)) * std::max(ny-2, 0)
                       * std::max(nz-2, 0) * std::max(iterations, 0);
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }
    if (iterations < 0 && iterations % 2) std::fill(input.begin(), input.end(), 0.0);
    if (printResults) reportGrid(input, layout, start, global, dims, rank, cart);

    int valid = 1;
    if (validate) {
        double minVal = std::numeric_limits<double>::infinity(), maxVal = -minVal;
        int finite = 1;
        for (int z = 0; z < block.extent[0]; ++z) {
            for (int y = 0; y < block.extent[1]; ++y) {
                size_t i = offset(z + start[0], y + start[1], start[2], layout);
                for (int x = 0; x < block.extent[2]; ++x) {
                    Real v = input[i+x];
                    finite &= std::isfinite(v);
                    minVal = std::min(minVal, v);
                    maxVal = std::max(maxVal, v);
                }
            }
        }
        double extrema[] = {-minVal, maxVal}, bounds[2];
        MPI_Reduce(extrema, bounds, 2, MPI_DOUBLE, MPI_MAX, 0, cart);
        MPI_Reduce(&finite, &valid, 1, MPI_INT, MPI_MIN, 0, cart);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!valid) {
                printf("Validation failed: found NaN or Inf value\n");
            } else {
                printf("Value range: [%.6f, %.6f]\n", -bounds[0], bounds[1]);
                valid = bounds[0] <= 1e6 && bounds[1] <= 1e6;
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, cart);
    }
    for (Face& face : faces) {
        MPI_Type_free(&face.sendType);
        MPI_Type_free(&face.recvType);
    }
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = benchmark(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
