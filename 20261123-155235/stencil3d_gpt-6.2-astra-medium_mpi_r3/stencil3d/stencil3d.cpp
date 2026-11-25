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
// MPI subarrays use C order, so dimensions are listed as z, y, x.
using Shape = std::array<int, 3>;

struct Box {
    Shape begin, end;  // half-open
};

// Minimize the total number of shared face cells. On ties, prefer
// contiguous z faces, then y faces, over strided x faces.
Shape processGrid(const Shape& global, int ranks) {
    for (int count = ranks; count > 0; --count) {
        Shape best{};
        double bestArea = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(count, global[2]); ++x) {
            if (count % x != 0) continue;
            const int yz = count / x;
            for (int y = 1; y <= std::min(yz, global[1]); ++y) {
                if (yz % y != 0) continue;
                int z = yz / y;
                if (z > global[0]) continue;
                double area = double(x-1)*global[0]*global[1]
                            + double(y-1)*global[0]*global[2]
                            + double(z-1)*global[1]*global[2];
                if (area < bestArea) {
                    bestArea = area;
                    best = {z, y, x};
                }
            }
        }
        if (best[0]) return best;
    }
    return {1, 1, 1};
}

struct Block {
    Shape start{}, size{}, offset{}, storage{};
    Box interior{}, core{};

    Block(const Shape& global, const Shape& dims, const Shape& coords) {
        for (int d = 0; d < 3; ++d) {
            const int base = global[d] / dims[d];
            const int extra = global[d] % dims[d];
            size[d] = base + (coords[d] < extra);
            start[d] = coords[d] * base + std::min(coords[d], extra);
            bool lower = coords[d] > 0, upper = coords[d]+1 < dims[d];
            offset[d] = lower;
            storage[d] = size[d] + lower + upper;
            interior.begin[d] = offset[d] + !lower;
            interior.end[d] = std::max(interior.begin[d], offset[d] + size[d] - !upper);
            // Exclude cells that depend on an incoming halo.
            core.begin[d] = std::min(interior.end[d], interior.begin[d] + lower);
            core.end[d] = std::max(core.begin[d], interior.end[d] - upper);
        }
    }
};

size_t volume(const Shape& s) {
    return size_t(s[0]) * s[1] * s[2];
}

size_t index3(const Shape& s, int z, int y, int x) {
    return (size_t(z) * s[1] + y) * s[2] + x;
}

MPI_Datatype subarray(const Shape& shape, const Shape& size, const Shape& start) {
    MPI_Datatype type;
    MPI_Type_create_subarray(3, shape.data(), size.data(), start.data(),
                             MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct Face {
    int peer, direction;
    size_t sendOffset, recvOffset;
    MPI_Datatype type;
};

std::vector<Face> makeFaces(MPI_Comm comm, const Block& block) {
    std::vector<Face> faces;
    for (int d = 0; d < 3; ++d) {
        int peers[2];
        MPI_Cart_shift(comm, d, 1, &peers[0], &peers[1]);
        for (int side = 0; side < 2; ++side) {
            if (peers[side] == MPI_PROC_NULL) continue;
            Shape size = block.size, send = block.offset, recv;
            size[d] = 1;
            if (side) send[d] += block.size[d] - 1;
            recv = send;
            recv[d] += side ? 1 : -1;
            faces.push_back({peers[side], 2*d + side,
                index3(block.storage, send[0], send[1], send[2]),
                index3(block.storage, recv[0], recv[1], recv[2]),
                subarray(block.storage, size, {0, 0, 0})});
        }
    }
    return faces;
}

void update(const Real* input, Real* output, const Shape& storage, const Box& box) {
    const size_t row = storage[2], plane = row * storage[1];
    for (int z = box.begin[0]; z < box.end[0]; ++z) {
        for (int y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t base = index3(storage, z, y, 0);
            for (int x = box.begin[2]; x < box.end[2]; ++x) {
                const size_t i = base + x;
                output[i] = (input[i] + input[i-1] + input[i+1] + input[i-row]
                            + input[i+row] + input[i-plane] + input[i+plane]) / 7.0;
            }
        }
    }
}

void iterate(MPI_Comm comm, const Block& block, const std::vector<Face>& faces,
             std::vector<Real>& input, std::vector<Real>& output) {
    MPI_Request requests[12];
    int n = 0;
    for (const auto& f : faces)
        MPI_Irecv(input.data()+f.recvOffset, 1, f.type, f.peer, f.direction ^ 1,
                  comm, &requests[n++]);
    for (const auto& f : faces)
        MPI_Isend(input.data()+f.sendOffset, 1, f.type, f.peer, f.direction,
                  comm, &requests[n++]);

    update(input.data(), output.data(), block.storage, block.core);
    MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
    // Peel disjoint faces from the interior until only the computed core remains.
    Box rest = block.interior;
    for (int d = 0; d < 3; ++d) {
        if (rest.begin[d] < block.core.begin[d]) {
            Box face = rest;
            face.end[d] = block.core.begin[d];
            rest.begin[d] = face.end[d];
            update(input.data(), output.data(), block.storage, face);
        }
        if (rest.end[d] > block.core.end[d]) {
            Box face = rest;
            face.begin[d] = block.core.end[d];
            rest.end[d] = face.begin[d];
            update(input.data(), output.data(), block.storage, face);
        }
    }
    input.swap(output);
}

void report(MPI_Comm comm, int rank, const Shape& global, const Shape& dims,
            const Block& block, const std::vector<Real>& grid) {
    MPI_Datatype owned = subarray(block.storage, block.size, block.offset);
    if (rank == 0) {
        std::vector<Real> result(volume(global));
        int ranks;
        MPI_Comm_size(comm, &ranks);
        for (int r = 0; r < ranks; ++r) {
            Shape coords;
            MPI_Cart_coords(comm, r, 3, coords.data());
            Block source(global, dims, coords);
            MPI_Datatype dest = subarray(global, source.size, source.start);
            if (r == 0)
                MPI_Sendrecv(grid.data(), 1, owned, 0, 0, result.data(), 1, dest,
                             0, 0, comm, MPI_STATUS_IGNORE);
            else
                MPI_Recv(result.data(), 1, dest, r, 0, comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&dest);
        }
        print_results(result, "Grid");
    } else {
        MPI_Send(grid.data(), 1, owned, 0, 0, comm);
    }
    MPI_Type_free(&owned);
}

bool validateResult(MPI_Comm comm, int rank, const Block& b, const std::vector<Real>& grid) {
    // One maximum reduction yields both extrema and the non-finite flag.
    Real stats[] = {-std::numeric_limits<Real>::infinity(),
                    -std::numeric_limits<Real>::infinity(), 0};
    for (int z = 0; z < b.size[0]; ++z)
        for (int y = 0; y < b.size[1]; ++y) {
            size_t i = index3(b.storage, z+b.offset[0], y+b.offset[1], b.offset[2]);
            for (int x = 0; x < b.size[2]; ++x) {
                Real v = grid[i+x];
                stats[0] = std::max(stats[0], -v);
                stats[1] = std::max(stats[1], v);
                if (!std::isfinite(v)) stats[2] = 1;
            }
        }
    MPI_Allreduce(MPI_IN_PLACE, stats, 3, MPI_DOUBLE, MPI_MAX, comm);
    bool valid = !stats[2] && stats[0] <= 1e6 && stats[1] <= 1e6;
    if (rank == 0) {
        if (stats[2]) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Value range: [%.6f, %.6f]\n", -stats[0], stats[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
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
    int nx = 128, ny = 0, nz = 0;
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
    
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }
    const Shape global{nz, ny, nx};
    const Shape dims = processGrid(global, ranks);
    // Cart_create excludes surplus ranks when no nonempty block fits them.
    MPI_Comm comm;
    const int periods[3]{};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods, 0, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    Shape coords;
    MPI_Cart_coords(comm, rank, 3, coords.data());
    Block block(global, dims, coords);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid(volume(block.storage));
    for (int z = 0; z < block.size[0]; ++z)
        for (int y = 0; y < block.size[1]; ++y) {
            size_t i = index3(block.storage, z+block.offset[0], y+block.offset[1], block.offset[2]);
            size_t g = index3(global, z+block.start[0], y+block.start[1], block.start[2]);
            for (int x = 0; x < block.size[2]; ++x)
                grid[i+x] = (g+x) % 19;
        }
    // Both buffers retain their initial physical boundaries across swaps.
    std::vector<Real> next = grid;
    auto faces = makeFaces(comm, block);
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    if (nx > 2 && ny > 2 && nz > 2)
        for (int iter = 0; iter < iterations; ++iter)
            iterate(comm, block, faces, grid, next);
    double elapsed = MPI_Wtime() - start, longest = 0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        double updates = double(std::max(0, nx-2)) * std::max(0, ny-2)
                       * std::max(0, nz-2) * std::max(0, iterations);
        printf("Computation time: %ld ms\n", static_cast<long>(longest * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", longest > 0 ? updates / longest / 1e6 : 0);
    }
    if (printResults) report(comm, rank, global, dims, block, grid);
    if (rank == 0 && validate) printf("Validating result...\n");
    bool valid = !validate || validateResult(comm, rank, block, grid);
    for (auto& f : faces) MPI_Type_free(&f.type);
    MPI_Comm_free(&comm);
    return valid ? 0 : 1;
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
