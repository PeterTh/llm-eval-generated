#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

struct Range {
    size_t begin, end;
    size_t size() const { return end - begin; }
};

Range partition(size_t n, int parts, int coord) {
    size_t width = n / parts, extra = n % parts;
    size_t begin = coord * width + std::min<size_t>(coord, extra);
    return {begin, begin + width + (size_t(coord) < extra)};
}

struct Layout {
    int py = 1, pz = 1;
};

// Use as many ranks as fit, then minimize face traffic. Favor contiguous
// Z faces when partitions have comparable communication costs.
Layout chooseLayout(size_t ny, size_t nz, int ranks) {
    Layout best;
    double bestCost = std::numeric_limits<double>::infinity();
    const int maxY = std::min<size_t>(ny, ranks);
    for (int py = 1; py <= maxY; ++py) {
        int pz = std::min<size_t>(nz, ranks / py);
        int used = py * pz;
        double cost = 1.25 * (py - 1) * nz + double(pz - 1) * ny;
        if (used > best.py * best.pz || (used == best.py * best.pz && cost < bestCost)) {
            best = {py, pz};
            bestCost = cost;
        }
    }
    return best;
}

struct Block {
    Range y, z;
    size_t nx, pitch, y0, z0;
    int lowY, highY, lowZ, highZ;

    Block(size_t x, size_t ny, size_t nz, Layout layout, int rank)
        : y(partition(ny, layout.py, rank % layout.py)),
          z(partition(nz, layout.pz, rank / layout.py)), nx(x),
          pitch(nx * (y.size() + (y.begin != 0) + (y.end != ny))),
          y0(y.begin != 0), z0(z.begin != 0),
          lowY(y.begin ? rank - 1 : MPI_PROC_NULL),
          highY(y.end < ny ? rank + 1 : MPI_PROC_NULL),
          lowZ(z.begin ? rank - layout.py : MPI_PROC_NULL),
          highZ(z.end < nz ? rank + layout.py : MPI_PROC_NULL) {}

    size_t row(size_t iz, size_t iy) const { return (iz + z0) * pitch + (iy + y0) * nx; }
};

void update(const Real* input, Real* output, const Block& b, Range zs, Range ys) {
    for (size_t z = zs.begin; z < zs.end; ++z) {
        for (size_t y = ys.begin; y < ys.end; ++y) {
            const size_t row = b.row(z, y);
            for (size_t i = row + 1; i + 1 < row + b.nx; ++i) {
                output[i] = (input[i] + input[i-1] + input[i+1] + input[i-b.nx]
                           + input[i+b.nx] + input[i-b.pitch] + input[i+b.pitch]) / 7.0;
            }
        }
    }
}

struct Face {
    int peer;
    size_t send, recv;
    MPI_Datatype type;
};

// Reassemble only for ordered external diagnostics. Each transfer covers
// one contiguous strip, even when a complete grid exceeds MPI's int count.
void report(const std::vector<Real>& grid, size_t nx, size_t ny, size_t nz,
            Layout layout, int rank) {
    if (layout.py * layout.pz == 1) {
        print_results(grid, "Grid");
        return;
    }
    std::vector<Real> full(rank == 0 ? nx * ny * nz : 0);
    int end = rank == 0 ? layout.py * layout.pz : rank + 1;
    for (int src = rank; src < end; ++src) {
        Block b(nx, ny, nz, layout, src);
        size_t width = nx * b.y.size();
        for (size_t z = 0; z < b.z.size(); ++z) {
            size_t dest = ((b.z.begin + z) * ny + b.y.begin) * nx;
            for (size_t offset = 0; offset < width;) {
                int count = std::min<size_t>(width - offset, std::numeric_limits<int>::max());
                if (src == 0)
                    std::copy_n(grid.data() + b.row(z, 0) + offset, count, full.data() + dest + offset);
                else if (rank == 0)
                    MPI_Recv(full.data() + dest + offset, count, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                else
                    MPI_Send(grid.data() + b.row(z, 0) + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                offset += count;
            }
        }
    }
    if (rank == 0) print_results(full, "Grid");
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    Layout layout = nx && ny && nz ? chooseLayout(ny, nz, ranks) : Layout{};
    bool active = rank < layout.py * layout.pz;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (!active) {
        MPI_Finalize();
        return 0;
    }
    Block b(nx, ny, nz, layout, rank);
    size_t height = b.y.size(), depth = b.z.size();
    // The initial formula also defines the first halos locally.
    size_t rows = height + b.y0 + (b.highY != MPI_PROC_NULL);
    size_t planes = depth + b.z0 + (b.highZ != MPI_PROC_NULL);
    std::vector<Real> grid(b.pitch * planes);
    for (size_t z = 0; z < planes; ++z) {
        for (size_t y = 0; y < rows; ++y) {
            size_t global = ((b.z.begin - b.z0 + z) * ny + b.y.begin - b.y0 + y) * nx;
            size_t local = z * b.pitch + y * nx;
            for (size_t x = 0; x < nx; ++x) grid[local + x] = (global + x) % 19;
        }
    }
    auto next = grid;  // Both buffers keep the fixed physical boundaries.

    MPI_Datatype yFace = MPI_DATATYPE_NULL, zFace = MPI_DATATYPE_NULL;
    std::vector<Face> faces;
    if (layout.py > 1) {
        MPI_Type_create_hvector(depth, nx, b.pitch * sizeof(Real), MPI_DOUBLE, &yFace);
        MPI_Type_commit(&yFace);
        if (b.lowY != MPI_PROC_NULL) faces.push_back({b.lowY, b.row(0, 0), b.row(0, 0) - nx, yFace});
        if (b.highY != MPI_PROC_NULL) faces.push_back({b.highY, b.row(0, height-1), b.row(0, height), yFace});
    }
    if (layout.pz > 1) {
        MPI_Type_vector(height, nx, nx, MPI_DOUBLE, &zFace);
        MPI_Type_commit(&zFace);
        if (b.lowZ != MPI_PROC_NULL) faces.push_back({b.lowZ, b.row(0, 0), b.row(0, 0) - b.pitch, zFace});
        if (b.highZ != MPI_PROC_NULL) faces.push_back({b.highZ, b.row(depth-1, 0), b.row(depth, 0), zFace});
    }

    Range ys{size_t(b.lowY == MPI_PROC_NULL && height > 0), height - (b.highY == MPI_PROC_NULL && height > 0)};
    Range zs{size_t(b.lowZ == MPI_PROC_NULL && depth > 0), depth - (b.highZ == MPI_PROC_NULL && depth > 0)};
    // Rows strictly inside the owned block need no remote values.
    Range coreY{1, height > 2 ? height - 1 : 1};
    Range coreZ{1, depth > 2 ? depth - 1 : 1};
    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter == 0 || faces.empty()) {
            update(grid.data(), next.data(), b, zs, ys);
        } else {
            std::array<MPI_Request, 8> requests;
            int count = 0;
            for (const Face& f : faces)
                MPI_Irecv(grid.data() + f.recv, 1, f.type, f.peer, 0, comm, &requests[count++]);
            for (const Face& f : faces)
                MPI_Isend(grid.data() + f.send, 1, f.type, f.peer, 0, comm, &requests[count++]);
            update(grid.data(), next.data(), b, coreZ, coreY);
            MPI_Waitall(count, requests.data(), MPI_STATUSES_IGNORE);
            for (size_t z = zs.begin; z < zs.end; ++z) {
                Range plane{z, z+1};
                if (z < coreZ.begin || z >= coreZ.end) {
                    update(grid.data(), next.data(), b, plane, ys);
                } else {
                    update(grid.data(), next.data(), b, plane, {ys.begin, std::min(coreY.begin, ys.end)});
                    update(grid.data(), next.data(), b, plane, {std::max(coreY.end, ys.begin), ys.end});
                }
            }
        }
        grid.swap(next);
    }
    double localTime = MPI_Wtime() - start, elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        double updates = nx > 2 && ny > 2 && nz > 2 ? double(nx-2) * (ny-2) * (nz-2) * iterations : 0;
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0);
    }
    // The original selects its zero-initialized second buffer for negative odd counts.
    if (iterations < 0 && iterations % 2 != 0) std::fill(grid.begin(), grid.end(), 0.0);
    if (printResults) report(grid, nx, ny, nz, layout, rank);

    bool valid = true;
    if (validate) {
        // A single maximum reduction gives the lower bound, upper bound and failure flag.
        double local[3] = {-INFINITY, -INFINITY, 0}, global[3];
        for (size_t z = 0; z < depth; ++z) {
            for (size_t y = 0; y < height; ++y) {
                size_t row = b.row(z, y);
                for (size_t x = 0; x < nx; ++x) {
                    Real v = grid[row + x];
                    local[0] = std::max(local[0], -v);
                    local[1] = std::max(local[1], v);
                    if (!std::isfinite(v)) local[2] = 1;
                }
            }
        }
        MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX, comm);
        valid = !global[2] && global[0] <= 1e6 && global[1] <= 1e6;
        if (rank == 0) {
            printf("Validating result...\n");
            if (global[2]) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Value range: [%.6f, %.6f]\n", -global[0], global[1]);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (yFace != MPI_DATATYPE_NULL) MPI_Type_free(&yFace);
    if (zFace != MPI_DATATYPE_NULL) MPI_Type_free(&zFace);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
