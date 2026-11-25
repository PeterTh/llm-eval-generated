#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

struct Layout {
    int py, pz;
};

// Prefer contiguous Z faces and choose a near-minimal communication surface.
Layout chooseLayout(int ny, int nz, int ranks) {
    int maxY = std::max(1, ny - 2);
    int maxZ = std::max(1, nz - 2);
    Layout best{1, 1};
    double bestCost = std::numeric_limits<double>::infinity();
    for (int py = 1; py <= std::min(ranks, maxY); ++py) {
        int pz = std::min(ranks / py, maxZ);
        double cost = 1.25 * (py - 1) * nz + double(pz - 1) * ny;
        if (py * pz > best.py * best.pz ||
            (py * pz == best.py * best.pz && cost < bestCost)) {
            best = {py, pz};
            bestCost = cost;
        }
    }
    return best;
}

struct Axis {
    int start, count;
    int lowerHalo, upperHalo;
    int extent() const { return lowerHalo + count + upperHalo; }
};

Axis partition(int n, int parts, int coord) {
    if (parts == 1) return {0, n, 0, 0};
    // Balance updateable points. The end ranks also own the fixed boundaries.
    int base = (n - 2) / parts, extra = (n - 2) % parts;
    int begin = 1 + coord * base + std::min(coord, extra);
    int end = begin + base + (coord < extra);
    if (coord == 0) begin = 0;
    if (coord == parts - 1) end = n;
    return {begin, end - begin, coord > 0, coord + 1 < parts};
}

// Describe a rectangle of full X rows repeated at a byte stride in Z.
MPI_Datatype blockType(int nx, int rows, int planes, size_t planeStride) {
    MPI_Datatype face, block;
    MPI_Type_vector(rows, nx, nx, MPI_DOUBLE, &face);
    MPI_Type_create_hvector(planes, 1, MPI_Aint(planeStride * sizeof(Real)), face, &block);
    MPI_Type_commit(&block);
    MPI_Type_free(&face);
    return block;
}

void update(const Real* __restrict__ in, Real* __restrict__ out, size_t nx,
            size_t plane, int y0, int y1, int z0, int z1) {
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            size_t row = z * plane + y * nx;
            for (size_t i = row + 1; i < row + nx - 1; ++i) {
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-nx] + in[i+nx]
                          + in[i-plane] + in[i+plane]) / 7.0;
            }
        }
    }
}

void iterate(std::vector<Real>& in, std::vector<Real>& out, int nx, Axis y, Axis z,
             int rank, Layout layout, MPI_Datatype yFace, MPI_Datatype zFace) {
    const size_t plane = size_t(nx) * y.extent();
    const int yEnd = y.extent() - 1, zEnd = z.extent() - 1;
    MPI_Request requests[8];
    int pending = 0;
    auto exchange = [&](size_t ghost, size_t owned, MPI_Datatype type, int peer) {
        MPI_Irecv(in.data() + ghost, 1, type, peer, 0, MPI_COMM_WORLD, &requests[pending++]);
        MPI_Isend(in.data() + owned, 1, type, peer, 0, MPI_COMM_WORLD, &requests[pending++]);
    };
    size_t firstPlane = z.lowerHalo * plane;
    if (y.lowerHalo) exchange(firstPlane, firstPlane + nx, yFace, rank - 1);
    if (y.upperHalo) exchange(firstPlane + size_t(yEnd) * nx,
                              firstPlane + size_t(yEnd - 1) * nx, yFace, rank + 1);
    size_t firstRow = size_t(y.lowerHalo) * nx;
    if (z.lowerHalo) exchange(firstRow, firstRow + plane, zFace, rank - layout.py);
    if (z.upperHalo) exchange(firstRow + zEnd * plane, firstRow + (zEnd - 1) * plane,
                              zFace, rank + layout.py);

    const int y0 = 1 + y.lowerHalo, y1 = std::max(y0, yEnd - y.upperHalo);
    const int z0 = 1 + z.lowerHalo, z1 = std::max(z0, zEnd - z.upperHalo);
    update(in.data(), out.data(), nx, plane, y0, y1, z0, z1);
    if (pending) {
        MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
        update(in.data(), out.data(), nx, plane, 1, yEnd, 1, z0);
        update(in.data(), out.data(), nx, plane, 1, yEnd, z1, zEnd);
        update(in.data(), out.data(), nx, plane, 1, y0, z0, z1);
        update(in.data(), out.data(), nx, plane, y1, yEnd, z0, z1);
    }
    in.swap(out);
}

void printGrid(const Real* first, int nx, int ny, int nz, Layout layout,
               Axis y, Axis z, int rank) {
    MPI_Datatype local = blockType(nx, y.count, z.count, size_t(nx) * y.extent());
    if (rank != 0) {
        MPI_Send(first, 1, local, 0, 1, MPI_COMM_WORLD);
    } else {
        size_t plane = size_t(nx) * ny;
        std::vector<Real> full(plane * nz);
        for (int peer = 0; peer < layout.py * layout.pz; ++peer) {
            Axis destY = partition(ny, layout.py, peer % layout.py);
            Axis destZ = partition(nz, layout.pz, peer / layout.py);
            Real* dest = full.data() + destZ.start * plane + size_t(destY.start) * nx;
            if (peer == 0) {
                for (int k = 0; k < z.count; ++k)
                    std::copy_n(first + size_t(k) * nx * y.extent(), size_t(nx) * y.count,
                                dest + k * plane);
            } else {
                MPI_Datatype target = blockType(nx, destY.count, destZ.count, plane);
                MPI_Recv(dest, 1, target, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Type_free(&target);
            }
        }
        print_results(full, "Grid");
    }
    MPI_Type_free(&local);
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
    int nx = 128;
    int ny = 0;  // Will be set to nx if not specified
    int nz = 0;  // Will be set to nx if not specified
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    Layout layout = chooseLayout(ny, nz, ranks);
    bool active = rank < layout.py * layout.pz;
    Axis y = active ? partition(ny, layout.py, rank % layout.py) : Axis{};
    Axis z = active ? partition(nz, layout.pz, rank / layout.py) : Axis{};
    size_t plane = size_t(nx) * y.extent();
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid(plane * z.extent());
    size_t first = z.lowerHalo * plane + size_t(y.lowerHalo) * nx;
    for (int k = 0; k < z.count; ++k) {
        for (int j = 0; j < y.count; ++j) {
            size_t global = (size_t(z.start + k) * ny + y.start + j) * nx;
            size_t row = first + k * plane + size_t(j) * nx;
            for (int x = 0; x < nx; ++x)
                grid[row + x] = (global + x) % 19;
        }
    }
    std::vector<Real> next = grid;
    MPI_Datatype yFace = MPI_DATATYPE_NULL, zFace = MPI_DATATYPE_NULL;
    if (active && layout.py > 1) yFace = blockType(nx, 1, z.count, plane);
    if (active && layout.pz > 1) zFace = blockType(nx, y.count, 1, plane);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (active && nx > 2 && ny > 2 && nz > 2) {
        for (int iter = 0; iter < iterations; ++iter)
            iterate(grid, next, nx, y, z, rank, layout, yFace, zFace);
    }
    double elapsed = MPI_Wtime() - start, duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double updates = double(std::max(0, nx-2)) * std::max(0, ny-2)
                         * std::max(0, nz-2) * iterations;
        printf("Computation time: %ld ms\n", long(duration * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", duration > 0 ? updates / duration / 1e6 : 0);
    }
    if (yFace != MPI_DATATYPE_NULL) MPI_Type_free(&yFace);
    if (zFace != MPI_DATATYPE_NULL) MPI_Type_free(&zFace);
    if (printResults && active)
        printGrid(grid.data() + first, nx, ny, nz, layout, y, z, rank);

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        Real minVal = std::numeric_limits<Real>::infinity();
        Real maxVal = -minVal;
        int finite = 1;
        for (int k = 0; k < z.count; ++k) {
            size_t offset = first + k * plane;
            for (size_t i = offset; i < offset + size_t(y.count) * nx; ++i) {
                finite &= std::isfinite(grid[i]);
                minVal = std::min(minVal, grid[i]);
                maxVal = std::max(maxVal, grid[i]);
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &finite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        bool valid = finite && minVal >= -1e6 && maxVal <= 1e6;
        if (rank == 0) {
            if (finite) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            else printf("Validation failed: found NaN or Inf value\n");
            if (finite && !valid) printf("Validation failed: values out of expected range\n");
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        return valid ? 0 : 1;
    }
    return 0;
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
