#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr std::size_t index3(const int x, const int y, const int z,
                                    const int nx, const int ny) noexcept {
    return static_cast<std::size_t>(z) * nx * ny + static_cast<std::size_t>(y) * nx + x;
}

struct Block {
    int n = 0;
    int begin = 0;
};

Block block_for(const int global, const int parts, const int coordinate) {
    const int base = global / parts;
    const int remainder = global % parts;
    return {base + (coordinate < remainder ? 1 : 0),
            coordinate * base + std::min(coordinate, remainder)};
}

void initialize_grid(std::vector<Real>& grid, const int lx, const int ly, const int lz,
                     const int gx0, const int gy0, const int gz0, const int nx, const int ny) {
    const int sx = lx + 2;
    const int sy = ly + 2;
    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            for (int x = 1; x <= lx; ++x) {
                const std::size_t local = index3(x, y, z, sx, sy);
                const std::size_t global = index3(gx0 + x - 1, gy0 + y - 1, gz0 + z - 1, nx, ny);
                grid[local] = static_cast<Real>(global % 19);
            }
        }
    }
}

inline void update_cell(const std::vector<Real>& in, std::vector<Real>& out, const int x,
                        const int y, const int z, const int sx, const int sy) {
    const std::size_t p = index3(x, y, z, sx, sy);
    out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - sx] + in[p + sx] +
              in[p - static_cast<std::size_t>(sx) * sy] +
              in[p + static_cast<std::size_t>(sx) * sy]) / 7.0;
}

void stencil_iteration(const std::vector<Real>& in, std::vector<Real>& out,
                       const int lx, const int ly, const int lz,
                       const int gx0, const int gy0, const int gz0,
                       const int nx, const int ny, const int nz) {
    const int sx = lx + 2;
    const int sy = ly + 2;
    // Compute the region that does not depend on newly received halo planes.
    for (int z = 2; z < lz; ++z)
        for (int y = 2; y < ly; ++y)
            for (int x = 2; x < lx; ++x)
                if (gx0 + x - 1 > 0 && gx0 + x - 1 < nx - 1 &&
                    gy0 + y - 1 > 0 && gy0 + y - 1 < ny - 1 &&
                    gz0 + z - 1 > 0 && gz0 + z - 1 < nz - 1)
                    update_cell(in, out, x, y, z, sx, sy);

}

void update_subdomain_boundary(const std::vector<Real>& in, std::vector<Real>& out,
                               const int lx, const int ly, const int lz,
                               const int gx0, const int gy0, const int gz0,
                               const int nx, const int ny, const int nz) {
    const int sx = lx + 2;
    const int sy = ly + 2;
    // Complete the local boundary of the subdomain after halo exchange.
    for (int z = 1; z <= lz; ++z) {
        for (int y = 1; y <= ly; ++y) {
            for (int x = 1; x <= lx; ++x) {
                if (x != 1 && x != lx && y != 1 && y != ly && z != 1 && z != lz) continue;
                const int gx = gx0 + x - 1, gy = gy0 + y - 1, gz = gz0 + z - 1;
                if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1 && gz > 0 && gz < nz - 1)
                    update_cell(in, out, x, y, z, sx, sy);
                else
                    out[index3(x, y, z, sx, sy)] = in[index3(x, y, z, sx, sy)];
            }
        }
    }
}

bool validate_result(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real min_value = grid[0], max_value = grid[0];
    for (const Real value : grid) {
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    std::printf("Value range: [%.6f, %.6f]\n", min_value, max_value);
    if (max_value > 1e6 || min_value < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void print_usage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int nx = 128, ny = 0, nz = 0, iterations = 10;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (rank == 0) print_usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); print_usage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 1 || ny < 1 || nz < 1 || iterations < 0) MPI_Abort(MPI_COMM_WORLD, 1);

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(world, 3, dims);
    if (dims[0] > nx || dims[1] > ny || dims[2] > nz) {
        if (rank == 0) std::fprintf(stderr, "Too many MPI ranks for the requested grid dimensions\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    const Block bx = block_for(nx, dims[0], coords[0]);
    const Block by = block_for(ny, dims[1], coords[1]);
    const Block bz = block_for(nz, dims[2], coords[2]);
    const int lx = bx.n, ly = by.n, lz = bz.n, sx = lx + 2, sy = ly + 2;
    std::vector<Real> grid1(static_cast<std::size_t>(sx) * sy * (lz + 2));
    std::vector<Real> grid2(grid1.size());
    initialize_grid(grid1, lx, ly, lz, bx.begin, by.begin, bz.begin, nx, ny);

    int xm, xp, ym, yp, zm, zp;
    MPI_Cart_shift(cart, 0, 1, &xm, &xp); MPI_Cart_shift(cart, 1, 1, &ym, &yp); MPI_Cart_shift(cart, 2, 1, &zm, &zp);
    MPI_Datatype xface, yface, zface;
    MPI_Type_vector(lz * sy, 1, sx, MPI_DOUBLE, &xface); MPI_Type_commit(&xface);
    MPI_Type_vector(lz, sx, sx * sy, MPI_DOUBLE, &yface); MPI_Type_commit(&yface);
    MPI_Type_contiguous(sx * sy, MPI_DOUBLE, &zface); MPI_Type_commit(&zface);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\nIterations: %d\nValidation: %s\nInitializing grid...\nRunning stencil computation...\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<MPI_Request> requests;
        requests.reserve(12);
        const std::size_t z0 = index3(0, 0, 1, sx, sy), z1 = index3(0, 0, lz, sx, sy);
        const std::size_t y0 = index3(0, 0, 0, sx, sy), y1 = index3(0, ly + 1, 0, sx, sy);
        const std::size_t x0 = index3(0, 0, 0, sx, sy), x1 = index3(lx + 1, 0, 0, sx, sy);
        auto recv = [&](int peer, std::size_t p, MPI_Datatype type, int tag) { MPI_Request q; MPI_Irecv(&grid1[p], 1, type, peer, tag, cart, &q); requests.push_back(q); };
        auto send = [&](int peer, std::size_t p, MPI_Datatype type, int tag) { MPI_Request q; MPI_Isend(&grid1[p], 1, type, peer, tag, cart, &q); requests.push_back(q); };
        const std::size_t interior_plane = static_cast<std::size_t>(sx) * sy;
        recv(xm, x0 + interior_plane, xface, 1); recv(xp, x1 + interior_plane, xface, 0);
        recv(ym, y0 + interior_plane, yface, 3); recv(yp, y1 + interior_plane, yface, 2);
        recv(zm, z0 - interior_plane, zface, 5); recv(zp, z1 + interior_plane, zface, 4);
        send(xm, x0 + interior_plane + 1, xface, 0); send(xp, x1 + interior_plane - 1, xface, 1);
        send(ym, y0 + interior_plane + sx, yface, 2); send(yp, y1 + interior_plane - sx, yface, 3);
        send(zm, z0, zface, 4); send(zp, z1, zface, 5);
        stencil_iteration(grid1, grid2, lx, ly, lz, bx.begin, by.begin, bz.begin, nx, ny, nz);
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        update_subdomain_boundary(grid1, grid2, lx, ly, lz, bx.begin, by.begin, bz.begin, nx, ny, nz);
        grid1.swap(grid2);
    }
    double elapsed = MPI_Wtime() - start, elapsed_max = 0.0;
    MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Type_free(&xface); MPI_Type_free(&yface); MPI_Type_free(&zface);

    std::vector<Real> final_grid;
    if (validate || print_results_flag) {
        const int local_count = lx * ly * lz;
        std::vector<Real> packed(static_cast<std::size_t>(local_count));
        const std::vector<Real>& local = (iterations % 2 == 0) ? grid1 : grid2;
        int at = 0;
        for (int z = 1; z <= lz; ++z) for (int y = 1; y <= ly; ++y)
            for (int x = 1; x <= lx; ++x) packed[at++] = local[index3(x, y, z, sx, sy)];
        std::vector<int> counts, displacements;
        if (rank == 0) { counts.resize(world); displacements.resize(world); }
        MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, cart);
        if (rank == 0) { int total = 0; for (int r = 0; r < world; ++r) { displacements[r] = total; total += counts[r]; } final_grid.resize(static_cast<std::size_t>(nx) * ny * nz); }
        std::vector<Real> all(rank == 0 ? static_cast<std::size_t>(nx) * ny * nz : 0);
        MPI_Gatherv(packed.data(), local_count, MPI_DOUBLE, rank == 0 ? all.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, cart);
        if (rank == 0) {
            for (int r = 0; r < world; ++r) { int c[3]; MPI_Cart_coords(cart, r, 3, c); Block rx = block_for(nx, dims[0], c[0]), ry = block_for(ny, dims[1], c[1]), rz = block_for(nz, dims[2], c[2]); int q = displacements[r]; for (int z = 0; z < rz.n; ++z) for (int y = 0; y < ry.n; ++y) for (int x = 0; x < rx.n; ++x) final_grid[index3(rx.begin+x, ry.begin+y, rz.begin+z, nx, ny)] = all[q++]; }
        }
    }
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed_max * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double cells = static_cast<double>(std::max(0, nx - 2)) * std::max(0, ny - 2) * std::max(0, nz - 2) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", elapsed_max > 0.0 ? cells / elapsed_max / 1e6 : 0.0);
        if (print_results_flag) print_results(final_grid, "Grid");
        if (validate) { std::printf("Validating result...\n"); bool valid = validate_result(final_grid); std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); MPI_Comm_free(&cart); MPI_Finalize(); return valid ? 0 : 1; }
    }
    MPI_Comm_free(&cart); MPI_Finalize(); return 0;
}
