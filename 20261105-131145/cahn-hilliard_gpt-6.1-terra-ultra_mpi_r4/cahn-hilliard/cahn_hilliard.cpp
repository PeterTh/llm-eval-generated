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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Split a dimension into consecutive, nearly equal blocks.
static size_t block_size(size_t n, int parts, int coord) {
    return n / parts + (static_cast<size_t>(coord) < n % parts);
}
static size_t block_start(size_t n, int parts, int coord) {
    return (n / parts) * coord + std::min(static_cast<size_t>(coord), n % parts);
}

// A Y-Z process grid leaves X rows contiguous. Select the largest usable
// number of ranks, then minimize the total area of inter-rank faces.
static void process_grid(size_t ny, size_t nz, int world_size, int& py, int& pz) {
    const int max_ranks = ny <= static_cast<size_t>(world_size) / nz
        ? static_cast<int>(ny * nz) : world_size;
    for (int n = max_ranks; n >= 1; --n) {
        double best = std::numeric_limits<double>::infinity();
        for (int y = 1; y <= n / y; ++y) {
            if (n % y != 0) continue;
            const int candidates[2] = {y, n / y};
            for (int a : candidates) {
                int b = n / a;
                if (static_cast<size_t>(a) > ny || static_cast<size_t>(b) > nz) continue;
                double area = static_cast<double>(a - 1) * nz + static_cast<double>(b - 1) * ny;
                if (area < best) { best = area; py = a; pz = b; }
            }
        }
        if (best != std::numeric_limits<double>::infinity()) return;
    }
}

struct Domain {
    size_t nx, ny, nz, sy, sz, global_ny, global_nz;
    int rank, py, pz, cy, cz;
    int low_y, high_y, low_z, high_z;
    size_t row, plane;
    MPI_Datatype y_face;
    MPI_Comm comm;

    size_t index(size_t x, size_t y, size_t z) const { return z * plane + y * row + x; }

    // Missing neighbors are clamped to the nearest cell, as in the serial code.
    void exchange_begin(std::vector<double>& a, MPI_Request (&requests)[8], int& count) const {
        if (low_y == MPI_PROC_NULL)
            for (size_t z = 1; z <= nz; ++z)
                std::memcpy(&a[index(0, 0, z)], &a[index(0, 1, z)], nx * sizeof(double));
        if (high_y == MPI_PROC_NULL)
            for (size_t z = 1; z <= nz; ++z)
                std::memcpy(&a[index(0, ny + 1, z)], &a[index(0, ny, z)], nx * sizeof(double));
        if (low_z == MPI_PROC_NULL)
            std::memcpy(&a[index(0, 1, 0)], &a[index(0, 1, 1)], nx * ny * sizeof(double));
        if (high_z == MPI_PROC_NULL)
            std::memcpy(&a[index(0, 1, nz + 1)], &a[index(0, 1, nz)], nx * ny * sizeof(double));
        count = 0;
        MPI_Irecv(&a[index(0, 0, 1)], 1, y_face, low_y, 1, comm, &requests[count++]);
        MPI_Irecv(&a[index(0, ny + 1, 1)], 1, y_face, high_y, 0, comm, &requests[count++]);
        MPI_Irecv(&a[index(0, 1, 0)], static_cast<int>(nx * ny), MPI_DOUBLE, low_z, 3, comm, &requests[count++]);
        MPI_Irecv(&a[index(0, 1, nz + 1)], static_cast<int>(nx * ny), MPI_DOUBLE, high_z, 2, comm, &requests[count++]);
        MPI_Isend(&a[index(0, 1, 1)], 1, y_face, low_y, 0, comm, &requests[count++]);
        MPI_Isend(&a[index(0, ny, 1)], 1, y_face, high_y, 1, comm, &requests[count++]);
        MPI_Isend(&a[index(0, 1, 1)], static_cast<int>(nx * ny), MPI_DOUBLE, low_z, 2, comm, &requests[count++]);
        MPI_Isend(&a[index(0, 1, nz)], static_cast<int>(nx * ny), MPI_DOUBLE, high_z, 3, comm, &requests[count++]);
    }
};

// Evaluate one stencil phase, overlapping communication with the local interior.
// The arithmetic order matches the original implementation.
template <bool chemical>
static void phase(const Domain& d, std::vector<double>& input, std::vector<double>& output,
                  const std::vector<double>& cold) {
    MPI_Request requests[8];
    int count;
    d.exchange_begin(input, requests, count);
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double* __restrict in = input.data();
    double* __restrict out = output.data();
    const double* old = cold.data();
    auto row = [&](size_t y, size_t z) {
        const size_t base = d.index(0, y, z);
        auto cell = [&](size_t x, size_t xn, size_t xp) {
            const size_t i = base + x;
            const double center = in[i];
            const double cxx = (in[base + xp] + in[base + xn] - 2.0 * center) / (1.0 * 1.0);
            const double cyy = (in[i + d.row] + in[i - d.row] - 2.0 * center) / (1.0 * 1.0);
            const double czz = (in[i + d.plane] + in[i - d.plane] - 2.0 * center) / (1.0 * 1.0);
            const double lap = cxx + cyy + czz;
            if constexpr (chemical) {
                const double cv = center;
                out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                            + 3.0 * cv + cv * cv * cv - 0.5 * lap;
            } else {
                out[i] = old[i] + 0.01 * 1.0 * lap;
            }
        };
        cell(0, 0, d.nx > 1 ? 1 : 0);
        for (size_t x = 1; x + 1 < d.nx; ++x) cell(x, x - 1, x + 1);
        if (d.nx > 1) cell(d.nx - 1, d.nx - 2, d.nx - 1);
    };
    for (size_t z = 2; z < d.nz; ++z)
        for (size_t y = 2; y < d.ny; ++y) row(y, z);
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    for (size_t z = 1; z <= d.nz; ++z)
        for (size_t y = 1; y <= d.ny; ++y)
            if (z == 1 || z == d.nz || y == 1 || y == d.ny) row(y, z);
}

// MPI point-to-point counts are ints. Chunk the optional result transfer.
static void transfer(double* data, size_t n, int peer, bool send, MPI_Comm comm) {
    while (n) {
        int chunk = static_cast<int>(std::min(n, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 10, comm);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 10, comm, MPI_STATUS_IGNORE);
        data += chunk;
        n -= chunk;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (world_rank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize(); return 1;
    }
    const size_t gridSize = nx * ny * nz;
    int py = 1, pz = 1;
    process_grid(ny, nz, world_size, py, pz);
    const int active = py * pz;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    const int cy = world_rank % py, cz = world_rank / py;
    Domain d{nx, block_size(ny, py, cy), block_size(nz, pz, cz),
             block_start(ny, py, cy), block_start(nz, pz, cz), ny, nz,
             world_rank, py, pz, cy, cz,
             cy ? world_rank - 1 : MPI_PROC_NULL, cy + 1 < py ? world_rank + 1 : MPI_PROC_NULL,
             cz ? world_rank - py : MPI_PROC_NULL, cz + 1 < pz ? world_rank + py : MPI_PROC_NULL,
             nx, 0, MPI_DATATYPE_NULL, comm};
    if (nx > INT_MAX || d.ny > INT_MAX || d.nz > INT_MAX ||
        d.ny + 2 > static_cast<size_t>(INT_MAX) / nx || d.ny > static_cast<size_t>(INT_MAX) / nx ||
        d.nz + 2 > SIZE_MAX / (nx * (d.ny + 2))) {
        if (world_rank == 0) fprintf(stderr, "Local grid exceeds MPI datatype limits\n");
        MPI_Abort(comm, 1);
    }
    d.plane = nx * (d.ny + 2);
    MPI_Type_vector(static_cast<int>(d.nz), static_cast<int>(nx), static_cast<int>(d.plane), MPI_DOUBLE, &d.y_face);
    MPI_Type_commit(&d.y_face);
    if (world_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold((d.nz + 2) * d.plane), cnew(cold.size()), mu(cold.size());
    for (size_t z = 1; z <= d.nz; ++z)
        for (size_t y = 1; y <= d.ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (d.sz + z - 1) * nx * ny + (d.sy + y - 1) * nx + x;
                const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
                cold[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
    if (world_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        phase<true>(d, cold, mu, cold);
        phase<false>(d, mu, cnew, cold);
        cold.swap(cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (world_rank == 0) {
        long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        double mcups = static_cast<double>(gridSize) * iterations / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) {
        std::vector<double> packed(nx * d.ny * d.nz);
        for (size_t z = 0; z < d.nz; ++z)
            for (size_t y = 0; y < d.ny; ++y)
                std::memcpy(&packed[(z * d.ny + y) * nx], &cold[d.index(0, y + 1, z + 1)], nx * sizeof(double));
        if (world_rank == 0) {
            std::vector<double> result(gridSize);
            for (int r = 0; r < active; ++r) {
                int ry = r % py, rz = r / py;
                size_t ly = block_size(ny, py, ry), lz = block_size(nz, pz, rz);
                size_t sy = block_start(ny, py, ry), sz = block_start(nz, pz, rz);
                std::vector<double> received;
                const double* source = packed.data();
                if (r != 0) {
                    received.resize(nx * ly * lz);
                    transfer(received.data(), received.size(), r, false, comm);
                    source = received.data();
                }
                for (size_t z = 0; z < lz; ++z)
                    for (size_t y = 0; y < ly; ++y)
                        std::memcpy(&result[((sz + z) * ny + sy + y) * nx], source + (z * ly + y) * nx, nx * sizeof(double));
            }
            print_results(result, "Concentration");
        } else transfer(packed.data(), packed.size(), 0, true, comm);
    }
    int status = 0;
    if (validate) {
        int nonfinite = 0;
        double minVal = std::numeric_limits<double>::infinity(), maxVal = -std::numeric_limits<double>::infinity();
        for (size_t z = 1; z <= d.nz; ++z)
            for (size_t y = 1; y <= d.ny; ++y)
                for (size_t x = 0; x < nx; ++x) {
                    double v = cold[d.index(x, y, z)];
                    if (!std::isfinite(v)) nonfinite = 1;
                    minVal = std::min(minVal, v); maxVal = std::max(maxVal, v);
                }
        int any_nonfinite;
        double global_min, global_max;
        MPI_Allreduce(&nonfinite, &any_nonfinite, 1, MPI_INT, MPI_MAX, comm);
        MPI_Reduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (world_rank == 0) {
            printf("Validating result...\n");
            if (any_nonfinite) { printf("Validation failed: found NaN or Inf value\n"); status = 1; }
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 10.0 || global_min < -10.0) { printf("Validation failed: values out of expected range\n"); status = 1; }
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    }
    MPI_Type_free(&d.y_face);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
