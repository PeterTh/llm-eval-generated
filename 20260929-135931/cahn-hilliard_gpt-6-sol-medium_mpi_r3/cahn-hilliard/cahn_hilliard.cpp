#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include "../common/results_output.hpp"

// Z planes, including one ghost plane at each end, are contiguous in memory.
static inline size_t index3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

static inline double laplacian(const double* a, size_t x, size_t y, size_t z,
                               size_t nx, size_t ny) {
    const size_t p = index3(x, y, z, nx, ny);
    const size_t xp = x + (x + 1 < nx);
    const size_t xm = x - (x != 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t ym = y - (y != 0);
    const size_t plane = nx * ny;
    const double cxx = a[index3(xp, y, z, nx, ny)] + a[index3(xm, y, z, nx, ny)] - 2.0 * a[p];
    const double cyy = a[index3(x, yp, z, nx, ny)] + a[index3(x, ym, z, nx, ny)] - 2.0 * a[p];
    const double czz = a[p + plane] + a[p - plane] - 2.0 * a[p];
    return cxx + cyy + czz;
}

static void chemical_planes(const double* c, double* mu, size_t nx, size_t ny,
                            size_t first, size_t last) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    for (size_t z = first; z < last; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = index3(x, y, z, nx, ny);
                const double v = c[p];
                mu[p] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                      + 3.0 * v + v * v * v - gamma * laplacian(c, x, y, z, nx, ny);
            }
}

static void update_planes(const double* c, const double* mu, double* next,
                          size_t nx, size_t ny, size_t first, size_t last) {
    constexpr double dt = 0.01;
    for (size_t z = first; z < last; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = index3(x, y, z, nx, ny);
                next[p] = c[p] + dt * laplacian(mu, x, y, z, nx, ny);
            }
}

// Post both receives before sends. The caller computes planes independent of
// the incoming halos while communication progresses.
static int exchange_begin(std::vector<double>& a, size_t local_z, size_t plane,
                          int rank, int ranks, MPI_Request requests[4]) {
    int count = 0;
    if (rank > 0) {
        MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[count++]);
        MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[count++]);
    } else {
        std::copy_n(a.data() + plane, plane, a.data());
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(a.data() + (local_z + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 1, MPI_COMM_WORLD, &requests[count++]);
        MPI_Isend(a.data() + local_z * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 0, MPI_COMM_WORLD, &requests[count++]);
    } else {
        std::copy_n(a.data() + local_z * plane, plane, a.data() + (local_z + 1) * plane);
    }
    return count;
}

static void print_usage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, print_results_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results_flag = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > SIZE_MAX / ny || nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > SIZE_MAX / (nx * ny)) {
        if (rank == 0) fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t grid_size = plane * nz;
    // Ranks without a Z plane do no work. Neighbors are the adjacent active ranks.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t local_z = rank < active ? nz / active + (static_cast<size_t>(rank) < nz % active) : 0;
    const size_t first_z = rank < active ? (nz / active) * rank + std::min(static_cast<size_t>(rank), nz % active) : 0;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold, cnew, mu;
    if (rank < active) {
        cold.resize((local_z + 2) * plane);
        cnew.resize((local_z + 2) * plane);
        mu.resize((local_z + 2) * plane);
        for (size_t z = 1; z <= local_z; ++z)
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x) {
                    const size_t global_id = (first_z + z - 1) * plane + y * nx + x;
                    const double pseudo = (((global_id + 1) * 1299709) % grid_size) / static_cast<double>(grid_size);
                    cold[index3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
                }
    }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    if (rank < active) {
        for (int t = 0; t < iterations; ++t) {
            MPI_Request requests[4];
            int pending = exchange_begin(cold, local_z, plane, rank, active, requests);
            if (local_z > 2) chemical_planes(cold.data(), mu.data(), nx, ny, 2, local_z);
            if (pending) MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
            chemical_planes(cold.data(), mu.data(), nx, ny, 1, 2);
            if (local_z > 1) chemical_planes(cold.data(), mu.data(), nx, ny, local_z, local_z + 1);

            pending = exchange_begin(mu, local_z, plane, rank, active, requests);
            if (local_z > 2) update_planes(cold.data(), mu.data(), cnew.data(), nx, ny, 2, local_z);
            if (pending) MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
            update_planes(cold.data(), mu.data(), cnew.data(), nx, ny, 1, 2);
            if (local_z > 1) update_planes(cold.data(), mu.data(), cnew.data(), nx, ny, local_z, local_z + 1);
            cold.swap(cnew);
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double local_seconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n",
               seconds > 0.0 ? static_cast<double>(grid_size) * iterations / seconds / 1e6 : 0.0);
    }

    if (print_results_flag) {
        // Gather only for external output; normal runs keep the distributed field.
        std::vector<double> all;
        if (rank == 0) {
            all.resize(grid_size);
            std::copy_n(cold.data() + plane, local_z * plane, all.data());
            for (int source = 1; source < active; ++source) {
                const size_t source_z = nz / active + (static_cast<size_t>(source) < nz % active);
                const size_t source_first = (nz / active) * source + std::min(static_cast<size_t>(source), nz % active);
                size_t received = 0;
                while (received < source_z * plane) {
                    const int chunk = static_cast<int>(std::min(source_z * plane - received,
                                                     static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(all.data() + source_first * plane + received, chunk, MPI_DOUBLE,
                             source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    received += chunk;
                }
            }
            print_results(all, "Concentration");
        } else if (rank < active) {
            size_t sent = 0;
            while (sent < local_z * plane) {
                const int chunk = static_cast<int>(std::min(local_z * plane - sent,
                                                 static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(cold.data() + plane + sent, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                sent += chunk;
            }
        }
    }

    int result = 0;
    if (validate) {
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        int local_finite = 1;
        for (size_t i = plane; i < (local_z + 1) * plane && rank < active; ++i) {
            const double v = cold[i];
            if (!std::isfinite(v)) local_finite = 0;
            else {
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
        double global_min, global_max;
        int global_finite;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_finite, &global_finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!global_finite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 10.0 || global_min < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            result = global_finite && global_min >= -10.0 && global_max <= 10.0 ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return result;
}
