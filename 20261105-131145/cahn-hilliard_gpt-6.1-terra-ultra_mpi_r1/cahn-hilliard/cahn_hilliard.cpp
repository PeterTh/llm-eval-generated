#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Local arrays have one ghost plane on each side of the owned z slabs.
struct Domain {
    size_t nx, ny, nz, plane, first_z, global_volume;
    int rank, ranks;
    MPI_Comm comm;
};

// Split large planes into messages representable by MPI's int count argument.
static void exchange_begin(std::vector<double>& field, const Domain& d,
                           std::vector<MPI_Request>& requests) {
    const int lower = d.rank == 0 ? MPI_PROC_NULL : d.rank - 1;
    const int upper = d.rank + 1 == d.ranks ? MPI_PROC_NULL : d.rank + 1;
    if (lower == MPI_PROC_NULL)
        std::copy_n(field.data() + d.plane, d.plane, field.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(field.data() + d.nz * d.plane, d.plane,
                    field.data() + (d.nz + 1) * d.plane);

    requests.clear();
    for (size_t offset = 0; offset < d.plane;) {
        const int count = static_cast<int>(std::min(d.plane - offset,
                                  static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Request req;
        if (lower != MPI_PROC_NULL) {
            MPI_Irecv(field.data() + offset, count, MPI_DOUBLE, lower, 1, d.comm, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + d.plane + offset, count, MPI_DOUBLE, lower, 0, d.comm, &req);
            requests.push_back(req);
        }
        if (upper != MPI_PROC_NULL) {
            MPI_Irecv(field.data() + (d.nz + 1) * d.plane + offset, count, MPI_DOUBLE, upper, 0, d.comm, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + d.nz * d.plane + offset, count, MPI_DOUBLE, upper, 1, d.comm, &req);
            requests.push_back(req);
        }
        offset += static_cast<size_t>(count);
    }
}

static void exchange_end(std::vector<MPI_Request>& requests) {
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

// The x and y faces are clamped here; the z faces are supplied by the halos.
static inline double laplacian(const double* f, size_t i, size_t x, size_t y,
                               const Domain& d) {
    const size_t xp = x + 1 < d.nx ? i + 1 : i;
    const size_t xn = x > 0 ? i - 1 : i;
    const size_t yp = y + 1 < d.ny ? i + d.nx : i;
    const size_t yn = y > 0 ? i - d.nx : i;
    const double cxx = (f[xp] + f[xn] - 2.0 * f[i]);
    const double cyy = (f[yp] + f[yn] - 2.0 * f[i]);
    const double czz = (f[i + d.plane] + f[i - d.plane] - 2.0 * f[i]);
    return cxx + cyy + czz;
}

static void chemical_planes(const std::vector<double>& c, std::vector<double>& mu,
                            const Domain& d, size_t begin, size_t end) {
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < d.ny; ++y)
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t i = z * d.plane + y * d.nx + x;
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c.data(), i, x, y, d);
            }
}

static void update_planes(std::vector<double>& next, const std::vector<double>& c,
                          const std::vector<double>& mu, const Domain& d,
                          size_t begin, size_t end) {
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < d.ny; ++y)
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t i = z * d.plane + y * d.nx + x;
                next[i] = c[i] + 0.01 * laplacian(mu.data(), i, x, y, d);
            }
}

static void printUsage(const char* progName) {
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
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (world_rank == 0) fprintf(stderr, "Invalid or excessively large grid size\n");
        MPI_Finalize();
        return 1;
    }

    // Ranks without a plane do not participate in the simulation communicator.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(world_size)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    int exit_code = 0;
    if (world_rank < active) {
        Domain d{nx, ny, nz / active + (static_cast<size_t>(world_rank) < nz % active),
                 nx * ny, static_cast<size_t>(world_rank) * (nz / active) +
                     std::min(static_cast<size_t>(world_rank), nz % active),
                 nx * ny * nz, world_rank, active, comm};
        std::vector<double> cold((d.nz + 2) * d.plane), cnew(cold.size()), mu(cold.size());
        if (world_rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (size_t i = 0; i < d.nz * d.plane; ++i) {
            const size_t linear_id = d.first_z * d.plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % d.global_volume) /
                                  static_cast<double>(d.global_volume);
            cold[d.plane + i] = -1.0 + 2.0 * pseudo;
        }
        if (world_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        std::vector<MPI_Request> requests;
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            exchange_begin(cold, d, requests);
            if (d.nz > 2) chemical_planes(cold, mu, d, 2, d.nz);
            exchange_end(requests);
            chemical_planes(cold, mu, d, 1, 2);
            if (d.nz > 1) chemical_planes(cold, mu, d, d.nz, d.nz + 1);

            exchange_begin(mu, d, requests);
            if (d.nz > 2) update_planes(cnew, cold, mu, d, 2, d.nz);
            exchange_end(requests);
            update_planes(cnew, cold, mu, d, 1, 2);
            if (d.nz > 1) update_planes(cnew, cold, mu, d, d.nz, d.nz + 1);
            cold.swap(cnew);
        }
        const double elapsed = MPI_Wtime() - start;
        double max_elapsed = 0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (world_rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(d.global_volume) * iterations / max_elapsed / 1e6);
        }
        if (printResults) {
            // Preserve the serial result format, including its order-dependent hash and sum.
            std::vector<double> global;
            if (world_rank == 0) {
                global.resize(d.global_volume);
                std::copy_n(cold.data() + d.plane, d.nz * d.plane, global.data());
                for (int rank = 1; rank < active; ++rank) {
                    const size_t count = (nz / active + (static_cast<size_t>(rank) < nz % active)) * d.plane;
                    const size_t first = (static_cast<size_t>(rank) * (nz / active) +
                                          std::min(static_cast<size_t>(rank), nz % active)) * d.plane;
                    for (size_t offset = 0; offset < count;) {
                        const int n = static_cast<int>(std::min(count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                        MPI_Recv(global.data() + first + offset, n, MPI_DOUBLE, rank, 2, comm, MPI_STATUS_IGNORE);
                        offset += static_cast<size_t>(n);
                    }
                }
                print_results(global, "Concentration");
            } else {
                const size_t count = d.nz * d.plane;
                for (size_t offset = 0; offset < count;) {
                    const int n = static_cast<int>(std::min(count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Send(cold.data() + d.plane + offset, n, MPI_DOUBLE, 0, 2, comm);
                    offset += static_cast<size_t>(n);
                }
            }
        }
        if (validate) {
            int local_bad = 0, global_bad = 0;
            double local_min = std::numeric_limits<double>::infinity();
            double local_max = -std::numeric_limits<double>::infinity();
            for (size_t i = d.plane; i < (d.nz + 1) * d.plane; ++i) {
                local_bad |= !std::isfinite(cold[i]);
                local_min = std::min(local_min, cold[i]);
                local_max = std::max(local_max, cold[i]);
            }
            double global_min, global_max;
            MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);
            MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (world_rank == 0) {
                printf("Validating result...\n");
                if (global_bad) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                    if (global_max > 10.0 || global_min < -10.0) {
                        printf("Validation failed: values out of expected range\n");
                        global_bad = 1;
                    }
                }
                printf("Validation: %s\n", global_bad ? "FAILED" : "PASSED");
                exit_code = global_bad ? 1 : 0;
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Finalize();
    return exit_code;
}
