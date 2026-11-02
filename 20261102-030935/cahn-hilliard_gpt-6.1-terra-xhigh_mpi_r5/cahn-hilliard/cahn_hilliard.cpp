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

// The local arrays contain one ghost plane on either side of the owned slab.
inline double laplacian(const double* a, size_t i, size_t x, size_t y,
                        size_t nx, size_t ny, size_t plane) {
    const double center = a[i];
    const double cxx = a[i + (x + 1 < nx)] + a[i - (x > 0)] - 2.0 * center;
    const double cyy = a[i + (y + 1 < ny ? nx : 0)] + a[i - (y > 0 ? nx : 0)] - 2.0 * center;
    const double czz = a[i + plane] + a[i - plane] - 2.0 * center;
    return cxx + cyy + czz;
}

// Compute a half-open range of owned planes. Keeping x contiguous helps vectorization.
void chemical(const std::vector<double>& c, std::vector<double>& mu,
              size_t nx, size_t ny, size_t begin, size_t end) {
    const size_t plane = nx * ny;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = z * plane + y * nx + x;
                const double cv = c[i];
                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - 0.5 * laplacian(c.data(), i, x, y, nx, ny, plane);
            }
}

void update(std::vector<double>& next, const std::vector<double>& c,
            const std::vector<double>& mu, size_t nx, size_t ny, size_t begin, size_t end) {
    const size_t plane = nx * ny;
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = z * plane + y * nx + x;
                next[i] = c[i] + 0.01 * laplacian(mu.data(), i, x, y, nx, ny, plane);
            }
}

// Receives and sends use different tags for each direction. Missing neighbors
// represent the original clamped boundary condition.
void startExchange(std::vector<double>& field, size_t plane, size_t local_z,
                   int rank, int ranks, MPI_Comm comm, MPI_Request requests[4]) {
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    if (lower == MPI_PROC_NULL)
        std::copy_n(field.data() + plane, plane, field.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(field.data() + local_z * plane, plane, field.data() + (local_z + 1) * plane);
    const int count = static_cast<int>(plane);
    MPI_Irecv(field.data(), count, MPI_DOUBLE, lower, 1, comm, &requests[0]);
    MPI_Irecv(field.data() + (local_z + 1) * plane, count, MPI_DOUBLE, upper, 0, comm, &requests[1]);
    MPI_Isend(field.data() + plane, count, MPI_DOUBLE, lower, 0, comm, &requests[2]);
    MPI_Isend(field.data() + local_z * plane, count, MPI_DOUBLE, upper, 1, comm, &requests[3]);
}

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
    if (nx == 0 || nx > static_cast<size_t>(INT_MAX) || ny > static_cast<size_t>(INT_MAX) / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny)) {
        if (world_rank == 0) fprintf(stderr, "Invalid grid size or MPI plane count exceeds INT_MAX\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    // Extra ranks do not own a slab and do not participate in the simulation.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(world_size)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) { MPI_Finalize(); return 0; }
    const int rank = world_rank, ranks = active;
    const size_t local_z = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t first_z = (nz / ranks) * rank + std::min(static_cast<size_t>(rank), nz % ranks);
    if (local_z > std::numeric_limits<size_t>::max() / plane - 2) {
        if (rank == 0) fprintf(stderr, "Local grid size is too large\n");
        MPI_Abort(comm, 1);
    }
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold((local_z + 2) * plane), cnew((local_z + 2) * plane), mu((local_z + 2) * plane);
    for (size_t i = 0; i < local_z * plane; ++i) {
        const size_t global_id = first_z * plane + i;
        const double pseudo = (((global_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[plane + i] = -1.0 + 2.0 * pseudo;
    }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[4];
        startExchange(cold, plane, local_z, rank, ranks, comm, requests);
        if (local_z > 2) chemical(cold, mu, nx, ny, 2, local_z);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        chemical(cold, mu, nx, ny, 1, 2);
        if (local_z > 1) chemical(cold, mu, nx, ny, local_z, local_z + 1);

        startExchange(mu, plane, local_z, rank, ranks, comm, requests);
        if (local_z > 2) update(cnew, cold, mu, nx, ny, 2, local_z);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        update(cnew, cold, mu, nx, ny, 1, 2);
        if (local_z > 1) update(cnew, cold, mu, nx, ny, local_z, local_z + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
    }
    if (printResults) {
        // Send in chunks to avoid MPI's int count limit and preserve global order.
        std::vector<double> global;
        if (rank == 0) {
            global.resize(gridSize);
            std::copy_n(cold.data() + plane, local_z * plane, global.data());
            for (int source = 1; source < ranks; ++source) {
                const size_t source_z = nz / ranks + (static_cast<size_t>(source) < nz % ranks);
                const size_t source_first = (nz / ranks) * source + std::min(static_cast<size_t>(source), nz % ranks);
                size_t offset = 0, count = source_z * plane;
                while (offset < count) {
                    int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(global.data() + source_first * plane + offset, chunk, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                    offset += chunk;
                }
            }
            print_results(global, "Concentration");
        } else {
            size_t offset = 0, count = local_z * plane;
            while (offset < count) {
                int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + plane + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                offset += chunk;
            }
        }
    }
    int status = 0;
    if (validate) {
        int local_bad = 0, any_bad = 0;
        double local_min = std::numeric_limits<double>::infinity(), local_max = -local_min;
        for (size_t i = plane; i < (local_z + 1) * plane; ++i) {
            double value = cold[i];
            if (!std::isfinite(value)) local_bad = 1;
            else { local_min = std::min(local_min, value); local_max = std::max(local_max, value); }
        }
        double global_min, global_max;
        MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        status = any_bad || global_max > 10.0 || global_min < -10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (any_bad) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (status) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", status ? "FAILED" : "PASSED");
        }
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
