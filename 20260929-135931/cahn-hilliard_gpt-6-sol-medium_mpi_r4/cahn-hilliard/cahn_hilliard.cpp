#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Local planes 1..local_z contain data; planes 0 and local_z+1 are halos.
struct Slab {
    size_t nx, ny, local_z, plane;
    int rank, ranks;
    MPI_Comm comm;

    size_t offset(size_t z, size_t y, size_t x) const {
        return z * plane + y * nx + x;
    }

    void beginExchange(std::vector<double>& field, MPI_Request requests[4]) const {
        std::fill(requests, requests + 4, MPI_REQUEST_NULL);
        if (rank > 0) {
            MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 0, comm, &requests[0]);
            MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank - 1, 1, comm, &requests[1]);
        } else {
            std::copy_n(field.data() + plane, plane, field.data());
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(field.data() + (local_z + 1) * plane,
                      static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1, comm, &requests[2]);
            MPI_Isend(field.data() + local_z * plane,
                      static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0, comm, &requests[3]);
        } else {
            std::copy_n(field.data() + local_z * plane, plane,
                        field.data() + (local_z + 1) * plane);
        }
    }

    double laplacian(const std::vector<double>& field, size_t z, size_t y, size_t x) const {
        const size_t p = offset(z, y, x);
        const double center = field[p];
        const double cxx = (field[offset(z, y, x + (x + 1 < nx))] +
                            field[offset(z, y, x - (x > 0))] - 2.0 * center);
        const double cyy = (field[offset(z, y + (y + 1 < ny), x)] +
                            field[offset(z, y - (y > 0), x)] - 2.0 * center);
        const double czz = (field[p + plane] + field[p - plane] - 2.0 * center);
        return cxx + cyy + czz;
    }

    void chemicalPlane(const std::vector<double>& c, std::vector<double>& mu, size_t z) const {
        constexpr double e_AA = -(2.0 / 9.0);
        constexpr double e_BB = -(2.0 / 9.0);
        constexpr double e_AB = 2.0 / 9.0;
        constexpr double gamma = 0.5;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = offset(z, y, x);
                const double cv = c[p];
                mu[p] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv - gamma * laplacian(c, z, y, x);
            }
        }
    }

    void updatePlane(const std::vector<double>& c, const std::vector<double>& mu,
                     std::vector<double>& next, size_t z) const {
        constexpr double dt = 0.01;
        constexpr double D = 1.0;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = offset(z, y, x);
                next[p] = c[p] + dt * D * laplacian(mu, z, y, x);
            }
        }
    }

    template <typename Work>
    void withHalos(std::vector<double>& field, Work work) const {
        MPI_Request requests[4];
        beginExchange(field, requests);
        for (size_t z = 2; z < local_z; ++z) work(z);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        work(1);
        if (local_z > 1) work(local_z);
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    int parse_error = 0;
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
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parse_error = 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    // MPI plane messages and result gathering use int counts/displacements.
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > SIZE_MAX / (nx * ny) ||
        nx * ny * nz > static_cast<size_t>(std::numeric_limits<int>::max())) parse_error = 1;
    if (parse_error) {
        if (world_rank == 0) fprintf(stderr, "Invalid grid size or option\n");
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(world_size)));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank >= active) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t local_z = base + (static_cast<size_t>(rank) < remainder);
    const size_t first_z = base * static_cast<size_t>(rank) +
                           std::min(static_cast<size_t>(rank), remainder);
    const Slab slab{nx, ny, local_z, plane, rank, ranks, comm};
    std::vector<double> cold((local_z + 2) * plane);
    std::vector<double> cnew((local_z + 2) * plane);
    std::vector<double> mu((local_z + 2) * plane);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    for (size_t z = 1; z <= local_z; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = (first_z + z - 1) * plane + y * nx + x;
                const double pseudo = (((global_id + 1) * 1299709) % gridSize) /
                                      static_cast<double>(gridSize);
                cold[slab.offset(z, y, x)] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        slab.withHalos(cold, [&](size_t z) { slab.chemicalPlane(cold, mu, z); });
        slab.withHalos(mu, [&](size_t z) { slab.updatePlane(cold, mu, cnew, z); });
        std::swap(cold, cnew);
    }
    const double local_seconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double mcups = static_cast<double>(gridSize) * iterations / seconds / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<int> counts, displacements;
        std::vector<double> result;
        if (rank == 0) {
            counts.resize(ranks);
            displacements.resize(ranks);
            for (int r = 0; r < ranks; ++r) {
                const size_t n = base + (static_cast<size_t>(r) < remainder);
                const size_t first = base * static_cast<size_t>(r) +
                                     std::min(static_cast<size_t>(r), remainder);
                counts[r] = static_cast<int>(n * plane);
                displacements[r] = static_cast<int>(first * plane);
            }
            result.resize(gridSize);
        }
        MPI_Gatherv(cold.data() + plane, static_cast<int>(local_z * plane), MPI_DOUBLE,
                    rank == 0 ? result.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(result, "Concentration");
    }

    int failed = 0;
    if (validate) {
        int local_nonfinite = 0;
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t p = plane; p < (local_z + 1) * plane; ++p) {
            const double value = cold[p];
            local_nonfinite |= !std::isfinite(value);
            local_min = std::min(local_min, value);
            local_max = std::max(local_max, value);
        }
        int nonfinite = 0;
        double min_value = 0.0, max_value = 0.0;
        MPI_Reduce(&local_nonfinite, &nonfinite, 1, MPI_INT, MPI_MAX, 0, comm);
        MPI_Reduce(&local_min, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_max, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (nonfinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
                if (max_value > 10.0 || min_value < -10.0)
                    printf("Validation failed: values out of expected range\n");
            }
            failed = nonfinite || max_value > 10.0 || min_value < -10.0;
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return failed;
}
