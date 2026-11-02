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

// Each rank owns consecutive Z planes, with one ghost plane on each side.
// The two stencils need separate exchanges: mu depends on neighboring c,
// and the update then depends on neighboring mu.
struct Slab {
    size_t nx, ny, plane, local_z, z_begin;
    int rank, active;
};

static void startExchange(std::vector<double>& field, const Slab& s,
                          std::vector<MPI_Request>& requests) {
    requests.clear();
    if (s.local_z == 0) return;
    double* data = field.data();
    const int lower = s.rank == 0 ? MPI_PROC_NULL : s.rank - 1;
    const int upper = s.rank + 1 == s.active ? MPI_PROC_NULL : s.rank + 1;
    if (lower == MPI_PROC_NULL)
        std::copy_n(data + s.plane, s.plane, data);
    if (upper == MPI_PROC_NULL)
        std::copy_n(data + s.local_z * s.plane, s.plane,
                    data + (s.local_z + 1) * s.plane);

    // Chunk large planes to respect MPI's int count limit.
    for (size_t offset = 0; offset < s.plane;) {
        const int count = static_cast<int>(std::min(s.plane - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request req;
        if (lower != MPI_PROC_NULL) {
            MPI_Irecv(data + offset, count, MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            MPI_Isend(data + s.plane + offset, count, MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        if (upper != MPI_PROC_NULL) {
            MPI_Irecv(data + (s.local_z + 1) * s.plane + offset, count, MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            MPI_Isend(data + s.local_z * s.plane + offset, count, MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        offset += count;
    }
}

static void finishExchange(std::vector<MPI_Request>& requests) {
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

// The benchmark uses unit spacing in all three dimensions. Process the X
// edges separately so the long interior loop can be vectorized.
template<bool Chemical>
static void stencil(const std::vector<double>& input, const std::vector<double>& old,
                    std::vector<double>& output, const Slab& s, size_t begin, size_t end) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double* __restrict__ src = input.data();
    const double* __restrict__ c = old.data();
    double* __restrict__ dst = output.data();
    for (size_t z = begin; z < end; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            const size_t row = z * s.plane + y * s.nx;
            const size_t yp = row + (y + 1 < s.ny ? s.nx : 0);
            const size_t yn = row - (y > 0 ? s.nx : 0);
            auto cell = [&](size_t x, size_t xp, size_t xn) {
                const size_t i = row + x;
                const double center = src[i];
                const double cxx = src[row + xp] + src[row + xn] - 2.0 * center;
                const double cyy = src[yp + x] + src[yn + x] - 2.0 * center;
                const double czz = src[i + s.plane] + src[i - s.plane] - 2.0 * center;
                const double lap = cxx + cyy + czz;
                if constexpr (Chemical) {
                    dst[i] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
                           + 3.0 * center + center * center * center - 0.5 * lap;
                } else {
                    dst[i] = c[i] + 0.01 * lap;
                }
            };
            cell(0, s.nx > 1 ? 1 : 0, 0);
            for (size_t x = 1; x + 1 < s.nx; ++x) cell(x, x + 1, x - 1);
            if (s.nx > 1) cell(s.nx - 1, s.nx - 1, s.nx - 2);
        }
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t base = nz / active, remainder = nz % active;
    const size_t local_z = rank < active ? base + (static_cast<size_t>(rank) < remainder) : 0;
    const size_t z_begin = rank < active ? static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder) : nz;
    const Slab s{nx, ny, plane, local_z, z_begin, rank, active};

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold(local_z ? (local_z + 2) * plane : 0);
    std::vector<double> cnew(cold.size()), mu(cold.size());
    for (size_t i = 0; i < local_z * plane; ++i) {
        const size_t linear_id = z_begin * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[plane + i] = -1.0 + 2.0 * pseudo;
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        startExchange(cold, s, requests);
        if (local_z > 2) stencil<true>(cold, cold, mu, s, 2, local_z);
        finishExchange(requests);
        if (local_z) {
            stencil<true>(cold, cold, mu, s, 1, 2);
            if (local_z > 1) stencil<true>(cold, cold, mu, s, local_z, local_z + 1);
        }
        startExchange(mu, s, requests);
        if (local_z > 2) stencil<false>(mu, cold, cnew, s, 2, local_z);
        finishExchange(requests);
        if (local_z) {
            stencil<false>(mu, cold, cnew, s, 1, 2);
            if (local_z > 1) stencil<false>(mu, cold, cnew, s, local_z, local_z + 1);
        }
        cold.swap(cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / maxElapsed / 1e6);
    }

    if (printResults) {
        // Gather in global Z order so the existing Kahan sum, samples and hash
        // have exactly the same semantics as the serial benchmark.
        if (rank == 0) {
            std::vector<double> global(gridSize);
            std::copy_n(cold.data() + plane, local_z * plane, global.data());
            size_t pos = local_z * plane;
            for (int r = 1; r < active; ++r) {
                size_t left = (base + (static_cast<size_t>(r) < remainder)) * plane;
                while (left) {
                    const int count = static_cast<int>(std::min(left, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(global.data() + pos, count, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += count;
                    left -= count;
                }
            }
            print_results(global, "Concentration");
        } else if (local_z) {
            size_t pos = plane, left = local_z * plane;
            while (left) {
                const int count = static_cast<int>(std::min(left, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + pos, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                pos += count;
                left -= count;
            }
        }
    }

    int valid = 1;
    if (validate) {
        int finite = 1, allFinite;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (size_t i = plane; i < plane + local_z * plane; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) finite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        double minVal, maxVal;
        MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        valid = allFinite && minVal >= -10.0 && maxVal <= 10.0;
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (!valid) printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
