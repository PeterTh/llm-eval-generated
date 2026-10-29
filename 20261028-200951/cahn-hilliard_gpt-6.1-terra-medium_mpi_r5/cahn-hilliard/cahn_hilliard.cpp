#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <limits>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Exchange the two neighboring planes while interior planes are processed.
struct HaloExchange {
    MPI_Request requests[4];
    int count = 0;
};

HaloExchange startExchange(std::vector<double>& field, size_t plane, size_t localZ,
                           int rank, int ranks, MPI_Comm comm) {
    HaloExchange exchange;
    if (rank > 0) {
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm,
                  &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm,
                  &exchange.requests[exchange.count++]);
    } else {
        std::copy_n(field.data() + plane, plane, field.data());
    }
    if (rank + 1 < ranks) {
        MPI_Irecv(field.data() + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 0, comm, &exchange.requests[exchange.count++]);
        MPI_Isend(field.data() + localZ * plane, static_cast<int>(plane), MPI_DOUBLE,
                  rank + 1, 1, comm, &exchange.requests[exchange.count++]);
    } else {
        std::copy_n(field.data() + localZ * plane, plane, field.data() + (localZ + 1) * plane);
    }
    return exchange;
}

void finishExchange(HaloExchange& exchange) {
    MPI_Waitall(exchange.count, exchange.requests, MPI_STATUSES_IGNORE);
}

// Compute one range of owned planes. Ghost planes supply the Z neighbors.
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t localZ, size_t begin, size_t end) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - 0.5 * computeLaplacian(c, nx, ny, localZ + 2, 1.0, 1.0, 1.0, x, y, z);
            }
}

void update(std::vector<double>& cnew, const std::vector<double>& cold,
            const std::vector<double>& mu, size_t nx, size_t ny, size_t localZ,
            size_t begin, size_t end) {
    for (size_t z = begin; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + 0.01 * computeLaplacian(mu, nx, ny, localZ + 2,
                                                                 1.0, 1.0, 1.0, x, y, z);
            }
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny > SIZE_MAX / nx || nx * ny > INT_MAX || nz > SIZE_MAX / (nx * ny) - 2) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size or MPI plane too large\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (worldRank >= activeRanks) { MPI_Finalize(); return 0; }
    const size_t localZ = nz / activeRanks + (static_cast<size_t>(worldRank) < nz % activeRanks);
    const size_t firstZ = (nz / activeRanks) * worldRank + std::min(static_cast<size_t>(worldRank), nz % activeRanks);
    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold((localZ + 2) * plane), cnew((localZ + 2) * plane), mu((localZ + 2) * plane);
    for (size_t i = 0; i < localZ * plane; ++i) {
        const size_t linear_id = firstZ * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[plane + i] = -1.0 + 2.0 * pseudo;
    }
    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        auto halo = startExchange(cold, plane, localZ, worldRank, activeRanks, comm);
        if (localZ > 2) computeChemicalPotential(cold, mu, nx, ny, localZ, 2, localZ);
        finishExchange(halo);
        computeChemicalPotential(cold, mu, nx, ny, localZ, 1, 2);
        if (localZ > 1) computeChemicalPotential(cold, mu, nx, ny, localZ, localZ, localZ + 1);
        halo = startExchange(mu, plane, localZ, worldRank, activeRanks, comm);
        if (localZ > 2) update(cnew, cold, mu, nx, ny, localZ, 2, localZ);
        finishExchange(halo);
        update(cnew, cold, mu, nx, ny, localZ, 1, 2);
        if (localZ > 1) update(cnew, cold, mu, nx, ny, localZ, localZ, localZ + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (worldRank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / maxElapsed / 1e6);
    }
    if (printResults) {
        // Chunk transfers so the global result need not fit in MPI's int counts.
        if (worldRank == 0) {
            std::vector<double> result(gridSize);
            std::copy_n(cold.data() + plane, localZ * plane, result.data());
            for (int r = 1; r < activeRanks; ++r) {
                size_t z = nz / activeRanks + (static_cast<size_t>(r) < nz % activeRanks);
                size_t offset = ((nz / activeRanks) * r + std::min(static_cast<size_t>(r), nz % activeRanks)) * plane;
                size_t remaining = z * plane;
                while (remaining) {
                    int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    remaining -= count; offset += count;
                }
            }
            print_results(result, "Concentration");
        } else {
            size_t remaining = localZ * plane, offset = plane;
            while (remaining) {
                int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + offset, count, MPI_DOUBLE, 0, 2, comm);
                remaining -= count; offset += count;
            }
        }
    }
    int valid = 1;
    if (validate) {
        double localMin = std::numeric_limits<double>::infinity(), localMax = -localMin;
        int finite = 1;
        for (size_t i = plane; i < (localZ + 1) * plane; ++i) {
            const double v = cold[i];
            if (!std::isfinite(v)) finite = 0;
            else { localMin = std::min(localMin, v); localMax = std::max(localMax, v); }
        }
        double minVal, maxVal;
        int allFinite;
        MPI_Reduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (worldRank == 0) {
            printf("Validating result...\n");
            if (!allFinite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
