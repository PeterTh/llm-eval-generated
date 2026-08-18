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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// The local arrays have a halo plane at z=0 and z=localNz+1.  X and Y use
// the same clamped boundaries as the original implementation.
inline double laplacian(const double* field, const size_t nx, const size_t ny,
                        const size_t plane, const double dx,
                        const double dy, const double dz, const size_t x,
                        const size_t y, const size_t z) noexcept {
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    // Physical-end halos are filled with the boundary plane, so z neighbours
    // are always the adjacent local-or-halo planes.
    const size_t zp = z + 1;
    const size_t zn = z - 1;
    const size_t center = z * plane + y * nx + x;
    return (field[z * plane + y * nx + xp] + field[z * plane + y * nx + xn] - 2.0 * field[center]) / (dx * dx)
         + (field[z * plane + yp * nx + x] + field[z * plane + yn * nx + x] - 2.0 * field[center]) / (dy * dy)
         + (field[zp * plane + y * nx + x] + field[zn * plane + y * nx + x] - 2.0 * field[center]) / (dz * dz);
}

std::array<MPI_Request, 4> beginHaloExchange(std::vector<double>& field, const size_t localNz, const size_t plane,
                                             const int lower, const int upper, MPI_Comm comm) {
    double* const data = field.data();
    std::array<MPI_Request, 4> requests{MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    if (lower == MPI_PROC_NULL) std::copy_n(data + plane, plane, data);
    else {
        MPI_Irecv(data, static_cast<int>(plane), MPI_DOUBLE, lower, 101, comm, &requests[0]);
        MPI_Isend(data + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 100, comm, &requests[1]);
    }
    if (upper == MPI_PROC_NULL) std::copy_n(data + localNz * plane, plane, data + (localNz + 1) * plane);
    else {
        MPI_Irecv(data + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 100, comm, &requests[2]);
        MPI_Isend(data + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 101, comm, &requests[3]);
    }
    return requests;
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t localNz, const size_t globalStartZ, const size_t globalNz) {
    const size_t plane = nx * ny;
    const size_t volume = plane * globalNz;
    for (size_t z = 1; z <= localNz; ++z) {
        const size_t globalZ = globalStartZ + z - 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = globalZ * plane + y * nx + x;
                const double pseudo = ((linearId + 1) * 1299709 % volume) / static_cast<double>(volume);
                c[z * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false, parseError = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else parseError = true;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || parseError || nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (worldRank == 0) {
            if (parseError) printf("Invalid command-line option or value\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) printf("Grid plane is too large for MPI message counts\n");
        MPI_Finalize();
        return 1;
    }

    // Ranks without an owned z plane do no work; active ranks form the compact
    // communicator used for nearest-neighbour halo traffic.
    const int active = worldRank < static_cast<int>(nz);
    MPI_Comm activeComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &activeComm);
    if (!active) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0, ranks = 1;
    MPI_Comm_rank(activeComm, &rank);
    MPI_Comm_size(activeComm, &ranks);

    const size_t baseNz = nz / static_cast<size_t>(ranks);
    const size_t remainder = nz % static_cast<size_t>(ranks);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t globalStartZ = static_cast<size_t>(rank) * baseNz + std::min(static_cast<size_t>(rank), remainder);
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\nMPI ranks: %d (%d active)\n", iterations, validate ? "enabled" : "disabled", worldSize, ranks);
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }

    std::vector<double> cold((localNz + 2) * plane);
    std::vector<double> cnew((localNz + 2) * plane);
    std::vector<double> mu((localNz + 2) * plane);
    initializeConcentration(cold, nx, ny, localNz, globalStartZ, nz);

    constexpr double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01, gamma = 0.5, diffusivity = 1.0;
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    MPI_Barrier(activeComm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        auto coldRequests = beginHaloExchange(cold, localNz, plane, lower, upper, activeComm);
        const auto calculateMu = [&](const size_t z) {
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const size_t i = z * plane + y * nx + x;
            const double cv = cold[i];
            mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) + 3.0 * cv + cv * cv * cv
                  - gamma * laplacian(cold.data(), nx, ny, plane, dx, dy, dz, x, y, z);
            }
        };
        for (size_t z = 2; z < localNz; ++z) calculateMu(z);
        MPI_Waitall(static_cast<int>(coldRequests.size()), coldRequests.data(), MPI_STATUSES_IGNORE);
        calculateMu(1);
        if (localNz > 1) calculateMu(localNz);

        auto muRequests = beginHaloExchange(mu, localNz, plane, lower, upper, activeComm);
        const auto updateConcentration = [&](const size_t z) {
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
            const size_t i = z * plane + y * nx + x;
            cnew[i] = cold[i] + dt * diffusivity * laplacian(mu.data(), nx, ny, plane, dx, dy, dz, x, y, z);
            }
        };
        for (size_t z = 2; z < localNz; ++z) updateConcentration(z);
        MPI_Waitall(static_cast<int>(muRequests.size()), muRequests.data(), MPI_STATUSES_IGNORE);
        updateConcentration(1);
        if (localNz > 1) updateConcentration(localNz);
        cold.swap(cnew);
    }
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, activeComm);

    const int localCount = static_cast<int>(localNz * plane);
    std::vector<int> counts, displacements;
    std::vector<double> global;
    if (rank == 0 && (printResults || validate)) {
        counts.resize(ranks); displacements.resize(ranks);
        for (int r = 0; r < ranks; ++r) {
            const size_t slabs = baseNz + (static_cast<size_t>(r) < remainder);
            counts[r] = static_cast<int>(slabs * plane);
            displacements[r] = static_cast<int>((static_cast<size_t>(r) * baseNz + std::min(static_cast<size_t>(r), remainder)) * plane);
        }
        global.resize(nx * ny * nz);
    }
    if (printResults || validate)
        MPI_Gatherv(cold.data() + plane, localCount, MPI_DOUBLE, global.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, activeComm);

    int result = 0;
    if (rank == 0) {
        const double cellUpdates = static_cast<double>(nx) * ny * nz * iterations;
        printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n", seconds * 1000.0, seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0);
        if (printResults) print_results(global, "Concentration");
        if (validate) {
            const auto [minIt, maxIt] = std::minmax_element(global.begin(), global.end());
            const bool finite = std::all_of(global.begin(), global.end(), [](double v) { return std::isfinite(v); });
            printf("Concentration range: [%.6f, %.6f]\n", *minIt, *maxIt);
            result = (!finite || *maxIt > 10.0 || *minIt < -10.0);
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, activeComm);
    MPI_Comm_free(&activeComm);
    MPI_Finalize();
    return result;
}
