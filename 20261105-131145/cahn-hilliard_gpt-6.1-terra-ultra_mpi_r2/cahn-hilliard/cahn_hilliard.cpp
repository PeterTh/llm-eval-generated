#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Each rank owns consecutive z planes, with one halo plane on each side.
// The initial condition is defined by the global, rather than local, index.
void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny,
                             size_t nz, size_t localZ, size_t firstZ) {
    const size_t plane = nx * ny;
    const size_t vol = plane * nz;
    for (size_t z = 0; z < localZ; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            const size_t linear_id = (firstZ + z) * plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol);
            c[(z + 1) * plane + i] = -1.0 + 2.0 * pseudo;
        }
    }
}

// Exchange the two face planes. MPI_PROC_NULL leaves physical boundary halos
// untouched; these are filled by copying the adjacent plane before the exchange.
void startExchange(std::vector<double>& field, size_t plane, size_t localZ,
                   size_t offset, int count, int previous, int next, MPI_Request requests[4]) {
    MPI_Irecv(field.data() + offset, count, MPI_DOUBLE, previous, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(field.data() + (localZ + 1) * plane + offset, count, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(field.data() + plane + offset, count, MPI_DOUBLE, previous, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(field.data() + localZ * plane + offset, count, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &requests[3]);
}

// Apply either half of the two-Laplacian update to a range of local planes.
// x and y boundaries and the halo-filled z boundaries use clamping exactly as
// in the original serial stencil. Grid spacings are all 1 in this benchmark.
template <bool Chemical>
void applyStencil(const std::vector<double>& input, const std::vector<double>& cold,
                  std::vector<double>& output, size_t nx, size_t ny,
                  size_t beginZ, size_t endZ) {
    const size_t plane = nx * ny;
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    const double* __restrict in = input.data();
    const double* __restrict old = cold.data();
    double* __restrict out = output.data();
    for (size_t z = beginZ; z < endZ; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t rowN = z * plane + (y > 0 ? y - 1 : 0) * nx;
            const size_t rowP = z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const auto cell = [&](size_t x, size_t xn, size_t xp) {
                const size_t i = row + x;
                const double v = in[i];
                const double cxx = in[row + xp] + in[row + xn] - 2.0 * v;
                const double cyy = in[rowP + x] + in[rowN + x] - 2.0 * v;
                const double czz = in[i + plane] + in[i - plane] - 2.0 * v;
                const double lap = cxx + cyy + czz;
                if constexpr (Chemical) {
                    out[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                           + 3.0 * v + v * v * v - 0.5 * lap;
                } else {
                    out[i] = old[i] + 0.01 * lap;
                }
            };
            cell(0, 0, nx > 1 ? 1 : 0);
            for (size_t x = 1; x + 1 < nx; ++x)
                cell(x, x - 1, x + 1);
            if (nx > 1)
                cell(nx - 1, nx - 2, nx - 1);
        }
    }
}

template <bool Chemical>
void stencilWithExchange(std::vector<double>& input, const std::vector<double>& cold,
                         std::vector<double>& output, size_t nx, size_t ny, size_t localZ,
                         int previous, int next) {
    const size_t plane = nx * ny;
    if (previous == MPI_PROC_NULL)
        std::copy_n(input.data() + plane, plane, input.data());
    if (next == MPI_PROC_NULL)
        std::copy_n(input.data() + localZ * plane, plane, input.data() + (localZ + 1) * plane);
    MPI_Request requests[4];
    const int firstCount = static_cast<int>(std::min(plane, static_cast<size_t>(INT_MAX)));
    startExchange(input, plane, localZ, 0, firstCount, previous, next, requests);
    if (localZ > 2)
        applyStencil<Chemical>(input, cold, output, nx, ny, 2, localZ);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    // MPI counts are int. Very large planes are exchanged in ordered chunks.
    for (size_t offset = firstCount; offset < plane;) {
        const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
        startExchange(input, plane, localZ, offset, count, previous, next, requests);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        offset += count;
    }
    applyStencil<Chemical>(input, cold, output, nx, ny, 1, 2);
    if (localZ > 1)
        applyStencil<Chemical>(input, cold, output, nx, ny, localZ, localZ + 1);
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    // Account for the two halo planes and the vector allocation size limit.
    const size_t maxElements = std::vector<double>().max_size();
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny ||
        nx * ny > SIZE_MAX / nz || maxElements / (nx * ny) < 2 ||
        nz / ranks + (nz % ranks != 0) > maxElements / (nx * ny) - 2 ||
        (printResults && nx * ny * nz > maxElements)) {
        if (rank == 0) fprintf(stderr, "Invalid or overflowing grid size\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t localZ = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
    const size_t firstZ = static_cast<size_t>(rank) * (nz / ranks) + std::min(static_cast<size_t>(rank), nz % ranks);
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const int previous = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int next = rank + 1 < activeRanks ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold(localZ ? (localZ + 2) * plane : 0);
    std::vector<double> cnew(cold.size());
    std::vector<double> mu(cold.size());
    if (localZ) initializeConcentration(cold, nx, ny, nz, localZ, firstZ);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        if (localZ) {
            stencilWithExchange<true>(cold, cold, mu, nx, ny, localZ, previous, next);
            stencilWithExchange<false>(mu, cold, cnew, nx, ny, localZ, previous, next);
            std::swap(cold, cnew);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000));
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / maxElapsed / 1e6);
    }

    // The external result hash and compensated sum are order dependent. Gather
    // only when requested, in global order, to retain their exact semantics.
    if (printResults) {
        constexpr size_t chunk = INT_MAX;
        if (rank == 0) {
            std::vector<double> result(gridSize);
            std::copy_n(cold.data() + plane, localZ * plane, result.data());
            for (int source = 1; source < activeRanks; ++source) {
                const size_t sourceZ = nz / ranks + (static_cast<size_t>(source) < nz % ranks);
                const size_t sourceFirst = static_cast<size_t>(source) * (nz / ranks) + std::min(static_cast<size_t>(source), nz % ranks);
                const size_t count = sourceZ * plane;
                for (size_t offset = 0; offset < count; offset += std::min(chunk, count - offset)) {
                    const int n = static_cast<int>(std::min(chunk, count - offset));
                    MPI_Recv(result.data() + sourceFirst * plane + offset, n, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            print_results(result, "Concentration");
        } else if (localZ) {
            const size_t count = localZ * plane;
            for (size_t offset = 0; offset < count; offset += std::min(chunk, count - offset)) {
                const int n = static_cast<int>(std::min(chunk, count - offset));
                MPI_Send(cold.data() + plane + offset, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    int valid = 1;
    if (validate) {
        int localFinite = 1;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < localZ * plane; ++i) {
            const double value = cold[plane + i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int finite = 0;
        double minVal = 0, maxVal = 0;
        MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = 0;
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
