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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local arrays include one ghost plane at either end of the z dimension.
inline double computeLaplacian(const std::vector<double>& c, size_t nx, size_t ny,
                               size_t x, size_t y, size_t z, size_t xn, size_t xp) {
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (1.0 * 1.0);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (1.0 * 1.0);
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (1.0 * 1.0);
    return cxx + cyy + czz;
}

// Begin communication before computing interior planes. Chunks keep MPI counts
// representable even if a plane has more than INT_MAX elements.
void exchangeBegin(std::vector<double>& field, size_t plane, size_t depth,
                   int rank, int ranks, MPI_Comm comm, std::vector<MPI_Request>& requests) {
    requests.clear();
    if (rank == 0) std::copy_n(field.data() + plane, plane, field.data());
    if (rank == ranks - 1) std::copy_n(field.data() + depth * plane, plane, field.data() + (depth + 1) * plane);
    for (size_t offset = 0; offset < plane;) {
        const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request req;
        if (rank > 0) {
            MPI_Irecv(field.data() + offset, count, MPI_DOUBLE, rank - 1, 1, comm, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + plane + offset, count, MPI_DOUBLE, rank - 1, 0, comm, &req);
            requests.push_back(req);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(field.data() + (depth + 1) * plane + offset, count, MPI_DOUBLE, rank + 1, 0, comm, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + depth * plane + offset, count, MPI_DOUBLE, rank + 1, 1, comm, &req);
            requests.push_back(req);
        }
        offset += static_cast<size_t>(count);
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
    int iterations = 20, earlyExit = -1;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); earlyExit = 0; break; }
        else {
            if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            earlyExit = 1; break;
        }
    }
    if (earlyExit >= 0) { MPI_Finalize(); return earlyExit; }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny > std::numeric_limits<size_t>::max() / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny)) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED, worldRank, &comm);
    int status = 0;
    if (worldRank < activeRanks) {
        const int rank = worldRank, ranks = activeRanks;
        const size_t depth = nz / ranks + (static_cast<size_t>(rank) < nz % ranks);
        const size_t zStart = (nz / ranks) * rank + std::min(static_cast<size_t>(rank), nz % ranks);
        if (std::numeric_limits<size_t>::max() / plane < 2 ||
            depth > std::numeric_limits<size_t>::max() / plane - 2) {
            fprintf(stderr, "Local grid too large\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::vector<double> cold((depth + 2) * plane), cnew((depth + 2) * plane), mu((depth + 2) * plane);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (size_t j = 0; j < depth * plane; ++j) {
            const size_t linear_id = zStart * plane + j;
            const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[plane + j] = -1.0 + 2.0 * pseudo;
        }
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        std::vector<MPI_Request> requests;
        const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
        auto chemical = [&](size_t z) {
            for (size_t y = 0; y < ny; ++y) {
                auto cell = [&](size_t x, size_t xn, size_t xp) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    const double cv = cold[idx];
                    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv - 0.5 * computeLaplacian(cold, nx, ny, x, y, z, xn, xp);
                };
                cell(0, 0, nx > 1 ? 1 : 0);
                for (size_t x = 1; x + 1 < nx; ++x) cell(x, x - 1, x + 1);
                if (nx > 1) cell(nx - 1, nx - 2, nx - 1);
            }
        };
        auto update = [&](size_t z) {
            for (size_t y = 0; y < ny; ++y) {
                auto cell = [&](size_t x, size_t xn, size_t xp) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    cnew[idx] = cold[idx] + 0.01 * 1.0 * computeLaplacian(mu, nx, ny, x, y, z, xn, xp);
                };
                cell(0, 0, nx > 1 ? 1 : 0);
                for (size_t x = 1; x + 1 < nx; ++x) cell(x, x - 1, x + 1);
                if (nx > 1) cell(nx - 1, nx - 2, nx - 1);
            }
        };
        for (int t = 0; t < iterations; ++t) {
            exchangeBegin(cold, plane, depth, rank, ranks, comm, requests);
            for (size_t z = 2; z < depth; ++z) chemical(z);
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            chemical(1); if (depth > 1) chemical(depth);
            exchangeBegin(mu, plane, depth, rank, ranks, comm, requests);
            for (size_t z = 2; z < depth; ++z) update(z);
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
            update(1); if (depth > 1) update(depth);
            cold.swap(cnew);
        }
        double elapsed = MPI_Wtime() - start, maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
            printf("Computation time: %ld ms\n", milliseconds);
            const double mcups = (double)gridSize * iterations / maxElapsed / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }
        if (printResults) {
            std::vector<double> result;
            if (rank == 0) {
                result.resize(gridSize);
                std::copy_n(cold.data() + plane, depth * plane, result.data());
                for (int src = 1; src < ranks; ++src) {
                    const size_t srcDepth = nz / ranks + (static_cast<size_t>(src) < nz % ranks);
                    const size_t srcStart = (nz / ranks) * src + std::min(static_cast<size_t>(src), nz % ranks);
                    for (size_t off = 0; off < srcDepth * plane;) {
                        const int count = static_cast<int>(std::min(srcDepth * plane - off, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + srcStart * plane + off, count, MPI_DOUBLE, src, 2, comm, MPI_STATUS_IGNORE);
                        off += static_cast<size_t>(count);
                    }
                }
                print_results(result, "Concentration");
            } else {
                for (size_t off = 0; off < depth * plane;) {
                    const int count = static_cast<int>(std::min(depth * plane - off, static_cast<size_t>(INT_MAX)));
                    MPI_Send(cold.data() + plane + off, count, MPI_DOUBLE, 0, 2, comm);
                    off += static_cast<size_t>(count);
                }
            }
        }
        if (validate) {
            int bad = 0, anyBad = 0;
            double minVal = cold[plane], maxVal = cold[plane];
            for (size_t j = plane; j < (depth + 1) * plane; ++j) {
                const double val = cold[j];
                if (!std::isfinite(val)) bad = 1;
                minVal = std::min(minVal, val); maxVal = std::max(maxVal, val);
            }
            double globalMin, globalMax;
            MPI_Reduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, 0, comm);
            MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                printf("Validating result...\n");
                if (anyBad) { printf("Validation failed: found NaN or Inf value\n"); status = 1; }
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                    if (globalMax > 10.0 || globalMin < -10.0) { printf("Validation failed: values out of expected range\n"); status = 1; }
                }
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
