#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Each rank owns complete XY planes, with one ghost plane on either side.
// Physical boundaries are clamped by copying the adjacent owned plane.
struct Slab {
    size_t nx, ny, nz, first_z, plane;
    int rank, ranks;
    MPI_Comm comm;
    std::vector<MPI_Request> requests;

    void exchange_start(std::vector<double>& field) {
        requests.clear();
        const int lower = rank > 0 ? rank - 1 : MPI_PROC_NULL;
        const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
        if (lower == MPI_PROC_NULL)
            std::copy_n(field.data() + plane, plane, field.data());
        if (upper == MPI_PROC_NULL)
            std::copy_n(field.data() + nz * plane, plane, field.data() + (nz + 1) * plane);

        // Chunk messages so MPI's int count does not limit the plane size.
        for (size_t offset = 0; offset < plane;) {
            const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
            MPI_Request req;
            if (lower != MPI_PROC_NULL) {
                MPI_Irecv(field.data() + offset, count, MPI_DOUBLE, lower, 1, comm, &req);
                requests.push_back(req);
                MPI_Isend(field.data() + plane + offset, count, MPI_DOUBLE, lower, 0, comm, &req);
                requests.push_back(req);
            }
            if (upper != MPI_PROC_NULL) {
                MPI_Irecv(field.data() + (nz + 1) * plane + offset, count, MPI_DOUBLE, upper, 0, comm, &req);
                requests.push_back(req);
                MPI_Isend(field.data() + nz * plane + offset, count, MPI_DOUBLE, upper, 1, comm, &req);
                requests.push_back(req);
            }
            offset += static_cast<size_t>(count);
        }
    }

    void exchange_finish() {
        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
    }
};

// The arithmetic and boundary conditions match the original two stencil passes.
// Z is local and starts at one; the ghost planes provide its neighbors.
template <bool Chemical>
void stencil(const Slab& s, const double* __restrict__ input,
             const double* __restrict__ cold, double* __restrict__ output,
             size_t zbegin, size_t zend) {
    constexpr double e_AA = -(2.0 / 9.0);
    constexpr double e_BB = -(2.0 / 9.0);
    constexpr double e_AB = (2.0 / 9.0);
    for (size_t z = zbegin; z < zend; ++z) {
        for (size_t y = 0; y < s.ny; ++y) {
            const size_t row = z * s.plane + y * s.nx;
            const size_t ym = y ? row - s.nx : row;
            const size_t yp = y + 1 < s.ny ? row + s.nx : row;
            for (size_t x = 0; x < s.nx; ++x) {
                const size_t i = row + x;
                const size_t xm = x ? i - 1 : i;
                const size_t xp = x + 1 < s.nx ? i + 1 : i;
                const double v = input[i];
                const double cxx = input[xp] + input[xm] - 2.0 * v;
                const double cyy = input[yp + x] + input[ym + x] - 2.0 * v;
                const double czz = input[i + s.plane] + input[i - s.plane] - 2.0 * v;
                const double laplacian = cxx + cyy + czz;
                if constexpr (Chemical) {
                    output[i] = 4.5 * ((v + 1.0) * e_AA + (v - 1.0) * e_BB - 2.0 * v * e_AB)
                              + 3.0 * v + v * v * v - 0.5 * laplacian;
                } else {
                    output[i] = cold[i] + 0.01 * laplacian;
                }
            }
        }
    }
}

template <bool Chemical>
void stencil_pass(Slab& s, std::vector<double>& input, const std::vector<double>& cold,
                  std::vector<double>& output) {
    s.exchange_start(input);
    if (s.nz > 2)
        stencil<Chemical>(s, input.data(), cold.data(), output.data(), 2, s.nz);
    s.exchange_finish();
    stencil<Chemical>(s, input.data(), cold.data(), output.data(), 1, 2);
    if (s.nz > 1)
        stencil<Chemical>(s, input.data(), cold.data(), output.data(), s.nz, s.nz + 1);
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
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if (nx == 0 || ny > maxSize / nx || nz > maxSize / (nx * ny) ||
        nx * ny > maxSize / sizeof(double) ||
        nz > maxSize / (nx * ny * sizeof(double))) {
        if (rank == 0) fprintf(stderr, "Invalid or too large grid size\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    // Extra ranks participate in world collectives but own no empty slabs.
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < activeRanks) {
        const size_t localNz = nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks);
        const size_t firstZ = static_cast<size_t>(rank) * (nz / activeRanks) +
                              std::min(static_cast<size_t>(rank), nz % activeRanks);
        if (localNz > maxSize / plane - 2 || (localNz + 2) * plane > maxSize / sizeof(double)) {
            fprintf(stderr, "Rank %d: local grid is too large\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Slab slab{nx, ny, localNz, firstZ, plane, rank, activeRanks, comm, {}};
        std::vector<double> cold((localNz + 2) * plane), cnew(cold.size()), mu(cold.size());
        for (size_t i = 0; i < localNz * plane; ++i) {
            const size_t linear_id = firstZ * plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[plane + i] = -1.0 + 2.0 * pseudo;
        }

        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            stencil_pass<true>(slab, cold, cold, mu);
            stencil_pass<false>(slab, mu, cold, cnew);
            cold.swap(cnew);
        }
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0.0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            const long duration = static_cast<long>(maxElapsed * 1000.0);
            printf("Computation time: %ld ms\n", duration);
            const double mcups = static_cast<double>(gridSize) * iterations / maxElapsed / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        if (printResults) {
            // Gather only for -r: print_results requires the original global order
            // for its serial Kahan sum and hash. Chunk transfers for large grids.
            if (rank == 0) {
                std::vector<double> global(gridSize);
                std::copy_n(cold.data() + plane, localNz * plane, global.data());
                size_t pos = localNz * plane;
                for (int source = 1; source < activeRanks; ++source) {
                    size_t remaining = (nz / activeRanks + (static_cast<size_t>(source) < nz % activeRanks)) * plane;
                    while (remaining) {
                        const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(global.data() + pos, count, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                        pos += static_cast<size_t>(count);
                        remaining -= static_cast<size_t>(count);
                    }
                }
                print_results(global, "Concentration");
            } else {
                size_t pos = plane, remaining = localNz * plane;
                while (remaining) {
                    const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Send(cold.data() + pos, count, MPI_DOUBLE, 0, 2, comm);
                    pos += static_cast<size_t>(count);
                    remaining -= static_cast<size_t>(count);
                }
            }
        }

        if (validate) {
            int invalid = 0;
            double minVal = std::numeric_limits<double>::infinity();
            double maxVal = -std::numeric_limits<double>::infinity();
            for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
                const double val = cold[i];
                if (!std::isfinite(val)) invalid = 1;
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            int anyInvalid = 0;
            double globalMin = 0.0, globalMax = 0.0;
            MPI_Reduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, 0, comm);
            MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) {
                printf("Validating result...\n");
                if (anyInvalid) {
                    printf("Validation failed: found NaN or Inf value\n");
                    status = 1;
                } else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                    if (globalMax > 10.0 || globalMin < -10.0) {
                        printf("Validation failed: values out of expected range\n");
                        status = 1;
                    }
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
