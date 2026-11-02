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

// Each active rank owns consecutive z planes, with one halo plane on each side.
struct Domain {
    size_t nx, ny, nz, first, local, plane;
    int rank, ranks;
    MPI_Comm comm;
};

static void startExchange(std::vector<double>& a, const Domain& d,
                          std::vector<MPI_Request>& requests) {
    const int below = d.rank > 0 ? d.rank - 1 : MPI_PROC_NULL;
    const int above = d.rank + 1 < d.ranks ? d.rank + 1 : MPI_PROC_NULL;
    if (below == MPI_PROC_NULL)
        std::copy_n(a.data() + d.plane, d.plane, a.data());
    if (above == MPI_PROC_NULL)
        std::copy_n(a.data() + d.local * d.plane, d.plane,
                    a.data() + (d.local + 1) * d.plane);

    requests.clear();
    // MPI counts are int; chunking also supports unusually large planes.
    for (size_t offset = 0; offset < d.plane;) {
        const int count = static_cast<int>(std::min(d.plane - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request req;
        if (below != MPI_PROC_NULL) {
            MPI_Irecv(a.data() + offset, count, MPI_DOUBLE, below, 1, d.comm, &req);
            requests.push_back(req);
            MPI_Isend(a.data() + d.plane + offset, count, MPI_DOUBLE, below, 0, d.comm, &req);
            requests.push_back(req);
        }
        if (above != MPI_PROC_NULL) {
            MPI_Irecv(a.data() + (d.local + 1) * d.plane + offset, count, MPI_DOUBLE, above, 0, d.comm, &req);
            requests.push_back(req);
            MPI_Isend(a.data() + d.local * d.plane + offset, count, MPI_DOUBLE, above, 1, d.comm, &req);
            requests.push_back(req);
        }
        offset += static_cast<size_t>(count);
    }
}

static void finishExchange(std::vector<MPI_Request>& requests) {
    if (!requests.empty())
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
}

// Process whole planes so that the interior can run while halos are in flight.
template<bool Chemical>
static void stencil(const std::vector<double>& input, const std::vector<double>& cold,
                    std::vector<double>& output, const Domain& d, size_t begin, size_t end) {
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    for (size_t z = begin; z < end; ++z) {
        for (size_t y = 0; y < d.ny; ++y) {
            const size_t row = z * d.plane + y * d.nx;
            const size_t ym = y ? row - d.nx : row;
            const size_t yp = y + 1 < d.ny ? row + d.nx : row;
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t i = row + x;
                const double v = input[i];
                const double cxx = input[row + (x + 1 < d.nx ? x + 1 : x)] + input[row + (x ? x - 1 : x)] - 2.0 * v;
                const double cyy = input[yp + x] + input[ym + x] - 2.0 * v;
                const double czz = input[i + d.plane] + input[i - d.plane] - 2.0 * v;
                const double lap = cxx + cyy + czz;
                if constexpr (Chemical)
                    output[i] = 4.5 * ((v + 1.0) * eAA + (v - 1.0) * eBB - 2.0 * v * eAB)
                                + 3.0 * v + v * v * v - 0.5 * lap;
                else
                    output[i] = cold[i] + 0.01 * lap;
            }
        }
    }
}

template<bool Chemical>
static void pass(std::vector<double>& input, const std::vector<double>& cold,
                 std::vector<double>& output, const Domain& d, std::vector<MPI_Request>& requests) {
    startExchange(input, d, requests);
    if (d.local > 2) stencil<Chemical>(input, cold, output, d, 2, d.local);
    finishExchange(requests);
    stencil<Chemical>(input, cold, output, d, 1, 2);
    if (d.local > 1) stencil<Chemical>(input, cold, output, d, d.local, d.local + 1);
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
    int iterations = 20, exitCode = 0;
    bool validate = false, printResults = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { help = true; break; }
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); exitCode = 1; break; }
    }
    if (help || exitCode) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return exitCode;
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny > SIZE_MAX / (std::min(nz, static_cast<size_t>(2)) + 2)) {
        if (rank == 0) fprintf(stderr, "Invalid or excessively large grid size\n");
        MPI_Finalize();
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank < active) {
        const size_t local = nz / active + (static_cast<size_t>(rank) < nz % active);
        const size_t first = static_cast<size_t>(rank) * (nz / active) + std::min(static_cast<size_t>(rank), nz % active);
        const Domain d{nx, ny, nz, first, local, nx * ny, rank, active, comm};
        if (d.plane > SIZE_MAX / (local + 2)) {
            fprintf(stderr, "Local grid size is too large\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        std::vector<double> cold((local + 2) * d.plane), cnew(cold.size()), mu(cold.size());
        for (size_t i = 0; i < local * d.plane; ++i) {
            const size_t linear = first * d.plane + i;
            const double pseudo = (((linear + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[d.plane + i] = -1.0 + 2.0 * pseudo;
        }
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        std::vector<MPI_Request> requests;
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            pass<true>(cold, cold, mu, d, requests);
            pass<false>(mu, cold, cnew, d, requests);
            cold.swap(cnew);
        }
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            const long millis = static_cast<long>(maxElapsed * 1000.0);
            printf("Computation time: %ld ms\n", millis);
            printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / maxElapsed / 1e6);
        }
        if (printResults) {
            // Gather in global order, preserving the original summation and hash.
            std::vector<double> result;
            if (rank == 0) {
                result.resize(gridSize);
                std::copy_n(cold.data() + d.plane, local * d.plane, result.data());
                for (int source = 1; source < active; ++source) {
                    const size_t n = (nz / active + (static_cast<size_t>(source) < nz % active)) * d.plane;
                    const size_t startPlane = static_cast<size_t>(source) * (nz / active) + std::min(static_cast<size_t>(source), nz % active);
                    for (size_t off = 0; off < n;) {
                        const int count = static_cast<int>(std::min(n - off, static_cast<size_t>(INT_MAX)));
                        MPI_Recv(result.data() + startPlane * d.plane + off, count, MPI_DOUBLE, source, 2, comm, MPI_STATUS_IGNORE);
                        off += static_cast<size_t>(count);
                    }
                }
                print_results(result, "Concentration");
            } else {
                const size_t n = local * d.plane;
                for (size_t off = 0; off < n;) {
                    const int count = static_cast<int>(std::min(n - off, static_cast<size_t>(INT_MAX)));
                    MPI_Send(cold.data() + d.plane + off, count, MPI_DOUBLE, 0, 2, comm);
                    off += static_cast<size_t>(count);
                }
            }
        }
        if (validate) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int invalid = 0;
            for (size_t i = d.plane; i < (local + 1) * d.plane; ++i) {
                const double v = cold[i];
                if (!std::isfinite(v)) invalid = 1;
                lo = std::min(lo, v); hi = std::max(hi, v);
            }
            double globalLo, globalHi;
            int globalInvalid;
            MPI_Allreduce(&lo, &globalLo, 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(&hi, &globalHi, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, comm);
            exitCode = globalInvalid || globalHi > 10.0 || globalLo < -10.0;
            if (rank == 0) {
                printf("Validating result...\n");
                if (globalInvalid) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Concentration range: [%.6f, %.6f]\n", globalLo, globalHi);
                    if (exitCode) printf("Validation failed: values out of expected range\n");
                }
                printf("Validation: %s\n", exitCode ? "FAILED" : "PASSED");
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
