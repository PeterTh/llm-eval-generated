#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx,
                             size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Compute one owned plane. Local planes 1..localNz correspond to global
// planes zBegin..zBegin+localNz-1; planes 0 and localNz+1 are halos.
inline void computePlane(const Real* __restrict in, Real* __restrict out,
                         size_t lz, size_t globalZ, size_t nx, size_t ny,
                         size_t nz) {
    const size_t plane = nx * ny;
    const size_t base = lz * plane;
    if (globalZ == 0 || globalZ + 1 == nz) {
        std::copy_n(in + base, plane, out + base);
        return;
    }

    // Global X/Y boundaries are invariant.
    std::copy_n(in + base, nx, out + base);
    std::copy_n(in + base + (ny - 1) * nx, nx,
                out + base + (ny - 1) * nx);
    for (size_t y = 1; y + 1 < ny; ++y) {
        const size_t row = base + y * nx;
        out[row] = in[row];
        out[row + nx - 1] = in[row + nx - 1];
        for (size_t x = 1; x + 1 < nx; ++x) {
            const size_t p = row + x;
            out[p] = (in[p] + in[p - 1] + in[p + 1] + in[p - nx] +
                      in[p + nx] + in[p - plane] + in[p + plane]) /
                     7.0;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false;
    int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc)
            nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc)
            ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc)
            nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else {
            if (worldRank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parseError = 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (help || parseError) {
        if (worldRank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError;
    }
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (worldRank == 0) std::fprintf(stderr, "Invalid grid size or iteration count\n");
        MPI_Finalize();
        return 1;
    }

    // Empty ranks are excluded, avoiding communication through zero-size slabs.
    const int activeSize = std::min<int>(worldSize, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const size_t baseNz = nz / static_cast<size_t>(size);
    const size_t remainder = nz % static_cast<size_t>(size);
    const size_t localNz = baseNz + (static_cast<size_t>(rank) < remainder);
    const size_t zBegin = static_cast<size_t>(rank) * baseNz +
                          std::min(static_cast<size_t>(rank), remainder);
    const size_t plane = nx * ny;
    std::vector<Real> a((localNz + 2) * plane);
    std::vector<Real> b((localNz + 2) * plane);
    for (size_t lz = 1; lz <= localNz; ++lz) {
        const size_t globalBase = (zBegin + lz - 1) * plane;
        Real* dst = a.data() + lz * plane;
        for (size_t p = 0; p < plane; ++p) dst[p] = Real((globalBase + p) % 19);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        std::printf("Iterations: %d\nValidation: %s\nMPI processes: %d\n",
                    iterations, validate ? "enabled" : "disabled", size);
        std::printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = (iter & 1) ? b.data() : a.data();
        Real* out = (iter & 1) ? a.data() : b.data();
        MPI_Request req[4];
        int nr = 0;
        if (rank > 0) {
            MPI_Irecv(in, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1, comm, &req[nr++]);
            MPI_Isend(in + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0, comm, &req[nr++]);
        }
        if (rank + 1 < size) {
            MPI_Irecv(in + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 0, comm, &req[nr++]);
            MPI_Isend(in + localNz * plane, static_cast<int>(plane), MPI_DOUBLE,
                      rank + 1, 1, comm, &req[nr++]);
        }
        // Planes independent of incoming halos overlap network progress.
        for (size_t lz = 2; lz < localNz; ++lz)
            computePlane(in, out, lz, zBegin + lz - 1, nx, ny, nz);
        if (nr) MPI_Waitall(nr, req, MPI_STATUSES_IGNORE);
        computePlane(in, out, 1, zBegin, nx, ny, nz);
        if (localNz > 1)
            computePlane(in, out, localNz, zBegin + localNz - 1, nx, ny, nz);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        std::printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        std::printf("Performance: %.3f MCellUpdates/s\n",
                    maxElapsed > 0 ? updates / maxElapsed / 1e6 : 0.0);
    }

    const Real* finalData = ((iterations & 1) ? b.data() : a.data()) + plane;
    if (printResults) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rn = baseNz + (static_cast<size_t>(r) < remainder);
            const size_t rz = static_cast<size_t>(r) * baseNz +
                              std::min(static_cast<size_t>(r), remainder);
            counts[r] = static_cast<int>(rn * plane);
            displs[r] = static_cast<int>(rz * plane);
        }
        std::vector<Real> global;
        if (rank == 0) global.resize(nx * ny * nz);
        MPI_Gatherv(finalData, static_cast<int>(localNz * plane), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, comm);
        if (rank == 0) print_results(global, "Grid");
    }

    int localValid = 1;
    Real localMin = finalData[0], localMax = finalData[0];
    if (validate) {
        for (size_t i = 0; i < localNz * plane; ++i) {
            localValid &= std::isfinite(finalData[i]);
            localMin = std::min(localMin, finalData[i]);
            localMax = std::max(localMax, finalData[i]);
        }
    }
    int globalValid = 1;
    Real globalMin = 0, globalMax = 0;
    if (validate) {
        MPI_Reduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, 0, comm);
        MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            globalValid &= globalMax <= 1e6 && globalMin >= -1e6;
            std::printf("Validating result...\nValue range: [%.6f, %.6f]\n", globalMin, globalMax);
            std::printf("Validation: %s\n", globalValid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&globalValid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return validate && !globalValid;
}
