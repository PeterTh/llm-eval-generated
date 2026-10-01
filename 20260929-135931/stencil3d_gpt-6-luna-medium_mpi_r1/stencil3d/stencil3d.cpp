#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, size_t nx, size_t ny, size_t localNz,
                    size_t globalZStart) {
    const size_t plane = nx * ny;
    for (size_t z = 0; z < localNz; ++z) {
        const size_t globalZ = globalZStart + z;
        for (size_t i = 0; i < plane; ++i)
            grid[(z + 1) * plane + i] = ((globalZ * plane + i) % 19) * 1.0;
    }
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      size_t nx, size_t ny, size_t localNz, size_t globalZStart,
                      size_t globalNz) {
    const size_t plane = nx * ny;
    for (size_t lz = 0; lz < localNz; ++lz) {
        const size_t gz = globalZStart + lz;
        const size_t z = lz + 1;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t p = z * plane + y * nx + x;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny ||
                    gz == 0 || gz + 1 == globalNz) {
                    output[p] = input[p];
                } else {
                    output[p] = (input[p] + input[p - 1] + input[p + 1] +
                                 input[p - nx] + input[p + nx] +
                                 input[p - plane] + input[p + plane]) / 7.0;
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) return false;
    for (Real v : grid) {
        if (!std::isfinite(v)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto bounds = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 1e6 || *bounds.first < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n"
                "  -y <num>     Grid size in Y dimension (default: same as X)\n"
                "  -z <num>     Grid size in Z dimension (default: same as X)\n"
                "  -i <num>     Number of iterations (default: 10)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false, help = false, badArgs = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); badArgs = true; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (rank == 0 && help) printUsage(argv[0]);
    if (help || badArgs) { MPI_Finalize(); return badArgs ? 1 : 0; }

    // Use balanced contiguous slabs. Each rank owns at least one plane.
    bool invalid = nx < 3 || ny < 3 || nz < 3 || ranks > static_cast<int>(nz) || iterations < 0;
    const bool xyOverflow = ny != 0 && nx > std::numeric_limits<size_t>::max() / ny;
    const size_t plane = xyOverflow ? 0 : nx * ny;
    invalid = invalid || xyOverflow || plane > static_cast<size_t>(std::numeric_limits<int>::max()) ||
              (plane != 0 && nz > std::numeric_limits<size_t>::max() / plane) ||
              (plane != 0 && nz <= std::numeric_limits<size_t>::max() / plane &&
               plane * nz > static_cast<size_t>(std::numeric_limits<int>::max()));
    int invalidInt = invalid ? 1 : 0, anyInvalid = 0;
    MPI_Allreduce(&invalidInt, &anyInvalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyInvalid) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions/iteration count are unsupported, or there are more MPI ranks than Z planes.\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = nz / static_cast<size_t>(ranks);
    const size_t extra = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    const size_t globalZStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const size_t localSize = (localNz + 2) * plane;
    std::vector<Real> grid1(localSize), grid2(localSize);
    initializeGrid(grid1, nx, ny, localNz, globalZStart);

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                    nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        std::printf("Running stencil computation on %d MPI ranks...\n", ranks);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = iter % 2 == 0 ? grid1 : grid2;
        std::vector<Real>& out = iter % 2 == 0 ? grid2 : grid1;
        MPI_Sendrecv(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                     in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                     in.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stencilIteration(in, out, nx, ny, localNz, globalZStart, nz);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const std::vector<Real>& localFinal = iterations % 2 == 0 ? grid1 : grid2;
    std::vector<Real> globalFinal;
    if (rank == 0) globalFinal.resize(nx * ny * nz);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rn = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        const size_t rz = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra);
        counts[r] = static_cast<int>(rn * plane);
        displs[r] = static_cast<int>(rz * plane);
    }
    MPI_Gatherv(localFinal.data() + plane, static_cast<int>(localNz * plane), MPI_DOUBLE,
                rank == 0 ? globalFinal.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Computation time: %.6f s\n", maxElapsed);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n", maxElapsed > 0 ? cellUpdates / maxElapsed / 1e6 : 0.0);
        if (printResults) print_results(globalFinal, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(globalFinal);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    MPI_Finalize();
    return 0;
}
