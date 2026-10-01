#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Arrays and Cartesian coordinates use x, y, z order; x is contiguous.
struct Domain {
    std::array<int, 3> n, begin;
    size_t row, plane;
    size_t index(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * row + x;
    }
};

Domain domainFor(const std::array<int, 3>& global, const int* dims,
                 const int* coords) {
    Domain d;
    for (int a = 0; a < 3; ++a) {
        const int base = global[a] / dims[a], extra = global[a] % dims[a];
        d.n[a] = base + (coords[a] < extra);
        d.begin[a] = coords[a] * base + std::min(coords[a], extra);
    }
    d.row = size_t(d.n[0]) + 2;
    d.plane = d.row * (size_t(d.n[1]) + 2);
    return d;
}

// Choose a shape with small face traffic, including for non-cubic grids.
// If necessary, leave excess ranks idle rather than creating empty domains.
int chooseTopology(const std::array<int, 3>& n, int ranks, int* dims) {
    for (int used = ranks; used > 0; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(n[0], used); ++x) {
            if (used % x) continue;
            const int yz = used / x;
            for (int y = 1; y <= std::min(n[1], yz); ++y) {
                if (yz % y) continue;
                const int z = yz / y;
                if (z > n[2]) continue;
                const double lx = std::ceil(double(n[0]) / x);
                const double ly = std::ceil(double(n[1]) / y);
                const double lz = std::ceil(double(n[2]) / z);
                const double traffic = (x > 1 ? ly * lz : 0) +
                    (y > 1 ? lx * lz : 0) + (z > 1 ? lx * ly : 0);
                // Also discourage imbalanced divisions at small local sizes.
                const double score = traffic + 0.01 * lx * ly * lz;
                if (score < best) {
                    best = score;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
}

void updateBox(const Real* __restrict__ input, Real* __restrict__ output,
               const Domain& d, const std::array<int, 3>& lo,
               const std::array<int, 3>& hi) {
    for (int z = lo[2]; z < hi[2]; ++z) {
        for (int y = lo[1]; y < hi[1]; ++y) {
            const size_t start = d.index(0, y, z);
            for (int x = lo[0]; x < hi[0]; ++x) {
                const size_t i = start + x;
                // Keep the serial expression's evaluation order for exact results.
                output[i] = (input[i] + input[i-1] + input[i+1] +
                    input[i-d.row] + input[i+d.row] +
                    input[i-d.plane] + input[i+d.plane]) / 7.0;
            }
        }
    }
}

// Gather only on request, in bounded messages, without MPI int displacement limits.
void printDistributed(const std::vector<Real>& grid, const Domain& local,
                      const std::array<int, 3>& global, const int* dims,
                      MPI_Comm comm, int rank, int ranks) {
    std::vector<Real> packed(size_t(local.n[0]) * local.n[1] * local.n[2]);
    size_t offset = 0;
    for (int z = 1; z <= local.n[2]; ++z)
        for (int y = 1; y <= local.n[1]; ++y) {
            std::copy_n(grid.data() + local.index(1, y, z), local.n[0],
                        packed.data() + offset);
            offset += local.n[0];
        }
    constexpr size_t chunk = 1 << 20;
    if (rank != 0) {
        for (size_t i = 0; i < packed.size(); i += chunk)
            MPI_Send(packed.data() + i, int(std::min(chunk, packed.size()-i)),
                     MPI_DOUBLE, 0, 20, comm);
        return;
    }
    std::vector<Real> full(size_t(global[0]) * global[1] * global[2]);
    for (int source = 0; source < ranks; ++source) {
        int coords[3];
        MPI_Cart_coords(comm, source, 3, coords);
        const Domain d = domainFor(global, dims, coords);
        packed.resize(size_t(d.n[0]) * d.n[1] * d.n[2]);
        if (source != 0)
            for (size_t i = 0; i < packed.size(); i += chunk)
                MPI_Recv(packed.data() + i, int(std::min(chunk, packed.size()-i)),
                         MPI_DOUBLE, source, 20, comm, MPI_STATUS_IGNORE);
        offset = 0;
        for (int z = 0; z < d.n[2]; ++z)
            for (int y = 0; y < d.n[1]; ++y) {
                const size_t dest = (size_t(z + d.begin[2]) * global[1] +
                                    y + d.begin[1]) * global[0] + d.begin[0];
                std::copy_n(packed.data() + offset, d.n[0], full.data() + dest);
                offset += d.n[0];
            }
    }
    print_results(full, "Grid");
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int worldRank, int worldSize) {
    std::array<int, 3> global = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i+1 < argc) {
            const char option = argv[i][1];
            char* end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX-2 ||
                (option == 'x' && value == 0)) {
                if (!worldRank) fprintf(stderr, "Invalid value for -%c: %s\n", option, argv[i]);
                return 1;
            }
            if (option == 'i') iterations = int(value);
            else global[option - 'x'] = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    size_t size = 1;
    for (int n : global) {
        if (size > std::numeric_limits<size_t>::max() / sizeof(Real) / size_t(n))
            throw std::runtime_error("Grid size exceeds addressable memory");
        size *= n;
    }
    int dims[3] = {}, periods[3] = {};
    const int active = chooseTopology(global, worldSize, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED,
                   worldRank, &workers);
    int result = 0;
    if (worldRank < active) {
        MPI_Comm comm;
        MPI_Cart_create(workers, 3, dims, periods, 0, &comm);
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        const Domain d = domainFor(global, dims, coords);
        if (d.plane > std::numeric_limits<size_t>::max() / sizeof(Real) / (size_t(d.n[2])+2))
            throw std::runtime_error("Local grid size exceeds addressable memory");
        const size_t localSize = d.plane * (size_t(d.n[2])+2);
        if (!rank) {
            printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", global[0], global[1], global[2]);
            printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (%d x %d x %d)\nInitializing grid...\n", active, dims[0], dims[1], dims[2]);
        }
        std::vector<Real> grid1(localSize), grid2(localSize);
        for (int z = 1; z <= d.n[2]; ++z)
            for (int y = 1; y <= d.n[1]; ++y)
                for (int x = 1; x <= d.n[0]; ++x) {
                    const size_t globalIndex = (size_t(z-1+d.begin[2]) * global[1] +
                        y-1+d.begin[1]) * global[0] + x-1+d.begin[0];
                    grid1[d.index(x,y,z)] = Real(globalIndex % 19);
                }
        // Fixed boundaries need no per-iteration copying.
        grid2 = grid1;
        MPI_Datatype faces[3];
        int neighbors[3][2];
        const int sizes[3] = {d.n[2]+2, d.n[1]+2, d.n[0]+2};
        const int starts[3] = {};
        for (int a = 0; a < 3; ++a) {
            int subsizes[3] = {d.n[2], d.n[1], d.n[0]};
            subsizes[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &faces[a]);
            MPI_Type_commit(&faces[a]);
            MPI_Cart_shift(comm, a, 1, &neighbors[a][0], &neighbors[a][1]);
        }
        std::array<int,3> lo, hi, coreLo, coreHi;
        bool hasCore = true;
        for (int a = 0; a < 3; ++a) {
            lo[a] = 1 + (d.begin[a] == 0);
            hi[a] = d.n[a]+1 - (d.begin[a]+d.n[a] == global[a]);
            coreLo[a] = std::max(lo[a], 2);
            coreHi[a] = std::min(hi[a], d.n[a]);
            hasCore = hasCore && coreLo[a] < coreHi[a];
        }
        if (!rank) printf("Running stencil computation...\n");
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            Real* input = grid1.data();
            Real* output = grid2.data();
            MPI_Request requests[12];
            int count = 0;
            for (int a = 0; a < 3; ++a) {
                for (int side = 0; side < 2; ++side) {
                    if (neighbors[a][side] == MPI_PROC_NULL) continue;
                    int receive[3] = {1,1,1}, send[3] = {1,1,1};
                    receive[a] = side ? d.n[a]+1 : 0;
                    send[a] = side ? d.n[a] : 1;
                    MPI_Irecv(input + d.index(receive[0], receive[1], receive[2]),
                              1, faces[a], neighbors[a][side], 2*a+1-side, comm, &requests[count++]);
                    MPI_Isend(input + d.index(send[0], send[1], send[2]),
                              1, faces[a], neighbors[a][side], 2*a+side, comm, &requests[count++]);
                }
            }
            if (hasCore) updateBox(input, output, d, coreLo, coreHi);
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            if (!hasCore) updateBox(input, output, d, lo, hi);
            else {
                // Six disjoint boxes cover the shell around the computed core.
                auto shellLo = lo, shellHi = hi;
                for (int a = 0; a < 3; ++a) {
                    auto end = shellHi;
                    end[a] = coreLo[a];
                    updateBox(input, output, d, shellLo, end);
                    auto begin = shellLo;
                    begin[a] = coreHi[a];
                    updateBox(input, output, d, begin, shellHi);
                    shellLo[a] = coreLo[a];
                    shellHi[a] = coreHi[a];
                }
            }
            grid1.swap(grid2);
        }
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            const double updates = double(std::max(0,global[0]-2)) *
                std::max(0,global[1]-2) * std::max(0,global[2]-2) * iterations;
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates / seconds / 1e6 : 0);
        }
        if (printResults) printDistributed(grid1, d, global, dims, comm, rank, active);
        if (validate) {
            Real localMin = std::numeric_limits<Real>::infinity();
            Real localMax = -localMin;
            int invalid = 0, anyInvalid;
            for (int z = 1; z <= d.n[2]; ++z)
                for (int y = 1; y <= d.n[1]; ++y)
                    for (int x = 1; x <= d.n[0]; ++x) {
                        const Real value = grid1[d.index(x,y,z)];
                        invalid |= !std::isfinite(value);
                        localMin = std::min(localMin, value);
                        localMax = std::max(localMax, value);
                    }
            Real minVal, maxVal;
            MPI_Reduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, 0, comm);
            MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (!rank) {
                printf("Validating result...\n");
                if (anyInvalid) printf("Validation failed: found NaN or Inf value\n");
                else printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                result = anyInvalid || minVal < -1e6 || maxVal > 1e6;
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
        for (auto& face : faces) MPI_Type_free(&face);
        MPI_Comm_free(&comm);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
