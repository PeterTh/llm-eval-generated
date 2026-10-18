#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Choose a surface-area-minimizing Cartesian grid with no empty subdomains.
// If the field has fewer cells than ranks, unused ranks wait in MPI_Finalize.
int chooseGrid(const size_t global[3], int ranks, int dims[3]) {
    const size_t volume = global[0] * global[1] * global[2];
    for (int p = static_cast<int>(std::min(volume, static_cast<size_t>(ranks))); p > 0; --p) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= p && static_cast<size_t>(x) <= global[0]; ++x) {
            if (p % x) continue;
            const int yz = p / x;
            for (int y = 1; y <= yz && static_cast<size_t>(y) <= global[1]; ++y) {
                if (yz % y) continue;
                const int z = yz / y;
                if (static_cast<size_t>(z) > global[2]) continue;
                const double surface = (x - 1.0) / global[0] +
                                       (y - 1.0) / global[1] + (z - 1.0) / global[2];
                if (surface < best) {
                    best = surface;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return p;
    }
    return 1;
}

struct Domain {
    MPI_Comm comm;
    int rank, dims[3], coord[3], neighbor[6];
    size_t global[3], n[3], offset[3], stride[3];
    MPI_Datatype face[3], owned;
    MPI_Request requests[12];
    int requestCount = 0;

    Domain(MPI_Comm active, const size_t sizes[3], const int grid[3]) {
        std::copy(grid, grid + 3, dims);
        std::copy(sizes, sizes + 3, global);
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coord);
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (static_cast<size_t>(coord[a]) < global[a] % dims[a]);
            offset[a] = coord[a] * (global[a] / dims[a]) +
                        std::min(static_cast<size_t>(coord[a]), global[a] % dims[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        stride[0] = 1;
        stride[1] = n[0] + 2;
        stride[2] = (n[0] + 2) * (n[1] + 2);
        int sizesWithHalo[3] = {static_cast<int>(n[2] + 2), static_cast<int>(n[1] + 2), static_cast<int>(n[0] + 2)};
        int starts[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) {
            int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
            subs[2-a] = 1;
            MPI_Type_create_subarray(3, sizesWithHalo, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
        int subs[3] = {static_cast<int>(n[2]), static_cast<int>(n[1]), static_cast<int>(n[0])};
        int ownedStart[3] = {1, 1, 1};
        MPI_Type_create_subarray(3, sizesWithHalo, subs, ownedStart, MPI_ORDER_C, MPI_DOUBLE, &owned);
        MPI_Type_commit(&owned);
    }

    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Type_free(&owned);
        MPI_Comm_free(&comm);
    }

    size_t index(size_t x, size_t y, size_t z) const {
        return x + y * stride[1] + z * stride[2];
    }

    void exchange(std::vector<double>& field) {
        requestCount = 0;
        const size_t base = index(1, 1, 1);
        // Only face halos are needed by the seven-point stencil, never edges/corners.
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                const int peer = neighbor[2*a+side];
                const size_t inside = base + (side ? (n[a]-1)*stride[a] : 0);
                const size_t halo = side ? inside + stride[a] : inside - stride[a];
                if (peer != MPI_PROC_NULL) {
                    MPI_Irecv(field.data() + halo, 1, face[a], peer, 2*a+1-side,
                              comm, &requests[requestCount++]);
                } else {
                    // Clamped physical boundaries for both concentration and mu.
                    const size_t ex = a == 0 ? 1 : n[0];
                    const size_t ey = a == 1 ? 1 : n[1];
                    const size_t ez = a == 2 ? 1 : n[2];
                    for (size_t z = 0; z < ez; ++z)
                        for (size_t y = 0; y < ey; ++y)
                            for (size_t x = 0; x < ex; ++x) {
                                const size_t delta = index(x, y, z);
                                field[halo + delta] = field[inside + delta];
                            }
                }
            }
        }
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                const int peer = neighbor[2*a+side];
                if (peer != MPI_PROC_NULL) {
                    const size_t inside = base + (side ? (n[a]-1)*stride[a] : 0);
                    MPI_Isend(field.data() + inside, 1, face[a], peer, 2*a+side,
                              comm, &requests[requestCount++]);
                }
            }
        }
    }

    void wait() { MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE); }
};

// Unit spatial spacing and the original physical constants. Keep the arithmetic
// order of the original stencil, without per-cell boundary tests.
template<bool chemical>
void computeBox(double* __restrict out, const double* field, const double* cold,
                const Domain& d, size_t xb, size_t xe, size_t yb, size_t ye, size_t zb, size_t ze) {
    const size_t sy = d.stride[1], sz = d.stride[2];
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (size_t z = zb; z < ze; ++z) {
        for (size_t y = yb; y < ye; ++y) {
            const size_t row = d.index(0, y, z);
            for (size_t x = xb; x < xe; ++x) {
                const size_t i = row + x;
                const double cv = field[i];
                const double cxx = field[i+1] + field[i-1] - 2.0 * cv;
                const double cyy = field[i+sy] + field[i-sy] - 2.0 * cv;
                const double czz = field[i+sz] + field[i-sz] - 2.0 * cv;
                const double lap = cxx + cyy + czz;
                if constexpr (chemical) {
                    out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    out[i] = cold[i] + 0.01 * lap;
                }
            }
        }
    }
}

template<bool chemical>
void stencil(std::vector<double>& out, std::vector<double>& field,
             const std::vector<double>& cold, Domain& d) {
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    d.exchange(field);
    computeBox<chemical>(out.data(), field.data(), cold.data(), d, 2, nx, 2, ny, 2, nz);
    d.wait();
    // Disjoint boundary slabs also handle one- and two-cell-wide subdomains.
    auto box = [&](size_t xb, size_t xe, size_t yb, size_t ye, size_t zb, size_t ze) {
        computeBox<chemical>(out.data(), field.data(), cold.data(), d, xb, xe, yb, ye, zb, ze);
    };
    box(1, nx+1, 1, ny+1, 1, 2);
    if (nz > 1) box(1, nx+1, 1, ny+1, nz, nz+1);
    box(1, nx+1, 1, 2, 2, nz);
    if (ny > 1) box(1, nx+1, ny, ny+1, 2, nz);
    box(1, 2, 2, ny, 2, nz);
    if (nx > 1) box(nx, nx+1, 2, ny, 2, nz);
}

void printDistributedResults(const std::vector<double>& c, const Domain& d) {
    // Gather only when requested. Derived receive types place each block in
    // global x-fastest order, preserving the serial hash and Kahan summation.
    if (d.rank != 0) {
        MPI_Send(c.data(), 1, d.owned, 0, 10, d.comm);
        return;
    }
    std::vector<double> result(d.global[0] * d.global[1] * d.global[2]);
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            std::copy_n(c.data() + d.index(1, y, z), d.n[0],
                        result.data() + (z-1)*d.global[0]*d.global[1] + (y-1)*d.global[0]);
    int count;
    MPI_Comm_size(d.comm, &count);
    for (int rank = 1; rank < count; ++rank) {
        int coord[3], sizes[3], subs[3], starts[3];
        MPI_Cart_coords(d.comm, rank, 3, coord);
        for (int a = 0; a < 3; ++a) {
            sizes[2-a] = static_cast<int>(d.global[a]);
            subs[2-a] = static_cast<int>(d.global[a] / d.dims[a] +
                                        (static_cast<size_t>(coord[a]) < d.global[a] % d.dims[a]));
            starts[2-a] = static_cast<int>(coord[a] * (d.global[a] / d.dims[a]) +
                                          std::min(static_cast<size_t>(coord[a]), d.global[a] % d.dims[a]));
        }
        MPI_Datatype block;
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(result.data(), 1, block, rank, 10, d.comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(result, "Concentration");
}

bool validateResult(const std::vector<double>& c, const Domain& d) {
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    int invalid = 0;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const double val = c[d.index(x,y,z)];
                invalid |= !std::isfinite(val);
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (d.rank == 0) {
        if (invalid) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !invalid && maxVal <= 10.0 && minVal >= -10.0;
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

int run(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // MPI subarray dimensions are ints; leave room for the two halo cells.
    const size_t limit = static_cast<size_t>(INT_MAX - 2);
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit || ny > limit || nz > limit ||
        nx > maxCells / ny || nx * ny > maxCells / nz) {
        if (worldRank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        return 1;
    }
    const size_t global[3] = {nx, ny, nz};
    int dims[3];
    const int activeSize = chooseGrid(global, worldSize, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeSize ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (active == MPI_COMM_NULL) return 0;
    Domain d(active, global, dims);
    MPI_Comm_free(&active);

    if (d.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, process grid: %d x %d x %d\n", activeSize, dims[0], dims[1], dims[2]);
        printf("Initializing concentration field...\n");
    }

    const size_t gridSize = nx * ny * nz;
    if (d.n[0]+2 > maxCells / (d.n[1]+2) ||
        (d.n[0]+2)*(d.n[1]+2) > maxCells / (d.n[2]+2)) {
        fprintf(stderr, "Local grid including halos is too large\n");
        MPI_Abort(d.comm, 1);
    }
    const size_t localSize = d.stride[2] * (d.n[2] + 2);
    std::vector<double> cold(localSize), cnew(localSize), mu(localSize);
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                const size_t linear_id = (z-1+d.offset[2])*nx*ny +
                                         (y-1+d.offset[1])*nx + x-1+d.offset[0];
                const double pseudo = (((linear_id + 1) * 1299709) % gridSize) /
                                      static_cast<double>(gridSize);
                cold[d.index(x,y,z)] = -1.0 + 2.0 * pseudo;
            }

    if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        stencil<true>(mu, cold, cold, d);
        stencil<false>(cnew, mu, cold, d);
        cold.swap(cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    if (d.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        const double mcups = static_cast<double>(gridSize) * iterations / duration / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) printDistributedResults(cold, d);
    if (validate) {
        if (d.rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(cold, d);
        if (d.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "Cahn-Hilliard failed: %s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
