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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0, ranks = 1;
    int dims[3] = {1, 1, 1};
    int global[3], n[3], start[3], neighbor[6];
    size_t sx, plane, cells;
    MPI_Datatype face[3], interior;

    static void block(int total, int parts, int coord, int& length, int& offset) {
        length = total / parts + (coord < total % parts);
        offset = coord * (total / parts) + std::min(coord, total % parts);
    }

    bool create(size_t nx, size_t ny, size_t nz, int worldRank, int worldSize) {
        global[0] = static_cast<int>(nx);
        global[1] = static_cast<int>(ny);
        global[2] = static_cast<int>(nz);
        // Select a grid that fits even thin domains. Among grids using the
        // most ranks, minimize estimated halo traffic and maximum block size.
        int used = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int pz = 1; pz <= std::min(global[2], worldSize); ++pz) {
            for (int py = 1; py <= std::min(global[1], worldSize / pz); ++py) {
                int px = std::min(global[0], worldSize / pz / py);
                int count = px * py * pz;
                double lx = double(nx) / px, ly = double(ny) / py, lz = double(nz) / pz;
                double traffic = (1.0 - 1.0 / px) * ly * lz
                               + (1.0 - 1.0 / py) * lx * lz
                               + (1.0 - 1.0 / pz) * lx * ly;
                double maxVolume = std::ceil(lx) * std::ceil(ly) * std::ceil(lz);
                double score = traffic + maxVolume / (lx * ly * lz);
                // Prefer longer contiguous rows when communication costs tie.
                bool betterLayout = score < best || (score == best &&
                    (px < dims[0] || (px == dims[0] && py < dims[1])));
                if (count > used || (count == used && betterLayout)) {
                    used = count;
                    best = score;
                    dims[0] = px; dims[1] = py; dims[2] = pz;
                }
            }
        }
        MPI_Comm active;
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < used ? 0 : MPI_UNDEFINED, worldRank, &active);
        if (active == MPI_COMM_NULL) return false;
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_free(&active);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        int coords[3];
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) {
            block(global[a], dims[a], coords[a], n[a], start[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        sx = size_t(n[0]) + 2;
        plane = sx * (size_t(n[1]) + 2);
        cells = plane * (size_t(n[2]) + 2);
        MPI_Datatype rows, column;
        MPI_Type_vector(n[1], 1, static_cast<int>(sx), MPI_DOUBLE, &column);
        MPI_Type_create_hvector(n[2], 1, MPI_Aint(plane * sizeof(double)), column, &face[0]);
        MPI_Type_free(&column);
        MPI_Type_create_hvector(n[2], n[0], MPI_Aint(plane * sizeof(double)), MPI_DOUBLE, &face[1]);
        MPI_Type_vector(n[1], n[0], static_cast<int>(sx), MPI_DOUBLE, &face[2]);
        MPI_Type_vector(n[1], n[0], static_cast<int>(sx), MPI_DOUBLE, &rows);
        MPI_Type_create_hvector(n[2], 1, MPI_Aint(plane * sizeof(double)), rows, &interior);
        MPI_Type_free(&rows);
        for (auto& type : face) MPI_Type_commit(&type);
        MPI_Type_commit(&interior);
        return true;
    }

    size_t index(size_t x, size_t y, size_t z) const { return z * plane + y * sx + x; }

    int exchange(std::vector<double>& field, MPI_Request* requests) const {
        double* f = field.data();
        const size_t stride[3] = {1, sx, plane};
        const size_t first = index(1, 1, 1);
        int count = 0;
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                const size_t owned = first + (side ? size_t(n[a]-1) * stride[a] : 0);
                const size_t halo = side ? owned + stride[a] : owned - stride[a];
                const int peer = neighbor[2*a+side];
                if (peer != MPI_PROC_NULL) {
                    MPI_Irecv(f + halo, 1, face[a], peer, 2*a+1-side, comm, &requests[count++]);
                    MPI_Isend(f + owned, 1, face[a], peer, 2*a+side, comm, &requests[count++]);
                } else {
                    // Clamp at physical boundaries, separately for c and mu.
                    if (a == 0) {
                        for (int z = 0; z < n[2]; ++z)
                            for (int y = 0; y < n[1]; ++y)
                                f[halo + z*plane + y*sx] = f[owned + z*plane + y*sx];
                    } else if (a == 1) {
                        for (int z = 0; z < n[2]; ++z)
                            std::copy_n(f + owned + z*plane, n[0], f + halo + z*plane);
                    } else {
                        for (int y = 0; y < n[1]; ++y)
                            std::copy_n(f + owned + y*sx, n[0], f + halo + y*sx);
                    }
                }
            }
        }
        return count;
    }

    void destroy() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Type_free(&interior);
        MPI_Comm_free(&comm);
    }
};

// Fixed-stride stencils allow vectorization of the contiguous X dimension.
// Interior and disjoint boundary boxes avoid per-cell boundary branches.
template<bool chemical>
void compute(const Domain& d, const double* __restrict__ field,
             const double* __restrict__ cold, double* __restrict__ out, bool boundary) {
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    auto box = [&](int xb, int xe, int yb, int ye, int zb, int ze) {
        for (int z = zb; z < ze; ++z) {
            for (int y = yb; y < ye; ++y) {
                size_t row = d.index(0, y, z);
                for (int x = xb; x < xe; ++x) {
                    size_t i = row + x;
                    double cv = field[i];
                    double cxx = field[i+1] + field[i-1] - 2.0 * cv;
                    double cyy = field[i+d.sx] + field[i-d.sx] - 2.0 * cv;
                    double czz = field[i+d.plane] + field[i-d.plane] - 2.0 * cv;
                    double lap = cxx + cyy + czz;
                    if constexpr (chemical) {
                        out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                               + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                    } else {
                        out[i] = cold[i] + 0.01 * lap;
                    }
                }
            }
        }
    };
    int nx = d.n[0], ny = d.n[1], nz = d.n[2];
    if (!boundary) {
        box(2, nx, 2, ny, 2, nz);
    } else {
        box(1, nx+1, 1, ny+1, 1, 2);
        if (nz > 1) box(1, nx+1, 1, ny+1, nz, nz+1);
        box(1, nx+1, 1, 2, 2, nz);
        if (ny > 1) box(1, nx+1, ny, ny+1, 2, nz);
        box(1, 2, 2, ny, 2, nz);
        if (nx > 1) box(nx, nx+1, 2, ny, 2, nz);
    }
}

// Preserve global traversal order for the Kahan sum, samples, and hash.
void printDistributedResults(const Domain& d, const std::vector<double>& local) {
    if (d.rank != 0) {
        MPI_Send(local.data() + d.index(1, 1, 1), 1, d.interior, 0, 10, d.comm);
        return;
    }
    std::vector<double> global(size_t(d.global[0]) * d.global[1] * d.global[2]);
    for (int rank = 0; rank < d.ranks; ++rank) {
        int coords[3], n[3], start[3];
        MPI_Cart_coords(d.comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) Domain::block(d.global[a], d.dims[a], coords[a], n[a], start[a]);
        if (rank == 0) {
            for (int z = 0; z < n[2]; ++z)
                for (int y = 0; y < n[1]; ++y)
                    std::copy_n(local.data() + d.index(1, y+1, z+1), n[0],
                                global.data() + idx3(start[0], start[1]+y, start[2]+z, d.global[0], d.global[1]));
        } else {
            int sizes[3] = {d.global[2], d.global[1], d.global[0]};
            int subsizes[3] = {n[2], n[1], n[0]};
            int offsets[3] = {start[2], start[1], start[0]};
            MPI_Datatype destination;
            MPI_Type_create_subarray(3, sizes, subsizes, offsets, MPI_ORDER_C, MPI_DOUBLE, &destination);
            MPI_Type_commit(&destination);
            MPI_Recv(global.data(), 1, destination, rank, 10, d.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&destination);
        }
    }
    print_results(global, "Concentration");
}

bool validateResult(const Domain& d, const std::vector<double>& c) {
    double local[3] = {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity(), 0.0};
    for (int z = 1; z <= d.n[2]; ++z)
        for (int y = 1; y <= d.n[1]; ++y)
            for (int x = 1; x <= d.n[0]; ++x) {
                double v = c[d.index(x, y, z)];
                if (!std::isfinite(v)) local[2] = -1.0;
                local[0] = std::min(local[0], v);
                local[1] = std::min(local[1], -v);
            }
    double global[3];
    MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MIN, d.comm);
    bool valid = global[2] == 0.0 && global[0] >= -10.0 && -global[1] <= 10.0;
    if (d.rank == 0) {
        if (global[2] != 0.0) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", global[0], -global[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
    }
    return valid;
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
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx + 2 > limit / (ny + 2) || (nx + 2) * (ny + 2) > limit / (nz + 2)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or oversized grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    Domain domain;
    if (!domain.create(nx, ny, nz, worldRank, worldSize)) {
        MPI_Finalize();
        return 0;
    }
    const size_t gridSize = nx * ny * nz;
    if (domain.rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d x %d)\n", domain.ranks,
               domain.dims[0], domain.dims[1], domain.dims[2]);
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold, cnew, mu;
    try {
        cold.resize(domain.cells);
        cnew.resize(domain.cells);
        mu.resize(domain.cells);
        for (int z = 1; z <= domain.n[2]; ++z)
            for (int y = 1; y <= domain.n[1]; ++y)
                for (int x = 1; x <= domain.n[0]; ++x) {
                    size_t id = idx3(domain.start[0]+x-1, domain.start[1]+y-1,
                                     domain.start[2]+z-1, nx, ny);
                    double pseudo = (((id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
                    cold[domain.index(x, y, z)] = -1.0 + 2.0 * pseudo;
                }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (domain.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(domain.comm);
    double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[12];
        int count = domain.exchange(cold, requests);
        compute<true>(domain, cold.data(), cold.data(), mu.data(), false);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        compute<true>(domain, cold.data(), cold.data(), mu.data(), true);

        count = domain.exchange(mu, requests);
        compute<false>(domain, mu.data(), cold.data(), cnew.data(), false);
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        compute<false>(domain, mu.data(), cold.data(), cnew.data(), true);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    double localTime = MPI_Wtime() - start, elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (domain.rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        double mcups = elapsed > 0.0 ? double(gridSize) * iterations / elapsed / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    if (printResults) {
        printDistributedResults(domain, cold);
    }
    
    // Validation
    bool valid = true;
    if (validate) {
        if (domain.rank == 0) printf("Validating result...\n");
        valid = validateResult(domain, cold);
        if (domain.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    domain.destroy();
    MPI_Finalize();
    return valid ? 0 : 1;
}
