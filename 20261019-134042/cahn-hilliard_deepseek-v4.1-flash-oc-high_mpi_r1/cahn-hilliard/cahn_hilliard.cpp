#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for the (global) unpadded layout
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Distributed domain description.
//
// The grid is decomposed into contiguous slabs along Z.  Every rank stores a
// padded local block:
//
//     x: 0 .. nx+1      (one ghost layer on each side)
//     y: 0 .. ny+1      (one ghost layer on each side)
//     z: 0 .. lz+1      (one ghost layer on each side)
//
// X/Y ghosts implement the clamped (Neumann-like) boundary condition, while
// Z ghosts are exchanged with the neighbouring ranks.  With this layout the
// Laplacian becomes a uniform 7-point stencil for every interior cell, which
// keeps the hot loops branch-free and vectorizable.
struct Domain {
    size_t nx, ny, nz;   // global dimensions
    size_t lx, ly, lz;   // local number of interior cells
    size_t z0;           // first global z plane owned by this rank
    size_t sx, sy, sz;   // padded strides
    size_t total;        // padded local allocation size
    int rank, size;
    int lower, upper;    // nearest active neighbours (MPI_PROC_NULL if none)
    MPI_Comm comm;
};

inline size_t lidx(const Domain& d, const size_t x, const size_t y, const size_t z) noexcept {
    return z * d.sz + y * d.sy + x;
}

// Fill the X/Y ghost layers by clamping to the adjacent interior value.
// This exactly reproduces the original boundary treatment where an out-of-range
// index is replaced by the cell itself.
void fillXYHalos(std::vector<double>& c, const Domain& d) {
    const size_t lx = d.lx;
    const size_t ly = d.ly;
    const size_t lz = d.lz;

    for (size_t z = 1; z <= lz; ++z) {
        const size_t zb = z * d.sz;
        for (size_t y = 1; y <= ly; ++y) {
            const size_t row = zb + y * d.sy;
            c[row] = c[row + 1];
            c[row + lx + 1] = c[row + lx];
        }
        for (size_t x = 1; x <= lx; ++x) {
            c[zb + x] = c[zb + d.sy + x];
            c[zb + (ly + 1) * d.sy + x] = c[zb + ly * d.sy + x];
        }
        // Corners are never read by the 7-point stencil, but keep the whole
        // ghost plane well-defined so it can be sent as one contiguous block.
        c[zb] = c[zb + d.sy + 1];
        c[zb + lx + 1] = c[zb + d.sy + lx];
        c[zb + (ly + 1) * d.sy] = c[zb + ly * d.sy + 1];
        c[zb + (ly + 1) * d.sy + lx + 1] = c[zb + ly * d.sy + lx];
    }
}

// Exchange the Z ghost layers.  At the global boundaries the ghost plane is a
// copy of the adjacent interior plane, matching the clamped boundary condition.
void exchangeZHalos(std::vector<double>& c, const Domain& d) {
    if (d.lz == 0) {
        return;
    }

    const int nplane = static_cast<int>(d.sz);

    MPI_Sendrecv(&c[d.sz], nplane, MPI_DOUBLE, d.lower, 0,
                 &c[0], nplane, MPI_DOUBLE, d.lower, 1,
                 d.comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(&c[d.lz * d.sz], nplane, MPI_DOUBLE, d.upper, 1,
                 &c[(d.lz + 1) * d.sz], nplane, MPI_DOUBLE, d.upper, 0,
                 d.comm, MPI_STATUS_IGNORE);

    if (d.lower == MPI_PROC_NULL) {
        std::copy_n(&c[d.sz], d.sz, &c[0]);
    }
    if (d.upper == MPI_PROC_NULL) {
        std::copy_n(&c[d.lz * d.sz], d.sz, &c[(d.lz + 1) * d.sz]);
    }
}

// Compute chemical potential on the local interior.
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const Domain& d,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const size_t sy = d.sy;
    const size_t sz = d.sz;

    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t zb = z * sz;
        for (size_t y = 1; y <= d.ly; ++y) {
            size_t i = zb + y * sy + 1;
            const size_t end = i + d.lx;
            for (; i < end; ++i) {
                const double cv = c[i];
                const double lap = (c[i - 1] + c[i + 1] - 2.0 * cv) / dx2
                                 + (c[i - sy] + c[i + sy] - 2.0 * cv) / dy2
                                 + (c[i - sz] + c[i + sz] - 2.0 * cv) / dz2;

                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                      + 3.0 * cv + cv * cv * cv
                      - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update on the local interior.
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const Domain& d,
                        const double D, const double dt,
                        const double dx, const double dy, const double dz) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const size_t sy = d.sy;
    const size_t sz = d.sz;

    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t zb = z * sz;
        for (size_t y = 1; y <= d.ly; ++y) {
            size_t i = zb + y * sy + 1;
            const size_t end = i + d.lx;
            for (; i < end; ++i) {
                const double mv = mu[i];
                const double lap = (mu[i - 1] + mu[i + 1] - 2.0 * mv) / dx2
                                 + (mu[i - sy] + mu[i + sy] - 2.0 * mv) / dy2
                                 + (mu[i - sz] + mu[i + sz] - 2.0 * mv) / dz2;

                cnew[i] = cold[i] + dt * D * lap;
            }
        }
    }
}

// Initialize the local interior using the same global pseudo-random formula as
// the original serial implementation.
void initializeConcentration(std::vector<double>& c, const Domain& d) {
    const size_t vol = d.nx * d.ny * d.nz;
    const size_t nxy = d.nx * d.ny;

    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t gz = d.z0 + z - 1;
        const size_t zb = z * d.sz;
        for (size_t y = 1; y <= d.ly; ++y) {
            size_t i = zb + y * d.sy + 1;
            const size_t gy = y - 1;
            for (size_t x = 1; x <= d.lx; ++x, ++i) {
                const size_t linear_id = gz * nxy + gy * d.nx + (x - 1);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[i] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

// Gather the local interior blocks on rank 0, reconstructing the full global
// field in the original z-major, y, x order.
std::vector<double> gatherField(const std::vector<double>& c, const Domain& d) {
    const size_t local_count = d.lx * d.ly * d.lz;
    std::vector<double> local(local_count);

    size_t n = 0;
    for (size_t z = 1; z <= d.lz; ++z) {
        const size_t zb = z * d.sz;
        for (size_t y = 1; y <= d.ly; ++y) {
            const size_t row = zb + y * d.sy + 1;
            for (size_t x = 0; x < d.lx; ++x) {
                local[n++] = c[row + x];
            }
        }
    }

    std::vector<double> global;
    if (d.rank == 0) {
        global.resize(d.nx * d.ny * d.nz);

        std::vector<int> counts(d.size);
        std::vector<int> displs(d.size);
        int offset = 0;
        for (int r = 0; r < d.size; ++r) {
            const size_t rz0 = static_cast<size_t>(r) * d.nz / d.size;
            const size_t rz1 = static_cast<size_t>(r + 1) * d.nz / d.size;
            const int cnt = static_cast<int>(d.nx * d.ny * (rz1 - rz0));
            counts[r] = cnt;
            displs[r] = offset;
            offset += cnt;
        }

        MPI_Gatherv(local.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    global.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, d.comm);
    } else {
        MPI_Gatherv(local.data(), static_cast<int>(local_count), MPI_DOUBLE,
                    nullptr, nullptr, nullptr, MPI_DOUBLE,
                    0, d.comm);
    }

    return global;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) {
            printf("Grid dimensions must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Build the distributed domain.
    Domain d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.lx = nx;
    d.ly = ny;
    d.z0 = static_cast<size_t>(rank) * nz / static_cast<size_t>(size);
    const size_t z1 = static_cast<size_t>(rank + 1) * nz / static_cast<size_t>(size);
    d.lz = z1 - d.z0;
    d.sx = 1;
    d.sy = nx + 2;
    d.sz = (nx + 2) * (ny + 2);
    d.total = d.sz * (d.lz + 2);
    d.rank = rank;
    d.size = size;
    d.comm = MPI_COMM_WORLD;

    // Find the nearest ranks that actually own Z planes.  This keeps the code
    // correct even if more ranks than Z planes are used.
    d.lower = MPI_PROC_NULL;
    for (int r = rank - 1; r >= 0; --r) {
        const size_t rz0 = static_cast<size_t>(r) * nz / static_cast<size_t>(size);
        const size_t rz1 = static_cast<size_t>(r + 1) * nz / static_cast<size_t>(size);
        if (rz1 > rz0) {
            d.lower = r;
            break;
        }
    }
    d.upper = MPI_PROC_NULL;
    for (int r = rank + 1; r < size; ++r) {
        const size_t rz0 = static_cast<size_t>(r) * nz / static_cast<size_t>(size);
        const size_t rz1 = static_cast<size_t>(r + 1) * nz / static_cast<size_t>(size);
        if (rz1 > rz0) {
            d.upper = r;
            break;
        }
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;

    // Allocate padded local arrays.
    std::vector<double> cold(d.total);
    std::vector<double> cnew(d.total);
    std::vector<double> mu(d.total);

    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold, d);

    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(d.comm);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Refresh the input field halos before computing mu.
        fillXYHalos(cold, d);
        exchangeZHalos(cold, d);

        // Compute chemical potential
        computeChemicalPotential(cold, mu, d, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);

        // Refresh the chemical-potential halos before the update.
        fillXYHalos(mu, d);
        exchangeZHalos(mu, d);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, d, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    const double end = MPI_Wtime();
    const double local_duration = end - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_duration, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);

    const long duration_ms = static_cast<long>(duration_seconds * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate performance
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results / validate on the root using the reconstructed global field.
    if (printResults || validate) {
        std::vector<double> global = gatherField(cold, d);

        if (rank == 0) {
            if (printResults) {
                print_results(global, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(global, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }

                int valid_int = valid ? 1 : 0;
                MPI_Bcast(&valid_int, 1, MPI_INT, 0, d.comm);
                MPI_Finalize();
                return valid_int ? 0 : 1;
            }
        } else if (validate) {
            int valid_int = 0;
            MPI_Bcast(&valid_int, 1, MPI_INT, 0, d.comm);
            MPI_Finalize();
            return valid_int ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
