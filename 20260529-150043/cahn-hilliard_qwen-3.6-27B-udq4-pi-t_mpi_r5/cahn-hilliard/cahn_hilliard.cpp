#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// 3D index calculation (local arrays include ghost cells in Z)
// ---------------------------------------------------------------------------
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Compute Laplacian with clamped boundary conditions
// x in [0, nx), y in [0, ny), z in [0, nz_local)
// Ghost cells at z=0 and z=nz_local-1 are set by exchangeHalos/clampBoundaryGhosts
// ---------------------------------------------------------------------------
double computeLaplacian(const std::vector<double>& c,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double dx, const double dy, const double dz,
                        const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz_local - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// ---------------------------------------------------------------------------
// Compute chemical potential (interior cells only, z=1..nz_local-2)
// ---------------------------------------------------------------------------
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB,
                              const double e_AB) {
    for (size_t z = 1; z < nz_local - 1; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                          3.0 * cv + cv * cv * cv -
                          gamma * computeLaplacian(c, nx, ny, nz_local, dx, dy, dz, x, y, z);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cahn-Hilliard update step (interior cells only, z=1..nz_local-2)
// ---------------------------------------------------------------------------
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy,
                        const double dz) {
    for (size_t z = 1; z < nz_local - 1; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D *
                            computeLaplacian(mu, nx, ny, nz_local, dx, dy, dz, x, y, z);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize concentration field (interior cells only)
// ---------------------------------------------------------------------------
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz_local, const size_t z_offset,
                             const size_t global_vol) {
    for (size_t z = 1; z < nz_local - 1; ++z) {
        const size_t gz = z_offset + (z - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo =
                    ((((linear_id + 1) * 1299709) % global_vol) / static_cast<double>(global_vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Exchange ghost (halo) cells between neighboring ranks
// Each rank sends its boundary interior cells to the neighbor's ghost cells:
//   - Send first interior (z=1) to rank-1's bottom ghost
//   - Send last interior (z=nz_local-2) to rank+1's top ghost
// Global boundary ghosts are clamped (copy from adjacent interior)
// ---------------------------------------------------------------------------
void exchangeHalos(std::vector<double>& field, const size_t nx, const size_t ny,
                   const size_t nz_local, int rank, int num_ranks) {
    const size_t plane_size = nx * ny;
    MPI_Request sends[2], recvs[2];
    int num_ops = 0;

    // Top ghost (z=0): receive first interior from rank-1
    if (rank > 0) {
        // Send our first interior (z=1) to rank-1's bottom ghost
        MPI_Isend(&field[idx3(0, 0, 1, nx, ny)], static_cast<int>(plane_size), MPI_DOUBLE,
                  rank - 1, 101, MPI_COMM_WORLD, &sends[num_ops]);
        // Receive rank-1's last interior into our top ghost (z=0)
        MPI_Irecv(&field[idx3(0, 0, 0, nx, ny)], static_cast<int>(plane_size), MPI_DOUBLE,
                  rank - 1, 100, MPI_COMM_WORLD, &recvs[num_ops]);
        num_ops++;
    } else {
        // Rank 0: clamp top ghost to first interior cell
        std::memcpy(field.data(), field.data() + plane_size, plane_size * sizeof(double));
    }

    // Bottom ghost (z=nz_local-1): receive last interior from rank+1
    if (rank < num_ranks - 1) {
        // Send our last interior (z=nz_local-2) to rank+1's top ghost
        MPI_Isend(&field[idx3(0, 0, nz_local - 2, nx, ny)], static_cast<int>(plane_size),
                  MPI_DOUBLE, rank + 1, 100, MPI_COMM_WORLD, &sends[num_ops]);
        // Receive rank+1's first interior into our bottom ghost (z=nz_local-1)
        MPI_Irecv(&field[idx3(0, 0, nz_local - 1, nx, ny)], static_cast<int>(plane_size),
                  MPI_DOUBLE, rank + 1, 101, MPI_COMM_WORLD, &recvs[num_ops]);
        num_ops++;
    } else {
        // Last rank: clamp bottom ghost to last interior cell
        std::memcpy(field.data() + (nz_local - 1) * plane_size,
                    field.data() + (nz_local - 2) * plane_size,
                    plane_size * sizeof(double));
    }

    if (num_ops > 0) {
        MPI_Waitall(num_ops, sends, MPI_STATUSES_IGNORE);
        MPI_Waitall(num_ops, recvs, MPI_STATUSES_IGNORE);
    }
}

// ---------------------------------------------------------------------------
// Gather all local interior fields onto rank 0
// ---------------------------------------------------------------------------
std::vector<double> gatherField(const std::vector<double>& localField, const size_t nx,
                                const size_t ny, const size_t nz_local, int rank,
                                int num_ranks, size_t global_nz) {
    const size_t plane_size = nx * ny;
    const size_t interior_nz = nz_local - 2;
    const size_t local_interior_vol = plane_size * interior_nz;
    const size_t global_vol = nx * ny * global_nz;

    // Copy interior data to a temporary buffer
    std::vector<double> interiorBuf(local_interior_vol);
    for (size_t z = 0; z < interior_nz; ++z) {
        std::memcpy(interiorBuf.data() + z * plane_size,
                    localField.data() + (z + 1) * plane_size,
                    plane_size * sizeof(double));
    }

    // Compute per-rank counts and displacements (ranks may differ in size)
    const size_t nz_base = global_nz / num_ranks;
    const size_t nz_remainder = global_nz % num_ranks;
    std::vector<int> counts(num_ranks);
    std::vector<int> displs(num_ranks, 0);
    for (int r = 0; r < num_ranks; ++r) {
        const size_t r_interior = (r < static_cast<int>(nz_remainder)) ? nz_base + 1 : nz_base;
        counts[r] = static_cast<int>(plane_size * r_interior);
        if (r > 0) displs[r] = displs[r - 1] + counts[r - 1];
    }

    std::vector<double> globalField;
    if (rank == 0) {
        globalField.resize(global_vol);
    }

    MPI_Gatherv(interiorBuf.data(), static_cast<int>(local_interior_vol), MPI_DOUBLE,
                rank == 0 ? globalField.data() : nullptr, counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Interior cells already include global boundary cells computed with
    // ghost-cell clamping, so no post-processing needed.

    return globalField;
}

// ---------------------------------------------------------------------------
// Validate result on gathered global data
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<double>& c) {
    if (c.empty()) return true;

    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
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

// ---------------------------------------------------------------------------
// Print usage
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse identically)
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

    // Domain decomposition along Z axis
    const size_t global_nx = nx;
    const size_t global_ny = ny;
    const size_t global_nz = nz;
    const size_t global_vol = global_nx * global_ny * global_nz;

    // Distribute Z cells across ranks (handle uneven division)
    // Each rank gets nz_interior interior cells + 2 ghost cells
    const size_t nz_base = global_nz / num_ranks;
    const size_t nz_remainder = global_nz % num_ranks;
    // First nz_remainder ranks get nz_base+1 cells, rest get nz_base
    const size_t nz_interior = (rank < static_cast<int>(nz_remainder)) ? nz_base + 1 : nz_base;
    const size_t nz_local = nz_interior + 2;

    // Compute z_offset (start of this rank's interior in global Z)
    size_t z_offset = 0;
    for (int r = 0; r < rank; ++r) {
        z_offset += (r < static_cast<int>(nz_remainder)) ? nz_base + 1 : nz_base;
    }

    const size_t local_vol = global_nx * global_ny * nz_local;

    // Print info on rank 0
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", global_nx, global_ny, global_nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_ranks);
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

    // Allocate local arrays (with ghost cells)
    std::vector<double> cold(local_vol);
    std::vector<double> cnew(local_vol);
    std::vector<double> mu(local_vol);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, global_nx, global_ny, nz_local, z_offset, global_vol);

    // Initialize cnew to match cold (interior + ghosts)
    std::copy(cold.begin(), cold.end(), cnew.begin());

    // Exchange ghost cells to populate halos
    exchangeHalos(cold, global_nx, global_ny, nz_local, rank, num_ranks);
    exchangeHalos(cnew, global_nx, global_ny, nz_local, rank, num_ranks);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential (uses ghost cells for boundary)
        computeChemicalPotential(cold, mu, global_nx, global_ny, nz_local,
                                 dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange mu ghost cells
        exchangeHalos(mu, global_nx, global_ny, nz_local, rank, num_ranks);

        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, global_nx, global_ny, nz_local,
                           D, dt, dx, dy, dz);

        // Exchange cnew ghost cells
        exchangeHalos(cnew, global_nx, global_ny, nz_local, rank, num_ranks);

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        // Calculate performance (global)
        double cellUpdates = static_cast<double>(global_vol) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather field for validation / results printing
    std::vector<double> globalField =
        gatherField(cold, global_nx, global_ny, nz_local, rank, num_ranks, global_nz);

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(globalField, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalField);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
