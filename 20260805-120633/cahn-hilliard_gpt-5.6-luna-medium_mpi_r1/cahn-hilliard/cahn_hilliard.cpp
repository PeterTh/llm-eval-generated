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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                        [[maybe_unused]] const size_t stride_z, const double inv_dx2, const double inv_dy2,
                        const double inv_dz2, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = z + 1;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = z - 1;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)] ) * inv_dx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)] ) * inv_dy2;
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)] ) * inv_dz2;
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              [[maybe_unused]] const size_t stride_z, const double inv_dx2, const double inv_dy2,
                              const double inv_dz2,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, ny, inv_dx2, inv_dy2, inv_dz2, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        [[maybe_unused]] const size_t stride_z, const double D, const double dt,
                        const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, ny, inv_dx2, inv_dy2, inv_dz2, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t global_z0, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = (global_z0 + z - 1) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz,
                    [[maybe_unused]] const size_t stride_z, MPI_Comm comm, const int rank) {
    // Check for NaN or Inf
    int localFinite = 1;
    for (size_t z = 1; z <= local_nz; ++z) {
      for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
        const auto val = c[idx3(x, y, z, nx, ny)];
        if (std::isnan(val) || std::isinf(val)) {
            localFinite = 0;
        }
      }
    }
    int finite = 0;
    MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_LAND, comm);
    if (!finite) { if (rank == 0) printf("Validation failed: found NaN or Inf value\n"); return false; }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();
    for (size_t z = 1; z <= local_nz; ++z) for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
        const double val = c[idx3(x, y, z, nx, ny)];
        minVal = std::min(minVal, val); maxVal = std::max(maxVal, val);
    }
    double globalMin, globalMax;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    
    // Values should generally stay within reasonable bounds
    const bool localInRange = globalMax <= 10.0 && globalMin >= -10.0;
    if (!localInRange) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

void exchangeHalos(std::vector<double>& field, const size_t plane, const size_t local_nz,
                   const int rank, const int active_ranks, MPI_Comm comm) {
    const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank + 1 < active_ranks) ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 101,
                 field.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 101,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(field.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 102,
                 field.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 102,
                 comm, MPI_STATUS_IGNORE);

    // A clamped boundary duplicates the nearest physical plane.
    if (rank == 0) std::copy_n(field.data() + plane, plane, field.data());
    if (rank + 1 == active_ranks) {
        std::copy_n(field.data() + local_nz * plane, plane,
                    field.data() + (local_nz + 1) * plane);
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
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse identically).
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
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        if (world_rank == 0) printf("Grid dimensions must be positive and time steps non-negative\n");
        MPI_Finalize();
        return 1;
    }

    // Only ranks with work join the active communicator. This also permits more
    // MPI ranks than Z planes without allocating empty domains.
    const int active_ranks = std::min<size_t>(static_cast<size_t>(world_size), nz);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_ranks ? 0 : MPI_UNDEFINED,
                   world_rank, &comm);
    if (world_rank >= active_ranks) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const size_t global_z0 = (nz * static_cast<size_t>(rank)) / active_ranks;
    const size_t global_z1 = (nz * static_cast<size_t>(rank + 1)) / active_ranks;
    const size_t local_nz = global_z1 - global_z0;
    const size_t plane = nx * ny;
    const size_t stride_z = local_nz + 2;
    const size_t localSize = plane * stride_z;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", active_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Physical parameters
    const double inv_dx2 = 1.0;
    const double inv_dy2 = 1.0;
    const double inv_dz2 = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    
    const size_t gridSize = nx * ny * nz;
    std::vector<double> cold(localSize, 0.0);
    std::vector<double> cnew(localSize, 0.0);
    std::vector<double> mu(localSize, 0.0);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, local_nz, global_z0, nz);
    exchangeHalos(cold, plane, local_nz, rank, active_ranks, comm);
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, plane, local_nz, rank, active_ranks, comm);
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, local_nz, stride_z,
                                inv_dx2, inv_dy2, inv_dz2,
                                gamma, e_AA, e_BB, e_AB);
        exchangeHalos(mu, plane, local_nz, rank, active_ranks, comm);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz, stride_z, D, dt,
                           inv_dx2, inv_dy2, inv_dz2);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    const double local_time = MPI_Wtime() - start;
    double duration_seconds = 0.0;
    MPI_Reduce(&local_time, &duration_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) printf("Computation time: %.0f ms\n", duration_seconds * 1000.0);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    if (rank == 0) {
        const double mcups = cellUpdates / std::max(duration_seconds, 1.0e-9) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Print results for external validation
    std::vector<double> global;
    std::vector<int> counts;
    std::vector<int> displacements;
    const int localCount = static_cast<int>(local_nz * plane);
    if (printResults && rank == 0) {
        global.resize(gridSize);
        counts.resize(active_ranks);
        displacements.resize(active_ranks);
        for (int r = 0; r < active_ranks; ++r) {
            const size_t z0 = (nz * static_cast<size_t>(r)) / active_ranks;
            const size_t z1 = (nz * static_cast<size_t>(r + 1)) / active_ranks;
            counts[r] = static_cast<int>((z1 - z0) * plane);
            displacements[r] = static_cast<int>(z0 * plane);
        }
    }
    if (printResults) MPI_Gatherv(cold.data() + plane, localCount, MPI_DOUBLE,
                                  rank == 0 ? global.data() : nullptr,
                                  rank == 0 ? counts.data() : nullptr,
                                  rank == 0 ? displacements.data() : nullptr,
                                  MPI_DOUBLE, 0, comm);
    if (printResults && rank == 0) print_results(global, "Concentration");
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, local_nz, stride_z, comm, rank);
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Comm_free(&comm);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Comm_free(&comm);
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return 0;
}
