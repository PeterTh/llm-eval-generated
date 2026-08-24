#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential for a z-slab range [lz_start, lz_end]
// Ghost layers handle z-boundary; x/y use clamped BC
void computeChemPotSlab(const double* c, double* mu,
                        const size_t nx, const size_t ny,
                        const size_t lz_start, const size_t lz_end,
                        const double dx, const double dy, const double dz,
                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t lz = lz_start; lz <= lz_end; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, lz, nx, ny);
                const double cv = c[i];

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : x;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t yn = (y > 0) ? y - 1 : y;

                const double cxx = (c[idx3(xp, y, lz, nx, ny)] + c[idx3(xn, y, lz, nx, ny)] - 2.0 * cv) / (dx * dx);
                const double cyy = (c[idx3(x, yp, lz, nx, ny)] + c[idx3(x, yn, lz, nx, ny)] - 2.0 * cv) / (dy * dy);
                const double czz = (c[idx3(x, y, lz + 1, nx, ny)] + c[idx3(x, y, lz - 1, nx, ny)] - 2.0 * cv) / (dz * dz);

                mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update for a z-slab range [lz_start, lz_end]
void updateSlab(double* cnew, const double* cold, const double* mu,
                const size_t nx, const size_t ny,
                const size_t lz_start, const size_t lz_end,
                const double D, const double dt,
                const double dx, const double dy, const double dz) {
    for (size_t lz = lz_start; lz <= lz_end; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = idx3(x, y, lz, nx, ny);

                const size_t xp = (x < nx - 1) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : x;
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t yn = (y > 0) ? y - 1 : y;

                const double mxx = (mu[idx3(xp, y, lz, nx, ny)] + mu[idx3(xn, y, lz, nx, ny)] - 2.0 * mu[i]) / (dx * dx);
                const double myy = (mu[idx3(x, yp, lz, nx, ny)] + mu[idx3(x, yn, lz, nx, ny)] - 2.0 * mu[i]) / (dy * dy);
                const double mzz = (mu[idx3(x, y, lz + 1, nx, ny)] + mu[idx3(x, y, lz - 1, nx, ny)] - 2.0 * mu[i]) / (dz * dz);

                cnew[i] = cold[i] + dt * D * (mxx + myy + mzz);
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Decompose z-dimension across MPI ranks
    const size_t base_nz = nz / static_cast<size_t>(nprocs);
    const size_t rem = nz % static_cast<size_t>(nprocs);
    size_t local_nz, z_start;
    if (static_cast<size_t>(rank) < rem) {
        local_nz = base_nz + 1;
        z_start = static_cast<size_t>(rank) * local_nz;
    } else {
        local_nz = base_nz;
        z_start = rem * (base_nz + 1) + (static_cast<size_t>(rank) - rem) * base_nz;
    }

    const size_t plane_size = nx * ny;
    // Local arrays: owned planes + 2 ghost planes (one on each z-side)
    const size_t local_total = plane_size * (local_nz + 2);

    std::vector<double> cold_local(local_total, 0.0);
    std::vector<double> cnew_local(local_total, 0.0);
    std::vector<double> mu_local(local_total, 0.0);

    // Initialize owned cells (local z indices 1..local_nz map to global z_start..z_start+local_nz-1)
    const size_t vol = nx * ny * nz;
    if (rank == 0) printf("Initializing concentration field...\n");

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local_idx = idx3(x, y, lz, nx, ny);
                const size_t linear_id = gz * plane_size + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cold_local[local_idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Determine neighbors (ranks with local_nz==0 are skipped)
    const int active_count = static_cast<int>(std::min(static_cast<size_t>(nprocs), nz));
    const int rank_prev = (rank > 0 && rank < active_count) ? rank - 1 : MPI_PROC_NULL;
    const int rank_next = (rank + 1 < active_count) ? rank + 1 : MPI_PROC_NULL;
    const bool is_first = (rank_prev == MPI_PROC_NULL && local_nz > 0);
    const bool is_last  = (rank_next == MPI_PROC_NULL && local_nz > 0);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int ps_int = static_cast<int>(plane_size);

    for (int t = 0; t < iterations; ++t) {
        if (local_nz > 0) {
            // --- Phase 1: Chemical potential from cold ---
            MPI_Request reqs[4];
            int nreqs = 0;

            // Exchange ghost layers of cold
            MPI_Irecv(cold_local.data(), ps_int, MPI_DOUBLE,
                      rank_prev, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(cold_local.data() + plane_size, ps_int, MPI_DOUBLE,
                      rank_prev, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(cold_local.data() + (local_nz + 1) * plane_size, ps_int, MPI_DOUBLE,
                      rank_next, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(cold_local.data() + local_nz * plane_size, ps_int, MPI_DOUBLE,
                      rank_next, 1, MPI_COMM_WORLD, &reqs[nreqs++]);

            // Overlap: compute interior slabs that don't depend on ghosts
            if (local_nz > 2) {
                computeChemPotSlab(cold_local.data(), mu_local.data(), nx, ny,
                                   2, local_nz - 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

            // Fill clamped BC ghost planes for domain boundaries
            if (is_first) {
                std::copy(cold_local.data() + plane_size,
                          cold_local.data() + 2 * plane_size, cold_local.data());
            }
            if (is_last) {
                std::copy(cold_local.data() + local_nz * plane_size,
                          cold_local.data() + (local_nz + 1) * plane_size,
                          cold_local.data() + (local_nz + 1) * plane_size);
            }

            // Compute boundary slabs (need ghost data)
            computeChemPotSlab(cold_local.data(), mu_local.data(), nx, ny,
                               1, 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            if (local_nz > 1) {
                computeChemPotSlab(cold_local.data(), mu_local.data(), nx, ny,
                                   local_nz, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }

            // --- Phase 2: Update concentration from mu ---
            nreqs = 0;

            MPI_Irecv(mu_local.data(), ps_int, MPI_DOUBLE,
                      rank_prev, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(mu_local.data() + plane_size, ps_int, MPI_DOUBLE,
                      rank_prev, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(mu_local.data() + (local_nz + 1) * plane_size, ps_int, MPI_DOUBLE,
                      rank_next, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Isend(mu_local.data() + local_nz * plane_size, ps_int, MPI_DOUBLE,
                      rank_next, 1, MPI_COMM_WORLD, &reqs[nreqs++]);

            if (local_nz > 2) {
                updateSlab(cnew_local.data(), cold_local.data(), mu_local.data(),
                           nx, ny, 2, local_nz - 1, D, dt, dx, dy, dz);
            }

            MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

            if (is_first) {
                std::copy(mu_local.data() + plane_size,
                          mu_local.data() + 2 * plane_size, mu_local.data());
            }
            if (is_last) {
                std::copy(mu_local.data() + local_nz * plane_size,
                          mu_local.data() + (local_nz + 1) * plane_size,
                          mu_local.data() + (local_nz + 1) * plane_size);
            }

            updateSlab(cnew_local.data(), cold_local.data(), mu_local.data(),
                       nx, ny, 1, 1, D, dt, dx, dy, dz);
            if (local_nz > 1) {
                updateSlab(cnew_local.data(), cold_local.data(), mu_local.data(),
                           nx, ny, local_nz, local_nz, D, dt, dx, dy, dz);
            }
        }

        std::swap(cold_local, cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        double cellUpdates = static_cast<double>(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (max_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full array on rank 0 for output/validation
    if (printResults || validate) {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        int local_count = static_cast<int>(local_nz * plane_size);
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }

        std::vector<double> global_c;
        if (rank == 0) {
            global_c.resize(nx * ny * nz);
        }

        MPI_Gatherv(cold_local.data() + plane_size, local_count, MPI_DOUBLE,
                     global_c.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global_c, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global_c, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
