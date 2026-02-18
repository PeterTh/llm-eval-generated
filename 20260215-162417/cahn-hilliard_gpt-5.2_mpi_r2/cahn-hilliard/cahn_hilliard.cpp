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

#if defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void exchange_halo_z(double* RESTRICT a, const size_t nx, const size_t ny, const size_t local_nz, const int rank,
                            const int size, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(double);

    const int prev = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? (rank + 1) : MPI_PROC_NULL;

    MPI_Request reqs[4];
    // Receive into ghosts
    MPI_Irecv(a + 0 * plane, static_cast<int>(plane), MPI_DOUBLE, prev, 1, comm, &reqs[0]);
    MPI_Irecv(a + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 2, comm, &reqs[1]);
    // Send interior boundary planes
    MPI_Isend(a + 1 * plane, static_cast<int>(plane), MPI_DOUBLE, prev, 2, comm, &reqs[2]);
    MPI_Isend(a + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1, comm, &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Clamped global boundary conditions at the domain ends
    if (prev == MPI_PROC_NULL) {
        std::memcpy(a + 0 * plane, a + 1 * plane, bytes);
    }
    if (next == MPI_PROC_NULL) {
        std::memcpy(a + (local_nz + 1) * plane, a + local_nz * plane, bytes);
    }
}

inline void initializeConcentration_local(double* RESTRICT c, const size_t nx, const size_t ny, const size_t nz,
                                          const size_t local_nz, const size_t z0) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z0 + (lz - 1);
        const size_t zbase_global = gz * plane;
        const size_t zbase_local = lz * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t ybase = y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = zbase_global + ybase + x;
                const double pseudo = ((((linear_id + 1) * static_cast<size_t>(1299709)) % vol) / static_cast<double>(vol));
                c[zbase_local + ybase + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

inline void computeChemicalPotential_local(const double* RESTRICT c, double* RESTRICT mu,
                                           const size_t nx, const size_t ny, const size_t local_nz,
                                           const double invdx2, const double invdy2, const double invdz2,
                                           const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;

    // Precompute constants in the algebraic term
    const double a0 = 4.5 * (e_AA - e_BB);
    const double a1 = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t zbase = lz * plane;
        const size_t zbase_m = (lz - 1) * plane;
        const size_t zbase_p = (lz + 1) * plane;

        for (size_t y = 0; y < ny; ++y) {
            const size_t ybase = y * nx;
            const size_t ybase_m = ((y > 0) ? (y - 1) : y) * nx;
            const size_t ybase_p = ((y + 1 < ny) ? (y + 1) : y) * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = (x > 0) ? (x - 1) : 0;
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;

                const size_t idx = zbase + ybase + x;
                const double cv = c[idx];

                const double cxx = (c[zbase + ybase + xp] + c[zbase + ybase + xm] - 2.0 * cv) * invdx2;
                const double cyy = (c[zbase + ybase_p + x] + c[zbase + ybase_m + x] - 2.0 * cv) * invdy2;
                const double czz = (c[zbase_p + ybase + x] + c[zbase_m + ybase + x] - 2.0 * cv) * invdz2;

                const double lap = cxx + cyy + czz;

                // 4.5*((cv+1)*e_AA + (cv-1)*e_BB - 2*cv*e_AB) + 3*cv + cv^3
                // = a0 + a1*cv + cv^3
                mu[idx] = (a0 + a1 * cv + cv * cv * cv) - gamma * lap;
            }
        }
    }
}

inline void cahnHilliardUpdate_local(double* RESTRICT cnew, const double* RESTRICT cold, const double* RESTRICT mu,
                                     const size_t nx, const size_t ny, const size_t local_nz,
                                     const double Ddt, const double invdx2, const double invdy2, const double invdz2) {
    const size_t plane = nx * ny;

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t zbase = lz * plane;
        const size_t zbase_m = (lz - 1) * plane;
        const size_t zbase_p = (lz + 1) * plane;

        for (size_t y = 0; y < ny; ++y) {
            const size_t ybase = y * nx;
            const size_t ybase_m = ((y > 0) ? (y - 1) : y) * nx;
            const size_t ybase_p = ((y + 1 < ny) ? (y + 1) : y) * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = (x > 0) ? (x - 1) : 0;
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;

                const size_t idx = zbase + ybase + x;
                const double muv = mu[idx];

                const double mxx = (mu[zbase + ybase + xp] + mu[zbase + ybase + xm] - 2.0 * muv) * invdx2;
                const double myy = (mu[zbase + ybase_p + x] + mu[zbase + ybase_m + x] - 2.0 * muv) * invdy2;
                const double mzz = (mu[zbase_p + ybase + x] + mu[zbase_m + ybase + x] - 2.0 * muv) * invdz2;

                cnew[idx] = cold[idx] + Ddt * (mxx + myy + mzz);
            }
        }
    }
}

bool validateResult_mpi(const double* RESTRICT c, const size_t nx, const size_t ny, const size_t local_nz,
                        MPI_Comm comm, const int rank) {
    const size_t plane = nx * ny;

    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t zbase = lz * plane;
        for (size_t i = 0; i < plane; ++i) {
            const double v = c[zbase + i];
            if (std::isnan(v) || std::isinf(v)) {
                local_bad = 1;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }
    }

    int global_bad = 0;
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);

    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (global_bad) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    return global_bad == 0;
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

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) printf("Error: nz (%zu) must be >= MPI ranks (%d)\n", nz, size);
        MPI_Finalize();
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t gridSize = nx * ny * nz;

    // Z-slab decomposition
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t z0 = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d\n", size);
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

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double Ddt = D * dt;

    // Local arrays with 2 Z ghost planes
    const size_t local_size = plane * (local_nz + 2);
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration_local(cold.data(), nx, ny, nz, local_nz, z0);
    exchange_halo_z(cold.data(), nx, ny, local_nz, rank, size, MPI_COMM_WORLD);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for c before computing mu (needed for Z Laplacian)
        exchange_halo_z(cold.data(), nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        computeChemicalPotential_local(cold.data(), mu.data(), nx, ny, local_nz, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);

        // Halo exchange for mu before updating c
        exchange_halo_z(mu.data(), nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        cahnHilliardUpdate_local(cnew.data(), cold.data(), mu.data(), nx, ny, local_nz, Ddt, invdx2, invdy2, invdz2);

        std::swap(cold, cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    const double local_sec = std::chrono::duration<double>(end - start).count();
    double max_sec = 0.0;
    MPI_Reduce(&local_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_sec * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_sec / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        // Gather full field to rank 0 for consistent hashing/statistics
        std::vector<double> global;
        if (rank == 0) global.resize(gridSize);

        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = base + ((static_cast<size_t>(r) < rem) ? 1 : 0);
                const size_t r_z0 = base * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), rem);
                const size_t elems = r_local_nz * plane;
                if (elems > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                    r_z0 * plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    printf("Error: problem size too large for MPI_Gatherv int counts\n");
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                counts[r] = static_cast<int>(elems);
                displs[r] = static_cast<int>(r_z0 * plane);
            }
        }

        MPI_Gatherv(cold.data() + 1 * plane, static_cast<int>(local_nz * plane), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool ok = validateResult_mpi(cold.data(), nx, ny, local_nz, MPI_COMM_WORLD, rank);
        int ok_i = ok ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok_i, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (rank == 0) {
            if (all_ok) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return all_ok ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
