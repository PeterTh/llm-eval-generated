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

// 3D index calculation (z-major), works for arrays that include halo planes in z.
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct DecompZ {
    size_t z0;
    size_t nz_local;
};

static inline DecompZ decompose_z(const size_t nz_global, const int rank, const int nranks) noexcept {
    const size_t P = static_cast<size_t>(nranks);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = nz_global / P;
    const size_t rem = nz_global % P;

    const size_t nz_local = base + (r < rem ? 1u : 0u);
    const size_t z0 = r * base + (r < rem ? r : rem);
    return {z0, nz_local};
}

struct HaloExchange {
    MPI_Request reqs[4];
    int nreq = 0;
};

static inline HaloExchange start_exchange_z_halos(double* a, const int plane_elems, const size_t nz_local,
                                                  const int prev, const int next, MPI_Comm comm) {
    HaloExchange hx;

    // Receive halos first.
    if (prev != MPI_PROC_NULL) {
        MPI_Irecv(a + 0ll * plane_elems, plane_elems, MPI_DOUBLE, prev, 100, comm, &hx.reqs[hx.nreq++]);
    } else {
        std::memcpy(a + 0ll * plane_elems, a + 1ll * plane_elems, static_cast<size_t>(plane_elems) * sizeof(double));
    }

    if (next != MPI_PROC_NULL) {
        MPI_Irecv(a + static_cast<long long>(nz_local + 1) * plane_elems, plane_elems, MPI_DOUBLE, next, 101, comm,
                  &hx.reqs[hx.nreq++]);
    } else {
        std::memcpy(a + static_cast<long long>(nz_local + 1) * plane_elems, a + static_cast<long long>(nz_local) * plane_elems,
                    static_cast<size_t>(plane_elems) * sizeof(double));
    }

    // Send boundary planes.
    if (prev != MPI_PROC_NULL) {
        MPI_Isend(a + 1ll * plane_elems, plane_elems, MPI_DOUBLE, prev, 101, comm, &hx.reqs[hx.nreq++]);
    }
    if (next != MPI_PROC_NULL) {
        MPI_Isend(a + static_cast<long long>(nz_local) * plane_elems, plane_elems, MPI_DOUBLE, next, 100, comm,
                  &hx.reqs[hx.nreq++]);
    }

    return hx;
}

static inline void wait_exchange(HaloExchange& hx) {
    if (hx.nreq > 0) {
        MPI_Waitall(hx.nreq, hx.reqs, MPI_STATUSES_IGNORE);
    }
}

static inline void compute_mu_range(double* __restrict mu, const double* __restrict c,
                                    const size_t nx, const size_t ny, const size_t z_begin, const size_t z_end,
                                    const double invdx2, const double invdy2, const double invdz2,
                                    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    if (z_begin > z_end) return;

    const size_t plane = nx * ny;

    for (size_t z = z_begin; z <= z_end; ++z) {
        const size_t zoff = z * plane;
        const size_t zoffp = (z + 1) * plane;
        const size_t zoffn = (z - 1) * plane;

        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = zoff + y * nx;
            const size_t yoffp = zoff + ((y + 1 < ny) ? (y + 1) : y) * nx;
            const size_t yoffn = zoff + ((y > 0) ? (y - 1) : y) * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = (x > 0) ? (x - 1) : x;
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;

                const double cv = c[yoff + x];
                const double c_xx = (c[yoff + xp] + c[yoff + xm] - 2.0 * cv) * invdx2;
                const double c_yy = (c[yoffp + x] + c[yoffn + x] - 2.0 * cv) * invdy2;
                const double c_zz = (c[zoffp + y * nx + x] + c[zoffn + y * nx + x] - 2.0 * cv) * invdz2;

                const double lap = c_xx + c_yy + c_zz;

                mu[yoff + x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv +
                               cv * cv * cv - gamma * lap;
            }
        }
    }
}

static inline void update_c_range(double* __restrict cnew, const double* __restrict cold, const double* __restrict mu,
                                  const size_t nx, const size_t ny, const size_t z_begin, const size_t z_end,
                                  const double invdx2, const double invdy2, const double invdz2, const double dtD) {
    if (z_begin > z_end) return;

    const size_t plane = nx * ny;

    for (size_t z = z_begin; z <= z_end; ++z) {
        const size_t zoff = z * plane;
        const size_t zoffp = (z + 1) * plane;
        const size_t zoffn = (z - 1) * plane;

        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = zoff + y * nx;
            const size_t yoffp = zoff + ((y + 1 < ny) ? (y + 1) : y) * nx;
            const size_t yoffn = zoff + ((y > 0) ? (y - 1) : y) * nx;

            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = (x > 0) ? (x - 1) : x;
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;

                const double mv = mu[yoff + x];
                const double m_xx = (mu[yoff + xp] + mu[yoff + xm] - 2.0 * mv) * invdx2;
                const double m_yy = (mu[yoffp + x] + mu[yoffn + x] - 2.0 * mv) * invdy2;
                const double m_zz = (mu[zoffp + y * nx + x] + mu[zoffn + y * nx + x] - 2.0 * mv) * invdz2;

                cnew[yoff + x] = cold[yoff + x] + dtD * (m_xx + m_yy + m_zz);
            }
        }
    }
}

static inline void initialize_concentration(double* cold, const size_t nx, const size_t ny, const size_t nz_global,
                                            const size_t z0, const size_t nz_local) {
    const size_t vol = nx * ny * nz_global;
    const size_t plane = nx * ny;

    for (size_t lz = 0; lz < nz_local; ++lz) {
        const size_t gz = z0 + lz;
        const size_t zoff = (lz + 1) * plane;

        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = zoff + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = gz * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709ull) % vol) / static_cast<double>(vol));
                cold[yoff + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

static inline bool validate_result_mpi(const double* cold, const size_t nx, const size_t ny, const size_t nz_local,
                                      MPI_Comm comm, const int rank) {
    const size_t plane = nx * ny;

    int local_invalid = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    for (size_t z = 1; z <= nz_local; ++z) {
        const size_t zoff = z * plane;
        for (size_t i = 0; i < plane; ++i) {
            const double v = cold[zoff + i];
            if (std::isnan(v) || std::isinf(v)) {
                local_invalid = 1;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }
    }

    int any_invalid = 0;
    MPI_Allreduce(&local_invalid, &any_invalid, 1, MPI_INT, MPI_MAX, comm);

    double global_min = 0.0;
    double global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        if (any_invalid) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }

        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
    }

    int valid_int = 1;
    if (rank == 0) {
        valid_int = (!any_invalid && global_max <= 10.0 && global_min >= -10.0) ? 1 : 0;
    }
    MPI_Bcast(&valid_int, 1, MPI_INT, 0, comm);
    return valid_int == 1;
}

static void printUsage(const char* progName) {
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

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Parameters parsed/broadcast from rank 0 for consistency.
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    int validate = 0;
    int printResults = 0;
    int showHelp = 0;
    int parseError = 0;

    if (world_rank == 0) {
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
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseError = 1;
                break;
            }
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;

        if (showHelp) {
            printUsage(argv[0]);
        } else if (parseError) {
            printUsage(argv[0]);
        }
    }

    // Broadcast parameters.
    unsigned long long params_u64[3] = {static_cast<unsigned long long>(nx), static_cast<unsigned long long>(ny),
                                        static_cast<unsigned long long>(nz)};
    int params_i32[4] = {iterations, validate, printResults, showHelp};
    MPI_Bcast(params_u64, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(params_i32, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseError, 1, MPI_INT, 0, MPI_COMM_WORLD);

    nx = static_cast<size_t>(params_u64[0]);
    ny = static_cast<size_t>(params_u64[1]);
    nz = static_cast<size_t>(params_u64[2]);
    iterations = params_i32[0];
    validate = params_i32[1];
    printResults = params_i32[2];
    showHelp = params_i32[3];

    int exit_code = 0;
    if (showHelp) {
        MPI_Finalize();
        return 0;
    }
    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    // Use only as many ranks as nz slices; extra ranks are idle.
    const int active_size = std::min(world_size, static_cast<int>(nz));
    const int active = (world_rank < active_size) ? 1 : 0;

    MPI_Comm active_comm = MPI_COMM_NULL;
    if (active) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &active_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, world_rank, &active_comm);
    }

    if (!active) {
        if (world_rank == 0 && active_size < world_size) {
            printf("Note: nz=%zu so only %d/%d MPI ranks will participate.\n", nz, active_size, world_size);
        }
        // Still participate in final exit-code broadcast below.
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exit_code;
    }

    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &nranks);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", nranks);
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
    const double dtD = dt * D;

    const size_t plane = nx * ny;
    if (plane > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Error: nx*ny too large for MPI count\n");
        }
        exit_code = 1;
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exit_code;
    }
    const int plane_elems = static_cast<int>(plane);

    const DecompZ d = decompose_z(nz, rank, nranks);
    const size_t z0 = d.z0;
    const size_t nz_local = d.nz_local;

    // Allocate with halo planes in z: [0] and [nz_local+1] are halos.
    std::vector<double> cold((nz_local + 2) * plane);
    std::vector<double> cnew((nz_local + 2) * plane);
    std::vector<double> mu((nz_local + 2) * plane);

    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initialize_concentration(cold.data(), nx, ny, nz, z0, nz_local);

    // Ensure halos are valid before first step.
    const int prev = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
    const int next = (rank + 1 < nranks) ? (rank + 1) : MPI_PROC_NULL;
    {
        HaloExchange hx = start_exchange_z_halos(cold.data(), plane_elems, nz_local, prev, next, active_comm);
        wait_exchange(hx);
    }

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(active_comm);
    const double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos and overlap with interior mu computation.
        HaloExchange hc = start_exchange_z_halos(cold.data(), plane_elems, nz_local, prev, next, active_comm);

        if (nz_local > 2) {
            compute_mu_range(mu.data(), cold.data(), nx, ny, 2, nz_local - 1, invdx2, invdy2, invdz2, gamma, e_AA, e_BB,
                             e_AB);
        }

        wait_exchange(hc);

        // Boundary planes (need halos): z=1 and z=nz_local.
        compute_mu_range(mu.data(), cold.data(), nx, ny, 1, 1, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        if (nz_local >= 2) {
            compute_mu_range(mu.data(), cold.data(), nx, ny, nz_local, nz_local, invdx2, invdy2, invdz2, gamma, e_AA, e_BB,
                             e_AB);
        }

        // Exchange mu halos and overlap with interior update.
        HaloExchange hm = start_exchange_z_halos(mu.data(), plane_elems, nz_local, prev, next, active_comm);

        if (nz_local > 2) {
            update_c_range(cnew.data(), cold.data(), mu.data(), nx, ny, 2, nz_local - 1, invdx2, invdy2, invdz2, dtD);
        }

        wait_exchange(hm);

        update_c_range(cnew.data(), cold.data(), mu.data(), nx, ny, 1, 1, invdx2, invdy2, invdz2, dtD);
        if (nz_local >= 2) {
            update_c_range(cnew.data(), cold.data(), mu.data(), nx, ny, nz_local, nz_local, invdx2, invdy2, invdz2, dtD);
        }

        std::swap(cold, cnew);
    }

    MPI_Barrier(active_comm);
    const double t1 = MPI_Wtime();
    const double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Allreduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, active_comm);

    if (rank == 0) {
        const long long ms = static_cast<long long>(max_time * 1000.0);
        printf("Computation time: %lld ms\n", ms);

        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz);
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results for external validation (gather to rank 0).
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> global;

        if (rank == 0) {
            counts.resize(nranks);
            displs.resize(nranks);
            for (int r = 0; r < nranks; ++r) {
                const DecompZ dr = decompose_z(nz, r, nranks);
                const unsigned long long c = static_cast<unsigned long long>(dr.nz_local) * static_cast<unsigned long long>(plane);
                const unsigned long long dpl = static_cast<unsigned long long>(dr.z0) * static_cast<unsigned long long>(plane);
                if (c > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                    dpl > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                    printf("Error: problem size too large for MPI_Gatherv counts/displs\n");
                    exit_code = 1;
                    break;
                }
                counts[r] = static_cast<int>(c);
                displs[r] = static_cast<int>(dpl);
            }

            if (exit_code == 0) {
                global.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz));
            }
        }

        MPI_Bcast(&exit_code, 1, MPI_INT, 0, active_comm);
        if (exit_code != 0) {
            MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return exit_code;
        }

        MPI_Gatherv(cold.data() + plane, static_cast<int>(nz_local * plane), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, active_comm);

        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }

        const bool valid = validate_result_mpi(cold.data(), nx, ny, nz_local, active_comm, rank);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exit_code = valid ? 0 : 1;
    }

    MPI_Comm_free(&active_comm);

    // Ensure all MPI ranks (including idle) return the same exit code.
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
