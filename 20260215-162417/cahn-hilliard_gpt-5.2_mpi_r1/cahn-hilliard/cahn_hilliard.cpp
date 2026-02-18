#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

namespace {

struct DecompZ {
    size_t z_start = 0;
    size_t local_nz = 0;
};

inline DecompZ decompose_z(const size_t nz, const int rank, const int size) {
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    DecompZ d;
    d.local_nz = base + (r < rem ? 1 : 0);
    d.z_start = r * base + (r < rem ? r : rem);
    return d;
}

inline void compute_mu_range(const double* __restrict__ c, double* __restrict__ mu,
                             const size_t nx, const size_t ny, const size_t plane,
                             const size_t lz_begin, const size_t lz_end,
                             const double invdx2, const double invdy2, const double invdz2,
                             const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t lz = lz_begin; lz <= lz_end; ++lz) {
        const size_t zoff = lz * plane;
        const size_t zoff_p = (lz + 1) * plane;
        const size_t zoff_n = (lz - 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = y * nx;
            const size_t yp = (y + 1 < ny) ? (y + 1) : y;
            const size_t yn = (y > 0) ? (y - 1) : 0;
            const size_t yoff_p = yp * nx;
            const size_t yoff_n = yn * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;
                const size_t xn = (x > 0) ? (x - 1) : 0;

                const size_t idx = zoff + yoff + x;
                const double cv = c[idx];

                const double cxx = (c[zoff + yoff + xp] + c[zoff + yoff + xn] - 2.0 * cv) * invdx2;
                const double cyy = (c[zoff + yoff_p + x] + c[zoff + yoff_n + x] - 2.0 * cv) * invdy2;
                const double czz = (c[zoff_p + yoff + x] + c[zoff_n + yoff + x] - 2.0 * cv) * invdz2;
                const double lap = cxx + cyy + czz;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv -
                          gamma * lap;
            }
        }
    }
}

inline void update_range(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                         const size_t nx, const size_t ny, const size_t plane,
                         const size_t lz_begin, const size_t lz_end,
                         const double invdx2, const double invdy2, const double invdz2,
                         const double D, const double dt) {
    const double coef = dt * D;
    for (size_t lz = lz_begin; lz <= lz_end; ++lz) {
        const size_t zoff = lz * plane;
        const size_t zoff_p = (lz + 1) * plane;
        const size_t zoff_n = (lz - 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = y * nx;
            const size_t yp = (y + 1 < ny) ? (y + 1) : y;
            const size_t yn = (y > 0) ? (y - 1) : 0;
            const size_t yoff_p = yp * nx;
            const size_t yoff_n = yn * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x + 1 < nx) ? (x + 1) : x;
                const size_t xn = (x > 0) ? (x - 1) : 0;

                const size_t idx = zoff + yoff + x;
                const double muv = mu[idx];

                const double mxx = (mu[zoff + yoff + xp] + mu[zoff + yoff + xn] - 2.0 * muv) * invdx2;
                const double myy = (mu[zoff + yoff_p + x] + mu[zoff + yoff_n + x] - 2.0 * muv) * invdy2;
                const double mzz = (mu[zoff_p + yoff + x] + mu[zoff_n + yoff + x] - 2.0 * muv) * invdz2;
                const double lap = mxx + myy + mzz;

                cnew[idx] = cold[idx] + coef * lap;
            }
        }
    }
}

inline void initialize_concentration_local(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
                                           const size_t plane, const size_t z_start, const size_t local_nz) {
    const size_t vol = nx * ny * nz;
    for (size_t lz = 1; lz <= local_nz; ++lz) {
        const size_t gz = z_start + (lz - 1);
        const size_t zoff = lz * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yoff = y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = zoff + yoff + x;
                const size_t linear_id = gz * plane + yoff + x;
                const double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

} // namespace

namespace {

struct HaloReq {
    MPI_Request reqs[4];
    int nreq = 0;
};

inline HaloReq exchange_halo_z_start(std::vector<double>& a, const size_t plane, const size_t local_nz,
                                    const size_t z_start, const size_t nz, const int rank, const int size, MPI_Comm comm) {
    const bool has_prev = (rank > 0);
    const bool has_next = (rank + 1 < size);

    // Clamp at global boundaries (Neumann: replicate edge plane into ghost).
    if (z_start == 0) {
        std::memcpy(a.data(), a.data() + plane, plane * sizeof(double));
    }
    if (z_start + local_nz == nz) {
        std::memcpy(a.data() + (local_nz + 1) * plane, a.data() + local_nz * plane, plane * sizeof(double));
    }

    HaloReq hr;

    if (has_prev && z_start != 0) {
        MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 10, comm, &hr.reqs[hr.nreq++]);
    }
    if (has_next && (z_start + local_nz != nz)) {
        MPI_Irecv(a.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 11, comm,
                  &hr.reqs[hr.nreq++]);
    }

    if (has_next && (z_start + local_nz != nz)) {
        MPI_Isend(a.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 10, comm,
                  &hr.reqs[hr.nreq++]);
    }
    if (has_prev && z_start != 0) {
        MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 11, comm, &hr.reqs[hr.nreq++]);
    }

    return hr;
}

inline void exchange_halo_z_finish(HaloReq& hr) {
    if (hr.nreq) MPI_Waitall(hr.nreq, hr.reqs, MPI_STATUSES_IGNORE);
}

} // namespace

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

    if (static_cast<size_t>(size) > nz) {
        if (rank == 0) {
            printf("Error: MPI ranks (%d) must be <= nz (%zu) for z-slab decomposition.\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    const DecompZ decomp = decompose_z(nz, rank, size);
    const size_t local_nz = decomp.local_nz;
    const size_t z_start = decomp.z_start;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
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

    const size_t localSize = (local_nz + 2) * plane;
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    if (rank == 0) printf("Initializing concentration field...\n");
    initialize_concentration_local(cold.data(), nx, ny, nz, plane, z_start, local_nz);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    const double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration and overlap with interior mu computation.
        HaloReq hc = exchange_halo_z_start(cold, plane, local_nz, z_start, nz, rank, size, MPI_COMM_WORLD);

        if (local_nz > 2) {
            compute_mu_range(cold.data(), mu.data(), nx, ny, plane, 2, local_nz - 1, invdx2, invdy2, invdz2, gamma, e_AA,
                             e_BB, e_AB);
        }

        exchange_halo_z_finish(hc);

        // Boundary planes (depend on halo).
        compute_mu_range(cold.data(), mu.data(), nx, ny, plane, 1, 1, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        if (local_nz > 1) {
            compute_mu_range(cold.data(), mu.data(), nx, ny, plane, local_nz, local_nz, invdx2, invdy2, invdz2, gamma, e_AA,
                             e_BB, e_AB);
        }

        // Exchange halos for mu and overlap with interior update.
        HaloReq hm = exchange_halo_z_start(mu, plane, local_nz, z_start, nz, rank, size, MPI_COMM_WORLD);

        if (local_nz > 2) {
            update_range(cnew.data(), cold.data(), mu.data(), nx, ny, plane, 2, local_nz - 1, invdx2, invdy2, invdz2, D, dt);
        }

        exchange_halo_z_finish(hm);

        update_range(cnew.data(), cold.data(), mu.data(), nx, ny, plane, 1, 1, invdx2, invdy2, invdz2, D, dt);
        if (local_nz > 1) {
            update_range(cnew.data(), cold.data(), mu.data(), nx, ny, plane, local_nz, local_nz, invdx2, invdy2, invdz2, D,
                         dt);
        }

        std::swap(cold, cnew);
    }

    const double t1 = MPI_Wtime();
    const double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        const double cellUpdates = static_cast<double>(gridSize) * static_cast<double>(iterations);
        const double mcups = cellUpdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
        }

        const int sendcount = static_cast<int>(local_nz * plane);
        if (rank == 0) {
            for (int r = 0; r < size; ++r) {
                const DecompZ d = decompose_z(nz, r, size);
                counts[r] = static_cast<int>(d.local_nz * plane);
                displs[r] = static_cast<int>(d.z_start * plane);
            }
        }

        std::vector<double> global;
        if (rank == 0) global.resize(gridSize);

        MPI_Gatherv(cold.data() + plane, sendcount, MPI_DOUBLE, rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    if (validate) {
        int local_ok = 1;
        double local_min = cold[plane];
        double local_max = cold[plane];

        for (size_t i = plane; i < (local_nz + 1) * plane; ++i) {
            const double v = cold[i];
            if (std::isnan(v) || std::isinf(v)) {
                local_ok = 0;
                break;
            }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }

        int global_ok = 0;
        double global_min = 0.0;
        double global_max = 0.0;
        MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            if (!global_ok) {
                printf("Validation failed: found NaN or Inf value\n");
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }

            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
            if (global_max > 10.0 || global_min < -10.0) {
                printf("Validation failed: values out of expected range\n");
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }

            printf("Validation: PASSED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
