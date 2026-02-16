#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation for packed slices
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local index for arrays with ghost layers in Z (ghost layers at z=0 and z=local_nz+1)
inline constexpr size_t idx3_local(const size_t x, const size_t y, const size_t z_local, const size_t nx, const size_t ny) noexcept {
    return z_local * (nx * ny) + y * nx + x;
}

// Compute Laplacian on local array using clamped/halo values
inline double computeLaplacian_local(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t local_nz_with_ghosts,
                                     const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z_local) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z_local + 1; // safe because ghost layers included
    const size_t zn = z_local - 1; // safe because ghost layers included

    const double center = c[idx3_local(x, y, z_local, nx, ny)];
    const double cxx = (c[idx3_local(xp, y, z_local, nx, ny)] + c[idx3_local(xn, y, z_local, nx, ny)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3_local(x, yp, z_local, nx, ny)] + c[idx3_local(x, yn, z_local, nx, ny)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3_local(x, y, zp, nx, ny)] + c[idx3_local(x, y, zn, nx, ny)] - 2.0 * center) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local real slices (z_local in [1..local_nz])
void computeChemicalPotential_local(const std::vector<double>& c_local, std::vector<double>& mu_local,
                                    const size_t nx, const size_t ny, const size_t local_nz, const size_t local_nz_with_ghosts,
                                    const double dx, const double dy, const double dz,
                                    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z_local = 1; z_local <= local_nz; ++z_local) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z_local, nx, ny);
                const double cv = c_local[idx];
                mu_local[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv
                                - gamma * computeLaplacian_local(c_local, nx, ny, local_nz_with_ghosts, dx, dy, dz, x, y, z_local);
            }
        }
    }
}

// Cahn-Hilliard update step on local real slices
void cahnHilliardUpdate_local(std::vector<double>& cnew_local, const std::vector<double>& cold_local,
                              const std::vector<double>& mu_local,
                              const size_t nx, const size_t ny, const size_t local_nz, const size_t local_nz_with_ghosts,
                              const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z_local = 1; z_local <= local_nz; ++z_local) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z_local, nx, ny);
                cnew_local[idx] = cold_local[idx] + dt * D * computeLaplacian_local(mu_local, nx, ny, local_nz_with_ghosts, dx, dy, dz, x, y, z_local);
            }
        }
    }
}

// Initialize concentration field on local real slices using global indexing for reproducibility
void initializeConcentration_local(std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz,
                                   const size_t z_start_global) {
    const size_t vol_global = nx * ny * (size_t)local_nz; // not used for modulo randomness per-rank
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        const size_t gz = z_start_global + (zl - 1);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, zl, nx, ny);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % (nx * ny * (size_t)local_nz)) / static_cast<double>(nx * ny * (size_t)local_nz));
                c_local[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange halo layers (one layer) in Z-direction for a given array
void exchangeHalos(std::vector<double>& arr, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size, MPI_Comm comm) {
    const int prev = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int next = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const size_t slice = nx * ny;

    // pointers into vector
    double* base = arr.data();
    double* send_down = base + idx3_local(0, 0, 1, nx, ny);               // first real layer
    double* recv_up = base + idx3_local(0, 0, local_nz + 1, nx, ny);     // top ghost
    double* send_up = base + idx3_local(0, 0, local_nz, nx, ny);         // last real
    double* recv_down = base + idx3_local(0, 0, 0, nx, ny);              // bottom ghost

    MPI_Status st;
    // send first real to prev receive into prev's top ghost -> recv_down from prev
    if (prev != MPI_PROC_NULL) {
        MPI_Sendrecv(send_down, (int)slice, MPI_DOUBLE, prev, 0,
                     recv_down, (int)slice, MPI_DOUBLE, prev, 0, comm, &st);
    } else {
        // clamp: copy first real into bottom ghost
        std::memcpy(recv_down, send_down, slice * sizeof(double));
    }

    // send last real to next receive into next's bottom ghost -> recv_up from next
    if (next != MPI_PROC_NULL) {
        MPI_Sendrecv(send_up, (int)slice, MPI_DOUBLE, next, 1,
                     recv_up, (int)slice, MPI_DOUBLE, next, 1, comm, &st);
    } else {
        // clamp: copy last real into top ghost
        std::memcpy(recv_up, send_up, slice * sizeof(double));
    }
}

bool validateResult_distributed(const std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz,
                                const int rank, const int size, MPI_Comm comm) {
    const size_t slice = nx * ny;
    bool local_bad = false;
    double local_min = 0.0, local_max = 0.0;
    bool first = true;
    // only consider real layers
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        for (size_t i = 0; i < slice; ++i) {
            double v = c_local[idx3_local(0, 0, zl, nx, ny) + i];
            if (std::isnan(v) || std::isinf(v)) local_bad = true;
            if (first) { local_min = local_max = v; first = false; }
            local_min = std::min(local_min, v);
            local_max = std::max(local_max, v);
        }
    }
    int any_bad = local_bad ? 1 : 0;
    int global_bad = 0;
    MPI_Allreduce(&any_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);

    double global_min = 0.0, global_max = 0.0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    }
    if (global_bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (global_max > 10.0 || global_min < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void gather_and_print_results(const std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz,
                              const std::vector<int>& counts, const std::vector<int>& displs, const int rank, MPI_Comm comm) {
    const size_t local_real_elems = (size_t)counts[rank];
    std::vector<double> gathered;
    if (rank == 0) gathered.resize((size_t)std::accumulate(counts.begin(), counts.end(), 0));

    // Build contiguous buffer of real layers from c_local
    std::vector<double> sendbuf(local_real_elems);
    size_t idxw = 0;
    for (size_t zl = 1; zl <= local_nz; ++zl) {
        const double* base = c_local.data() + idx3_local(0, 0, zl, nx, ny);
        std::copy(base, base + nx * ny, sendbuf.begin() + idxw);
        idxw += nx * ny;
    }

    MPI_Gatherv(sendbuf.data(), (int)local_real_elems, MPI_DOUBLE,
                gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        print_results(gathered, "Concentration");
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
    int rank = 0, size = 1;
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

    const size_t slice = nx * ny;
    const size_t total_grid = nx * ny * nz;

    // Decompose Nz among ranks (slab decomposition)
    std::vector<int> counts_int(size);
    std::vector<int> displs_int(size);
    std::vector<size_t> local_nz_arr(size);
    size_t base = nz / (size_t)size;
    size_t rem = nz % (size_t)size;
    size_t z_start = 0;
    for (int r = 0; r < size; ++r) {
        size_t lnz = base + (r < (int)rem ? 1 : 0);
        local_nz_arr[r] = lnz;
        counts_int[r] = (int)(lnz * slice);
        displs_int[r] = (int)(z_start * slice);
        z_start += lnz;
    }

    const size_t local_nz = local_nz_arr[rank];
    // allocate with 2 ghost layers in Z
    const size_t local_nz_with_ghosts = local_nz + 2;
    const size_t local_elems_with_ghosts = local_nz_with_ghosts * slice;

    std::vector<double> cold_local(local_elems_with_ghosts);
    std::vector<double> cnew_local(local_elems_with_ghosts);
    std::vector<double> mu_local(local_elems_with_ghosts);

    // Compute global z start for this rank
    size_t z_start_global = 0;
    for (int r = 0; r < rank; ++r) z_start_global += local_nz_arr[r];

    if (rank == 0) printf("Initializing concentration field...\n");
    // Initialize real layers
    initializeConcentration_local(cold_local, nx, ny, local_nz, z_start_global);
    // For ghost layers, clamp copies
    // bottom ghost = first real, top ghost = last real
    std::memcpy(cold_local.data() + idx3_local(0,0,0,nx,ny), cold_local.data() + idx3_local(0,0,1,nx,ny), slice * sizeof(double));
    std::memcpy(cold_local.data() + idx3_local(0,0,local_nz+1,nx,ny), cold_local.data() + idx3_local(0,0,local_nz,nx,ny), slice * sizeof(double));

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // exchange halos for cold
        exchangeHalos(cold_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);

        // compute chemical potential on local real slices
        computeChemicalPotential_local(cold_local, mu_local, nx, ny, local_nz, local_nz_with_ghosts, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // exchange halos for mu
        exchangeHalos(mu_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);

        // update concentration on local real slices
        cahnHilliardUpdate_local(cnew_local, cold_local, mu_local, nx, ny, local_nz, local_nz_with_ghosts, D, dt, dx, dy, dz);

        // swap buffers (including ghosts) - keep ghosts for next iteration
        std::swap(cold_local, cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double duration_sec = t1 - t0;

    if (rank == 0) {
        printf("Computation time: %lld ms\n", (long long)(duration_sec * 1000.0));
        double cellUpdates = (double)total_grid * iterations;
        double mcups = cellUpdates / duration_sec / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        // prepare counts and displs as int vectors for MPI_Gatherv
        gather_and_print_results(cold_local, nx, ny, local_nz, counts_int, displs_int, rank, MPI_COMM_WORLD);
    }

    bool global_valid = true;
    if (validate) {
        bool ok = validateResult_distributed(cold_local, nx, ny, local_nz, rank, size, MPI_COMM_WORLD);
        int ok_int = ok ? 1 : 0;
        int all_ok = 0;
        MPI_Allreduce(&ok_int, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        global_valid = (all_ok == 1);
        if (rank == 0) {
            if (global_valid) printf("Validation: PASSED\n"); else printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return global_valid ? 0 : 1;
}
