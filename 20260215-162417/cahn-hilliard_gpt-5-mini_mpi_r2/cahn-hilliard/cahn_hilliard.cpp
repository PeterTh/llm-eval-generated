#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Local 3D index with ghost planes in Z: z in [0 .. nz_local+1]
inline size_t idx3_local(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny, const size_t nz_local_with_ghost) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Exchange ghost planes in Z for a contiguous 3D array with layout z*(nx*ny)+y*nx+x
static void exchange_ghost_planes(std::vector<double>& arr, const size_t nx, const size_t ny, const size_t nz_local, int rank, int size, MPI_Comm comm) {
    const int plane_count = static_cast<int>(nx * ny);
    MPI_Status status;

    // send last interior plane (z = nz_local) to right neighbor, receive into ghost after (z = nz_local+1)
    if (rank < size - 1 && nz_local > 0) {
        double* sendbuf = &arr[idx3_local(0, 0, nz_local, nx, ny, nz_local + 2)];
        double* recvbuf = &arr[idx3_local(0, 0, nz_local + 1, nx, ny, nz_local + 2)];
        MPI_Sendrecv(sendbuf, plane_count, MPI_DOUBLE, rank + 1, 0,
                     recvbuf, plane_count, MPI_DOUBLE, rank + 1, 1, comm, &status);
    } else if (nz_local > 0) {
        // boundary: clamp to last interior plane
        double* src = &arr[idx3_local(0, 0, nz_local, nx, ny, nz_local + 2)];
        double* dst = &arr[idx3_local(0, 0, nz_local + 1, nx, ny, nz_local + 2)];
        std::memcpy(dst, src, plane_count * sizeof(double));
    }

    // send first interior plane (z = 1) to left neighbor, receive into ghost before (z = 0)
    if (rank > 0 && nz_local > 0) {
        double* sendbuf = &arr[idx3_local(0, 0, 1, nx, ny, nz_local + 2)];
        double* recvbuf = &arr[idx3_local(0, 0, 0, nx, ny, nz_local + 2)];
        MPI_Sendrecv(sendbuf, plane_count, MPI_DOUBLE, rank - 1, 1,
                     recvbuf, plane_count, MPI_DOUBLE, rank - 1, 0, comm, &status);
    } else if (nz_local > 0) {
        // boundary: clamp to first interior plane
        double* src = &arr[idx3_local(0, 0, 1, nx, ny, nz_local + 2)];
        double* dst = &arr[idx3_local(0, 0, 0, nx, ny, nz_local + 2)];
        std::memcpy(dst, src, plane_count * sizeof(double));
    }
}

// Compute Laplacian for local array with ghost planes in Z
static double computeLaplacian_local(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local_with_ghost,
                                     const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz_local_with_ghost - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double center = c[idx3_local(x, y, z, nx, ny, nz_local_with_ghost)];
    const double cxx = (c[idx3_local(xp, y, z, nx, ny, nz_local_with_ghost)] + c[idx3_local(xn, y, z, nx, ny, nz_local_with_ghost)] - 2.0 * center) / (dx * dx);
    const double cyy = (c[idx3_local(x, yp, z, nx, ny, nz_local_with_ghost)] + c[idx3_local(x, yn, z, nx, ny, nz_local_with_ghost)] - 2.0 * center) / (dy * dy);
    const double czz = (c[idx3_local(x, y, zp, nx, ny, nz_local_with_ghost)] + c[idx3_local(x, y, zn, nx, ny, nz_local_with_ghost)] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

// Compute chemical potential on local interior (z=1..nz_local)
static void computeChemicalPotential_local(const std::vector<double>& cold_local, std::vector<double>& mu_local,
                                           const size_t nx, const size_t ny, const size_t nz_local,
                                           const double dx, const double dy, const double dz,
                                           const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nz_with_ghost = nz_local + 2;
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z, nx, ny, nz_with_ghost);
                const double cv = cold_local[idx];
                mu_local[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv
                                - gamma * computeLaplacian_local(cold_local, nx, ny, nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update on local interior
static void cahnHilliardUpdate_local(std::vector<double>& cnew_local, const std::vector<double>& cold_local,
                                     const std::vector<double>& mu_local,
                                     const size_t nx, const size_t ny, const size_t nz_local,
                                     const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nz_with_ghost = nz_local + 2;
    for (size_t z = 1; z <= nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, z, nx, ny, nz_with_ghost);
                cnew_local[idx] = cold_local[idx] + dt * D * computeLaplacian_local(mu_local, nx, ny, nz_with_ghost, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration local interior using global indexing and same pseudo-random formula
static void initializeConcentration_local(std::vector<double>& cold_local, const size_t nx, const size_t ny, const size_t nz_local,
                                          const size_t z_offset, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    const size_t nz_with_ghost = nz_local + 2;
    // initialize ghosts and interior to safe values
    std::fill(cold_local.begin(), cold_local.end(), 0.0);
    for (size_t local_z = 1; local_z <= nz_local; ++local_z) {
        const size_t gz = z_offset + (local_z - 1); // global z
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3_local(x, y, local_z, nx, ny, nz_with_ghost);
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cold_local[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    // clamp ghosts to nearest interior plane if exists (will be refreshed by exchange as needed)
    if (nz_local > 0) {
        // before first
        double* first_interior = &cold_local[idx3_local(0, 0, 1, nx, ny, nz_with_ghost)];
        double* ghost_before = &cold_local[idx3_local(0, 0, 0, nx, ny, nz_with_ghost)];
        std::memcpy(ghost_before, first_interior, nx * ny * sizeof(double));
        // after last
        double* last_interior = &cold_local[idx3_local(0, 0, nz_local, nx, ny, nz_with_ghost)];
        double* ghost_after = &cold_local[idx3_local(0, 0, nz_local + 1, nx, ny, nz_with_ghost)];
        std::memcpy(ghost_after, last_interior, nx * ny * sizeof(double));
    }
}

// Root-side validation by gathering full field and using existing validateResult
bool gather_and_validate(std::vector<double>& cold_local, const size_t nx, const size_t ny, const size_t nz_local,
                         const std::vector<int>& counts_elems, const std::vector<int>& displs_elems, const size_t nz_global, int rank, MPI_Comm comm) {
    const int world = 0;
    const size_t local_elems = nx * ny * nz_local;
    std::vector<double> full;
    if (rank == 0) full.resize(nx * ny * nz_global);

    // send buffer must skip ghosts; if nz_local==0, send nullptr with count 0
    double* sendbuf = nullptr;
    if (nz_local > 0) sendbuf = &cold_local[idx3_local(0, 0, 1, nx, ny, nz_local + 2)];

    MPI_Gatherv(sendbuf, static_cast<int>(local_elems), MPI_DOUBLE,
                full.data(), counts_elems.data(), displs_elems.data(), MPI_DOUBLE,
                0, comm);

    if (rank == 0) {
        return validateResult(full, nx, ny, nz_global);
    }
    return true; // non-root will rely on root's result
}

void print_usage_root(int rank, const char* progName) {
    if (rank == 0) {
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
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Only rank 0 parses and then broadcasts options
    if (rank == 0) {
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
                print_usage_root(0, argv[0]);
                MPI_Finalize();
                return 0;
            } else if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                print_usage_root(0, argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

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

    // Partition along Z
    std::vector<int> counts(size), displs(size);
    int base = static_cast<int>(nz) / size;
    int rem = static_cast<int>(nz) % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r - 1] + counts[r - 1];

    const int nz_local = counts[rank];
    const int z_offset = displs[rank];

    // Each local array includes two ghost planes
    const size_t nz_with_ghost = (nz_local > 0) ? (nz_local + 2) : 0;
    const size_t local_alloc = (nz_with_ghost == 0) ? 0 : nx * ny * nz_with_ghost;

    std::vector<double> cold_local, cnew_local, mu_local;
    if (local_alloc > 0) {
        cold_local.resize(local_alloc);
        cnew_local.resize(local_alloc);
        mu_local.resize(local_alloc);
    }

    // Initialize concentration local
    if (nz_local > 0) {
        initializeConcentration_local(cold_local, nx, ny, nz_local, z_offset, nz);
    }

    // Prepare counts and displacements for gather in elements
    std::vector<int> counts_elems(size), displs_elems(size);
    for (int r = 0; r < size; ++r) {
        counts_elems[r] = static_cast<int>(nx * ny * counts[r]);
        displs_elems[r] = static_cast<int>(nx * ny * displs[r]);
    }

    // Warm up exchange to set ghosts
    if (nz_local > 0) exchange_ghost_planes(cold_local, nx, ny, nz_local, rank, size, MPI_COMM_WORLD);

    // Run simulation with MPI parallelism
    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Ensure ghost planes for cold are up-to-date
        if (nz_local > 0) exchange_ghost_planes(cold_local, nx, ny, nz_local, rank, size, MPI_COMM_WORLD);

        // Compute chemical potential on interior
        if (nz_local > 0) computeChemicalPotential_local(cold_local, mu_local, nx, ny, nz_local, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange ghost planes of mu before update
        if (nz_local > 0) exchange_ghost_planes(mu_local, nx, ny, nz_local, rank, size, MPI_COMM_WORLD);

        // Update concentration interior
        if (nz_local > 0) cahnHilliardUpdate_local(cnew_local, cold_local, mu_local, nx, ny, nz_local, D, dt, dx, dy, dz);

        // Swap interior buffers (swap entire vectors including ghosts for simplicity)
        if (nz_local > 0) cold_local.swap(cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tend = MPI_Wtime();
    double elapsed = tend - tstart;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f s\n", max_elapsed);
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Print results by gathering to root
    if (printResults) {
        // Gather interior cold_local (skip ghosts) to root and print
        std::vector<double> full;
        if (rank == 0) full.resize(gridSize);
        double* sendbuf = nullptr;
        if (nz_local > 0) sendbuf = &cold_local[idx3_local(0, 0, 1, nx, ny, nz_with_ghost)];
        MPI_Gatherv(sendbuf, nx * ny * nz_local, MPI_DOUBLE,
                    full.data(), counts_elems.data(), displs_elems.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) print_results(full, "Concentration");
    }

    // Validation: gather to root and run existing validateResult
    int exit_code = 0;
    if (validate) {
        std::vector<double> full;
        if (rank == 0) full.resize(gridSize);
        double* sendbuf = nullptr;
        if (nz_local > 0) sendbuf = &cold_local[idx3_local(0, 0, 1, nx, ny, nz_with_ghost)];
        MPI_Gatherv(sendbuf, nx * ny * nz_local, MPI_DOUBLE,
                    full.data(), counts_elems.data(), displs_elems.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            bool ok = validateResult(full, nx, ny, nz);
            if (ok) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Finalize();
    return exit_code;
}
