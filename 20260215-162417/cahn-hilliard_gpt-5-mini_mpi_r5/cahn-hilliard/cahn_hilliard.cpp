#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
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

// Compute chemical potential for local domain (nz includes 2 ghost layers)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_with_ghosts,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    // loop only over real z slices (1 .. nz_with_ghosts-2)
    for (size_t z = 1; z + 1 < nz_with_ghosts; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_with_ghosts, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step for local domain (nz includes 2 ghost layers)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_with_ghosts,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z + 1 < nz_with_ghosts; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_with_ghosts, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field for local domain; nz_with_ghosts includes ghosts, start_z is global z index of local z=1
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_with_ghosts, const size_t start_z, const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    const size_t local_nz = (nz_with_ghosts >= 2) ? (nz_with_ghosts - 2) : 0;
    
    for (size_t lz = 0; lz < local_nz; ++lz) {
        const size_t gz = start_z + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx_local = idx3(x, y, lz + 1, nx, ny); // +1 for ghost
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx_local] = -1.0 + 2.0 * pseudo;
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

// Exchange ghost layers in z-direction. arrays are sized with nz_with_ghosts = local_nz + 2
static void exchange_ghost_layer(std::vector<double>& arr, const size_t nx, const size_t ny, const size_t nz_with_ghosts, int prev_rank, int next_rank) {
    const int slice_count = static_cast<int>(nx * ny);
    MPI_Status st1, st2;

    // send first real slice (z=1) to prev_rank -> it becomes their top ghost; receive into z=0 from prev_rank
    MPI_Sendrecv(arr.data() + idx3(0,0,1,nx,ny), slice_count, MPI_DOUBLE, (prev_rank>=0?prev_rank:MPI_PROC_NULL), 0,
                 arr.data() + idx3(0,0,0,nx,ny), slice_count, MPI_DOUBLE, (prev_rank>=0?prev_rank:MPI_PROC_NULL), 0,
                 MPI_COMM_WORLD, &st1);

    // send last real slice (z=nz_with_ghosts-2) to next_rank -> it becomes their bottom ghost; receive into z=nz_with_ghosts-1 from next_rank
    MPI_Sendrecv(arr.data() + idx3(0,0,nz_with_ghosts-2,nx,ny), slice_count, MPI_DOUBLE, (next_rank>=0?next_rank:MPI_PROC_NULL), 1,
                 arr.data() + idx3(0,0,nz_with_ghosts-1,nx,ny), slice_count, MPI_DOUBLE, (next_rank>=0?next_rank:MPI_PROC_NULL), 1,
                 MPI_COMM_WORLD, &st2);

    // Clamp boundaries if neighbor is MPI_PROC_NULL
    if (prev_rank < 0) {
        // copy z=1 into z=0
        std::memcpy(arr.data() + idx3(0,0,0,nx,ny), arr.data() + idx3(0,0,1,nx,ny), slice_count * sizeof(double));
    }
    if (next_rank < 0) {
        std::memcpy(arr.data() + idx3(0,0,nz_with_ghosts-1,nx,ny), arr.data() + idx3(0,0,nz_with_ghosts-2,nx,ny), slice_count * sizeof(double));
    }
}

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    int rank = 0, size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute per-rank partitioning along z
    std::vector<int> local_nz_per_rank(size);
    const int base = static_cast<int>(nz) / size;
    const int rem = static_cast<int>(nz) % size;
    for (int r = 0; r < size; ++r) {
        local_nz_per_rank[r] = base + (r < rem ? 1 : 0);
    }

    int local_nz = local_nz_per_rank[rank];
    // compute start z (global) for this rank
    int start_z = 0;
    for (int r = 0; r < rank; ++r) start_z += local_nz_per_rank[r];

    const size_t nz_with_ghosts = static_cast<size_t>(local_nz > 0 ? local_nz + 2 : 2);
    const size_t local_storage = nx * ny * nz_with_ghosts;

    // Neighbors that actually have data
    int prev_rank = (rank > 0 && local_nz_per_rank[rank-1] > 0) ? (rank - 1) : -1;
    int next_rank = (rank + 1 < size && local_nz_per_rank[rank+1] > 0) ? (rank + 1) : -1;

    if (rank == 0) {
        if (rank == 0) {
            if (rank == 0) {} // silence unused warning
        }
    }

    if (rank == 0) {
        if (rank == 0) {
            // only rank 0 prints header
            if (rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
                printf("Time steps: %d\n", iterations);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
            }
        }
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

    // Allocate local arrays (with ghosts)
    std::vector<double> cold_local(local_storage);
    std::vector<double> cnew_local(local_storage);
    std::vector<double> mu_local(local_storage);

    // Initialize concentration field on local real slices
    if (local_nz > 0) {
        initializeConcentration(cold_local, nx, ny, nz_with_ghosts, static_cast<size_t>(start_z), nz);
    }
    // Ensure ghost layers set (clamped)
    if (local_nz > 0) {
        // for top ghost (z=0) copy from z=1, bottom ghost copy from last real slice
        std::memcpy(cold_local.data() + idx3(0,0,0,nx,ny), cold_local.data() + idx3(0,0,1,nx,ny), nx*ny*sizeof(double));
        std::memcpy(cold_local.data() + idx3(0,0,nz_with_ghosts-1,nx,ny), cold_local.data() + idx3(0,0,nz_with_ghosts-2,nx,ny), nx*ny*sizeof(double));
    } else {
        // no local data: set ghosts to zero
        std::fill(cold_local.begin(), cold_local.end(), 0.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();

    // Main time stepping
    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for concentration
        if (local_nz > 0) exchange_ghost_layer(cold_local, nx, ny, nz_with_ghosts, prev_rank, next_rank);

        // Compute chemical potential on local real slices
        if (local_nz > 0) computeChemicalPotential(cold_local, mu_local, nx, ny, nz_with_ghosts, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange halos for mu
        if (local_nz > 0) exchange_ghost_layer(mu_local, nx, ny, nz_with_ghosts, prev_rank, next_rank);

        // Update concentration on local real slices
        if (local_nz > 0) cahnHilliardUpdate(cnew_local, cold_local, mu_local, nx, ny, nz_with_ghosts, D, dt, dx, dy, dz);

        // For ghost layers we keep them as is; swap local buffers for next step
        if (local_nz > 0) std::swap(cold_local, cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tend = MPI_Wtime();
    double elapsed = tend - tstart;

    // Reduce max time across ranks
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to root for validation/printing
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = static_cast<int>(nx * ny * local_nz_per_rank[r]);
        // displacements in elements
        displs[r] = 0;
        for (int k = 0; k < r; ++k) displs[r] += sendcounts[k];
    }

    std::vector<double> global_data;
    if (rank == 0) {
        global_data.resize(static_cast<size_t>(nx) * ny * nz);
    }

    // Prepare send pointer (skip ghost layer). If local_nz==0 sendcount is zero and pointer may be nullptr.
    double* sendptr = nullptr;
    if (local_nz > 0) sendptr = cold_local.data() + idx3(0,0,1,nx,ny);

    MPI_Gatherv(sendptr, sendcounts[rank], MPI_DOUBLE,
                global_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", max_elapsed * 1000.0);
        double cellUpdates = static_cast<double>(nx) * ny * nz * iterations;
        double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(global_data, "Concentration");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_data, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
