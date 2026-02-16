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

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
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
    int rank, size;
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
    
    // Domain decomposition (z-slab)
    size_t z_per_rank = nz / size;
    size_t z_rem = nz % size;
    size_t z_start = rank * z_per_rank + (rank < z_rem ? rank : z_rem);
    size_t z_count = z_per_rank + (rank < z_rem ? 1 : 0);
    size_t z_end = z_start + z_count;
    size_t localGridSize = nx * ny * z_count;
    size_t gridSize = nx * ny * nz;

    // Allocate arrays
    std::vector<double> cold(localGridSize);
    std::vector<double> cnew(localGridSize);
    std::vector<double> mu(localGridSize);

    // Initialize concentration field (global, then scatter)
    std::vector<double> global_init;
    if (rank == 0) {
        global_init.resize(gridSize);
        initializeConcentration(global_init, nx, ny, nz);
    }
    // Scatter initial data
    std::vector<int> sendcounts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t zs = r * z_per_rank + (r < z_rem ? r : z_rem);
        size_t zc = z_per_rank + (r < z_rem ? 1 : 0);
        sendcounts[r] = nx * ny * zc;
        displs[r] = nx * ny * zs;
    }
    MPI_Scatterv(rank == 0 ? global_init.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 cold.data(), localGridSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Initializing concentration field...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Buffers for halo exchange
    std::vector<double> halo_top(nx * ny), halo_bottom(nx * ny);
    for (int t = 0; t < iterations; ++t) {
        // Halo exchange (top and bottom z-planes)
        if (size > 1) {
            // Send top, receive bottom
            if (rank < size - 1) {
                MPI_Sendrecv(&cold[(z_count-1)*nx*ny], nx*ny, MPI_DOUBLE, rank+1, 0,
                             halo_top.data(), nx*ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank > 0) {
                MPI_Sendrecv(&cold[0], nx*ny, MPI_DOUBLE, rank-1, 1,
                             halo_bottom.data(), nx*ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // Compute chemical potential (with halo)
        for (size_t z = 0; z < z_count; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t global_z = z_start + z;
                    size_t idx = z * (nx * ny) + y * nx + x;
                    double cv = cold[idx];
                    // Laplacian with halo
                    size_t zp = (z < z_count-1) ? z+1 : z;
                    size_t zn = (z > 0) ? z-1 : z;
                    double czz;
                    if (z == 0 && rank > 0) {
                        czz = (cold[idx3(x, y, zp, nx, ny)] + halo_bottom[y*nx + x] - 2.0 * cv) / (dz * dz);
                    } else if (z == z_count-1 && rank < size-1) {
                        czz = (halo_top[y*nx + x] + cold[idx3(x, y, zn, nx, ny)] - 2.0 * cv) / (dz * dz);
                    } else {
                        czz = (cold[idx3(x, y, zp, nx, ny)] + cold[idx3(x, y, zn, nx, ny)] - 2.0 * cv) / (dz * dz);
                    }
                    double cxx = (cold[idx3((x < nx-1) ? x+1 : x, y, z, nx, ny)] +
                                   cold[idx3((x > 0) ? x-1 : 0, y, z, nx, ny)] - 2.0 * cv) / (dx * dx);
                    double cyy = (cold[idx3(x, (y < ny-1) ? y+1 : y, z, nx, ny)] +
                                   cold[idx3(x, (y > 0) ? y-1 : 0, z, nx, ny)] - 2.0 * cv) / (dy * dy);
                    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv
                              - gamma * (cxx + cyy + czz);
                }
            }
        }
        // Update concentration
        for (size_t z = 0; z < z_count; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    size_t idx = z * (nx * ny) + y * nx + x;
                    // Laplacian of mu (reuse halo)
                    double muv = mu[idx];
                    size_t zp = (z < z_count-1) ? z+1 : z;
                    size_t zn = (z > 0) ? z-1 : z;
                    double czz;
                    if (z == 0 && rank > 0) {
                        czz = (mu[idx3(x, y, zp, nx, ny)] + halo_bottom[y*nx + x] - 2.0 * muv) / (dz * dz);
                    } else if (z == z_count-1 && rank < size-1) {
                        czz = (halo_top[y*nx + x] + mu[idx3(x, y, zn, nx, ny)] - 2.0 * muv) / (dz * dz);
                    } else {
                        czz = (mu[idx3(x, y, zp, nx, ny)] + mu[idx3(x, y, zn, nx, ny)] - 2.0 * muv) / (dz * dz);
                    }
                    double cxx = (mu[idx3((x < nx-1) ? x+1 : x, y, z, nx, ny)] +
                                   mu[idx3((x > 0) ? x-1 : 0, y, z, nx, ny)] - 2.0 * muv) / (dx * dx);
                    double cyy = (mu[idx3(x, (y < ny-1) ? y+1 : y, z, nx, ny)] +
                                   mu[idx3(x, (y > 0) ? y-1 : 0, z, nx, ny)] - 2.0 * muv) / (dy * dy);
                    cnew[idx] = cold[idx] + dt * D * (cxx + cyy + czz);
                }
            }
        }
        std::swap(cold, cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results to rank 0
    std::vector<double> global_cold;
    if (rank == 0) global_cold.resize(gridSize);
    MPI_Gatherv(cold.data(), localGridSize, MPI_DOUBLE,
                rank == 0 ? global_cold.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if (printResults) {
            print_results(global_cold, "Concentration");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_cold, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
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
