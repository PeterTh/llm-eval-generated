#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI helper structure
struct MPIContext {
    int rank = 0;
    int size = 1;
    size_t local_z_start = 0;
    size_t local_z_count = 0;
};

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with proper boundary handling
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t global_nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, 
                        const size_t global_z, const MPIContext& mpi) {
    // Handle global Z boundaries with clamping
    const size_t zp = (global_z < global_nz - 1) ? global_z + 1 : global_z;
    const size_t zn = (global_z > 0) ? global_z - 1 : 0;
    
    // Convert global_z to local_z
    const size_t local_z = global_z - mpi.local_z_start;
    const size_t local_zp = (zp >= mpi.local_z_start && zp < mpi.local_z_start + mpi.local_z_count) 
                           ? (zp - mpi.local_z_start) : local_z;
    const size_t local_zn = (zn >= mpi.local_z_start && zn < mpi.local_z_start + mpi.local_z_count) 
                           ? (zn - mpi.local_z_start) : local_z;
    
    // X and Y boundaries with clamping
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, local_z, nx, ny)] + c[idx3(xn, y, local_z, nx, ny)] - 
                  2.0 * c[idx3(x, y, local_z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, local_z, nx, ny)] + c[idx3(x, yn, local_z, nx, ny)] - 
                  2.0 * c[idx3(x, y, local_z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, local_zp, nx, ny)] + c[idx3(x, y, local_zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, local_z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Synchronize boundary planes between MPI processes
void syncBoundaryPlanes(const std::vector<double>& local_data, std::vector<double>& boundary_data_above,
                       std::vector<double>& boundary_data_below, const size_t nx, const size_t ny, 
                       const size_t local_nz, const MPIContext& mpi) {
    if (mpi.size == 1) return;
    
    const size_t plane_size = nx * ny;
    
    // Exchange using non-blocking communication
    MPI_Request reqs[4];
    int req_count = 0;
    
    // Send bottom plane up (to lower rank)
    if (mpi.rank > 0) {
        MPI_Isend(local_data.data(), plane_size, MPI_DOUBLE, mpi.rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    // Receive from above (from higher rank)
    if (mpi.rank < mpi.size - 1) {
        MPI_Irecv(boundary_data_above.data(), plane_size, MPI_DOUBLE, mpi.rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    // Send top plane down (to higher rank)
    if (mpi.rank < mpi.size - 1) {
        MPI_Isend(local_data.data() + (local_nz - 1) * plane_size, plane_size, MPI_DOUBLE,
                 mpi.rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    // Receive from below (from lower rank)
    if (mpi.rank > 0) {
        MPI_Irecv(boundary_data_below.data(), plane_size, MPI_DOUBLE, mpi.rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
}

// Compute chemical potential (parallelized)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t local_nz, const size_t global_nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const MPIContext& mpi) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_z = mpi.local_z_start + z;
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, global_nz, dx, dy, dz, x, y, global_z, mpi);
            }
        }
    }
}

// Cahn-Hilliard update step (parallelized)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t local_nz, const size_t global_nz,
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const MPIContext& mpi) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_z = mpi.local_z_start + z;
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, global_nz, dx, dy, dz, x, y, global_z, mpi);
            }
        }
    }
}

// Initialize concentration field (parallelized)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, 
                            const size_t local_nz, const size_t global_z_start, 
                            const size_t global_nz) {
    const size_t vol = nx * ny * global_nz;
    
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t linear_id = (global_z_start + z) * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t local_nz, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0) {
                printf("Validation failed: found NaN or Inf value\n");
            }
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
    
    // Reduce to find global min/max
    double global_min = minVal, global_max = maxVal;
    MPI_Allreduce(&minVal, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&maxVal, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
        
        if (global_max > 10.0 || global_min < -10.0) {
            printf("Validation failed: values out of expected range\n");
            return false;
        }
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPIContext mpi;
    mpi.rank = rank;
    mpi.size = size;
    
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0)
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
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Decompose domain along Z axis
    mpi.local_z_count = nz / size;
    const size_t remainder = nz % size;
    if (rank < (int)remainder) {
        mpi.local_z_count++;
        mpi.local_z_start = rank * (nz / size + 1);
    } else {
        mpi.local_z_start = remainder * (nz / size + 1) + (rank - remainder) * (nz / size);
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu (total: %lu cells)\n", nx, ny, nz, nx*ny*nz);
        printf("MPI processes: %d\n", size);
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
    
    size_t local_gridSize = nx * ny * mpi.local_z_count;
    
    // Allocate local arrays
    std::vector<double> cold(local_gridSize);
    std::vector<double> cnew(local_gridSize);
    std::vector<double> mu(local_gridSize);
    
    // Allocate boundary buffers for ghost cell exchange
    std::vector<double> boundary_above(nx * ny, 0.0);
    std::vector<double> boundary_below(nx * ny, 0.0);
    
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    
    // Initialize concentration field
    initializeConcentration(cold, nx, ny, mpi.local_z_count, mpi.local_z_start, nz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Synchronize boundary planes before chemical potential computation
        syncBoundaryPlanes(cold, boundary_above, boundary_below, nx, ny, mpi.local_z_count, mpi);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, mpi.local_z_count, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB, mpi);
        
        // Synchronize boundary planes before update
        syncBoundaryPlanes(mu, boundary_above, boundary_below, nx, ny, mpi.local_z_count, mpi);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, mpi.local_z_count, nz, D, dt, dx, dy, dz, mpi);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)nx * ny * nz * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0 for validation and output
    std::vector<double> global_result;
    if (rank == 0) {
        global_result.resize(nx * ny * nz);
    }
    
    // Calculate counts and displacements for Gatherv
    std::vector<int> counts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        size_t z_count = nz / size;
        if (i < (int)(nz % size)) z_count++;
        counts[i] = nx * ny * z_count;
        
        if (i == 0) {
            displs[i] = 0;
        } else {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }
    
    MPI_Gatherv(cold.data(), cold.size(), MPI_DOUBLE, global_result.data(), 
               counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(global_result, "Concentration");
    }
    
    // Validation
    if (validate) {
        bool local_valid = validateResult(cold, mpi.local_z_count, nx, ny);
        int global_valid = local_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &global_valid, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return global_valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
