#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI globals
int rank = 0;
int num_procs = 1;
size_t z_start = 0;
size_t z_end = 0;

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

// Exchange halo regions between neighboring ranks for Z-decomposition
void exchangeHalo(std::vector<double>& data, const size_t nx, const size_t ny, const size_t nz_local) {
    if (num_procs == 1) return;
    
    const size_t plane_size = nx * ny;
    std::vector<MPI_Request> requests;
    
    // Send bottom plane to left neighbor
    if (rank > 0) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(data.data(), static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 0,
                  MPI_COMM_WORLD, &requests.back());
    }
    
    // Receive from right neighbor into top halo
    if (rank < num_procs - 1) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(&data[nz_local * plane_size], static_cast<int>(plane_size), MPI_DOUBLE, 
                  rank + 1, 0, MPI_COMM_WORLD, &requests.back());
    }
    
    // Send top plane to right neighbor
    if (rank < num_procs - 1) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&data[(nz_local - 1) * plane_size], static_cast<int>(plane_size), MPI_DOUBLE, 
                  rank + 1, 1, MPI_COMM_WORLD, &requests.back());
    }
    
    // Receive from left neighbor into bottom halo
    if (rank > 0) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Irecv(&data[(nz_local + 1) * plane_size], static_cast<int>(plane_size), MPI_DOUBLE, 
                  rank - 1, 1, MPI_COMM_WORLD, &requests.back());
    }
    
    // Wait for all communications to complete
    if (!requests.empty()) {
        std::vector<MPI_Status> statuses(requests.size());
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), statuses.data());
    }
}

// Compute chemical potential (distributed version)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz_local,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz_local + 2, dx, dy, dz, x, y, z + 1);
            }
        }
    }
}

// Cahn-Hilliard update step (distributed version)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz_local,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz_local + 2, dx, dy, dz, x, y, z + 1);
            }
        }
    }
}

// Initialize concentration field (distributed version)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz_local,
                             const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const size_t global_z = z_start + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
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
    
    // Distribute Z dimension across ranks
    size_t nz_local = nz / num_procs;
    size_t nz_remainder = nz % num_procs;
    if (rank < static_cast<int>(nz_remainder)) {
        nz_local++;
        z_start = rank * (nz_local);
    } else {
        z_start = nz_remainder * (nz / num_procs + 1) + (static_cast<size_t>(rank) - nz_remainder) * nz_local;
    }
    z_end = z_start + nz_local;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_procs);
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
    
    // Allocate arrays with halo regions for Z-decomposition
    // Each rank stores nz_local + 2 planes (one halo on each side)
    size_t plane_size = nx * ny;
    size_t local_size = plane_size * (nz_local + 2);
    
    // Initialize with halo regions
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);
    
    // Initialize concentration field (skip halo regions)
    if (rank == 0) printf("Initializing concentration field...\n");
    for (size_t z = 0; z < nz_local; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z + 1, nx, ny);
                const size_t global_z = z_start + z;
                const size_t linear_id = global_z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % (nx * ny * nz)) / static_cast<double>(nx * ny * nz));
                cold[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
    
    // Initialize halo regions with boundary conditions
    if (rank == 0) {
        for (size_t i = 0; i < plane_size; ++i) {
            cold[i] = cold[plane_size];
        }
    }
    if (rank == num_procs - 1) {
        size_t last_internal = (nz_local) * plane_size;
        for (size_t i = 0; i < plane_size; ++i) {
            cold[(nz_local + 1) * plane_size + i] = cold[last_internal - plane_size + i];
        }
    }
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Exchange halo regions
        exchangeHalo(cold, nx, ny, nz_local);
        
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz_local, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);
        
        // Exchange halos for mu
        exchangeHalo(mu, nx, ny, nz_local);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather timing information (use max across all ranks)
    long local_time = duration.count();
    long global_time = 0;
    MPI_Reduce(&local_time, &global_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_time);
        
        // Calculate performance
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (global_time / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results for validation (only needed at rank 0)
    std::vector<double> full_result;
    if (rank == 0) {
        full_result.resize(nx * ny * nz);
    }
    
    if (validate || printResults) {
        // Gather all data to rank 0
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displs(num_procs);
        
        for (int i = 0; i < num_procs; ++i) {
            size_t local_nz = nz / num_procs;
            size_t remainder = nz % num_procs;
            if (i < static_cast<int>(remainder)) local_nz++;
            recvcounts[i] = static_cast<int>(local_nz * plane_size);
            
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }
        
        // Gather the actual data (without halos)
        std::vector<double> send_buf(nz_local * plane_size);
        for (size_t i = 0; i < nz_local * plane_size; ++i) {
            send_buf[i] = cold[i + plane_size];
        }
        
        // Allocate receive buffer on all ranks (only rank 0 uses it)
        std::vector<double> recv_buf;
        if (rank == 0) {
            recv_buf = full_result;
        } else {
            recv_buf.resize(1);
        }
        
        MPI_Gatherv(send_buf.data(), static_cast<int>(nz_local * plane_size), MPI_DOUBLE,
                    recv_buf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            full_result = recv_buf;
        }
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(full_result, "Concentration");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(full_result, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
