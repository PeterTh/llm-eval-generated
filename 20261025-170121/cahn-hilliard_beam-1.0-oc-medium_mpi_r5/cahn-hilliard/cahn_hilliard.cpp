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
    for (size_t z = 1; z + 1 < nz; ++z) {
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
    for (size_t z = 1; z + 1 < nz; ++z) {
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
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                            const size_t z_start, const size_t local_nz) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;
    
    for (size_t lz = 0; lz < local_nz; ++lz) {
        const size_t gz = z_start + lz;
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz + 1, nx, ny);
                const size_t linear_id = gz * plane + y * nx + x;
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
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
    
    // Partition z dimension across ranks
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    size_t z_start = rank * base + (rank < (int)rem ? rank : (int)rem);
    
    const size_t plane = nx * ny;
    const size_t nz_local_total = local_nz + 2;
    const size_t localSize = plane * nz_local_total;
    
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);
    
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, z_start, local_nz);
    
    // halos clamp initially
    if (local_nz > 0) {
        std::copy(cold.begin() + plane, cold.begin() + 2*plane, cold.begin()); // z=0 halo = first owned
        std::copy(cold.begin() + local_nz*plane, cold.begin() + (local_nz+1)*plane, cold.begin() + (local_nz+1)*plane); // last halo
    }
    
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // halo exchange for cold
        if (size > 1 && local_nz > 0) {
            if (rank > 0)
                MPI_Sendrecv(cold.data() + plane, plane, MPI_DOUBLE, rank-1, 0,
                             cold.data() + (local_nz+1)*plane, plane, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank < size-1)
                MPI_Sendrecv(cold.data() + local_nz*plane, plane, MPI_DOUBLE, rank+1, 1,
                             cold.data(), plane, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == 0 && local_nz > 0)
            std::copy(cold.begin()+plane, cold.begin()+2*plane, cold.begin());
        if (rank == size-1 && local_nz > 0)
            std::copy(cold.begin()+local_nz*plane, cold.begin()+(local_nz+1)*plane, cold.begin()+(local_nz+1)*plane);
        
        computeChemicalPotential(cold, mu, nx, ny, nz_local_total, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        
        // halo exchange for mu
        if (size > 1 && local_nz > 0) {
            if (rank > 0)
                MPI_Sendrecv(mu.data() + plane, plane, MPI_DOUBLE, rank-1, 2,
                             mu.data() + (local_nz+1)*plane, plane, MPI_DOUBLE, rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rank < size-1)
                MPI_Sendrecv(mu.data() + local_nz*plane, plane, MPI_DOUBLE, rank+1, 3,
                             mu.data(), plane, MPI_DOUBLE, rank+1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank == 0 && local_nz > 0)
            std::copy(mu.begin()+plane, mu.begin()+2*plane, mu.begin());
        if (rank == size-1 && local_nz > 0)
            std::copy(mu.begin()+local_nz*plane, mu.begin()+(local_nz+1)*plane, mu.begin()+(local_nz+1)*plane);
        
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz_local_total, D, dt, dx, dy, dz);
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)((t1-t0)*1000.0));
        double cellUpdates = (double)(nx*ny*nz) * iterations;
        double secs = t1 - t0;
        double mcups = cellUpdates / secs / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    // Gather results to rank 0
    std::vector<int> counts(size), displs(size);
    // need local_nz per rank
    std::vector<size_t> lnz(size);
    for (int r=0;r<size;++r){ size_t b=nz/size, remm=nz%size; lnz[r]=b+(r<(int)remm?1:0); }
    int disp=0;
    for (int r=0;r<size;++r){ counts[r]= (int)(lnz[r]*plane); displs[r]=disp; disp += counts[r]; }
    
    std::vector<double> globalGrid;
    if (rank == 0) globalGrid.resize(nx*ny*nz);
    MPI_Gatherv(cold.data() + plane, (int)(local_nz*plane), MPI_DOUBLE,
                rank==0?globalGrid.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (printResults) print_results(globalGrid, "Concentration");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalGrid, nx, ny, nz);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }
    
    MPI_Finalize();
    if (rank == 0 && validate) {
        // return code not propagated; okay
    }
    return 0;
}
