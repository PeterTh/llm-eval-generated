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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nz < static_cast<size_t>(ranks)) { if (rank == 0) fprintf(stderr, "Z dimension must be >= MPI process count\n"); MPI_Finalize(); return 1; }
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
    
    size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zStart = rank * base + std::min(static_cast<size_t>(rank), rem);
    
    // Allocate arrays
    std::vector<double> cold((localNz + 2) * plane);
    std::vector<double> cnew(cold.size());
    std::vector<double> mu(cold.size());
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    for (size_t z=0; z<localNz; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
        const size_t id=(zStart+z)*plane+y*nx+x;
        cold[(z+1)*plane+y*nx+x] = -1.0 + 2.0 * ((((id+1)*1299709) % gridSize) / static_cast<double>(gridSize));
    }
    
    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        MPI_Sendrecv(cold.data()+plane, plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 10,
                     cold.data()+(localNz+1)*plane, plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(cold.data()+localNz*plane, plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 11,
                     cold.data(), plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank == 0) std::copy(cold.begin()+plane, cold.begin()+2*plane, cold.begin());
        if (rank == ranks-1) std::copy(cold.begin()+localNz*plane, cold.begin()+(localNz+1)*plane, cold.begin()+(localNz+1)*plane);
        for (size_t z=1; z<=localNz; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
            const size_t i=z*plane+y*nx+x;
            const double lap=(cold[i+1- (x+1==nx)] + cold[i-(x?1:0)] - 2*cold[i])/(dx*dx)
                +(cold[i+(y+1<ny?nx:0)] + cold[i-(y?nx:0)]-2*cold[i])/(dy*dy)
                +(cold[i+plane]+cold[i-plane]-2*cold[i])/(dz*dz);
            mu[i]=4.5*((cold[i]+1)*e_AA+(cold[i]-1)*e_BB-2*cold[i]*e_AB)+3*cold[i]+cold[i]*cold[i]*cold[i]-gamma*lap;
        }
        MPI_Sendrecv(mu.data()+plane, plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 12,
                     mu.data()+(localNz+1)*plane, plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(mu.data()+localNz*plane, plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 13,
                     mu.data(), plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank == 0) std::copy(mu.begin()+plane, mu.begin()+2*plane, mu.begin());
        if (rank == ranks-1) std::copy(mu.begin()+localNz*plane, mu.begin()+(localNz+1)*plane, mu.begin()+(localNz+1)*plane);
        for (size_t z=1; z<=localNz; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
            const size_t i=z*plane+y*nx+x;
            const double lap=(mu[i+(x+1<nx?1:0)]+mu[i-(x?1:0)]-2*mu[i])/(dx*dx)
                +(mu[i+(y+1<ny?nx:0)]+mu[i-(y?nx:0)]-2*mu[i])/(dy*dy)
                +(mu[i+plane]+mu[i-plane]-2*mu[i])/(dz*dz);
            cnew[i]=cold[i]+dt*D*lap;
        }
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0; r<ranks; ++r) { size_t rn=nz/ranks+(static_cast<size_t>(r)<nz%ranks); size_t rs=r*(nz/ranks)+std::min(static_cast<size_t>(r),nz%ranks); counts[r]=static_cast<int>(rn*plane); displs[r]=static_cast<int>(rs*plane); }
    std::vector<double> result(rank==0?gridSize:0);
    MPI_Gatherv(cold.data()+plane, static_cast<int>(localNz*plane), MPI_DOUBLE, result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    if (rank == 0) printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = rank == 0 ? validateResult(result, nx, ny, nz) : true;
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
