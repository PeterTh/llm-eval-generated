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
    MPI_Init(&argc, &argv); int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) printf("Cahn-Hilliard Phase Separation Benchmark\n");
    if (rank == 0) { printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz); printf("Time steps: %d\n", iterations); printf("MPI ranks: %d\n", nranks); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
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
    
    if (nx == 0 || ny == 0 || nz == 0 || nranks > static_cast<int>(nz)) { if (rank == 0) fprintf(stderr,"Invalid grid or too many ranks\n"); MPI_Finalize(); return 1; }
    const size_t base=nz/nranks, rem=nz%nranks, localZ=base+(static_cast<size_t>(rank)<rem), z0=rank*base+std::min(static_cast<size_t>(rank),rem), plane=nx*ny;
    
    // Allocate arrays
    std::vector<double> cold((localZ+2)*plane), cnew((localZ+2)*plane), mu((localZ+2)*plane);
    
    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    for(size_t z=1;z<=localZ;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){size_t id=(z0+z-1)*plane+y*nx+x;cold[idx3(x,y,z,nx,ny)]=-1+2*((((id+1)*1299709)%(nx*ny*nz))/static_cast<double>(nx*ny*nz));}
    auto exchange=[&](std::vector<double>& a){int lo=rank?rank-1:MPI_PROC_NULL,hi=rank+1<nranks?rank+1:MPI_PROC_NULL;MPI_Sendrecv(a.data()+plane,plane,MPI_DOUBLE,lo,0,a.data()+(localZ+1)*plane,plane,MPI_DOUBLE,hi,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);MPI_Sendrecv(a.data()+localZ*plane,plane,MPI_DOUBLE,hi,1,a.data(),plane,MPI_DOUBLE,lo,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);if(!rank)std::copy(a.data()+plane,a.data()+2*plane,a.data());if(rank==nranks-1)std::copy(a.data()+localZ*plane,a.data()+(localZ+1)*plane,a.data()+(localZ+1)*plane);};
    exchange(cold);
    
    // Run simulation
    if(rank==0)printf("Running Cahn-Hilliard simulation...\n"); MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, localZ+2, dx, dy, dz,
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        exchange(mu); cahnHilliardUpdate(cnew, cold, mu, nx, ny, localZ+2, D, dt, dx, dy, dz);
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    double elapsed=MPI_Wtime()-start,duration=0; MPI_Reduce(&elapsed,&duration,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    
    if(rank==0)printf("Computation time: %.0f ms\n",duration*1000);
    
    // Calculate performance
    if(rank==0)printf("Performance: %.3f MCellUpdates/s\n",static_cast<double>(nx*ny*nz)*iterations/duration/1e6);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> all(rank?0:nx*ny*nz);std::vector<int> counts(rank?0:nranks),displs(rank?0:nranks);for(int r=0;r<nranks&&rank==0;++r){size_t n=base+(static_cast<size_t>(r)<rem);counts[r]=n*plane;displs[r]=(r*base+std::min(static_cast<size_t>(r),rem))*plane;}MPI_Gatherv(cold.data()+plane,localZ*plane,MPI_DOUBLE,rank?nullptr:all.data(),rank?nullptr:counts.data(),rank?nullptr:displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);if(rank==0)print_results(all,"Concentration");
    }
    
    // Validation
    if (validate) {
        if(rank==0)printf("Validating result...\n"); double lmin=*std::min_element(cold.begin()+plane,cold.end()-plane),lmax=*std::max_element(cold.begin()+plane,cold.end()-plane),gmin,gmax;MPI_Reduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);MPI_Reduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);bool valid=true;if(rank==0){printf("Concentration range: [%.6f, %.6f]\n",gmin,gmax);valid=gmin>=-10&&gmax<=10;}MPI_Bcast(&valid,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
        
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
