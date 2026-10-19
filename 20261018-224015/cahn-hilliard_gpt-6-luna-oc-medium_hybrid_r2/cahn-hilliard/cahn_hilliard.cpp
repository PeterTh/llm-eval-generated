#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ inline size_t didx(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}
__device__ double lap(const double* a, size_t x, size_t y, size_t z, size_t nx, size_t ny, size_t nz) {
    const size_t p=didx(x,y,z,nx,ny), xp=didx(x+1<nx?x+1:x,y,z,nx,ny), xm=didx(x?x-1:0,y,z,nx,ny);
    const size_t yp=didx(x,y+1<ny?y+1:y,z,nx,ny), ym=didx(x,y?y-1:0,z,nx,ny);
    const size_t zp=didx(x,y,z+1<nz?z+1:z,nx,ny), zm=didx(x,y,z?z-1:0,nx,ny);
    return a[xp]+a[xm]+a[yp]+a[ym]+a[zp]+a[zm]-6.0*a[p];
}
__global__ void chemicalKernel(const double* c, double* mu, size_t nx,size_t ny,size_t nz) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x, n=nx*ny*nz; if(i>=n)return;
    size_t x=i%nx,y=(i/nx)%ny,z=i/(nx*ny); double v=c[i];
    mu[i]=4.5*((v+1.0)*(-2.0/9.0)+(v-1.0)*(-2.0/9.0)-2.0*v*(2.0/9.0))+3.0*v+v*v*v-0.5*lap(c,x,y,z,nx,ny,nz);
}
__global__ void updateKernel(const double* c,const double* mu,double* out,size_t nx,size_t ny,size_t nz) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x,n=nx*ny*nz;if(i>=n)return;
    size_t x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);out[i]=c[i]+0.01*lap(mu,x,y,z,nx,ny,nz);
}
static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,1);} }

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
    #pragma omp parallel for collapse(2) schedule(static)
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
    MPI_Init(&argc,&argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
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
    
    if(rank==0) printf("Cahn-Hilliard Phase Separation Benchmark\n");
    if(rank==0){printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);printf("Time steps: %d\n", iterations);printf("Validation: %s\n", validate ? "enabled" : "disabled");}
    
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
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    
    // Initialize concentration field
    if(rank==0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    if(rank==0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    double *dc=nullptr,*dn=nullptr,*dm=nullptr;
    cudaCheck(cudaMalloc(&dc,gridSize*sizeof(double))); cudaCheck(cudaMalloc(&dn,gridSize*sizeof(double))); cudaCheck(cudaMalloc(&dm,gridSize*sizeof(double)));
    cudaCheck(cudaMemcpy(dc,cold.data(),gridSize*sizeof(double),cudaMemcpyHostToDevice));
    const int threads=256; const int blocks=static_cast<int>((gridSize+threads-1)/threads);
    for(int t=0;t<iterations;++t){chemicalKernel<<<blocks,threads>>>(dc,dm,nx,ny,nz);updateKernel<<<blocks,threads>>>(dc,dm,dn,nx,ny,nz);cudaCheck(cudaGetLastError());std::swap(dc,dn);}
    cudaCheck(cudaDeviceSynchronize()); cudaCheck(cudaMemcpy(cold.data(),dc,gridSize*sizeof(double),cudaMemcpyDeviceToHost));
    cudaFree(dc);cudaFree(dn);cudaFree(dm);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long local_ms = duration.count();
    long max_ms = local_ms;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("Computation time: %ld ms\n", max_ms);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
    if(rank==0) printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults && rank==0) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate && rank==0) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
