#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Process interior points (not on boundaries)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];
                
                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
    
    // Copy boundary values
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    output[idx] = input[idx];
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nz < static_cast<size_t>(ranks) || nx < 3 || ny < 3 || nz < 3) {
        if (rank == 0) printf("Grid dimensions must be >= 3 and Z must have at least one plane per MPI rank\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / ranks, rem = nz % ranks;
    const size_t localNz = base + (static_cast<size_t>(rank) < rem);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny, localSize = (localNz + 2) * plane;
    std::vector<Real> a(localSize), b(localSize);
    for (size_t lz = 1; lz <= localNz; ++lz)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                size_t global = (zStart + lz - 1) * plane + y * nx + x;
                a[lz * plane + y * nx + x] = static_cast<Real>(global % 19);
            }
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx, ny, nz, iterations, validate ? "enabled" : "disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    int prev = rank ? rank - 1 : MPI_PROC_NULL, next = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = (iter % 2 == 0) ? a.data() : b.data();
        Real* out = (iter % 2 == 0) ? b.data() : a.data();
        MPI_Sendrecv(in + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                     in + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(in + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                     in, static_cast<int>(plane), MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t z = 1; z <= localNz; ++z) {
            size_t gz = zStart + z - 1;
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
                size_t q = z * plane + y * nx + x;
                if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || gz == 0 || gz + 1 == nz) out[q] = in[q];
                else out[q] = (in[q] + in[q-1] + in[q+1] + in[q-nx] + in[q+nx] + in[q-plane] + in[q+plane]) / 7.0;
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end-start).count(), maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const Real* local = (iterations % 2 == 0) ? a.data() : b.data();
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0; r<ranks; ++r) { size_t n=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*plane); }
    std::vector<Real> full;
    if (rank == 0 && (printResults || validate)) full.resize(nx*ny*nz);
    MPI_Gatherv(const_cast<Real*>(local+plane), counts[rank], MPI_DOUBLE, rank==0 && !full.empty()?full.data():nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
        long ms = static_cast<long>(maxElapsed*1000.0);
        printf("Computation time: %ld ms\n", ms);
        double updates = static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations;
        printf("Performance: %.3f MCellUpdates/s\n", updates/(maxElapsed>0?maxElapsed:1e-9)/1e6);
        if (printResults) print_results(full, "Grid");
        if (validate) { printf("Validating result...\n"); result=validateResult(full,nx,ny,nz)?0:1; printf("Validation: %s\n",result==0?"PASSED":"FAILED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
