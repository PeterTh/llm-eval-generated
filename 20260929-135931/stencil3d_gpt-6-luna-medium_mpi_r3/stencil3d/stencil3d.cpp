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
    
    if (nz < static_cast<size_t>(ranks)) {
        if (rank == 0) std::fprintf(stderr, "Error: Z dimension must be at least the MPI rank count\n");
        MPI_Finalize(); return 1;
    }
    const size_t base = nz / static_cast<size_t>(ranks), rem = nz % static_cast<size_t>(ranks);
    const size_t localNz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t plane = nx * ny;
    const size_t localSize = plane * localNz;
    std::vector<Real> a(plane * (localNz + 2)), b(plane * (localNz + 2));
    for (size_t lz = 0; lz < localNz; ++lz) {
        size_t gz = zStart + lz;
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x)
            a[idx3(x,y,lz+1,nx,ny)] = ((gz * plane + y * nx + x) % 19) * 1.0;
    }
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    if (rank == 0) {
        printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n", nx,ny,nz,iterations,validate?"enabled":"disabled");
        printf("Initializing grid...\nRunning stencil computation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Real>& in = (iter % 2 == 0) ? a : b;
        std::vector<Real>& out = (iter % 2 == 0) ? b : a;
        MPI_Request req[4];
        MPI_Irecv(in.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 10, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(in.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 11, MPI_COMM_WORLD, &req[1]);
        MPI_Isend(in.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 11, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(in.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &req[3]);
        for (size_t lz = 0; lz < localNz; ++lz) {
            const size_t gz = zStart + lz;
            if (gz == 0 || gz + 1 == nz) continue;
            for (size_t y=1; y+1<ny; ++y) for(size_t x=1; x+1<nx; ++x) {
                const size_t q=idx3(x,y,lz+1,nx,ny);
                out[q]=(in[q]+in[q-1]+in[q+1]+in[q-nx]+in[q+nx]+in[q-plane]+in[q+plane])/7.0;
            }
        }
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        // Compute any local boundary plane that depends on a received halo.
        for (size_t lz = 0; lz < localNz; ++lz) {
            size_t gz = zStart + lz;
            if (gz == 0 || gz + 1 == nz || (lz != 0 && lz + 1 != localNz)) continue;
            for (size_t y=1; y+1<ny; ++y) for(size_t x=1; x+1<nx; ++x) {
                size_t q=idx3(x,y,lz+1,nx,ny);
                out[q]=(in[q]+in[q-1]+in[q+1]+in[q-nx]+in[q+nx]+in[q-plane]+in[q+plane])/7.0;
            }
        }
        // Preserve every non-computed cell exactly, including global and XY faces.
        for (size_t lz=0; lz<localNz; ++lz) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x)
            if (x==0 || x+1==nx || y==0 || y+1==ny || zStart+lz==0 || zStart+lz+1==nz)
                out[idx3(x,y,lz+1,nx,ny)] = in[idx3(x,y,lz+1,nx,ny)];
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    const std::vector<Real>& localFinal = (iterations % 2 == 0) ? a : b;
    std::vector<Real> finalGrid;
    if (rank == 0) finalGrid.resize(nx*ny*nz);
    std::vector<int> counts(ranks), displs(ranks);
    for(int r=0;r<ranks;++r) { size_t n=base+(static_cast<size_t>(r)<rem?1:0); counts[r]=static_cast<int>(n*plane); size_t off=(static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*plane; displs[r]=static_cast<int>(off); }
    MPI_Gatherv(localFinal.data()+plane,static_cast<int>(localSize),MPI_DOUBLE,rank==0?finalGrid.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int localOk=1, allOk=1;
    if(validate) { for(size_t q=plane;q<plane*(localNz+1);++q) { Real v=localFinal[q]; if(!std::isfinite(v)||v>1e6||v< -1e6) localOk=0; } MPI_Allreduce(&localOk,&allOk,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD); }
    if(rank==0) {
        long duration=static_cast<long>(maxElapsed*1000.0);
        printf("Computation time: %ld ms\n",duration);
        double updates=static_cast<double>((nx>2?nx-2:0)*(ny>2?ny-2:0)*(nz>2?nz-2:0))*iterations;
        printf("Performance: %.3f MCellUpdates/s\n", maxElapsed>0?updates/maxElapsed/1e6:0.0);
        if(printResults) print_results(finalGrid,"Grid");
        if(validate) { printf("Validating result...\nValue range: [%.6f, %.6f]\n", *std::min_element(finalGrid.begin(),finalGrid.end()), *std::max_element(finalGrid.begin(),finalGrid.end())); printf("Validation: %s\n",allOk?"PASSED":"FAILED"); }
    }
    MPI_Finalize();
    return validate && !allOk ? 1 : 0;
}
