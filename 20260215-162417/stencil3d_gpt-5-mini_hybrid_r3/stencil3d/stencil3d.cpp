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

using Real = double;

// 3D index calculation (local)
inline __host__ __device__ size_t idx3_loc(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for interior stencil computation (skips global boundaries)
__global__ void stencilKernel(const double* input, double* output, int nx, int ny, int local_nz, int global_z0, int nz_global) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1; // skip x=0
    int iy = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip y=0
    int iz = blockIdx.z * blockDim.z + threadIdx.z + 1; // skip halo z=0

    if (ix >= nx-1 || iy >= ny-1 || iz >= local_nz-1) return; // skip edges and top halo

    int global_z = global_z0 + (iz - 1);
    if (global_z == 0 || global_z == nz_global - 1) return; // global z-boundaries

    size_t idx = idx3_loc(ix, iy, iz, nx, ny);

    double center = input[idx];
    double left = input[idx3_loc(ix-1, iy, iz, nx, ny)];
    double right = input[idx3_loc(ix+1, iy, iz, nx, ny)];
    double front = input[idx3_loc(ix, iy-1, iz, nx, ny)];
    double back = input[idx3_loc(ix, iy+1, iz, nx, ny)];
    double bottom = input[idx3_loc(ix, iy, iz-1, nx, ny)];
    double top = input[idx3_loc(ix, iy, iz+1, nx, ny)];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    // Initialize MPI early
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 then broadcast)
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
            }
        }
    }
    // Broadcast params
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition Z dimension among MPI ranks
    int base = nz / world;
    int rem = nz % world;
    int local_nz = base + (rank < rem ? 1 : 0); // number of real slices for this rank

    // compute global z start
    int z_start = 0;
    for (int r = 0; r < rank; ++r) z_start += base + (r < rem ? 1 : 0);

    // include halo layers
    int local_nz_with_halo = local_nz + 2;
    size_t planeSize = nx * ny; // doubles per z-slice
    size_t localSize = (size_t)local_nz_with_halo * planeSize;

    // Allocate managed memory so CUDA and host can access
    double* gridA = nullptr;
    double* gridB = nullptr;
    cudaMallocManaged(&gridA, localSize * sizeof(double));
    cudaMallocManaged(&gridB, localSize * sizeof(double));

    // Initialize local grids using OpenMP
    #pragma omp parallel for collapse(3)
    for (int z = 0; z < local_nz_with_halo; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                int global_z = z_start + (z - 1);
                size_t idx = idx3_loc(x, y, z, nx, ny);
                if (global_z < 0 || global_z >= (int)nz) {
                    // halo outside global domain (set to 0)
                    gridA[idx] = 0.0;
                    gridB[idx] = 0.0;
                } else {
                    size_t global_idx = (size_t)global_z * (nx * ny) + y * nx + x;
                    gridA[idx] = (global_idx % 19) * 1.0;
                    gridB[idx] = gridA[idx];
                }
            }
        }
    }
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Main iteration loop
    for (int iter = 0; iter < iterations; ++iter) {
        double* input = (iter % 2 == 0) ? gridA : gridB;
        double* output = (iter % 2 == 0) ? gridB : gridA;

        // Exchange halo planes with neighbors
        MPI_Status status;
        // Send first real slice to rank-1, receive into halo z=0
        if (rank > 0) {
            MPI_Sendrecv(input + idx3_loc(0,0,1,nx,ny), planeSize, MPI_DOUBLE, rank-1, 0,
                         input + idx3_loc(0,0,0,nx,ny), planeSize, MPI_DOUBLE, rank-1, 0,
                         MPI_COMM_WORLD, &status);
        } else {
            // replicate boundary into halo
            #pragma omp parallel for
            for (size_t i = 0; i < planeSize; ++i) input[i] = input[planeSize + i];
        }
        // Send last real slice to rank+1, receive into top halo
        if (rank < world - 1) {
            MPI_Sendrecv(input + idx3_loc(0,0,local_nz,nx,ny), planeSize, MPI_DOUBLE, rank+1, 0,
                         input + idx3_loc(0,0,local_nz+1,nx,ny), planeSize, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, &status);
        } else {
            size_t offset_last = idx3_loc(0,0,local_nz,nx,ny);
            size_t offset_top = idx3_loc(0,0,local_nz+1,nx,ny);
            #pragma omp parallel for
            for (size_t i = 0; i < planeSize; ++i) input[offset_top + i] = input[offset_last + i];
        }

        // Launch CUDA kernel over interior
        dim3 block(16, 4, 4);
        dim3 grid((nx + block.x - 3) / block.x, (ny + block.y - 3) / block.y, (local_nz_with_halo + block.z - 3) / block.z);
        stencilKernel<<<grid, block>>>(input, output, (int)nx, (int)ny, local_nz_with_halo, z_start, (int)nz);
        cudaDeviceSynchronize();

        // Copy boundary values (x,y edges and global z boundaries) on host using OpenMP
        #pragma omp parallel for collapse(3)
        for (int z = 0; z < local_nz_with_halo; ++z) {
            int global_z = z_start + (z - 1);
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || global_z == 0 || global_z == (int)nz-1) {
                        size_t idx = idx3_loc(x,y,z,nx,ny);
                        output[idx] = input[idx];
                    }
                }
            }
        }

        // proceed to next iteration
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        double duration = (t1 - t0) * 1000.0; // ms
        printf("Computation time: %.0f ms\n", duration);
        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / ((t1 - t0)) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grids (exclude halos) to rank 0 for validation/printing
    size_t local_real_size = (size_t)local_nz * planeSize;
    std::vector<int> recvcounts(world);
    std::vector<int> displs(world);
    std::vector<double> gatherbuf;
    if (rank == 0) {
        for (int r = 0; r < world; ++r) {
            int r_nz = base + (r < rem ? 1 : 0);
            recvcounts[r] = r_nz * (int)planeSize;
        }
        displs[0] = 0;
        for (int r = 1; r < world; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
        gatherbuf.resize((size_t)nz * planeSize);
    }

    // Prepare send buffer pointer to first real slice
    double* finalLocal = ((iterations % 2 == 0) ? gridA : gridB) + idx3_loc(0,0,1,nx,ny);
    MPI_Gatherv(finalLocal, (int)local_real_size, MPI_DOUBLE,
                rank==0 ? gatherbuf.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Reconstruct full grid for validation/printing
        std::vector<Real> finalGrid((size_t)nx * ny * nz);
        // gatherbuf already ordered by z-slices per rank contiguous
        memcpy(finalGrid.data(), gatherbuf.data(), finalGrid.size() * sizeof(double));

        if (printResults) {
            print_results(finalGrid, "Grid");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            // simple sanity check for NaN/Inf using OpenMP
            #pragma omp parallel for reduction(&&:valid)
            for (size_t i = 0; i < finalGrid.size(); ++i) {
                double v = finalGrid[i];
                if (std::isnan(v) || std::isinf(v)) valid = false;
            }
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    cudaFree(gridA);
    cudaFree(gridB);

    MPI_Finalize();
    return 0;
}
