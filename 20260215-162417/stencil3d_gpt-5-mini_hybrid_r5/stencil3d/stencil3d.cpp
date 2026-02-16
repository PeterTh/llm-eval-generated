#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <omp.h>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation for contiguous host buffers (z major)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize local buffer using global z offset so results match single-rank run
void initializeGridLocal(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t z_offset, const size_t nz_total) {
    // grid includes halos: layout z in [0..local_nz+1]
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz + 2; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                long long global_z = (long long)z + (long long)z_offset - 1; // halo at z=0 corresponds to global z = z_offset-1
                size_t idx = idx3(x, y, z, nx, ny);
                if (global_z < 0 || global_z >= (long long)nz_total) {
                    grid[idx] = 0.0; // halo or out-of-range; will be overwritten by MPI
                } else {
                    size_t global_idx = (size_t)global_z * (nx * ny) + y * nx + x;
                    grid[idx] = (global_idx % 19) * 1.0;
                }
            }
        }
    }
}

#if defined(__CUDACC__)
// CUDA kernel for stencil on interior points (excluding global boundaries and halos)
extern "C" __global__ void stencil_kernel(const Real* input, Real* output, int nx, int ny, int local_nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1; // skip x=0
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1; // skip y=0
    int z = blockIdx.z * blockDim.z + threadIdx.z + 1; // local z in [1..local_nz]

    if (x >= nx-1 || y >= ny-1 || z > local_nz) return;

    int slice = nx * ny;
    int idx = z * slice + y * nx + x;

    Real center = input[idx];
    Real left = input[idx - 1];
    Real right = input[idx + 1];
    Real front = input[idx - nx];
    Real back = input[idx + nx];
    Real bottom = input[idx - slice];
    Real top = input[idx + slice];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}
#endif

bool validateResultFull(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    // reuse previous validation semantics
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
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
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse args (only on rank 0 will be printed later)
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
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    size_t gridSize = nx * ny * nz;

    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition along Z
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    size_t z_offset = rank * base + std::min<size_t>(rank, rem);

    // local buffers include 2 halo planes
    size_t slice = nx * ny;
    size_t local_with_halo = (local_nz + 2) * slice;

    std::vector<Real> grid1(local_with_halo);
    std::vector<Real> grid2(local_with_halo);

    // Initialize local grid based on global coordinates
    // halos are filled with neighbors later; initialize halos conservatively
    initializeGridLocal(grid1, nx, ny, local_nz, z_offset, nz);
    initializeGridLocal(grid2, nx, ny, local_nz, z_offset, nz);

#if defined(__CUDACC__)
    // Setup CUDA device per rank (map rank to device 0..)
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    int device_id = 0;
    if (device_count > 0) {
        device_id = rank % device_count;
        cudaSetDevice(device_id);
    }

    // Allocate device buffers
    Real* d_input = nullptr;
    Real* d_output = nullptr;
    cudaMalloc(&d_input, local_with_halo * sizeof(Real));
    cudaMalloc(&d_output, local_with_halo * sizeof(Real));
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Prepare MPI datatypes for a single slice
    MPI_Datatype slice_type;
    MPI_Type_contiguous((int)slice, MPI_DOUBLE, &slice_type);
    MPI_Type_commit(&slice_type);

    int prev = (rank - 1 + size) % size;
    int next = (rank + 1) % size;

#if defined(__CUDACC__)
    // Kernel launch parameters
    dim3 block(16, 8, 4);
    dim3 grid((nx + block.x - 3) / block.x, (ny + block.y - 3) / block.y, (local_nz + block.z - 1) / block.z);
#endif

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halo slices: send/recv inner slices (z=1 and z=local_nz)
        MPI_Sendrecv(&grid1[idx3(0,0,1,nx,ny)], 1, slice_type, prev, 0,
                     &grid1[idx3(0,0,0,nx,ny)], 1, slice_type, prev, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&grid1[idx3(0,0,local_nz,nx,ny)], 1, slice_type, next, 1,
                     &grid1[idx3(0,0,local_nz+1,nx,ny)], 1, slice_type, next, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#if defined(__CUDACC__)
        // copy to device
        cudaMemcpy(d_input, grid1.data(), local_with_halo * sizeof(Real), cudaMemcpyHostToDevice);

        // launch kernel to compute interior points
        stencil_kernel<<<grid, block>>>(d_input, d_output, (int)nx, (int)ny, (int)local_nz);
        cudaDeviceSynchronize();

        // copy back
        cudaMemcpy(grid2.data(), d_output, local_with_halo * sizeof(Real), cudaMemcpyDeviceToHost);
#else
        // CPU fallback compute using OpenMP
        #pragma omp parallel for collapse(3)
        for (size_t z = 1; z <= local_nz; ++z) {
            for (size_t y = 1; y < ny-1; ++y) {
                for (size_t x = 1; x < nx-1; ++x) {
                    size_t idx = idx3(x,y,z,nx,ny);
                    Real center = grid1[idx];
                    Real left = grid1[idx3(x-1,y,z,nx,ny)];
                    Real right = grid1[idx3(x+1,y,z,nx,ny)];
                    Real front = grid1[idx3(x,y-1,z,nx,ny)];
                    Real back = grid1[idx3(x,y+1,z,nx,ny)];
                    Real bottom = grid1[idx3(x,y,z-1,nx,ny)];
                    Real top = grid1[idx3(x,y,z+1,nx,ny)];
                    grid2[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
#endif

        // copy boundary values on host (x/y boundaries and global z boundaries)
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < local_nz + 2; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx-1 || y == 0 || y == ny-1) {
                        size_t idx = idx3(x,y,z,nx,ny);
                        grid2[idx] = grid1[idx];
                    }
                    // global z boundaries: if this rank owns the global boundary slice, copy halo accordingly
                    if ((z == 1 && z_offset == 0) || (z == local_nz && (z_offset + local_nz) == nz)) {
                        size_t idx = idx3(x,y,z,nx,ny);
                        grid2[idx] = grid1[idx];
                    }
                }
            }
        }

        // swap buffers
        std::swap(grid1, grid2);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    if (rank == 0) {
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 for printing/validation
    // prepare counts and displacements
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            size_t r_local = base + (r < (int)rem ? 1 : 0);
            recvcounts[r] = (int)(r_local * slice);
        }
        displs[0] = 0;
        for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];
    }

    std::vector<Real> gathered;
    if (rank == 0) gathered.resize(gridSize);

    // send pointer to inner region (exclude halos)
    MPI_Gatherv(&grid1[idx3(0,0,1,nx,ny)], (int)(local_nz*slice), MPI_DOUBLE,
                gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        print_results(gathered, "Grid");
    }

    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResultFull(gathered, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

#if defined(__CUDACC__)
    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
#endif
    MPI_Type_free(&slice_type);
    MPI_Finalize();
    return 0;
}
