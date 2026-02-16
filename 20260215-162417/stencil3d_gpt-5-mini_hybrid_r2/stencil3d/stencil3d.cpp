#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#if defined(__has_include)
  #if __has_include(<cuda_runtime.h>)
    #include <cuda_runtime.h>
    #define HAVE_CUDA_RUNTIME 1
  #else
    #define HAVE_CUDA_RUNTIME 0
  #endif
#else
  #define HAVE_CUDA_RUNTIME 0
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation for contiguous x fastest
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize local portion of grid; global_z_offset is the global index of local z=0 (excluding halos)
void initializeLocalGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t local_nz, const size_t global_z_offset) {
    // grid has size nx * ny * (local_nz + 2) to include halos
    const size_t lz_with_halo = local_nz + 2;
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < lz_with_halo; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = global_z_offset + (z == 0 ? 0 : (z == lz_with_halo - 1 ? local_nz - 1 : z - 1));
                size_t idx = idx3(x, y, z, nx, ny);
                // Use a deterministic initialization that matches original when gathered
                grid[idx] = ((global_z * nx * ny + y * nx + x) % 19) * 1.0;
            }
        }
    }
}

// CUDA kernel for stencil on local domain (including halos). Works on interior points only (z in [1, local_nz])
#if HAVE_CUDA_RUNTIME || defined(__CUDACC__)
extern "C" __global__ void stencil_kernel(const Real* input, Real* output, size_t nx, size_t ny, size_t lz_with_halo) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z; // local index including halo

    if (x >= 1 && x < nx-1 && y >= 1 && y < ny-1 && z >= 1 && z < lz_with_halo-1) {
        size_t idx = z * (nx * ny) + y * nx + x;
        Real center = input[idx];
        Real left = input[idx - 1];
        Real right = input[idx + 1];
        Real front = input[idx - nx];
        Real back = input[idx + nx];
        Real bottom = input[idx - (nx * ny)];
        Real top = input[idx + (nx * ny)];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}
#endif

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid.empty() ? 0 : grid[0];
    Real maxVal = grid.empty() ? 0 : grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 1e12 || minVal < -1e12) {
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

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Decompose along Z dimension
    size_t base = (nz / world_size) * world_rank + std::min((size_t)world_rank, nz % world_size);
    size_t base_next = (nz / world_size) * (world_rank + 1) + std::min((size_t)(world_rank + 1), nz % world_size);
    size_t local_nz = (base_next > base) ? (base_next - base) : 0;

    if (local_nz == 0) {
        if (world_rank == 0) printf("No work assigned to rank %d; exiting.\n", world_rank);
        MPI_Finalize();
        return 0;
    }

    if (world_rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("MPI ranks: %d, iterations: %d\n", world_size, iterations);
    }

    // Each local buffer includes two halo planes
    const size_t lz_with_halo = local_nz + 2;
    std::vector<Real> local_grid_in(nx * ny * lz_with_halo);
    std::vector<Real> local_grid_out(nx * ny * lz_with_halo);

    // Initialize local grid deterministically so that gather matches original
    initializeLocalGrid(local_grid_in, nx, ny, local_nz, base);

    // Ensure boundaries (global) are set in halos for global boundaries
    if (base == 0) {
        // global z == 0 corresponds to local z==1's bottom halo at index 0
        // already initialized by initializeLocalGrid to correct values
    }
    if (base + local_nz == nz) {
        // top global boundary similarly in halo at lz_with_halo-1
    }

    // Halo exchange buffers (one plane)
    std::vector<Real> send_lower(nx * ny), send_upper(nx * ny), recv_lower(nx * ny), recv_upper(nx * ny);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

#ifdef __CUDACC__
    // Allocate device buffers when compiling with NVCC
    Real* d_in = nullptr;
    Real* d_out = nullptr;
    size_t bytes = nx * ny * lz_with_halo * sizeof(Real);
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    dim3 block(16, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (lz_with_halo + block.z - 1) / block.z);
#else
    double bytes = 0; (void)bytes;
#endif

    for (int iter = 0; iter < iterations; ++iter) {
        // Prepare halos to send
        #pragma omp parallel for
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) send_lower[y * nx + x] = local_grid_in[idx3(x, y, 1, nx, ny)];
        #pragma omp parallel for
        for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) send_upper[y * nx + x] = local_grid_in[idx3(x, y, local_nz, nx, ny)];

        int prev = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
        int next = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

        MPI_Request reqs[4];
        MPI_Irecv(recv_lower.data(), nx * ny, MPI_DOUBLE, prev, 10, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(send_lower.data(), nx * ny, MPI_DOUBLE, prev, 11, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(recv_upper.data(), nx * ny, MPI_DOUBLE, next, 11, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(send_upper.data(), nx * ny, MPI_DOUBLE, next, 10, MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (prev != MPI_PROC_NULL) {
            #pragma omp parallel for
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) local_grid_in[idx3(x, y, 0, nx, ny)] = recv_lower[y * nx + x];
        }
        if (next != MPI_PROC_NULL) {
            #pragma omp parallel for
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) local_grid_in[idx3(x, y, local_nz + 1, nx, ny)] = recv_upper[y * nx + x];
        }

#ifdef __CUDACC__
        // Copy to device
        cudaMemcpy(d_in, local_grid_in.data(), bytes, cudaMemcpyHostToDevice);

        // Launch kernel
        stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, lz_with_halo);
        cudaDeviceSynchronize();

        // Copy back
        cudaMemcpy(local_grid_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

        // Copy boundaries (x/y/global z boundaries) from input to output to preserve original semantics
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < lz_with_halo; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || 
                        (base == 0 && z == 1 && (base + 0) == 0 && z == 1 && z == 1 && (0 == 0)) ||
                        (base + local_nz == nz && z == lz_with_halo - 2 && (base + local_nz - 1) == nz - 1)) {
                        // Preserve global boundaries and x/y boundaries
                        local_grid_out[idx3(x, y, z, nx, ny)] = local_grid_in[idx3(x, y, z, nx, ny)];
                    }
                }
            }
        }
#else
        // CPU fallback using OpenMP parallel loops
        #pragma omp parallel for collapse(3)
        for (size_t z = 1; z < lz_with_halo - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t x = 1; x < nx - 1; ++x) {
                    size_t idx = idx3(x, y, z, nx, ny);
                    Real center = local_grid_in[idx];
                    Real left = local_grid_in[idx3(x-1, y, z, nx, ny)];
                    Real right = local_grid_in[idx3(x+1, y, z, nx, ny)];
                    Real front = local_grid_in[idx3(x, y-1, z, nx, ny)];
                    Real back = local_grid_in[idx3(x, y+1, z, nx, ny)];
                    Real bottom = local_grid_in[idx3(x, y, z-1, nx, ny)];
                    Real top = local_grid_in[idx3(x, y, z+1, nx, ny)];
                    local_grid_out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
        // Copy boundaries
        #pragma omp parallel for collapse(3)
        for (size_t z = 0; z < lz_with_halo; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
                        (base == 0 && z == 1) || (base + local_nz == nz && z == lz_with_halo - 2)) {
                        local_grid_out[idx3(x, y, z, nx, ny)] = local_grid_in[idx3(x, y, z, nx, ny)];
                    }
                }
            }
        }
#endif
        // Swap buffers
        local_grid_in.swap(local_grid_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (world_rank == 0) {
        double duration = (t1 - t0) * 1000.0;
        printf("Computation time: %.0f ms\n", duration);
        double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        double mcups = cellUpdates / ((t1 - t0)) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final data (exclude halos) to rank 0 for validation/printing
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    int elems_per_rank = (int)(nx * ny * local_nz);
    // Build counts and displacements collectively
    std::vector<int> all_local_nz(world_size);
    int my_local_nz = (int)local_nz;
    MPI_Gather(&my_local_nz, 1, MPI_INT, all_local_nz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Real> gathered;
    if (world_rank == 0) {
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            recvcounts[r] = all_local_nz[r] * nx * ny;
            displs[r] = offset;
            offset += recvcounts[r];
        }
        gathered.resize((size_t)offset);
    }

    // Prepare send buffer of local data (exclude halos): copy into contiguous sendbuf
    std::vector<Real> sendbuf(nx * ny * local_nz);
    #pragma omp parallel for
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                sendbuf[idx3(x, y, z, nx, ny)] = local_grid_in[idx3(x, y, z + 1, nx, ny)];
            }
        }
    }

    MPI_Gatherv(sendbuf.data(), (int)sendbuf.size(), MPI_DOUBLE,
                gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        if (printResults) print_results(gathered, "Grid");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(gathered, nx, ny, nz);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

#ifdef __CUDACC__
    cudaFree(d_in);
    cudaFree(d_out);
#endif

    MPI_Finalize();
    return 0;
}
