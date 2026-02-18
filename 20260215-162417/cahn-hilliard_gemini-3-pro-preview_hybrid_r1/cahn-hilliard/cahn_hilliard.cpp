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

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
__device__ double computeLaplacian(const double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz_global,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z,
                        const size_t z_offset, const size_t local_nz) {
    // x and y are local and global indices (no decomposition in x, y)
    // z is the local index in the array (including ghost layers)
    // z_global_idx is the global index of the current cell
    
    // Determine neighbors in local array
    // Since we have ghost layers at z=0 and z=local_nz+1 (if applicable), we can access z-1 and z+1 freely 
    // UNLESS we are at the global boundary.
    
    size_t z_global_idx = z_offset + (z - 1); // z starts at 1 for the first real plane

    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t xn = (x > 0) ? x - 1 : 0;
    
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t yn = (y > 0) ? y - 1 : 0;
    
    // For Z, we use the local array which has ghost cells.
    // If we are at global boundary, we clamp.
    // Otherwise, we use the neighbor in the local array (which might be a ghost cell).
    
    size_t zp_local = z + 1;
    size_t zn_local = z - 1;

    // Apply global boundary conditions for Z
    if (z_global_idx == 0) zn_local = z; // Clamp at global bottom
    if (z_global_idx == nz_global - 1) zp_local = z; // Clamp at global top
    
    // Note: c includes ghost layers. z goes from 1 to local_nz.
    const double c_center = c[idx3(x, y, z, nx, ny)];
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * c_center) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * c_center) / (dy * dy);
    const double czz = (c[idx3(x, y, zp_local, nx, ny)] + c[idx3(x, y, zn_local, nx, ny)] - 2.0 * c_center) / (dz * dz);
    
    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz_global,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                               const size_t z_offset, const size_t local_nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z; // z is 0 to local_nz-1

    if (x >= nx || y >= ny || z >= local_nz) return;

    // Shift z to point to the correct plane in the array (skipping bottom ghost layer)
    size_t z_array = z + 1;

    const size_t idx = idx3(x, y, z_array, nx, ny);
    const double cv = c[idx];
    
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
             + 3.0 * cv + cv * cv * cv
             - gamma * computeLaplacian(c, nx, ny, nz_global, dx, dy, dz, x, y, z_array, z_offset, local_nz);
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz_global,
                                         const double D, const double dt, const double dx, const double dy, const double dz,
                                         const size_t z_offset, const size_t local_nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= local_nz) return;

    size_t z_array = z + 1;
    const size_t idx = idx3(x, y, z_array, nx, ny);
    
    cnew[idx] = cold[idx] + dt * D * 
               computeLaplacian(mu, nx, ny, nz_global, dx, dy, dz, x, y, z_array, z_offset, local_nz);
}

// Helper to update halos
void updateHalos(double* d_data, double* h_send_top, double* h_send_bottom, double* h_recv_top, double* h_recv_bottom,
                 const size_t nx, const size_t ny, const size_t local_nz,
                 const int rank, const int size) {
    size_t plane_size = nx * ny;
    size_t bytes = plane_size * sizeof(double);

    // Copy from device to host buffers
    // Top inner plane (z = local_nz) -> send to rank + 1
    // Bottom inner plane (z = 1) -> send to rank - 1
    CHECK_CUDA(cudaMemcpy(h_send_top, d_data + (local_nz * plane_size), bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_send_bottom, d_data + (1 * plane_size), bytes, cudaMemcpyDeviceToHost));
    
    // MPI Exchange
    MPI_Request reqs[4];
    int nreqs = 0;
    
    int top_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int bottom_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;

    // Send to top, receive from top
    MPI_Isend(h_send_top, plane_size, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    MPI_Irecv(h_recv_top, plane_size, MPI_DOUBLE, top_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);

    // Send to bottom, receive from bottom
    MPI_Isend(h_send_bottom, plane_size, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
    MPI_Irecv(h_recv_bottom, plane_size, MPI_DOUBLE, bottom_neighbor, 0, MPI_COMM_WORLD, &reqs[nreqs++]);

    MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

    // Copy from host buffers to device ghost layers
    // Top ghost layer (z = local_nz + 1)
    if (top_neighbor != MPI_PROC_NULL) {
        CHECK_CUDA(cudaMemcpy(d_data + ((local_nz + 1) * plane_size), h_recv_top, bytes, cudaMemcpyHostToDevice));
    }
    // Bottom ghost layer (z = 0)
    if (bottom_neighbor != MPI_PROC_NULL) {
        CHECK_CUDA(cudaMemcpy(d_data, h_recv_bottom, bytes, cudaMemcpyHostToDevice));
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool valid = true;
    double minVal = 1e30;
    double maxVal = -1e30;

    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Ranks: %d\n", size);
        
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("CUDA Devices: %d\n", deviceCount);
    }
    
    // Set device
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    cudaSetDevice(rank % num_devices);

    // Decomposition
    size_t local_nz = nz / size;
    size_t remainder = nz % size;
    size_t z_offset = rank * local_nz + std::min((size_t)rank, remainder);
    if ((size_t)rank < remainder) local_nz++;
    
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
    
    size_t plane_size = nx * ny;
    // +2 for ghost layers (top and bottom)
    size_t local_grid_size_with_halo = plane_size * (local_nz + 2); 
    size_t local_grid_size_without_halo = plane_size * local_nz;

    // Host buffers
    std::vector<double> h_cold;
    std::vector<double> h_final;
    
    if (rank == 0 || validate || printResults) {
        h_cold.resize(nz * plane_size); // Full grid on rank 0 for init/verification
    }
    
    // Initialize concentration field globally on rank 0 (or all ranks for simplicity of code)
    // To match original, we should generate the whole field.
    // Efficient way: each rank generates its part.
    // Since pseudo random depends on linear index, it is deterministic.
    
    std::vector<double> h_local_c(local_grid_size_without_halo);
    
    // Use OpenMP to initialize locally
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t global_z = z_offset + z;
                size_t idx = idx3(x, y, z, nx, ny);
                size_t linear_id = global_z * (nx * ny) + y * nx + x;
                size_t vol = nx * ny * nz_global; // Use global volume
                double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                h_local_c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Device allocations
    double *d_cold, *d_cnew, *d_mu;
    CHECK_CUDA(cudaMalloc(&d_cold, local_grid_size_with_halo * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_cnew, local_grid_size_with_halo * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_mu, local_grid_size_with_halo * sizeof(double)));

    // Copy initial data to device (offset by 1 plane for bottom ghost)
    CHECK_CUDA(cudaMemcpy(d_cold + plane_size, h_local_c.data(), local_grid_size_without_halo * sizeof(double), cudaMemcpyHostToDevice));

    // Halo exchange buffers (Host)
    double *h_send_top, *h_send_bottom, *h_recv_top, *h_recv_bottom;
    CHECK_CUDA(cudaMallocHost(&h_send_top, plane_size * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_send_bottom, plane_size * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_recv_top, plane_size * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_recv_bottom, plane_size * sizeof(double)));

    // Fill initial halos
    // For initial step, we need to make sure boundaries are correct.
    // We can do a halo exchange on d_cold before starting loop.
    updateHalos(d_cold, h_send_top, h_send_bottom, h_recv_top, h_recv_bottom, nx, ny, local_nz, rank, size);

    // Initialize cnew same as cold
    CHECK_CUDA(cudaMemcpy(d_cnew, d_cold, local_grid_size_with_halo * sizeof(double), cudaMemcpyDeviceToDevice));
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (local_nz + block.z - 1) / block.z);

    for (int t = 0; t < iterations; ++t) {
        // Compute Chemical Potential
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, z_offset, local_nz);
        CHECK_CUDA(cudaGetLastError());
        
        // Halo exchange for mu
        updateHalos(d_mu, h_send_top, h_send_bottom, h_recv_top, h_recv_bottom, nx, ny, local_nz, rank, size);

        // Update Concentration
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz, z_offset, local_nz);
        CHECK_CUDA(cudaGetLastError());
        
        // Halo exchange for cnew (which becomes cold next iter)
        updateHalos(d_cnew, h_send_top, h_send_bottom, h_recv_top, h_recv_bottom, nx, ny, local_nz, rank, size);
        
        // Swap pointers
        std::swap(d_cold, d_cnew);
    }
    
    // Ensure GPU is done
    CHECK_CUDA(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)(nx * ny * nz) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results for validation/printing
    if (validate || printResults) {
        // Copy back from device
        CHECK_CUDA(cudaMemcpy(h_local_c.data(), d_cold + plane_size, local_grid_size_without_halo * sizeof(double), cudaMemcpyDeviceToHost));
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int my_count = local_grid_size_without_halo;
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
            h_final.resize(nz * plane_size);
        }
        
        MPI_Gatherv(h_local_c.data(), my_count, MPI_DOUBLE, 
                    h_final.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            if (printResults) {
                print_results(h_final, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(h_final, nx, ny, nz);
                if (valid) printf("Validation: PASSED\n");
                else printf("Validation: FAILED\n");
            }
        }
    }

    CHECK_CUDA(cudaFree(d_cold));
    CHECK_CUDA(cudaFree(d_cnew));
    CHECK_CUDA(cudaFree(d_mu));
    CHECK_CUDA(cudaFreeHost(h_send_top));
    CHECK_CUDA(cudaFreeHost(h_send_bottom));
    CHECK_CUDA(cudaFreeHost(h_recv_top));
    CHECK_CUDA(cudaFreeHost(h_recv_bottom));

    MPI_Finalize();
    return 0;
}
