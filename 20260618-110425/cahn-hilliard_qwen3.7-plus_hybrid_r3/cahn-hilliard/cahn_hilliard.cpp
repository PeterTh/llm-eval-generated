#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char errstr[MPI_MAX_ERROR_STRING]; \
        int errlen; \
        MPI_Error_string(mpi_err, errstr, &errlen); \
        fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index calculation (device and host)
__device__ __host__ __forceinline__ size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: Initialize concentration field
// Data layout: array has (local_nz+2) z-slices; index 0 = lower ghost, 1..local_nz = owned, local_nz+1 = upper ghost
__global__ void initConcentrationKernel(double* __restrict__ c,
                                         size_t nx, size_t ny, size_t local_nz,
                                         size_t z_offset, size_t vol) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t global_z = z + z_offset;
        size_t linear_id = global_z * (nx * ny) + y * nx + x;
        double pseudo = ((((linear_id + 1) * 1299709) % vol) / (double)vol);
        c[idx3(x, y, z + 1, nx, ny)] = -1.0 + 2.0 * pseudo;
    }
}

// CUDA kernel: Compute chemical potential (fused with Laplacian of c)
// Reads c (with ghost layers), writes mu (at owned locations z=1..local_nz)
__global__ void computeMuKernel(const double* __restrict__ c,
                                 double* __restrict__ mu,
                                 size_t nx, size_t ny, size_t local_nz,
                                 double inv_dx2, double inv_dy2, double inv_dz2,
                                 double gamma, double e_AA, double e_BB, double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t za = z + 1;
        size_t cidx = idx3(x, y, za, nx, ny);

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t yn = (y > 0) ? y - 1 : 0;

        double cv = c[cidx];

        double cxx = (c[idx3(xp, y, za, nx, ny)] + c[idx3(xn, y, za, nx, ny)] - 2.0 * cv) * inv_dx2;
        double cyy = (c[idx3(x, yp, za, nx, ny)] + c[idx3(x, yn, za, nx, ny)] - 2.0 * cv) * inv_dy2;
        double czz = (c[idx3(x, y, za + 1, nx, ny)] + c[idx3(x, y, za - 1, nx, ny)] - 2.0 * cv) * inv_dz2;

        double lap = cxx + cyy + czz;

        mu[cidx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma * lap;
    }
}

// CUDA kernel: Cahn-Hilliard update (fused with Laplacian of mu)
// Reads mu (with ghost layers) and cold (at owned locations), writes cnew (at owned locations)
__global__ void updateKernel(double* __restrict__ cnew,
                              const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              size_t nx, size_t ny, size_t local_nz,
                              double DtD,
                              double inv_dx2, double inv_dy2, double inv_dz2) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t za = z + 1;
        size_t cidx = idx3(x, y, za, nx, ny);

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t yn = (y > 0) ? y - 1 : 0;

        double mu_c = mu[cidx];

        double muxx = (mu[idx3(xp, y, za, nx, ny)] + mu[idx3(xn, y, za, nx, ny)] - 2.0 * mu_c) * inv_dx2;
        double muyy = (mu[idx3(x, yp, za, nx, ny)] + mu[idx3(x, yn, za, nx, ny)] - 2.0 * mu_c) * inv_dy2;
        double muzz = (mu[idx3(x, y, za + 1, nx, ny)] + mu[idx3(x, y, za - 1, nx, ny)] - 2.0 * mu_c) * inv_dz2;

        double lap_mu = muxx + muyy + muzz;

        cnew[cidx] = cold[cidx] + DtD * lap_mu;
    }
}

// CUDA kernel: Fill ghost layers with clamped boundary conditions at global domain boundaries
__global__ void fillBoundaryGhostKernel(double* __restrict__ data,
                                         size_t nx, size_t ny, size_t local_nz,
                                         bool do_lower, bool do_upper) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny) {
        if (do_lower) {
            data[idx3(x, y, 0, nx, ny)] = data[idx3(x, y, 1, nx, ny)];
        }
        if (do_upper) {
            data[idx3(x, y, local_nz + 1, nx, ny)] = data[idx3(x, y, local_nz, nx, ny)];
        }
    }
}

// Halo exchange: exchanges ghost layers between MPI neighbors along Z
// Uses pinned host buffers for D2H/H2D transfers
void haloExchange(double* d_data, double* h_send, double* h_recv,
                  size_t nx, size_t ny, size_t local_nz,
                  int rank_up, int rank_down, cudaStream_t stream) {
    size_t slice_elems = nx * ny;
    size_t slice_bytes = slice_elems * sizeof(double);

    // Exchange with upper neighbor:
    // Send my highest owned slice (z=local_nz) to rank_up
    // Receive upper ghost (z=local_nz+1) from rank_up (its lowest owned)
    CUDA_CHECK(cudaMemcpyAsync(h_send, d_data + local_nz * slice_elems,
                                slice_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Sendrecv(h_send, (int)slice_elems, MPI_DOUBLE, rank_up, 0,
                 h_recv, (int)slice_elems, MPI_DOUBLE, rank_up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    CUDA_CHECK(cudaMemcpyAsync(d_data + (local_nz + 1) * slice_elems, h_recv,
                                slice_bytes, cudaMemcpyHostToDevice, stream));

    // Exchange with lower neighbor:
    // Send my lowest owned slice (z=1) to rank_down
    // Receive lower ghost (z=0) from rank_down (its highest owned)
    CUDA_CHECK(cudaMemcpyAsync(h_send, d_data + 1 * slice_elems,
                                slice_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Sendrecv(h_send, (int)slice_elems, MPI_DOUBLE, rank_down, 1,
                 h_recv, (int)slice_elems, MPI_DOUBLE, rank_down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    CUDA_CHECK(cudaMemcpyAsync(d_data + 0 * slice_elems, h_recv,
                                slice_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Parallel NaN/Inf check and min/max reduction using OpenMP
    bool hasNanInf = false;
    double minVal = c[0];
    double maxVal = c[0];

    #pragma omp parallel for reduction(||:hasNanInf) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            hasNanInf = true;
        }
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }

    if (hasNanInf) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    // Initialize MPI with thread support
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int my_rank, comm_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));

    // Parse command line arguments (all ranks get same args via mpirun)
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = (size_t)atoi(argv[++i]);
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
        } else {
            if (my_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Domain decomposition along Z axis
    size_t base_nz = nz / comm_size;
    size_t extra = nz % comm_size;
    size_t local_nz, z_offset;
    if ((size_t)my_rank < extra) {
        local_nz = base_nz + 1;
        z_offset = (size_t)my_rank * (base_nz + 1);
    } else {
        local_nz = base_nz;
        z_offset = extra * (base_nz + 1) + ((size_t)my_rank - extra) * base_nz;
    }

    int rank_up = (my_rank < comm_size - 1) ? my_rank + 1 : MPI_PROC_NULL;
    int rank_down = (my_rank > 0) ? my_rank - 1 : MPI_PROC_NULL;
    bool is_lower_boundary = (my_rank == 0);
    bool is_upper_boundary = (my_rank == comm_size - 1);

    if (my_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d, OpenMP threads: %d\n", comm_size, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU device based on rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    CUDA_CHECK(cudaSetDevice(my_rank % num_gpus));

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma_val = 0.5;
    const double D = 1.0;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double DtD = dt * D;

    size_t slice_elems = nx * ny;
    size_t array_nz = local_nz + 2; // including ghost layers
    size_t array_size = slice_elems * array_nz;

    // Allocate device memory
    double *d_c, *d_mu, *d_cnew;
    CUDA_CHECK(cudaMalloc(&d_c, array_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, array_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, array_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_c, 0, array_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, array_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, array_size * sizeof(double)));

    // Allocate pinned host buffers for halo exchange
    double *h_send, *h_recv;
    CUDA_CHECK(cudaMallocHost(&h_send, slice_elems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv, slice_elems * sizeof(double)));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize concentration field on GPU
    if (my_rank == 0) printf("Initializing concentration field...\n");

    {
        dim3 block(16, 16, 1);
        dim3 grid((unsigned int)((nx + 15) / 16), (unsigned int)((ny + 15) / 16), (unsigned int)local_nz);
        initConcentrationKernel<<<grid, block, 0, stream>>>(
            d_c, nx, ny, local_nz, z_offset, nx * ny * nz);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Fill initial ghost layers for boundary ranks
    {
        dim3 block2d(16, 16);
        dim3 grid2d((unsigned int)((nx + 15) / 16), (unsigned int)((ny + 15) / 16));
        fillBoundaryGhostKernel<<<grid2d, block2d, 0, stream>>>(
            d_c, nx, ny, local_nz, is_lower_boundary, is_upper_boundary);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Run simulation
    if (my_rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start_time = MPI_Wtime();

    dim3 block3d(16, 16, 1);
    dim3 grid3d((unsigned int)((nx + 15) / 16), (unsigned int)((ny + 15) / 16), (unsigned int)local_nz);
    dim3 block2d(16, 16);
    dim3 grid2d((unsigned int)((nx + 15) / 16), (unsigned int)((ny + 15) / 16));

    double* d_cold_ptr = d_c;
    double* d_cnew_ptr = d_cnew;

    for (int t = 0; t < iterations; ++t) {
        // Halo exchange for concentration
        haloExchange(d_cold_ptr, h_send, h_recv, nx, ny, local_nz,
                     rank_up, rank_down, stream);

        // Fill clamped BC ghosts for boundary ranks
        if (is_lower_boundary || is_upper_boundary) {
            fillBoundaryGhostKernel<<<grid2d, block2d, 0, stream>>>(
                d_cold_ptr, nx, ny, local_nz, is_lower_boundary, is_upper_boundary);
        }

        // Compute chemical potential (fused with Laplacian of c)
        computeMuKernel<<<grid3d, block3d, 0, stream>>>(
            d_cold_ptr, d_mu, nx, ny, local_nz,
            inv_dx2, inv_dy2, inv_dz2,
            gamma_val, e_AA, e_BB, e_AB);

        // Halo exchange for chemical potential
        haloExchange(d_mu, h_send, h_recv, nx, ny, local_nz,
                     rank_up, rank_down, stream);

        // Fill clamped BC ghosts for mu at boundary ranks
        if (is_lower_boundary || is_upper_boundary) {
            fillBoundaryGhostKernel<<<grid2d, block2d, 0, stream>>>(
                d_mu, nx, ny, local_nz, is_lower_boundary, is_upper_boundary);
        }

        // Update concentration (fused with Laplacian of mu)
        updateKernel<<<grid3d, block3d, 0, stream>>>(
            d_cnew_ptr, d_cold_ptr, d_mu, nx, ny, local_nz,
            DtD, inv_dx2, inv_dy2, inv_dz2);

        // Swap pointers
        double* tmp = d_cold_ptr;
        d_cold_ptr = d_cnew_ptr;
        d_cnew_ptr = tmp;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // Copy result from device to host (owned slices only)
    std::vector<double> local_result(local_nz * slice_elems);
    for (size_t z = 0; z < local_nz; ++z) {
        CUDA_CHECK(cudaMemcpyAsync(local_result.data() + z * slice_elems,
                                    d_cold_ptr + (z + 1) * slice_elems,
                                    slice_elems * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Gather results to rank 0
    size_t global_size = nx * ny * nz;
    std::vector<double> global_result;
    if (my_rank == 0) {
        global_result.resize(global_size);
    }

    // Compute displacements for Gatherv
    std::vector<int> recv_counts(comm_size);
    std::vector<int> displs(comm_size);
    for (int r = 0; r < comm_size; ++r) {
        size_t r_base = nz / comm_size;
        size_t r_extra = nz % comm_size;
        size_t r_local_nz = r_base + ((size_t)r < r_extra ? 1 : 0);
        recv_counts[r] = (int)(r_local_nz * slice_elems);
        size_t r_offset;
        if ((size_t)r < r_extra) {
            r_offset = (size_t)r * (r_base + 1);
        } else {
            r_offset = r_extra * (r_base + 1) + ((size_t)r - r_extra) * r_base;
        }
        displs[r] = (int)(r_offset * slice_elems);
    }

    MPI_CHECK(MPI_Gatherv(local_result.data(), (int)(local_nz * slice_elems), MPI_DOUBLE,
                           global_result.data(), recv_counts.data(), displs.data(),
                           MPI_DOUBLE, 0, MPI_COMM_WORLD));

    // Report timing and performance from rank 0
    if (my_rank == 0) {
        long duration_ms = (long)(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        double cellUpdates = (double)global_size * iterations;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        // Print results for external validation
        if (printResults) {
            print_results(global_result, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_result, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));

    MPI_CHECK(MPI_Finalize());
    return 0;
}
