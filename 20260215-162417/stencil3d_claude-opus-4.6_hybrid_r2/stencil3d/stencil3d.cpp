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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// 3D index calculation
__host__ __device__
inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel: 7-point stencil on interior points of the local subdomain
// local_nz includes 2 ghost layers (one at bottom z=0, one at top z=local_nz-1)
__global__
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   const size_t nx, const size_t ny, const size_t local_nz,
                   const size_t z_start, const size_t z_end) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z + z_start;

    if (x < nx - 1 && y < ny - 1 && z < z_end) {
        size_t i = idx3(x, y, z, nx, ny);
        Real center = input[i];
        Real left   = input[idx3(x-1, y, z, nx, ny)];
        Real right  = input[idx3(x+1, y, z, nx, ny)];
        Real front  = input[idx3(x, y-1, z, nx, ny)];
        Real back   = input[idx3(x, y+1, z, nx, ny)];
        Real bottom = input[idx3(x, y, z-1, nx, ny)];
        Real top    = input[idx3(x, y, z+1, nx, ny)];
        output[i] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// CUDA kernel: copy boundary faces (x=0, x=nx-1, y=0, y=ny-1 and global z boundaries)
__global__
void copyBoundaryKernel(const Real* __restrict__ input, Real* __restrict__ output,
                        const size_t nx, const size_t ny, const size_t local_nz,
                        const bool is_z_bottom_boundary, const bool is_z_top_boundary) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t slice_size = nx * ny;
    size_t total = slice_size * local_nz;
    if (tid >= total) return;

    size_t z = tid / slice_size;
    size_t rem = tid % slice_size;
    size_t y = rem / nx;
    size_t x = rem % nx;

    bool is_boundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
    if (!is_boundary && is_z_bottom_boundary && z == 0) is_boundary = true;
    if (!is_boundary && is_z_top_boundary && z == local_nz - 1) is_boundary = true;

    if (is_boundary) {
        output[tid] = input[tid];
    }
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    (void)nx; (void)ny; (void)nz;
    bool valid = true;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&&:valid)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            valid = false;
        }
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    if (!valid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Assign one GPU per rank (round-robin if more ranks than GPUs)
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA Hybrid)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("MPI ranks: %d, GPUs per rank: 1 (of %d available)\n", num_procs, num_devices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Domain decomposition along Z axis
    // Each rank gets a contiguous slab of Z planes
    size_t base_nz = nz / num_procs;
    size_t remainder = nz % num_procs;

    // z_offset and local z-count in the global grid (without ghost layers)
    size_t local_nz_no_ghost = base_nz + ((size_t)rank < remainder ? 1 : 0);
    size_t z_offset = 0;
    for (int r = 0; r < rank; ++r) {
        z_offset += base_nz + ((size_t)r < remainder ? 1 : 0);
    }

    // With ghost layers: need one extra plane on each side for halo exchange
    // except at global boundaries
    bool has_lower_neighbor = (rank > 0);
    bool has_upper_neighbor = (rank < num_procs - 1);
    size_t ghost_bottom = has_lower_neighbor ? 1 : 0;
    size_t ghost_top = has_upper_neighbor ? 1 : 0;
    size_t local_nz = local_nz_no_ghost + ghost_bottom + ghost_top;

    size_t slice_size = nx * ny;
    size_t local_grid_size = slice_size * local_nz;

    // Determine stencil computation bounds in local coordinates
    // z_comp_start/end are the local z indices where interior stencil can be computed
    // The stencil needs z-1 and z+1, so we need at least z=1 and z=local_nz-2
    // But we also need to respect global boundaries
    size_t z_comp_start, z_comp_end;
    if (has_lower_neighbor) {
        z_comp_start = 1; // ghost layer at z=0, first real data at z=1
    } else {
        z_comp_start = 1; // global boundary at z=0
    }
    if (has_upper_neighbor) {
        z_comp_end = local_nz - 1; // ghost layer at z=local_nz-1
    } else {
        z_comp_end = local_nz - 1; // global boundary at z=local_nz-1
    }

    // Initialize the full grid on rank 0, then scatter
    std::vector<Real> full_grid;
    if (rank == 0) {
        full_grid.resize(nx * ny * nz);
        initializeGrid(full_grid, nx, ny, nz);
    }

    // Local host buffer for this rank's subdomain (with ghost layers)
    std::vector<Real> local_host(local_grid_size);

    // Scatter: distribute Z-slabs from rank 0 to all ranks
    {
        // Compute send counts and displacements for MPI_Scatterv
        std::vector<int> sendcounts(num_procs), displs(num_procs);
        size_t off = 0;
        for (int r = 0; r < num_procs; ++r) {
            size_t rnz = base_nz + ((size_t)r < remainder ? 1 : 0);
            sendcounts[r] = (int)(rnz * slice_size);
            displs[r] = (int)(off * slice_size);
            off += rnz;
        }

        // Receive into the appropriate location (after ghost_bottom planes)
        std::vector<Real> recv_buf(local_nz_no_ghost * slice_size);
        MPI_Scatterv(rank == 0 ? full_grid.data() : nullptr,
                     sendcounts.data(), displs.data(), MPI_DOUBLE,
                     recv_buf.data(), (int)(local_nz_no_ghost * slice_size), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Copy into local buffer at correct offset (after ghost bottom)
        memcpy(local_host.data() + ghost_bottom * slice_size,
               recv_buf.data(),
               local_nz_no_ghost * slice_size * sizeof(Real));
    }

    // Exchange initial ghost layers
    {
        MPI_Request reqs[4];
        int nreqs = 0;

        // Send my bottom real plane to lower neighbor's top ghost
        if (has_lower_neighbor) {
            MPI_Isend(local_host.data() + ghost_bottom * slice_size,
                      (int)slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(local_host.data(),
                      (int)slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        // Send my top real plane to upper neighbor's bottom ghost
        if (has_upper_neighbor) {
            MPI_Isend(local_host.data() + (ghost_bottom + local_nz_no_ghost - 1) * slice_size,
                      (int)slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(local_host.data() + (local_nz - 1) * slice_size,
                      (int)slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);
    }

    // Allocate device memory (double buffering)
    Real *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, local_grid_size * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_output, local_grid_size * sizeof(Real)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_input, local_host.data(), local_grid_size * sizeof(Real),
                          cudaMemcpyHostToDevice));

    // Create CUDA streams for computation and halo exchange overlap
    cudaStream_t stream_compute, stream_halo;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_halo));

    // Host-pinned buffers for async halo exchange
    Real *h_send_bottom, *h_send_top, *h_recv_bottom, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, slice_size * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, slice_size * sizeof(Real)));

    // Determine global boundary flags for boundary copy kernel
    bool is_z_bottom_boundary = !has_lower_neighbor;
    bool is_z_top_boundary = !has_upper_neighbor;

    // Kernel launch configuration
    dim3 block(16, 16, 1);
    dim3 grid_interior(
        ((unsigned int)(nx - 2) + block.x - 1) / block.x,
        ((unsigned int)(ny - 2) + block.y - 1) / block.y,
        ((unsigned int)(z_comp_end - z_comp_start) + block.z - 1) / block.z
    );

    unsigned int boundary_threads = 256;
    unsigned int boundary_blocks = ((unsigned int)local_grid_size + boundary_threads - 1) / boundary_threads;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        Real *cur_in = (iter % 2 == 0) ? d_input : d_output;
        Real *cur_out = (iter % 2 == 0) ? d_output : d_input;

        // Launch stencil kernel for interior
        stencilKernel<<<grid_interior, block, 0, stream_compute>>>(
            cur_in, cur_out, nx, ny, local_nz, z_comp_start, z_comp_end);

        // Copy boundary values
        copyBoundaryKernel<<<boundary_blocks, boundary_threads, 0, stream_compute>>>(
            cur_in, cur_out, nx, ny, local_nz, is_z_bottom_boundary, is_z_top_boundary);

        // Halo exchange: copy halo planes from device to pinned host
        if (has_lower_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(h_send_bottom,
                cur_out + ghost_bottom * slice_size,
                slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_compute));
        }
        if (has_upper_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(h_send_top,
                cur_out + (ghost_bottom + local_nz_no_ghost - 1) * slice_size,
                slice_size * sizeof(Real), cudaMemcpyDeviceToHost, stream_compute));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        // MPI halo exchange
        MPI_Request reqs[4];
        int nreqs = 0;
        if (has_lower_neighbor) {
            MPI_Isend(h_send_bottom, (int)slice_size, MPI_DOUBLE, rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_bottom, (int)slice_size, MPI_DOUBLE, rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (has_upper_neighbor) {
            MPI_Isend(h_send_top, (int)slice_size, MPI_DOUBLE, rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
            MPI_Irecv(h_recv_top, (int)slice_size, MPI_DOUBLE, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Copy received halos to device
        if (has_lower_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(cur_out,
                h_recv_bottom, slice_size * sizeof(Real),
                cudaMemcpyHostToDevice, stream_halo));
        }
        if (has_upper_neighbor) {
            CUDA_CHECK(cudaMemcpyAsync(cur_out + (local_nz - 1) * slice_size,
                h_recv_top, slice_size * sizeof(Real),
                cudaMemcpyHostToDevice, stream_halo));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_halo));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Copy result back to host
    Real *d_final = (iterations % 2 == 0) ? d_input : d_output;
    CUDA_CHECK(cudaMemcpy(local_host.data(), d_final, local_grid_size * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    // Gather results back to rank 0
    std::vector<Real> finalGrid;
    {
        std::vector<int> recvcounts(num_procs), displs(num_procs);
        size_t off = 0;
        for (int r = 0; r < num_procs; ++r) {
            size_t rnz = base_nz + ((size_t)r < remainder ? 1 : 0);
            recvcounts[r] = (int)(rnz * slice_size);
            displs[r] = (int)(off * slice_size);
            off += rnz;
        }

        if (rank == 0) {
            finalGrid.resize(nx * ny * nz);
        }

        // Send only the real data (without ghost layers)
        MPI_Gatherv(local_host.data() + ghost_bottom * slice_size,
                    (int)(local_nz_no_ghost * slice_size), MPI_DOUBLE,
                    rank == 0 ? finalGrid.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);

        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (global_duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);

        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(finalGrid, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(d_input));
                CUDA_CHECK(cudaFree(d_output));
                CUDA_CHECK(cudaFreeHost(h_send_bottom));
                CUDA_CHECK(cudaFreeHost(h_send_top));
                CUDA_CHECK(cudaFreeHost(h_recv_bottom));
                CUDA_CHECK(cudaFreeHost(h_recv_top));
                CUDA_CHECK(cudaStreamDestroy(stream_compute));
                CUDA_CHECK(cudaStreamDestroy(stream_halo));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_halo));

    MPI_Finalize();
    return 0;
}
