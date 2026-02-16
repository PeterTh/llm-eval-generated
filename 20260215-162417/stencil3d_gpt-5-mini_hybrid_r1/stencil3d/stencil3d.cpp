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
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

// Device kernel: operates on local grid with halo layers in z (z=0 and z=local_nz+1 are halos)
__global__ void stencil_kernel(const Real* in, Real* out, size_t nx, size_t ny, size_t local_nz, size_t start_z, size_t global_nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z_local = blockIdx.z * blockDim.z + threadIdx.z; // ranges 0..local_nz-1 map to local z 1..local_nz

    if (x <= 0 || x >= (int)nx-1) return;
    if (y <= 0 || y >= (int)ny-1) return;
    if (z_local < 0 || z_local >= (int)local_nz) return;

    // map to index with halo: z = z_local + 1
    size_t z = (size_t)z_local + 1;
    size_t nxny = nx * ny;
    size_t idx = z * nxny + y * nx + x;

    size_t global_z = start_z + (z - 1);
    if (global_z == 0 || global_z == global_nz - 1) return; // global boundary, copied elsewhere

    Real center = in[idx];
    Real left = in[idx - 1];
    Real right = in[idx + 1];
    Real front = in[idx - nx];
    Real back = in[idx + nx];
    Real bottom = in[idx - nxny];
    Real top = in[idx + nxny];

    out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeLocalGrid(std::vector<Real>& local, const size_t nx, const size_t ny, const size_t local_nz, const size_t start_z, const size_t global_nz) {
    size_t nxny = nx * ny;
    // local size includes 2 halo layers in z: stored as size (local_nz + 2)
    for (size_t z = 0; z < local_nz + 2; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = z * nxny + y * nx + x;
                // compute global z index for interior z
                if (z == 0) {
                    size_t gz = (start_z == 0) ? 0 : (start_z - 1);
                    local[idx] = ((idx3(x,y,gz,nx,ny)) % 19) * 1.0;
                } else if (z == local_nz + 1) {
                    size_t gz = (start_z + local_nz >= global_nz) ? global_nz - 1 : (start_z + local_nz);
                    local[idx] = ((idx3(x,y,gz,nx,ny)) % 19) * 1.0;
                } else {
                    size_t gz = start_z + (z - 1);
                    local[idx] = ((idx3(x,y,gz,nx,ny)) % 19) * 1.0;
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            fprintf(stderr, "Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    // 2. Range checks
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e9 || minVal < -1e9) {
        fprintf(stderr, "Validation failed: values out of expected range\n");
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

    MPI_Init(&argc, &argv);

    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Partition along Z among ranks
    size_t base = nz / world_size;
    size_t rem = nz % world_size;
    size_t local_nz = base + (world_rank < (int)rem ? 1 : 0);
    size_t start_z = (size_t)world_rank * base + (world_rank < (int)rem ? world_rank : rem);

    size_t nxny = nx * ny;
    size_t local_with_halo = (local_nz + 2);
    size_t local_size = nxny * local_with_halo;

    // allocate host local buffers (with halo layers)
    std::vector<Real> local_in(local_size);
    std::vector<Real> local_out(local_size);

    // initialize local grid (including halos)
    initializeLocalGrid(local_in, nx, ny, local_nz, start_z, nz);
    initializeLocalGrid(local_out, nx, ny, local_nz, start_z, nz);

    // setup device buffers
    Real* d_in = nullptr;
    Real* d_out = nullptr;
    CUDA_CHECK(cudaSetDevice(0)); // assume one GPU per MPI rank; in multi-GPU you would map ranks to GPUs
    CUDA_CHECK(cudaMalloc((void**)&d_in, sizeof(Real) * local_size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(Real) * local_size));

    // Prepare halo MPI datatypes: a single XY plane
    MPI_Datatype xy_plane;
    MPI_Type_contiguous((int)nxny, MPI_DOUBLE, &xy_plane);
    MPI_Type_commit(&xy_plane);

    int prev = (world_rank == 0) ? MPI_PROC_NULL : world_rank - 1;
    int next = (world_rank == world_size - 1) ? MPI_PROC_NULL : world_rank + 1;

    // Synchronize and start timed region
    MPI_Barrier(MPI_COMM_WORLD);
    auto tstart = std::chrono::high_resolution_clock::now();

    // iteration loop
    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (local_nz + block.z - 1) / block.z);

    for (int iter = 0; iter < iterations; ++iter) {
        // copy host to device
        CUDA_CHECK(cudaMemcpy(d_in, local_in.data(), sizeof(Real) * local_size, cudaMemcpyHostToDevice));

        // launch kernel to compute interior (skips x/y boundaries and global z boundaries)
        stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, local_nz, start_z, nz);
        CUDA_CHECK(cudaGetLastError());

        // copy back
        CUDA_CHECK(cudaMemcpy(local_out.data(), d_out, sizeof(Real) * local_size, cudaMemcpyDeviceToHost));

        // Copy x/y boundary values from input to output (host) to preserve boundaries
        #pragma omp parallel for collapse(2)
        for (size_t z = 0; z < local_with_halo; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                // left and right columns
                local_out[z * nxny + y * nx + 0] = local_in[z * nxny + y * nx + 0];
                local_out[z * nxny + y * nx + (nx - 1)] = local_in[z * nxny + y * nx + (nx - 1)];
            }
        }
        #pragma omp parallel for collapse(2)
        for (size_t z = 0; z < local_with_halo; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                // front and back rows
                local_out[z * nxny + 0 * nx + x] = local_in[z * nxny + 0 * nx + x];
                local_out[z * nxny + (ny - 1) * nx + x] = local_in[z * nxny + (ny - 1) * nx + x];
            }
        }

        // Exchange halo planes (send/recv using XY plane contiguous type)
        // send our z=1 plane to prev (becomes their top halo), receive into z=0
        MPI_Sendrecv(
            &local_out[1 * nxny], 1, xy_plane, prev, 0,
            &local_out[0 * nxny], 1, xy_plane, prev, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send our last interior plane z=local_nz to next, receive into z=local_nz+1
        MPI_Sendrecv(
            &local_out[local_nz * nxny], 1, xy_plane, next, 0,
            &local_out[(local_nz + 1) * nxny], 1, xy_plane, next, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // swap buffers for next iteration
        std::swap(local_in, local_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto tend = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);

    // compute global performance metrics
    double local_cells = (double)((nx - 2) * (ny - 2) * (int)local_nz) * iterations;
    double global_cells = 0.0;
    MPI_Reduce(&local_cells, &global_cells, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double mcups = global_cells / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 for validation/printing
    std::vector<Real> finalGrid;
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    int local_count = (int)(nxny * local_nz);
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        int total = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
        finalGrid.resize(total);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) displs[i] = displs[i-1] + recvcounts[i-1];
    }

    // prepare send buffer without halos: copy interior z=1..local_nz
    std::vector<Real> sendbuf(nxny * local_nz);
    for (size_t z = 0; z < local_nz; ++z) {
        std::memcpy(&sendbuf[z * nxny], &local_in[(z+1) * nxny], sizeof(Real) * nxny);
    }

    MPI_Gatherv(sendbuf.data(), local_count, MPI_DOUBLE, finalGrid.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
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
            }
        }
    }

    // cleanup
    MPI_Type_free(&xy_plane);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    MPI_Finalize();
    return 0;
}
