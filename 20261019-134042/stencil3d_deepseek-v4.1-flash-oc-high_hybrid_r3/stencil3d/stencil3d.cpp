#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

// Optional CUDA-aware MPI support.  When the MPI library provides it, the halo
// planes are exchanged directly between GPUs; otherwise pinned host staging is
// used.  Both paths are always available at runtime.
#if defined(__has_include)
#  if __has_include(<mpi-ext.h>)
#    include <mpi-ext.h>
#    if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#      define STENCIL_HAVE_CUDA_AWARE_MPI 1
#    endif
#  endif
#endif
#ifndef STENCIL_HAVE_CUDA_AWARE_MPI
#  define STENCIL_HAVE_CUDA_AWARE_MPI 0
#endif

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA stencil kernel
//
// The local grid contains local_nz + 2 z-planes.  Plane p corresponds to the
// global plane z0 + p - 1, so p == 0 and p == local_nz + 1 are halo planes.
// Only interior x/y points are updated; the global boundaries are never
// written, exactly as in the reference implementation.
// ---------------------------------------------------------------------------
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const int nx,
                              const int ny,
                              const int firstPlane) {
    const int x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) {
        return;
    }

    const int p = firstPlane + blockIdx.z;
    const size_t pstride = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(p) * pstride
                     + static_cast<size_t>(y) * static_cast<size_t>(nx)
                     + static_cast<size_t>(x);

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - static_cast<size_t>(nx)];
    const Real back   = input[idx + static_cast<size_t>(nx)];
    const Real bottom = input[idx - pstride];
    const Real top    = input[idx + pstride];

    // Keep the exact same operation order as the reference implementation.
    output[idx] = ((((((center + left) + right) + front) + back) + bottom) + top) / 7.0;
}

static inline void exchangeHalos(Real* grid,
                                 const size_t local_nz,
                                 const size_t plane,
                                 const int rank,
                                 const int active,
                                 Real* send_lo,
                                 Real* send_hi,
                                 Real* recv_lo,
                                 Real* recv_hi,
                                 cudaStream_t stream,
                                 MPI_Comm comm,
                                 const bool cuda_aware_mpi) {
    const bool has_prev = (rank > 0);
    const bool has_next = (rank + 1 < active);
    if (!has_prev && !has_next) {
        return;
    }

    const int count = static_cast<int>(plane);
    const size_t bytes = plane * sizeof(Real);

    MPI_Request requests[4];
    int nreq = 0;

    if (cuda_aware_mpi) {
        // Make sure the previous kernel has produced the planes that are sent.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (has_prev) {
            MPI_Isend(grid + plane, count, MPI_DOUBLE, rank - 1, 1, comm,
                      &requests[nreq++]);
            MPI_Irecv(grid, count, MPI_DOUBLE, rank - 1, 0, comm,
                      &requests[nreq++]);
        }
        if (has_next) {
            MPI_Isend(grid + local_nz * plane, count, MPI_DOUBLE, rank + 1, 0,
                      comm, &requests[nreq++]);
            MPI_Irecv(grid + (local_nz + 1) * plane, count, MPI_DOUBLE,
                      rank + 1, 1, comm, &requests[nreq++]);
        }

        MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    // Pack the two boundary planes that are sent to the neighbouring ranks.
    if (has_prev) {
        CUDA_CHECK(cudaMemcpyAsync(send_lo, grid + plane, bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    if (has_next) {
        CUDA_CHECK(cudaMemcpyAsync(send_hi, grid + local_nz * plane, bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Exchange with the previous rank: our lower plane <-> its upper plane.
    if (has_prev) {
        MPI_Isend(send_lo, count, MPI_DOUBLE, rank - 1, 1,
                  comm, &requests[nreq++]);
        MPI_Irecv(recv_lo, count, MPI_DOUBLE, rank - 1, 0,
                  comm, &requests[nreq++]);
    }
    // Exchange with the next rank: our upper plane <-> its lower plane.
    if (has_next) {
        MPI_Isend(send_hi, count, MPI_DOUBLE, rank + 1, 0,
                  comm, &requests[nreq++]);
        MPI_Irecv(recv_hi, count, MPI_DOUBLE, rank + 1, 1,
                  comm, &requests[nreq++]);
    }

    MPI_Waitall(nreq, requests, MPI_STATUSES_IGNORE);

    if (has_prev) {
        CUDA_CHECK(cudaMemcpyAsync(grid, recv_lo, bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
    if (has_next) {
        CUDA_CHECK(cudaMemcpyAsync(grid + (local_nz + 1) * plane, recv_hi, bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

static inline bool queryCudaAwareMpi() {
#if STENCIL_HAVE_CUDA_AWARE_MPI
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
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

// Local initialisation used by the distributed code.  Only the owned planes are
// filled; halo planes stay zero until the first exchange.
static void initializeLocalGrid(std::vector<Real>& grid,
                                const size_t local_nz,
                                const size_t z0,
                                const size_t nx,
                                const size_t ny) {
    const size_t plane = nx * ny;
    const long long lnz = static_cast<long long>(local_nz);
    const long long lny = static_cast<long long>(ny);
    const long long lnx = static_cast<long long>(nx);

    #pragma omp parallel for collapse(3) schedule(static)
    for (long long p = 1; p <= lnz; ++p) {
        for (long long y = 0; y < lny; ++y) {
            for (long long x = 0; x < lnx; ++x) {
                const size_t gz = z0 + static_cast<size_t>(p) - 1;
                const size_t gidx = gz * plane
                                  + static_cast<size_t>(y) * nx
                                  + static_cast<size_t>(x);
                const size_t lidx = static_cast<size_t>(p) * plane
                                  + static_cast<size_t>(y) * nx
                                  + static_cast<size_t>(x);
                grid[lidx] = (gidx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation (reference serial version, retained for clarity).
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

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real val = grid[static_cast<size_t>(i)];
        if (std::isnan(val) || std::isinf(val)) {
            bad = 1;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (long long i = 0; i < static_cast<long long>(grid.size()); ++i) {
        const Real val = grid[static_cast<size_t>(i)];
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

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nz == 0) nz = 1;

    if (world_rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Map each MPI rank to a GPU on its node.  If several ranks share a node,
    // they are distributed round-robin over the visible devices.
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &node_comm);
    int local_rank = 0;
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(node_comm, &local_rank);
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < 1) {
        if (world_rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return 1;
    }
    const int device = local_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device));
    if (node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&node_comm);
    }
    const bool cuda_aware_mpi = queryCudaAwareMpi();

    // Distribute the global z dimension over the MPI ranks.  Ranks without any
    // plane (possible only when size > nz) stay idle but participate in the
    // final gather so that the global result remains well defined.
    const int active = static_cast<int>(std::min<size_t>(static_cast<size_t>(world_size), nz));
    const bool am_active = (world_rank < active);
    const size_t base = nz / static_cast<size_t>(active);
    const size_t rem = nz % static_cast<size_t>(active);
    const size_t local_nz = am_active
                          ? base + (static_cast<size_t>(world_rank) < rem ? 1 : 0)
                          : 0;
    const size_t z0 = am_active
                    ? static_cast<size_t>(world_rank) * base
                      + std::min<size_t>(static_cast<size_t>(world_rank), rem)
                    : 0;

    const size_t plane = nx * ny;
    const size_t local_planes = local_nz + 2;
    const size_t local_elements = local_planes * plane;

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    cudaStream_t stream = nullptr;
    Real* h_send_lo = nullptr;
    Real* h_send_hi = nullptr;
    Real* h_recv_lo = nullptr;
    Real* h_recv_hi = nullptr;

    if (am_active) {
        std::vector<Real> host_init(local_elements, Real(0));
        initializeLocalGrid(host_init, local_nz, z0, nx, ny);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_grid1), local_elements * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_grid2), local_elements * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_grid1, host_init.data(), local_elements * sizeof(Real),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid2, host_init.data(), local_elements * sizeof(Real),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreate(&stream));

        if (active > 1 && !cuda_aware_mpi) {
            const size_t bytes = plane * sizeof(Real);
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_send_lo), bytes));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_send_hi), bytes));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_recv_lo), bytes));
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_recv_hi), bytes));
        }
    }

    // Determine which local planes contain interior global z planes.
    int first_plane = 0;
    int num_planes = 0;
    if (am_active) {
        first_plane = (world_rank == 0) ? 2 : 1;
        const int last_exclusive = (world_rank == active - 1)
                                 ? static_cast<int>(local_nz)
                                 : static_cast<int>(local_nz + 1);
        num_planes = last_exclusive - first_plane;
        if (num_planes < 0) num_planes = 0;
    }

    Real* d_input = d_grid1;
    Real* d_output = d_grid2;

    if (world_rank == 0) {
        printf("Initializing grid...\n");
        printf("Running stencil computation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (am_active) {
            exchangeHalos(d_input, local_nz, plane, world_rank, active,
                          h_send_lo, h_send_hi, h_recv_lo, h_recv_hi,
                          stream, MPI_COMM_WORLD, cuda_aware_mpi);

            if (num_planes > 0 && nx >= 3 && ny >= 3) {
                const dim3 block(32, 8, 1);
                const unsigned gx = (static_cast<unsigned>(nx - 2) + block.x - 1) / block.x;
                const unsigned gy = (static_cast<unsigned>(ny - 2) + block.y - 1) / block.y;
                const int max_grid_z = 65535;
                for (int p0 = first_plane; p0 < first_plane + num_planes; p0 += max_grid_z) {
                    const int pn = std::min(max_grid_z, first_plane + num_planes - p0);
                    const dim3 grid(gx, gy, static_cast<unsigned>(pn));
                    stencilKernel<<<grid, block, 0, stream>>>(d_input, d_output,
                                                              static_cast<int>(nx),
                                                              static_cast<int>(ny),
                                                              p0);
                }
            }
        }

        std::swap(d_input, d_output);
    }

    if (am_active) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    const double elapsed_seconds = max_duration_ms / 1000.0;

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
        double mcups = cellUpdates / (elapsed_seconds > 0.0 ? elapsed_seconds : 1e-12) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // The final grid is d_input after the swaps (grid1 for an even number of
    // iterations, grid2 for an odd number), matching the reference program.
    const bool need_result = printResults || validate;
    std::vector<Real> local_result;
    if (am_active) {
        local_result.resize(local_nz * plane);
        if (local_nz > 0) {
            CUDA_CHECK(cudaMemcpy(local_result.data(), d_input + plane,
                                  local_nz * plane * sizeof(Real),
                                  cudaMemcpyDeviceToHost));
        }
    }

    int valid = 1;
    if (need_result) {
        std::vector<int> recvcounts(static_cast<size_t>(world_size), 0);
        std::vector<int> displs(static_cast<size_t>(world_size), 0);

        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            size_t r_nz = 0;
            if (r < active) {
                r_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            }
            const size_t count = r_nz * plane;
            recvcounts[static_cast<size_t>(r)] = static_cast<int>(count);
            displs[static_cast<size_t>(r)] = offset;
            offset += static_cast<int>(count);
        }

        std::vector<Real> full_grid;
        if (world_rank == 0) {
            full_grid.resize(nx * ny * nz);
        }

        const int send_count = static_cast<int>(local_nz * plane);
        MPI_Gatherv(am_active ? local_result.data() : nullptr, send_count, MPI_DOUBLE,
                    (world_rank == 0) ? full_grid.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            if (printResults) {
                print_results(full_grid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool ok = validateResult(full_grid, nx, ny, nz);
                valid = ok ? 1 : 0;
                if (ok) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }

    int global_valid = 1;
    MPI_Allreduce(&valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (am_active) {
        if (h_send_lo) CUDA_CHECK(cudaFreeHost(h_send_lo));
        if (h_send_hi) CUDA_CHECK(cudaFreeHost(h_send_hi));
        if (h_recv_lo) CUDA_CHECK(cudaFreeHost(h_recv_lo));
        if (h_recv_hi) CUDA_CHECK(cudaFreeHost(h_recv_hi));
        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
        if (d_grid1) CUDA_CHECK(cudaFree(d_grid1));
        if (d_grid2) CUDA_CHECK(cudaFree(d_grid2));
    }

    MPI_Finalize();
    return (validate && !global_valid) ? 1 : 0;
}
