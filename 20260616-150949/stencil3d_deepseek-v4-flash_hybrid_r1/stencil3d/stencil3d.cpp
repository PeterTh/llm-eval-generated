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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = call;                                                  \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// 3D index calculation (global grid)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                              const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernel: hybrid 7-point stencil + global boundary copy
// ---------------------------------------------------------------------------
// Each thread handles one (x,y) column; loops over local z = 1..nz_local.
// Points on global domain boundaries (x=0, x=nx-1, y=0, y=ny-1,
// global_z=0, global_z=nz-1) are simply copied from input to output.
// All other points are computed via the 7-point averaging stencil.
// ---------------------------------------------------------------------------
__global__ void stencil_kernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               size_t nx, size_t ny, size_t nz_local,
                               size_t plane_size, size_t z_start,
                               size_t global_nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) return;

    for (int zl = 1; zl <= static_cast<int>(nz_local); ++zl) {
        size_t global_z = z_start + static_cast<size_t>(zl) - 1;
        size_t idx = static_cast<size_t>(zl) * plane_size +
                     static_cast<size_t>(y) * nx + static_cast<size_t>(x);

        if (x == 0 || x == static_cast<int>(nx) - 1 ||
            y == 0 || y == static_cast<int>(ny) - 1 ||
            global_z == 0 || global_z == global_nz - 1) {
            // Global boundary — copy from input
            output[idx] = input[idx];
        } else {
            // Interior — 7-point stencil
            Real val = input[idx]
                     + input[idx - 1]
                     + input[idx + 1]
                     + input[idx - nx]
                     + input[idx + static_cast<size_t>(nx)]
                     + input[idx - plane_size]
                     + input[idx + plane_size];
            output[idx] = val / static_cast<Real>(7.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Domain decomposition helpers
// ---------------------------------------------------------------------------
static void compute_domain(size_t nz, int rank, int num_procs,
                            size_t& z_start, size_t& nz_local) {
    size_t base = nz / static_cast<size_t>(num_procs);
    size_t rem = nz % static_cast<size_t>(num_procs);
    nz_local = base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    z_start = 0;
    for (int r = 0; r < rank; ++r) {
        z_start += base + (static_cast<size_t>(r) < rem ? 1 : 0);
    }
}

static void initialize_grid_local(std::vector<Real>& grid,
                                   size_t nx, size_t ny,
                                   size_t nz_local, size_t z_start,
                                   size_t global_nz) {
    size_t plane_size = nx * ny;
    // Interior layers (z_local = 1 .. nz_local)
#pragma omp parallel for collapse(2)
    for (int zl = 1; zl <= static_cast<int>(nz_local); ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            size_t global_z = z_start + static_cast<size_t>(zl) - 1;
            for (size_t x = 0; x < nx; ++x) {
                size_t lidx = static_cast<size_t>(zl) * plane_size + y * nx + x;
                size_t gidx = idx3(x, y, global_z, nx, ny);
                grid[lidx] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }
    // Bottom ghost layer (z_local = 0) — only if there is a lower neighbour
    if (z_start > 0) {
        size_t ghost_global_z = z_start - 1;
#pragma omp parallel for
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t lidx = y * nx + x;
                size_t gidx = idx3(x, y, ghost_global_z, nx, ny);
                grid[lidx] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }
    // Top ghost layer (z_local = nz_local + 1) — only if there is an upper neighbour
    if (z_start + nz_local < global_nz) {
        size_t ghost_global_z = z_start + nz_local;
#pragma omp parallel for
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t lidx = (nz_local + 1) * plane_size + y * nx + x;
                size_t gidx = idx3(x, y, ghost_global_z, nx, ny);
                grid[lidx] = static_cast<Real>((gidx % 19) * 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (per-rank, then reduce across ranks)
// ---------------------------------------------------------------------------
static bool validate_result_local(const std::vector<Real>& grid,
                                   size_t total_elements) {
    bool ok = true;
    Real local_min = grid[0];
    Real local_max = grid[0];

#pragma omp parallel for reduction(&& : ok) reduction(min : local_min) reduction(max : local_max)
    for (size_t i = 0; i < total_elements; ++i) {
        Real v = grid[i];
        if (std::isnan(v) || std::isinf(v)) ok = false;
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    // Reduce across MPI ranks
    bool global_ok;
    MPI_Allreduce(&ok, &global_ok, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

    Real global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Value range: [%.6f, %.6f]\n", global_min, global_max);
        if (!global_ok) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        if (global_max > 1e6 || global_min < -1e6) {
            printf("Validation failed: values out of expected range\n");
            global_ok = false;
        }
        printf("Validation: %s\n", global_ok ? "PASSED" : "FAILED");
    }
    return global_ok;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* progName) {
    // Only rank 0 prints — but we let all callers decide
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

// ===========================================================================
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // MPI initialisation
    // -----------------------------------------------------------------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // -----------------------------------------------------------------------
    // CUDA device selection – round-robin across available GPUs
    // -----------------------------------------------------------------------
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    if (n_gpus == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev_id = rank % n_gpus;
    CUDA_CHECK(cudaSetDevice(dev_id));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev_id);
    if (rank == 0) {
        printf("Using CUDA device %d: %s (rank mapping)\n", dev_id, prop.name);
    }

    // -----------------------------------------------------------------------
    // Parse command-line arguments (all ranks, only rank 0 prints banner)
    // -----------------------------------------------------------------------
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
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

    // Domain decomposition along Z
    size_t z_start = 0, nz_local = 0;
    compute_domain(nz, rank, num_procs, z_start, nz_local);

    if (nz_local == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: grid Z dimension too small for %d ranks\n",
                    num_procs);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Ranks: %d, OpenMP threads: %d, GPU devices available: %d\n",
               num_procs, omp_get_max_threads(), n_gpus);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t plane_size = nx * ny;
    size_t local_elements = plane_size * (nz_local + 2);  // includes ghost layers

    // -----------------------------------------------------------------------
    // Host memory (page-locked for fast GPU transfers + MPI exchange)
    // -----------------------------------------------------------------------
    std::vector<Real> h_grid(local_elements);
    initialize_grid_local(h_grid, nx, ny, nz_local, z_start, nz);

    // Pinned buffers for MPI halo exchange
    Real *h_send = nullptr, *h_recv = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_send, plane_size * sizeof(Real),
                              cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv, plane_size * sizeof(Real),
                              cudaHostAllocDefault));

    // -----------------------------------------------------------------------
    // Device memory
    // -----------------------------------------------------------------------
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    size_t dev_bytes = local_elements * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_grid1, dev_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, dev_bytes));

    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid.data(), dev_bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, h_grid.data(), dev_bytes,
                           cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Main computation loop
    // -----------------------------------------------------------------------
    Real* d_in  = d_grid1;
    Real* d_out = d_grid2;

    dim3 block(16, 16);
    dim3 grid((static_cast<unsigned int>(nx) + 15) / 16,
              (static_cast<unsigned int>(ny) + 15) / 16);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        // ------- CUDA stencil kernel ---------------------------------------
        stencil_kernel<<<grid, block>>>(d_in, d_out, nx, ny, nz_local,
                                         plane_size, z_start, nz);
        CUDA_CHECK(cudaGetLastError());

        // ------- MPI halo exchange (ghost layers in Z) ---------------------
        //
        // After the kernel, d_out contains the correct values for the
        // interior (z_local = 1 .. nz_local).  Ghost cells (z_local = 0 and
        // z_local = nz_local+1) are stale and must be refreshed via MPI.
        //
        //  bottom neighbour (rank-1) :  I send my z_local=1 plane down and
        //                                receive into my z_local=0 ghost.
        //  top    neighbour (rank+1) :  I send my z_local=nz_local plane up
        //                                and receive into my z_local=nz_local+1
        //                                ghost.

        // -- bottom exchange (higher rank sends down, receives from below) --
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpy(h_send,
                                   d_out + plane_size * 1,  // z_local = 1
                                   plane_size * sizeof(Real),
                                   cudaMemcpyDeviceToHost));
            // I am the higher rank (r); lower neighbour is (r-1).
            // Convention: lower→higher uses tag 100, higher→lower uses tag 101.
            // So I send to (r-1) with tag 101 and receive from (r-1) with tag 100.
            MPI_Sendrecv(h_send, static_cast<int>(plane_size), MPI_DOUBLE,
                         rank - 1, 101,
                         h_recv, static_cast<int>(plane_size), MPI_DOUBLE,
                         rank - 1, 100,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(d_out,                // z_local = 0
                                   h_recv,
                                   plane_size * sizeof(Real),
                                   cudaMemcpyHostToDevice));
        }

        // -- top exchange (lower rank sends up, receives from above) --
        if (rank < num_procs - 1) {
            CUDA_CHECK(cudaMemcpy(h_send,
                                   d_out + plane_size * nz_local,  // z_local=nz_local
                                   plane_size * sizeof(Real),
                                   cudaMemcpyDeviceToHost));
            // I am the lower rank (r); higher neighbour is (r+1).
            // Convention: lower→higher uses tag 100, higher→lower uses tag 101.
            // So I send to (r+1) with tag 100 and receive from (r+1) with tag 101.
            MPI_Sendrecv(h_send, static_cast<int>(plane_size), MPI_DOUBLE,
                         rank + 1, 100,
                         h_recv, static_cast<int>(plane_size), MPI_DOUBLE,
                         rank + 1, 101,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpy(d_out + plane_size * (nz_local + 1),
                                   h_recv,
                                   plane_size * sizeof(Real),
                                   cudaMemcpyHostToDevice));
        }

        // Swap buffers for next iteration
        std::swap(d_in, d_out);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // Timing & performance (rank 0 only)
    // -----------------------------------------------------------------------
    double elapsed = std::chrono::duration<double>(end - start).count();
    double elapsed_ms = elapsed * 1.0e3;

    if (rank == 0) {
        double cellUpdates =
            static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) *
            static_cast<double>(iterations);
        double mcups = cellUpdates / elapsed / 1.0e6;

        printf("Computation time: %.3f ms\n", elapsed_ms);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // -----------------------------------------------------------------------
    // Retrieve final grid (d_in after last swap)
    // -----------------------------------------------------------------------
    // d_in points to the buffer that contains the most recent output.
    // After the loop: if iterations is even, d_in == d_grid1; else d_in == d_grid2.
    // Copy non-ghost portion (z_local = 1 .. nz_local) back to h_grid.
    CUDA_CHECK(cudaMemcpy(h_grid.data(), d_in, dev_bytes,
                           cudaMemcpyDeviceToHost));

    // -----------------------------------------------------------------------
    // Gather full grid to rank 0 for validation / result printing
    // -----------------------------------------------------------------------
    bool need_full_grid = printResults || validate;
    std::vector<Real> full_grid;
    std::vector<int> recv_counts(num_procs);
    std::vector<int> recv_displs(num_procs);

    if (need_full_grid) {
        // Each rank contributes exactly nz_local * plane_size non-ghost elements
        for (int r = 0; r < num_procs; ++r) {
            size_t base = nz / static_cast<size_t>(num_procs);
            size_t rem = nz % static_cast<size_t>(num_procs);
            size_t nz_r =
                base + (static_cast<size_t>(r) < rem ? 1 : 0);
            recv_counts[r] =
                static_cast<int>(nz_r * plane_size);
            recv_displs[r] =
                (r == 0) ? 0
                         : (recv_displs[r - 1] + recv_counts[r - 1]);
        }

        if (rank == 0) {
            full_grid.resize(nx * ny * nz);
        }

        size_t my_count = nz_local * plane_size;
        MPI_Gatherv(h_grid.data() + plane_size,  // skip bottom ghost
                    static_cast<int>(my_count), MPI_DOUBLE,
                    full_grid.data(),
                    recv_counts.data(), recv_displs.data(),
                    MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // -----------------------------------------------------------------------
    // Print results (rank 0)
    // -----------------------------------------------------------------------
    if (printResults && rank == 0) {
        print_results(full_grid, "Grid");
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------
    bool valid = true;
    if (validate) {
        // Validate the non-ghost portion (z_local = 1 .. nz_local)
        size_t my_non_ghost = nz_local * plane_size;
        size_t ghost_offset = plane_size;  // skip bottom ghost
        valid = validate_result_local(
            std::vector<Real>(h_grid.begin() + static_cast<ptrdiff_t>(ghost_offset),
                              h_grid.begin() + static_cast<ptrdiff_t>(ghost_offset + my_non_ghost)),
            my_non_ghost);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaFreeHost(h_send));
    CUDA_CHECK(cudaFreeHost(h_recv));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    MPI_Finalize();

    if (validate) {
        return valid ? 0 : 1;
    }
    return 0;
}
