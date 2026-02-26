#include <algorithm>
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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t _err = (call);                                                  \
        if (_err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            MPI_Abort(MPI_COMM_WORLD, 2);                                                 \
        }                                                                                \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static __global__ void copy_boundaries_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                              int nx, int ny, int local_nz, int z0_global, int nz_global) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int z_local = (int)(blockIdx.z * blockDim.z + threadIdx.z) + 1;  // [1, local_nz]

    if (x >= nx || y >= ny || z_local > local_nz) return;

    const int gz = z0_global + (z_local - 1);
    const bool z_boundary = (gz == 0) || (gz == nz_global - 1);
    const bool xy_boundary = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1);

    if (z_boundary || xy_boundary) {
        const size_t idx = ((size_t)z_local * (size_t)nx * (size_t)ny) + (size_t)y * (size_t)nx + (size_t)x;
        out[idx] = in[idx];
    }
}

static __global__ void stencil_interior_kernel(const Real* __restrict__ in, Real* __restrict__ out,
                                               int nx, int ny, int z_first_local, int z_count) {
    // Computes only interior (x,y not on boundary), over a contiguous local-z range.
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1;  // [1, nx-2]
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1;  // [1, ny-2]
    const int z_local = (int)(blockIdx.z * blockDim.z + threadIdx.z) + z_first_local;

    if (x >= nx - 1 || y >= ny - 1 || z_local >= z_first_local + z_count) return;

    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t idx = (size_t)z_local * plane + (size_t)y * (size_t)nx + (size_t)x;

    const Real center = in[idx];
    const Real left = in[idx - 1];
    const Real right = in[idx + 1];
    const Real front = in[idx - (size_t)nx];
    const Real back = in[idx + (size_t)nx];
    const Real bottom = in[idx - plane];
    const Real top = in[idx + plane];

    out[idx] = (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned long long nx = 128;
    unsigned long long ny = 0;
    unsigned long long nz = 0;
    int iterations = 10;
    int validate = 0;
    int printResults = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx = (unsigned long long)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny = (unsigned long long)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz = (unsigned long long)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("3D Stencil Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Grid size: %llu x %llu x %llu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Select GPU based on intra-node rank.
    int local_rank = 0;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % deviceCount));

    const size_t nx_s = (size_t)nx;
    const size_t ny_s = (size_t)ny;
    const size_t nz_s = (size_t)nz;
    const size_t plane_elems = nx_s * ny_s;

    // Z-slab decomposition across ranks.
    const size_t base = nz_s / (size_t)size;
    const size_t rem = nz_s % (size_t)size;
    const size_t local_nz = base + ((size_t)rank < rem ? 1u : 0u);
    const size_t start_z = (size_t)rank * base + ((size_t)rank < rem ? (size_t)rank : rem);

    if (local_nz == 0) {
        if (rank == 0) fprintf(stderr, "Too many MPI ranks for nz=%zu\n", nz_s);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t local_with_halo = local_nz + 2;
    const size_t local_elems_with_halo = local_with_halo * plane_elems;
    const size_t plane_bytes = plane_elems * sizeof(Real);

    std::vector<Real> h_init(local_elems_with_halo, 0.0);

    // Initialize only owned planes [start_z, start_z+local_nz-1] into local z=[1,local_nz].
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z_local = 1; z_local <= local_nz; ++z_local) {
        for (size_t y = 0; y < ny_s; ++y) {
            for (size_t x = 0; x < nx_s; ++x) {
                const size_t gz = start_z + (z_local - 1);
                const size_t gidx = idx3(x, y, gz, nx_s, ny_s);
                const size_t lidx = z_local * plane_elems + y * nx_s + x;
                h_init[lidx] = (Real)((gidx % 19u)) * 1.0;
            }
        }
    }

    Real* d_in = nullptr;
    Real* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, local_elems_with_halo * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_out, local_elems_with_halo * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_in, h_init.data(), local_elems_with_halo * sizeof(Real), cudaMemcpyHostToDevice));

    // Host pinned buffers for halo exchange.
    Real* h_send_lo = nullptr;
    Real* h_send_hi = nullptr;
    Real* h_recv_lo = nullptr;
    Real* h_recv_hi = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_send_lo, plane_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_send_hi, plane_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_lo, plane_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_hi, plane_bytes, cudaHostAllocDefault));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    auto exchange_halos = [&](Real* d_buf) {
        if (size == 1) return;

        // Device -> host for boundary planes.
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(h_send_lo, d_buf + plane_elems * 1, plane_bytes, cudaMemcpyDeviceToHost, stream));
        }
        if (rank < size - 1) {
            CUDA_CHECK(cudaMemcpyAsync(h_send_hi, d_buf + plane_elems * (local_nz), plane_bytes, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Status st;
        constexpr int TAG_DOWN = 100;
        constexpr int TAG_UP = 101;

        // Exchange with previous rank (receive lower halo into z=0).
        if (rank > 0) {
            MPI_Sendrecv(h_send_lo, (int)plane_elems, MPI_DOUBLE, rank - 1, TAG_DOWN,
                         h_recv_lo, (int)plane_elems, MPI_DOUBLE, rank - 1, TAG_UP,
                         MPI_COMM_WORLD, &st);
        }

        // Exchange with next rank (receive upper halo into z=local_nz+1).
        if (rank < size - 1) {
            MPI_Sendrecv(h_send_hi, (int)plane_elems, MPI_DOUBLE, rank + 1, TAG_UP,
                         h_recv_hi, (int)plane_elems, MPI_DOUBLE, rank + 1, TAG_DOWN,
                         MPI_COMM_WORLD, &st);
        }

        // Host -> device halos.
        if (rank > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf + plane_elems * 0, h_recv_lo, plane_bytes, cudaMemcpyHostToDevice, stream));
        }
        if (rank < size - 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf + plane_elems * (local_nz + 1), h_recv_hi, plane_bytes, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // Ensure initial halos are ready for ranks that need them.
    exchange_halos(d_in);

    // Determine interior-z range for this rank (global z in [1, nz-2]).
    const size_t global_interior_z0 = (start_z < 1 ? 1 : start_z);
    const size_t global_interior_z1 = (start_z + local_nz - 1 > nz_s - 2 ? nz_s - 2 : start_z + local_nz - 1);
    const int z_first_local = (global_interior_z0 <= global_interior_z1) ? (int)((global_interior_z0 - start_z) + 1) : 1;
    const int z_count = (global_interior_z0 <= global_interior_z1) ? (int)(global_interior_z1 - global_interior_z0 + 1) : 0;

    const dim3 blockB(32, 4, 2);
    const dim3 gridB((unsigned)((nx_s + blockB.x - 1) / blockB.x),
                     (unsigned)((ny_s + blockB.y - 1) / blockB.y),
                     (unsigned)((local_nz + blockB.z - 1) / blockB.z));

    const dim3 blockI(32, 4, 2);
    const dim3 gridI((unsigned)(((nx_s - 2) + blockI.x - 1) / blockI.x),
                     (unsigned)(((ny_s - 2) + blockI.y - 1) / blockI.y),
                     (unsigned)(((size_t)z_count + blockI.z - 1) / blockI.z));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int iter = 0; iter < iterations; ++iter) {
        // Copy boundary values for this iteration.
        copy_boundaries_kernel<<<gridB, blockB, 0, stream>>>(d_in, d_out, (int)nx_s, (int)ny_s, (int)local_nz, (int)start_z, (int)nz_s);
        CUDA_CHECK(cudaGetLastError());

        // Compute interior (GPU).
        if (z_count > 0 && nx_s >= 3 && ny_s >= 3) {
            stencil_interior_kernel<<<gridI, blockI, 0, stream>>>(d_in, d_out, (int)nx_s, (int)ny_s, z_first_local, z_count);
            CUDA_CHECK(cudaGetLastError());
        }

        // Prepare halos for next iteration (on d_out).
        exchange_halos(d_out);

        std::swap(d_in, d_out);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    double local_sec = t1 - t0;

    double max_sec = 0.0;
    MPI_Reduce(&local_sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = (long)llround(max_sec * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double cellUpdates = (double)((nx_s - 2) * (ny_s - 2) * (nz_s - 2)) * (double)iterations;
        const double mcups = cellUpdates / max_sec / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final grid to rank 0 for printing/validation (owned planes only).
    std::vector<Real> local_final(local_nz * plane_elems);
    CUDA_CHECK(cudaMemcpyAsync(local_final.data(), d_in + plane_elems * 1, local_final.size() * sizeof(Real), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> recvcounts;
    std::vector<int> displs;
    std::vector<Real> finalGrid;

    if (rank == 0) {
        recvcounts.resize((size_t)size);
        displs.resize((size_t)size);
        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            const size_t r_local_nz = base + ((size_t)r < rem ? 1u : 0u);
            recvcounts[(size_t)r] = (int)(r_local_nz * plane_elems);
            displs[(size_t)r] = (int)disp;
            disp += r_local_nz * plane_elems;
        }
        finalGrid.resize(nz_s * plane_elems);
    }

    MPI_Gatherv(local_final.data(), (int)local_final.size(), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool ok = validateResult(finalGrid, nx_s, ny_s, nz_s);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaStreamDestroy(stream));
            CUDA_CHECK(cudaFreeHost(h_send_lo));
            CUDA_CHECK(cudaFreeHost(h_send_hi));
            CUDA_CHECK(cudaFreeHost(h_recv_lo));
            CUDA_CHECK(cudaFreeHost(h_recv_hi));
            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
            MPI_Finalize();
            return ok ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    MPI_Finalize();
    return 0;
}
