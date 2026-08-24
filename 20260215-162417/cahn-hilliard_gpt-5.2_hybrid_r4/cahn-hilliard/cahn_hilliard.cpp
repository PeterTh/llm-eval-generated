#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

// 3D index calculation (row-major, x fastest)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

static inline void decompose_1d_z(size_t nz, int rank, int nprocs, size_t& z_start, size_t& local_nz) {
    const size_t base = nz / static_cast<size_t>(nprocs);
    const size_t rem = nz % static_cast<size_t>(nprocs);
    const size_t r = static_cast<size_t>(rank);
    local_nz = base + (r < rem ? 1 : 0);
    z_start = base * r + (r < rem ? r : rem);
}

static inline void set_device_for_rank(int rank) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % ndev));
}

static inline double* plane_ptr(double* base, size_t z, size_t plane_elems) {
    return base + z * plane_elems;
}
static inline const double* plane_ptr(const double* base, size_t z, size_t plane_elems) {
    return base + z * plane_elems;
}

__global__ void compute_mu_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                 size_t nx, size_t ny, size_t local_nz,
                                 size_t z_begin, size_t z_end,
                                 double invdx2, double invdy2, double invdz2,
                                 double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + z_begin;

    if (x >= nx || y >= ny || z < z_begin || z > z_end || z > local_nz) return;

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double cv = c[idx];

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * cv) * invdx2;
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * cv) * invdy2;
    const double czz = (c[idx3(x, y, z + 1, nx, ny)] + c[idx3(x, y, z - 1, nx, ny)] - 2.0 * cv) * invdz2;

    const double lap = cxx + cyy + czz;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv
            - gamma * lap;
}

__global__ void update_c_kernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                               size_t nx, size_t ny, size_t local_nz,
                               size_t z_begin, size_t z_end,
                               double invdx2, double invdy2, double invdz2,
                               double Ddt) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + z_begin;

    if (x >= nx || y >= ny || z < z_begin || z > z_end || z > local_nz) return;

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t idx = idx3(x, y, z, nx, ny);
    const double muv = mu[idx];

    const double mxx = (mu[idx3(xp, y, z, nx, ny)] + mu[idx3(xn, y, z, nx, ny)] - 2.0 * muv) * invdx2;
    const double myy = (mu[idx3(x, yp, z, nx, ny)] + mu[idx3(x, yn, z, nx, ny)] - 2.0 * muv) * invdy2;
    const double mzz = (mu[idx3(x, y, z + 1, nx, ny)] + mu[idx3(x, y, z - 1, nx, ny)] - 2.0 * muv) * invdz2;

    cnew[idx] = cold[idx] + Ddt * (mxx + myy + mzz);
}

static inline void launch_mu_range(const double* d_c, double* d_mu,
                                  size_t nx, size_t ny, size_t local_nz,
                                  size_t z_begin, size_t z_end,
                                  double dx, double dy, double dz,
                                  double gamma, double e_AA, double e_BB, double e_AB,
                                  cudaStream_t stream) {
    if (z_begin > z_end || z_begin < 1 || z_end > local_nz) return;

    const dim3 block(32, 4, 2);
    const size_t z_span = z_end - z_begin + 1;
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (z_span + block.z - 1) / block.z);

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    compute_mu_kernel<<<grid, block, 0, stream>>>(d_c, d_mu, nx, ny, local_nz, z_begin, z_end,
                                                  invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

static inline void launch_update_range(double* d_cnew, const double* d_cold, const double* d_mu,
                                      size_t nx, size_t ny, size_t local_nz,
                                      size_t z_begin, size_t z_end,
                                      double dx, double dy, double dz,
                                      double D, double dt,
                                      cudaStream_t stream) {
    if (z_begin > z_end || z_begin < 1 || z_end > local_nz) return;

    const dim3 block(32, 4, 2);
    const size_t z_span = z_end - z_begin + 1;
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (z_span + block.z - 1) / block.z);

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    update_c_kernel<<<grid, block, 0, stream>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, z_begin, z_end,
                                                invdx2, invdy2, invdz2, D * dt);
    CUDA_CHECK(cudaGetLastError());
}

static inline void memcpy_plane_async(void* dst, const void* src, size_t plane_elems, cudaMemcpyKind kind, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, plane_elems * sizeof(double), kind, stream));
}

static inline void copy_plane_device_to_device_async(double* d_base, size_t z_dst, size_t z_src, size_t plane_elems, cudaStream_t stream) {
    memcpy_plane_async(plane_ptr(d_base, z_dst, plane_elems), plane_ptr(d_base, z_src, plane_elems),
                       plane_elems, cudaMemcpyDeviceToDevice, stream);
}

static inline void exchange_halos_z_overlap(double* d_field,
                                           size_t nx, size_t ny, size_t local_nz,
                                           int rank, int nprocs,
                                           double* h_send_lo, double* h_recv_lo,
                                           double* h_send_hi, double* h_recv_hi,
                                           cudaStream_t stream_halo) {
    const size_t plane = nx * ny;

    // Global clamped boundaries: ghost plane mirrors edge interior plane.
    if (rank == 0) {
        copy_plane_device_to_device_async(d_field, 0, 1, plane, stream_halo);
    }
    if (rank == nprocs - 1) {
        copy_plane_device_to_device_async(d_field, local_nz + 1, local_nz, plane, stream_halo);
    }

    // Start D2H copies for planes that will be sent.
    if (rank > 0) {
        memcpy_plane_async(h_send_lo, plane_ptr(d_field, 1, plane), plane, cudaMemcpyDeviceToHost, stream_halo);
    }
    if (rank < nprocs - 1) {
        memcpy_plane_async(h_send_hi, plane_ptr(d_field, local_nz, plane), plane, cudaMemcpyDeviceToHost, stream_halo);
    }

    // Ensure host send buffers are ready, then do MPI exchange.
    CUDA_CHECK(cudaStreamSynchronize(stream_halo));

    if (rank > 0) {
        MPI_Sendrecv(h_send_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 301,
                     h_recv_lo, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 300,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy_plane_async(plane_ptr(d_field, 0, plane), h_recv_lo, plane, cudaMemcpyHostToDevice, stream_halo);
    }
    if (rank < nprocs - 1) {
        MPI_Sendrecv(h_send_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 300,
                     h_recv_hi, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 301,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        memcpy_plane_async(plane_ptr(d_field, local_nz + 1, plane), h_recv_hi, plane, cudaMemcpyHostToDevice, stream_halo);
    }

    // Ensure ghosts are updated before any boundary-plane computation uses them.
    CUDA_CHECK(cudaStreamSynchronize(stream_halo));
}

static inline void initialize_concentration_local(std::vector<double>& host_with_halos,
                                                  size_t nx, size_t ny, size_t nz_global,
                                                  size_t z_start, size_t local_nz) {
    const size_t plane = nx * ny;
    const size_t vol = nx * ny * nz_global;

    // Fill interior.
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t zg = z_start + z;
                const size_t idx_local = idx3(x, y, z + 1, nx, ny);
                const size_t linear_id = zg * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) / static_cast<double>(vol));
                host_with_halos[idx_local] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // Initialize halos by clamping.
    std::copy_n(host_with_halos.data() + 1 * plane, plane, host_with_halos.data() + 0 * plane);
    std::copy_n(host_with_halos.data() + local_nz * plane, plane, host_with_halos.data() + (local_nz + 1) * plane);
}

static inline bool validate_distributed(const double* d_c,
                                       size_t nx, size_t ny, size_t local_nz,
                                       int rank) {
    const size_t plane = nx * ny;
    const size_t local_elems = local_nz * plane;

    std::vector<double> local(local_elems);
    CUDA_CHECK(cudaMemcpy(local.data(), plane_ptr(d_c, 1, plane), local_elems * sizeof(double), cudaMemcpyDeviceToHost));

    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    #pragma omp parallel for reduction(|:local_bad) reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < local_elems; ++i) {
        const double v = local[i];
        if (std::isnan(v) || std::isinf(v)) local_bad = 1;
        local_min = std::min(local_min, v);
        local_max = std::max(local_max, v);
    }

    int any_bad = 0;
    MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&local_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        if (any_bad) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
        if (gmax > 10.0 || gmin < -10.0) {
            printf("Validation failed: values out of expected range\n");
            any_bad = 1;
        }
    }

    MPI_Bcast(&any_bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return any_bad == 0;
}

static inline void gather_and_print_results(const double* d_c,
                                           size_t nx, size_t ny, size_t nz,
                                           size_t local_nz,
                                           int rank, int nprocs) {
    const size_t plane = nx * ny;
    const size_t local_elems = local_nz * plane;

    std::vector<double> local(local_elems);
    CUDA_CHECK(cudaMemcpy(local.data(), plane_ptr(d_c, 1, plane), local_elems * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> global;

    if (rank == 0) {
        counts.resize(nprocs);
        displs.resize(nprocs);
        size_t disp = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t zs = 0, lnz = 0;
            decompose_1d_z(nz, r, nprocs, zs, lnz);
            counts[r] = static_cast<int>(lnz * plane);
            displs[r] = static_cast<int>(disp);
            disp += static_cast<size_t>(counts[r]);
        }
        global.resize(nx * ny * nz);
    }

    MPI_Gatherv(local.data(), static_cast<int>(local_elems), MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_results(global, "Concentration");
    }
}

static void printUsage(const char* progName) {
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    set_device_for_rank(rank);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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

    if (static_cast<size_t>(nprocs) > nz) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds nz (%zu).\n", nprocs, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", nprocs);
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
    }

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

    size_t z_start = 0, local_nz = 0;
    decompose_1d_z(nz, rank, nprocs, z_start, local_nz);

    const size_t plane = nx * ny;
    const size_t local_with_halos = (local_nz + 2) * plane;

    // Host init buffer (with halos) then copy to device.
    std::vector<double> h_cold(local_with_halos, 0.0);
    initialize_concentration_local(h_cold, nx, ny, nz, z_start, local_nz);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_with_halos * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_with_halos * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_with_halos * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_cold, h_cold.data(), local_with_halos * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned host buffers for halo exchange (portable without CUDA-aware MPI)
    double *h_send_lo = nullptr, *h_recv_lo = nullptr, *h_send_hi = nullptr, *h_recv_hi = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_send_lo, plane * sizeof(double), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_recv_lo, plane * sizeof(double), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_send_hi, plane * sizeof(double), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_recv_hi, plane * sizeof(double), cudaHostAllocPortable));

    cudaStream_t stream_compute;
    cudaStream_t stream_halo;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_halo, cudaStreamNonBlocking));

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Overlap: compute interior mu while exchanging cold halos (needed for boundary planes).
        const bool has_interior = (local_nz >= 3);
        if (has_interior) {
            launch_mu_range(d_cold, d_mu, nx, ny, local_nz, 2, local_nz - 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB, stream_compute);
        }

        exchange_halos_z_overlap(d_cold, nx, ny, local_nz, rank, nprocs, h_send_lo, h_recv_lo, h_send_hi, h_recv_hi, stream_halo);

        // Boundary planes of mu require fresh halos.
        launch_mu_range(d_cold, d_mu, nx, ny, local_nz, 1, 1, dx, dy, dz, gamma, e_AA, e_BB, e_AB, stream_compute);
        launch_mu_range(d_cold, d_mu, nx, ny, local_nz, local_nz, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, stream_compute);

        // Ensure all mu is ready before update computations use it.
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        // Overlap: compute interior update while exchanging mu halos (needed for boundary update planes).
        if (has_interior) {
            launch_update_range(d_cnew, d_cold, d_mu, nx, ny, local_nz, 2, local_nz - 1, dx, dy, dz, D, dt, stream_compute);
        }

        exchange_halos_z_overlap(d_mu, nx, ny, local_nz, rank, nprocs, h_send_lo, h_recv_lo, h_send_hi, h_recv_hi, stream_halo);

        // Boundary planes of update require fresh mu halos.
        launch_update_range(d_cnew, d_cold, d_mu, nx, ny, local_nz, 1, 1, dx, dy, dz, D, dt, stream_compute);
        launch_update_range(d_cnew, d_cold, d_mu, nx, ny, local_nz, local_nz, local_nz, dx, dy, dz, D, dt, stream_compute);

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_elapsed = t1 - t0;
    double global_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double elapsed_ms = global_elapsed * 1000.0;
        printf("Computation time: %.3f ms\n", elapsed_ms);

        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz);
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double mcups = cellUpdates / global_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        gather_and_print_results(d_cold, nx, ny, nz, local_nz, rank, nprocs);
    }

    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool ok = validate_distributed(d_cold, nx, ny, local_nz, rank);
        if (rank == 0) printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        ret = ok ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_halo));

    CUDA_CHECK(cudaFreeHost(h_send_lo));
    CUDA_CHECK(cudaFreeHost(h_recv_lo));
    CUDA_CHECK(cudaFreeHost(h_send_hi));
    CUDA_CHECK(cudaFreeHost(h_recv_hi));

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return ret;
}
