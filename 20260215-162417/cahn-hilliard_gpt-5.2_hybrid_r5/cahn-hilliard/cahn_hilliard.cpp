#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int TAG_LOWER = 100; // message fills receiver's lower halo (z=0)
constexpr int TAG_UPPER = 101; // message fills receiver's upper halo (z=local_nz+1)

inline void mpi_check(const int err, const char* what) {
    if (err != MPI_SUCCESS) {
        char msg[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(err, msg, &len);
        fprintf(stderr, "MPI error in %s: %.*s\n", what, len, msg);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define MPI_CHECK(call) mpi_check((call), #call)

inline void cuda_check(const cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call)

__device__ __forceinline__ int clampi(const int v, const int lo, const int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void kernel_compute_mu(double* __restrict__ mu,
                                 const double* __restrict__ c,
                                 int nx, int ny, int z_start, int z_count,
                                 double invdx2, double invdy2, double invdz2,
                                 double gamma, double e_AA, double e_BB, double e_AB) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int z_local = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z_local >= z_count) return;

    const int z = z_start + z_local; // includes halos in indexing; interior z in [1, local_nz]
    const size_t plane = (size_t)nx * (size_t)ny;

    const int xp = (x + 1 < nx) ? (x + 1) : x;
    const int xn = (x > 0) ? (x - 1) : 0;
    const int yp = (y + 1 < ny) ? (y + 1) : y;
    const int yn = (y > 0) ? (y - 1) : 0;

    const size_t base_z = (size_t)z * plane;
    const size_t idx = base_z + (size_t)y * (size_t)nx + (size_t)x;

    const double cv = c[idx];

    const size_t base_y  = base_z + (size_t)y * (size_t)nx;
    const size_t base_yp = base_z + (size_t)yp * (size_t)nx;
    const size_t base_yn = base_z + (size_t)yn * (size_t)nx;
    const size_t base_zp = (size_t)(z + 1) * plane + (size_t)y * (size_t)nx;
    const size_t base_zn = (size_t)(z - 1) * plane + (size_t)y * (size_t)nx;

    const double cxx = (c[base_y + (size_t)xp] + c[base_y + (size_t)xn] - 2.0 * cv) * invdx2;
    const double cyy = (c[base_yp + (size_t)x] + c[base_yn + (size_t)x] - 2.0 * cv) * invdy2;
    const double czz = (c[base_zp + (size_t)x] + c[base_zn + (size_t)x] - 2.0 * cv) * invdz2;
    const double lap = cxx + cyy + czz;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
}

__global__ void kernel_update(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu,
                             int nx, int ny, int z_start, int z_count,
                             double invdx2, double invdy2, double invdz2,
                             double D, double dt) {
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int z_local = (int)(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= nx || y >= ny || z_local >= z_count) return;

    const int z = z_start + z_local;
    const size_t plane = (size_t)nx * (size_t)ny;

    const int xp = (x + 1 < nx) ? (x + 1) : x;
    const int xn = (x > 0) ? (x - 1) : 0;
    const int yp = (y + 1 < ny) ? (y + 1) : y;
    const int yn = (y > 0) ? (y - 1) : 0;

    const size_t base_z = (size_t)z * plane;
    const size_t idx = base_z + (size_t)y * (size_t)nx + (size_t)x;

    const double muv = mu[idx];

    const size_t base_y  = base_z + (size_t)y * (size_t)nx;
    const size_t base_yp = base_z + (size_t)yp * (size_t)nx;
    const size_t base_yn = base_z + (size_t)yn * (size_t)nx;
    const size_t base_zp = (size_t)(z + 1) * plane + (size_t)y * (size_t)nx;
    const size_t base_zn = (size_t)(z - 1) * plane + (size_t)y * (size_t)nx;

    const double mxx = (mu[base_y + (size_t)xp] + mu[base_y + (size_t)xn] - 2.0 * muv) * invdx2;
    const double myy = (mu[base_yp + (size_t)x] + mu[base_yn + (size_t)x] - 2.0 * muv) * invdy2;
    const double mzz = (mu[base_zp + (size_t)x] + mu[base_zn + (size_t)x] - 2.0 * muv) * invdz2;
    const double lap = mxx + myy + mzz;

    cnew[idx] = cold[idx] + dt * D * lap;
}

inline void printUsage(const char* progName) {
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

inline bool validateResult(const std::vector<double>& c) {
    // Check for NaN or Inf
    bool ok = true;
#pragma omp parallel for reduction(&& : ok)
    for (size_t i = 0; i < c.size(); ++i) {
        const double v = c[i];
        if (std::isnan(v) || std::isinf(v)) ok = false;
    }
    if (!ok) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel
    {
        double tmin = minVal;
        double tmax = maxVal;
#pragma omp for nowait
        for (size_t i = 0; i < c.size(); ++i) {
            tmin = std::min(tmin, c[i]);
            tmax = std::max(tmax, c[i]);
        }
#pragma omp critical
        {
            minVal = std::min(minVal, tmin);
            maxVal = std::max(maxVal, tmax);
        }
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

struct Decomp {
    int rank = 0;
    int size = 1;
    int nz_local = 0;
    int z0 = 0;
};

inline Decomp decompose_z(const int nz, const int rank, const int size) {
    const int base = nz / size;
    const int rem = nz % size;
    const int nz_local = base + (rank < rem ? 1 : 0);
    const int z0 = rank * base + (rank < rem ? rank : rem);
    return Decomp{rank, size, nz_local, z0};
}

inline void exchange_halo_z(double* d_field,
                           const size_t plane_elems,
                           const int nz_local,
                           const int rank,
                           const int size,
                           double* send_lo_h,
                           double* send_hi_h,
                           double* recv_lo_h,
                           double* recv_hi_h,
                           cudaStream_t stream_copy) {
    const size_t bytes = plane_elems * sizeof(double);
    const size_t off_lo = plane_elems * (size_t)1;
    const size_t off_hi = plane_elems * (size_t)nz_local;

    MPI_Request reqs[4];
    int nreq = 0;

    // Pack device boundary planes to host send buffers
    if (size > 1) {
        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(send_lo_h, d_field + off_lo, bytes, cudaMemcpyDeviceToHost, stream_copy));
        if (rank < size - 1) CUDA_CHECK(cudaMemcpyAsync(send_hi_h, d_field + off_hi, bytes, cudaMemcpyDeviceToHost, stream_copy));
        CUDA_CHECK(cudaStreamSynchronize(stream_copy));

        if (rank > 0) MPI_CHECK(MPI_Irecv(recv_lo_h, (int)plane_elems, MPI_DOUBLE, rank - 1, TAG_LOWER, MPI_COMM_WORLD, &reqs[nreq++]));
        if (rank < size - 1) MPI_CHECK(MPI_Irecv(recv_hi_h, (int)plane_elems, MPI_DOUBLE, rank + 1, TAG_UPPER, MPI_COMM_WORLD, &reqs[nreq++]));

        if (rank > 0) MPI_CHECK(MPI_Isend(send_lo_h, (int)plane_elems, MPI_DOUBLE, rank - 1, TAG_UPPER, MPI_COMM_WORLD, &reqs[nreq++]));
        if (rank < size - 1) MPI_CHECK(MPI_Isend(send_hi_h, (int)plane_elems, MPI_DOUBLE, rank + 1, TAG_LOWER, MPI_COMM_WORLD, &reqs[nreq++]));

        if (nreq) MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

        if (rank > 0) CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)0, recv_lo_h, bytes, cudaMemcpyHostToDevice, stream_copy));
        else CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)0, d_field + off_lo, bytes, cudaMemcpyDeviceToDevice, stream_copy));

        if (rank < size - 1) CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)(nz_local + 1), recv_hi_h, bytes, cudaMemcpyHostToDevice, stream_copy));
        else CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)(nz_local + 1), d_field + off_hi, bytes, cudaMemcpyDeviceToDevice, stream_copy));

        CUDA_CHECK(cudaStreamSynchronize(stream_copy));
        return;
    }

    // Single rank: clamped boundary conditions in z via halo copies
    CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)0, d_field + off_lo, bytes, cudaMemcpyDeviceToDevice, stream_copy));
    CUDA_CHECK(cudaMemcpyAsync(d_field + plane_elems * (size_t)(nz_local + 1), d_field + off_hi, bytes, cudaMemcpyDeviceToDevice, stream_copy));
    CUDA_CHECK(cudaStreamSynchronize(stream_copy));
}

inline void launch_mu(cudaStream_t stream_compute,
                     double* d_mu,
                     const double* d_c,
                     int nx, int ny,
                     int z_start, int z_count,
                     double invdx2, double invdy2, double invdz2,
                     double gamma, double e_AA, double e_BB, double e_AB) {
    if (z_count <= 0) return;
    const dim3 block(8, 8, 4);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)((z_count + (int)block.z - 1) / (int)block.z));
    kernel_compute_mu<<<grid, block, 0, stream_compute>>>(d_mu, d_c, nx, ny, z_start, z_count, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
}

inline void launch_update(cudaStream_t stream_compute,
                         double* d_cnew,
                         const double* d_cold,
                         const double* d_mu,
                         int nx, int ny,
                         int z_start, int z_count,
                         double invdx2, double invdy2, double invdz2,
                         double D, double dt) {
    if (z_count <= 0) return;
    const dim3 block(8, 8, 4);
    const dim3 grid((unsigned)((nx + block.x - 1) / block.x),
                    (unsigned)((ny + block.y - 1) / block.y),
                    (unsigned)((z_count + (int)block.z - 1) / (int)block.z));
    kernel_update<<<grid, block, 0, stream_compute>>>(d_cnew, d_cold, d_mu, nx, ny, z_start, z_count, invdx2, invdy2, invdz2, D, dt);
}

} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse command line (rank 0) then broadcast
    uint64_t nx_u = 64, ny_u = 0, nz_u = 0;
    int iterations = 20;
    int validate_i = 0;
    int printResults_i = 0;
    int early_exit = -1; // -1 continue, 0 help, 1 error

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                nx_u = (uint64_t)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                ny_u = (uint64_t)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                nz_u = (uint64_t)atoll(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                early_exit = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                early_exit = 1;
                break;
            }
        }
        if (early_exit == -1) {
            if (ny_u == 0) ny_u = nx_u;
            if (nz_u == 0) nz_u = nx_u;
        }
    }

    MPI_CHECK(MPI_Bcast(&early_exit, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (early_exit != -1) {
        MPI_CHECK(MPI_Finalize());
        return early_exit;
    }
    MPI_CHECK(MPI_Bcast(&nx_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&ny_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&nz_u, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD));

    const int nx = (int)nx_u;
    const int ny = (int)ny_u;
    const int nz = (int)nz_u;
    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    if (nx <= 0 || ny <= 0 || nz <= 0 || iterations < 0) {
        if (rank == 0) fprintf(stderr, "Invalid parameters\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const Decomp dc = decompose_z(nz, rank, size);
    if (dc.nz_local <= 0) {
        if (rank == 0) fprintf(stderr, "MPI size (%d) larger than nz (%d); unsupported.\n", size, nz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU per-rank
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    cudaStream_t stream_compute{}, stream_copy{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_copy, cudaStreamNonBlocking));

    // Physical parameters (same as original)
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    const size_t plane = (size_t)nx * (size_t)ny;
    const size_t local_with_halo = plane * (size_t)(dc.nz_local + 2);
    const size_t local_interior = plane * (size_t)dc.nz_local;

    // Host initialization (interior only), consistent with original global indexing
    std::vector<double> h_cold(local_with_halo, 0.0);

    const size_t vol = (size_t)nx * (size_t)ny * (size_t)nz;
#pragma omp parallel for collapse(3)
    for (int z = 0; z < dc.nz_local; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int gz = dc.z0 + z;
                const size_t linear_id = (size_t)gz * plane + (size_t)y * (size_t)nx + (size_t)x;
                const double pseudo = (double)((((linear_id + 1) * (size_t)1299709) % vol)) / (double)vol;
                const double v = -1.0 + 2.0 * pseudo;
                h_cold[plane * (size_t)(z + 1) + (size_t)y * (size_t)nx + (size_t)x] = v;
            }
        }
    }

    // Device arrays (include z halos)
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_with_halo * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(d_cold, h_cold.data(), local_with_halo * sizeof(double), cudaMemcpyHostToDevice, stream_copy));
    CUDA_CHECK(cudaStreamSynchronize(stream_copy));

    // Pinned host buffers for halo exchange
    double *send_lo_h = nullptr, *send_hi_h = nullptr, *recv_lo_h = nullptr, *recv_hi_h = nullptr;
    CUDA_CHECK(cudaHostAlloc(&send_lo_h, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&send_hi_h, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recv_lo_h, plane * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&recv_hi_h, plane * sizeof(double), cudaHostAllocDefault));

    // Warm up halos
    exchange_halo_z(d_cold, plane, dc.nz_local, rank, size, send_lo_h, send_hi_h, recv_lo_h, recv_hi_h, stream_copy);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t0 = MPI_Wtime();

    for (int it = 0; it < iterations; ++it) {
        // Exchange halos for concentration field
        // Overlap: compute interior z=[2..nz_local-1] while exchanging boundary planes
        const int interior_count = dc.nz_local - 2;
        if (interior_count > 0) {
            launch_mu(stream_compute, d_mu, d_cold, nx, ny, 2, interior_count, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        }
        exchange_halo_z(d_cold, plane, dc.nz_local, rank, size, send_lo_h, send_hi_h, recv_lo_h, recv_hi_h, stream_copy);
        // Boundary planes
        launch_mu(stream_compute, d_mu, d_cold, nx, ny, 1, 1, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        if (dc.nz_local > 1) {
            launch_mu(stream_compute, d_mu, d_cold, nx, ny, dc.nz_local, 1, invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        }

        CUDA_CHECK(cudaGetLastError());

        // Exchange halos for chemical potential
        if (interior_count > 0) {
            launch_update(stream_compute, d_cnew, d_cold, d_mu, nx, ny, 2, interior_count, invdx2, invdy2, invdz2, D, dt);
        }
        exchange_halo_z(d_mu, plane, dc.nz_local, rank, size, send_lo_h, send_hi_h, recv_lo_h, recv_hi_h, stream_copy);
        launch_update(stream_compute, d_cnew, d_cold, d_mu, nx, ny, 1, 1, invdx2, invdy2, invdz2, D, dt);
        if (dc.nz_local > 1) {
            launch_update(stream_compute, d_cnew, d_cold, d_mu, nx, ny, dc.nz_local, 1, invdx2, invdy2, invdz2, D, dt);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double t1 = MPI_Wtime();

    double local_s = t1 - t0;
    double max_s = 0.0;
    MPI_CHECK(MPI_Reduce(&local_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)llround(max_s * 1000.0));
        const double gridSize = (double)nx * (double)ny * (double)nz;
        const double cellUpdates = gridSize * (double)iterations;
        const double mcups = cellUpdates / max_s / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather final concentration to rank 0 for printing/validation (semantics match original output)
    std::vector<double> h_local_interior(local_interior);
    CUDA_CHECK(cudaMemcpyAsync(h_local_interior.data(), d_cold + plane, local_interior * sizeof(double), cudaMemcpyDeviceToHost, stream_copy));
    CUDA_CHECK(cudaStreamSynchronize(stream_copy));

    std::vector<int> counts, displs;
    if (rank == 0) {
        counts.resize((size_t)size);
        displs.resize((size_t)size);
        int disp = 0;
        for (int r = 0; r < size; ++r) {
            const Decomp dr = decompose_z(nz, r, size);
            counts[(size_t)r] = (int)(plane * (size_t)dr.nz_local);
            displs[(size_t)r] = disp;
            disp += counts[(size_t)r];
        }
    }

    std::vector<double> h_global;
    if (rank == 0) h_global.resize((size_t)nx * (size_t)ny * (size_t)nz);

    MPI_CHECK(MPI_Gatherv(h_local_interior.data(), (int)local_interior, MPI_DOUBLE,
                         rank == 0 ? h_global.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr,
                         MPI_DOUBLE, 0, MPI_COMM_WORLD));

    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(h_global, "Concentration");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(h_global);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            ret = valid ? 0 : 1;
        }
    }

    MPI_CHECK(MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaFreeHost(send_lo_h));
    CUDA_CHECK(cudaFreeHost(send_hi_h));
    CUDA_CHECK(cudaFreeHost(recv_lo_h));
    CUDA_CHECK(cudaFreeHost(recv_hi_h));

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_copy));

    MPI_CHECK(MPI_Finalize());
    return ret;
}
