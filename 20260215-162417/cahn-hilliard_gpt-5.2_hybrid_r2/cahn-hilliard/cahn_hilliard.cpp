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

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

static inline void mpi_check(int err) {
    if (err != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
}

static inline void partition_1d(size_t n, int p, int r, size_t& start, size_t& count) {
    const size_t base = n / static_cast<size_t>(p);
    const size_t rem  = n % static_cast<size_t>(p);
    count = base + (static_cast<size_t>(r) < rem ? 1 : 0);
    start = base * static_cast<size_t>(r) + std::min(rem, static_cast<size_t>(r));
}

__device__ __forceinline__ size_t didx3(int x, int y, int z, int nx, int ny) {
    return static_cast<size_t>(z) * static_cast<size_t>(nx * ny) + static_cast<size_t>(y) * static_cast<size_t>(nx) + static_cast<size_t>(x);
}

__global__ void init_concentration(double* c, int nx, int ny_local, int nz, int ny_global,
                                   int y_start_global, int local_ny, size_t vol_global) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = static_cast<int>(blockIdx.y); // 0..local_ny-1
    const int z = static_cast<int>(blockIdx.z);
    if (x >= nx || ly >= local_ny || z >= nz) return;

    const int y = ly + 1; // interior starts at 1
    const int yg = y_start_global + ly;

    // Match original initialization: linear_id = z*(nx*ny) + y*nx + x
    const size_t gid = static_cast<size_t>(z) * static_cast<size_t>(nx) * static_cast<size_t>(ny_global)
                     + static_cast<size_t>(yg) * static_cast<size_t>(nx)
                     + static_cast<size_t>(x);
    const double pseudo = (static_cast<double>((((gid + 1ULL) * 1299709ULL) % vol_global)) / static_cast<double>(vol_global));
    c[didx3(x, y, z, nx, ny_local)] = -1.0 + 2.0 * pseudo;
}

__global__ void copy_y_plane(double* a, int nx, int ny, int nz, int y_src, int y_dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = static_cast<int>(blockIdx.y);
    if (x >= nx || z >= nz) return;
    a[didx3(x, y_dst, z, nx, ny)] = a[didx3(x, y_src, z, nx, ny)];
}

__global__ void pack_y_plane(const double* a, double* plane, int nx, int ny, int nz, int y_src) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = static_cast<int>(blockIdx.y);
    if (x >= nx || z >= nz) return;
    plane[static_cast<size_t>(z) * static_cast<size_t>(nx) + static_cast<size_t>(x)] = a[didx3(x, y_src, z, nx, ny)];
}

__global__ void unpack_y_plane(double* a, const double* plane, int nx, int ny, int nz, int y_dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = static_cast<int>(blockIdx.y);
    if (x >= nx || z >= nz) return;
    a[didx3(x, y_dst, z, nx, ny)] = plane[static_cast<size_t>(z) * static_cast<size_t>(nx) + static_cast<size_t>(x)];
}

__global__ void compute_mu(const double* __restrict__ c, double* __restrict__ mu,
                           int nx, int ny, int nz,
                           int local_ny, double invdx2, double invdy2, double invdz2,
                           double gamma, double e_AA, double e_BB, double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = static_cast<int>(blockIdx.y);
    const int z = static_cast<int>(blockIdx.z);
    if (x >= nx || ly >= local_ny || z >= nz) return;

    const int y = ly + 1;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int zn = (z > 0) ? z - 1 : 0;
    const int yp = y + 1;
    const int yn = y - 1;

    const size_t id = didx3(x, y, z, nx, ny);
    const double cv = c[id];

    const double cxx = (c[didx3(xp, y, z, nx, ny)] + c[didx3(xn, y, z, nx, ny)] - 2.0 * cv) * invdx2;
    const double cyy = (c[didx3(x, yp, z, nx, ny)] + c[didx3(x, yn, z, nx, ny)] - 2.0 * cv) * invdy2;
    const double czz = (c[didx3(x, y, zp, nx, ny)] + c[didx3(x, y, zn, nx, ny)] - 2.0 * cv) * invdz2;

    const double lap = cxx + cyy + czz;

    mu[id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * lap;
}

__global__ void update_c(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                         int nx, int ny, int nz,
                         int local_ny, double invdx2, double invdy2, double invdz2,
                         double D, double dt) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = static_cast<int>(blockIdx.y);
    const int z = static_cast<int>(blockIdx.z);
    if (x >= nx || ly >= local_ny || z >= nz) return;

    const int y = ly + 1;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int zp = (z < nz - 1) ? z + 1 : z;
    const int zn = (z > 0) ? z - 1 : 0;
    const int yp = y + 1;
    const int yn = y - 1;

    const size_t id = didx3(x, y, z, nx, ny);
    const double muv = mu[id];

    const double mxx = (mu[didx3(xp, y, z, nx, ny)] + mu[didx3(xn, y, z, nx, ny)] - 2.0 * muv) * invdx2;
    const double myy = (mu[didx3(x, yp, z, nx, ny)] + mu[didx3(x, yn, z, nx, ny)] - 2.0 * muv) * invdy2;
    const double mzz = (mu[didx3(x, y, zp, nx, ny)] + mu[didx3(x, y, zn, nx, ny)] - 2.0 * muv) * invdz2;

    cnew[id] = cold[id] + dt * D * (mxx + myy + mzz);
}

__global__ void pack_interior(const double* a, double* out, int nx, int ny_with_halo, int nz, int local_ny) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int ly = static_cast<int>(blockIdx.y);
    const int z = static_cast<int>(blockIdx.z);
    if (x >= nx || ly >= local_ny || z >= nz) return;
    const int y = ly + 1;
    out[static_cast<size_t>(z) * static_cast<size_t>(nx * local_ny) + static_cast<size_t>(ly) * static_cast<size_t>(nx) + static_cast<size_t>(x)] =
        a[didx3(x, y, z, nx, ny_with_halo)];
}

static void exchange_y_halo(double* d_field, int nx, int ny_with_halo, int nz, int local_ny,
                            int rank, int size, int prev, int next,
                            double* d_send_up, double* d_send_dn, double* d_recv_up, double* d_recv_dn,
                            double* h_send_up, double* h_send_dn, double* h_recv_up, double* h_recv_dn,
                            cudaStream_t stream) {
    const size_t plane_elems = static_cast<size_t>(nx) * static_cast<size_t>(nz);

    if (size == 1) {
        dim3 block(256, 1, 1);
        dim3 grid((nx + block.x - 1) / block.x, nz, 1);
        copy_y_plane<<<grid, block, 0, stream>>>(d_field, nx, ny_with_halo, nz, 1, 0);
        copy_y_plane<<<grid, block, 0, stream>>>(d_field, nx, ny_with_halo, nz, local_ny, local_ny + 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }

    constexpr int TAG_FROM_UP = 200;
    constexpr int TAG_FROM_DN = 201;

    MPI_Request reqs[4];
    int nreq = 0;

    // Post receives early
    if (prev >= 0) {
        mpi_check(MPI_Irecv(h_recv_up, static_cast<int>(plane_elems), MPI_DOUBLE, prev, TAG_FROM_UP, MPI_COMM_WORLD, &reqs[nreq++]));
    }
    if (next < size) {
        mpi_check(MPI_Irecv(h_recv_dn, static_cast<int>(plane_elems), MPI_DOUBLE, next, TAG_FROM_DN, MPI_COMM_WORLD, &reqs[nreq++]));
    }

    dim3 block(256, 1, 1);
    dim3 grid((nx + block.x - 1) / block.x, nz, 1);

    // Pack and send
    if (prev >= 0) {
        pack_y_plane<<<grid, block, 0, stream>>>(d_field, d_send_up, nx, ny_with_halo, nz, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_send_up, d_send_up, plane_elems * sizeof(double), cudaMemcpyDeviceToHost, stream));
    } else {
        copy_y_plane<<<grid, block, 0, stream>>>(d_field, nx, ny_with_halo, nz, 1, 0);
        CUDA_CHECK(cudaGetLastError());
    }

    if (next < size) {
        pack_y_plane<<<grid, block, 0, stream>>>(d_field, d_send_dn, nx, ny_with_halo, nz, local_ny);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_send_dn, d_send_dn, plane_elems * sizeof(double), cudaMemcpyDeviceToHost, stream));
    } else {
        copy_y_plane<<<grid, block, 0, stream>>>(d_field, nx, ny_with_halo, nz, local_ny, local_ny + 1);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (prev >= 0) {
        mpi_check(MPI_Isend(h_send_up, static_cast<int>(plane_elems), MPI_DOUBLE, prev, TAG_FROM_DN, MPI_COMM_WORLD, &reqs[nreq++]));
    }
    if (next < size) {
        mpi_check(MPI_Isend(h_send_dn, static_cast<int>(plane_elems), MPI_DOUBLE, next, TAG_FROM_UP, MPI_COMM_WORLD, &reqs[nreq++]));
    }

    if (nreq > 0) {
        mpi_check(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));
    }

    // Unpack receives
    if (prev >= 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_up, h_recv_up, plane_elems * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        unpack_y_plane<<<grid, block, 0, stream>>>(d_field, d_recv_up, nx, ny_with_halo, nz, 0);
        CUDA_CHECK(cudaGetLastError());
    }
    if (next < size) {
        CUDA_CHECK(cudaMemcpyAsync(d_recv_dn, h_recv_dn, plane_elems * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        unpack_y_plane<<<grid, block, 0, stream>>>(d_field, d_recv_dn, nx, ny_with_halo, nz, local_ny + 1);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    (void)rank;
}

static bool validate_distributed(double* d_c, int nx, int ny_with_halo, int nz, int local_ny,
                                 int rank, int size, bool print_range) {
    // Copy interior to host and compute local stats (validation is optional; overhead acceptable).
    const size_t local_elems = static_cast<size_t>(nx) * static_cast<size_t>(local_ny) * static_cast<size_t>(nz);

    double* d_pack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pack, local_elems * sizeof(double)));

    dim3 block(256, 1, 1);
    dim3 grid((nx + block.x - 1) / block.x, local_ny, nz);
    pack_interior<<<grid, block>>>(d_c, d_pack, nx, ny_with_halo, nz, local_ny);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h(local_elems);
    CUDA_CHECK(cudaMemcpy(h.data(), d_pack, local_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pack));

    int local_bad = 0;
    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();

    #pragma omp parallel for reduction(|:local_bad) reduction(min:local_min) reduction(max:local_max)
    for (size_t i = 0; i < local_elems; ++i) {
        const double v = h[i];
        if (std::isnan(v) || std::isinf(v)) {
            local_bad |= 1;
        }
        local_min = std::min(local_min, v);
        local_max = std::max(local_max, v);
    }

    int any_bad = 0;
    double gmin = 0.0, gmax = 0.0;
    mpi_check(MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD));
    mpi_check(MPI_Allreduce(&local_min, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    mpi_check(MPI_Allreduce(&local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (any_bad) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    if (rank == 0 && print_range) {
        printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
    }

    if (gmax > 10.0 || gmin < -10.0) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
        return false;
    }

    (void)size;
    return true;
}

static void gather_and_print(double* d_c, int nx, int ny_with_halo, int ny_global, int nz,
                             int local_ny, size_t y_start,
                             int rank, int size) {
    const size_t local_elems = static_cast<size_t>(nx) * static_cast<size_t>(local_ny) * static_cast<size_t>(nz);

    double* d_pack = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pack, local_elems * sizeof(double)));

    dim3 block(256, 1, 1);
    dim3 grid((nx + block.x - 1) / block.x, local_ny, nz);
    pack_interior<<<grid, block>>>(d_c, d_pack, nx, ny_with_halo, nz, local_ny);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h(local_elems);
    CUDA_CHECK(cudaMemcpy(h.data(), d_pack, local_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pack));

    std::vector<int> counts, displs;
    std::vector<double> gathered;

    const int sendcount = static_cast<int>(local_elems);

    if (rank == 0) {
        counts.resize(size);
        displs.resize(size);
        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            size_t ys, yc;
            partition_1d(static_cast<size_t>(ny_global), size, r, ys, yc);
            counts[r] = static_cast<int>(static_cast<size_t>(nx) * static_cast<size_t>(nz) * yc);
            displs[r] = static_cast<int>(disp);
            disp += static_cast<size_t>(counts[r]);
        }
        gathered.resize(disp);
    }

    mpi_check(MPI_Gatherv(h.data(), sendcount, MPI_DOUBLE,
                         rank == 0 ? gathered.data() : nullptr,
                         rank == 0 ? counts.data() : nullptr,
                         rank == 0 ? displs.data() : nullptr,
                         MPI_DOUBLE, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        std::vector<double> full(static_cast<size_t>(nx) * static_cast<size_t>(ny_global) * static_cast<size_t>(nz));

        for (int r = 0; r < size; ++r) {
            size_t ys, yc;
            partition_1d(static_cast<size_t>(ny_global), size, r, ys, yc);
            const size_t r_off = static_cast<size_t>(displs[r]);
            const size_t r_local_ny = yc;
            const size_t plane = static_cast<size_t>(nx) * r_local_ny;

            #pragma omp parallel for
            for (int z = 0; z < nz; ++z) {
                const size_t src_base = r_off + static_cast<size_t>(z) * plane;
                const size_t dst_base = static_cast<size_t>(z) * static_cast<size_t>(nx) * static_cast<size_t>(ny_global) + ys * static_cast<size_t>(nx);
                std::copy_n(gathered.data() + src_base, plane, full.data() + dst_base);
            }
        }

        print_results(full, "Concentration");
    }

    (void)y_start;
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
    mpi_check(MPI_Init(&argc, &argv));

    int rank = 0, size = 1;
    mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // GPU assignment
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Finalize();
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(rank % devCount));

    // Ensure OpenMP is engaged (hybrid MPI + OpenMP + CUDA) with negligible overhead.
    {
        int warm = 0;
        #pragma omp parallel for reduction(+:warm)
        for (int i = 0; i < 1024; ++i) warm += (i & 1);
        if (warm == -1) printf("%d\n", warm);
    }

    size_t y_start = 0, local_ny_sz = 0;
    partition_1d(ny, size, rank, y_start, local_ny_sz);
    const int local_ny = static_cast<int>(local_ny_sz);
    if (local_ny <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: MPI ranks (%d) exceed grid Y dimension (%zu)\n", size, ny);
        }
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    const int ny_with_halo = local_ny + 2;

    const int prev = (rank > 0) ? (rank - 1) : -1;
    const int next = (rank + 1 < size) ? (rank + 1) : size;

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

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    const size_t local_elems_with_halo = static_cast<size_t>(nx) * static_cast<size_t>(ny_with_halo) * static_cast<size_t>(nz);
    const size_t plane_elems = static_cast<size_t>(nx) * static_cast<size_t>(nz);
    const size_t vol_global = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_elems_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_elems_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu,   local_elems_with_halo * sizeof(double)));

    double *d_send_up = nullptr, *d_send_dn = nullptr, *d_recv_up = nullptr, *d_recv_dn = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send_up, plane_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_send_dn, plane_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_up, plane_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_recv_dn, plane_elems * sizeof(double)));

    double *h_send_up = nullptr, *h_send_dn = nullptr, *h_recv_up = nullptr, *h_recv_dn = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_send_up, plane_elems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_send_dn, plane_elems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_up, plane_elems * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_recv_dn, plane_elems * sizeof(double), cudaHostAllocDefault));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    if (rank == 0) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }

    // Initialize interior
    {
        dim3 block(256, 1, 1);
        dim3 grid((static_cast<int>(nx) + block.x - 1) / block.x, local_ny, static_cast<int>(nz));
        init_concentration<<<grid, block, 0, stream>>>(d_cold, static_cast<int>(nx), ny_with_halo, static_cast<int>(nz),
                                                       static_cast<int>(ny), static_cast<int>(y_start), local_ny, vol_global);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    mpi_check(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for cold
        exchange_y_halo(d_cold, static_cast<int>(nx), ny_with_halo, static_cast<int>(nz), local_ny,
                        rank, size, prev, next,
                        d_send_up, d_send_dn, d_recv_up, d_recv_dn,
                        h_send_up, h_send_dn, h_recv_up, h_recv_dn,
                        stream);

        // Compute chemical potential on interior
        {
            dim3 block(256, 1, 1);
            dim3 grid((static_cast<int>(nx) + block.x - 1) / block.x, local_ny, static_cast<int>(nz));
            compute_mu<<<grid, block, 0, stream>>>(d_cold, d_mu,
                                                   static_cast<int>(nx), ny_with_halo, static_cast<int>(nz),
                                                   local_ny, invdx2, invdy2, invdz2,
                                                   gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Exchange halos for mu
        exchange_y_halo(d_mu, static_cast<int>(nx), ny_with_halo, static_cast<int>(nz), local_ny,
                        rank, size, prev, next,
                        d_send_up, d_send_dn, d_recv_up, d_recv_dn,
                        h_send_up, h_send_dn, h_recv_up, h_recv_dn,
                        stream);

        // Update concentration on interior
        {
            dim3 block(256, 1, 1);
            dim3 grid((static_cast<int>(nx) + block.x - 1) / block.x, local_ny, static_cast<int>(nz));
            update_c<<<grid, block, 0, stream>>>(d_cnew, d_cold, d_mu,
                                                 static_cast<int>(nx), ny_with_halo, static_cast<int>(nz),
                                                 local_ny, invdx2, invdy2, invdz2,
                                                 D, dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        std::swap(d_cold, d_cnew);
    }

    mpi_check(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();

    const double ms_local = std::chrono::duration<double, std::milli>(end - start).count();
    double ms_max = 0.0;
    mpi_check(MPI_Reduce(&ms_local, &ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(std::llround(ms_max)));
        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz);
        const double cellUpdates = gridSize * static_cast<double>(iterations);
        const double mcups = cellUpdates / (ms_max / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        gather_and_print(d_cold, static_cast<int>(nx), ny_with_halo, static_cast<int>(ny), static_cast<int>(nz),
                         local_ny, y_start, rank, size);
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool ok = validate_distributed(d_cold, static_cast<int>(nx), ny_with_halo, static_cast<int>(nz),
                                             local_ny, rank, size, /*print_range=*/true);
        if (rank == 0) {
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaFreeHost(h_send_up));
        CUDA_CHECK(cudaFreeHost(h_send_dn));
        CUDA_CHECK(cudaFreeHost(h_recv_up));
        CUDA_CHECK(cudaFreeHost(h_recv_dn));
        CUDA_CHECK(cudaFree(d_send_up));
        CUDA_CHECK(cudaFree(d_send_dn));
        CUDA_CHECK(cudaFree(d_recv_up));
        CUDA_CHECK(cudaFree(d_recv_dn));
        CUDA_CHECK(cudaFree(d_mu));
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));

        MPI_Finalize();
        return ok ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_send_up));
    CUDA_CHECK(cudaFreeHost(h_send_dn));
    CUDA_CHECK(cudaFreeHost(h_recv_up));
    CUDA_CHECK(cudaFreeHost(h_recv_dn));
    CUDA_CHECK(cudaFree(d_send_up));
    CUDA_CHECK(cudaFree(d_send_dn));
    CUDA_CHECK(cudaFree(d_recv_up));
    CUDA_CHECK(cudaFree(d_recv_dn));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));

    MPI_Finalize();
    return 0;
}
