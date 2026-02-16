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

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while (0)

// 3D local index calculation (x fastest)
inline constexpr size_t idx3_loc(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CPU laplacian for local arrays with halos (z includes halos 0..nz_local+1)
inline double laplacian_local(const std::vector<double>& a, size_t nx, size_t ny, size_t nz_local_with_halo,
                              size_t x, size_t y, size_t z, double dx, double dy, double dz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz_local_with_halo - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double c = a[idx3_loc(x,y,z,nx,ny)];
    const double cxx = (a[idx3_loc(xp,y,z,nx,ny)] + a[idx3_loc(xn,y,z,nx,ny)] - 2.0 * c) / (dx*dx);
    const double cyy = (a[idx3_loc(x,yp,z,nx,ny)] + a[idx3_loc(x,yn,z,nx,ny)] - 2.0 * c) / (dy*dy);
    const double czz = (a[idx3_loc(x,y,zp,nx,ny)] + a[idx3_loc(x,y,zn,nx,ny)] - 2.0 * c) / (dz*dz);
    return cxx + cyy + czz;
}

// CUDA kernel to perform the update step on device for interior z slices (1..nz_local)
extern "C" __global__ void update_kernel(const double* cold, const double* mu, double* cnew,
                                           size_t nx, size_t ny, size_t nz_with_halo,
                                           double D, double dt, double dx, double dy, double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z; // z indexes interior starting at 1

    if (x >= nx || y >= ny || z >= nz_with_halo - 2) return; // z in [0..local_nz-1] interior index offset

    size_t z_local = z + 1; // shift to account for halo at 0
    size_t idx = z_local * (nx * ny) + y * nx + x;

    // compute laplacian of mu
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = z_local + 1;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = z_local - 1;

    double center = mu[idx];
    double cxx = (mu[z_local*(nx*ny) + y*nx + xp] + mu[z_local*(nx*ny) + y*nx + xn] - 2.0 * center) / (dx*dx);
    double cyy = (mu[z_local*(nx*ny) + yp*nx + x] + mu[z_local*(nx*ny) + yn*nx + x] - 2.0 * center) / (dy*dy);
    double czz = (mu[zp*(nx*ny) + y*nx + x] + mu[zn*(nx*ny) + y*nx + x] - 2.0 * center) / (dz*dz);

    double lap = cxx + cyy + czz;
    cnew[idx] = cold[idx] + dt * D * lap;
}

// Initialize concentration pseudo-randomly (only interior slices) in local buffer with halos
void initialize_local(std::vector<double>& cold_local, size_t nx, size_t ny, size_t local_nz, size_t z_offset_global) {
    size_t vol = nx * ny * local_nz; // local volume (excluding halos)

    // interior z indices 1..local_nz
    #pragma omp parallel for collapse(3)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx3_loc(x,y,z,nx,ny);
                size_t global_z = z_offset_global + (z-1);
                size_t linear_id = global_z * (nx * ny) + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709ULL) % (nx*ny*local_nz)) / static_cast<double>(nx*ny*local_nz));
                cold_local[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Compute chemical potential on CPU for interior slices given cold_local (with halos)
void computeChemicalPotential_local(const std::vector<double>& cold_local, std::vector<double>& mu_local,
                                    size_t nx, size_t ny, size_t local_nz_with_halo,
                                    double dx, double dy, double dz, double gamma,
                                    double e_AA, double e_BB, double e_AB) {
    size_t local_nz = local_nz_with_halo - 2;
    #pragma omp parallel for collapse(3)
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t idx = idx3_loc(x,y,z,nx,ny);
                double cv = cold_local[idx];
                double chem = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv*cv*cv;
                double lap = laplacian_local(cold_local, nx, ny, local_nz_with_halo, x, y, z, dx, dy, dz);
                mu_local[idx] = chem - gamma * lap;
            }
        }
    }
}

// Validate across ranks by reducing min/max and NaN/Inf detection
bool validate_global(MPI_Comm comm, const std::vector<double>& cold_local, size_t nx, size_t ny, size_t local_nz_with_halo, size_t local_nz) {
    // Compute local min/max and NaN presence over interior only
    double local_min = 1e300, local_max = -1e300;
    int local_bad = 0;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                double v = cold_local[idx3_loc(x,y,z,nx,ny)];
                if (std::isnan(v) || std::isinf(v)) local_bad = 1;
                local_min = std::min(local_min, v);
                local_max = std::max(local_max, v);
            }
        }
    }

    double global_min, global_max;
    int global_bad;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_MAX, comm);

    if (global_bad) {
        if (MPI::COMM_WORLD.Get_rank() == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (global_max > 10.0 || global_min < -10.0) {
        if (MPI::COMM_WORLD.Get_rank() == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    if (MPI::COMM_WORLD.Get_rank() == 0) printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse args (only rank 0 prints usage)
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
            if (rank==0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n  -x <num> Grid X (default 64)\n  -y <num> Grid Y (default X)\n  -z <num> Grid Z (default X)\n  -i <num> iterations (default 20)\n  -v validate\n  -r print results\n");
            }
            MPI_Finalize();
            return 0;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = (2.0/9.0);
    const double gamma = 0.5, D = 1.0;

    // Decompose Z among ranks (contiguous blocks)
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t local_nz = base + (rank < (int)rem ? 1 : 0);
    size_t z_offset = rank * base + std::min((size_t)rank, rem);

    // local arrays include two halos
    size_t local_nz_with_halo = local_nz + 2;
    size_t local_size = nx * ny * local_nz_with_halo;

    // allocate local buffers (host)
    std::vector<double> cold_local(local_size, 0.0);
    std::vector<double> cnew_local(local_size, 0.0);
    std::vector<double> mu_local(local_size, 0.0);

    // initialize interior values
    initialize_local(cold_local, nx, ny, local_nz, z_offset);

    // set halo values using clamped boundaries (copy edge slices in absence of neighbor)
    auto fill_halos = [&](std::vector<double>& a) {
        // top halo at z=0 copies interior z=1 if rank==0 else will be filled by recv
        for (size_t y=0;y<ny;++y) for (size_t x=0;x<nx;++x) {
            a[idx3_loc(x,y,0,nx,ny)] = a[idx3_loc(x,y,1,nx,ny)];
            a[idx3_loc(x,y,local_nz_with_halo-1,nx,ny)] = a[idx3_loc(x,y,local_nz_with_halo-2,nx,ny)];
        }
    };
    fill_halos(cold_local);

    // Allocate device buffers
    double *d_cold=nullptr, *d_mu=nullptr, *d_cnew=nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&d_cold, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_mu, local_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_cnew, local_size * sizeof(double)));

    // Host buffers for halo exchange (one XY slice)
    size_t slice = nx * ny;
    std::vector<double> send_top(slice), send_bot(slice), recv_top(slice), recv_bot(slice);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int iter=0; iter<iterations; ++iter) {
        // Exchange cold halos with neighbors
        // pack top interior slice (z=1) to send_top, bottom interior slice (z=local_nz) to send_bot
        for (size_t i=0;i<slice;++i) { send_top[i] = cold_local[idx3_loc(i % nx, (i / nx)%ny, 1, nx, ny)]; }
        for (size_t i=0;i<slice;++i) { send_bot[i] = cold_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-2, nx, ny)]; }

        int prev = (rank==0) ? MPI_PROC_NULL : rank-1;
        int next = (rank==size-1) ? MPI_PROC_NULL : rank+1;
        MPI_Sendrecv(send_top.data(), slice, MPI_DOUBLE, prev, 0,
                     recv_bot.data(), slice, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot.data(), slice, MPI_DOUBLE, next, 1,
                     recv_top.data(), slice, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // unpack received halos or clamp if MPI_PROC_NULL
        if (prev == MPI_PROC_NULL) {
            // clamp top
            for (size_t i=0;i<slice;++i) cold_local[idx3_loc(i % nx, (i / nx)%ny, 0, nx, ny)] = cold_local[idx3_loc(i % nx, (i / nx)%ny, 1, nx, ny)];
        } else {
            for (size_t i=0;i<slice;++i) cold_local[idx3_loc(i % nx, (i / nx)%ny, 0, nx, ny)] = recv_top[i];
        }
        if (next == MPI_PROC_NULL) {
            // clamp bottom
            for (size_t i=0;i<slice;++i) cold_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-1, nx, ny)] = cold_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-2, nx, ny)];
        } else {
            for (size_t i=0;i<slice;++i) cold_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-1, nx, ny)] = recv_bot[i];
        }

        // Compute chemical potential on CPU (OpenMP parallel)
        computeChemicalPotential_local(cold_local, mu_local, nx, ny, local_nz_with_halo, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange mu halos for update
        for (size_t i=0;i<slice;++i) { send_top[i] = mu_local[idx3_loc(i % nx, (i / nx)%ny, 1, nx, ny)]; }
        for (size_t i=0;i<slice;++i) { send_bot[i] = mu_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-2, nx, ny)]; }
        MPI_Sendrecv(send_top.data(), slice, MPI_DOUBLE, prev, 2,
                     recv_bot.data(), slice, MPI_DOUBLE, next, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_bot.data(), slice, MPI_DOUBLE, next, 3,
                     recv_top.data(), slice, MPI_DOUBLE, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (prev == MPI_PROC_NULL) {
            for (size_t i=0;i<slice;++i) mu_local[idx3_loc(i % nx, (i / nx)%ny, 0, nx, ny)] = mu_local[idx3_loc(i % nx, (i / nx)%ny, 1, nx, ny)];
        } else {
            for (size_t i=0;i<slice;++i) mu_local[idx3_loc(i % nx, (i / nx)%ny, 0, nx, ny)] = recv_top[i];
        }
        if (next == MPI_PROC_NULL) {
            for (size_t i=0;i<slice;++i) mu_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-1, nx, ny)] = mu_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-2, nx, ny)];
        } else {
            for (size_t i=0;i<slice;++i) mu_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-1, nx, ny)] = recv_bot[i];
        }

        // Copy cold_local and mu_local to device
        CUDA_CHECK(cudaMemcpy(d_cold, cold_local.data(), local_size * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mu, mu_local.data(), local_size * sizeof(double), cudaMemcpyHostToDevice));

        // Launch kernel over interior region sized (nx,ny,local_nz)
        dim3 block(16,8,4);
        dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y, (local_nz + block.z -1)/block.z);
        update_kernel<<<grid, block>>>(d_cold, d_mu, d_cnew, nx, ny, local_nz_with_halo, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy device cnew back to host
        CUDA_CHECK(cudaMemcpy(cnew_local.data(), d_cnew, local_size * sizeof(double), cudaMemcpyDeviceToHost));

        // For halo slices keep previous values clamped
        for (size_t i=0;i<slice;++i) {
            cnew_local[idx3_loc(i % nx, (i / nx)%ny, 0, nx, ny)] = cnew_local[idx3_loc(i % nx, (i / nx)%ny, 1, nx, ny)];
            cnew_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-1, nx, ny)] = cnew_local[idx3_loc(i % nx, (i / nx)%ny, local_nz_with_halo-2, nx, ny)];
        }

        // swap cold_local and cnew_local
        std::swap(cold_local, cnew_local);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute global performance
    double global_cellupdates = (double)nx * (double)ny * (double)nz * (double)iterations;
    if (rank==0) {
        printf("Computation time (max over ranks): %.3f s\n", max_time);
        double mcups = global_cellupdates / max_time / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Optional print results: gather to rank 0
    if (printResults) {
        // gather counts and displacements
        std::vector<int> recvcounts(size), displs(size);
        for (int r=0;r<size;++r) {
            size_t rnz = base + (r < (int)rem ? 1 : 0);
            recvcounts[r] = (int)(rnz * nx * ny);
            displs[r] = (r==0)?0:displs[r-1]+recvcounts[r-1];
        }
        std::vector<double> gathered;
        if (rank==0) gathered.assign(nx*ny*nz,0.0);
        // pack local interior (skip halos) into temporary contiguous buffer
        std::vector<double> tmp(local_nz * nx * ny);
        for (size_t z=0;z<local_nz;++z) for (size_t i=0;i<slice;++i) tmp[z*slice + i] = cold_local[idx3_loc(i % nx, (i / nx)%ny, z+1, nx, ny)];
        MPI_Gatherv(tmp.data(), (int)tmp.size(), MPI_DOUBLE, gathered.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank==0) print_results(gathered, "Concentration");
    }

    // Validation
    if (validate) {
        bool ok = validate_global(MPI_COMM_WORLD, cold_local, nx, ny, local_nz_with_halo, local_nz);
        if (rank==0) printf("Validation: %s\n", ok?"PASSED":"FAILED");
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_mu)); CUDA_CHECK(cudaFree(d_cnew));
    MPI_Finalize();
    return 0;
}
