#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ======================== CUDA Kernels ========================

__global__ void cudaInitializeConcentration(double* c, size_t nx, size_t ny, size_t local_nz,
                                            size_t z_offset, size_t vol, size_t slice_size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t dev_idx = (z + 1) * slice_size + y * nx + x;
        size_t global_z = z + z_offset;
        size_t linear_id = global_z * (nx * ny) + y * nx + x;
        double pseudo = ((((linear_id + 1) * 1299709ULL) % vol) / (double)vol);
        c[dev_idx] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void cudaComputeChemicalPotential(
    const double* __restrict__ c, double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz, size_t z_offset, size_t global_nz, size_t slice_size,
    double dx, double dy, double dz,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t global_z = z + z_offset;
        size_t dev_z = z + 1;
        size_t idx = dev_z * slice_size + y * nx + x;
        double cv = c[idx];

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t zp_dev = (global_z < global_nz - 1) ? (dev_z + 1) : dev_z;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yn = (y > 0) ? y - 1 : 0;
        size_t zn_dev = (global_z > 0) ? (dev_z - 1) : dev_z;

        double c0 = c[idx];
        double cxp = c[dev_z * slice_size + y * nx + xp];
        double cxn = c[dev_z * slice_size + y * nx + xn];
        double cyp = c[dev_z * slice_size + yp * nx + x];
        double cyn = c[dev_z * slice_size + yn * nx + x];
        double czp = c[zp_dev * slice_size + y * nx + x];
        double czn = c[zn_dev * slice_size + y * nx + x];

        double cxx = (cxp + cxn - 2.0 * c0) / (dx * dx);
        double cyy = (cyp + cyn - 2.0 * c0) / (dy * dy);
        double czz = (czp + czn - 2.0 * c0) / (dz * dz);

        double lap = cxx + cyy + czz;

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * lap;
    }
}

__global__ void cudaCahnHilliardUpdate(
    double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
    size_t nx, size_t ny, size_t local_nz, size_t z_offset, size_t global_nz, size_t slice_size,
    double D, double dt, double dx, double dy, double dz)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < local_nz) {
        size_t global_z = z + z_offset;
        size_t dev_z = z + 1;
        size_t idx = dev_z * slice_size + y * nx + x;
        double mu0 = mu[idx];

        size_t xp = (x < nx - 1) ? x + 1 : x;
        size_t yp = (y < ny - 1) ? y + 1 : y;
        size_t zp_dev = (global_z < global_nz - 1) ? (dev_z + 1) : dev_z;
        size_t xn = (x > 0) ? x - 1 : 0;
        size_t yn = (y > 0) ? y - 1 : 0;
        size_t zn_dev = (global_z > 0) ? (dev_z - 1) : dev_z;

        double muxp = mu[dev_z * slice_size + y * nx + xp];
        double muxn = mu[dev_z * slice_size + y * nx + xn];
        double muyp = mu[dev_z * slice_size + yp * nx + x];
        double muyn = mu[dev_z * slice_size + yn * nx + x];
        double muzp = mu[zp_dev * slice_size + y * nx + x];
        double muzn = mu[zn_dev * slice_size + y * nx + x];

        double lap = (muxp + muxn - 2.0 * mu0) / (dx * dx)
                   + (muyp + muyn - 2.0 * mu0) / (dy * dy)
                   + (muzp + muzn - 2.0 * mu0) / (dz * dz);

        cnew[idx] = cold[idx] + dt * D * lap;
    }
}

__global__ void cudaCopySlice(const double* src, double* dst, size_t nx, size_t ny,
                               size_t src_dev_z, size_t dst_dev_z, size_t slice_size) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        size_t src_idx = src_dev_z * slice_size + y * nx + x;
        size_t dst_idx = dst_dev_z * slice_size + y * nx + x;
        dst[dst_idx] = src[src_idx];
    }
}

// ======================== Host helper functions ========================

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool found_invalid = false;
    #pragma omp parallel for reduction(||:found_invalid)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            found_invalid = true;
        }
    }
    if (found_invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }

    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        if (rank == 0) fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Finalize();
        return 1;
    }
    int device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
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
            }
        }
    }

    MPI_Bcast(&nx, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, num_threads, num_devices);
    }

    const double dx = 1.0, dy = 1.0, dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma_val = 0.5;
    const double D = 1.0;

    // ======================== MPI Domain Decomposition ========================
    std::vector<int> z_counts(nprocs);
    std::vector<int> z_displs(nprocs);
    size_t base_nz = nz / nprocs;
    size_t remainder = nz % nprocs;
    for (int p = 0; p < nprocs; ++p) {
        z_counts[p] = (int)base_nz + (p < (int)remainder ? 1 : 0);
    }
    z_displs[0] = 0;
    for (int p = 1; p < nprocs; ++p) {
        z_displs[p] = z_displs[p - 1] + z_counts[p - 1];
    }

    int local_nz = z_counts[rank];
    int z_offset = z_displs[rank];
    int left_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    size_t slice_size = nx * ny;
    size_t local_grid_size = slice_size * local_nz;
    size_t local_grid_with_halo = slice_size * (local_nz + 2);

    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, local_grid_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_grid_with_halo * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_grid_with_halo * sizeof(double)));

    // Host buffers for MPI communication
    std::vector<double> h_send_top(slice_size);
    std::vector<double> h_send_bot(slice_size);
    std::vector<double> h_recv_top(slice_size);
    std::vector<double> h_recv_bot(slice_size);

    std::vector<double> h_cold_full;
    if (rank == 0 && (printResults || validate)) {
        h_cold_full.resize(nx * ny * nz);
    }

    if (rank == 0) printf("Initializing concentration field...\n");

    {
        dim3 block(8, 8, 4);
        dim3 grid_dim(
            (nx + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (local_nz + block.z - 1) / block.z);
        size_t vol = nx * ny * nz;
        cudaInitializeConcentration<<<grid_dim, block>>>(d_cold, nx, ny, local_nz, z_offset, vol, slice_size);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    dim3 block(8, 8, 4);
    dim3 grid_dim(
        (nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y,
        (local_nz + block.z - 1) / block.z);

    dim3 block_2d(16, 16);
    dim3 grid_2d((slice_size + 255) / 256, 1);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // ---- Halo exchange for cold ----
        // Copy z=0 (owned, dev_z=1) to host send buffer
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(h_send_top.data(), d_cold + slice_size,
                slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        // Copy z=local_nz-1 (owned, dev_z=local_nz) to host send buffer
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(h_send_bot.data(), d_cold + local_nz * slice_size,
                slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // MPI exchange
        MPI_Request reqs[4];
        int nreqs = 0;

        if (left_rank != MPI_PROC_NULL) {
            MPI_Isend(h_send_top.data(), slice_size, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            MPI_Isend(h_send_bot.data(), slice_size, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_top.data(), slice_size, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (left_rank != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_bot.data(), slice_size, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Copy received halo data to device arrays
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_cold, h_recv_bot.data(),
                slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_cold + (local_nz + 1) * slice_size, h_recv_top.data(),
                slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        // ---- Compute chemical potential ----
        cudaComputeChemicalPotential<<<grid_dim, block>>>(
            d_cold, d_mu, nx, ny, local_nz, z_offset, nz, slice_size,
            dx, dy, dz, gamma_val, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // ---- Halo exchange for mu ----
        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(h_send_top.data(), d_mu + slice_size,
                slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(h_send_bot.data(), d_mu + local_nz * slice_size,
                slice_size * sizeof(double), cudaMemcpyDeviceToHost));
        }

        nreqs = 0;
        if (left_rank != MPI_PROC_NULL) {
            MPI_Isend(h_send_top.data(), slice_size, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            MPI_Isend(h_send_bot.data(), slice_size, MPI_DOUBLE, right_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (right_rank != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_top.data(), slice_size, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (left_rank != MPI_PROC_NULL) {
            MPI_Irecv(h_recv_bot.data(), slice_size, MPI_DOUBLE, left_rank, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        if (left_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_mu, h_recv_bot.data(),
                slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (right_rank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_mu + (local_nz + 1) * slice_size, h_recv_top.data(),
                slice_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        // ---- Cahn-Hilliard update ----
        cudaCahnHilliardUpdate<<<grid_dim, block>>>(
            d_cnew, d_cold, d_mu, nx, ny, local_nz, z_offset, nz, slice_size,
            D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        // ---- Swap buffers ----
        double* tmp = d_cold;
        d_cold = d_cnew;
        d_cnew = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_ms);
        size_t gridSize = nx * ny * nz;
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults || validate) {
        std::vector<double> h_local(local_grid_size);
        CUDA_CHECK(cudaMemcpy(h_local.data(),
            d_cold + slice_size,
            local_grid_size * sizeof(double),
            cudaMemcpyDeviceToHost));

        if (rank == 0) {
            std::copy(h_local.begin(), h_local.end(), h_cold_full.begin());
            for (int p = 1; p < nprocs; ++p) {
                int p_nz = z_counts[p];
                int p_offset = z_displs[p];
                size_t p_size = nx * ny * p_nz;
                MPI_Recv(h_cold_full.data() + p_offset * slice_size,
                         p_size, MPI_DOUBLE, p, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(h_local.data(), local_grid_size, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD);
        }
    }

    if (printResults && rank == 0) {
        print_results(h_cold_full, "Concentration");
    }

    if (validate) {
        bool valid = false;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(h_cold_full, nx, ny, nz);
        }
        MPI_Bcast(&valid, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            CUDA_CHECK(cudaFree(d_cold));
            CUDA_CHECK(cudaFree(d_cnew));
            CUDA_CHECK(cudaFree(d_mu));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return 0;
}
