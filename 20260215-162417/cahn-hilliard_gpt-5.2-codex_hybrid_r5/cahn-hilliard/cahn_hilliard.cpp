#include <algorithm>
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

// 3D index calculation
inline constexpr __host__ __device__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void compute_mu_kernel(const double* c, double* mu,
                                  const size_t nx, const size_t ny, const size_t local_nz,
                                  const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                  const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z_local = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z_local >= local_nz) {
        return;
    }

    const size_t z = z_local + 1;
    const size_t slice = nx * ny;
    const size_t idx = z * slice + y * nx + x;

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double center = c[idx];
    const double lap = (c[z * slice + y * nx + xp] + c[z * slice + y * nx + xn] - 2.0 * center) * inv_dx2
                     + (c[z * slice + yp * nx + x] + c[z * slice + yn * nx + x] - 2.0 * center) * inv_dy2
                     + (c[zp * slice + y * nx + x] + c[zn * slice + y * nx + x] - 2.0 * center) * inv_dz2;

    mu[idx] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
            + 3.0 * center + center * center * center
            - gamma * lap;
}

__global__ void update_kernel(double* cnew, const double* cold, const double* mu,
                              const size_t nx, const size_t ny, const size_t local_nz,
                              const double inv_dx2, const double inv_dy2, const double inv_dz2,
                              const double coeff) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z_local = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z_local >= local_nz) {
        return;
    }

    const size_t z = z_local + 1;
    const size_t slice = nx * ny;
    const size_t idx = z * slice + y * nx + x;

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : x;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : y;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double center = mu[idx];
    const double lap = (mu[z * slice + y * nx + xp] + mu[z * slice + y * nx + xn] - 2.0 * center) * inv_dx2
                     + (mu[z * slice + yp * nx + x] + mu[z * slice + yn * nx + x] - 2.0 * center) * inv_dy2
                     + (mu[zp * slice + y * nx + x] + mu[zn * slice + y * nx + x] - 2.0 * center) * inv_dz2;

    cnew[idx] = cold[idx] + coeff * lap;
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t local_nz, const size_t z_start, const size_t nz_global) {
    const size_t vol = nx * ny * nz_global;
    const size_t slice = nx * ny;

    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + z;
                const size_t idx = z * slice + y * nx + x;
                const size_t linear_id = global_z * slice + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeHalos(double* d_field,
                   double* h_send_low, double* h_send_high,
                   double* h_recv_low, double* h_recv_high,
                   const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size, MPI_Comm comm) {
    const size_t slice = nx * ny;
    const size_t bytes = slice * sizeof(double);
    const int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int next = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    const int slice_count = static_cast<int>(slice);

    if (prev != MPI_PROC_NULL) {
        cudaCheck(cudaMemcpy(h_send_low, d_field + slice, bytes, cudaMemcpyDeviceToHost), "halo D2H low");
    }
    if (next != MPI_PROC_NULL) {
        cudaCheck(cudaMemcpy(h_send_high, d_field + local_nz * slice, bytes, cudaMemcpyDeviceToHost), "halo D2H high");
    }

    MPI_Sendrecv(h_send_low, slice_count, MPI_DOUBLE, prev, 0,
                 h_recv_high, slice_count, MPI_DOUBLE, next, 0,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(h_send_high, slice_count, MPI_DOUBLE, next, 1,
                 h_recv_low, slice_count, MPI_DOUBLE, prev, 1,
                 comm, MPI_STATUS_IGNORE);

    if (next != MPI_PROC_NULL) {
        cudaCheck(cudaMemcpy(d_field + (local_nz + 1) * slice, h_recv_high, bytes, cudaMemcpyHostToDevice), "halo H2D high");
    } else {
        cudaCheck(cudaMemcpy(d_field + (local_nz + 1) * slice, d_field + local_nz * slice, bytes, cudaMemcpyDeviceToDevice), "halo clamp high");
    }

    if (prev != MPI_PROC_NULL) {
        cudaCheck(cudaMemcpy(d_field, h_recv_low, bytes, cudaMemcpyHostToDevice), "halo H2D low");
    } else {
        cudaCheck(cudaMemcpy(d_field, d_field + slice, bytes, cudaMemcpyDeviceToDevice), "halo clamp low");
    }
}

bool validateResult(const std::vector<double>& c, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    double local_min = std::numeric_limits<double>::infinity();
    double local_max = -std::numeric_limits<double>::infinity();
    int local_bad = 0;

    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(|:local_bad)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            local_bad = 1;
        }
        local_min = std::min(local_min, val);
        local_max = std::max(local_max, val);
    }

    double global_min = 0.0;
    double global_max = 0.0;
    int global_bad = 0;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_bad, &global_bad, 1, MPI_INT, MPI_LOR, comm);

    if (rank == 0) {
        printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
    }

    if (global_bad) {
        if (rank == 0) {
            printf("Validation failed: found NaN or Inf value\n");
        }
        return false;
    }

    if (global_max > 10.0 || global_min < -10.0) {
        if (rank == 0) {
            printf("Validation failed: values out of expected range\n");
        }
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
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", size);
    }

    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + ((static_cast<size_t>(rank) < rem) ? 1 : 0);
    const size_t z_start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    if (local_nz == 0) {
        if (rank == 0) {
            printf("Error: MPI size exceeds nz; reduce ranks or increase nz.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t slice = nx * ny;
    const size_t local_size = local_nz * slice;
    const size_t total_size = (local_nz + 2) * slice;

    std::vector<double> cold_host(local_size);
    initializeConcentration(cold_host, nx, ny, local_nz, z_start, nz);

    int device_count = 0;
    cudaCheck(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        if (rank == 0) {
            printf("No CUDA devices found.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(rank % device_count), "cudaSetDevice");

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    cudaCheck(cudaMalloc(&d_cold, total_size * sizeof(double)), "cudaMalloc d_cold");
    cudaCheck(cudaMalloc(&d_cnew, total_size * sizeof(double)), "cudaMalloc d_cnew");
    cudaCheck(cudaMalloc(&d_mu, total_size * sizeof(double)), "cudaMalloc d_mu");
    cudaCheck(cudaMemcpy(d_cold + slice, cold_host.data(), local_size * sizeof(double), cudaMemcpyHostToDevice),
              "cudaMemcpy cold to device");

    std::vector<double> send_low(slice);
    std::vector<double> send_high(slice);
    std::vector<double> recv_low(slice);
    std::vector<double> recv_high(slice);

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double coeff = dt * D;

    const dim3 block(8, 8, 8);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, send_low.data(), send_high.data(), recv_low.data(), recv_high.data(),
                      nx, ny, local_nz, rank, size, MPI_COMM_WORLD);

        compute_mu_kernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        cudaCheck(cudaGetLastError(), "compute_mu_kernel launch");

        exchangeHalos(d_mu, send_low.data(), send_high.data(), recv_low.data(), recv_high.data(),
                      nx, ny, local_nz, rank, size, MPI_COMM_WORLD);

        update_kernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, coeff);
        cudaCheck(cudaGetLastError(), "update_kernel launch");

        std::swap(d_cold, d_cnew);
    }

    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));
        const double cell_updates = static_cast<double>(nx * ny * nz) * iterations;
        const double mcups = cell_updates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    cudaCheck(cudaMemcpy(cold_host.data(), d_cold + slice, local_size * sizeof(double), cudaMemcpyDeviceToHost),
              "cudaMemcpy result to host");

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> global;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = base + ((static_cast<size_t>(r) < rem) ? 1 : 0);
                const size_t r_count = r_local_nz * slice;
                counts[r] = static_cast<int>(r_count);
                displs[r] = static_cast<int>(offset);
                offset += r_count;
            }
            global.resize(nx * ny * nz);
        }
        MPI_Gatherv(cold_host.data(), static_cast<int>(local_size), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    int exit_code = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(cold_host, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exit_code = valid ? 0 : 1;
    }

    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);
    MPI_Finalize();
    return exit_code;
}
