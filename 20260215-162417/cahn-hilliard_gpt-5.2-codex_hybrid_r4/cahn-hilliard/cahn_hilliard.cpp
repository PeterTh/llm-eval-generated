#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation (supports host and device code)
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void compute_mu_kernel(const double* __restrict__ c, double* __restrict__ mu,
                                  const size_t nx, const size_t ny, const size_t local_nz,
                                  const double dx, const double dy, const double dz,
                                  const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) {
        return;
    }

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t plane = nx * ny;
    const size_t idx = idx3(x, y, z, nx, ny);

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                        2.0 * c[idx]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                        2.0 * c[idx]) / (dy * dy);
    const double czz = (c[idx + plane] + c[idx - plane] - 2.0 * c[idx]) / (dz * dz);

    const double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * (cxx + cyy + czz);
}

__global__ void update_kernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu, const size_t nx, const size_t ny,
                              const size_t local_nz, const double D, const double dt,
                              const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;
    if (x >= nx || y >= ny || z > local_nz) {
        return;
    }

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t plane = nx * ny;
    const size_t idx = idx3(x, y, z, nx, ny);

    const double mxx = (mu[idx3(xp, y, z, nx, ny)] + mu[idx3(xn, y, z, nx, ny)] -
                        2.0 * mu[idx]) / (dx * dx);
    const double myy = (mu[idx3(x, yp, z, nx, ny)] + mu[idx3(x, yn, z, nx, ny)] -
                        2.0 * mu[idx]) / (dy * dy);
    const double mzz = (mu[idx + plane] + mu[idx - plane] - 2.0 * mu[idx]) / (dz * dz);

    cnew[idx] = cold[idx] + dt * D * (mxx + myy + mzz);
}

void exchangeHalos(double* d_field, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size,
                   std::vector<double>& send_low, std::vector<double>& send_high,
                   std::vector<double>& recv_low, std::vector<double>& recv_high) {
    const size_t plane_size = nx * ny;
    const size_t plane_bytes = plane_size * sizeof(double);

    if (size == 1) {
        checkCuda(cudaMemcpy(d_field, d_field + plane_size, plane_bytes, cudaMemcpyDeviceToDevice), "clamp low halo");
        checkCuda(cudaMemcpy(d_field + (local_nz + 1) * plane_size, d_field + local_nz * plane_size,
                             plane_bytes, cudaMemcpyDeviceToDevice), "clamp high halo");
        return;
    }

    if (rank > 0) {
        checkCuda(cudaMemcpy(send_low.data(), d_field + plane_size, plane_bytes, cudaMemcpyDeviceToHost), "copy send low");
    }
    if (rank < size - 1) {
        checkCuda(cudaMemcpy(send_high.data(), d_field + local_nz * plane_size, plane_bytes, cudaMemcpyDeviceToHost), "copy send high");
    }

    if (rank > 0) {
        MPI_Sendrecv(send_low.data(), static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 0,
                     recv_low.data(), static_cast<int>(plane_size), MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Sendrecv(send_high.data(), static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 1,
                     recv_high.data(), static_cast<int>(plane_size), MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank > 0) {
        checkCuda(cudaMemcpy(d_field, recv_low.data(), plane_bytes, cudaMemcpyHostToDevice), "copy recv low");
    } else {
        checkCuda(cudaMemcpy(d_field, d_field + plane_size, plane_bytes, cudaMemcpyDeviceToDevice), "clamp low halo");
    }
    if (rank < size - 1) {
        checkCuda(cudaMemcpy(d_field + (local_nz + 1) * plane_size, recv_high.data(), plane_bytes, cudaMemcpyHostToDevice), "copy recv high");
    } else {
        checkCuda(cudaMemcpy(d_field + (local_nz + 1) * plane_size, d_field + local_nz * plane_size,
                             plane_bytes, cudaMemcpyDeviceToDevice), "clamp high halo");
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t local_nz, const size_t z_start) {
    const size_t plane = nx * ny;
    const size_t vol = nx * ny * nz;
    std::fill(c.begin(), c.end(), 0.0);

    #pragma omp parallel for schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = z_start + z;
                const size_t lz = z + 1;
                const size_t idx = idx3(x, y, lz, nx, ny);
                const size_t linear_id = gz * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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
    bool showHelp = false;
    bool parseOk = true;
    char unknownOpt[128] = {};

    if (rank == 0) {
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
                showHelp = true;
            } else {
                parseOk = false;
                std::snprintf(unknownOpt, sizeof(unknownOpt), "%s", argv[i]);
                break;
            }
        }

        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }

    int parseOkInt = parseOk ? 1 : 0;
    int showHelpInt = showHelp ? 1 : 0;
    MPI_Bcast(&parseOkInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelpInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOkInt || showHelpInt) {
        if (rank == 0) {
            if (!parseOkInt) {
                printf("Unknown option: %s\n", unknownOpt);
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseOkInt ? 0 : 1;
    }

    uint64_t nx_u = static_cast<uint64_t>(nx);
    uint64_t ny_u = static_cast<uint64_t>(ny);
    uint64_t nz_u = static_cast<uint64_t>(nz);
    MPI_Bcast(&nx_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz_u, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    nx = static_cast<size_t>(nx_u);
    ny = static_cast<size_t>(ny_u);
    nz = static_cast<size_t>(nz_u);
    validate = (validateInt != 0);
    printResults = (printResultsInt != 0);

    int configOk = 1;
    if (rank == 0 && size > static_cast<int>(nz)) {
        printf("Error: MPI size (%d) exceeds global Z dimension (%zu)\n", size, nz);
        configOk = 0;
    }
    MPI_Bcast(&configOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!configOk) {
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
    if (deviceErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA device not available\n");
        }
        MPI_Finalize();
        return 1;
    }
    checkCuda(cudaSetDevice(rank % deviceCount), "set device");

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices: %d\n", size, deviceCount);
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

    const size_t plane_size = nx * ny;
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);

    const size_t local_with_halo = local_nz + 2;
    const size_t local_total = local_with_halo * plane_size;

    std::vector<double> h_cold(local_total);
    std::vector<double> send_low(plane_size);
    std::vector<double> send_high(plane_size);
    std::vector<double> recv_low(plane_size);
    std::vector<double> recv_high(plane_size);

    printf("Rank %d initializing concentration field...\n", rank);
    initializeConcentration(h_cold, nx, ny, nz, local_nz, z_start);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    checkCuda(cudaMalloc(&d_cold, local_total * sizeof(double)), "alloc cold");
    checkCuda(cudaMalloc(&d_cnew, local_total * sizeof(double)), "alloc cnew");
    checkCuda(cudaMalloc(&d_mu, local_total * sizeof(double)), "alloc mu");
    checkCuda(cudaMemcpy(d_cold, h_cold.data(), local_total * sizeof(double), cudaMemcpyHostToDevice), "copy cold");

    dim3 block(8, 8, 4);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (local_nz + block.z - 1) / block.z);

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, nx, ny, local_nz, rank, size, send_low, send_high, recv_low, recv_high);
        compute_mu_kernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        checkCuda(cudaGetLastError(), "compute mu kernel");

        exchangeHalos(d_mu, nx, ny, local_nz, rank, size, send_low, send_high, recv_low, recv_high);
        update_kernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        checkCuda(cudaGetLastError(), "update kernel");

        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "sync");
    const double end = MPI_Wtime();
    double elapsed = end - start;
    double max_elapsed = 0.0;
    MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        const double cellUpdates = static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(nz) * iterations;
        const double mcups = cellUpdates / max_elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> local_final;
    if (printResults || validate) {
        local_final.resize(local_nz * plane_size);
        checkCuda(cudaMemcpy(local_final.data(), d_cold + plane_size, local_nz * plane_size * sizeof(double),
                             cudaMemcpyDeviceToHost), "copy final");
    }

    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> global_field;
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t r_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
                counts[r] = static_cast<int>(r_nz * plane_size);
                displs[r] = static_cast<int>(offset);
                offset += r_nz * plane_size;
            }
            global_field.resize(nx * ny * nz);
        }

        MPI_Gatherv(local_final.data(), static_cast<int>(local_final.size()), MPI_DOUBLE,
                    rank == 0 ? global_field.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(global_field, "Concentration");
        }
    }

    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        double local_min = local_final.empty() ? 0.0 : local_final[0];
        double local_max = local_final.empty() ? 0.0 : local_final[0];
        int local_invalid = 0;

        #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) reduction(|:local_invalid)
        for (size_t i = 0; i < local_final.size(); ++i) {
            const double val = local_final[i];
            if (std::isnan(val) || std::isinf(val)) {
                local_invalid = 1;
            }
            local_min = std::min(local_min, val);
            local_max = std::max(local_max, val);
        }

        double global_min = 0.0;
        double global_max = 0.0;
        int global_invalid = 0;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

        const int range_ok = (global_max <= 10.0 && global_min >= -10.0) ? 1 : 0;
        const int valid_all = (global_invalid == 0 && range_ok == 1) ? 1 : 0;

        if (rank == 0) {
            if (global_invalid) {
                printf("Validation failed: found NaN or Inf value\n");
            }

            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
            if (!range_ok) {
                printf("Validation failed: values out of expected range\n");
            }
            printf("Validation: %s\n", valid_all ? "PASSED" : "FAILED");
        }

        if (!valid_all) {
            checkCuda(cudaFree(d_cold), "free cold");
            checkCuda(cudaFree(d_cnew), "free cnew");
            checkCuda(cudaFree(d_mu), "free mu");
            MPI_Finalize();
            return 1;
        }
    }

    checkCuda(cudaFree(d_cold), "free cold");
    checkCuda(cudaFree(d_cnew), "free cnew");
    checkCuda(cudaFree(d_mu), "free mu");

    MPI_Finalize();
    return 0;
}
