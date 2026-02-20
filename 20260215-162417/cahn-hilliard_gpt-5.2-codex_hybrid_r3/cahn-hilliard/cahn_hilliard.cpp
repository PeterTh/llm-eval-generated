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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

constexpr int kTagDown = 100;
constexpr int kTagUp = 101;

__device__ inline double laplacianDevice(const double* field, const size_t nx, const size_t ny,
                                         const size_t x, const size_t y, const size_t z,
                                         const size_t plane, const double inv_dx2,
                                         const double inv_dy2, const double inv_dz2) {
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t idx = z * plane + y * nx + x;
    const double center = field[idx];
    const double cxx = (field[z * plane + y * nx + xp] + field[z * plane + y * nx + xn] - 2.0 * center) * inv_dx2;
    const double cyy = (field[z * plane + yp * nx + x] + field[z * plane + yn * nx + x] - 2.0 * center) * inv_dy2;
    const double czz = (field[(z + 1) * plane + y * nx + x] + field[(z - 1) * plane + y * nx + x] - 2.0 * center) * inv_dz2;

    return cxx + cyy + czz;
}

__global__ void computeChemicalPotentialKernel(const double* c, double* mu, const size_t nx, const size_t ny,
                                               const size_t local_nz, const double inv_dx2, const double inv_dy2,
                                               const double inv_dz2, const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = plane * local_nz;
    if (idx >= total) {
        return;
    }

    const size_t z = idx / plane + 1;
    const size_t rem = idx - (z - 1) * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;

    const size_t offset = z * plane + rem;
    const double cv = c[offset];
    const double lap = laplacianDevice(c, nx, ny, x, y, z, plane, inv_dx2, inv_dy2, inv_dz2);

    mu[offset] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * lap;
}

__global__ void cahnHilliardUpdateKernel(double* cnew, const double* cold, const double* mu,
                                        const size_t nx, const size_t ny, const size_t local_nz,
                                        const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                        const double D, const double dt) {
    const size_t plane = nx * ny;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = plane * local_nz;
    if (idx >= total) {
        return;
    }

    const size_t z = idx / plane + 1;
    const size_t rem = idx - (z - 1) * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;

    const size_t offset = z * plane + rem;
    const double lap_mu = laplacianDevice(mu, nx, ny, x, y, z, plane, inv_dx2, inv_dy2, inv_dz2);

    cnew[offset] = cold[offset] + dt * D * lap_mu;
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z_offset) {
    const size_t vol = nx * ny * nz;
    const size_t plane = nx * ny;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < c.size() / plane; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_offset + z;
                const size_t linear_id = global_z * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx3(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeHalos(double* d_field, const size_t nx, const size_t ny, const size_t local_nz,
                   const int rank, const int size, const MPI_Comm comm,
                   std::vector<double>& send_lower, std::vector<double>& send_upper,
                   std::vector<double>& recv_lower, std::vector<double>& recv_upper) {
    const size_t plane = nx * ny;

    if (rank > 0) {
        CUDA_CHECK(cudaMemcpy(send_lower.data(), d_field + plane, plane * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (rank < size - 1) {
        CUDA_CHECK(cudaMemcpy(send_upper.data(), d_field + plane * local_nz, plane * sizeof(double), cudaMemcpyDeviceToHost));
    }

    if (rank > 0) {
        MPI_Sendrecv(send_lower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, kTagDown,
                     recv_lower.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, kTagUp,
                     comm, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_field, recv_lower.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field, d_field + plane, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    if (rank < size - 1) {
        MPI_Sendrecv(send_upper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, kTagUp,
                     recv_upper.data(), static_cast<int>(plane), MPI_DOUBLE, rank + 1, kTagDown,
                     comm, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_field + plane * (local_nz + 1), recv_upper.data(), plane * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_field + plane * (local_nz + 1), d_field + plane * local_nz, plane * sizeof(double), cudaMemcpyDeviceToDevice));
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
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

    if (size > static_cast<int>(nz)) {
        if (rank == 0) {
            printf("MPI size (%d) must not exceed nz (%zu).\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
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

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;

    // Domain decomposition along Z
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_offset = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    const size_t local_cells = local_nz * plane;

    if (local_cells > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Local problem size too large for MPI counts.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold_local(local_cells);
    initializeConcentration(cold_local, nx, ny, nz, z_offset);

    // Select CUDA device per local MPI rank
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        if (rank == 0) {
            printf("No CUDA devices available.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    const size_t halo_nz = local_nz + 2;
    const size_t device_cells = halo_nz * plane;
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    CUDA_CHECK(cudaMalloc(&d_cold, device_cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, device_cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, device_cells * sizeof(double)));

    CUDA_CHECK(cudaMemset(d_cnew, 0, device_cells * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, device_cells * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cold + plane, cold_local.data(), local_cells * sizeof(double), cudaMemcpyHostToDevice));

    std::vector<double> send_lower(plane);
    std::vector<double> send_upper(plane);
    std::vector<double> recv_lower(plane);
    std::vector<double> recv_upper(plane);

    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const size_t threads = 256;
    const size_t blocks = (local_cells + threads - 1) / threads;

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, nx, ny, local_nz, rank, size, MPI_COMM_WORLD,
                      send_lower, send_upper, recv_lower, recv_upper);

        computeChemicalPotentialKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads)>>>(
            d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        exchangeHalos(d_mu, nx, ny, local_nz, rank, size, MPI_COMM_WORLD,
                      send_lower, send_upper, recv_lower, recv_upper);

        cahnHilliardUpdateKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads)>>>(
            d_cnew, d_cold, d_mu, nx, ny, local_nz, inv_dx2, inv_dy2, inv_dz2, D, dt);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpy(cold_local.data(), d_cold + plane, local_cells * sizeof(double), cudaMemcpyDeviceToHost));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_time * 1000.0));
        if (max_time > 0.0) {
            const double cellUpdates = static_cast<double>(gridSize) * iterations;
            const double mcups = cellUpdates / max_time / 1e6;
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        } else {
            printf("Performance: 0.000 MCellUpdates/s\n");
        }
    }

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> cold_global;

    if (printResults || validate) {
        if (rank == 0) {
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t r_local_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
                const size_t r_cells = r_local_nz * plane;
                counts[r] = static_cast<int>(r_cells);
                displs[r] = static_cast<int>(offset);
                offset += r_cells;
            }
            cold_global.resize(gridSize);
        }

        MPI_Gatherv(cold_local.data(), static_cast<int>(local_cells), MPI_DOUBLE,
                    rank == 0 ? cold_global.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        print_results(cold_global, "Concentration");
    }

    int valid_int = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(cold_global, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
                valid_int = 1;
            } else {
                printf("Validation: FAILED\n");
                valid_int = 0;
            }
        }
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return validate ? (valid_int ? 0 : 1) : 0;
}
