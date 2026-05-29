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

// ============================================================================
// CUDA kernels
// ============================================================================

__device__ inline size_t idx3d(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__device__ inline double laplacian(const double* c, size_t nx, size_t ny, size_t nz,
                                    double dx, double dy, double dz,
                                    size_t x, size_t y, size_t z) {
    size_t xp = (x < nx - 1) ? x + 1 : x;
    size_t yp = (y < ny - 1) ? y + 1 : y;
    size_t zp = (z < nz - 1) ? z + 1 : z;
    size_t xn = (x > 0) ? x - 1 : 0;
    size_t yn = (y > 0) ? y - 1 : 0;
    size_t zn = (z > 0) ? z - 1 : 0;

    double cxx = (c[idx3d(xp, y, z, nx, ny)] + c[idx3d(xn, y, z, nx, ny)] -
                  2.0 * c[idx3d(x, y, z, nx, ny)]) / (dx * dx);
    double cyy = (c[idx3d(x, yp, z, nx, ny)] + c[idx3d(x, yn, z, nx, ny)] -
                  2.0 * c[idx3d(x, y, z, nx, ny)]) / (dy * dy);
    double czz = (c[idx3d(x, y, zp, nx, ny)] + c[idx3d(x, y, zn, nx, ny)] -
                  2.0 * c[idx3d(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void chemPotentialKernel(const double* c, double* mu,
                                    size_t nx, size_t ny, size_t nz,
                                    double dx, double dy, double dz,
                                    double gamma, double e_AA, double e_BB, double e_AB) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = idx3d(x, y, z, nx, ny);
    double cv = c[idx];
    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * laplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void updateKernel(const double* cold, const double* mu, double* cnew,
                              size_t nx, size_t ny, size_t nz,
                              double D, double dt, double dx, double dy, double dz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = idx3d(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
                laplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
}

__global__ void initKernel(double* c, size_t nx, size_t ny, size_t nz, size_t vol) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t linear_id = z * (nx * ny) + y * nx + x;
    double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[idx3d(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
}

// ============================================================================
// Host helpers
// ============================================================================

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny,
                         const size_t nz, const double dx, const double dy, const double dz,
                         const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                        2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

// ============================================================================
// Domain decomposition (Z-axis, 1 halo layer)
// ============================================================================

struct SubDomain {
    size_t nx, ny, nz;          // global
    int rank, nrank;            // MPI info
    size_t local_z_start;       // first owned z (inclusive of halo)
    size_t local_z_end;         // last owned z (exclusive, inclusive of halo)
    size_t local_owned_start;   // first owned z (no halo)
    size_t local_owned_end;     // last owned z (exclusive, no halo)
    size_t local_size;          // (local_z_end - local_z_start) * ny * nx
    size_t owned_size;          // (local_owned_end - local_owned_start) * ny * nx
    int rank_above, rank_below; // MPI neighbours (-1 if none)

    void compute(size_t gnx, size_t gny, size_t gnz, int r, int nr) {
        nx = gnx; ny = gny; nz = gnz;
        rank = r; nrank = nr;

        size_t base_z = gnz / nr;
        size_t remainder = gnz % nr;
        size_t start = 0;
        for (int i = 0; i < r; ++i) {
            start += base_z + (i < (int)remainder ? 1 : 0);
        }
        size_t end = start + base_z + (r < (int)remainder ? 1 : 0);

        // Owned region (no halo)
        local_owned_start = start;
        local_owned_end = end;

        // With halo
        local_z_start = (start > 0) ? start - 1 : start;
        local_z_end = (end < gnz) ? end + 1 : end;

        local_size = (local_z_end - local_z_start) * ny * nx;
        owned_size = (local_owned_end - local_owned_start) * ny * nx;

        rank_below = (r > 0) ? r - 1 : -1;
        rank_above = (r < nr - 1) ? r + 1 : -1;
    }

    // Map (x,y,z) -> local index; z must be within [local_z_start, local_z_end)
    inline size_t local_idx(size_t x, size_t y, size_t z) const {
        return (z - local_z_start) * ny * nx + y * nx + x;
    }
};

// ============================================================================
// Main
// ============================================================================

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

    int rank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse)
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
        printf("MPI ranks: %d\n", nrank);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
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
    const size_t gridSize = nx * ny * nz;

    // Domain decomposition
    SubDomain dom;
    dom.compute(nx, ny, nz, rank, nrank);

    // Local buffers (with halo)
    std::vector<double> cold(dom.local_size);
    std::vector<double> cnew(dom.local_size);
    std::vector<double> mu(dom.local_size);

    // Initialize concentration field (with halo)
    printf("Rank %d: Initializing concentration field (local_z %zu..%zu)...\n",
           rank, dom.local_z_start, dom.local_z_end);
    for (size_t z = dom.local_z_start; z < dom.local_z_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t linear_id = z * (nx * ny) + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % gridSize) /
                                 static_cast<double>(gridSize));
                cold[dom.local_idx(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    // CUDA: allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    cudaMalloc(&d_cold, dom.local_size * sizeof(double));
    cudaMalloc(&d_cnew, dom.local_size * sizeof(double));
    cudaMalloc(&d_mu, dom.local_size * sizeof(double));

    // Copy initial data to device
    cudaMemcpy(d_cold, cold.data(), dom.local_size * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA kernel launch config
    dim3 blockSize(8, 8, 4);
    dim3 gridSize_cuda((nx + blockSize.x - 1) / blockSize.x,
                        (ny + blockSize.y - 1) / blockSize.y,
                        (dom.local_z_end - dom.local_z_start + blockSize.z - 1) / blockSize.z);

    // MPI request objects for halo exchange
    MPI_Request req_send_top, req_recv_top, req_send_bot, req_recv_bot;

    // Temporary halo buffers
    size_t halo_size = nx * ny;
    std::vector<double> send_top(halo_size), recv_top(halo_size);
    std::vector<double> send_bot(halo_size), recv_bot(halo_size);

    printf("Rank %d: Running Cahn-Hilliard simulation...\n", rank);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // ---- Halo exchange for cold ----
        // Pack top halo (z = local_owned_end - 1) -> send to rank_above
        // Pack bottom halo (z = local_owned_start) -> send to rank_below
        bool has_top = (dom.rank_above >= 0);
        bool has_bot = (dom.rank_below >= 0);

        if (has_top) {
            size_t z_top = dom.local_owned_end - 1;
            for (size_t j = 0; j < ny * nx; ++j) {
                send_top[j] = cold[dom.local_idx(j % nx, j / nx, z_top)];
            }
        }
        if (has_bot) {
            size_t z_bot = dom.local_owned_start;
            for (size_t j = 0; j < ny * nx; ++j) {
                send_bot[j] = cold[dom.local_idx(j % nx, j / nx, z_bot)];
            }
        }

        // Non-blocking send/recv (top)
        if (has_top) {
            MPI_Isend(send_top.data(), halo_size, MPI_DOUBLE, dom.rank_above, 0,
                      MPI_COMM_WORLD, &req_send_top);
            MPI_Irecv(recv_top.data(), halo_size, MPI_DOUBLE, dom.rank_above, 0,
                      MPI_COMM_WORLD, &req_recv_top);
        }
        // Non-blocking send/recv (bottom)
        if (has_bot) {
            MPI_Isend(send_bot.data(), halo_size, MPI_DOUBLE, dom.rank_below, 0,
                      MPI_COMM_WORLD, &req_send_bot);
            MPI_Irecv(recv_bot.data(), halo_size, MPI_DOUBLE, dom.rank_below, 0,
                      MPI_COMM_WORLD, &req_recv_bot);
        }

        // Wait for receives
        if (has_top) {
            MPI_Wait(&req_recv_top, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send_top, MPI_STATUS_IGNORE);
            // Unpack into halo layer at z = local_owned_end
            size_t z_recv = dom.local_owned_end;
            for (size_t j = 0; j < ny * nx; ++j) {
                cold[dom.local_idx(j % nx, j / nx, z_recv)] = recv_top[j];
            }
        }
        if (has_bot) {
            MPI_Wait(&req_recv_bot, MPI_STATUS_IGNORE);
            MPI_Wait(&req_send_bot, MPI_STATUS_IGNORE);
            // Unpack into halo layer at z = local_z_start
            size_t z_recv = dom.local_z_start;
            for (size_t j = 0; j < ny * nx; ++j) {
                cold[dom.local_idx(j % nx, j / nx, z_recv)] = recv_bot[j];
            }
        }

        // Copy cold to device
        cudaMemcpy(d_cold, cold.data(), dom.local_size * sizeof(double),
                   cudaMemcpyHostToDevice);

        // CUDA: compute chemical potential on local domain (with halo)
        chemPotentialKernel<<<gridSize_cuda, blockSize>>>(
            d_cold, d_mu,
            nx, ny, dom.local_z_end - dom.local_z_start,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // CUDA: update concentration
        updateKernel<<<gridSize_cuda, blockSize>>>(
            d_cold, d_mu, d_cnew,
            nx, ny, dom.local_z_end - dom.local_z_start,
            D, dt, dx, dy, dz);

        // Copy cnew back from device
        cudaMemcpy(cnew.data(), d_cnew, dom.local_size * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Rank %d: Computation time: %ld ms\n", rank, duration.count());

    // Calculate performance
    double cellUpdates = (double)dom.owned_size * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Rank %d: Performance: %.3f MCellUpdates/s\n", rank, mcups);

    // Gather results on rank 0 for validation/printing
    std::vector<double> global_cold;
    if (rank == 0) {
        global_cold.resize(gridSize);
    }

    // Gather cold from all ranks (owned region only)
    std::vector<int> owned_sizes(nrank);
    owned_sizes[rank] = (int)dom.owned_size;
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  owned_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Compute displacements
    std::vector<int> displs(nrank, 0);
    for (int i = 1; i < nrank; ++i) {
        displs[i] = displs[i - 1] + owned_sizes[i - 1];
    }

    // Extract owned region from local buffer
    std::vector<double> owned_cold(dom.owned_size);
    for (size_t z = dom.local_owned_start; z < dom.local_owned_end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t local_i = (z - dom.local_owned_start) * ny * nx + y * nx + x;
                owned_cold[local_i] = cold[dom.local_idx(x, y, z)];
            }
        }
    }

    MPI_Gatherv(owned_cold.data(), (int)dom.owned_size, MPI_DOUBLE,
                rank == 0 ? global_cold.data() : nullptr,
                owned_sizes.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results on rank 0
    if (rank == 0 && printResults) {
        print_results(global_cold, "Concentration");
    }

    // Validation on rank 0
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = true;
        double minVal = global_cold[0];
        double maxVal = global_cold[0];

#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
        for (size_t i = 0; i < global_cold.size(); ++i) {
            if (std::isnan(global_cold[i]) || std::isinf(global_cold[i])) {
                printf("Validation failed: found NaN or Inf value\n");
            }
            minVal = std::min(minVal, global_cold[i]);
            maxVal = std::max(maxVal, global_cold[i]);
        }

        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

        if (maxVal > 10.0 || minVal < -10.0) {
            printf("Validation failed: values out of expected range\n");
            valid = false;
        }

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);

    MPI_Finalize();
    return 0;
}
