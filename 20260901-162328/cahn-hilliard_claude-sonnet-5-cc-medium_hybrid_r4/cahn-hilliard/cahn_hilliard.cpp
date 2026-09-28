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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                         \
    } while (0)

// 3D index calculation (local, ghost-inclusive along z)
inline __host__ __device__ constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential for the local sub-domain (z-layer decomposition with 1 ghost layer on each side)
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                const size_t nx, const size_t ny, const size_t local_nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zl >= local_nz) return;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t zg = zl + 1;  // ghost-inclusive z index (ghost planes at 0 and local_nz+1)

    const double cv = c[idx3(x, y, zg, nx, ny)];

    const double cxx = (c[idx3(xp, y, zg, nx, ny)] + c[idx3(xn, y, zg, nx, ny)] - 2.0 * cv) / (dx * dx);
    const double cyy = (c[idx3(x, yp, zg, nx, ny)] + c[idx3(x, yn, zg, nx, ny)] - 2.0 * cv) / (dy * dy);
    const double czz = (c[idx3(x, y, zg + 1, nx, ny)] + c[idx3(x, y, zg - 1, nx, ny)] - 2.0 * cv) / (dz * dz);

    const double lap = cxx + cyy + czz;

    mu[idx3(x, y, zg, nx, ny)] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * lap;
}

// Cahn-Hilliard update step for the local sub-domain
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                                          const double* __restrict__ mu,
                                          const size_t nx, const size_t ny, const size_t local_nz,
                                          const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zl >= local_nz) return;

    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t zg = zl + 1;

    const double mv = mu[idx3(x, y, zg, nx, ny)];

    const double cxx = (mu[idx3(xp, y, zg, nx, ny)] + mu[idx3(xn, y, zg, nx, ny)] - 2.0 * mv) / (dx * dx);
    const double cyy = (mu[idx3(x, yp, zg, nx, ny)] + mu[idx3(x, yn, zg, nx, ny)] - 2.0 * mv) / (dy * dy);
    const double czz = (mu[idx3(x, y, zg + 1, nx, ny)] + mu[idx3(x, y, zg - 1, nx, ny)] - 2.0 * mv) / (dz * dz);

    const double lap = cxx + cyy + czz;

    cnew[idx3(x, y, zg, nx, ny)] = cold[idx3(x, y, zg, nx, ny)] + dt * D * lap;
}

// Initialize concentration field for the local z-slab (matches original global pseudo-random formula)
void initializeConcentrationLocal(std::vector<double>& c_local, const size_t nx, const size_t ny,
                                   const size_t nz_global, const size_t z_start, const size_t local_nz) {
    const size_t vol = nx * ny * nz_global;
    const size_t plane = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < local_nz; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t zg = z_start + zl;
                const size_t linear_id = zg * plane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c_local[zl * plane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange ghost (halo) planes with z-neighbors; at global domain boundaries, the boundary
// plane is duplicated into the ghost slot to reproduce the original clamped boundary condition.
void exchangeGhosts(double* d_buf, double* h_send_bottom, double* h_send_top,
                     double* h_recv_bottom, double* h_recv_top,
                     const size_t nx, const size_t ny, const size_t local_nz,
                     const int down_rank, const int up_rank, MPI_Comm comm) {
    const size_t plane = nx * ny;
    const size_t bytes = plane * sizeof(double);

    // Pack boundary real planes from device to host
    CUDA_CHECK(cudaMemcpy(h_send_bottom, d_buf + 1 * plane, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_send_top, d_buf + local_nz * plane, bytes, cudaMemcpyDeviceToHost));

    MPI_Request reqs[4];
    int nreq = 0;
    const int TAG_A = 100;  // bottom-real -> down neighbor's top ghost
    const int TAG_B = 101;  // top-real -> up neighbor's bottom ghost

    MPI_Isend(h_send_bottom, static_cast<int>(plane), MPI_DOUBLE, down_rank, TAG_A, comm, &reqs[nreq++]);
    MPI_Isend(h_send_top, static_cast<int>(plane), MPI_DOUBLE, up_rank, TAG_B, comm, &reqs[nreq++]);
    MPI_Irecv(h_recv_top, static_cast<int>(plane), MPI_DOUBLE, up_rank, TAG_A, comm, &reqs[nreq++]);
    MPI_Irecv(h_recv_bottom, static_cast<int>(plane), MPI_DOUBLE, down_rank, TAG_B, comm, &reqs[nreq++]);

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    // Clamp at global domain boundaries (no MPI neighbor): duplicate own edge plane
    if (down_rank == MPI_PROC_NULL) {
        std::memcpy(h_recv_bottom, h_send_bottom, bytes);
    }
    if (up_rank == MPI_PROC_NULL) {
        std::memcpy(h_recv_top, h_send_top, bytes);
    }

    CUDA_CHECK(cudaMemcpy(d_buf + 0 * plane, h_recv_bottom, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_buf + (local_nz + 1) * plane, h_recv_top, bytes, cudaMemcpyHostToDevice));
}

bool validateResultDistributed(const std::vector<double>& c_local, MPI_Comm comm, double& outMin, double& outMax) {
    double localMin = c_local.empty() ? 0.0 : c_local[0];
    double localMax = c_local.empty() ? 0.0 : c_local[0];
    int localHasNanInf = 0;

    #pragma omp parallel
    {
        double tMin = localMin, tMax = localMax;
        int tNan = 0;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < c_local.size(); ++i) {
            const double val = c_local[i];
            if (std::isnan(val) || std::isinf(val)) tNan = 1;
            tMin = std::min(tMin, val);
            tMax = std::max(tMax, val);
        }
        #pragma omp critical
        {
            localMin = std::min(localMin, tMin);
            localMax = std::max(localMax, tMax);
            localHasNanInf |= tNan;
        }
    }

    double globalMin, globalMax;
    int globalHasNanInf;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localHasNanInf, &globalHasNanInf, 1, MPI_INT, MPI_LOR, comm);

    outMin = globalMin;
    outMax = globalMax;

    if (globalHasNanInf) {
        return false;
    }
    if (globalMax > 10.0 || globalMin < -10.0) {
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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

    if (nz < static_cast<size_t>(size)) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid depth nz (%zu)\n", size, nz);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices visible: %d, OpenMP threads/rank: %d\n",
               size, deviceCount, omp_get_max_threads());
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
    const size_t plane = nx * ny;

    // 1D domain decomposition along z
    std::vector<int> counts(size), displs(size);
    {
        const size_t base = nz / static_cast<size_t>(size);
        const size_t rem = nz % static_cast<size_t>(size);
        size_t offset = 0;
        for (int r = 0; r < size; ++r) {
            const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(offset);
            offset += c;
        }
    }
    const size_t local_nz = static_cast<size_t>(counts[rank]);
    const size_t z_start = static_cast<size_t>(displs[rank]);
    const int down_rank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up_rank = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t local_padded = (local_nz + 2) * plane;

    // Device buffers (ghost-padded along z)
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_padded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_padded * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_padded * sizeof(double)));

    // Pinned host halo-exchange buffers
    double *h_send_bottom, *h_send_top, *h_recv_bottom, *h_recv_top;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, plane * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, plane * sizeof(double)));

    // Initialize concentration field (OpenMP-parallel on host, then upload)
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> h_init(local_nz * plane);
    initializeConcentrationLocal(h_init, nx, ny, nz, z_start, local_nz);
    CUDA_CHECK(cudaMemcpy(d_cold + plane, h_init.data(), local_nz * plane * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 block(8, 8, 4);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>((local_nz + block.z - 1) / block.z));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchangeGhosts(d_cold, h_send_bottom, h_send_top, h_recv_bottom, h_recv_top,
                        nx, ny, local_nz, down_rank, up_rank, MPI_COMM_WORLD);

        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz, dx, dy, dz,
                                                          gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        exchangeGhosts(d_mu, h_send_bottom, h_send_top, h_recv_bottom, h_recv_top,
                        nx, ny, local_nz, down_rank, up_rank, MPI_COMM_WORLD);

        cahnHilliardUpdateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localMs = duration.count();
    long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double mcups = cellUpdates / (maxMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy final local result back to host (real region only)
    std::vector<double> local_result(local_nz * plane);
    CUDA_CHECK(cudaMemcpy(local_result.data(), d_cold + plane, local_nz * plane * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        std::vector<int> elemCounts(size), elemDispls(size);
        for (int r = 0; r < size; ++r) {
            elemCounts[r] = counts[r] * static_cast<int>(plane);
            elemDispls[r] = displs[r] * static_cast<int>(plane);
        }
        std::vector<double> global;
        if (rank == 0) global.resize(gridSize);
        MPI_Gatherv(local_result.data(), static_cast<int>(local_result.size()), MPI_DOUBLE,
                    rank == 0 ? global.data() : nullptr, elemCounts.data(), elemDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results(global, "Concentration");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        double gMin, gMax;
        bool valid = validateResultDistributed(local_result, MPI_COMM_WORLD, gMin, gMax);
        if (rank == 0) {
            printf("Concentration range: [%.6f, %.6f]\n", gMin, gMax);
            if (!valid) {
                printf("Validation failed: found NaN/Inf or out-of-range values\n");
                printf("Validation: FAILED\n");
            } else {
                printf("Validation: PASSED\n");
            }
        }
        exitCode = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaFreeHost(h_send_bottom));
    CUDA_CHECK(cudaFreeHost(h_send_top));
    CUDA_CHECK(cudaFreeHost(h_recv_bottom));
    CUDA_CHECK(cudaFreeHost(h_recv_top));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    MPI_Finalize();
    return exitCode;
}
