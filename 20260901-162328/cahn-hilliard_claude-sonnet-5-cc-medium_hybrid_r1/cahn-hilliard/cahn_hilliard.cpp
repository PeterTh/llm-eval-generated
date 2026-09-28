#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),        \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// 3D index calculation
__host__ __device__ inline size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute chemical potential for the owned slab (local z in [1, local_nz]) using a
// local array that carries one halo plane on each side (local z = 0 and local z = local_nz + 1).
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                                const size_t nx, const size_t ny, const size_t local_nz,
                                                const double dx, const double dy, const double dz,
                                                const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zl >= local_nz) return;

    const size_t z = zl + 1;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const size_t id = idx3(x, y, z, nx, ny);
    const double cv = c[id];

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 2.0 * cv) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 2.0 * cv) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 2.0 * cv) / (dz * dz);
    const double lap = cxx + cyy + czz;

    mu[id] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv - gamma * lap;
}

// Cahn-Hilliard update step for the owned slab.
__global__ void cahnHilliardUpdateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                                          double* __restrict__ cnew,
                                          const size_t nx, const size_t ny, const size_t local_nz,
                                          const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zl = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || zl >= local_nz) return;

    const size_t z = zl + 1;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const size_t id = idx3(x, y, z, nx, ny);
    const double muv = mu[id];

    const double cxx = (mu[idx3(xp, y, z, nx, ny)] + mu[idx3(xn, y, z, nx, ny)] - 2.0 * muv) / (dx * dx);
    const double cyy = (mu[idx3(x, yp, z, nx, ny)] + mu[idx3(x, yn, z, nx, ny)] - 2.0 * muv) / (dy * dy);
    const double czz = (mu[idx3(x, y, zp, nx, ny)] + mu[idx3(x, y, zn, nx, ny)] - 2.0 * muv) / (dz * dz);
    const double lap = cxx + cyy + czz;

    cnew[id] = cold[id] + dt * D * lap;
}

// Exchange the z-boundary planes of a local (halo-padded) field with the neighboring
// MPI ranks. Ranks at the global z boundary have no neighbor (MPI_PROC_NULL) and
// instead replicate their own edge plane into the halo, matching the clamped boundary
// condition of the original single-process implementation.
void exchangeHalo(double* d_field, const size_t nx, const size_t ny, const size_t local_nz,
                   const int down, const int up,
                   double* h_send_bottom, double* h_send_top,
                   double* h_recv_bottom, double* h_recv_top) {
    const size_t planeSize = nx * ny;
    const size_t planeBytes = planeSize * sizeof(double);

    double* d_bottom_owned = d_field + idx3(0, 0, 1, nx, ny);
    double* d_top_owned = d_field + idx3(0, 0, local_nz, nx, ny);
    double* d_bottom_halo = d_field + idx3(0, 0, 0, nx, ny);
    double* d_top_halo = d_field + idx3(0, 0, local_nz + 1, nx, ny);

    if (down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(h_send_bottom, d_bottom_owned, planeBytes, cudaMemcpyDeviceToHost));
    }
    if (up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(h_send_top, d_top_owned, planeBytes, cudaMemcpyDeviceToHost));
    }

    MPI_Sendrecv(h_send_bottom, static_cast<int>(planeSize), MPI_DOUBLE, down, 0,
                 h_recv_top, static_cast<int>(planeSize), MPI_DOUBLE, up, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(h_send_top, static_cast<int>(planeSize), MPI_DOUBLE, up, 1,
                 h_recv_bottom, static_cast<int>(planeSize), MPI_DOUBLE, down, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (up != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_top_halo, h_recv_top, planeBytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_top_halo, d_top_owned, planeBytes, cudaMemcpyDeviceToDevice));
    }
    if (down != MPI_PROC_NULL) {
        CUDA_CHECK(cudaMemcpy(d_bottom_halo, h_recv_bottom, planeBytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_bottom_halo, d_bottom_owned, planeBytes, cudaMemcpyDeviceToDevice));
    }
}

// Initialize the concentration field for the locally owned slab (OpenMP-parallel on host).
// Uses global (x, y, z) coordinates so the generated values are bit-identical to the
// original single-process implementation regardless of the domain decomposition.
void initializeConcentration(std::vector<double>& c_local, const size_t nx, const size_t ny, const size_t local_nz,
                              const size_t z_start, const size_t nz) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t zl = 0; zl < local_nz; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t z = z_start + zl;
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                const size_t id = idx3(x, y, zl + 1, nx, ny);
                c_local[id] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    bool hasInvalid = false;

    #pragma omp parallel for reduction(||:hasInvalid) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
            hasInvalid = true;
        }
    }

    if (hasInvalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < c.size(); ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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

    // Parse command line arguments (identical on every rank via mpirun's argv broadcast)
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

    if (static_cast<size_t>(size) > nz) {
        if (rank == 0) {
            fprintf(stderr, "Error: number of MPI ranks (%d) exceeds grid size in Z (%zu)\n", size, nz);
        }
        MPI_Finalize();
        return 1;
    }

    // Assign one GPU per rank (round-robin across the GPUs visible on this node)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices/node: %d, OpenMP threads/rank: %d\n",
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

    // Domain decomposition along z: contiguous slabs, remainder distributed to the first ranks
    const size_t base = nz / static_cast<size_t>(size);
    const size_t rem = nz % static_cast<size_t>(size);
    const size_t local_nz = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t z_start = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int up = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    const size_t planeSize = nx * ny;
    const size_t localAllocSize = planeSize * (local_nz + 2);

    // Host staging buffer used for initialization and gather/validate
    std::vector<double> h_local(planeSize * local_nz);

    // Device buffers, sized with one halo plane on each side
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, localAllocSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localAllocSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localAllocSize * sizeof(double)));

    // Pinned host halo staging buffers for fast MPI transfers
    double* h_send_bottom = nullptr;
    double* h_send_top = nullptr;
    double* h_recv_bottom = nullptr;
    double* h_recv_top = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_send_bottom, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_send_top, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_bottom, planeSize * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recv_top, planeSize * sizeof(double)));

    // Initialize concentration field on host (OpenMP-parallel), then transfer to device
    if (rank == 0) printf("Initializing concentration field...\n");
    {
        std::vector<double> c_local_init(localAllocSize, 0.0);
        initializeConcentration(c_local_init, nx, ny, local_nz, z_start, nz);
        CUDA_CHECK(cudaMemcpy(d_cold, c_local_init.data(), localAllocSize * sizeof(double), cudaMemcpyHostToDevice));
    }

    const dim3 block(8, 8, 4);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((local_nz + block.z - 1) / block.z));

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Exchange concentration halos, then compute chemical potential
        exchangeHalo(d_cold, nx, ny, local_nz, down, up, h_send_bottom, h_send_top, h_recv_bottom, h_recv_top);
        computeChemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz, dx, dy, dz,
                                                         gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        // Exchange chemical potential halos, then update concentration
        exchangeHalo(d_mu, nx, ny, local_nz, down, up, h_send_bottom, h_send_top, h_recv_bottom, h_recv_top);
        cahnHilliardUpdateKernel<<<grid, block>>>(d_cold, d_mu, d_cnew, nx, ny, local_nz, D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double elapsedMs = (end - start) * 1000.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the owned (halo-free) slab back to host; the owned range is a single
    // contiguous block starting at local z = 1.
    CUDA_CHECK(cudaMemcpy(h_local.data(), d_cold + idx3(0, 0, 1, nx, ny),
                          planeSize * local_nz * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather the full field to rank 0 for validation / result printing
    std::vector<double> full;
    std::vector<int> recvcounts;
    std::vector<int> displs;
    if (rank == 0) {
        full.resize(gridSize);
        recvcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            const size_t r_local_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            const size_t r_z_start = static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), rem);
            recvcounts[r] = static_cast<int>(planeSize * r_local_nz);
            displs[r] = static_cast<int>(planeSize * r_z_start);
        }
    }
    MPI_Gatherv(h_local.data(), static_cast<int>(planeSize * local_nz), MPI_DOUBLE,
                rank == 0 ? full.data() : nullptr, rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(full, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full);

            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
