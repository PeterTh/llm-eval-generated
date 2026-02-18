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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: compute chemical potential for local slab
// The local domain has nz_local real z-planes plus ghost layers.
// d_c has layout: [ghost_lo (1 plane)] [nz_local real planes] [ghost_hi (1 plane)]
// total z-extent in arrays = nz_local + 2, with real data at z_offset 1..nz_local
// For boundary clamping we need global z coordinates.
__global__ void kernelChemicalPotential(
    const double* __restrict__ d_c, double* __restrict__ d_mu,
    size_t nx, size_t ny, size_t nz_local,
    size_t nz_global, size_t z_global_start,
    double dx, double dy, double dz,
    double gamma, double e_AA, double e_BB, double e_AB)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z; // local z in [0, nz_local)

    if (x >= nx || y >= ny || zl >= nz_local) return;

    // In the padded array, real data starts at z-offset 1
    size_t zp = zl + 1; // padded z index for this point

    size_t idx = zp * (nx * ny) + y * nx + x;
    double cv = d_c[idx];

    // Neighbor indices in x,y with clamping to global boundaries
    size_t xp_coord = (x < nx - 1) ? x + 1 : x;
    size_t xn_coord = (x > 0) ? x - 1 : x; // clamped: min is 0
    size_t yp_coord = (y < ny - 1) ? y + 1 : y;
    size_t yn_coord = (y > 0) ? y - 1 : y;

    // Z neighbors with clamping to global boundaries
    size_t z_global = z_global_start + zl;
    // If at global boundary, clamp (use same plane = zp). Otherwise use adjacent padded plane.
    size_t zp_up = (z_global < nz_global - 1) ? zp + 1 : zp;
    size_t zp_dn = (z_global > 0) ? zp - 1 : zp;

    double cxp = d_c[zp * (nx * ny) + y * nx + xp_coord];
    double cxn = d_c[zp * (nx * ny) + y * nx + xn_coord];
    double cyp = d_c[zp * (nx * ny) + yp_coord * nx + x];
    double cyn = d_c[zp * (nx * ny) + yn_coord * nx + x];
    double czp = d_c[zp_up * (nx * ny) + y * nx + x];
    double czn = d_c[zp_dn * (nx * ny) + y * nx + x];

    double cxx = (cxp + cxn - 2.0 * cv) / (dx * dx);
    double cyy = (cyp + cyn - 2.0 * cv) / (dy * dy);
    double czz = (czp + czn - 2.0 * cv) / (dz * dz);
    double laplacian = cxx + cyy + czz;

    d_mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
              + 3.0 * cv + cv * cv * cv
              - gamma * laplacian;
}

// CUDA kernel: Cahn-Hilliard update step
__global__ void kernelCahnHilliardUpdate(
    double* __restrict__ d_cnew, const double* __restrict__ d_cold,
    const double* __restrict__ d_mu,
    size_t nx, size_t ny, size_t nz_local,
    size_t nz_global, size_t z_global_start,
    double D, double dt, double dx, double dy, double dz)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t zl = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || zl >= nz_local) return;

    size_t zp = zl + 1;
    size_t idx = zp * (nx * ny) + y * nx + x;

    size_t xp_coord = (x < nx - 1) ? x + 1 : x;
    size_t xn_coord = (x > 0) ? x - 1 : x;
    size_t yp_coord = (y < ny - 1) ? y + 1 : y;
    size_t yn_coord = (y > 0) ? y - 1 : y;

    size_t z_global = z_global_start + zl;
    size_t zp_up = (z_global < nz_global - 1) ? zp + 1 : zp;
    size_t zp_dn = (z_global > 0) ? zp - 1 : zp;

    double mu_c = d_mu[idx];
    double mxp = d_mu[zp * (nx * ny) + y * nx + xp_coord];
    double mxn = d_mu[zp * (nx * ny) + y * nx + xn_coord];
    double myp = d_mu[zp * (nx * ny) + yp_coord * nx + x];
    double myn = d_mu[zp * (nx * ny) + yn_coord * nx + x];
    double mzp = d_mu[zp_up * (nx * ny) + y * nx + x];
    double mzn = d_mu[zp_dn * (nx * ny) + y * nx + x];

    double mxx = (mxp + mxn - 2.0 * mu_c) / (dx * dx);
    double myy = (myp + myn - 2.0 * mu_c) / (dy * dy);
    double mzz = (mzp + mzn - 2.0 * mu_c) / (dz * dz);

    d_cnew[idx] = d_cold[idx] + dt * D * (mxx + myy + mzz);
}

// Initialize concentration field for a local slab
// z_global_start is the global z index of the first real plane in this rank
void initializeConcentration(double* c_padded, size_t nx, size_t ny,
                             size_t nz_local, size_t nz_global,
                             size_t z_global_start) {
    const size_t vol = nx * ny * nz_global;
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t zl = 0; zl < nz_local; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                size_t zp = zl + 1; // offset by ghost layer
                size_t idx = zp * (nx * ny) + y * nx + x;
                size_t z_global = z_global_start + zl;
                size_t linear_id = z_global * (nx * ny) + y * nx + x;
                double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c_padded[idx] = -1.0 + 2.0 * pseudo;
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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

    // Assign one GPU per rank (round-robin if more ranks than GPUs)
    int ndevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));

    // Domain decomposition along Z
    size_t nz_local = nz / nprocs;
    size_t remainder = nz % nprocs;
    size_t z_global_start;
    if ((size_t)rank < remainder) {
        nz_local += 1;
        z_global_start = (size_t)rank * nz_local;
    } else {
        z_global_start = remainder * (nz_local + 1) + ((size_t)rank - remainder) * nz_local;
    }

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per rank: 1, OpenMP threads: %d\n",
               nprocs, omp_get_max_threads());
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma_param = 0.5;
    const double D = 1.0;

    size_t planeSize = nx * ny;
    // Padded local array: 1 ghost plane below + nz_local real planes + 1 ghost plane above
    size_t nz_padded = nz_local + 2;
    size_t localPaddedSize = nz_padded * planeSize;

    // Host buffers for initialization and halo exchange
    std::vector<double> h_cold(localPaddedSize, 0.0);
    std::vector<double> h_send_lo(planeSize);
    std::vector<double> h_send_hi(planeSize);
    std::vector<double> h_recv_lo(planeSize);
    std::vector<double> h_recv_hi(planeSize);

    // Initialize concentration field on host
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(h_cold.data(), nx, ny, nz_local, nz, z_global_start);

    // Allocate device memory
    double *d_cold, *d_cnew, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_cold, localPaddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, localPaddedSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localPaddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cnew, 0, localPaddedSize * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localPaddedSize * sizeof(double)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_cold, h_cold.data(), localPaddedSize * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Create CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Kernel launch configuration
    dim3 blockDim(8, 8, 4);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                 (nz_local + blockDim.z - 1) / blockDim.z);

    int rank_lo = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_hi = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;

    // Halo exchange helper: exchange ghost planes for a given device array
    auto exchangeHalos = [&](double* d_arr) {
        // Copy boundary planes from device to host
        // First real plane (z-padded index 1) -> send to lower neighbor
        CUDA_CHECK(cudaMemcpy(h_send_lo.data(),
                              d_arr + 1 * planeSize,
                              planeSize * sizeof(double),
                              cudaMemcpyDeviceToHost));
        // Last real plane (z-padded index nz_local) -> send to upper neighbor
        CUDA_CHECK(cudaMemcpy(h_send_hi.data(),
                              d_arr + nz_local * planeSize,
                              planeSize * sizeof(double),
                              cudaMemcpyDeviceToHost));

        MPI_Request reqs[4];
        int nreqs = 0;

        // Send lo, recv lo
        MPI_Isend(h_send_lo.data(), (int)planeSize, MPI_DOUBLE, rank_lo, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_lo.data(), (int)planeSize, MPI_DOUBLE, rank_lo, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        // Send hi, recv hi
        MPI_Isend(h_send_hi.data(), (int)planeSize, MPI_DOUBLE, rank_hi, 1,
                  MPI_COMM_WORLD, &reqs[nreqs++]);
        MPI_Irecv(h_recv_hi.data(), (int)planeSize, MPI_DOUBLE, rank_hi, 0,
                  MPI_COMM_WORLD, &reqs[nreqs++]);

        MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE);

        // Copy received ghost planes to device
        // Ghost plane below (z-padded index 0)
        if (rank_lo != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_arr,
                                  h_recv_lo.data(),
                                  planeSize * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        // Ghost plane above (z-padded index nz_local+1)
        if (rank_hi != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(d_arr + (nz_local + 1) * planeSize,
                                  h_recv_hi.data(),
                                  planeSize * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    };

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for c
        exchangeHalos(d_cold);

        // Compute chemical potential
        kernelChemicalPotential<<<gridDim, blockDim, 0, stream>>>(
            d_cold, d_mu, nx, ny, nz_local, nz, z_global_start,
            dx, dy, dz, gamma_param, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Exchange halos for mu
        exchangeHalos(d_mu);

        // Update concentration
        kernelCahnHilliardUpdate<<<gridDim, blockDim, 0, stream>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz_local, nz, z_global_start,
            D, dt, dx, dy, dz);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    size_t gridSize = nx * ny * nz;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results to rank 0 for validation/output
    if (printResults || validate) {
        // Copy local real data from device to host
        std::vector<double> h_local(nz_local * planeSize);
        CUDA_CHECK(cudaMemcpy(h_local.data(),
                              d_cold + 1 * planeSize,
                              nz_local * planeSize * sizeof(double),
                              cudaMemcpyDeviceToHost));

        std::vector<double> gathered;
        if (rank == 0) {
            gathered.resize(gridSize);
        }

        // Gather using variable-length gather
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        int my_count = (int)(nz_local * planeSize);
        MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }

        MPI_Gatherv(h_local.data(), my_count, MPI_DOUBLE,
                     gathered.data(), recvcounts.data(), displs.data(),
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(gathered, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                // Check for NaN or Inf
                bool valid = true;
                double minVal = gathered[0];
                double maxVal = gathered[0];
                #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(&&:valid)
                for (size_t i = 0; i < gathered.size(); ++i) {
                    if (std::isnan(gathered[i]) || std::isinf(gathered[i])) {
                        valid = false;
                    }
                    minVal = std::min(minVal, gathered[i]);
                    maxVal = std::max(maxVal, gathered[i]);
                }
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (valid && maxVal <= 10.0 && minVal >= -10.0) {
                    printf("Validation: PASSED\n");
                } else {
                    if (!valid) printf("Validation failed: found NaN or Inf value\n");
                    else printf("Validation failed: values out of expected range\n");
                    printf("Validation: FAILED\n");
                    CUDA_CHECK(cudaFree(d_cold));
                    CUDA_CHECK(cudaFree(d_cnew));
                    CUDA_CHECK(cudaFree(d_mu));
                    CUDA_CHECK(cudaStreamDestroy(stream));
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
