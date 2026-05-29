#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA kernels – operate on a local slab [0..nx-1] x [0..ny-1] x [0..nz_loc-1]
// The caller maintains halo layers at z=0 and z=nz_loc-1
// ---------------------------------------------------------------------------

// 3D index (local slab)
__device__ inline size_t idx3d(const size_t x, const size_t y, const size_t z,
                               const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device Laplacian with clamped boundaries
__device__ inline double dev_laplacian(const double* __restrict__ c,
                                       const size_t nx, const size_t ny, const size_t nz,
                                       const double dx, const double dy, const double dz,
                                       const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3d(xp, y, z, nx, ny)] + c[idx3d(xn, y, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3d(x, yp, z, nx, ny)] + c[idx3d(x, yn, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3d(x, y, zp, nx, ny)] + c[idx3d(x, y, zn, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dz * dz);
    return cxx + cyy + czz;
}

// Kernel: initialize concentration
__global__ void init_kernel(double* __restrict__ c,
                            const size_t nx, const size_t ny, const size_t nz_loc,
                            const ssize_t z_offset, const size_t vol_global) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz_loc) return;

    const ssize_t global_z = z_offset + static_cast<ssize_t>(z);
    const size_t linear_id = global_z >= 0 ?
        static_cast<size_t>(global_z) * (nx * ny) + y * nx + x : 0;
    const double pseudo = ((((linear_id + 1) * 1299709) % vol_global) / static_cast<double>(vol_global));
    c[idx3d(x, y, z, nx, ny)] = -1.0 + 2.0 * pseudo;
}

// Kernel: compute chemical potential
__global__ void chemical_potential_kernel(const double* __restrict__ c,
                                          double* __restrict__ mu,
                                          const size_t nx, const size_t ny, const size_t nz_loc,
                                          const double dx, const double dy, const double dz,
                                          const double gamma, const double e_AA,
                                          const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz_loc) return;

    const double cv = c[idx3d(x, y, z, nx, ny)];
    mu[idx3d(x, y, z, nx, ny)] =
        4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
        + 3.0 * cv + cv * cv * cv
        - gamma * dev_laplacian(c, nx, ny, nz_loc, dx, dy, dz, x, y, z);
}

// Kernel: Cahn-Hilliard update
__global__ void update_kernel(const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              double* __restrict__ cnew,
                              const size_t nx, const size_t ny, const size_t nz_loc,
                              const double D, const double dt,
                              const double dx, const double dy, const double dz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz_loc) return;

    const size_t idx = idx3d(x, y, z, nx, ny);
    cnew[idx] = cold[idx] + dt * D *
                dev_laplacian(mu, nx, ny, nz_loc, dx, dy, dz, x, y, z);
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

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

// Gather full result on rank 0 and validate / print
bool validateAndPrint(std::vector<double>& full_c,
                      const size_t nx, const size_t ny, const size_t nz,
                      const int rank, const int nprocs,
                      const bool validate, const bool printResults) {
    if (nprocs > 1) {
        if (rank == 0) {
            // Receive from other ranks
            for (int r = 1; r < nprocs; ++r) {
                int count;
                MPI_Recv(&count, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<double> buf(count);
                MPI_Recv(buf.data(), count, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                full_c.insert(full_c.end(), buf.begin(), buf.end());
            }
        } else {
            int count = static_cast<int>(full_c.size());
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(full_c.data(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        if (printResults) {
            print_results(full_c, "Concentration");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            double minVal = full_c[0];
            double maxVal = full_c[0];

#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
            for (size_t i = 0; i < full_c.size(); ++i) {
                if (std::isnan(full_c[i]) || std::isinf(full_c[i])) {
                    valid = false;
                }
                minVal = std::min(minVal, full_c[i]);
                maxVal = std::max(maxVal, full_c[i]);
            }

            if (!valid) {
                printf("Validation failed: found NaN or Inf value\n");
            }
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

            if (maxVal > 10.0 || minVal < -10.0) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }

            if (valid) {
                printf("Validation: PASSED\n");
                return true;
            } else {
                printf("Validation: FAILED\n");
                return false;
            }
        }
    }
    return true;
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

    // Parse command line arguments (all ranks)
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
        printf("MPI ranks: %d\n", nprocs);
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

    // Domain decomposition: split nz among ranks
    // Each rank gets a core domain plus 2 halo layers (top/bottom)
    const size_t base_nz = nz / nprocs;
    const size_t remainder = nz % nprocs;
    const size_t my_nz_core = base_nz + (rank < static_cast<int>(remainder) ? 1 : 0);
    const size_t my_nz = my_nz_core + 2;  // +2 for halo layers

    // z_offset: global z where my local slab starts (top halo maps to z_offset-1 globally)
    // For the init kernel, we pass (z_offset - 1) so that local z=0 corresponds to global z=z_offset-1
    // We use signed arithmetic to avoid underflow for rank 0
    ssize_t z_offset_signed = 0;
    for (int r = 0; r < rank; ++r) {
        z_offset_signed += static_cast<ssize_t>(base_nz) + (r < static_cast<int>(remainder) ? 1 : 0);
    }

    const size_t local_size = nx * ny * my_nz;
    const size_t global_size = nx * ny * nz;

    // Host buffers (for halo exchange)
    std::vector<double> h_cold(local_size);
    std::vector<double> h_cnew(local_size);

    // Device buffers
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cudaMalloc(&d_cold, local_size * sizeof(double));
    cudaMalloc(&d_cnew, local_size * sizeof(double));
    cudaMalloc(&d_mu, local_size * sizeof(double));

    // Halo send/recv buffers (nx * ny doubles per face)
    const size_t face_size = nx * ny;
    std::vector<double> halo_send_top(face_size);
    std::vector<double> halo_send_bot(face_size);
    std::vector<double> halo_recv_bot(face_size);  // from rank_dn → my bottom halo
    std::vector<double> halo_recv_top(face_size);  // from rank_up → my top halo

    // CUDA launch configuration
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (my_nz + block.z - 1) / block.z);

    // Initialize concentration field on GPU
    if (rank == 0) printf("Initializing concentration field...\n");
    init_kernel<<<grid, block>>>(d_cold, nx, ny, my_nz, z_offset_signed - 1, global_size);
    cudaDeviceSynchronize();

    // Copy to host for halo management
    cudaMemcpy(h_cold.data(), d_cold, local_size * sizeof(double), cudaMemcpyDeviceToHost);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // --- Halo exchange ---
        int rank_up = (rank < nprocs - 1) ? rank + 1 : MPI_PROC_NULL;
        int rank_dn = (rank > 0) ? rank - 1 : MPI_PROC_NULL;

        // Pack top halo (z = my_nz - 1) for sending to rank_up
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                halo_send_top[j * nx + i] = h_cold[idx3d(i, j, my_nz - 1, nx, ny)];
            }
        }

        // Pack bottom halo (z = 0) for sending to rank_dn
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                halo_send_bot[j * nx + i] = h_cold[idx3d(i, j, 0, nx, ny)];
            }
        }

        // Non-blocking send/receive for halos
        MPI_Request reqs[4];
        int nreq = 0;

        // Send top halo to rank_up
        if (rank_up != MPI_PROC_NULL) {
            MPI_Isend(halo_send_top.data(), face_size, MPI_DOUBLE, rank_up, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
        // Receive from rank_dn into bottom halo
        if (rank_dn != MPI_PROC_NULL) {
            MPI_Irecv(halo_recv_bot.data(), face_size, MPI_DOUBLE, rank_dn, 0,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }

        // Send bottom halo to rank_dn
        if (rank_dn != MPI_PROC_NULL) {
            MPI_Isend(halo_send_bot.data(), face_size, MPI_DOUBLE, rank_dn, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }
        // Receive from rank_up into top halo
        if (rank_up != MPI_PROC_NULL) {
            MPI_Irecv(halo_recv_top.data(), face_size, MPI_DOUBLE, rank_up, 1,
                      MPI_COMM_WORLD, &reqs[nreq++]);
        }

        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Unpack received halos
        // Data from rank_dn goes into bottom halo (z=0)
        if (rank_dn != MPI_PROC_NULL) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    h_cold[idx3d(i, j, 0, nx, ny)] = halo_recv_bot[j * nx + i];
                }
            }
        }
        // Data from rank_up goes into top halo (z=my_nz-1)
        if (rank_up != MPI_PROC_NULL) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    h_cold[idx3d(i, j, my_nz - 1, nx, ny)] = halo_recv_top[j * nx + i];
                }
            }
        }

        // Copy updated host data to device
        cudaMemcpy(d_cold, h_cold.data(), local_size * sizeof(double), cudaMemcpyHostToDevice);

        // Compute chemical potential on GPU
        chemical_potential_kernel<<<grid, block>>>(
            d_cold, d_mu, nx, ny, my_nz,
            dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Update concentration on GPU
        update_kernel<<<grid, block>>>(
            d_cold, d_mu, d_cnew,
            nx, ny, my_nz, D, dt, dx, dy, dz);

        cudaDeviceSynchronize();

        // Copy result back to host
        cudaMemcpy(h_cnew.data(), d_cnew, local_size * sizeof(double), cudaMemcpyDeviceToHost);

        // Swap: h_cold <-> h_cnew
        std::swap(h_cold, h_cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = static_cast<double>(global_size) * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather results on rank 0 for validation/printing
    if (validate || printResults) {
        // Extract core domain (strip halos) for final output
        std::vector<double> core_cold;
        if (my_nz_core > 0) {
            const size_t core_size = nx * ny * my_nz_core;
            core_cold.resize(core_size);
            for (size_t z = 0; z < my_nz_core; ++z) {
                const size_t src_z = z + 1;  // skip top halo
                std::memcpy(core_cold.data() + z * nx * ny,
                            h_cold.data() + src_z * nx * ny,
                            nx * ny * sizeof(double));
            }
        }
        validateAndPrint(core_cold, nx, ny, nz, rank, nprocs, validate, printResults);
    }

    // Cleanup
    cudaFree(d_cold);
    cudaFree(d_cnew);
    cudaFree(d_mu);

    MPI_Finalize();
    return 0;
}
