#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions (device version).
// The z direction is always an interior index because the local array has
// one ghost layer on each side; global clamped boundaries are enforced by
// the halo-exchange routine.
__device__ __forceinline__ double computeLaplacianDevice(const double* __restrict__ c,
                                                         const size_t nx, const size_t ny,
                                                         const size_t x, const size_t y, const size_t lz,
                                                         const double invdx2, const double invdy2,
                                                         const double invdz2) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t nplane = nx * ny;
    const size_t zbase = lz * nplane;
    const size_t ybase = y * nx;
    const size_t center_idx = zbase + ybase + x;
    const double center = c[center_idx];

    const double cxx = (c[zbase + ybase + xp] + c[zbase + ybase + xn] - 2.0 * center) * invdx2;
    const double cyy = (c[zbase + yp * nx + x] + c[zbase + yn * nx + x] - 2.0 * center) * invdy2;
    const double czz = (c[(lz + 1) * nplane + ybase + x] +
                        c[(lz - 1) * nplane + ybase + x] - 2.0 * center) * invdz2;

    return cxx + cyy + czz;
}

// Compute chemical potential on the local physical subdomain.
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nzl,
                                               const double invdx2, const double invdy2, const double invdz2,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;      // physical local z index (0 .. nzl-1)
    if (x >= nx || y >= ny || z >= nzl) return;

    const size_t lz = z + 1;          // array index including lower ghost
    const size_t idx = idx3(x, y, lz, nx, ny);

    const double cv = c[idx];
    const double lap = computeLaplacianDevice(c, nx, ny, x, y, lz,
                                              invdx2, invdy2, invdz2);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
             + 3.0 * cv + cv * cv * cv
             - gamma * lap;
}

// Cahn-Hilliard update step on the local physical subdomain.
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nzl,
                                         const double D, const double dt,
                                         const double invdx2, const double invdy2,
                                         const double invdz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z;
    if (x >= nx || y >= ny || z >= nzl) return;

    const size_t lz = z + 1;
    const size_t idx = idx3(x, y, lz, nx, ny);

    const double lap = computeLaplacianDevice(mu, nx, ny, x, y, lz,
                                              invdx2, invdy2, invdz2);
    cnew[idx] = cold[idx] + dt * D * lap;
}

// Return the NUMA node of the CPU currently executing this process, or -1
// when NUMA information is unavailable.
int processNumaNode() {
    const int cpu = sched_getcpu();
    if (cpu < 0) return -1;

    for (int node = 0; node < 256; ++node) {
        char dir[128];
        snprintf(dir, sizeof(dir), "/sys/devices/system/node/node%d", node);
        if (access(dir, F_OK) != 0) {
            if (node == 0) return -1;
            break;
        }
        char cpu_path[160];
        snprintf(cpu_path, sizeof(cpu_path),
                 "/sys/devices/system/node/node%d/cpu%d", node, cpu);
        if (access(cpu_path, F_OK) == 0) return node;
    }
    return -1;
}

// Choose a GPU whose host NUMA node matches the process NUMA node.  This
// avoids costly cross-socket staging copies on multi-socket GPU nodes.
int selectDevice(int local_rank, int local_size, int device_count,
                 int my_numa, const std::vector<int>& rank_numas) {
    std::vector<int> gpu_numas(static_cast<size_t>(device_count), -1);
    for (int d = 0; d < device_count; ++d) {
        char bus_id[64] = {0};
        int numa = -1;
        if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), d) == cudaSuccess) {
            for (char* p = bus_id; *p != '\0'; ++p) {
                if (*p >= 'A' && *p <= 'F') *p = static_cast<char>(*p - 'A' + 'a');
            }
            char path[256];
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bus_id);
            FILE* f = fopen(path, "r");
            if (f != nullptr) {
                if (fscanf(f, "%d", &numa) != 1) numa = -1;
                fclose(f);
            }
        }
        gpu_numas[d] = numa;
    }

    if (my_numa >= 0) {
        int position = 0;
        int count = 0;
        for (int q = 0; q < local_size; ++q) {
            if (rank_numas[q] == my_numa) {
                if (q == local_rank) position = count;
                ++count;
            }
        }

        std::vector<int> candidates;
        for (int d = 0; d < device_count; ++d) {
            if (gpu_numas[d] == my_numa) candidates.push_back(d);
        }
        if (!candidates.empty()) {
            return candidates[position % static_cast<int>(candidates.size())];
        }
    }

    return local_rank % device_count;
}

// Initialize the local physical concentration values.  The formula uses the
// global linear index so that the result is identical regardless of the MPI
// decomposition.
void initializeConcentrationLocal(std::vector<double>& c,
                                  const size_t nx, const size_t ny, const size_t nz,
                                  const size_t z0, const size_t nzl) {
    const size_t nplane = nx * ny;
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t lz = 0; lz < nzl; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t gz = z0 + lz;
                const size_t linear_id = gz * nplane + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[(lz + 1) * nplane + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Exchange one ghost layer in the z direction.  Host staging buffers are
// pinned memory so that the device copies and MPI messages stay efficient.
void exchangeHalos(double* d_field, const size_t nplane, const size_t nzl,
                   const int rank, const int active_ranks, MPI_Comm comm,
                   double* h_send, double* h_recv) {
    if (nzl == 0) return;

    const size_t bytes = nplane * sizeof(double);
    const int count = static_cast<int>(nplane);
    const bool has_lower = (rank > 0);
    const bool has_upper = (rank + 1 < active_ranks);

    // Pack both boundary planes on the device and copy them to pinned host
    // memory in one asynchronous batch.
    if (has_lower) {
        cudaMemcpyAsync(h_send, d_field + nplane, bytes,
                        cudaMemcpyDeviceToHost, 0);
    }
    if (has_upper) {
        cudaMemcpyAsync(h_send + nplane, d_field + nzl * nplane, bytes,
                        cudaMemcpyDeviceToHost, 0);
    }
    cudaStreamSynchronize(0);

    // Post all receives first, then sends, so both directions are in flight
    // simultaneously across the ring.
    MPI_Request requests[4];
    int nrequests = 0;
    if (has_lower) {
        MPI_Irecv(h_recv, count, MPI_DOUBLE, rank - 1, 101,
                  comm, &requests[nrequests++]);
    }
    if (has_upper) {
        MPI_Irecv(h_recv + nplane, count, MPI_DOUBLE, rank + 1, 100,
                  comm, &requests[nrequests++]);
    }
    if (has_lower) {
        MPI_Isend(h_send, count, MPI_DOUBLE, rank - 1, 100,
                  comm, &requests[nrequests++]);
    }
    if (has_upper) {
        MPI_Isend(h_send + nplane, count, MPI_DOUBLE, rank + 1, 101,
                  comm, &requests[nrequests++]);
    }
    MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);

    // Unpack the received ghost planes (or apply the global clamped boundary
    // conditions) in one asynchronous batch.
    if (has_lower) {
        cudaMemcpyAsync(d_field, h_recv, bytes, cudaMemcpyHostToDevice, 0);
    } else {
        cudaMemcpyAsync(d_field, d_field + nplane, bytes,
                        cudaMemcpyDeviceToDevice, 0);
    }
    if (has_upper) {
        cudaMemcpyAsync(d_field + (nzl + 1) * nplane, h_recv + nplane, bytes,
                        cudaMemcpyHostToDevice, 0);
    } else {
        cudaMemcpyAsync(d_field + (nzl + 1) * nplane, d_field + nzl * nplane,
                        bytes, cudaMemcpyDeviceToDevice, 0);
    }
    cudaStreamSynchronize(0);
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) printf("Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
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

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    
    const size_t gridSize = nx * ny * nz;
    const size_t nplane = nx * ny;

    // Build a node-local view of the process and GPU topology.
    MPI_Comm shmcomm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);
    int local_rank = 0;
    int local_size = 1;
    MPI_Comm_rank(shmcomm, &local_rank);
    MPI_Comm_size(shmcomm, &local_size);

    const int my_numa = processNumaNode();
    std::vector<int> rank_numas(static_cast<size_t>(local_size), -1);
    MPI_Allgather(&my_numa, 1, MPI_INT, rank_numas.data(), 1, MPI_INT, shmcomm);

    // Identify the physical node without relying on the global rank layout.
    int node_index = 0;
    MPI_Comm leader_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, (local_rank == 0) ? 0 : MPI_UNDEFINED,
                   rank, &leader_comm);
    if (leader_comm != MPI_COMM_NULL) {
        int leader_rank = 0;
        MPI_Comm_rank(leader_comm, &leader_rank);
        node_index = leader_rank;
        MPI_Comm_free(&leader_comm);
    }
    MPI_Bcast(&node_index, 1, MPI_INT, 0, shmcomm);

    // Reorder ranks by node and NUMA socket (then by local rank).  Consecutive
    // z blocks therefore stay on the same socket as much as possible, which
    // keeps most halo messages inside a NUMA node.
    const long long numa_key = static_cast<long long>(my_numa >= 0 ? my_numa : 1000000);
    const long long sort_key = numa_key * (local_size + 1) + local_rank;
    MPI_Comm decomp_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, node_index, sort_key, &decomp_comm);
    int cart_rank = 0;
    int cart_size = 1;
    MPI_Comm_rank(decomp_comm, &cart_rank);
    MPI_Comm_size(decomp_comm, &cart_size);

    // Decompose the z dimension across the reordered ranks.  Consecutive cart
    // ranks own consecutive z blocks, which keeps the output gather ordered.
    const size_t base = nz / static_cast<size_t>(cart_size);
    const size_t rem = nz % static_cast<size_t>(cart_size);
    const size_t z0 = static_cast<size_t>(cart_rank) * base +
                      std::min(static_cast<size_t>(cart_rank), rem);
    const size_t nzl = base + (static_cast<size_t>(cart_rank) < rem ? 1 : 0);
    const int active_ranks = static_cast<int>(std::min<size_t>(nz, static_cast<size_t>(cart_size)));
    const size_t localSize = (nzl + 2) * nplane;

    // Assign one GPU per rank on each node, preferring a GPU on the same NUMA
    // socket as the process.
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        if (rank == 0) printf("CUDA error: no CUDA-capable device found\n");
        MPI_Comm_free(&decomp_comm);
        MPI_Comm_free(&shmcomm);
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(selectDevice(local_rank, local_size, device_count,
                               my_numa, rank_numas));

    // Allocate device arrays (with one ghost layer at each z boundary).
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    cudaMalloc(&d_cold, localSize * sizeof(double));
    cudaMalloc(&d_cnew, localSize * sizeof(double));
    cudaMalloc(&d_mu, localSize * sizeof(double));

    // Pinned host staging buffers for halo exchanges.
    double* h_send = nullptr;
    double* h_recv = nullptr;
    cudaMallocHost(&h_send, 2 * nplane * sizeof(double));
    cudaMallocHost(&h_recv, 2 * nplane * sizeof(double));

    // Initialize the concentration field on the host with OpenMP, then upload.
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> h_cold(localSize);
    initializeConcentrationLocal(h_cold, nx, ny, nz, z0, nzl);
    cudaMemcpy(d_cold, h_cold.data(), localSize * sizeof(double), cudaMemcpyHostToDevice);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(nzl));

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, nplane, nzl, cart_rank, active_ranks,
                      decomp_comm, h_send, h_recv);

        if (nzl > 0) {
            computeChemicalPotentialKernel<<<grid, block>>>(
                d_cold, d_mu, nx, ny, nzl, invdx2, invdy2, invdz2,
                gamma, e_AA, e_BB, e_AB);
        }

        exchangeHalos(d_mu, nplane, nzl, cart_rank, active_ranks,
                      decomp_comm, h_send, h_recv);

        if (nzl > 0) {
            cahnHilliardUpdateKernel<<<grid, block>>>(
                d_cnew, d_cold, d_mu, nx, ny, nzl, D, dt,
                invdx2, invdy2, invdz2);
        }

        std::swap(d_cold, d_cnew);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count(), global_ms = 0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the final concentration field onto rank 0 for output/validation.
    std::vector<double> globalC;
    if (printResults || validate) {
        std::vector<double> localPhysical(nzl * nplane);
        if (nzl > 0) {
            cudaMemcpy(localPhysical.data(), d_cold + nplane,
                       nzl * nplane * sizeof(double), cudaMemcpyDeviceToHost);
        }

        int localCount = static_cast<int>(nzl * nplane);
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<int> cart_ranks;
        if (rank == 0) {
            counts.resize(nranks);
            displs.resize(nranks);
            cart_ranks.resize(nranks);
        }
        MPI_Gather(&cart_rank, 1, MPI_INT,
                   cart_ranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int r = 0; r < nranks; ++r) {
                const size_t cr = static_cast<size_t>(cart_ranks[r]);
                const size_t rz0 = cr * base + std::min(cr, rem);
                const size_t rnzl = base + (cr < rem ? 1 : 0);
                counts[r] = static_cast<int>(rnzl * nplane);
                displs[r] = static_cast<int>(rz0 * nplane);
            }
            globalC.resize(gridSize);
        }
        MPI_Gatherv(localPhysical.data(), localCount, MPI_DOUBLE,
                    globalC.data(), counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_ms);
        
        double cellUpdates = static_cast<double>(gridSize) * iterations;
        double elapsed = global_ms / 1000.0;
        if (elapsed <= 0.0) elapsed = 1e-9;
        double mcups = cellUpdates / elapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        
        // Print results for external validation
        if (printResults) {
            print_results(globalC, "Concentration");
        }
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalC, nx, ny, nz) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    cudaFreeHost(h_send);
    cudaFreeHost(h_recv);
    cudaFree(d_mu);
    cudaFree(d_cnew);
    cudaFree(d_cold);
    MPI_Comm_free(&decomp_comm);
    MPI_Comm_free(&shmcomm);
    MPI_Finalize();

    return (validate && !valid) ? 1 : 0;
}
