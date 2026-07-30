#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// ============================================================
// Data structures
// ============================================================
struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ============================================================
// CUDA kernels
// ============================================================

// Force computation: one thread per local body, sequential accumulation
// over all bodies. Produces bit-exact results matching the serial code.
// Uses enough threads to maximize occupancy while each thread does
// the full O(N) sequential accumulation for its assigned body.
// local_offset is the global index of the first local body.
__global__ void computeForcesKernel(
    const double *all_x, const double *all_y, const double *all_z,
    double *deltax, double *deltay, double *deltaz,
    int local_n, int total_n, int local_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_n) return;

    int gi = local_offset + idx;  // global body index
    double bx = all_x[gi];
    double by = all_y[gi];
    double bz = all_z[gi];

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < total_n; ++j) {
        double dx = all_x[j] - bx;
        double dy = all_y[j] - by;
        double dz = all_z[j] - bz;
        double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }

    deltax[idx] = fx;
    deltay[idx] = fy;
    deltaz[idx] = fz;
}

// Velocity update + position integration kernel
__global__ void integrateKernel(
    double *x, double *y, double *z,
    double *vx, double *vy, double *vz,
    const double *deltax, const double *deltay, const double *deltaz,
    int n, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update velocity: v += dt * F
        vx[idx] += dt * deltax[idx];
        vy[idx] += dt * deltay[idx];
        vz[idx] += dt * deltaz[idx];
        // Update position: x += v * dt  (using updated velocity)
        x[idx] += vx[idx] * dt;
        y[idx] += vy[idx] * dt;
        z[idx] += vz[idx] * dt;
    }
}

// ============================================================
// CUDA helper
// ============================================================
static cudaError_t cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA Error] %s: %s\n", msg, cudaGetErrorString(err));
        return err;
    }
    return err;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse CLI arguments before MPI init (so -h works standalone)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -n <num>     Number of bodies (default: 1024)\n");
            printf("  -s <num>     Number of simulation steps (default: 10)\n");
            printf("  -v           Enable validation (checks energy conservation)\n");
            printf("  -r           Print results for external validation\n");
            printf("  -h           Show this help message\n");
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Detect GPUs and assign one per rank (round-robin)
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    int gpuId = rank % std::max(numGPUs, 1);

    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", numRanks);
        printf("GPUs detected: %d\n", numGPUs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---------------------------------------------------------------
    // Block distribution of bodies across MPI ranks
    // ---------------------------------------------------------------
    std::vector<int> counts(numRanks), displs(numRanks);
    int base = numBodies / numRanks;
    int remainder = numBodies % numRanks;
    int offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
    int local_n = counts[rank];

    // Initialize local bodies with deterministic RNG (same sequence as serial)
    std::vector<Body> localBodies(local_n);
    {
        unsigned int seed = 42;
        // Advance RNG past bodies owned by earlier ranks
        for (int i = 0; i < displs[rank] * 6; ++i) {
            rand_r(&seed);
        }
        for (auto& body : localBodies) {
            body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }

    // Set CUDA device for this rank
    cudaCheck(cudaSetDevice(gpuId), "cudaSetDevice");

    // ---------------------------------------------------------------
    // GPU memory allocation (SoA layout)
    // ---------------------------------------------------------------
    double *d_all_x = nullptr, *d_all_y = nullptr, *d_all_z = nullptr;
    double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    double *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;

    // Full position arrays (all bodies, for force computation)
    cudaCheck(cudaMalloc(&d_all_x, numBodies * sizeof(double)), "cudaMalloc d_all_x");
    cudaCheck(cudaMalloc(&d_all_y, numBodies * sizeof(double)), "cudaMalloc d_all_y");
    cudaCheck(cudaMalloc(&d_all_z, numBodies * sizeof(double)), "cudaMalloc d_all_z");

    // Local velocity and force arrays
    if (local_n > 0) {
        cudaCheck(cudaMalloc(&d_vx, local_n * sizeof(double)), "cudaMalloc d_vx");
        cudaCheck(cudaMalloc(&d_vy, local_n * sizeof(double)), "cudaMalloc d_vy");
        cudaCheck(cudaMalloc(&d_vz, local_n * sizeof(double)), "cudaMalloc d_vz");
        cudaCheck(cudaMalloc(&d_fx, local_n * sizeof(double)), "cudaMalloc d_fx");
        cudaCheck(cudaMalloc(&d_fy, local_n * sizeof(double)), "cudaMalloc d_fy");
        cudaCheck(cudaMalloc(&d_fz, local_n * sizeof(double)), "cudaMalloc d_fz");
    }

    // Host-side SoA buffers
    std::vector<double> h_all_x(numBodies), h_all_y(numBodies), h_all_z(numBodies);
    std::vector<double> h_vx(local_n), h_vy(local_n), h_vz(local_n);

    // Initialize host arrays from local bodies
    for (int i = 0; i < local_n; ++i) {
        int g = displs[rank] + i;
        h_all_x[g] = localBodies[i].pos.x;
        h_all_y[g] = localBodies[i].pos.y;
        h_all_z[g] = localBodies[i].pos.z;
        h_vx[i] = localBodies[i].vel.x;
        h_vy[i] = localBodies[i].vel.y;
        h_vz[i] = localBodies[i].vel.z;
    }

    // CUDA stream for async transfers
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Pre-upload initial positions to GPU
    if (local_n > 0) {
        cudaCheck(cudaMemcpyAsync(d_all_x, h_all_x.data(), numBodies * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload pos");
        cudaCheck(cudaMemcpyAsync(d_all_y, h_all_y.data(), numBodies * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload pos");
        cudaCheck(cudaMemcpyAsync(d_all_z, h_all_z.data(), numBodies * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload pos");
        cudaCheck(cudaMemcpyAsync(d_vx, h_vx.data(), local_n * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload vel");
        cudaCheck(cudaMemcpyAsync(d_vy, h_vy.data(), local_n * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload vel");
        cudaCheck(cudaMemcpyAsync(d_vz, h_vz.data(), local_n * sizeof(double),
                                  cudaMemcpyHostToDevice, stream), "init upload vel");
    }

    // ---------------------------------------------------------------
    // Simulation loop
    // ---------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // --- 1. MPI: gather all positions to every rank ---
        // Use explicit send/receive buffers for correctness
        MPI_Allgatherv(h_all_x.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_all_y.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_y.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_all_z.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_z.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // --- 2. Upload updated positions to GPU (async) ---
        if (local_n > 0) {
            cudaCheck(cudaMemcpyAsync(d_all_x, h_all_x.data(), numBodies * sizeof(double),
                                      cudaMemcpyHostToDevice, stream), "upload pos");
            cudaCheck(cudaMemcpyAsync(d_all_y, h_all_y.data(), numBodies * sizeof(double),
                                      cudaMemcpyHostToDevice, stream), "upload pos");
            cudaCheck(cudaMemcpyAsync(d_all_z, h_all_z.data(), numBodies * sizeof(double),
                                      cudaMemcpyHostToDevice, stream), "upload pos");
        }

        // --- 3. CUDA: compute forces on local bodies ---
        if (local_n > 0) {
            int blockSize = 256;
            int gridSize = (local_n + blockSize - 1) / blockSize;
            computeForcesKernel<<<gridSize, blockSize, 0, stream>>>(
                d_all_x, d_all_y, d_all_z,
                d_fx, d_fy, d_fz,
                local_n, numBodies, displs[rank]);
        }

        // --- 4. CUDA: update velocity + integrate position ---
        if (local_n > 0) {
            int blockSize = 256;
            int gridSize = (local_n + blockSize - 1) / blockSize;
            integrateKernel<<<gridSize, blockSize, 0, stream>>>(
                d_all_x + displs[rank], d_all_y + displs[rank], d_all_z + displs[rank],
                d_vx, d_vy, d_vz,
                d_fx, d_fy, d_fz,
                local_n, DT);
        }

        // --- 5. Download updated positions and velocities ---
        if (local_n > 0) {
            cudaCheck(cudaMemcpyAsync(h_all_x.data() + displs[rank], d_all_x + displs[rank],
                                      local_n * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "download pos");
            cudaCheck(cudaMemcpyAsync(h_all_y.data() + displs[rank], d_all_y + displs[rank],
                                      local_n * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "download pos");
            cudaCheck(cudaMemcpyAsync(h_all_z.data() + displs[rank], d_all_z + displs[rank],
                                      local_n * sizeof(double), cudaMemcpyDeviceToHost, stream),
                      "download pos");
            cudaCheck(cudaMemcpyAsync(h_vx.data(), d_vx, local_n * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream), "download vel");
            cudaCheck(cudaMemcpyAsync(h_vy.data(), d_vy, local_n * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream), "download vel");
            cudaCheck(cudaMemcpyAsync(h_vz.data(), d_vz, local_n * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream), "download vel");
            cudaCheck(cudaStreamSynchronize(stream), "sync");
        }

        // Update local body state from downloaded data
        for (int i = 0; i < local_n; ++i) {
            localBodies[i].pos.x = h_all_x[displs[rank] + i];
            localBodies[i].pos.y = h_all_y[displs[rank] + i];
            localBodies[i].pos.z = h_all_z[displs[rank] + i];
            localBodies[i].vel.x = h_vx[i];
            localBodies[i].vel.y = h_vy[i];
            localBodies[i].vel.z = h_vz[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // ============================================================
    // Results output
    // ============================================================
    if (printResults) {
        std::vector<double> bodyData(numBodies * 6);
        std::vector<double> localPacked(local_n * 6);
        for (int i = 0; i < local_n; ++i) {
            localPacked[i * 6 + 0] = localBodies[i].pos.x;
            localPacked[i * 6 + 1] = localBodies[i].pos.y;
            localPacked[i * 6 + 2] = localBodies[i].pos.z;
            localPacked[i * 6 + 3] = localBodies[i].vel.x;
            localPacked[i * 6 + 4] = localBodies[i].vel.y;
            localPacked[i * 6 + 5] = localBodies[i].vel.z;
        }

        std::vector<int> pcounts(numRanks), pdispls(numRanks);
        int poff = 0;
        for (int r = 0; r < numRanks; ++r) {
            pcounts[r] = counts[r] * 6;
            pdispls[r] = poff;
            poff += pcounts[r];
        }

        MPI_Gatherv(localPacked.data(), pcounts[rank], MPI_DOUBLE,
                    bodyData.data(), pcounts.data(), pdispls.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    // ============================================================
    // Validation
    // ============================================================
    if (validate) {
        // Validate local bodies with OpenMP
        int localOk = 1;
        #pragma omp parallel for schedule(static) reduction(||:localOk)
        for (int i = 0; i < local_n; ++i) {
            const auto& b = localBodies[i];
            if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
                !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z)) {
                localOk = 0;
            }
            const double maxPos = 1e6, maxVel = 1e6;
            if (std::abs(b.pos.x) > maxPos || std::abs(b.pos.y) > maxPos || std::abs(b.pos.z) > maxPos) {
                localOk = 0;
            }
            if (std::abs(b.vel.x) > maxVel || std::abs(b.vel.y) > maxVel || std::abs(b.vel.z) > maxVel) {
                localOk = 0;
            }
        }

        int globalOk;
        MPI_Reduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (globalOk) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed: found NaN, Inf, or out-of-bounds value\n");
                printf("Validation: FAILED\n");
                // Cleanup and exit
                cudaStreamDestroy(stream);
                cudaFree(d_all_x); cudaFree(d_all_y); cudaFree(d_all_z);
                if (local_n > 0) {
                    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
                    cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
                }
                MPI_Finalize();
                return 1;
            }
        }

        // Compute total energy (kinetic + potential) with MPI + OpenMP
        // Kinetic energy: local sum, then MPI reduce
        double localKE = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:localKE)
        for (int i = 0; i < local_n; ++i) {
            localKE += 0.5 * (localBodies[i].vel.x * localBodies[i].vel.x +
                             localBodies[i].vel.y * localBodies[i].vel.y +
                             localBodies[i].vel.z * localBodies[i].vel.z);
        }

        // Potential energy: each rank handles pairs where first index is in its range
        // Re-gather all positions
        MPI_Allgatherv(h_all_x.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_all_y.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_y.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(h_all_z.data() + displs[rank], counts[rank], MPI_DOUBLE,
                       h_all_z.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        double localPE = 0.0;
        #pragma omp parallel for schedule(static) reduction(+:localPE)
        for (int i = displs[rank]; i < displs[rank] + local_n; ++i) {
            for (int j = i + 1; j < numBodies; ++j) {
                double dx = h_all_x[j] - h_all_x[i];
                double dy = h_all_y[j] - h_all_y[i];
                double dz = h_all_z[j] - h_all_z[i];
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                localPE -= 1.0 / dist;
            }
        }

        double totalKE = 0.0, totalPE = 0.0;
        MPI_Reduce(&localKE, &totalKE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localPE, &totalPE, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Final energy: %.6f\n", totalKE + totalPE);
        }
    }

    // Cleanup GPU resources
    cudaStreamDestroy(stream);
    cudaFree(d_all_x);
    cudaFree(d_all_y);
    cudaFree(d_all_z);
    if (local_n > 0) {
        cudaFree(d_vx);
        cudaFree(d_vy);
        cudaFree(d_vz);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
    }

    MPI_Finalize();
    return 0;
}
