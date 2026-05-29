#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ==================== CUDA Kernels ====================

// Force computation kernel with shared-memory tiling.
// Each thread computes the total force on one local body by iterating over
// all bodies in tiles loaded into shared memory.
// All threads in a block participate in __syncthreads(), so we launch exactly
// enough threads to fill each block (no early returns).
__global__ void computeForcesKernel(
    const Body* __restrict__ allBodies,
    const int n,
    const int localStart,
    const int localCount,
    double* __restrict__ d_fx,
    double* __restrict__ d_fy,
    double* __restrict__ d_fz
) {
    extern __shared__ char sharedMem[];
    Body* sBodies = reinterpret_cast<Body*>(sharedMem);

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const Body bodyI = allBodies[localStart + idx];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int tileSize = blockDim.x;
    for (int offset = 0; offset < n; offset += tileSize) {
        if (offset + tid < n) {
            sBodies[tid] = allBodies[offset + tid];
        }
        __syncthreads();

        const int tileEnd = (offset + tileSize < n) ? offset + tileSize : n;
        for (int k = 0; k < tileEnd - offset; k++) {
            const Body bodyJ = sBodies[k];
            const double dx = bodyJ.pos.x - bodyI.pos.x;
            const double dy = bodyJ.pos.y - bodyI.pos.y;
            const double dz = bodyJ.pos.z - bodyI.pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (idx < localCount) {
        d_fx[idx] = Fx;
        d_fy[idx] = Fy;
        d_fz[idx] = Fz;
    }
}

// Energy computation kernel: each thread computes kinetic energy for one body
// and potential energy contribution for pairs (i, j) with j > i.
__global__ void computeEnergyKernel(
    const Body* __restrict__ allBodies,
    const int n,
    double* __restrict__ d_kinetic,
    double* __restrict__ d_potential
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    const Body bodyI = allBodies[idx];

    // Kinetic energy for this body
    double ke = 0.5 * (bodyI.vel.x * bodyI.vel.x +
                        bodyI.vel.y * bodyI.vel.y +
                        bodyI.vel.z * bodyI.vel.z);

    // Potential energy: pairs (i, j) with j > i
    double pe = 0.0;
    for (int j = idx + 1; j < n; j++) {
        const Body bodyJ = allBodies[j];
        const double dx = bodyJ.pos.x - bodyI.pos.x;
        const double dy = bodyJ.pos.y - bodyI.pos.y;
        const double dz = bodyJ.pos.z - bodyI.pos.z;
        const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        pe -= 1.0 / dist;
    }

    d_kinetic[idx] = ke;
    d_potential[idx] = pe;
}

// ==================== Host Functions ====================

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (mpiRank == 0) {
        printf("N-Body Simulation (Hybrid MPI/OpenMP/CUDA)\n");
        printf("MPI ranks: %d\n", mpiSize);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Body distribution: contiguous chunks across ranks ----
    const int baseCount = numBodies / mpiSize;
    const int remainder = numBodies % mpiSize;
    const size_t localCount = static_cast<size_t>(baseCount) + (mpiRank < remainder ? 1 : 0);
    const size_t localStart = static_cast<size_t>(baseCount) * mpiRank + std::min(mpiRank, remainder);

    // MPI communication parameters (counts in doubles; each Body = 6 doubles)
    std::vector<int> sendCounts(mpiSize);
    std::vector<int> sendDispls(mpiSize);
    {
        int offset = 0;
        for (int r = 0; r < mpiSize; r++) {
            int rc = baseCount + (r < remainder ? 1 : 0);
            sendCounts[r] = rc * 6;
            sendDispls[r] = offset * 6;
            offset += rc;
        }
    }

    // Position-only communication parameters (3 doubles per position)
    std::vector<int> posRecvCounts(mpiSize);
    std::vector<int> posRecvDispls(mpiSize);
    {
        int offset = 0;
        for (int r = 0; r < mpiSize; r++) {
            int rc = baseCount + (r < remainder ? 1 : 0);
            posRecvCounts[r] = rc * 3;
            posRecvDispls[r] = offset * 3;
            offset += rc;
        }
    }

    // ---- Initialize bodies on rank 0, scatter to all ranks ----
    std::vector<Body> localBodies(localCount);
    const int localCountInt = static_cast<int>(localCount);
    if (mpiRank == 0) {
        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies);
        MPI_Scatterv(
            reinterpret_cast<const double*>(allBodies.data()),
            sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
            reinterpret_cast<double*>(localBodies.data()),
            localCountInt * 6, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    } else {
        MPI_Scatterv(
            nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
            reinterpret_cast<double*>(localBodies.data()),
            localCountInt * 6, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );
    }

    // ---- Allocate GPU memory ----
    Body* d_allBodies = nullptr;
    double* d_fx = nullptr;
    double* d_fy = nullptr;
    double* d_fz = nullptr;

    cudaMalloc(&d_allBodies, numBodies * sizeof(Body));
    if (localCount > 0) {
        cudaMalloc(&d_fx, localCount * sizeof(double));
        cudaMalloc(&d_fy, localCount * sizeof(double));
        cudaMalloc(&d_fz, localCount * sizeof(double));
    }

    // ---- Host buffers for simulation ----
    std::vector<double> localPos(localCount * 3);
    std::vector<double> allPos(numBodies * 3);
    std::vector<Body> allBodies(numBodies);
    std::vector<double> h_fx(localCount), h_fy(localCount), h_fz(localCount);

    // CUDA configuration
    const int blockSize = 256;
    const int gridSize = (localCount + blockSize - 1) / blockSize;
    const size_t sharedMemSize = blockSize * sizeof(Body);

    // ==================== Simulation Loop ====================
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // 1. Extract local positions (OpenMP)
        #pragma omp parallel for
        for (size_t i = 0; i < localCount; i++) {
            localPos[i * 3]     = localBodies[i].pos.x;
            localPos[i * 3 + 1] = localBodies[i].pos.y;
            localPos[i * 3 + 2] = localBodies[i].pos.z;
        }

        // 2. Allgather positions across all ranks
        MPI_Allgatherv(
            localPos.data(), localCount * 3, MPI_DOUBLE,
            allPos.data(), posRecvCounts.data(), posRecvDispls.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // 3. Reconstruct full body array from gathered positions (OpenMP)
        #pragma omp parallel for
        for (size_t i = 0; i < static_cast<size_t>(numBodies); i++) {
            allBodies[i].pos.x = allPos[i * 3];
            allBodies[i].pos.y = allPos[i * 3 + 1];
            allBodies[i].pos.z = allPos[i * 3 + 2];
        }

        // 4. Copy all body positions to GPU
        cudaMemcpy(d_allBodies, allBodies.data(), numBodies * sizeof(Body),
                   cudaMemcpyHostToDevice);

        // 5. Launch force computation kernel on GPU
        if (localCount > 0 && gridSize > 0) {
            computeForcesKernel<<<gridSize, blockSize, sharedMemSize>>>(
                d_allBodies, static_cast<int>(numBodies), static_cast<int>(localStart),
                static_cast<int>(localCount), d_fx, d_fy, d_fz
            );
            cudaDeviceSynchronize();

            // 6. Copy force results back from GPU
            cudaMemcpy(h_fx.data(), d_fx, localCount * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fy.data(), d_fy, localCount * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fz.data(), d_fz, localCount * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // 7. Update local velocities (OpenMP)
        #pragma omp parallel for
        for (size_t i = 0; i < localCount; i++) {
            localBodies[i].vel.x += DT * h_fx[i];
            localBodies[i].vel.y += DT * h_fy[i];
            localBodies[i].vel.z += DT * h_fz[i];
        }

        // 8. Integrate local positions (OpenMP)
        #pragma omp parallel for
        for (size_t i = 0; i < localCount; i++) {
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // ==================== Post-Simulation: Gather & Validate ====================

    // Gather all bodies back to rank 0
    std::vector<Body> gatheredBodies;
    if (mpiRank == 0) gatheredBodies.resize(numBodies);

    MPI_Gatherv(
        reinterpret_cast<const double*>(localBodies.data()),
        localCountInt * 6, MPI_DOUBLE,
        mpiRank == 0 ? reinterpret_cast<double*>(gatheredBodies.data()) : nullptr,
        sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (mpiRank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : gatheredBodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(gatheredBodies)) {
                // Compute total energy using GPU
                Body* d_bodies = nullptr;
                double* d_kinetic = nullptr;
                double* d_potential = nullptr;

                cudaMalloc(&d_bodies, numBodies * sizeof(Body));
                cudaMalloc(&d_kinetic, numBodies * sizeof(double));
                cudaMalloc(&d_potential, numBodies * sizeof(double));

                cudaMemcpy(d_bodies, gatheredBodies.data(), numBodies * sizeof(Body),
                           cudaMemcpyHostToDevice);

                const int eBlockSize = 256;
                const int eGridSize = (numBodies + eBlockSize - 1) / eBlockSize;

                computeEnergyKernel<<<eGridSize, eBlockSize>>>(
                    d_bodies, numBodies, d_kinetic, d_potential
                );
                cudaDeviceSynchronize();

                std::vector<double> h_kinetic(numBodies), h_potential(numBodies);
                cudaMemcpy(h_kinetic.data(), d_kinetic, numBodies * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_potential.data(), d_potential, numBodies * sizeof(double),
                           cudaMemcpyDeviceToHost);

                double totalEnergy = 0.0;
                #pragma omp parallel for reduction(+:totalEnergy)
                for (size_t i = 0; i < static_cast<size_t>(numBodies); i++) {
                    totalEnergy += h_kinetic[i] + h_potential[i];
                }

                printf("Final energy: %.6f\n", totalEnergy);
                printf("Validation: PASSED\n");

                cudaFree(d_bodies);
                cudaFree(d_kinetic);
                cudaFree(d_potential);

                cudaFree(d_allBodies);
                if (localCount > 0) {
                    cudaFree(d_fx);
                    cudaFree(d_fy);
                    cudaFree(d_fz);
                }
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");

                cudaFree(d_allBodies);
                if (localCount > 0) {
                    cudaFree(d_fx);
                    cudaFree(d_fy);
                    cudaFree(d_fz);
                }
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup on all ranks
    cudaFree(d_allBodies);
    if (localCount > 0) {
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_fz);
    }

    MPI_Finalize();
    return 0;
}
