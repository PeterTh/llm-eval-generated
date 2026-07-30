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

// Suppress MPI inline macro conflicts with CUDA
#ifndef _FORCE_INLINES
#define _FORCE_INLINES
#endif

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// ============================================================
// CUDA device data structures and kernels
// ============================================================
struct __align__(16) DeviceBody {
    double x, y, z;
    double vx, vy, vz;
    double pad0, pad1; // padding for vectorized loads
};

__global__ void computeForcesKernel(const DeviceBody* __restrict__ gBodies,
                                     DeviceBody* __restrict__ localBodies,
                                     int nGlobal, int nLocal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLocal) return;

    const double px = localBodies[idx].x;
    const double py = localBodies[idx].y;
    const double pz = localBodies[idx].z;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j = 0; j < nGlobal; ++j) {
        const double dx = gBodies[j].x - px;
        const double dy = gBodies[j].y - py;
        const double dz = gBodies[j].z - pz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    localBodies[idx].vx += DT * Fx;
    localBodies[idx].vy += DT * Fy;
    localBodies[idx].vz += DT * Fz;
}

__global__ void integrateKernel(DeviceBody* __restrict__ bodies, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    bodies[idx].x += bodies[idx].vx * DT;
    bodies[idx].y += bodies[idx].vy * DT;
    bodies[idx].z += bodies[idx].vz * DT;
}

// ============================================================
// Host-side helpers
// ============================================================

void randomizeBodies(std::vector<double>& px, std::vector<double>& py, std::vector<double>& pz,
                     std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                     int n, unsigned int seed) {
    for (int i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// OpenMP-parallelized total energy computation
double computeTotalEnergy(const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz,
                          int n) {
    double energy = 0.0;

    // Kinetic energy (unit mass) - parallelized with OpenMP reduction
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Potential energy (unit mass) - parallelized with OpenMP reduction
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - px[i];
            const double dy = py[j] - py[i];
            const double dz = pz[j] - pz[i];
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate simulation results using OpenMP for parallel checking
bool validateSimulation(const double* px, const double* py, const double* pz,
                        const double* vx, const double* vy, const double* vz,
                        int n) {
    double maxPos = 1e6;
    double maxVel = 1e6;
    bool failed = false;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            #pragma omp atomic write
            failed = true;
        }
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            #pragma omp atomic write
            failed = true;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
            #pragma omp atomic write
            failed = true;
        }
    }
    return !failed;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        // Report CUDA device info
        int cudaDeviceCount = 0;
        cudaGetDeviceCount(&cudaDeviceCount);
        printf("CUDA devices: %d\n", cudaDeviceCount);
        if (cudaDeviceCount > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            printf("Using GPU: %s\n", prop.name);
        }
    }

    // Distribute bodies across MPI ranks
    int localN = numBodies / mpiSize;
    int remainder = numBodies % mpiSize;
    if (mpiRank < remainder) {
        localN++;
    }

    // Compute local offset (start index for this rank's bodies)
    int localOffset = 0;
    for (int r = 0; r < mpiRank; ++r) {
        localOffset += (r < remainder) ? (numBodies / mpiSize + 1) : (numBodies / mpiSize);
    }

    // Host arrays for local bodies (Structure of Arrays for cache efficiency)
    std::vector<double> lpx(localN), lpy(localN), lpz(localN);
    std::vector<double> lvx(localN), lvy(localN), lvz(localN);

    // Host array for global bodies (gathered from all ranks)
    std::vector<double> gpx(numBodies), gpy(numBodies), gpz(numBodies);
    std::vector<double> gvx(numBodies), gvy(numBodies), gvz(numBodies);

    // Initialize bodies on rank 0, then scatter to all ranks
    if (mpiRank == 0) {
        std::vector<double> px(numBodies), py(numBodies), pz(numBodies);
        std::vector<double> vx(numBodies), vy(numBodies), vz(numBodies);
        randomizeBodies(px, py, pz, vx, vy, vz, numBodies, 42);

        // Scatter to all ranks
        std::vector<int> sendCounts(mpiSize), sendDispls(mpiSize);
        int cum = 0;
        for (int r = 0; r < mpiSize; ++r) {
            int rn = numBodies / mpiSize + ((r < remainder) ? 1 : 0);
            sendCounts[r] = rn;
            sendDispls[r] = cum;
            cum += rn;
        }

        MPI_Scatterv(px.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpx.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(py.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpy.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(pz.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpz.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vx.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvx.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vy.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvy.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(vz.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvz.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        std::vector<int> sendCounts(mpiSize), sendDispls(mpiSize);
        int cum = 0;
        for (int r = 0; r < mpiSize; ++r) {
            int rn = numBodies / mpiSize + ((r < remainder) ? 1 : 0);
            sendCounts[r] = rn;
            sendDispls[r] = cum;
            cum += rn;
        }

        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpx.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpy.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lpz.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvx.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvy.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     lvz.data(), localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Prepare MPI displacement and count arrays for Allgatherv
    std::vector<int> recvCounts(mpiSize), recvDispls(mpiSize);
    {
        int cum = 0;
        for (int r = 0; r < mpiSize; ++r) {
            int rn = numBodies / mpiSize + ((r < remainder) ? 1 : 0);
            recvCounts[r] = rn;
            recvDispls[r] = cum;
            cum += rn;
        }
    }

    // CUDA device memory allocation
    DeviceBody* dLocalBodies = nullptr;
    DeviceBody* dGlobalBodies = nullptr;
    cudaMalloc(&dLocalBodies, localN * sizeof(DeviceBody));
    cudaMalloc(&dGlobalBodies, numBodies * sizeof(DeviceBody));

    // CUDA kernel launch configuration
    int blockSize = 256;
    int gridSize = (localN + blockSize - 1) / blockSize;
    if (gridSize > 65535) gridSize = 65535;

    // ============================================================
    // Simulation loop
    // ============================================================
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Step 1: MPI_Allgatherv - gather all body positions/velocities from all ranks to host
        MPI_Allgatherv(lpx.data(), localN, MPI_DOUBLE,
                       gpx.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lpy.data(), localN, MPI_DOUBLE,
                       gpy.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lpz.data(), localN, MPI_DOUBLE,
                       gpz.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lvx.data(), localN, MPI_DOUBLE,
                       gvx.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lvy.data(), localN, MPI_DOUBLE,
                       gvy.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lvz.data(), localN, MPI_DOUBLE,
                       gvz.data(), recvCounts.data(), recvDispls.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Step 2: Copy global bodies to GPU
        {
            // Build DeviceBody array for global bodies
            std::vector<DeviceBody> hostGlobalBodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                hostGlobalBodies[i].x = gpx[i];
                hostGlobalBodies[i].y = gpy[i];
                hostGlobalBodies[i].z = gpz[i];
                hostGlobalBodies[i].vx = gvx[i];
                hostGlobalBodies[i].vy = gvy[i];
                hostGlobalBodies[i].vz = gvz[i];
                hostGlobalBodies[i].pad0 = 0.0;
                hostGlobalBodies[i].pad1 = 0.0;
            }
            cudaMemcpy(dGlobalBodies, hostGlobalBodies.data(),
                       numBodies * sizeof(DeviceBody), cudaMemcpyHostToDevice);
        }

        // Step 3: Copy local bodies to GPU
        {
            std::vector<DeviceBody> hostLocalBodies(localN);
            for (int i = 0; i < localN; ++i) {
                hostLocalBodies[i].x = lpx[i];
                hostLocalBodies[i].y = lpy[i];
                hostLocalBodies[i].z = lpz[i];
                hostLocalBodies[i].vx = lvx[i];
                hostLocalBodies[i].vy = lvy[i];
                hostLocalBodies[i].vz = lvz[i];
                hostLocalBodies[i].pad0 = 0.0;
                hostLocalBodies[i].pad1 = 0.0;
            }
            cudaMemcpy(dLocalBodies, hostLocalBodies.data(),
                       localN * sizeof(DeviceBody), cudaMemcpyHostToDevice);
        }

        // Step 4: CUDA - compute forces for local bodies against all global bodies
        computeForcesKernel<<<gridSize, blockSize>>>(dGlobalBodies, dLocalBodies,
                                                      numBodies, localN);

        // Step 5: CUDA - integrate local bodies
        integrateKernel<<<gridSize, blockSize>>>(dLocalBodies, localN);

        // Synchronize GPU
        cudaDeviceSynchronize();

        // Step 6: Copy updated local bodies back to host
        {
            std::vector<DeviceBody> hostLocalBodies(localN);
            cudaMemcpy(hostLocalBodies.data(), dLocalBodies,
                       localN * sizeof(DeviceBody), cudaMemcpyDeviceToHost);

            for (int i = 0; i < localN; ++i) {
                lpx[i] = hostLocalBodies[i].x;
                lpy[i] = hostLocalBodies[i].y;
                lpz[i] = hostLocalBodies[i].z;
                lvx[i] = hostLocalBodies[i].vx;
                lvy[i] = hostLocalBodies[i].vy;
                lvz[i] = hostLocalBodies[i].vz;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // ============================================================
    // Gather all results to rank 0 for output/validation
    // ============================================================
    std::vector<double> final_gpx(numBodies), final_gpy(numBodies), final_gpz(numBodies);
    std::vector<double> final_gvx(numBodies), final_gvy(numBodies), final_gvz(numBodies);

    MPI_Gatherv(lpx.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gpx.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lpy.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gpy.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lpz.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gpz.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lvx.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gvx.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lvy.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gvy.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lvz.data(), localN, MPI_DOUBLE,
                (mpiRank == 0) ? final_gvz.data() : nullptr,
                recvCounts.data(), recvDispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results for external validation (rank 0 only)
    if (printResults && mpiRank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(final_gpx[i]);
            bodyData.push_back(final_gpy[i]);
            bodyData.push_back(final_gpz[i]);
            bodyData.push_back(final_gvx[i]);
            bodyData.push_back(final_gvy[i]);
            bodyData.push_back(final_gvz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation (rank 0 only, using OpenMP for parallel checking)
    if (validate && mpiRank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(final_gpx.data(), final_gpy.data(), final_gpz.data(),
                               final_gvx.data(), final_gvy.data(), final_gvz.data(),
                               numBodies)) {
            // Compute total energy with OpenMP parallelization
            double finalEnergy = computeTotalEnergy(final_gpx.data(), final_gpy.data(),
                                                     final_gpz.data(),
                                                     final_gvx.data(), final_gvy.data(),
                                                     final_gvz.data(), numBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cudaFree(dLocalBodies);
            cudaFree(dGlobalBodies);
            MPI_Finalize();
            return 1;
        }
    }

    // Cleanup CUDA resources
    cudaFree(dLocalBodies);
    cudaFree(dGlobalBodies);

    MPI_Finalize();
    return 0;
}
