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

#include "common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

/* ------------------------------------------------------------------ */
/*  CUDA kernel – O(n^2) force computation on local bodies            */
/* ------------------------------------------------------------------ */
__global__ void computeForcesKernel(
    const Body* d_allBodies, Body* d_localBodies, int totalN, int localN)
{
    int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
          + static_cast<int>(threadIdx.x);
    if (i < localN) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int j = 0; j < totalN; ++j) {
            const double dx = d_allBodies[j].pos.x - d_localBodies[i].pos.x;
            const double dy = d_allBodies[j].pos.y - d_localBodies[i].pos.y;
            const double dz = d_allBodies[j].pos.z - d_localBodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist  = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        d_localBodies[i].vel.x += DT * Fx;
        d_localBodies[i].vel.y += DT * Fy;
        d_localBodies[i].vel.z += DT * Fz;
    }
}

/* ------------------------------------------------------------------ */
/*  main – hybrid MPI / CUDA / OpenMP                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    // ---- MPI init --------------------------------------------------
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies  = 1024;
    int numSteps   = 10;
    bool validate   = false;
    bool printResults = false;

    // ---- Parse CLI args (identical interface) ----------------------
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps  = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of bodies (default: 1024)\n");
                printf("  -s <num>     Number of simulation steps (default: 10)\n");
                printf("  -v           Enable validation (checks energy conservation)\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of bodies (default: 1024)\n");
                printf("  -s <num>     Number of simulation steps (default: 10)\n");
                printf("  -v           Enable validation (checks energy conservation)\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    // ---- Block-distribute bodies across MPI ranks ------------------
    int baseN     = numBodies / numRanks;
    int remainder = numBodies % numRanks;
    int localN    = baseN + (rank < remainder ? 1 : 0);

    // Global start index for this rank's bodies
    int globalStart = baseN * rank + std::min(rank, remainder);

    // ---- Local storage ---------------------------------------------
    std::vector<Body> localBodies(localN);

    // Deterministic randomization – each rank advances the shared seed
    // to the correct position so the result is bit-identical to the
    // original sequential code.
    {
        unsigned int seed = 42;
        for (int k = 0; k < globalStart * 6; ++k) rand_r(&seed);
        for (auto& body : localBodies) {
            body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }

    // ---- MPI Allgatherv bookkeeping (counts & displs in bytes) -----
    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    {
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            int rn = baseN + (r < remainder ? 1 : 0);
            recvCounts[r] = rn * static_cast<int>(sizeof(Body));
            displs[r]     = offset * static_cast<int>(sizeof(Body));
            offset += rn;
        }
    }

    // Buffer that holds every body after Allgatherv
    std::vector<Body> allBodies(numBodies);

    // ---- CUDA device memory ----------------------------------------
    // Single allocation for all bodies; d_localBodies points into it.
    Body* d_allBodies   = nullptr;
    cudaMalloc(&d_allBodies, numBodies * sizeof(Body));
    Body* d_localBodies = d_allBodies + globalStart;

    // ---- Banner (rank 0 only) --------------------------------------
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ================================================================
    //  SIMULATION LOOP
    //    MPI   – distribute bodies across ranks (Allgatherv each step)
    //    CUDA  – O(n^2) force kernel on GPU
    //    OpenMP – O(n) integration on CPU cores
    // ================================================================
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // 1) MPI: gather every rank's bodies so each rank has the full
        //    configuration (all-to-all N-body interaction).
        MPI_Allgatherv(
            localBodies.data(),
            localN * static_cast<int>(sizeof(Body)), MPI_BYTE,
            allBodies.data(),
            recvCounts.data(), displs.data(), MPI_BYTE,
            MPI_COMM_WORLD);

        // 2) CUDA: transfer full configuration to GPU.
        cudaMemcpy(d_allBodies, allBodies.data(),
                   numBodies * sizeof(Body), cudaMemcpyHostToDevice);

        // 3) CUDA: compute forces on local bodies.
        if (localN > 0) {
            const int blockSize = 256;
            const int numBlocks = (localN + blockSize - 1) / blockSize;
            computeForcesKernel<<<numBlocks, blockSize>>>(
                d_allBodies, d_localBodies, numBodies, localN);
            cudaDeviceSynchronize();

            // Copy updated local bodies back to host.
            cudaMemcpy(localBodies.data(), d_localBodies,
                       localN * sizeof(Body), cudaMemcpyDeviceToHost);
        }

        // 4) OpenMP: integrate positions of local bodies.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    if (rank == 0)
        printf("Simulation time: %ld ms\n", duration.count());

    // ================================================================
    //  POST-SIMULATION (gather → rank 0 for output / validation)
    // ================================================================
    int exitCode = 0;

    if (printResults || validate) {
        std::vector<Body> gatheredBodies;
        if (rank == 0) gatheredBodies.resize(numBodies);

        MPI_Gatherv(
            localBodies.data(),
            localN * static_cast<int>(sizeof(Body)), MPI_BYTE,
            rank == 0 ? gatheredBodies.data() : nullptr,
            recvCounts.data(), displs.data(), MPI_BYTE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            // ---- Print results -------------------------------------
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

            // ---- Validation ----------------------------------------
            if (validate) {
                printf("Validating simulation results...\n");

                bool valid = true;
                for (const auto& body : gatheredBodies) {
                    if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
                        !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
                        !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
                        printf(
                            "Validation failed: found NaN or Inf value in body state\n");
                        valid = false;
                        break;
                    }
                    const double maxPos = 1e6, maxVel = 1e6;
                    if (std::abs(body.pos.x) > maxPos ||
                        std::abs(body.pos.y) > maxPos ||
                        std::abs(body.pos.z) > maxPos) {
                        printf(
                            "Validation failed: body position exceeds reasonable bounds\n");
                        valid = false;
                        break;
                    }
                    if (std::abs(body.vel.x) > maxVel ||
                        std::abs(body.vel.y) > maxVel ||
                        std::abs(body.vel.z) > maxVel) {
                        printf(
                            "Validation failed: body velocity exceeds reasonable bounds\n");
                        valid = false;
                        break;
                    }
                }

                if (valid) {
                    // Kinetic energy (OpenMP)
                    double energy = 0.0;
                    #pragma omp parallel for reduction(+ : energy) schedule(static)
                    for (int i = 0; i < numBodies; ++i) {
                        energy += 0.5 * (gatheredBodies[i].vel.x * gatheredBodies[i].vel.x +
                                         gatheredBodies[i].vel.y * gatheredBodies[i].vel.y +
                                         gatheredBodies[i].vel.z * gatheredBodies[i].vel.z);
                    }

                    // Potential energy (OpenMP)
                    double potEnergy = 0.0;
                    #pragma omp parallel for reduction(+ : potEnergy) schedule(static)
                    for (int i = 0; i < numBodies; ++i) {
                        for (int j = i + 1; j < numBodies; ++j) {
                            double dx = gatheredBodies[j].pos.x - gatheredBodies[i].pos.x;
                            double dy = gatheredBodies[j].pos.y - gatheredBodies[i].pos.y;
                            double dz = gatheredBodies[j].pos.z - gatheredBodies[i].pos.z;
                            double dist =
                                std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                            potEnergy -= 1.0 / dist;
                        }
                    }

                    printf("Final energy: %.6f\n", energy + potEnergy);
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    // Broadcast exit code so all ranks return consistently.
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Cleanup ---------------------------------------------------
    cudaFree(d_allBodies);
    MPI_Finalize();
    return exitCode;
}
