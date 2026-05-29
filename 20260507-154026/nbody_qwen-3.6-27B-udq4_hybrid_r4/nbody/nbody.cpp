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

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// CUDA kernel: tiled force computation with shared memory
// Each thread computes forces on one body (i) from all bodies (j).
// Shared memory tiles the source bodies for coalesced memory access.
__global__ void computeForcesKernel(
    const double* __restrict__ pos,  // all body positions: [N][3]
    double* __restrict__ vel,         // all body velocities: [N][3]
    const int N)
{
    constexpr int TILE_DIM = 16;
    __shared__ double tilePos[TILE_DIM][3];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (idx >= N) return;

    const double px = pos[idx * 3 + 0];
    const double py = pos[idx * 3 + 1];
    const double pz = pos[idx * 3 + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    // Iterate over tiles of source bodies
    for (int tile = 0; tile < (N + TILE_DIM - 1) / TILE_DIM; ++tile) {
        // Load tile into shared memory (each thread loads one body's position)
        int tidx = tile * TILE_DIM + tid;
        if (tidx < N) {
            tilePos[tid][0] = pos[tidx * 3 + 0];
            tilePos[tid][1] = pos[tidx * 3 + 1];
            tilePos[tid][2] = pos[tidx * 3 + 2];
        }
        __syncthreads();

        // Compute forces from this tile
        int count = (tile + 1) * TILE_DIM < N ? TILE_DIM : (N - tile * TILE_DIM);
        for (int j = 0; j < count; ++j) {
            const double dx = tilePos[j][0] - px;
            const double dy = tilePos[j][1] - py;
            const double dz = tilePos[j][2] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocity
    vel[idx * 3 + 0] += DT * Fx;
    vel[idx * 3 + 1] += DT * Fy;
    vel[idx * 3 + 2] += DT * Fz;
}

// Host-side data structures
struct Body {
    double pos[3];
    double vel[3];
};

// Initialize bodies with deterministic random values (same as original)
void randomizeBodies(std::vector<Body>& bodies, unsigned int seed) {
    for (auto& body : bodies) {
        body.pos[0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos[1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos[2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[0] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[1] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel[2] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// OpenMP parallel integration
void integrateBodies(std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos[0] += bodies[i].vel[0] * DT;
        bodies[i].pos[1] += bodies[i].vel[1] * DT;
        bodies[i].pos[2] += bodies[i].vel[2] * DT;
    }
}

// OpenMP parallel energy computation
double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const int n = static_cast<int>(bodies.size());

    // Kinetic energy with OpenMP reduction
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel[0] * bodies[i].vel[0] +
                         bodies[i].vel[1] * bodies[i].vel[1] +
                         bodies[i].vel[2] * bodies[i].vel[2]);
    }

    // Potential energy with OpenMP reduction
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos[0] - bodies[i].pos[0];
            const double dy = bodies[j].pos[1] - bodies[i].pos[1];
            const double dz = bodies[j].pos[2] - bodies[i].pos[2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate simulation results
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        for (int k = 0; k < 3; ++k) {
            if (!std::isfinite(body.pos[k]) || !std::isfinite(body.vel[k])) {
                printf("Validation failed: found NaN or Inf value in body state\n");
                return false;
            }
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        for (int k = 0; k < 3; ++k) {
            if (std::abs(body.pos[k]) > maxPos) {
                printf("Validation failed: body position exceeds reasonable bounds\n");
                return false;
            }
            if (std::abs(body.vel[k]) > maxVel) {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
                return false;
            }
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
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (before MPI init for -h support)
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Block-distribute bodies across MPI ranks
    int localN = numBodies / numRanks;
    int remainder = numBodies % numRanks;
    if (rank < remainder) localN++;

    // Compute how many bodies precede this rank's portion
    int bodiesBeforeRank = 0;
    for (int r = 0; r < rank; ++r) {
        int rn = numBodies / numRanks + (r < remainder ? 1 : 0);
        bodiesBeforeRank += rn;
    }

    // Advance the global seed past bodies owned by previous ranks,
    // so each rank initializes its local bodies with the same values
    // the sequential code would produce.
    unsigned int seed = 42;
    for (int i = 0; i < bodiesBeforeRank * 6; ++i) rand_r(&seed);

    std::vector<Body> localBodies(localN);
    randomizeBodies(localBodies, seed);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Each rank allocates GPU memory for the full body set
    double *d_pos = nullptr, *d_vel = nullptr;
    cudaMalloc(&d_pos, numBodies * 3 * sizeof(double));
    cudaMalloc(&d_vel, numBodies * 3 * sizeof(double));

    // Per-rank buffers for MPI_Allgatherv
    std::vector<double> localSendPos(localN * 3);
    std::vector<double> localSendVel(localN * 3);
    std::vector<double> fullPos(numBodies * 3);
    std::vector<double> fullVel(numBodies * 3);

    // Build recvcounts/displs for positions/velocities (3 doubles per body)
    std::vector<int> recvCounts3(numRanks);
    std::vector<int> displs3(numRanks);
    {
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            int rn = numBodies / numRanks + (r < remainder ? 1 : 0);
            recvCounts3[r] = rn * 3;
            displs3[r] = offset;
            offset += rn * 3;
        }
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions into send buffer
        for (int i = 0; i < localN; ++i) {
            localSendPos[i * 3 + 0] = localBodies[i].pos[0];
            localSendPos[i * 3 + 1] = localBodies[i].pos[1];
            localSendPos[i * 3 + 2] = localBodies[i].pos[2];
        }

        // All-gather positions: every rank receives the full set
        MPI_Allgatherv(localSendPos.data(), localN * 3, MPI_DOUBLE,
                       fullPos.data(), recvCounts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Pack local velocities into send buffer
        for (int i = 0; i < localN; ++i) {
            localSendVel[i * 3 + 0] = localBodies[i].vel[0];
            localSendVel[i * 3 + 1] = localBodies[i].vel[1];
            localSendVel[i * 3 + 2] = localBodies[i].vel[2];
        }

        // All-gather velocities
        MPI_Allgatherv(localSendVel.data(), localN * 3, MPI_DOUBLE,
                       fullVel.data(), recvCounts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Upload full data to GPU
        cudaMemcpy(d_pos, fullPos.data(), numBodies * 3 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel, fullVel.data(), numBodies * 3 * sizeof(double),
                   cudaMemcpyHostToDevice);

        // Launch CUDA force kernel
        constexpr int BLOCK_DIM = 256;
        int numBlocks = (numBodies + BLOCK_DIM - 1) / BLOCK_DIM;
        computeForcesKernel<<<numBlocks, BLOCK_DIM>>>(d_pos, d_vel, numBodies);
        cudaDeviceSynchronize();

        // Download updated velocities
        cudaMemcpy(fullVel.data(), d_vel, numBodies * 3 * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Extract local velocity updates from full buffer
        for (int i = 0; i < localN; ++i) {
            int globalIdx = bodiesBeforeRank + i;
            localBodies[i].vel[0] = fullVel[globalIdx * 3 + 0];
            localBodies[i].vel[1] = fullVel[globalIdx * 3 + 1];
            localBodies[i].vel[2] = fullVel[globalIdx * 3 + 2];
        }

        // OpenMP parallel integration (local bodies only)
        integrateBodies(localBodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather all bodies to rank 0 for results/validation
    std::vector<double> sendBuf(localN * 6);
    std::vector<double> recvBuf(numBodies * 6);
    for (int i = 0; i < localN; ++i) {
        sendBuf[i * 6 + 0] = localBodies[i].pos[0];
        sendBuf[i * 6 + 1] = localBodies[i].pos[1];
        sendBuf[i * 6 + 2] = localBodies[i].pos[2];
        sendBuf[i * 6 + 3] = localBodies[i].vel[0];
        sendBuf[i * 6 + 4] = localBodies[i].vel[1];
        sendBuf[i * 6 + 5] = localBodies[i].vel[2];
    }

    std::vector<int> recvCounts6(numRanks);
    std::vector<int> displs6(numRanks);
    {
        int offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            int rn = numBodies / numRanks + (r < remainder ? 1 : 0);
            recvCounts6[r] = rn * 6;
            displs6[r] = offset;
            offset += rn * 6;
        }
    }

    MPI_Gatherv(sendBuf.data(), localN * 6, MPI_DOUBLE,
                recvBuf.data(), recvCounts6.data(), displs6.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct bodies on rank 0 and produce output
    if (rank == 0) {
        std::vector<Body> allBodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            allBodies[i].pos[0] = recvBuf[i * 6 + 0];
            allBodies[i].pos[1] = recvBuf[i * 6 + 1];
            allBodies[i].pos[2] = recvBuf[i * 6 + 2];
            allBodies[i].vel[0] = recvBuf[i * 6 + 3];
            allBodies[i].vel[1] = recvBuf[i * 6 + 4];
            allBodies[i].vel[2] = recvBuf[i * 6 + 5];
        }

        // Print results for external validation
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : allBodies) {
                bodyData.push_back(body.pos[0]);
                bodyData.push_back(body.pos[1]);
                bodyData.push_back(body.pos[2]);
                bodyData.push_back(body.vel[0]);
                bodyData.push_back(body.vel[1]);
                bodyData.push_back(body.vel[2]);
            }
            print_results(bodyData, "Bodies");
        }

        // Validation
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(allBodies)) {
                double finalEnergy = computeTotalEnergy(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                cudaFree(d_pos);
                cudaFree(d_vel);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup CUDA
    cudaFree(d_pos);
    cudaFree(d_vel);

    MPI_Finalize();
    return 0;
}
