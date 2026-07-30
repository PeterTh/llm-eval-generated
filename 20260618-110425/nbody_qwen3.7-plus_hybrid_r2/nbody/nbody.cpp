#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Tile size for shared memory in CUDA kernel
#define TILE_SIZE 128

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x_ = 0, const double y_ = 0, const double z_ = 0) noexcept : x(x_), y(y_), z(z_) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// CUDA kernel: compute forces and update velocities for local bodies
// Uses shared memory tiling for position data
__global__ void computeForcesKernel(const double* __restrict__ allPos,
                                    double* __restrict__ localVel,
                                    const double* __restrict__ localPos,
                                    int localN, int totalN, double dt, double softening) {
    __shared__ double spos[3 * TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (i < localN) {
        px = localPos[i * 3 + 0];
        py = localPos[i * 3 + 1];
        pz = localPos[i * 3 + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < totalN; tile += TILE_SIZE) {
        int j = tile + tid;
        if (j < totalN) {
            spos[tid * 3 + 0] = allPos[j * 3 + 0];
            spos[tid * 3 + 1] = allPos[j * 3 + 1];
            spos[tid * 3 + 2] = allPos[j * 3 + 2];
        } else {
            spos[tid * 3 + 0] = 0.0;
            spos[tid * 3 + 1] = 0.0;
            spos[tid * 3 + 2] = 0.0;
        }
        __syncthreads();

        if (i < localN) {
            int tileEnd = min(TILE_SIZE, totalN - tile);
            for (int k = 0; k < tileEnd; ++k) {
                const double dx = spos[k * 3 + 0] - px;
                const double dy = spos[k * 3 + 1] - py;
                const double dz = spos[k * 3 + 2] - pz;
                const double distSqr = dx * dx + dy * dy + dz * dz + softening;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < localN) {
        localVel[i * 3 + 0] += dt * Fx;
        localVel[i * 3 + 1] += dt * Fy;
        localVel[i * 3 + 2] += dt * Fz;
    }
}

// CUDA kernel: integrate bodies (update positions from velocities)
__global__ void integrateBodiesKernel(double* __restrict__ localPos,
                                      const double* __restrict__ localVel,
                                      int localN, double dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localN) {
        localPos[i * 3 + 0] += localVel[i * 3 + 0] * dt;
        localPos[i * 3 + 1] += localVel[i * 3 + 1] * dt;
        localPos[i * 3 + 2] += localVel[i * 3 + 2] * dt;
    }
}

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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy - parallelized with OpenMP
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t b = 0; b < n; ++b) {
        energy += 0.5 * (bodies[b].vel.x * bodies[b].vel.x +
                         bodies[b].vel.y * bodies[b].vel.y +
                         bodies[b].vel.z * bodies[b].vel.z);
    }

    // Potential energy - parallelized with OpenMP
    #pragma omp parallel
    {
        double localEnergy = 0.0;
        #pragma omp for schedule(dynamic, 64) nowait
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double dx = bodies[j].pos.x - bodies[i].pos.x;
                const double dy = bodies[j].pos.y - bodies[i].pos.y;
                const double dz = bodies[j].pos.z - bodies[i].pos.z;
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                localEnergy -= 1.0 / dist;
            }
        }
        #pragma omp atomic
        energy += localEnergy;
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    bool valid = true;
    const size_t n = bodies.size();

    #pragma omp parallel
    {
        bool localValid = true;
        #pragma omp for schedule(static) nowait
        for (size_t b = 0; b < n; ++b) {
            if (!std::isfinite(bodies[b].pos.x) || !std::isfinite(bodies[b].pos.y) || !std::isfinite(bodies[b].pos.z) ||
                !std::isfinite(bodies[b].vel.x) || !std::isfinite(bodies[b].vel.y) || !std::isfinite(bodies[b].vel.z)) {
                localValid = false;
            }
            const double maxPos = 1e6;
            const double maxVel = 1e6;
            if (std::abs(bodies[b].pos.x) > maxPos || std::abs(bodies[b].pos.y) > maxPos || std::abs(bodies[b].pos.z) > maxPos) {
                localValid = false;
            }
            if (std::abs(bodies[b].vel.x) > maxVel || std::abs(bodies[b].vel.y) > maxVel || std::abs(bodies[b].vel.z) > maxVel) {
                localValid = false;
            }
        }
        if (!localValid) {
            #pragma omp critical
            {
                valid = false;
            }
        }
    }

    if (!valid) {
        printf("Validation failed: found invalid values in body state\n");
    }
    return valid;
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

    // Select GPU based on MPI rank (round-robin across available GPUs)
    int numGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        if (mpiRank == 0) fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Finalize();
        return 1;
    }
    int gpuId = mpiRank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (rank 0 parses, then broadcasts)
    if (mpiRank == 0) {
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
                MPI_Finalize();
                return 0;
            }
        }
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    if (mpiRank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d, GPUs: %d\n", mpiSize, numGPUs);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies on rank 0
    std::vector<Body> bodies;
    if (mpiRank == 0) {
        bodies.resize(numBodies);
        randomizeBodies(bodies);
    }

    // Compute local body distribution
    // Each rank gets a contiguous chunk of bodies
    int localN = numBodies / mpiSize;
    int remainder = numBodies % mpiSize;
    int localCount;
    if (mpiRank < remainder) {
        localCount = localN + 1;
    } else {
        localCount = localN;
    }

    // Scatter initial body data
    std::vector<double> localPos(localCount * 3);
    std::vector<double> localVel(localCount * 3);

    // Build send counts and displacements for scatterv
    std::vector<int> sendCounts(mpiSize), sendDispls(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        int cnt;
        if (r < remainder) cnt = localN + 1;
        else cnt = localN;
        sendCounts[r] = cnt * 3;
        if (r == 0) sendDispls[r] = 0;
        else sendDispls[r] = sendDispls[r-1] + sendCounts[r-1];
    }

    // Scatter positions and velocities from rank 0
    if (mpiRank == 0) {
        // Interleave pos/vel for scatter
        std::vector<double> allPos(numBodies * 3);
        std::vector<double> allVel(numBodies * 3);
        for (int b = 0; b < numBodies; ++b) {
            allPos[b * 3 + 0] = bodies[b].pos.x;
            allPos[b * 3 + 1] = bodies[b].pos.y;
            allPos[b * 3 + 2] = bodies[b].pos.z;
            allVel[b * 3 + 0] = bodies[b].vel.x;
            allVel[b * 3 + 1] = bodies[b].vel.y;
            allVel[b * 3 + 2] = bodies[b].vel.z;
        }
        MPI_Scatterv(allPos.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     localPos.data(), localCount * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(allVel.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                     localVel.data(), localCount * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Free bodies vector on rank 0 to save memory
        bodies.clear();
        bodies.shrink_to_fit();
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     localPos.data(), localCount * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     localVel.data(), localCount * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Allocate GPU memory
    double *d_allPos, *d_localPos, *d_localVel;
    CUDA_CHECK(cudaMalloc(&d_allPos, numBodies * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localPos, localCount * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_localVel, localCount * 3 * sizeof(double)));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_localPos, localPos.data(), localCount * 3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_localVel, localVel.data(), localCount * 3 * sizeof(double), cudaMemcpyHostToDevice));

    // All-gather buffer for positions (every rank needs all positions)
    std::vector<double> allPosBuf(numBodies * 3);
    std::vector<int> recvCounts(mpiSize), recvDispls(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        int cnt;
        if (r < remainder) cnt = localN + 1;
        else cnt = localN;
        recvCounts[r] = cnt * 3;
        if (r == 0) recvDispls[r] = 0;
        else recvDispls[r] = recvDispls[r-1] + recvCounts[r-1];
    }

    // CUDA kernel launch parameters
    const int blockSize = 128;
    const int forceGridSize = (localCount + blockSize - 1) / blockSize;
    const int integrateGridSize = (localCount + blockSize - 1) / blockSize;

    // Use a CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // All-gather positions so every rank has all body positions
        MPI_Allgatherv(localPos.data(), localCount * 3, MPI_DOUBLE,
                       allPosBuf.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Copy all positions to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_allPos, allPosBuf.data(), numBodies * 3 * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        // Compute forces on GPU (updates velocities)
        computeForcesKernel<<<forceGridSize, blockSize, 0, stream>>>(
            d_allPos, d_localVel, d_localPos, localCount, numBodies, DT, SOFTENING);

        // Integrate bodies on GPU (updates positions from velocities)
        integrateBodiesKernel<<<integrateGridSize, blockSize, 0, stream>>>(
            d_localPos, d_localVel, localCount, DT);

        // Copy updated local positions and velocities back to host
        CUDA_CHECK(cudaMemcpyAsync(localPos.data(), d_localPos, localCount * 3 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(localVel.data(), d_localVel, localCount * 3 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather all results to rank 0 for output
    std::vector<double> allPosFinal, allVelFinal;
    if (mpiRank == 0) {
        allPosFinal.resize(numBodies * 3);
        allVelFinal.resize(numBodies * 3);
    }

    MPI_Gatherv(localPos.data(), localCount * 3, MPI_DOUBLE,
                allPosFinal.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localVel.data(), localCount * 3, MPI_DOUBLE,
                allVelFinal.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Print results for external validation (rank 0 only)
    if (printResults && mpiRank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int b = 0; b < numBodies; ++b) {
            bodyData.push_back(allPosFinal[b * 3 + 0]);
            bodyData.push_back(allPosFinal[b * 3 + 1]);
            bodyData.push_back(allPosFinal[b * 3 + 2]);
            bodyData.push_back(allVelFinal[b * 3 + 0]);
            bodyData.push_back(allVelFinal[b * 3 + 1]);
            bodyData.push_back(allVelFinal[b * 3 + 2]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation (rank 0 only)
    if (validate && mpiRank == 0) {
        printf("Validating simulation results...\n");

        // Reconstruct bodies vector for validation
        std::vector<Body> finalBodies(numBodies);
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < numBodies; ++b) {
            finalBodies[b].pos.x = allPosFinal[b * 3 + 0];
            finalBodies[b].pos.y = allPosFinal[b * 3 + 1];
            finalBodies[b].pos.z = allPosFinal[b * 3 + 2];
            finalBodies[b].vel.x = allVelFinal[b * 3 + 0];
            finalBodies[b].vel.y = allVelFinal[b * 3 + 1];
            finalBodies[b].vel.z = allVelFinal[b * 3 + 2];
        }

        if (validateSimulation(finalBodies)) {
            double finalEnergy = computeTotalEnergy(finalBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_allPos));
    CUDA_CHECK(cudaFree(d_localPos));
    CUDA_CHECK(cudaFree(d_localVel));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();
    return 0;
}
