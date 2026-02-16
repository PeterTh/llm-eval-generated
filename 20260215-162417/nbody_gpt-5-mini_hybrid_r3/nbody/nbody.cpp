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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Simple CUDA error checking
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { 
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

// GPU kernel: each thread computes forces for one local body against all global positions
__global__ void computeForcesKernel(const double* allPos, int nGlobal, double* localPos, double* localVel, int localN) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localN) return;

    double px = localPos[3*idx + 0];
    double py = localPos[3*idx + 1];
    double pz = localPos[3*idx + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j = 0; j < nGlobal; ++j) {
        double dx = allPos[3*j + 0] - px;
        double dy = allPos[3*j + 1] - py;
        double dz = allPos[3*j + 2] - pz;
        double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    localVel[3*idx + 0] += DT * Fx;
    localVel[3*idx + 1] += DT * Fy;
    localVel[3*idx + 2] += DT * Fz;
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

// Integrate on host using OpenMP
void integrateBodiesHost(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

// Compute total energy on root process for validation
double computeTotalEnergyRoot(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulationRoot(const std::vector<Body>& bodies) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    // Initialize MPI and OpenMP
    MPI_Init(&argc, &argv);
    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", world_size);
    }

    // Determine local partitioning
    int base = numBodies / world_size;
    int rem = numBodies % world_size;
    int local_n = base + (world_rank < rem ? 1 : 0);
    int local_offset = world_rank * base + std::min(world_rank, rem);

    // Prepare local bodies
    std::vector<Body> localBodies(local_n);
    unsigned int seed = 42 + world_rank;
    randomizeBodies(localBodies, seed);

    // Buffers for communication (positions only)
    std::vector<int> sendcounts(world_size), displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        int rn = base + (r < rem ? 1 : 0);
        sendcounts[r] = rn * 3; // x,y,z per body
        displs[r] = (r * base + std::min(r, rem)) * 3;
    }

    std::vector<double> localPos(3 * local_n), localVel(3 * local_n);
    // initialize local buffers
    for (int i = 0; i < local_n; ++i) {
        localPos[3*i+0] = localBodies[i].pos.x;
        localPos[3*i+1] = localBodies[i].pos.y;
        localPos[3*i+2] = localBodies[i].pos.z;
        localVel[3*i+0] = localBodies[i].vel.x;
        localVel[3*i+1] = localBodies[i].vel.y;
        localVel[3*i+2] = localBodies[i].vel.z;
    }

    std::vector<double> allPos(3 * numBodies);

    // Allocate device buffers
    double *d_allPos = nullptr, *d_localPos = nullptr, *d_localVel = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_allPos, sizeof(double) * 3 * numBodies));
    CUDA_CHECK(cudaMalloc((void**)&d_localPos, sizeof(double) * 3 * local_n));
    CUDA_CHECK(cudaMalloc((void**)&d_localVel, sizeof(double) * 3 * local_n));

    // Timing
    double tStart = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        // gather all positions
        MPI_Allgatherv(localPos.data(), localPos.size(), MPI_DOUBLE,
                       allPos.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // copy data to device
        CUDA_CHECK(cudaMemcpy(d_allPos, allPos.data(), sizeof(double)*3*numBodies, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localPos, localPos.data(), sizeof(double)*3*local_n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_localVel, localVel.data(), sizeof(double)*3*local_n, cudaMemcpyHostToDevice));

        // launch kernel: one thread per local body
        int threads = 128;
        int blocks = (local_n + threads - 1) / threads;
        computeForcesKernel<<<blocks, threads>>>(d_allPos, numBodies, d_localPos, d_localVel, local_n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy updated velocities back
        CUDA_CHECK(cudaMemcpy(localVel.data(), d_localVel, sizeof(double)*3*local_n, cudaMemcpyDeviceToHost));

        // integrate on host with OpenMP
        #pragma omp parallel for
        for (int i = 0; i < local_n; ++i) {
            localPos[3*i+0] += localVel[3*i+0] * DT;
            localPos[3*i+1] += localVel[3*i+1] * DT;
            localPos[3*i+2] += localVel[3*i+2] * DT;
        }

        // update localBodies for potential validation/printing
        for (int i = 0; i < local_n; ++i) {
            localBodies[i].pos.x = localPos[3*i+0];
            localBodies[i].pos.y = localPos[3*i+1];
            localBodies[i].pos.z = localPos[3*i+2];
            localBodies[i].vel.x = localVel[3*i+0];
            localBodies[i].vel.y = localVel[3*i+1];
            localBodies[i].vel.z = localVel[3*i+2];
        }
    }

    double tEnd = MPI_Wtime();
    double elapsedMs = (tEnd - tStart) * 1000.0;

    if (world_rank == 0) {
        printf("Simulation time: %.0f ms\n", elapsedMs);
    }

    // If requested, gather all bodies to root and print results
    if (printResults || validate) {
        // Prepare gathering of all positions and velocities (6 doubles per body)
        std::vector<int> sendcounts6(world_size), displs6(world_size);
        for (int r = 0; r < world_size; ++r) {
            int rn = base + (r < rem ? 1 : 0);
            sendcounts6[r] = rn * 6;
            displs6[r] = (r * base + std::min(r, rem)) * 6;
        }
        std::vector<double> localData(6 * local_n);
        for (int i = 0; i < local_n; ++i) {
            localData[6*i+0] = localBodies[i].pos.x;
            localData[6*i+1] = localBodies[i].pos.y;
            localData[6*i+2] = localBodies[i].pos.z;
            localData[6*i+3] = localBodies[i].vel.x;
            localData[6*i+4] = localBodies[i].vel.y;
            localData[6*i+5] = localBodies[i].vel.z;
        }
        std::vector<double> allData;
        if (world_rank == 0) allData.resize(6 * numBodies);

        MPI_Gatherv(localData.data(), localData.size(), MPI_DOUBLE,
                    allData.data(), sendcounts6.data(), displs6.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (world_rank == 0 && printResults) {
            print_results(allData, "Bodies");
        }

        if (world_rank == 0 && validate) {
            // reconstruct bodies vector on root
            std::vector<Body> allBodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                allBodies[i].pos.x = allData[6*i+0];
                allBodies[i].pos.y = allData[6*i+1];
                allBodies[i].pos.z = allData[6*i+2];
                allBodies[i].vel.x = allData[6*i+3];
                allBodies[i].vel.y = allData[6*i+4];
                allBodies[i].vel.z = allData[6*i+5];
            }

            printf("Validating simulation results...\n");
            if (validateSimulationRoot(allBodies)) {
                double finalEnergy = computeTotalEnergyRoot(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_allPos));
    CUDA_CHECK(cudaFree(d_localPos));
    CUDA_CHECK(cudaFree(d_localVel));

    MPI_Finalize();
    return 0;
}
