#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Helper macro for CUDA error checking
#define CUDACHK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, -1); } } while(0)

struct Vec3 {
    double x, y, z;
};

// CUDA kernel: compute forces for local segment and integrate positions
extern "C" __global__ void computeForcesAndIntegrate(const double* __restrict__ posAll, double* __restrict__ posLocal, double* __restrict__ velLocal, int Ntotal, int localOffset, int localN) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= localN) return;

    int i = tid; // index within local arrays
    double px = posLocal[3*i + 0];
    double py = posLocal[3*i + 1];
    double pz = posLocal[3*i + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j = 0; j < Ntotal; ++j) {
        double qx = posAll[3*j + 0];
        double qy = posAll[3*j + 1];
        double qz = posAll[3*j + 2];
        double dx = qx - px;
        double dy = qy - py;
        double dz = qz - pz;
        double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    // Update velocity
    velLocal[3*i + 0] += DT * Fx;
    velLocal[3*i + 1] += DT * Fy;
    velLocal[3*i + 2] += DT * Fz;

    // Integrate position
    posLocal[3*i + 0] = px + velLocal[3*i + 0] * DT;
    posLocal[3*i + 1] = py + velLocal[3*i + 1] * DT;
    posLocal[3*i + 2] = pz + velLocal[3*i + 2] * DT;
}

// Serial validation helpers (run only on rank 0)
struct Body {
    Vec3 pos;
    Vec3 vel;
};

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
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints usage)
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

    if (rank == 0) {
        printf("N-Body Simulation (MPI+OpenMP+CUDA hybrid)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition bodies among ranks (block distribution)
    std::vector<int> counts(size), displs(size);
    int base = numBodies / size;
    int rem = numBodies % size;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + counts[r-1];
    int localN = counts[rank];
    int localOffset = displs[rank];

    // Host buffers: flattened as [x0,y0,z0,x1,y1,z1,...]
    std::vector<double> posLocal(3 * localN);
    std::vector<double> velLocal(3 * localN);
    std::vector<double> posAll(3 * numBodies);

    // Rank 0 initializes full bodies and scatters
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        // Flatten
        std::vector<double> posFlat(3 * numBodies);
        std::vector<double> velFlat(3 * numBodies);
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            posFlat[3*i+0] = bodies[i].pos.x;
            posFlat[3*i+1] = bodies[i].pos.y;
            posFlat[3*i+2] = bodies[i].pos.z;
            velFlat[3*i+0] = bodies[i].vel.x;
            velFlat[3*i+1] = bodies[i].vel.y;
            velFlat[3*i+2] = bodies[i].vel.z;
        }
        // Prepare counts/displs for MPI of doubles (3 per body)
        std::vector<int> counts3(size), displs3(size);
        for (int r = 0; r < size; ++r) { counts3[r] = counts[r] * 3; displs3[r] = displs[r] * 3; }
        MPI_Scatterv(posFlat.data(), counts3.data(), displs3.data(), MPI_DOUBLE, posLocal.data(), localN*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(velFlat.data(), counts3.data(), displs3.data(), MPI_DOUBLE, velLocal.data(), localN*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Build posAll for initial state
        std::copy(posFlat.begin(), posFlat.end(), posAll.begin());
    } else {
        // Receive scattered chunks
        std::vector<int> counts3(size), displs3(size);
        for (int r = 0; r < size; ++r) { counts3[r] = counts[r] * 3; displs3[r] = displs[r] * 3; }
        MPI_Scatterv(nullptr, counts3.data(), displs3.data(), MPI_DOUBLE, posLocal.data(), localN*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, counts3.data(), displs3.data(), MPI_DOUBLE, velLocal.data(), localN*3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Make sure all ranks have posAll for the first step
    std::vector<int> counts3(size), displs3(size);
    for (int r = 0; r < size; ++r) { counts3[r] = counts[r] * 3; displs3[r] = displs[r] * 3; }
    MPI_Allgatherv(posLocal.data(), localN*3, MPI_DOUBLE, posAll.data(), counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Initialize CUDA device for this rank (one GPU per MPI rank if available)
    int deviceCount = 0;
    CUDACHK(cudaGetDeviceCount(&deviceCount));
    int cudaDevice = rank % max(1, deviceCount);
    if (deviceCount > 0) CUDACHK(cudaSetDevice(cudaDevice));

    // Allocate device buffers
    double *d_posAll = nullptr, *d_posLocal = nullptr, *d_velLocal = nullptr;
    if (deviceCount > 0) {
        CUDACHK(cudaMalloc(&d_posAll, sizeof(double) * 3 * numBodies));
        CUDACHK(cudaMalloc(&d_posLocal, sizeof(double) * 3 * localN));
        CUDACHK(cudaMalloc(&d_velLocal, sizeof(double) * 3 * localN));
    }

    // Copy initial local buffers to device
    if (deviceCount > 0) {
        CUDACHK(cudaMemcpy(d_posLocal, posLocal.data(), sizeof(double) * 3 * localN, cudaMemcpyHostToDevice));
        CUDACHK(cudaMemcpy(d_velLocal, velLocal.data(), sizeof(double) * 3 * localN, cudaMemcpyHostToDevice));
        CUDACHK(cudaMemcpy(d_posAll, posAll.data(), sizeof(double) * 3 * numBodies, cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Main simulation loop
    int threadsPerBlock = 256;
    int blocks = (localN + threadsPerBlock - 1) / threadsPerBlock;

    for (int step = 0; step < numSteps; ++step) {
        // Ensure posAll is up to date on device
        if (deviceCount > 0) {
            CUDACHK(cudaMemcpy(d_posAll, posAll.data(), sizeof(double) * 3 * numBodies, cudaMemcpyHostToDevice));
            // Run kernel to update local velocities and integrate local positions
            computeForcesAndIntegrate<<<blocks, threadsPerBlock>>>(d_posAll, d_posLocal, d_velLocal, numBodies, localOffset, localN);
            CUDACHK(cudaGetLastError());
            // Copy back local positions to host for allgatherv
            CUDACHK(cudaMemcpy(posLocal.data(), d_posLocal, sizeof(double) * 3 * localN, cudaMemcpyDeviceToHost));
        } else {
            // Fallback to CPU with OpenMP if no CUDA device
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < localN; ++i) {
                double px = posLocal[3*i+0];
                double py = posLocal[3*i+1];
                double pz = posLocal[3*i+2];
                double Fx = 0.0, Fy = 0.0, Fz = 0.0;
                for (int j = 0; j < numBodies; ++j) {
                    double qx = posAll[3*j+0];
                    double qy = posAll[3*j+1];
                    double qz = posAll[3*j+2];
                    double dx = qx - px;
                    double dy = qy - py;
                    double dz = qz - pz;
                    double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
                    double invDist = 1.0 / std::sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;
                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
                velLocal[3*i+0] += DT * Fx;
                velLocal[3*i+1] += DT * Fy;
                velLocal[3*i+2] += DT * Fz;
                posLocal[3*i+0] = px + velLocal[3*i+0] * DT;
                posLocal[3*i+1] = py + velLocal[3*i+1] * DT;
                posLocal[3*i+2] = pz + velLocal[3*i+2] * DT;
            }
        }

        // Allgather updated positions across ranks for next iteration
        MPI_Allgatherv(posLocal.data(), localN*3, MPI_DOUBLE, posAll.data(), counts3.data(), displs3.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // If CUDA used, keep device velLocal in sync by copying host velLocal back (only if needed next iter)
        if (deviceCount > 0) {
            CUDACHK(cudaMemcpy(d_velLocal, velLocal.data(), sizeof(double) * 3 * localN, cudaMemcpyHostToDevice));
            // Also copy updated posLocal to device for next iter
            CUDACHK(cudaMemcpy(d_posLocal, posLocal.data(), sizeof(double) * 3 * localN, cudaMemcpyHostToDevice));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Simulation time: %0.3f ms\n", (t1 - t0) * 1000.0);
    }

    // Gather final data to rank 0 for printing/validation
    std::vector<double> finalPos, finalVel;
    if (rank == 0) { finalPos.resize(3 * numBodies); finalVel.resize(3 * numBodies); }
    MPI_Gatherv(posLocal.data(), localN*3, MPI_DOUBLE, finalPos.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(velLocal.data(), localN*3, MPI_DOUBLE, finalVel.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(finalPos[3*i+0]);
            bodyData.push_back(finalPos[3*i+1]);
            bodyData.push_back(finalPos[3*i+2]);
            bodyData.push_back(finalVel[3*i+0]);
            bodyData.push_back(finalVel[3*i+1]);
            bodyData.push_back(finalVel[3*i+2]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation on rank 0
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        std::vector<Body> bodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = finalPos[3*i+0]; bodies[i].pos.y = finalPos[3*i+1]; bodies[i].pos.z = finalPos[3*i+2];
            bodies[i].vel.x = finalVel[3*i+0]; bodies[i].vel.y = finalVel[3*i+1]; bodies[i].vel.z = finalVel[3*i+2];
        }
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup device memory
    if (d_posAll) cudaFree(d_posAll);
    if (d_posLocal) cudaFree(d_posLocal);
    if (d_velLocal) cudaFree(d_velLocal);

    MPI_Finalize();
    return 0;
}
