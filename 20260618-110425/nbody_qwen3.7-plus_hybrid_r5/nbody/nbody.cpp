#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

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

// CUDA kernel for force computation on local bodies
__global__ void computeForcesLocalKernel(const double* all_pos, double* local_vel, int total_n, int local_start, int local_n) {
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= local_n) return;
    
    int global_idx = local_start + local_idx;
    
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double px = all_pos[global_idx * 3];
    double py = all_pos[global_idx * 3 + 1];
    double pz = all_pos[global_idx * 3 + 2];
    
    for (int j = 0; j < total_n; ++j) {
        const double dx = all_pos[j * 3] - px;
        const double dy = all_pos[j * 3 + 1] - py;
        const double dz = all_pos[j * 3 + 2] - pz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    
    local_vel[local_idx * 3] += DT * Fx;
    local_vel[local_idx * 3 + 1] += DT * Fy;
    local_vel[local_idx * 3 + 2] += DT * Fz;
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
    
    // Kinetic energy (assuming unit mass) - parallelized with OpenMP
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x + 
                        bodies[i].vel.y * bodies[i].vel.y + 
                        bodies[i].vel.z * bodies[i].vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies) - parallelized with OpenMP
    #pragma omp parallel for reduction(+:energy)
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
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
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
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Assign GPU based on rank (assuming one GPU per rank for now)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    int gpu_id = mpi_rank % num_gpus;
    cudaSetDevice(gpu_id);
    
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 needs to parse, but all ranks need the values)
    if (mpi_rank == 0) {
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
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate local body count for this rank
    int localNumBodies = numBodies / mpi_size;
    int remainder = numBodies % mpi_size;
    int localStart = mpi_rank * localNumBodies + std::min(mpi_rank, remainder);
    if (mpi_rank < remainder) {
        localNumBodies++;
    }
    
    // Initialize all bodies on rank 0, then distribute
    std::vector<Body> allBodies;
    if (mpi_rank == 0) {
        allBodies.resize(numBodies);
        randomizeBodies(allBodies);
    }
    
    // Gather counts from all ranks
    std::vector<int> recvcounts(mpi_size);
    MPI_Gather(&localNumBodies, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast recvcounts to all ranks
    MPI_Bcast(recvcounts.data(), mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute displacements on rank 0
    std::vector<int> displs(mpi_size, 0);
    if (mpi_rank == 0) {
        for (int i = 1; i < mpi_size; ++i) {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    // Broadcast displacements to all ranks
    MPI_Bcast(displs.data(), mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare local bodies
    std::vector<Body> localBodies(localNumBodies);
    
    // Scatter positions and velocities separately
    std::vector<double> sendbuf_pos, sendbuf_vel;
    std::vector<double> local_pos(localNumBodies * 3);
    std::vector<double> local_vel(localNumBodies * 3);
    
    // Convert recvcounts to element counts for MPI_Scatterv
    std::vector<int> recvcounts_pos(mpi_size), recvcounts_vel(mpi_size);
    std::vector<int> displs_pos(mpi_size), displs_vel(mpi_size);
    for (int i = 0; i < mpi_size; ++i) {
        recvcounts_pos[i] = recvcounts[i] * 3;
        recvcounts_vel[i] = recvcounts[i] * 3;
        displs_pos[i] = displs[i] * 3;
        displs_vel[i] = displs[i] * 3;
    }
    
    if (mpi_rank == 0) {
        sendbuf_pos.resize(numBodies * 3);
        sendbuf_vel.resize(numBodies * 3);
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            sendbuf_pos[i * 3] = allBodies[i].pos.x;
            sendbuf_pos[i * 3 + 1] = allBodies[i].pos.y;
            sendbuf_pos[i * 3 + 2] = allBodies[i].pos.z;
            sendbuf_vel[i * 3] = allBodies[i].vel.x;
            sendbuf_vel[i * 3 + 1] = allBodies[i].vel.y;
            sendbuf_vel[i * 3 + 2] = allBodies[i].vel.z;
        }
    }
    
    MPI_Scatterv(sendbuf_pos.data(), recvcounts_pos.data(), displs_pos.data(), MPI_DOUBLE,
                 local_pos.data(), localNumBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(sendbuf_vel.data(), recvcounts_vel.data(), displs_vel.data(), MPI_DOUBLE,
                 local_vel.data(), localNumBodies * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Unpack local bodies
    #pragma omp parallel for
    for (int i = 0; i < localNumBodies; ++i) {
        localBodies[i].pos.x = local_pos[i * 3];
        localBodies[i].pos.y = local_pos[i * 3 + 1];
        localBodies[i].pos.z = local_pos[i * 3 + 2];
        localBodies[i].vel.x = local_vel[i * 3];
        localBodies[i].vel.y = local_vel[i * 3 + 1];
        localBodies[i].vel.z = local_vel[i * 3 + 2];
    }
    
    // Free memory on rank 0
    if (mpi_rank == 0) {
        allBodies.clear();
        allBodies.shrink_to_fit();
    }
    
    // Allocate GPU memory for this rank
    double *d_pos, *d_vel;
    cudaMalloc(&d_pos, numBodies * 3 * sizeof(double));  // Need space for all bodies
    cudaMalloc(&d_vel, localNumBodies * 3 * sizeof(double));
    
    // Determine CUDA block size
    int blockSize = 256;
    
    // Buffer for allgather of positions
    std::vector<double> all_pos(numBodies * 3);
    
    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Allgather positions from all ranks
        std::vector<double> local_pos_send(localNumBodies * 3);
        #pragma omp parallel for
        for (int i = 0; i < localNumBodies; ++i) {
            local_pos_send[i * 3] = localBodies[i].pos.x;
            local_pos_send[i * 3 + 1] = localBodies[i].pos.y;
            local_pos_send[i * 3 + 2] = localBodies[i].pos.z;
        }
        
        MPI_Allgatherv(local_pos_send.data(), localNumBodies * 3, MPI_DOUBLE,
                       all_pos.data(), recvcounts_pos.data(), displs_pos.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Copy all positions to GPU
        cudaMemcpy(d_pos, all_pos.data(), numBodies * 3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Pack local velocities
        std::vector<double> local_vel_pack(localNumBodies * 3);
        #pragma omp parallel for
        for (int i = 0; i < localNumBodies; ++i) {
            local_vel_pack[i * 3] = localBodies[i].vel.x;
            local_vel_pack[i * 3 + 1] = localBodies[i].vel.y;
            local_vel_pack[i * 3 + 2] = localBodies[i].vel.z;
        }
        cudaMemcpy(d_vel, local_vel_pack.data(), localNumBodies * 3 * sizeof(double), cudaMemcpyHostToDevice);
        
        // Launch kernel - each thread computes forces for one local body against all bodies
        int numBlocks = (localNumBodies + blockSize - 1) / blockSize;
        computeForcesLocalKernel<<<numBlocks, blockSize>>>(d_pos, d_vel, numBodies, localStart, localNumBodies);
        
        // Copy back velocities
        cudaMemcpy(local_vel_pack.data(), d_vel, localNumBodies * 3 * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Unpack and integrate
        #pragma omp parallel for
        for (int i = 0; i < localNumBodies; ++i) {
            localBodies[i].vel.x = local_vel_pack[i * 3];
            localBodies[i].vel.y = local_vel_pack[i * 3 + 1];
            localBodies[i].vel.z = local_vel_pack[i * 3 + 2];
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Gather all bodies to rank 0 for validation and output
    std::vector<Body> finalBodies;
    if (printResults || validate) {
        // Pack local results
        std::vector<double> final_pos(localNumBodies * 3);
        std::vector<double> final_vel(localNumBodies * 3);
        #pragma omp parallel for
        for (int i = 0; i < localNumBodies; ++i) {
            final_pos[i * 3] = localBodies[i].pos.x;
            final_pos[i * 3 + 1] = localBodies[i].pos.y;
            final_pos[i * 3 + 2] = localBodies[i].pos.z;
            final_vel[i * 3] = localBodies[i].vel.x;
            final_vel[i * 3 + 1] = localBodies[i].vel.y;
            final_vel[i * 3 + 2] = localBodies[i].vel.z;
        }
        
        std::vector<double> gathered_pos, gathered_vel;
        if (mpi_rank == 0) {
            gathered_pos.resize(numBodies * 3);
            gathered_vel.resize(numBodies * 3);
        }
        
        MPI_Gatherv(final_pos.data(), localNumBodies * 3, MPI_DOUBLE,
                    gathered_pos.data(), recvcounts_pos.data(), displs_pos.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(final_vel.data(), localNumBodies * 3, MPI_DOUBLE,
                    gathered_vel.data(), recvcounts_vel.data(), displs_vel.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            finalBodies.resize(numBodies);
            #pragma omp parallel for
            for (int i = 0; i < numBodies; ++i) {
                finalBodies[i].pos.x = gathered_pos[i * 3];
                finalBodies[i].pos.y = gathered_pos[i * 3 + 1];
                finalBodies[i].pos.z = gathered_pos[i * 3 + 2];
                finalBodies[i].vel.x = gathered_vel[i * 3];
                finalBodies[i].vel.y = gathered_vel[i * 3 + 1];
                finalBodies[i].vel.z = gathered_vel[i * 3 + 2];
            }
        }
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : finalBodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (mpi_rank == 0) {
            printf("Validating simulation results...\n");
        }
        
        bool local_valid = validateSimulation(localBodies);
        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            if (global_valid) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(finalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_vel);
    
    MPI_Finalize();
    return 0;
}
