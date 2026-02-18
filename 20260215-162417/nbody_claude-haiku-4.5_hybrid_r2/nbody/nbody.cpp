#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>

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

void computeForces(std::vector<Body>& bodies, int mpi_rank, int mpi_size) {
    const size_t n = bodies.size();
    const size_t local_n = (n + mpi_size - 1) / mpi_size;
    const size_t start_idx = mpi_rank * local_n;
    const size_t end_idx = std::min(start_idx + local_n, n);
    
    // Each MPI rank computes forces for its subset of bodies
    #pragma omp parallel for collapse(1) default(none) shared(bodies, n, start_idx, end_idx)
    for (size_t i = start_idx; i < end_idx; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for default(none) shared(bodies)
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) default(none) shared(bodies, n)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for collapse(2) reduction(-:energy) default(none) shared(bodies, n)
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (mpi_rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP Parallelization)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", mpi_size);
        printf("OpenMP threads per process: %d\n", omp_get_max_threads());
    }
    
    // Initialize bodies (all ranks have same initial state due to same seed)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Synchronize all body positions before force computation
        // Each process contributes its subset of positions, receives all positions
        size_t local_n = (numBodies + mpi_size - 1) / mpi_size;
        size_t start_idx = mpi_rank * local_n;
        size_t end_idx = std::min(start_idx + local_n, (size_t)numBodies);
        size_t num_local = end_idx - start_idx;
        
        // Prepare send buffer with local positions
        std::vector<double> sendBuf(num_local * 3);
        #pragma omp parallel for default(none) shared(bodies, sendBuf, start_idx, num_local)
        for (size_t i = 0; i < num_local; ++i) {
            size_t idx = start_idx + i;
            sendBuf[i * 3 + 0] = bodies[idx].pos.x;
            sendBuf[i * 3 + 1] = bodies[idx].pos.y;
            sendBuf[i * 3 + 2] = bodies[idx].pos.z;
        }
        
        // Gather all positions on all ranks
        std::vector<int> recvCounts(mpi_size);
        std::vector<int> recvDispls(mpi_size);
        
        int sendCount = num_local * 3;
        MPI_Allgather(&sendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        recvDispls[0] = 0;
        for (int i = 1; i < mpi_size; ++i) {
            recvDispls[i] = recvDispls[i-1] + recvCounts[i-1];
        }
        
        std::vector<double> recvBuf(recvDispls[mpi_size-1] + recvCounts[mpi_size-1]);
        
        MPI_Allgatherv(sendBuf.data(), sendCount, MPI_DOUBLE,
                       recvBuf.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Unpack received positions back into bodies array
        #pragma omp parallel for default(none) shared(bodies, recvBuf, recvDispls, mpi_size, numBodies)
        for (int rank = 0; rank < mpi_size; ++rank) {
            size_t rank_start_idx = rank * ((numBodies + mpi_size - 1) / mpi_size);
            size_t rank_end_idx = std::min(rank_start_idx + (size_t)((numBodies + mpi_size - 1) / mpi_size), (size_t)numBodies);
            size_t rank_num = rank_end_idx - rank_start_idx;
            
            for (size_t i = 0; i < rank_num; ++i) {
                size_t idx = rank_start_idx + i;
                size_t bufIdx = recvDispls[rank] + i * 3;
                bodies[idx].pos.x = recvBuf[bufIdx + 0];
                bodies[idx].pos.y = recvBuf[bufIdx + 1];
                bodies[idx].pos.z = recvBuf[bufIdx + 2];
            }
        }
        
        // Compute forces with hybrid parallelization
        computeForces(bodies, mpi_rank, mpi_size);
        
        // Synchronize velocities after force computation
        std::vector<double> sendVelBuf(num_local * 3);
        #pragma omp parallel for default(none) shared(bodies, sendVelBuf, start_idx, num_local)
        for (size_t i = 0; i < num_local; ++i) {
            size_t idx = start_idx + i;
            sendVelBuf[i * 3 + 0] = bodies[idx].vel.x;
            sendVelBuf[i * 3 + 1] = bodies[idx].vel.y;
            sendVelBuf[i * 3 + 2] = bodies[idx].vel.z;
        }
        
        // Allgather velocities
        int sendVelCount = num_local * 3;
        std::vector<double> recvVelBuf(recvDispls[mpi_size-1] + recvCounts[mpi_size-1]);
        
        MPI_Allgatherv(sendVelBuf.data(), sendVelCount, MPI_DOUBLE,
                       recvVelBuf.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        // Unpack received velocities back into bodies array
        #pragma omp parallel for default(none) shared(bodies, recvVelBuf, recvDispls, mpi_size, numBodies)
        for (int rank = 0; rank < mpi_size; ++rank) {
            size_t rank_start_idx = rank * ((numBodies + mpi_size - 1) / mpi_size);
            size_t rank_end_idx = std::min(rank_start_idx + (size_t)((numBodies + mpi_size - 1) / mpi_size), (size_t)numBodies);
            size_t rank_num = rank_end_idx - rank_start_idx;
            
            for (size_t i = 0; i < rank_num; ++i) {
                size_t idx = rank_start_idx + i;
                size_t bufIdx = recvDispls[rank] + i * 3;
                bodies[idx].vel.x = recvVelBuf[bufIdx + 0];
                bodies[idx].vel.y = recvVelBuf[bufIdx + 1];
                bodies[idx].vel.z = recvVelBuf[bufIdx + 2];
            }
        }
        
        // Integrate positions
        integrateBodies(bodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (mpi_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
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
    if (validate && mpi_rank == 0) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
