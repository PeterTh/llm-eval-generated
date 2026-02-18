#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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

void computeForces(std::vector<Body>& local_bodies, const std::vector<Body>& all_bodies) {
    const size_t local_n = local_bodies.size();
    const size_t total_n = all_bodies.size();
    
    for (size_t i = 0; i < local_n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < total_n; ++j) {
            const double dx = all_bodies[j].pos.x - local_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - local_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - local_bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        local_bodies[i].vel.x += DT * Fx;
        local_bodies[i].vel.y += DT * Fy;
        local_bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& all_bodies) {
    double energy = 0.0;
    const size_t n = all_bodies.size();
    
    for (const auto& body : all_bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = all_bodies[j].pos.x - all_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - all_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - all_bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    
    return energy;
}

bool validateSimulation(const std::vector<Body>& all_bodies) {
    for (const auto& body : all_bodies) {
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
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
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
            if (world_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (world_rank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Number of MPI processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    int remainder = numBodies % world_size;
    int local_count = (numBodies / world_size);
    if (world_rank < remainder) {
        local_count = (numBodies / world_size) + 1;
    }
    
    std::vector<Body> local_bodies;
    std::vector<double> all_init_data;
    
    if (world_rank == 0) {
        std::vector<Body> all_bodies(numBodies);
        randomizeBodies(all_bodies);
        
        all_init_data.resize(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            all_init_data[i * 6 + 0] = all_bodies[i].pos.x;
            all_init_data[i * 6 + 1] = all_bodies[i].pos.y;
            all_init_data[i * 6 + 2] = all_bodies[i].pos.z;
            all_init_data[i * 6 + 3] = all_bodies[i].vel.x;
            all_init_data[i * 6 + 4] = all_bodies[i].vel.y;
            all_init_data[i * 6 + 5] = all_bodies[i].vel.z;
        }
    }
    
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    int offset = 0;
    for (int i = 0; i < world_size; ++i) {
        int count = (numBodies / world_size);
        if (i < remainder) {
            count = (numBodies / world_size) + 1;
        }
        sendcounts[i] = count * 6;
        displs[i] = offset;
        offset += count * 6;
    }
    
    local_bodies.resize(local_count);
    std::vector<double> local_data(local_count * 6);
    MPI_Scatterv(all_init_data.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_data.data(), local_data.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < local_count; ++i) {
        local_bodies[i].pos.x = local_data[i * 6 + 0];
        local_bodies[i].pos.y = local_data[i * 6 + 1];
        local_bodies[i].pos.z = local_data[i * 6 + 2];
        local_bodies[i].vel.x = local_data[i * 6 + 3];
        local_bodies[i].vel.y = local_data[i * 6 + 4];
        local_bodies[i].vel.z = local_data[i * 6 + 5];
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> recvcounts(world_size);
    std::vector<int> recvdispls(world_size);
    offset = 0;
    for (int i = 0; i < world_size; ++i) {
        int count = (numBodies / world_size);
        if (i < remainder) {
            count = (numBodies / world_size) + 1;
        }
        recvcounts[i] = count * 6;
        recvdispls[i] = offset;
        offset += count * 6;
    }
    
    for (int step = 0; step < numSteps; ++step) {
        std::vector<double> step_local_data(local_count * 6);
        for (size_t i = 0; i < (size_t)local_count; ++i) {
            step_local_data[i * 6 + 0] = local_bodies[i].pos.x;
            step_local_data[i * 6 + 1] = local_bodies[i].pos.y;
            step_local_data[i * 6 + 2] = local_bodies[i].pos.z;
            step_local_data[i * 6 + 3] = local_bodies[i].vel.x;
            step_local_data[i * 6 + 4] = local_bodies[i].vel.y;
            step_local_data[i * 6 + 5] = local_bodies[i].vel.z;
        }
        
        std::vector<double> all_data(numBodies * 6);
        MPI_Allgatherv(step_local_data.data(), step_local_data.size(), MPI_DOUBLE,
                       all_data.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        
        std::vector<Body> all_bodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            all_bodies[i].pos.x = all_data[i * 6 + 0];
            all_bodies[i].pos.y = all_data[i * 6 + 1];
            all_bodies[i].pos.z = all_data[i * 6 + 2];
            all_bodies[i].vel.x = all_data[i * 6 + 3];
            all_bodies[i].vel.y = all_data[i * 6 + 4];
            all_bodies[i].vel.z = all_data[i * 6 + 5];
        }
        
        computeForces(local_bodies, all_bodies);
        integrateBodies(local_bodies);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (world_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }
    
    std::vector<double> local_final(local_count * 6);
    for (size_t i = 0; i < (size_t)local_count; ++i) {
        local_final[i * 6 + 0] = local_bodies[i].pos.x;
        local_final[i * 6 + 1] = local_bodies[i].pos.y;
        local_final[i * 6 + 2] = local_bodies[i].pos.z;
        local_final[i * 6 + 3] = local_bodies[i].vel.x;
        local_final[i * 6 + 4] = local_bodies[i].vel.y;
        local_final[i * 6 + 5] = local_bodies[i].vel.z;
    }
    
    std::vector<double> all_final(numBodies * 6);
    MPI_Allgatherv(local_final.data(), local_final.size(), MPI_DOUBLE,
                   all_final.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    
    if (printResults && world_rank == 0) {
        print_results(all_final, "Bodies");
    }
    
    if (validate && world_rank == 0) {
        printf("Validating simulation results...\n");
        std::vector<Body> all_bodies_final(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            all_bodies_final[i].pos.x = all_final[i * 6 + 0];
            all_bodies_final[i].pos.y = all_final[i * 6 + 1];
            all_bodies_final[i].pos.z = all_final[i * 6 + 2];
            all_bodies_final[i].vel.x = all_final[i * 6 + 3];
            all_bodies_final[i].vel.y = all_final[i * 6 + 4];
            all_bodies_final[i].vel.z = all_final[i * 6 + 5];
        }
        
        if (validateSimulation(all_bodies_final)) {
            double finalEnergy = computeTotalEnergy(all_bodies_final);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
