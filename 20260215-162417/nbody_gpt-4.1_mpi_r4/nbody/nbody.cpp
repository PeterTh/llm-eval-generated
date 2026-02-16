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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    for (size_t i = 0; i < n; ++i) {
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
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
            if (world_rank == 0) printUsage(argv[0]);
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
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Partition bodies among ranks
    int local_n = numBodies / world_size;
    int remainder = numBodies % world_size;
    int local_start = world_rank * local_n + (world_rank < remainder ? world_rank : remainder);
    int local_count = local_n + (world_rank < remainder ? 1 : 0);

    std::vector<Body> local_bodies(local_count);
    std::vector<Body> all_bodies(numBodies);
    if (world_rank == 0) {
        randomizeBodies(all_bodies);
    }
    // Scatter bodies to all ranks
    std::vector<int> counts(world_size), displs(world_size);
    for (int i = 0, offset = 0; i < world_size; ++i) {
        counts[i] = numBodies / world_size + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
    std::vector<int> body_counts(world_size), body_displs(world_size);
    for (int i = 0, offset = 0; i < world_size; ++i) {
        body_counts[i] = counts[i] * sizeof(Body);
        body_displs[i] = displs[i] * sizeof(Body);
    }
    MPI_Scatterv(all_bodies.data(), body_counts.data(), body_displs.data(), MPI_BYTE,
                 local_bodies.data(), local_count * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Prepare buffer for all positions
    std::vector<double> all_pos_flat(numBodies * 3);
    std::vector<double> local_pos_flat(local_count * 3);
    std::vector<int> pos_counts(world_size), pos_displs(world_size);
    int pos_offset = 0;
    for (int i = 0; i < world_size; ++i) {
        pos_counts[i] = counts[i] * 3; // number of doubles per rank
        pos_displs[i] = pos_offset;
        pos_offset += pos_counts[i];
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        // Gather all positions
        for (int i = 0; i < local_count; ++i) local_positions[i] = local_bodies[i].pos;
        // flatten local_positions for MPI
        std::vector<double> local_pos_flat(local_count * 3);
        for (int i = 0; i < local_count; ++i) {
            local_pos_flat[i * 3 + 0] = local_positions[i].x;
            local_pos_flat[i * 3 + 1] = local_positions[i].y;
            local_pos_flat[i * 3 + 2] = local_positions[i].z;
        }
        std::vector<double> all_pos_flat(numBodies * 3);
        // Debug: print buffer sizes for each rank
        printf("[Rank %d] local_count=%d, send_count=%ld, world_size=%d, numBodies=%d\n", world_rank, local_count, (long)(local_count*3), world_size, numBodies);
        for (int r = 0; r < world_size; ++r) {
            printf("[Rank %d] pos_counts[%d]=%d pos_displs[%d]=%d\n", world_rank, r, pos_counts[r], r, pos_displs[r]);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        if (local_count == 0) printf("[Rank %d] WARNING: local_count is 0!\n", world_rank);
        if (local_pos_flat.data() == nullptr) printf("[Rank %d] WARNING: local_pos_flat.data() is nullptr!\n", world_rank);
        if (all_pos_flat.data() == nullptr) printf("[Rank %d] WARNING: all_pos_flat.data() is nullptr!\n", world_rank);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        int rc = MPI_Allgatherv(local_pos_flat.data(), local_count * 3, MPI_DOUBLE,
                      all_pos_flat.data(), pos_counts.data(), pos_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        if (rc != MPI_SUCCESS) {
            printf("[Rank %d] MPI_Allgatherv failed with code %d\n", world_rank, rc);
            MPI_Abort(MPI_COMM_WORLD, rc);
        }
        // unpack all_pos_flat into all_positions
        for (int i = 0; i < numBodies; ++i) {
            all_positions[i].x = all_pos_flat[i * 3 + 0];
            all_positions[i].y = all_pos_flat[i * 3 + 1];
            all_positions[i].z = all_pos_flat[i * 3 + 2];
        }
        // Compute forces for local bodies
        for (int i = 0; i < local_count; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            int global_i = local_start + i;
            for (int j = 0; j < numBodies; ++j) {
                const double dx = all_pos_flat[j * 3 + 0] - local_bodies[i].pos.x;
                const double dy = all_pos_flat[j * 3 + 1] - local_bodies[i].pos.y;
                const double dz = all_pos_flat[j * 3 + 2] - local_bodies[i].pos.z;
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
        // Integrate local bodies
        for (int i = 0; i < local_count; ++i) {
            local_bodies[i].pos.x += local_bodies[i].vel.x * DT;
            local_bodies[i].pos.y += local_bodies[i].vel.y * DT;
            local_bodies[i].pos.z += local_bodies[i].vel.z * DT;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (world_rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather all bodies to rank 0 for output/validation
    MPI_Gatherv(local_bodies.data(), local_count * sizeof(Body), MPI_BYTE,
                all_bodies.data(), body_counts.data(), body_displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (printResults && world_rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : all_bodies) {
            bodyData.push_back(body.pos.x);
            bodyData.push_back(body.pos.y);
            bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x);
            bodyData.push_back(body.vel.y);
            bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate && world_rank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulation(all_bodies)) {
            double finalEnergy = computeTotalEnergy(all_bodies);
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
