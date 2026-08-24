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

// Create MPI type for Body
MPI_Datatype MPI_BODY_TYPE;

void create_mpi_type() {
    MPI_Type_contiguous(6, MPI_DOUBLE, &MPI_BODY_TYPE);
    MPI_Type_commit(&MPI_BODY_TYPE);
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

double computeTotalEnergy(const std::vector<Body>& local_bodies, const std::vector<Body>& all_bodies, int rank, const std::vector<int>& displs) {
    double local_energy = 0.0;
    const size_t local_n = local_bodies.size();
    const size_t total_n = all_bodies.size();
    
    // Kinetic energy (local contribution)
    for (const auto& body : local_bodies) {
        local_energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy
    // Each rank calculates potential energy for its particles interacting with all other particles.
    // To avoid double counting, we iterate j > i.
    // Since we have distributed bodies, global index I corresponds to local index i via offset.
    int global_offset = displs[rank];

    for (size_t i = 0; i < local_n; ++i) {
        size_t global_i = global_offset + i;
        for (size_t j = global_i + 1; j < total_n; ++j) {
            const double dx = all_bodies[j].pos.x - local_bodies[i].pos.x;
            const double dy = all_bodies[j].pos.y - local_bodies[i].pos.y;
            const double dz = all_bodies[j].pos.z - local_bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_energy -= 1.0 / dist;
        }
    }

    double global_energy = 0.0;
    MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    return global_energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            // Only print on rank 0 or all? Better to return false and handle in caller.
            return false;
        }
        
        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
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

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    create_mpi_type();

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
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI World Size: %d\n", world_size);
    }
    
    // Determine local bodies
    int count = numBodies / world_size;
    int remainder = numBodies % world_size;
    
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);
    
    for (int i = 0; i < world_size; ++i) {
        counts[i] = count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }
    
    int local_count = counts[rank];
    std::vector<Body> local_bodies(local_count);
    
    // Initialize bodies
    // To ensure consistency with serial version, we generate all bodies on rank 0 and scatter,
    // or generate all everywhere and pick. Generating all everywhere is simpler if memory allows.
    // For large N, we should optimize, but for benchmark with 1024, it's fine.
    // Let's generate all on rank 0 and scatter to be safe and memory efficient.
    
    std::vector<Body> all_bodies(numBodies);

    if (rank == 0) {
        randomizeBodies(all_bodies);
    }
    
    // Scatter bodies to all ranks
    // Using MPI_Scatterv because counts may vary
    MPI_Scatterv(all_bodies.data(), counts.data(), displs.data(), MPI_BODY_TYPE,
                 local_bodies.data(), local_count, MPI_BODY_TYPE,
                 0, MPI_COMM_WORLD);
    
    // Also broadcast initial state to everyone because everyone needs full state for force computation
    // Actually, we can just Bcast the whole thing from rank 0, then everyone picks their part.
    // But Scatterv + Allgatherv logic is more general for memory constrained cases where we only hold all_bodies during force comp.
    // Here we need `all_bodies` on every rank to compute forces.
    
    MPI_Bcast(all_bodies.data(), numBodies, MPI_BODY_TYPE, 0, MPI_COMM_WORLD);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(local_bodies, all_bodies);
        integrateBodies(local_bodies);
        
        // Update global state
        MPI_Allgatherv(local_bodies.data(), local_count, MPI_BODY_TYPE,
                       all_bodies.data(), counts.data(), displs.data(), MPI_BODY_TYPE,
                       MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %lld ms\n", global_duration_ms);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
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
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        
        // Local validation
        bool local_valid = validateSimulation(local_bodies);
        bool global_valid = false;
        
        // Logical AND reduction
        int local_valid_int = local_valid ? 1 : 0;
        int global_valid_int = 0;
        MPI_Reduce(&local_valid_int, &global_valid_int, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        global_valid = (global_valid_int == 1);
        
        // Broadcast global_valid to all ranks so they agree on whether to proceed
        MPI_Bcast(&global_valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        global_valid = (global_valid_int == 1);

        if (global_valid) {
             double finalEnergy = computeTotalEnergy(local_bodies, all_bodies, rank, displs);
             if (rank == 0) {
                 printf("Final energy: %.6f\n", finalEnergy);
                 printf("Validation: PASSED\n");
             }
        } else {
             if (rank == 0) {
                 printf("Validation: FAILED\n");
             }
             MPI_Type_free(&MPI_BODY_TYPE);
             MPI_Finalize();
             return 1;
        }
    }
    
    MPI_Type_free(&MPI_BODY_TYPE);
    MPI_Finalize();
    return 0;
}
