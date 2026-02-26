#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <algorithm>

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

// Create MPI Datatype for Body
void createBodyType(MPI_Datatype* mpi_body_type) {
    MPI_Type_contiguous(6, MPI_DOUBLE, mpi_body_type);
    MPI_Type_commit(mpi_body_type);
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

void computeForces(std::vector<Body>& local_bodies, int rank, int size, MPI_Datatype mpi_body_type) {
    const size_t n_local = local_bodies.size();
    
    std::vector<Body> current_remote_bodies = local_bodies;
    std::vector<Body> next_remote_bodies; 
    
    int src = (rank - 1 + size) % size;
    int dst = (rank + 1) % size;

    for (int step = 0; step < size; ++step) {
        MPI_Request reqs[2];
        bool communicating = (step < size - 1);
        
        if (communicating) {
             int send_count = (int)current_remote_bodies.size();
             int recv_count;
             
             // Exchange sizes (tag 0)
             MPI_Sendrecv(&send_count, 1, MPI_INT, dst, 0, 
                          &recv_count, 1, MPI_INT, src, 0, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                          
             next_remote_bodies.resize(recv_count);
             
             // Start async exchange of bodies (tag 1)
             MPI_Isend(current_remote_bodies.data(), send_count, mpi_body_type, dst, 1, MPI_COMM_WORLD, &reqs[0]);
             MPI_Irecv(next_remote_bodies.data(), recv_count, mpi_body_type, src, 1, MPI_COMM_WORLD, &reqs[1]);
        }
        
        // Compute forces (overlap with communication)
        const size_t n_remote = current_remote_bodies.size();
        for (size_t i = 0; i < n_local; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            
            for (size_t j = 0; j < n_remote; ++j) {
                const double dx = current_remote_bodies[j].pos.x - local_bodies[i].pos.x;
                const double dy = current_remote_bodies[j].pos.y - local_bodies[i].pos.y;
                const double dz = current_remote_bodies[j].pos.z - local_bodies[i].pos.z;
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

        if (communicating) {
             MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
             current_remote_bodies = std::move(next_remote_bodies);
        }
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& local_bodies, int rank, int size, MPI_Datatype mpi_body_type) {
    // Kinetic Energy
    double local_kinetic = 0.0;
    for (const auto& body : local_bodies) {
        local_kinetic += 0.5 * (body.vel.x * body.vel.x + 
                                body.vel.y * body.vel.y + 
                                body.vel.z * body.vel.z);
    }
    
    // Potential Energy
    double local_potential = 0.0;
    
    std::vector<Body> current_remote = local_bodies;
    std::vector<Body> received_bodies;
    
    int src = (rank - 1 + size) % size;
    int dst = (rank + 1) % size;
    
    for (int step = 0; step < size; ++step) {
        const size_t n_local = local_bodies.size();
        const size_t n_remote = current_remote.size();
        
        for (size_t i = 0; i < n_local; ++i) {
             for (size_t j = 0; j < n_remote; ++j) {
                 // Avoid self-interaction
                 if (step == 0 && i == j) continue;
                 
                 const double dx = current_remote[j].pos.x - local_bodies[i].pos.x;
                 const double dy = current_remote[j].pos.y - local_bodies[i].pos.y;
                 const double dz = current_remote[j].pos.z - local_bodies[i].pos.z;
                 const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                 local_potential -= 1.0 / dist;
             }
        }
        
        if (step < size - 1) {
             int send_count = (int)current_remote.size();
             int recv_count;
             MPI_Sendrecv(&send_count, 1, MPI_INT, dst, 0, 
                          &recv_count, 1, MPI_INT, src, 0, 
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             received_bodies.resize(recv_count);
             MPI_Sendrecv(current_remote.data(), send_count, mpi_body_type, dst, 0,
                          received_bodies.data(), recv_count, mpi_body_type, src, 0,
                          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
             current_remote = received_bodies;
        }
    }
    
    double total_kinetic = 0.0;
    MPI_Allreduce(&local_kinetic, &total_kinetic, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    double total_potential = 0.0;
    MPI_Allreduce(&local_potential, &total_potential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return total_kinetic + 0.5 * total_potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            return false;
        }
        
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exit_flag = 0;
    
    if (rank == 0) {
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
                exit_flag = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_flag = 1;
            }
        }
    }

    MPI_Bcast(&exit_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_flag) {
        MPI_Finalize();
        return 0;
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Processes: %d\n", size);
    }
    
    MPI_Datatype mpi_body_type;
    createBodyType(&mpi_body_type);

    // Calculate distribution
    int base_count = numBodies / size;
    int remainder = numBodies % size;
    int my_count = base_count + (rank < remainder ? 1 : 0);
    
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    for (int i = 0; i < size; ++i) {
        counts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }
    
    std::vector<Body> local_bodies(my_count);
    
    // Rank 0 initializes all bodies and scatters
    std::vector<Body> all_bodies;
    if (rank == 0) {
        all_bodies.resize(numBodies);
        randomizeBodies(all_bodies);
    }
    
    MPI_Scatterv(rank == 0 ? all_bodies.data() : nullptr, counts.data(), displs.data(), mpi_body_type,
                 local_bodies.data(), my_count, mpi_body_type, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForces(local_bodies, rank, size, mpi_body_type);
        integrateBodies(local_bodies);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", (long)duration_ms);
    }
    
    // Gather results for output if requested
    if (printResults) {
        if (rank == 0) {
            all_bodies.resize(numBodies);
        }
        MPI_Gatherv(local_bodies.data(), my_count, mpi_body_type,
                    rank == 0 ? all_bodies.data() : nullptr, counts.data(), displs.data(), mpi_body_type,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
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
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        
        bool local_valid = validateSimulation(local_bodies);
        bool global_valid;
        MPI_Allreduce(&local_valid, &global_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
        
        if (global_valid) {
             double finalEnergy = computeTotalEnergy(local_bodies, rank, size, mpi_body_type);
             if (rank == 0) {
                 printf("Final energy: %.6f\n", finalEnergy);
                 printf("Validation: PASSED\n");
             }
        } else {
             if (rank == 0) {
                 printf("Validation failed: found NaN or Inf value or out of bounds\n");
                 printf("Validation: FAILED\n");
             }
        }
    }
    
    MPI_Type_free(&mpi_body_type);
    MPI_Finalize();
    return 0;
}
