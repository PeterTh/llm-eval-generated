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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
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
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int retcode = 0;
    
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
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Initialize all bodies identically on every rank (deterministic seed)
    std::vector<Body> all_bodies(numBodies);
    randomizeBodies(all_bodies);
    
    // Partition bodies across ranks
    std::vector<int> body_counts(nprocs), body_displs(nprocs);
    const int base_count = numBodies / nprocs;
    const int remainder = numBodies % nprocs;
    for (int r = 0; r < nprocs; r++) {
        body_counts[r] = base_count + (r < remainder ? 1 : 0);
        body_displs[r] = (r == 0) ? 0 : body_displs[r - 1] + body_counts[r - 1];
    }
    
    const int local_start = body_displs[rank];
    const int local_count = body_counts[rank];
    
    // Extract local slice
    std::vector<Body> local_bodies(all_bodies.begin() + local_start,
                                   all_bodies.begin() + local_start + local_count);
    all_bodies.clear();
    
    // Allgatherv parameters for positions (3 doubles per body)
    std::vector<int> pos_counts(nprocs), pos_displs(nprocs);
    for (int r = 0; r < nprocs; r++) {
        pos_counts[r] = body_counts[r] * 3;
        pos_displs[r] = body_displs[r] * 3;
    }
    
    std::vector<double> all_pos(numBodies * 3);
    std::vector<double> local_pos(local_count * 3);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions
        for (int i = 0; i < local_count; i++) {
            local_pos[i * 3 + 0] = local_bodies[i].pos.x;
            local_pos[i * 3 + 1] = local_bodies[i].pos.y;
            local_pos[i * 3 + 2] = local_bodies[i].pos.z;
        }
        
        // Gather all positions from every rank
        MPI_Allgatherv(local_pos.data(), local_count * 3, MPI_DOUBLE,
                       all_pos.data(), pos_counts.data(), pos_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Compute forces on local bodies from all bodies (same j-order as serial)
        for (int i = 0; i < local_count; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double px = local_bodies[i].pos.x;
            const double py = local_bodies[i].pos.y;
            const double pz = local_bodies[i].pos.z;
            
            for (int j = 0; j < numBodies; ++j) {
                const double dx = all_pos[j * 3 + 0] - px;
                const double dy = all_pos[j * 3 + 1] - py;
                const double dz = all_pos[j * 3 + 2] - pz;
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
    long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_duration_ms);
    }
    
    // Gather all bodies to rank 0 for output/validation
    if (printResults || validate) {
        std::vector<double> local_data(local_count * 6);
        for (int i = 0; i < local_count; i++) {
            local_data[i * 6 + 0] = local_bodies[i].pos.x;
            local_data[i * 6 + 1] = local_bodies[i].pos.y;
            local_data[i * 6 + 2] = local_bodies[i].pos.z;
            local_data[i * 6 + 3] = local_bodies[i].vel.x;
            local_data[i * 6 + 4] = local_bodies[i].vel.y;
            local_data[i * 6 + 5] = local_bodies[i].vel.z;
        }
        
        std::vector<int> data_counts(nprocs), data_displs(nprocs);
        for (int r = 0; r < nprocs; r++) {
            data_counts[r] = body_counts[r] * 6;
            data_displs[r] = body_displs[r] * 6;
        }
        
        std::vector<double> all_data;
        if (rank == 0) all_data.resize(numBodies * 6);
        
        MPI_Gatherv(local_data.data(), local_count * 6, MPI_DOUBLE,
                    all_data.data(), data_counts.data(), data_displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::vector<Body> bodies(numBodies);
            for (int i = 0; i < numBodies; i++) {
                bodies[i].pos.x = all_data[i * 6 + 0];
                bodies[i].pos.y = all_data[i * 6 + 1];
                bodies[i].pos.z = all_data[i * 6 + 2];
                bodies[i].vel.x = all_data[i * 6 + 3];
                bodies[i].vel.y = all_data[i * 6 + 4];
                bodies[i].vel.z = all_data[i * 6 + 5];
            }
            
            if (printResults) {
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
            
            if (validate) {
                printf("Validating simulation results...\n");
                if (validateSimulation(bodies)) {
                    double finalEnergy = computeTotalEnergy(bodies);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    retcode = 1;
                }
            }
        }
    }
    
    MPI_Finalize();
    return retcode;
}
