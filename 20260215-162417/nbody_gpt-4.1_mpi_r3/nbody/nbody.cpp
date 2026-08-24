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
    const size_t n = all_bodies.size();
    for (size_t i = 0; i < local_bodies.size(); ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
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

void integrateBodies(std::vector<Body>& local_bodies) {
    for (auto& body : local_bodies) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition bodies among ranks
    int local_n = numBodies / size;
    int remainder = numBodies % size;
    int local_start = rank * local_n + (rank < remainder ? rank : remainder);
    int local_count = local_n + (rank < remainder ? 1 : 0);

    std::vector<Body> local_bodies(local_count);
    std::vector<Body> all_bodies(numBodies);
    static_assert(sizeof(Body) % sizeof(double) == 0, "Body must be a multiple of double for MPI");

    if (rank == 0) {
        std::vector<Body> tmp_bodies(numBodies);
        randomizeBodies(tmp_bodies);
        // Scatter initial bodies
        for (int r = 0; r < size; ++r) {
            int start = r * local_n + (r < remainder ? r : remainder);
            int count = local_n + (r < remainder ? 1 : 0);
            if (r == 0) {
                std::copy(tmp_bodies.begin(), tmp_bodies.begin() + count, local_bodies.begin());
            } else {
                MPI_Send(tmp_bodies.data() + start, count * sizeof(Body), MPI_BYTE, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(local_bodies.data(), local_count * sizeof(Body), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Simulation
    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        // Gather all bodies' positions to all ranks
        // Gather all bodies' positions to all ranks
        std::vector<int> counts(size), displs(size);
        for (int r = 0, disp = 0; r < size; ++r) {
            counts[r] = local_n + (r < remainder ? 1 : 0);
            displs[r] = disp;
            disp += counts[r];
        }
        for (int r = 0; r < size; ++r) {
            counts[r] = (local_n + (r < remainder ? 1 : 0)) * sizeof(Body);
            displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
        }
        int body_doubles = sizeof(Body) / sizeof(double);
        for (int r = 0; r < size; ++r) {
            counts[r] = (local_n + (r < remainder ? 1 : 0)) * body_doubles;
            displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
        }
        MPI_Allgatherv(reinterpret_cast<double*>(local_bodies.data()), local_count * body_doubles, MPI_DOUBLE,
                      reinterpret_cast<double*>(all_bodies.data()), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        computeForces(local_bodies, all_bodies);
        integrateBodies(local_bodies);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", global_duration_ms);
    }

    // Gather all bodies to rank 0 for output/validation
    if (printResults || validate) {
        std::vector<Body> gathered_bodies;
        if (rank == 0) gathered_bodies.resize(numBodies);
        std::vector<int> counts(size), displs(size);
        for (int r = 0, disp = 0; r < size; ++r) {
            counts[r] = (local_n + (r < remainder ? 1 : 0)) * sizeof(Body);
            displs[r] = disp;
            disp += counts[r];
        }
        for (int r = 0; r < size; ++r) {
            counts[r] = (local_n + (r < remainder ? 1 : 0)) * sizeof(Body);
            displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
        }
        int body_doubles = sizeof(Body) / sizeof(double);
        for (int r = 0; r < size; ++r) {
            counts[r] = (local_n + (r < remainder ? 1 : 0)) * body_doubles;
            displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
        }
        MPI_Gatherv(reinterpret_cast<double*>(local_bodies.data()), local_count * body_doubles, MPI_DOUBLE,
                    rank == 0 ? reinterpret_cast<double*>(gathered_bodies.data()) : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) {
                std::vector<double> bodyData;
                bodyData.reserve(numBodies * 6);
                for (const auto& body : gathered_bodies) {
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
                if (validateSimulation(gathered_bodies)) {
                    double finalEnergy = computeTotalEnergy(gathered_bodies);
                    printf("Final energy: %.6f\n", finalEnergy);
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }
    }
    MPI_Finalize();
    return 0;
}
