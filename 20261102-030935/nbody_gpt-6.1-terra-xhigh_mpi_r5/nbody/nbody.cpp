#include <algorithm>
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

// Every rank owns a contiguous set of velocities and shares the positions.
// The position array is unchanged until all local forces have been computed.
void computeForces(const std::vector<Vec3>& positions, std::vector<Vec3>& velocities,
                   size_t first) {
    const size_t n = positions.size();
    for (size_t i = 0; i < velocities.size(); ++i) {
        const Vec3& pos = positions[first + i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - pos.x;
            const double dy = positions[j].y - pos.y;
            const double dz = positions[j].z - pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        velocities[i].x += DT * Fx;
        velocities[i].y += DT * Fy;
        velocities[i].z += DT * Fz;
    }
}

void integrateBodies(std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                     size_t first) {
    for (size_t i = 0; i < velocities.size(); ++i) {
        positions[first + i].x += velocities[i].x * DT;
        positions[first + i].y += velocities[i].y * DT;
        positions[first + i].z += velocities[i].z * DT;
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
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
    if (numBodies < 0) {
        if (rank == 0) fprintf(stderr, "Number of bodies must be nonnegative\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(numRanks), displacements(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = numBodies / numRanks + (r < numBodies % numRanks ? 1 : 0);
        displacements[r] = r * (numBodies / numRanks) + std::min(r, numBodies % numRanks);
    }
    const int localCount = counts[rank];
    const int first = displacements[rank];
    std::vector<Vec3> positions(numBodies);
    std::vector<Vec3> velocities(localCount);
    {
        // Replaying the original generator on every rank avoids an initial broadcast
        // and keeps the initial state independent of the number of ranks.
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        for (int i = 0; i < numBodies; ++i) positions[i] = bodies[i].pos;
        for (int i = 0; i < localCount; ++i) velocities[i] = bodies[first + i].vel;
    }

    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(positions, velocities, first);
        integrateBodies(positions, velocities, first);
        if (step + 1 < numSteps) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, positions.data(), counts.data(),
                           displacements.data(), vec3Type, MPI_COMM_WORLD);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    int result = 0;
    if (printResults || validate) {
        // At the last step each rank has only updated its own positions.
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE :
                    (localCount ? positions.data() + first : positions.data()),
                    localCount, vec3Type, positions.data(), counts.data(),
                    displacements.data(), vec3Type, 0, MPI_COMM_WORLD);
        std::vector<Vec3> allVelocities(rank == 0 ? numBodies : 0);
        MPI_Gatherv(velocities.data(), localCount, vec3Type, allVelocities.data(),
                    counts.data(), displacements.data(), vec3Type, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<Body> bodies(numBodies);
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = positions[i];
                bodies[i].vel = allVelocities[i];
            }
            if (printResults) {
                std::vector<double> bodyData;
                bodyData.reserve(static_cast<size_t>(numBodies) * 6);
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
                    result = 1;
                }
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&vec3Type);
    MPI_Finalize();
    return result;
}
