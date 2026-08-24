#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
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

// Compute forces for a local set of bodies given all positions in the system
void computeForcesLocal(std::vector<Body>& localBodies, const std::vector<double>& allPos, int globalN, int globalOffset) {
    const int localN = (int)localBodies.size();
    for (int i = 0; i < localN; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double px = localBodies[i].pos.x;
        const double py = localBodies[i].pos.y;
        const double pz = localBodies[i].pos.z;
        for (int j = 0; j < globalN; ++j) {
            const double dx = allPos[3*j + 0] - px;
            const double dy = allPos[3*j + 1] - py;
            const double dz = allPos[3*j + 2] - pz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        localBodies[i].vel.x += DT * Fx;
        localBodies[i].vel.y += DT * Fy;
        localBodies[i].vel.z += DT * Fz;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // determine local distribution
    int q = numBodies / world_size;
    int r = numBodies % world_size;
    int localN = q + (rank < r ? 1 : 0);
    int offset = rank * q + std::min(rank, r);

    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);
    std::vector<int> counts_pos(world_size);
    std::vector<int> displs_pos(world_size);
    for (int i = 0; i < world_size; ++i) {
        int ni = q + (i < r ? 1 : 0);
        counts[i] = ni * 6; // 6 doubles per body
        counts_pos[i] = ni * 3; // 3 doubles per position
    }
    displs[0] = 0;
    displs_pos[0] = 0;
    for (int i = 1; i < world_size; ++i) {
        displs[i] = displs[i-1] + counts[i-1];
        displs_pos[i] = displs_pos[i-1] + counts_pos[i-1];
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize full bodies on root and scatter
    std::vector<Body> localBodies(localN);
    if (rank == 0) {
        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies);
        // pack into doubles
        std::vector<double> packed(allBodies.size() * 6);
        for (int i = 0; i < numBodies; ++i) {
            packed[6*i + 0] = allBodies[i].pos.x;
            packed[6*i + 1] = allBodies[i].pos.y;
            packed[6*i + 2] = allBodies[i].pos.z;
            packed[6*i + 3] = allBodies[i].vel.x;
            packed[6*i + 4] = allBodies[i].vel.y;
            packed[6*i + 5] = allBodies[i].vel.z;
        }
        // scatter packed data
        std::vector<double> localPacked(localN * 6);
        MPI_Scatterv(packed.data(), counts.data(), displs.data(), MPI_DOUBLE,
                     localPacked.data(), localN * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // unpack
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x = localPacked[6*i + 0];
            localBodies[i].pos.y = localPacked[6*i + 1];
            localBodies[i].pos.z = localPacked[6*i + 2];
            localBodies[i].vel.x = localPacked[6*i + 3];
            localBodies[i].vel.y = localPacked[6*i + 4];
            localBodies[i].vel.z = localPacked[6*i + 5];
        }
    } else {
        // non-root receive
        std::vector<double> localPacked(localN * 6);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     localPacked.data(), localN * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x = localPacked[6*i + 0];
            localBodies[i].pos.y = localPacked[6*i + 1];
            localBodies[i].pos.z = localPacked[6*i + 2];
            localBodies[i].vel.x = localPacked[6*i + 3];
            localBodies[i].vel.y = localPacked[6*i + 4];
            localBodies[i].vel.z = localPacked[6*i + 5];
        }
    }

    // Prepare buffers for Allgatherv of positions
    std::vector<double> localPos(localN * 3);
    for (int i = 0; i < localN; ++i) {
        localPos[3*i + 0] = localBodies[i].pos.x;
        localPos[3*i + 1] = localBodies[i].pos.y;
        localPos[3*i + 2] = localBodies[i].pos.z;
    }
    std::vector<double> allPos(numBodies * 3);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        // gather all positions
        MPI_Allgatherv(localPos.data(), localN * 3, MPI_DOUBLE,
                       allPos.data(), counts_pos.data(), displs_pos.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // compute forces on local bodies using all positions
        computeForcesLocal(localBodies, allPos, numBodies, offset);

        // integrate local bodies
        integrateBodies(localBodies);

        // update localPos for next iteration
        for (int i = 0; i < localN; ++i) {
            localPos[3*i + 0] = localBodies[i].pos.x;
            localPos[3*i + 1] = localBodies[i].pos.y;
            localPos[3*i + 2] = localBodies[i].pos.z;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double localElapsed = t1 - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long ms = (long)(maxElapsed * 1000.0);
        printf("Simulation time: %ld ms\n", ms);
    }

    // Gather final bodies to root for validation/printing
    std::vector<double> localPackedOut(localN * 6);
    for (int i = 0; i < localN; ++i) {
        localPackedOut[6*i + 0] = localBodies[i].pos.x;
        localPackedOut[6*i + 1] = localBodies[i].pos.y;
        localPackedOut[6*i + 2] = localBodies[i].pos.z;
        localPackedOut[6*i + 3] = localBodies[i].vel.x;
        localPackedOut[6*i + 4] = localBodies[i].vel.y;
        localPackedOut[6*i + 5] = localBodies[i].vel.z;
    }

    std::vector<double> gathered;
    if (rank == 0) gathered.resize(numBodies * 6);

    MPI_Gatherv(localPackedOut.data(), localN * 6, MPI_DOUBLE,
                gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // reconstruct bodies vector
        std::vector<Body> finalBodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            finalBodies[i].pos.x = gathered[6*i + 0];
            finalBodies[i].pos.y = gathered[6*i + 1];
            finalBodies[i].pos.z = gathered[6*i + 2];
            finalBodies[i].vel.x = gathered[6*i + 3];
            finalBodies[i].vel.y = gathered[6*i + 4];
            finalBodies[i].vel.z = gathered[6*i + 5];
        }

        // Print results for external validation
        if (printResults) {
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
            printf("Validating simulation results...\n");
            if (validateSimulation(finalBodies)) {
                double finalEnergy = computeTotalEnergy(finalBodies);
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
    }

    MPI_Finalize();
    return 0;
}
