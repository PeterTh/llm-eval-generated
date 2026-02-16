#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"
#include <mpi.h>

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
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numBodies = 1024;
    int numSteps = 10;
    int validate_i = 0;
    int printResults_i = 0;

    // Parse arguments on rank 0 and broadcast
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_i = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_i = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else if (i == 1) {
                // allow unknown options to be handled normally
            } else {
                if (rank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                MPI_Finalize();
                return 1;
            }
        }
    }
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const bool validate = (validate_i != 0);
    const bool printResults = (printResults_i != 0);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute distribution of bodies across ranks
    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    int base = numBodies / worldSize;
    int rem = numBodies % worldSize;
    int offset = 0;
    for (int i = 0; i < worldSize; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
    const int localN = counts[rank];

    // Root initializes all bodies and scatters
    std::vector<Body> localBodies(localN);
    std::vector<double> sendbuf;
    if (rank == 0) {
        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies);
        sendbuf.reserve(numBodies * 6);
        for (const auto& b : allBodies) {
            sendbuf.push_back(b.pos.x);
            sendbuf.push_back(b.pos.y);
            sendbuf.push_back(b.pos.z);
            sendbuf.push_back(b.vel.x);
            sendbuf.push_back(b.vel.y);
            sendbuf.push_back(b.vel.z);
        }
    }

    std::vector<int> sendcounts_doubles(worldSize);
    std::vector<int> displs_doubles(worldSize);
    for (int i = 0; i < worldSize; ++i) {
        sendcounts_doubles[i] = counts[i] * 6;
        displs_doubles[i] = displs[i] * 6;
    }

    std::vector<double> recvbuf(localN * 6);
    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, sendcounts_doubles.data(), displs_doubles.data(), MPI_DOUBLE,
                 recvbuf.data(), localN * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Unpack local bodies
    for (int i = 0; i < localN; ++i) {
        const int idx = i * 6;
        localBodies[i].pos.x = recvbuf[idx + 0];
        localBodies[i].pos.y = recvbuf[idx + 1];
        localBodies[i].pos.z = recvbuf[idx + 2];
        localBodies[i].vel.x = recvbuf[idx + 3];
        localBodies[i].vel.y = recvbuf[idx + 4];
        localBodies[i].vel.z = recvbuf[idx + 5];
    }

    // Prepare position gather parameters (3 doubles per body)
    std::vector<int> pos_counts(worldSize), pos_displs(worldSize);
    for (int i = 0; i < worldSize; ++i) {
        pos_counts[i] = counts[i] * 3;
        pos_displs[i] = displs[i] * 3;
    }
    std::vector<double> allPositions(numBodies * 3);
    std::vector<double> localPositions(localN * 3);

    // Synchronize and time the simulation on rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Pack local positions
        for (int i = 0; i < localN; ++i) {
            localPositions[3*i + 0] = localBodies[i].pos.x;
            localPositions[3*i + 1] = localBodies[i].pos.y;
            localPositions[3*i + 2] = localBodies[i].pos.z;
        }

        // Gather all positions
        MPI_Allgatherv(localPositions.data(), localN * 3, MPI_DOUBLE,
                       allPositions.data(), pos_counts.data(), pos_displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute forces on local bodies using allPositions
        for (int i = 0; i < localN; ++i) {
            double Fx = 0.0, Fy = 0.0, Fz = 0.0;
            const double ix = localBodies[i].pos.x;
            const double iy = localBodies[i].pos.y;
            const double iz = localBodies[i].pos.z;
            for (int j = 0; j < numBodies; ++j) {
                const double jx = allPositions[3*j + 0];
                const double jy = allPositions[3*j + 1];
                const double jz = allPositions[3*j + 2];
                const double dx = jx - ix;
                const double dy = jy - iy;
                const double dz = jz - iz;
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

        // Integrate local bodies
        for (int i = 0; i < localN; ++i) {
            localBodies[i].pos.x += localBodies[i].vel.x * DT;
            localBodies[i].pos.y += localBodies[i].vel.y * DT;
            localBodies[i].pos.z += localBodies[i].vel.z * DT;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Gather final bodies back to root for printing or validation
    std::vector<double> finalRecv;
    if (rank == 0) finalRecv.resize(numBodies * 6);
    for (int i = 0; i < localN; ++i) {
        const int idx = i * 6;
        recvbuf[idx + 0] = localBodies[i].pos.x;
        recvbuf[idx + 1] = localBodies[i].pos.y;
        recvbuf[idx + 2] = localBodies[i].pos.z;
        recvbuf[idx + 3] = localBodies[i].vel.x;
        recvbuf[idx + 4] = localBodies[i].vel.y;
        recvbuf[idx + 5] = localBodies[i].vel.z;
    }

    MPI_Gatherv(recvbuf.data(), localN * 6, MPI_DOUBLE,
                rank == 0 ? finalRecv.data() : nullptr, sendcounts_doubles.data(), displs_doubles.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (printResults && rank == 0) {
        // finalRecv already in the required format
        print_results(finalRecv, "Bodies");
    }

    if (validate && rank == 0) {
        // Reconstruct Body vector for validation and energy computation
        std::vector<Body> allBodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            const int idx = i * 6;
            allBodies[i].pos.x = finalRecv[idx + 0];
            allBodies[i].pos.y = finalRecv[idx + 1];
            allBodies[i].pos.z = finalRecv[idx + 2];
            allBodies[i].vel.x = finalRecv[idx + 3];
            allBodies[i].vel.y = finalRecv[idx + 4];
            allBodies[i].vel.z = finalRecv[idx + 5];
        }

        printf("Validating simulation results...\n");
        if (validateSimulation(allBodies)) {
            double finalEnergy = computeTotalEnergy(allBodies);
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
