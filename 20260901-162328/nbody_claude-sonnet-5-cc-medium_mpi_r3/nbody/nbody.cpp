#include <chrono>
#include <cmath>
#include <cstddef>
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

// Compute forces for the local range [localStart, localStart + localCount) of
// bodies, using the globally replicated `bodies` array, updating only the
// locally owned velocities in place.
void computeForces(std::vector<Body>& bodies, int localStart, int localCount) {
    const size_t n = bodies.size();

    for (int li = 0; li < localCount; ++li) {
        const size_t i = static_cast<size_t>(localStart + li);
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

// Integrate the local range of bodies in place (only the locally owned
// bodies' positions are updated).
void integrateBodies(std::vector<Body>& bodies, int localStart, int localCount) {
    for (int li = 0; li < localCount; ++li) {
        Body& body = bodies[static_cast<size_t>(localStart + li)];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

// Compute the total energy (kinetic + potential) across all ranks. Each rank
// contributes the kinetic energy of its local bodies and the potential energy
// of pairs (i, j) with i in the local range and j > i; the partial sums are
// then reduced across all ranks so every process ends up with the same total.
double computeTotalEnergy(const std::vector<Body>& bodies, int localStart, int localCount, MPI_Comm comm) {
    double localEnergy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass) for locally owned bodies
    for (int li = 0; li < localCount; ++li) {
        const Body& body = bodies[static_cast<size_t>(localStart + li)];
        localEnergy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), partitioned by i
    for (int li = 0; li < localCount; ++li) {
        const size_t i = static_cast<size_t>(localStart + li);
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            localEnergy -= 1.0 / dist;
        }
    }

    double totalEnergy = 0.0;
    MPI_Allreduce(&localEnergy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, comm);
    return totalEnergy;
}

// Validate that the locally owned bodies produce finite, reasonable values
bool validateSimulationLocal(const std::vector<Body>& bodies, int localStart, int localCount) {
    for (int li = 0; li < localCount; ++li) {
        const Body& body = bodies[static_cast<size_t>(localStart + li)];

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

// Compute a block distribution of `numBodies` across `numRanks` ranks,
// filling in per-rank counts and displacements (in number of bodies).
void computeDistribution(int numBodies, int numRanks, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(numRanks);
    displs.resize(numRanks);
    const int base = numBodies / numRanks;
    const int rem = numBodies % numRanks;
    int offset = 0;
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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

    // Determine this rank's slice of the bodies (block distribution)
    std::vector<int> counts, displs;
    computeDistribution(numBodies, worldSize, counts, displs);
    const int localStart = displs[rank];
    const int localCount = counts[rank];

    // Derived MPI datatype describing one Body's `pos` field (3 contiguous
    // doubles), resized so consecutive elements stride across the array by
    // sizeof(Body). This lets us gather/broadcast only positions in place
    // inside the interleaved Body array, matching the original data layout
    // (and therefore its floating-point evaluation order) exactly.
    MPI_Datatype vec3Contig, posType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Contig);
    MPI_Type_create_resized(vec3Contig, 0, sizeof(Body), &posType);
    MPI_Type_commit(&posType);
    MPI_Type_free(&vec3Contig);

    // Initialize bodies identically to the serial version: generate the full
    // set on rank 0 (matching the original single-seed RNG sequence exactly),
    // then broadcast the whole array so every rank has identical initial state.
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), numBodies * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, localStart, localCount);
        integrateBodies(bodies, localStart, localCount);

        // Synchronize the updated positions across all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                        bodies.data(), counts.data(), displs.data(), posType,
                        MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Print results for external validation
    if (printResults) {
        // Gather velocities (positions are already globally in sync)
        MPI_Datatype vec3ContigV, velType;
        MPI_Type_contiguous(3, MPI_DOUBLE, &vec3ContigV);
        MPI_Type_create_resized(vec3ContigV, 0, sizeof(Body), &velType);
        MPI_Type_commit(&velType);
        MPI_Type_free(&vec3ContigV);

        char* velBase = reinterpret_cast<char*>(bodies.data()) + offsetof(Body, vel);
        void* sendPtr = velBase + static_cast<size_t>(localStart) * sizeof(Body);
        MPI_Gatherv(sendPtr, localCount, velType,
                    velBase, counts.data(), displs.data(), velType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&velType);

        if (rank == 0) {
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
    }

    // Validation: check that simulation produces finite, reasonable values
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        const bool localValid = validateSimulationLocal(bodies, localStart, localCount);
        int localValidInt = localValid ? 1 : 0;
        int allValidInt = 0;
        MPI_Allreduce(&localValidInt, &allValidInt, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (allValidInt) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, localStart, localCount, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            exitCode = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Type_free(&posType);
    MPI_Finalize();
    return exitCode;
}
