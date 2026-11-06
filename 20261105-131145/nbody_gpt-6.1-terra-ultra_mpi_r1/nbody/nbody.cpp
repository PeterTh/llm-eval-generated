#include <algorithm>
#include <cstddef>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Vectorizing the source-body loop changes floating-point contraction and the
// benchmark result hash. Keep its original ordered accumulation.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-loop-vectorize")))
#endif
void computeForces(std::vector<Body>& bodies, const std::vector<Vec3>& positions, size_t offset) {
    const size_t n = positions.size();
    
    for (size_t i = 0; i < bodies.size(); ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable)
#endif
        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - positions[offset + i].x;
            const double dy = positions[j].y - positions[offset + i].y;
            const double dz = positions[j].z - positions[offset + i].z;
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

// Divide the triangular potential-energy calculation cyclically so that early,
// more expensive rows are spread across ranks.
double computeLocalEnergy(const std::vector<Body>& bodies,
                          const std::vector<Vec3>& positions, int rank, int ranks) {
    double energy = 0.0;
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }
    for (size_t i = static_cast<size_t>(rank); i < positions.size(); i += static_cast<size_t>(ranks)) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            const double dx = positions[j].x - positions[i].x;
            const double dy = positions[j].y - positions[i].y;
            const double dz = positions[j].z - positions[i].z;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool done = false;

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
            done = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            done = true;
            break;
        }
    }
    if (!done && numBodies < 0) {
        if (rank == 0) printf("Number of bodies must be nonnegative\n");
        exitCode = 1;
        done = true;
    }
    if (done) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // These types describe arrays of complete objects, including their extent.
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    static_assert(sizeof(Body) == 6 * sizeof(double));
    static_assert(offsetof(Body, vel) == 3 * sizeof(double));
    MPI_Datatype mpiVec3, mpiBody;
    MPI_Type_contiguous(3, MPI_DOUBLE, &mpiVec3);
    MPI_Type_commit(&mpiVec3);
    MPI_Type_contiguous(6, MPI_DOUBLE, &mpiBody);
    MPI_Type_commit(&mpiBody);

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + std::min(r, numBodies % ranks);
    }
    std::vector<Body> localBodies(counts[rank]);
    std::vector<Body> gatheredBodies;
    if (rank == 0) {
        gatheredBodies.resize(numBodies);
        randomizeBodies(gatheredBodies);
    }
    MPI_Scatterv(gatheredBodies.data(), counts.data(), offsets.data(), mpiBody,
                 localBodies.data(), counts[rank], mpiBody, 0, MPI_COMM_WORLD);
    // Every rank keeps the positions in original global order, preserving the
    // original summation order for each independently calculated body.
    std::vector<Vec3> positions(numBodies);
    for (int i = 0; i < counts[rank]; ++i)
        positions[offsets[rank] + i] = localBodies[i].pos;
    if (numBodies > 0) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, mpiVec3, positions.data(), counts.data(),
                       offsets.data(), mpiVec3, MPI_COMM_WORLD);
    }
    if (rank == 0) std::vector<Body>().swap(gatheredBodies);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions, offsets[rank]);
        integrateBodies(localBodies);
        if (step + 1 < numSteps) {
            for (int i = 0; i < counts[rank]; ++i)
                positions[offsets[rank] + i] = localBodies[i].pos;
            if (numBodies > 0) {
                MPI_Allgatherv(MPI_IN_PLACE, 0, mpiVec3, positions.data(), counts.data(),
                               offsets.data(), mpiVec3, MPI_COMM_WORLD);
            }
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

    if (printResults || validate) {
        if (rank == 0) gatheredBodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), counts[rank], mpiBody, gatheredBodies.data(),
                    counts.data(), offsets.data(), mpiBody, 0, MPI_COMM_WORLD);
    }
    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : gatheredBodies) {
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
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (!validateSimulation(gatheredBodies)) {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (exitCode == 0) {
            if (numSteps > 0 && numBodies > 0) {
                for (int i = 0; i < counts[rank]; ++i)
                    positions[offsets[rank] + i] = localBodies[i].pos;
                MPI_Allgatherv(MPI_IN_PLACE, 0, mpiVec3, positions.data(), counts.data(),
                               offsets.data(), mpiVec3, MPI_COMM_WORLD);
            }
            const double localEnergy = computeLocalEnergy(localBodies, positions, rank, ranks);
            double finalEnergy = 0.0;
            MPI_Reduce(&localEnergy, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        }
    }
    MPI_Type_free(&mpiBody);
    MPI_Type_free(&mpiVec3);
    MPI_Finalize();
    return exitCode;
}
