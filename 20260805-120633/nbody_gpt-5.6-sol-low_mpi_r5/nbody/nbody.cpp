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
    constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double));

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

void computeForces(std::vector<Body>& localBodies, const std::vector<Vec3>& positions) {
    const size_t n = positions.size();
    for (auto& body : localBodies) {
        const double px = body.pos.x;
        const double py = body.pos.y;
        const double pz = body.pos.z;
        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (size_t j = 0; j < n; ++j) {
            const double dx = positions[j].x - px;
            const double dy = positions[j].y - py;
            const double dz = positions[j].z - pz;
            const double invDist = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        body.vel.x += DT * fx;
        body.vel.y += DT * fy;
        body.vel.z += DT * fz;
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
    for (const auto& body : bodies)
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
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
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else { if (rank == 0) printf("Unknown option: %s\n", argv[i]); parseStatus = 1; }
    }
    if (numBodies < 0 || numSteps < 0) {
        if (rank == 0) printf("Number of bodies and steps must be non-negative\n");
        parseStatus = 1;
    }
    if (parseStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 1 ? 1 : 0;
    }

    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\n", numBodies, numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        offsets[r] = r * (numBodies / ranks) + (r < numBodies % ranks ? r : numBodies % ranks);
    }
    std::vector<int> bodyCounts(ranks), bodyOffsets(ranks), posCounts(ranks), posOffsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        bodyCounts[r] = counts[r] * 6; bodyOffsets[r] = offsets[r] * 6;
        posCounts[r] = counts[r] * 3; posOffsets[r] = offsets[r] * 3;
    }

    std::vector<Body> allBodies;
    if (rank == 0) { allBodies.resize(numBodies); randomizeBodies(allBodies); }
    std::vector<Body> localBodies(counts[rank]);
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                 localBodies.data(), bodyCounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Vec3> positions(numBodies), localPositions(counts[rank]);
    for (size_t i = 0; i < localBodies.size(); ++i) localPositions[i] = localBodies[i].pos;
    MPI_Allgatherv(localPositions.data(), posCounts[rank], MPI_DOUBLE,
                   positions.data(), posCounts.data(), posOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    allBodies.clear();

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(localBodies, positions);
        integrateBodies(localBodies);
        for (size_t i = 0; i < localBodies.size(); ++i) localPositions[i] = localBodies[i].pos;
        MPI_Allgatherv(localPositions.data(), posCounts[rank], MPI_DOUBLE,
                       positions.data(), posCounts.data(), posOffsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));

    if (printResults || validate) {
        if (rank == 0) allBodies.resize(numBodies);
        MPI_Gatherv(localBodies.data(), bodyCounts[rank], MPI_DOUBLE,
                    rank == 0 ? allBodies.data() : nullptr, bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    int result = 0;
    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (const auto& body : allBodies) {
            bodyData.insert(bodyData.end(), {body.pos.x, body.pos.y, body.pos.z,
                                             body.vel.x, body.vel.y, body.vel.z});
        }
        print_results(bodyData, "Bodies");
    }
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(allBodies)) {
            printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(allBodies));
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
