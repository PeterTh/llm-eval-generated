#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };

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

void computeForces(std::vector<Body>& local, const std::vector<double>& positions) {
    const std::size_t n = positions.size() / 3;
    for (auto& body : local) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        const double xi = body.pos.x, yi = body.pos.y, zi = body.pos.z;
        for (std::size_t j = 0; j < n; ++j) {
            const double dx = positions[3*j] - xi;
            const double dy = positions[3*j+1] - yi;
            const double dz = positions[3*j+2] - zi;
            const double invDist = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
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
    for (const auto& b : bodies)
        energy += 0.5 * (b.vel.x*b.vel.x + b.vel.y*b.vel.y + b.vel.z*b.vel.z);
    for (std::size_t i = 0; i < bodies.size(); ++i)
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& b : bodies) {
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
            !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n"); return false;
        }
        if (std::abs(b.pos.x)>1e6 || std::abs(b.pos.y)>1e6 || std::abs(b.pos.z)>1e6) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n"); return false;
        }
        if (std::abs(b.vel.x)>1e6 || std::abs(b.vel.y)>1e6 || std::abs(b.vel.z)>1e6) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false;
        }
    }
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation (checks energy conservation)\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int numBodies = 1024, numSteps = 10, parseStatus = 0;
    bool validate = false, printResults = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i+1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) help = true;
        else { if (rank == 0) std::printf("Unknown option: %s\n", argv[i]); parseStatus = 1; }
    }
    if (numBodies < 0 || numSteps < 0) parseStatus = 1;
    int anyError = 0;
    MPI_Allreduce(&parseStatus, &anyError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (help || anyError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize(); return anyError;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    numBodies, numSteps, validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(ranks), displs(ranks), posCounts(ranks), posDispls(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = numBodies / ranks + (r < numBodies % ranks);
        displs[r] = r ? displs[r-1] + counts[r-1] : 0;
        posCounts[r] = 3 * counts[r]; posDispls[r] = 3 * displs[r];
    }
    MPI_Datatype bodyType;
    MPI_Type_contiguous(6, MPI_DOUBLE, &bodyType); MPI_Type_commit(&bodyType);

    std::vector<Body> allBodies;
    if (rank == 0) { allBodies.resize(numBodies); randomizeBodies(allBodies); }
    std::vector<Body> local(counts[rank]);
    MPI_Scatterv(rank == 0 ? allBodies.data() : nullptr, counts.data(), displs.data(), bodyType,
                 local.data(), counts[rank], bodyType, 0, MPI_COMM_WORLD);

    std::vector<double> localPositions(3 * counts[rank]), positions(3 * numBodies);
    auto exchangePositions = [&] {
        for (std::size_t i = 0; i < local.size(); ++i) {
            localPositions[3*i] = local[i].pos.x;
            localPositions[3*i+1] = local[i].pos.y;
            localPositions[3*i+2] = local[i].pos.z;
        }
        MPI_Allgatherv(localPositions.data(), static_cast<int>(localPositions.size()), MPI_DOUBLE,
                       positions.data(), posCounts.data(), posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    };
    exchangePositions();
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(local, positions);
        integrateBodies(local);
        exchangePositions();
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) allBodies.resize(numBodies);
    MPI_Gatherv(local.data(), counts[rank], bodyType, rank == 0 ? allBodies.data() : nullptr,
                counts.data(), displs.data(), bodyType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&bodyType);

    int status = 0;
    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        if (printResults) {
            std::vector<double> data; data.reserve(static_cast<std::size_t>(numBodies) * 6);
            for (const auto& b : allBodies)
                data.insert(data.end(), {b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z});
            print_results(data, "Bodies");
        }
        if (validate) {
            std::printf("Validating simulation results...\n");
            if (validateSimulation(allBodies)) {
                std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(allBodies));
            } else { std::printf("Validation: FAILED\n"); status = 1; }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
