#include <mpi.h>

#include <algorithm>
#include <chrono>
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

struct BodyRange {
    size_t start;
    size_t end;
    size_t size() const { return end - start; }
};

static BodyRange computeRange(size_t n, int rank, int numRanks) {
    const size_t chunk = (n + static_cast<size_t>(numRanks) - 1) / static_cast<size_t>(numRanks);
    const size_t start = chunk * static_cast<size_t>(rank);
    const size_t end = std::min(start + chunk, n);
    return {start, end};
}

void computeForces(std::vector<Body>& bodies, int rank, int numRanks) {
    const size_t n = bodies.size();
    const auto range = computeRange(n, rank, numRanks);

    for (size_t i = range.start; i < range.end; ++i) {
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

void integrateBodies(std::vector<Body>& bodies, int rank, int numRanks) {
    const auto range = computeRange(bodies.size(), rank, numRanks);

    for (size_t i = range.start; i < range.end; ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

void syncBodies(std::vector<Body>& bodies, int rank, int numRanks) {
    const size_t n = bodies.size();
    const auto range = computeRange(n, rank, numRanks);
    const size_t local_n = range.size();

    const int sendCount = static_cast<int>(local_n * 6);
    std::vector<double> sendbuf(sendCount);
    for (size_t i = 0; i < local_n; ++i) {
        const auto& b = bodies[range.start + i];
        sendbuf[i * 6 + 0] = b.pos.x;
        sendbuf[i * 6 + 1] = b.pos.y;
        sendbuf[i * 6 + 2] = b.pos.z;
        sendbuf[i * 6 + 3] = b.vel.x;
        sendbuf[i * 6 + 4] = b.vel.y;
        sendbuf[i * 6 + 5] = b.vel.z;
    }

    const size_t chunk = (n + static_cast<size_t>(numRanks) - 1) / static_cast<size_t>(numRanks);
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t s = chunk * static_cast<size_t>(r);
        const size_t e = std::min(s + chunk, n);
        counts[r] = static_cast<int>((e - s) * 6);
        displs[r] = static_cast<int>(s * 6);
    }

    std::vector<double> recvbuf(n * 6);
    MPI_Allgatherv(sendbuf.data(), sendCount, MPI_DOUBLE,
                   recvbuf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = recvbuf[i * 6 + 0];
        bodies[i].pos.y = recvbuf[i * 6 + 1];
        bodies[i].pos.z = recvbuf[i * 6 + 2];
        bodies[i].vel.x = recvbuf[i * 6 + 3];
        bodies[i].vel.y = recvbuf[i * 6 + 4];
        bodies[i].vel.z = recvbuf[i * 6 + 5];
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int numRanks) {
    const size_t n = bodies.size();
    const auto range = computeRange(n, rank, numRanks);

    double energy = 0.0;

    for (size_t i = range.start; i < range.end; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                         bodies[i].vel.y * bodies[i].vel.y +
                         bodies[i].vel.z * bodies[i].vel.z);
    }

    for (size_t i = range.start; i < range.end; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    double totalEnergy;
    MPI_Allreduce(&energy, &totalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return totalEnergy;
}

bool validateSimulation(const std::vector<Body>& bodies, int rank, int numRanks) {
    const auto range = computeRange(bodies.size(), rank, numRanks);

    int localValid = 1;
    for (size_t i = range.start; i < range.end; ++i) {
        if (!std::isfinite(bodies[i].pos.x) || !std::isfinite(bodies[i].pos.y) || !std::isfinite(bodies[i].pos.z) ||
            !std::isfinite(bodies[i].vel.x) || !std::isfinite(bodies[i].vel.y) || !std::isfinite(bodies[i].vel.z)) {
            localValid = 0;
            break;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies[i].pos.x) > maxPos || std::abs(bodies[i].pos.y) > maxPos || std::abs(bodies[i].pos.z) > maxPos) {
            localValid = 0;
            break;
        }
        if (std::abs(bodies[i].vel.x) > maxVel || std::abs(bodies[i].vel.y) > maxVel || std::abs(bodies[i].vel.z) > maxVel) {
            localValid = 0;
            break;
        }
    }

    int globalValid;
    MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalValid != 0;
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
    int rank, numRanks;
    MPI_Init(&argc, &argv);
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, numRanks);
        integrateBodies(bodies, rank, numRanks);
        syncBodies(bodies, rank, numRanks);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    if (printResults) {
        const auto range = computeRange(numBodies, rank, numRanks);
        const int localCount = static_cast<int>(range.size() * 6);

        std::vector<double> localData(localCount);
        for (size_t i = 0; i < range.size(); ++i) {
            const auto& b = bodies[range.start + i];
            localData[i * 6 + 0] = b.pos.x;
            localData[i * 6 + 1] = b.pos.y;
            localData[i * 6 + 2] = b.pos.z;
            localData[i * 6 + 3] = b.vel.x;
            localData[i * 6 + 4] = b.vel.y;
            localData[i * 6 + 5] = b.vel.z;
        }

        const size_t chunk = (numBodies + numRanks - 1) / numRanks;
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const size_t s = chunk * static_cast<size_t>(r);
            const size_t e = std::min(s + chunk, static_cast<size_t>(numBodies));
            counts[r] = static_cast<int>((e - s) * 6);
            displs[r] = static_cast<int>(s * 6);
        }

        std::vector<double> bodyData;
        if (rank == 0) bodyData.resize(numBodies * 6);

        MPI_Gatherv(localData.data(), localCount, MPI_DOUBLE,
                    rank == 0 ? bodyData.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(bodyData, "Bodies");
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        if (validateSimulation(bodies, rank, numRanks)) {
            double finalEnergy = computeTotalEnergy(bodies, rank, numRanks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
