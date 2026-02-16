#include <mpi.h>
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

// compute forces for local range using global double-array representation (6 doubles per body)
static inline void computeForcesLocal(const std::vector<double>& globalData, std::vector<double>& localData,
                                     int numBodies, int localOffsetBodies, int localBodies) {
    // localData layout: [pos.x,pos.y,pos.z,vel.x,vel.y,vel.z]*localBodies
    for (int li = 0; li < localBodies; ++li) {
        int gi = localOffsetBodies + li;
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double ix = globalData[6 * gi + 0];
        const double iy = globalData[6 * gi + 1];
        const double iz = globalData[6 * gi + 2];

        for (int j = 0; j < numBodies; ++j) {
            const double jx = globalData[6 * j + 0];
            const double jy = globalData[6 * j + 1];
            const double jz = globalData[6 * j + 2];
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

        // update velocity in localData
        localData[6 * li + 3] += DT * Fx;
        localData[6 * li + 4] += DT * Fy;
        localData[6 * li + 5] += DT * Fz;
    }
}

static inline void integrateLocal(std::vector<double>& localData, int localBodies) {
    for (int li = 0; li < localBodies; ++li) {
        localData[6 * li + 0] += localData[6 * li + 3] * DT;
        localData[6 * li + 1] += localData[6 * li + 4] * DT;
        localData[6 * li + 2] += localData[6 * li + 5] * DT;
    }
}

// helpers to convert between vector<double> layout and Body vector
static inline void fillBodyVectorFromDoubles(const std::vector<double>& data, std::vector<Body>& bodies) {
    const int n = (int)bodies.size();
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = data[6 * i + 0];
        bodies[i].pos.y = data[6 * i + 1];
        bodies[i].pos.z = data[6 * i + 2];
        bodies[i].vel.x = data[6 * i + 3];
        bodies[i].vel.y = data[6 * i + 4];
        bodies[i].vel.z = data[6 * i + 5];
    }
}

static inline void fillDoublesFromBodyVector(const std::vector<Body>& bodies, std::vector<double>& data) {
    const int n = (int)bodies.size();
    for (int i = 0; i < n; ++i) {
        data[6 * i + 0] = bodies[i].pos.x;
        data[6 * i + 1] = bodies[i].pos.y;
        data[6 * i + 2] = bodies[i].pos.z;
        data[6 * i + 3] = bodies[i].vel.x;
        data[6 * i + 4] = bodies[i].vel.y;
        data[6 * i + 5] = bodies[i].vel.z;
    }
}

// reuse existing validation/energy functions by converting to Body vectors on rank 0
double computeTotalEnergyFromDoubles(const std::vector<double>& data) {
    const int n = (int)(data.size() / 6);
    double energy = 0.0;
    // kinetic
    for (int i = 0; i < n; ++i) {
        const double vx = data[6 * i + 3];
        const double vy = data[6 * i + 4];
        const double vz = data[6 * i + 5];
        energy += 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    // potential
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = data[6 * j + 0] - data[6 * i + 0];
            const double dy = data[6 * j + 1] - data[6 * i + 1];
            const double dz = data[6 * j + 2] - data[6 * i + 2];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
}

bool validateSimulationFromDoubles(const std::vector<double>& data) {
    const int n = (int)(data.size() / 6);
    for (int i = 0; i < n; ++i) {
        const double px = data[6 * i + 0];
        const double py = data[6 * i + 1];
        const double pz = data[6 * i + 2];
        const double vx = data[6 * i + 3];
        const double vy = data[6 * i + 4];
        const double vz = data[6 * i + 5];
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) || !std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px) > maxPos || std::abs(py) > maxPos || std::abs(pz) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx) > maxVel || std::abs(vy) > maxVel || std::abs(vz) > maxVel) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (done on all ranks)
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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("N-Body Simulation (MPI)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // compute partitioning
    std::vector<int> countsBodies(worldSize);
    std::vector<int> displsBodies(worldSize);
    int base = numBodies / worldSize;
    int rem = numBodies % worldSize;
    int offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        countsBodies[r] = base + (r < rem ? 1 : 0);
        displsBodies[r] = offset;
        offset += countsBodies[r];
    }

    // counts in doubles (6 doubles per body)
    std::vector<int> countsD(worldSize), displsD(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        countsD[r] = countsBodies[r] * 6;
        displsD[r] = displsBodies[r] * 6;
    }

    const int localBodies = countsBodies[worldRank];
    const int localD = countsD[worldRank];

    // prepare local buffer
    std::vector<double> localData(localD);
    std::vector<double> globalData(numBodies * 6);

    // root initializes and scatters
    if (worldRank == 0) {
        std::vector<Body> allBodies(numBodies);
        randomizeBodies(allBodies);
        std::vector<double> allData(numBodies * 6);
        fillDoublesFromBodyVector(allBodies, allData);
        MPI_Scatterv(allData.data(), countsD.data(), displsD.data(), MPI_DOUBLE,
                     localData.data(), localD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     localData.data(), localD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // build global view for the first step
    MPI_Allgatherv(localData.data(), localD, MPI_DOUBLE,
                   globalData.data(), countsD.data(), displsD.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // synchronize and time on rank 0
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // compute forces for local partition based on globalData
        computeForcesLocal(globalData, localData, numBodies, displsBodies[worldRank], localBodies);
        // integrate local positions
        integrateLocal(localData, localBodies);
        // exchange updated local blocks to form new globalData
        MPI_Allgatherv(localData.data(), localD, MPI_DOUBLE,
                       globalData.data(), countsD.data(), displsD.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (worldRank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // rank 0 has globalData already
    if (printResults && worldRank == 0) {
        std::vector<double> bodyData = globalData; // copy
        print_results(bodyData, "Bodies");
    }

    if (validate && worldRank == 0) {
        printf("Validating simulation results...\n");
        if (validateSimulationFromDoubles(globalData)) {
            double finalEnergy = computeTotalEnergyFromDoubles(globalData);
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
