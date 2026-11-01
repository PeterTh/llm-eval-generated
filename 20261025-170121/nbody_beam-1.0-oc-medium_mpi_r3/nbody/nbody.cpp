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

void computeForces(std::vector<Body>& bodies, const std::vector<double>& allPos, int offset, int nGlobal) {
    const size_t nLocal = bodies.size();
    for (size_t li = 0; li < nLocal; ++li) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        const double pix = bodies[li].pos.x;
        const double piy = bodies[li].pos.y;
        const double piz = bodies[li].pos.z;
        for (int j = 0; j < nGlobal; ++j) {
            const double dx = allPos[j*3+0] - pix;
            const double dy = allPos[j*3+1] - piy;
            const double dz = allPos[j*3+2] - piz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        bodies[li].vel.x += DT * Fx;
        bodies[li].vel.y += DT * Fy;
        bodies[li].vel.z += DT * Fz;
    }
    (void)offset;
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
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }
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

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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

    // Partition
    int base = numBodies / size;
    int rem = numBodies % size;
    int localN = base + (rank < rem ? 1 : 0);
    int offset = rank * base + (rank < rem ? rank : rem);

    // init full on rank0
    std::vector<Body> full;
    if (rank == 0) {
        full.assign(numBodies, Body());
        randomizeBodies(full);
    }
    // broadcast full as doubles
    std::vector<double> fullBuf;
    if (rank == 0) {
        fullBuf.resize((size_t)numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            fullBuf[i*6+0]=full[i].pos.x; fullBuf[i*6+1]=full[i].pos.y; fullBuf[i*6+2]=full[i].pos.z;
            fullBuf[i*6+3]=full[i].vel.x; fullBuf[i*6+4]=full[i].vel.y; fullBuf[i*6+5]=full[i].vel.z;
        }
    } else {
        fullBuf.resize((size_t)numBodies * 6);
    }
    MPI_Bcast(fullBuf.data(), numBodies*6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Body> local(localN);
    for (int i = 0; i < localN; ++i) {
        int g = offset + i;
        local[i].pos.x = fullBuf[g*6+0];
        local[i].pos.y = fullBuf[g*6+1];
        local[i].pos.z = fullBuf[g*6+2];
        local[i].vel.x = fullBuf[g*6+3];
        local[i].vel.y = fullBuf[g*6+4];
        local[i].vel.z = fullBuf[g*6+5];
    }
    full.clear(); full.shrink_to_fit();
    fullBuf.clear(); fullBuf.shrink_to_fit();

    // counts for allgatherv positions
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int ln = base + (r < rem ? 1 : 0);
        counts[r] = ln * 3;
        displs[r] = (r*base + (r < rem ? r : rem)) * 3;
    }
    std::vector<double> allPos((size_t)numBodies * 3);
    std::vector<double> localPos((size_t)localN * 3);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < localN; ++i) {
            localPos[i*3+0] = local[i].pos.x;
            localPos[i*3+1] = local[i].pos.y;
            localPos[i*3+2] = local[i].pos.z;
        }
        MPI_Allgatherv(localPos.data(), localN*3, MPI_DOUBLE,
                       allPos.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        computeForces(local, allPos, offset, numBodies);
        integrateBodies(local);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", maxDuration);

    // Gather all bodies to rank0
    std::vector<int> bcounts(size), bdispls(size);
    for (int r = 0; r < size; ++r) {
        int ln = base + (r < rem ? 1 : 0);
        bcounts[r] = ln * 6;
        bdispls[r] = (r*base + (r < rem ? r : rem)) * 6;
    }
    std::vector<double> allBuf;
    if (rank == 0) allBuf.resize((size_t)numBodies * 6);
    std::vector<double> localBuf((size_t)localN * 6);
    for (int i = 0; i < localN; ++i) {
        localBuf[i*6+0]=local[i].pos.x; localBuf[i*6+1]=local[i].pos.y; localBuf[i*6+2]=local[i].pos.z;
        localBuf[i*6+3]=local[i].vel.x; localBuf[i*6+4]=local[i].vel.y; localBuf[i*6+5]=local[i].vel.z;
    }
    MPI_Gatherv(localBuf.data(), localN*6, MPI_DOUBLE,
                allBuf.data(), bcounts.data(), bdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos.x = allBuf[i*6+0];
            bodies[i].pos.y = allBuf[i*6+1];
            bodies[i].pos.z = allBuf[i*6+2];
            bodies[i].vel.x = allBuf[i*6+3];
            bodies[i].vel.y = allBuf[i*6+4];
            bodies[i].vel.z = allBuf[i*6+5];
        }

        if (printResults) {
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

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
