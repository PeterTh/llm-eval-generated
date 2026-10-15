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

constexpr int BLK = 8;

// Compute forces on bodies [lo, hi) from all n bodies (positions pos, 3 doubles per body)
// and update their velocities. Each body's sum runs over j in the original order, so results
// match the serial code bit for bit. Several i-bodies are processed at once for ILP/SIMD.
void computeForces(const double* __restrict pos, Vec3* __restrict vel, size_t n, size_t lo, size_t hi) {
    for (size_t i0 = lo; i0 < hi; i0 += BLK) {
        double xi[BLK], yi[BLK], zi[BLK];
        double Fx[BLK], Fy[BLK], Fz[BLK];
        for (int k = 0; k < BLK; ++k) {
            const size_t i = (i0 + k < hi) ? i0 + k : hi - 1;
            xi[k] = pos[3 * i];
            yi[k] = pos[3 * i + 1];
            zi[k] = pos[3 * i + 2];
            Fx[k] = Fy[k] = Fz[k] = 0.0;
        }
        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[3 * j], yj = pos[3 * j + 1], zj = pos[3 * j + 2];
#pragma omp simd
            for (int k = 0; k < BLK; ++k) {
                const double dx = xj - xi[k];
                const double dy = yj - yi[k];
                const double dz = zj - zi[k];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx[k] += dx * invDist3;
                Fy[k] += dy * invDist3;
                Fz[k] += dz * invDist3;
            }
        }
        for (int k = 0; k < BLK && i0 + k < hi; ++k) {
            Vec3& v = vel[i0 + k - lo];
            v.x += DT * Fx[k];
            v.y += DT * Fy[k];
            v.z += DT * Fz[k];
        }
    }
}

void integrateBodies(double* pos, const Vec3* vel, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
        pos[3 * i] += vel[i - lo].x * DT;
        pos[3 * i + 1] += vel[i - lo].y * DT;
        pos[3 * i + 2] += vel[i - lo].z * DT;
    }
}

// Distributed energy: kinetic part (full velocities, rank 0 only) plus the pair potential,
// with outer index i distributed cyclically over ranks for load balance. Result valid on rank 0.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int size) {
    double energy = 0.0;
    const size_t n = bodies.size();

    if (rank == 0) {
        for (const auto& body : bodies) {
            energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
        }
    }

    double pot = 0.0;
    for (size_t i = rank; i < n; i += size) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            pot -= 1.0 / dist;
        }
    }
    double totalPot = 0.0;
    MPI_Reduce(&pot, &totalPot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return energy + totalPot;
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically)
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

    const size_t n = numBodies > 0 ? (size_t)numBodies : 0;

    // Every rank generates the identical initial state (deterministic), keeps all positions,
    // and owns the velocities of its contiguous slice.
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    std::vector<double> pos(3 * n + 3);
    for (size_t i = 0; i < n; ++i) {
        pos[3 * i] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }

    std::vector<int> counts(size), displs(size);       // in bodies
    std::vector<int> counts3(size), displs3(size);     // in doubles
    std::vector<int> countsV(size), displsV(size);     // velocity gather, in doubles
    for (int r = 0; r < size; ++r) {
        const size_t l = n * r / size, h = n * (r + 1) / size;
        counts[r] = (int)(h - l);
        displs[r] = (int)l;
        counts3[r] = 3 * counts[r];
        displs3[r] = 3 * displs[r];
    }
    const size_t lo = displs[rank], hi = lo + counts[rank];
    std::vector<Vec3> vel(hi - lo);
    for (size_t i = lo; i < hi; ++i) vel[i - lo] = bodies[i].vel;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(pos.data(), vel.data(), n, lo, hi);
        integrateBodies(pos.data(), vel.data(), lo, hi);
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", (long)duration.count());

    // Collect final state: positions are already replicated; gather velocities on rank 0.
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = pos[3 * i];
        bodies[i].pos.y = pos[3 * i + 1];
        bodies[i].pos.z = pos[3 * i + 2];
    }
    if (printResults || validate) {
        std::vector<double> vbuf(3 * n + 3), mine(3 * (hi - lo) + 1);
        for (size_t i = lo; i < hi; ++i) {
            mine[3 * (i - lo)] = vel[i - lo].x;
            mine[3 * (i - lo) + 1] = vel[i - lo].y;
            mine[3 * (i - lo) + 2] = vel[i - lo].z;
        }
        MPI_Gatherv(mine.data(), 3 * (int)(hi - lo), MPI_DOUBLE, vbuf.data(), counts3.data(), displs3.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (size_t i = 0; i < n; ++i) {
                bodies[i].vel = Vec3(vbuf[3 * i], vbuf[3 * i + 1], vbuf[3 * i + 2]);
            }
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
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

    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");
        int ok = 1;
        if (rank == 0) ok = validateSimulation(bodies) ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            double finalEnergy = computeTotalEnergy(bodies, rank, size);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
