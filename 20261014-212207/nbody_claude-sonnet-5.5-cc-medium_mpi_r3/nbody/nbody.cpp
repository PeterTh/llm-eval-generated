#include <algorithm>
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

// Compute velocity updates for bodies [lo, hi) given all positions (interleaved xyz).
// Per-body accumulation order over j matches the serial code; tiles of TILE bodies
// are processed together so the compiler can vectorize across i.
constexpr int TILE = 8;

void computeForces(const double* __restrict pos, double* __restrict vel, size_t n, size_t lo, size_t hi) {
    for (size_t i0 = lo; i0 < hi; i0 += TILE) {
        double xi[TILE], yi[TILE], zi[TILE];
        double Fx[TILE], Fy[TILE], Fz[TILE];
        for (int k = 0; k < TILE; ++k) {
            size_t i = std::min(i0 + k, hi - 1);
            xi[k] = pos[3 * i];
            yi[k] = pos[3 * i + 1];
            zi[k] = pos[3 * i + 2];
            Fx[k] = Fy[k] = Fz[k] = 0.0;
        }
        for (size_t j = 0; j < n; ++j) {
            const double xj = pos[3 * j], yj = pos[3 * j + 1], zj = pos[3 * j + 2];
            for (int k = 0; k < TILE; ++k) {
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
        const size_t cnt = std::min<size_t>(TILE, hi - i0);
        for (size_t k = 0; k < cnt; ++k) {
            double* v = vel + 3 * (i0 + k - lo);
            v[0] += DT * Fx[k];
            v[1] += DT * Fy[k];
            v[2] += DT * Fz[k];
        }
    }
}

void integrateBodies(double* __restrict pos, const double* __restrict vel, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) {
        double* p = pos + 3 * i;
        const double* v = vel + 3 * (i - lo);
        p[0] += v[0] * DT;
        p[1] += v[1] * DT;
        p[2] += v[2] * DT;
    }
}

// Kinetic energy over all bodies plus this rank's share (cyclic) of potential energy;
// caller reduces across ranks.
double computePartialEnergy(const std::vector<Body>& bodies, int rank, int size) {
    double energy = 0.0;
    const size_t n = bodies.size();
    if (rank == 0) {
        for (const auto& body : bodies) {
            energy += 0.5 * (body.vel.x * body.vel.x +
                            body.vel.y * body.vel.y +
                            body.vel.z * body.vel.z);
        }
    }
    for (size_t i = rank; i < n; i += size) {
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    // Initialize bodies (deterministic; every rank generates the same full state)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    const size_t n = numBodies;

    // Block distribution of bodies; counts/displacements in doubles (3 per body)
    std::vector<int> counts(size), displs(size);
    std::vector<size_t> first(size + 1);
    for (int r = 0; r <= size; ++r) first[r] = n * r / size;
    for (int r = 0; r < size; ++r) {
        counts[r] = 3 * (int)(first[r + 1] - first[r]);
        displs[r] = 3 * (int)first[r];
    }
    const size_t lo = first[rank], hi = first[rank + 1];

    std::vector<double> pos(3 * n), vel(3 * (hi - lo));
    for (size_t i = 0; i < n; ++i) {
        pos[3 * i] = bodies[i].pos.x;
        pos[3 * i + 1] = bodies[i].pos.y;
        pos[3 * i + 2] = bodies[i].pos.z;
    }
    for (size_t i = lo; i < hi; ++i) {
        vel[3 * (i - lo)] = bodies[i].vel.x;
        vel[3 * (i - lo) + 1] = bodies[i].vel.y;
        vel[3 * (i - lo) + 2] = bodies[i].vel.z;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (hi > lo) {
            computeForces(pos.data(), vel.data(), n, lo, hi);
            integrateBodies(pos.data(), vel.data(), lo, hi);
        }
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), counts.data(), displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_ms);

    // Gather final velocities everywhere (positions are already replicated)
    std::vector<double> allVel(3 * n);
    MPI_Allgatherv(vel.data(), 3 * (int)(hi - lo), MPI_DOUBLE, allVel.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        bodies[i].vel = Vec3(allVel[3 * i], allVel[3 * i + 1], allVel[3 * i + 2]);
    }

    int rc = 0;

    // Print results for external validation
    if (printResults && rank == 0) {
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

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        bool ok = (rank == 0) ? validateSimulation(bodies) : true;
        int okFlag = ok;
        MPI_Bcast(&okFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (okFlag) {
            double local = computePartialEnergy(bodies, rank, size), finalEnergy = 0.0;
            MPI_Reduce(&local, &finalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
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
