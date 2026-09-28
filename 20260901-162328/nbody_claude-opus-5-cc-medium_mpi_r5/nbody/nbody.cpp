#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Number of local bodies processed simultaneously in the force kernel. Blocking over i
// lets the compiler vectorize across i while the summation order over j stays identical
// to the sequential code, which keeps results bit-for-bit equivalent.
constexpr size_t IBLOCK = 8;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Structure-of-arrays state. Positions are replicated on every rank (they are needed by
// every force evaluation), velocities are distributed: rank r owns [offset, offset+count).
struct State {
    std::vector<double> px, py, pz; // global, size n
    std::vector<double> vx, vy, vz; // local, size count
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

// Computes the forces acting on the locally owned bodies and applies them to the local
// velocities. All positions of all bodies must be up to date in state.p*.
void computeForces(State& state, const size_t n, const size_t offset, const size_t count) {
    const double* __restrict px = state.px.data();
    const double* __restrict py = state.py.data();
    const double* __restrict pz = state.pz.data();

    for (size_t i0 = 0; i0 < count; i0 += IBLOCK) {
        const size_t ib = std::min(IBLOCK, count - i0);

        double xi[IBLOCK], yi[IBLOCK], zi[IBLOCK];
        double Fx[IBLOCK] = {}, Fy[IBLOCK] = {}, Fz[IBLOCK] = {};

        // The tail block is padded with a copy of its first body so that the kernel below
        // always runs over the full block width; the padded results are discarded.
        for (size_t k = 0; k < IBLOCK; ++k) {
            const size_t idx = offset + i0 + (k < ib ? k : 0);
            xi[k] = px[idx];
            yi[k] = py[idx];
            zi[k] = pz[idx];
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = px[j];
            const double yj = py[j];
            const double zj = pz[j];

            for (size_t k = 0; k < IBLOCK; ++k) {
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

        for (size_t k = 0; k < ib; ++k) {
            state.vx[i0 + k] += DT * Fx[k];
            state.vy[i0 + k] += DT * Fy[k];
            state.vz[i0 + k] += DT * Fz[k];
        }
    }
}

void integrateBodies(State& state, const size_t offset, const size_t count) {
    double* __restrict px = state.px.data() + offset;
    double* __restrict py = state.py.data() + offset;
    double* __restrict pz = state.pz.data() + offset;
    const double* __restrict vx = state.vx.data();
    const double* __restrict vy = state.vy.data();
    const double* __restrict vz = state.vz.data();

    for (size_t i = 0; i < count; ++i) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
        printf("MPI ranks: %d\n", numRanks);
    }

    const size_t n = numBodies > 0 ? (size_t)numBodies : 0;

    // Block distribution of bodies over ranks; the remainder is spread over the first ranks.
    std::vector<int> counts(numRanks), displs(numRanks);
    std::vector<int> counts3(numRanks), displs3(numRanks);
    {
        const size_t base = n / (size_t)numRanks;
        const size_t rem = n % (size_t)numRanks;
        size_t off = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t c = base + ((size_t)r < rem ? 1 : 0);
            counts[r] = (int)c;
            displs[r] = (int)off;
            counts3[r] = (int)(3 * c);
            displs3[r] = (int)(3 * off);
            off += c;
        }
    }
    const size_t offset = (size_t)displs[rank];
    const size_t count = (size_t)counts[rank];

    // Initialize bodies: the generator is deterministic, so every rank reproduces the
    // exact same initial state and simply keeps the slice it owns.
    State state;
    state.px.resize(n);
    state.py.resize(n);
    state.pz.resize(n);
    state.vx.resize(count);
    state.vy.resize(count);
    state.vz.resize(count);
    {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (size_t i = 0; i < n; ++i) {
            state.px[i] = bodies[i].pos.x;
            state.py[i] = bodies[i].pos.y;
            state.pz[i] = bodies[i].pos.z;
        }
        for (size_t i = 0; i < count; ++i) {
            state.vx[i] = bodies[offset + i].vel.x;
            state.vy[i] = bodies[offset + i].vel.y;
            state.vz[i] = bodies[offset + i].vel.z;
        }
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(state, n, offset, count);
        integrateBodies(state, offset, count);

        // Replicate the updated positions so that every rank can evaluate all pair forces.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.px.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.py.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.pz.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Collect the final state on rank 0 for output and validation. Positions are already
    // replicated everywhere; only the velocities have to be gathered.
    std::vector<Body> bodies;
    if (printResults || validate) {
        std::vector<double> localVel(3 * count);
        for (size_t i = 0; i < count; ++i) {
            localVel[3 * i + 0] = state.vx[i];
            localVel[3 * i + 1] = state.vy[i];
            localVel[3 * i + 2] = state.vz[i];
        }
        std::vector<double> allVel(rank == 0 ? 3 * n : 0);
        MPI_Gatherv(localVel.data(), (int)(3 * count), MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            bodies.resize(n);
            for (size_t i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(state.px[i], state.py[i], state.pz[i]);
                bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
            }
        }
    }

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
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

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
