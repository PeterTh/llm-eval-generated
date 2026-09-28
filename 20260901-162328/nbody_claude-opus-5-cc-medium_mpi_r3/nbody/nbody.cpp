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

// Number of bodies processed simultaneously in the force kernel. Each lane keeps
// its own accumulator and iterates over all j in the original order, so results
// are bit-identical to the sequential version while allowing SIMD over i.
constexpr int VLEN = 8;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Structure-of-arrays state. Positions are replicated on every rank, velocities
// are only kept for the locally owned block of bodies.
struct State {
    std::vector<double> px, py, pz; // size n, replicated
    std::vector<double> vx, vy, vz; // size localCount
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

// Computes the forces acting on the locally owned bodies [begin, end) and
// updates the local velocities. All positions must be up to date.
void computeForces(State& s, const size_t n, const size_t begin, const size_t end) {
    const double* __restrict px = s.px.data();
    const double* __restrict py = s.py.data();
    const double* __restrict pz = s.pz.data();
    double* __restrict vx = s.vx.data();
    double* __restrict vy = s.vy.data();
    double* __restrict vz = s.vz.data();

    // The position arrays are padded by VLEN entries, so the last (partial) block
    // can be processed with the exact same kernel; surplus lanes are discarded.
    for (size_t i = begin; i < end; i += VLEN) {
        double xi[VLEN], yi[VLEN], zi[VLEN];
        double Fx[VLEN] = {}, Fy[VLEN] = {}, Fz[VLEN] = {};
        for (int l = 0; l < VLEN; ++l) {
            xi[l] = px[i + l];
            yi[l] = py[i + l];
            zi[l] = pz[i + l];
        }

        for (size_t j = 0; j < n; ++j) {
            const double xj = px[j], yj = py[j], zj = pz[j];
            for (int l = 0; l < VLEN; ++l) {
                const double dx = xj - xi[l];
                const double dy = yj - yi[l];
                const double dz = zj - zi[l];
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx[l] += dx * invDist3;
                Fy[l] += dy * invDist3;
                Fz[l] += dz * invDist3;
            }
        }

        // Velocities are padded as well, so surplus lanes can be stored blindly
        for (int l = 0; l < VLEN; ++l) {
            vx[i - begin + l] += DT * Fx[l];
            vy[i - begin + l] += DT * Fy[l];
            vz[i - begin + l] += DT * Fz[l];
        }
    }
}

void integrateBodies(State& s, const size_t begin, const size_t end) {
    for (size_t i = begin; i < end; ++i) {
        s.px[i] += s.vx[i - begin] * DT;
        s.py[i] += s.vy[i - begin] * DT;
        s.pz[i] += s.vz[i - begin] * DT;
    }
}

// Distributed total energy. Rows of the potential-energy triangle are assigned
// round-robin to balance the triangular work; partial sums are combined on rank
// 0 in rank order so the result is deterministic for a given rank count.
double computeTotalEnergy(const State& s, const size_t n, const size_t begin, const size_t end, const int rank,
    const int size) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass) over the locally owned bodies
    for (size_t i = 0; i < end - begin; ++i) {
        energy += 0.5 * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    }

    // Potential energy (assuming unit mass for all bodies)
    const double* __restrict px = s.px.data();
    const double* __restrict py = s.py.data();
    const double* __restrict pz = s.pz.data();
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size)) {
        double partial = 0.0;
        const double xi = px[i], yi = py[i], zi = pz[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            partial -= 1.0 / dist;
        }
        energy += partial;
    }

    std::vector<double> partials(rank == 0 ? size : 0);
    MPI_Gather(&energy, 1, MPI_DOUBLE, partials.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double total = 0.0;
    if (rank == 0) {
        for (const double p : partials) total += p;
    }
    return total;
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

    const size_t n = numBodies > 0 ? static_cast<size_t>(numBodies) : 0;

    // Block distribution of bodies over ranks
    std::vector<int> counts(size), displs(size);
    {
        const size_t base = size > 0 ? n / static_cast<size_t>(size) : 0;
        const size_t rem = size > 0 ? n % static_cast<size_t>(size) : 0;
        size_t off = 0;
        for (int r = 0; r < size; ++r) {
            const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(c);
            displs[r] = static_cast<int>(off);
            off += c;
        }
    }
    const size_t begin = static_cast<size_t>(displs[rank]);
    const size_t end = begin + static_cast<size_t>(counts[rank]);
    const size_t localCount = end - begin;

    // Initialize bodies: every rank runs the identical deterministic generator,
    // so no communication is required here.
    State state;
    // VLEN padding entries allow the force kernel to read whole blocks of i
    state.px.resize(n + VLEN, 0.0);
    state.py.resize(n + VLEN, 0.0);
    state.pz.resize(n + VLEN, 0.0);
    state.vx.resize(localCount + VLEN, 0.0);
    state.vy.resize(localCount + VLEN, 0.0);
    state.vz.resize(localCount + VLEN, 0.0);
    {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (size_t i = 0; i < n; ++i) {
            state.px[i] = bodies[i].pos.x;
            state.py[i] = bodies[i].pos.y;
            state.pz[i] = bodies[i].pos.z;
        }
        for (size_t i = 0; i < localCount; ++i) {
            state.vx[i] = bodies[begin + i].vel.x;
            state.vy[i] = bodies[begin + i].vel.y;
            state.vz[i] = bodies[begin + i].vel.z;
        }
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(state, n, begin, end);
        integrateBodies(state, begin, end);
        // Exchange updated positions so every rank sees the full system again
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.px.data(), counts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.py.data(), counts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.pz.data(), counts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start);
    long localMs = static_cast<long>(duration.count());
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %ld ms\n", maxMs);

    // Collect the full state on rank 0 for output and validation
    std::vector<Body> bodies;
    if (printResults || validate) {
        std::vector<double> gvx(rank == 0 ? n : 0), gvy(rank == 0 ? n : 0), gvz(rank == 0 ? n : 0);
        MPI_Gatherv(state.vx.data(), counts[rank], MPI_DOUBLE, gvx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
        MPI_Gatherv(state.vy.data(), counts[rank], MPI_DOUBLE, gvy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
        MPI_Gatherv(state.vz.data(), counts[rank], MPI_DOUBLE, gvz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
        if (rank == 0) {
            bodies.resize(n);
            for (size_t i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(state.px[i], state.py[i], state.pz[i]);
                bodies[i].vel = Vec3(gvx[i], gvy[i], gvz[i]);
            }
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = 1;
        if (rank == 0) ok = validateSimulation(bodies) ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (ok) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(state, n, begin, end, rank, size);
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
