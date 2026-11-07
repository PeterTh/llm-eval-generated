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

// Structure-of-arrays body storage.
// Positions are replicated on every rank (all-gathered after each step);
// velocities are only stored for the locally owned block of bodies.
struct Bodies {
    size_t n = 0;            // total number of bodies
    size_t begin = 0;        // first locally owned body (global index)
    size_t count = 0;        // number of locally owned bodies
    std::vector<double> px, py, pz;  // global positions (size n)
    std::vector<double> vx, vy, vz;  // local velocities (size count)
};

// Block decomposition of [0, n) over `size` ranks: returns counts/displacements.
void blockPartition(size_t n, int size, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    const size_t base = n / size;
    const size_t rem = n % size;
    size_t off = 0;
    for (int r = 0; r < size; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(off);
        off += c;
    }
}

// Generates exactly the same sequence as the original implementation: every
// rank replays the generator, but only stores the data it needs (full positions,
// local velocities).
// (2.0 * r - 1.0 is written as an explicit fma to match the original build.)
static inline double nextUniform(unsigned int& seed) {
    return std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
}

void randomizeBodies(Bodies& b, unsigned int seed = 42) {
    const size_t end = b.begin + b.count;
    for (size_t i = 0; i < b.n; ++i) {
        const double x = nextUniform(seed);
        const double y = nextUniform(seed);
        const double z = nextUniform(seed);
        const double vx = nextUniform(seed);
        const double vy = nextUniform(seed);
        const double vz = nextUniform(seed);
        b.px[i] = x;
        b.py[i] = y;
        b.pz[i] = z;
        if (i >= b.begin && i < end) {
            b.vx[i - b.begin] = vx;
            b.vy[i - b.begin] = vy;
            b.vz[i - b.begin] = vz;
        }
    }
}

// Number of bodies processed simultaneously by the force kernel. The inner
// j loop is executed lane-wise for LANES different bodies i, so the compiler can
// vectorize across i while every lane performs exactly the scalar sequence of
// operations of the original implementation.
constexpr size_t LANES = 8;

// Accumulates the force on LANES bodies (positions xi/yi/zi) from all n bodies.
// The arithmetic is spelled out with explicit fma() calls to match the
// fused-multiply-add pattern of the original (-O3 -march=native) scalar loop,
// so the results are bitwise identical to the original program.
static inline void forceKernel(const size_t n,
                               const double* __restrict px,
                               const double* __restrict py,
                               const double* __restrict pz,
                               const double* __restrict xi,
                               const double* __restrict yi,
                               const double* __restrict zi,
                               double* __restrict Fx,
                               double* __restrict Fy,
                               double* __restrict Fz) {
    for (size_t l = 0; l < LANES; ++l) {
        Fx[l] = 0.0;
        Fy[l] = 0.0;
        Fz[l] = 0.0;
    }
    for (size_t j = 0; j < n; ++j) {
        const double xj = px[j], yj = py[j], zj = pz[j];
        for (size_t l = 0; l < LANES; ++l) {
            const double dx = xj - xi[l];
            const double dy = yj - yi[l];
            const double dz = zj - zi[l];
            double distSqr = dy * dy;
            distSqr = std::fma(dx, dx, distSqr);
            distSqr = std::fma(dz, dz, distSqr);
            distSqr += SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx[l] = std::fma(dx, invDist3, Fx[l]);
            Fy[l] = std::fma(dy, invDist3, Fy[l]);
            Fz[l] = std::fma(dz, invDist3, Fz[l]);
        }
    }
}

// Computes the force on every locally owned body from all n bodies and updates
// the local velocities. Each lane runs over j in the original order, so the
// floating-point accumulation order (and thus the result) is unchanged.
void computeForces(Bodies& b) {
    const size_t n = b.n;
    const double* __restrict px = b.px.data();
    const double* __restrict py = b.py.data();
    const double* __restrict pz = b.pz.data();

    alignas(64) double xi[LANES], yi[LANES], zi[LANES];
    alignas(64) double Fx[LANES], Fy[LANES], Fz[LANES];

    for (size_t li = 0; li < b.count; li += LANES) {
        const size_t lanes = std::min(LANES, b.count - li);
        for (size_t l = 0; l < LANES; ++l) {
            // Pad the tail block with copies of its last valid body.
            const size_t i = b.begin + li + std::min(l, lanes - 1);
            xi[l] = px[i];
            yi[l] = py[i];
            zi[l] = pz[i];
        }

        forceKernel(n, px, py, pz, xi, yi, zi, Fx, Fy, Fz);

        for (size_t l = 0; l < lanes; ++l) {
            b.vx[li + l] += DT * Fx[l];
            b.vy[li + l] += DT * Fy[l];
            b.vz[li + l] += DT * Fz[l];
        }
    }
}

// Integrates the locally owned positions in place (inside the global arrays).
// pos += vel * DT is written as an explicit fma to match the original build.
void integrateBodies(Bodies& b) {
    for (size_t li = 0; li < b.count; ++li) {
        const size_t i = b.begin + li;
        b.px[i] = std::fma(b.vx[li], DT, b.px[i]);
        b.py[i] = std::fma(b.vy[li], DT, b.py[i]);
        b.pz[i] = std::fma(b.vz[li], DT, b.pz[i]);
    }
}

// Exchanges the updated positions so every rank holds the full set again.
void exchangePositions(Bodies& b, const std::vector<int>& counts, const std::vector<int>& displs) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

// Total energy (unit masses). Kinetic part from local velocities, potential part
// over pairs i<j with the i rows distributed cyclically for load balance; the
// partial sums are reduced onto rank 0 (only rank 0's return value is meaningful).
double computeTotalEnergy(const Bodies& b, int rank, int size) {
    double energy = 0.0;
    const size_t n = b.n;

    // Kinetic energy (assuming unit mass)
    for (size_t li = 0; li < b.count; ++li) {
        energy += 0.5 * (b.vx[li] * b.vx[li] +
                         b.vy[li] * b.vy[li] +
                         b.vz[li] * b.vz[li]);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size)) {
        const double xi = b.px[i], yi = b.py[i], zi = b.pz[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = b.px[j] - xi;
            const double dy = b.py[j] - yi;
            const double dz = b.pz[j] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    double total = 0.0;
    MPI_Reduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values (local bodies only;
// the caller combines the per-rank verdicts).
bool validateSimulation(const Bodies& b) {
    for (size_t li = 0; li < b.count; ++li) {
        const size_t i = b.begin + li;
        // Check for NaN or Inf values
        if (!std::isfinite(b.px[i]) || !std::isfinite(b.py[i]) || !std::isfinite(b.pz[i]) ||
            !std::isfinite(b.vx[li]) || !std::isfinite(b.vy[li]) || !std::isfinite(b.vz[li])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(b.px[i]) > maxPos || std::abs(b.py[i]) > maxPos || std::abs(b.pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(b.vx[li]) > maxVel || std::abs(b.vy[li]) > maxVel || std::abs(b.vz[li]) > maxVel) {
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

    // Parse command line arguments (identical on every rank)
    int exitCode = -1;
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
            exitCode = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            break;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    // Domain decomposition: contiguous block of bodies per rank
    std::vector<int> counts, displs;
    blockPartition(static_cast<size_t>(numBodies), size, counts, displs);

    Bodies bodies;
    bodies.n = static_cast<size_t>(numBodies);
    bodies.begin = static_cast<size_t>(displs[rank]);
    bodies.count = static_cast<size_t>(counts[rank]);
    bodies.px.resize(bodies.n);
    bodies.py.resize(bodies.n);
    bodies.pz.resize(bodies.n);
    bodies.vx.resize(bodies.count);
    bodies.vy.resize(bodies.count);
    bodies.vz.resize(bodies.count);
    randomizeBodies(bodies);

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
        exchangePositions(bodies, counts, displs);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long global_ms;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", global_ms);

    // Print results for external validation
    if (printResults) {
        // Gather velocities onto rank 0 (positions are already replicated)
        std::vector<double> gvx, gvy, gvz;
        if (rank == 0) {
            gvx.resize(bodies.n);
            gvy.resize(bodies.n);
            gvz.resize(bodies.n);
        }
        MPI_Gatherv(bodies.vx.data(), counts[rank], MPI_DOUBLE, gvx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(bodies.vy.data(), counts[rank], MPI_DOUBLE, gvy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(bodies.vz.data(), counts[rank], MPI_DOUBLE, gvz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(bodies.n * 6);
            for (size_t i = 0; i < bodies.n; ++i) {
                bodyData.push_back(bodies.px[i]);
                bodyData.push_back(bodies.py[i]);
                bodyData.push_back(bodies.pz[i]);
                bodyData.push_back(gvx[i]);
                bodyData.push_back(gvy[i]);
                bodyData.push_back(gvz[i]);
            }
            print_results(bodyData, "Bodies");
        }
    }

    // Validation: check that simulation produces finite, reasonable values
    int result = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        const int localOk = validateSimulation(bodies) ? 1 : 0;
        int allOk = 0;
        MPI_Allreduce(&localOk, &allOk, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (allOk) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, size);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            result = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            result = 1;
        }
    }

    MPI_Finalize();
    return result;
}
