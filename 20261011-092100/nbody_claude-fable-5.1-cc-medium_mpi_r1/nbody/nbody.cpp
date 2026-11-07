#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Floating-point note: the file is compiled with -ffp-contract=off and every
// fused multiply-add is written explicitly with std::fma, reproducing the
// contraction pattern of the original serial build. Per-body results are
// therefore bitwise identical to the serial program for any rank count.
constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// Structure-of-arrays body storage. Positions are replicated on every rank
// (needed for the all-pairs force evaluation); velocities are only stored for
// the bodies owned by this rank.
struct Bodies {
    int n = 0;                      // total number of bodies
    int localStart = 0;             // first body owned by this rank
    int localCount = 0;             // number of bodies owned by this rank
    std::vector<double> px, py, pz; // all positions (size n)
    std::vector<double> vx, vy, vz; // local velocities (size localCount)
};

// MPI decomposition info (block distribution of bodies over ranks)
struct Decomp {
    int rank = 0;
    int size = 1;
    std::vector<int> counts;  // bodies per rank
    std::vector<int> displs;  // offset of each rank's block
};

static Decomp makeDecomp(int n, MPI_Comm comm) {
    Decomp d;
    MPI_Comm_rank(comm, &d.rank);
    MPI_Comm_size(comm, &d.size);
    d.counts.resize(d.size);
    d.displs.resize(d.size);
    const int base = n / d.size;
    const int rem = n % d.size;
    int off = 0;
    for (int r = 0; r < d.size; ++r) {
        d.counts[r] = base + (r < rem ? 1 : 0);
        d.displs[r] = off;
        off += d.counts[r];
    }
    return d;
}

// Every rank generates the full (deterministic) initial state so that all
// positions are replicated; each rank keeps only its own velocities.
void randomizeBodies(Bodies& b, unsigned int seed = 42) {
    const int n = b.n;
    b.px.resize(n);
    b.py.resize(n);
    b.pz.resize(n);
    b.vx.resize(b.localCount);
    b.vy.resize(b.localCount);
    b.vz.resize(b.localCount);
    for (int i = 0; i < n; ++i) {
        b.px[i] = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        b.py[i] = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        b.pz[i] = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        const double vx = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        const double vy = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        const double vz = std::fma(2.0, rand_r(&seed) / (double)RAND_MAX, -1.0);
        const int li = i - b.localStart;
        if (li >= 0 && li < b.localCount) {
            b.vx[li] = vx;
            b.vy[li] = vy;
            b.vz[li] = vz;
        }
    }
}

// Compute forces on the locally owned bodies from all bodies and update the
// local velocities. Summation order over j is identical to the serial code.
void computeForces(Bodies& b) {
    const int n = b.n;
    const double* __restrict px = b.px.data();
    const double* __restrict py = b.py.data();
    const double* __restrict pz = b.pz.data();

    // Block width for the SIMD-friendly phase of the inner loop.
    constexpr int B = 16;

    for (int li = 0; li < b.localCount; ++li) {
        const int i = b.localStart + li;
        const double xi = px[i], yi = py[i], zi = pz[i];
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        for (int j0 = 0; j0 < n; j0 += B) {
            const int cnt = std::min(B, n - j0);
            double dxs[B], dys[B], dzs[B], inv3[B];

            // Phase 1: independent per-j work (vectorizable: sqrt/div).
            for (int k = 0; k < cnt; ++k) {
                const int j = j0 + k;
                const double dx = px[j] - xi;
                const double dy = py[j] - yi;
                const double dz = pz[j] - zi;
                const double distSqr = std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                dxs[k] = dx;
                dys[k] = dy;
                dzs[k] = dz;
                inv3[k] = invDist * invDist * invDist;
            }

            // Phase 2: sequential fused accumulation (same rounding as the
            // original scalar loop, in the original j order).
            for (int k = 0; k < cnt; ++k) {
                Fx = std::fma(dxs[k], inv3[k], Fx);
                Fy = std::fma(dys[k], inv3[k], Fy);
                Fz = std::fma(dzs[k], inv3[k], Fz);
            }
        }

        b.vx[li] += DT * Fx;
        b.vy[li] += DT * Fy;
        b.vz[li] += DT * Fz;
    }
}

// Advance locally owned positions, then replicate them to all ranks.
void integrateBodies(Bodies& b, const Decomp& d, MPI_Comm comm) {
    for (int li = 0; li < b.localCount; ++li) {
        const int i = b.localStart + li;
        b.px[i] = std::fma(b.vx[li], DT, b.px[i]);
        b.py[i] = std::fma(b.vy[li], DT, b.py[i]);
        b.pz[i] = std::fma(b.vz[li], DT, b.pz[i]);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.px.data(), d.counts.data(), d.displs.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.py.data(), d.counts.data(), d.displs.data(),
                   MPI_DOUBLE, comm);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b.pz.data(), d.counts.data(), d.displs.data(),
                   MPI_DOUBLE, comm);
}

// Total energy (unit masses): kinetic from local velocities, potential from a
// cyclically distributed share of the i-rows of the pair loop, reduced to rank 0.
double computeTotalEnergy(const Bodies& b, const Decomp& d, MPI_Comm comm) {
    const int n = b.n;
    double kinetic = 0.0;
    for (int li = 0; li < b.localCount; ++li) {
        kinetic += 0.5 * std::fma(b.vz[li], b.vz[li], std::fma(b.vy[li], b.vy[li], b.vx[li] * b.vx[li]));
    }

    const double* __restrict px = b.px.data();
    const double* __restrict py = b.py.data();
    const double* __restrict pz = b.pz.data();
    double potential = 0.0;
    for (int i = d.rank; i < n; i += d.size) {
        const double xi = px[i], yi = py[i], zi = pz[i];
        double rowSum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = px[j] - xi;
            const double dy = py[j] - yi;
            const double dz = pz[j] - zi;
            const double dist = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)) + SOFTENING);
            rowSum -= 1.0 / dist;
        }
        potential += rowSum;
    }

    const double local = kinetic + potential;
    double total = 0.0;
    MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    return total;
}

// Validate that the (gathered, full) simulation state has finite, reasonable
// values. Called on rank 0 only.
bool validateSimulation(const std::vector<double>& bodyData) {
    const size_t n = bodyData.size() / 6;
    for (size_t i = 0; i < n; ++i) {
        const double* s = &bodyData[i * 6];
        // Check for NaN or Inf values
        for (int k = 0; k < 6; ++k) {
            if (!std::isfinite(s[k])) {
                printf("Validation failed: found NaN or Inf value in body state\n");
                return false;
            }
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(s[0]) > maxPos || std::abs(s[1]) > maxPos || std::abs(s[2]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(s[3]) > maxVel || std::abs(s[4]) > maxVel || std::abs(s[5]) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

// Gather the full interleaved body state (pos.xyz, vel.xyz per body) on rank 0.
std::vector<double> gatherBodyState(const Bodies& b, const Decomp& d, MPI_Comm comm) {
    std::vector<double> allVx, allVy, allVz;
    if (d.rank == 0) {
        allVx.resize(b.n);
        allVy.resize(b.n);
        allVz.resize(b.n);
    }
    MPI_Gatherv(b.vx.data(), b.localCount, MPI_DOUBLE, allVx.data(), d.counts.data(), d.displs.data(),
                MPI_DOUBLE, 0, comm);
    MPI_Gatherv(b.vy.data(), b.localCount, MPI_DOUBLE, allVy.data(), d.counts.data(), d.displs.data(),
                MPI_DOUBLE, 0, comm);
    MPI_Gatherv(b.vz.data(), b.localCount, MPI_DOUBLE, allVz.data(), d.counts.data(), d.displs.data(),
                MPI_DOUBLE, 0, comm);

    std::vector<double> bodyData;
    if (d.rank == 0) {
        bodyData.reserve(static_cast<size_t>(b.n) * 6);
        for (int i = 0; i < b.n; ++i) {
            bodyData.push_back(b.px[i]);
            bodyData.push_back(b.py[i]);
            bodyData.push_back(b.pz[i]);
            bodyData.push_back(allVx[i]);
            bodyData.push_back(allVy[i]);
            bodyData.push_back(allVz[i]);
        }
    }
    return bodyData;
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
    const MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on all ranks)
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
    numBodies = std::max(numBodies, 0);

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Decompose and initialize bodies
    const Decomp decomp = makeDecomp(numBodies, comm);
    Bodies bodies;
    bodies.n = numBodies;
    bodies.localStart = decomp.displs[decomp.rank];
    bodies.localCount = decomp.counts[decomp.rank];
    randomizeBodies(bodies);

    // Run simulation
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies, decomp, comm);
    }

    MPI_Barrier(comm);
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);

    if (rank == 0) printf("Simulation time: %ld ms\n", durationMs);

    // Gather full state on rank 0 (needed for results output and validation)
    std::vector<double> bodyData;
    if (printResults || validate) bodyData = gatherBodyState(bodies, decomp, comm);

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = 1;
        if (rank == 0) ok = validateSimulation(bodyData) ? 1 : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);

        if (ok) {
            // Report final energy for reference
            const double finalEnergy = computeTotalEnergy(bodies, decomp, comm);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
