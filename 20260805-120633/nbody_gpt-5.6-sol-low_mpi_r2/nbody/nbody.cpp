#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Body {
    double px, py, pz;
    double vx, vy, vz;
};

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& b : bodies) {
        b.px = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.py = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.pz = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vx = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vy = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        b.vz = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation (checks energy conservation)\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& b : bodies) {
        if (!std::isfinite(b.px) || !std::isfinite(b.py) || !std::isfinite(b.pz) ||
            !std::isfinite(b.vx) || !std::isfinite(b.vy) || !std::isfinite(b.vz)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        if (std::abs(b.px) > 1e6 || std::abs(b.py) > 1e6 || std::abs(b.pz) > 1e6) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(b.vx) > 1e6 || std::abs(b.vy) > 1e6 || std::abs(b.vz) > 1e6) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    for (const auto& b : bodies)
        energy += 0.5 * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        for (std::size_t j = i + 1; j < bodies.size(); ++j) {
            const double dx = bodies[j].px - bodies[i].px;
            const double dy = bodies[j].py - bodies[i].py;
            const double dz = bodies[j].pz - bodies[i].pz;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) parseStatus = 2;
        else {
            if (rank == 0) std::printf("Unknown option: %s\n", argv[i]);
            parseStatus = 1;
            break;
        }
    }
    if (numBodies < 0 || numSteps < 0 || numBodies > std::numeric_limits<int>::max() / 6) {
        if (rank == 0) std::printf("Body and step counts must be non-negative and representable by MPI counts.\n");
        parseStatus = 1;
    }
    if (parseStatus != 0) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        std::printf("N-Body Simulation\n");
        std::printf("Number of bodies: %d\n", numBodies);
        std::printf("Number of steps: %d\n", numSteps);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(nranks), offsets(nranks), bodyCounts(nranks), bodyOffsets(nranks);
    for (int r = 0; r < nranks; ++r) {
        counts[r] = numBodies / nranks + (r < numBodies % nranks);
        offsets[r] = r * (numBodies / nranks) + std::min(r, numBodies % nranks);
        bodyCounts[r] = 6 * counts[r];
        bodyOffsets[r] = 6 * offsets[r];
    }
    const int localN = counts[rank];

    std::vector<Body> initial;
    std::vector<double> packedInitial;
    if (rank == 0) {
        initial.resize(numBodies);
        randomizeBodies(initial);
        packedInitial.resize(static_cast<std::size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) {
            packedInitial[6 * i + 0] = initial[i].px; packedInitial[6 * i + 1] = initial[i].py;
            packedInitial[6 * i + 2] = initial[i].pz; packedInitial[6 * i + 3] = initial[i].vx;
            packedInitial[6 * i + 4] = initial[i].vy; packedInitial[6 * i + 5] = initial[i].vz;
        }
    }
    std::vector<double> localPacked(static_cast<std::size_t>(localN) * 6);
    MPI_Scatterv(packedInitial.data(), bodyCounts.data(), bodyOffsets.data(), MPI_DOUBLE,
                 localPacked.data(), 6 * localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    initial.clear(); packedInitial.clear();

    std::vector<double> lx(localN), ly(localN), lz(localN), lvx(localN), lvy(localN), lvz(localN);
    for (int i = 0; i < localN; ++i) {
        lx[i] = localPacked[6*i]; ly[i] = localPacked[6*i+1]; lz[i] = localPacked[6*i+2];
        lvx[i] = localPacked[6*i+3]; lvy[i] = localPacked[6*i+4]; lvz[i] = localPacked[6*i+5];
    }
    localPacked.clear(); localPacked.shrink_to_fit();
    std::vector<double> gx(numBodies), gy(numBodies), gz(numBodies);

    MPI_Request requests[3];
    auto exchangePositions = [&]() {
        MPI_Iallgatherv(lx.data(), localN, MPI_DOUBLE, gx.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD, &requests[0]);
        MPI_Iallgatherv(ly.data(), localN, MPI_DOUBLE, gy.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD, &requests[1]);
        MPI_Iallgatherv(lz.data(), localN, MPI_DOUBLE, gz.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD, &requests[2]);
        MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
    };
    exchangePositions();
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        for (int i = 0; i < localN; ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            const double xi = lx[i], yi = ly[i], zi = lz[i];
            for (int j = 0; j < numBodies; ++j) {
                const double dx = gx[j] - xi, dy = gy[j] - yi, dz = gz[j] - zi;
                const double inv = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
            }
            lvx[i] += DT * fx; lvy[i] += DT * fy; lvz[i] += DT * fz;
            lx[i] += DT * lvx[i]; ly[i] += DT * lvy[i]; lz[i] += DT * lvz[i];
        }
        if (step + 1 < numSteps) exchangePositions();
    }
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> result;
    if (printResults || validate) {
        std::vector<double> localResult(static_cast<std::size_t>(localN) * 6);
        for (int i = 0; i < localN; ++i) {
            localResult[6*i] = lx[i]; localResult[6*i+1] = ly[i]; localResult[6*i+2] = lz[i];
            localResult[6*i+3] = lvx[i]; localResult[6*i+4] = lvy[i]; localResult[6*i+5] = lvz[i];
        }
        if (rank == 0) result.resize(static_cast<std::size_t>(numBodies) * 6);
        MPI_Gatherv(localResult.data(), 6 * localN, MPI_DOUBLE, result.data(), bodyCounts.data(),
                    bodyOffsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        if (printResults) print_results(result, "Bodies");
        if (validate) {
            std::printf("Validating simulation results...\n");
            std::vector<Body> bodies(numBodies);
            for (int i = 0; i < numBodies; ++i)
                bodies[i] = {result[6*i], result[6*i+1], result[6*i+2], result[6*i+3], result[6*i+4], result[6*i+5]};
            if (validateSimulation(bodies)) {
                std::printf("Final energy: %.6f\n", computeTotalEnergy(bodies));
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
