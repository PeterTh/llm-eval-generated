#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

struct Domain {
    size_t n[3], start[3], pitch, plane;
    int parts[3], coord[3], neighbor[6];
    MPI_Comm comm;
    MPI_Datatype face[3];

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * pitch + x;
    }
};

// Choose the largest usable number of ranks, then minimize the total surface
// area of the subdomains. This also permits thin and non-cubic grids.
static int chooseDecomposition(const size_t global[3], int ranks, int parts[3]) {
    ranks = int(std::min(size_t(ranks), global[0] * global[1] * global[2]));
    for (int used = ranks; used >= 1; --used) {
        double best = std::numeric_limits<double>::infinity();
        bool found = false;
        // Prefer contiguous z faces when several layouts have equal area.
        for (int pz = int(std::min(size_t(used), global[2])); pz >= 1; --pz) {
            if (used % pz) continue;
            int rest = used / pz;
            for (int py = int(std::min(size_t(rest), global[1])); py >= 1; --py) {
                if (rest % py) continue;
                int px = rest / py;
                if (static_cast<size_t>(px) > global[0]) continue;
                double surface = double(px - 1) * double(global[1]) * double(global[2])
                               + double(py - 1) * double(global[0]) * double(global[2])
                               + double(pz - 1) * double(global[0]) * double(global[1]);
                if (surface < best) {
                    best = surface;
                    parts[0] = px; parts[1] = py; parts[2] = pz;
                    found = true;
                }
            }
        }
        if (found) return used;
    }
    return 0;
}

static void setupDomain(Domain& d, const size_t global[3], const int parts[3], int rank, MPI_Comm comm) {
    d.comm = comm;
    d.coord[0] = rank % parts[0];
    d.coord[1] = (rank / parts[0]) % parts[1];
    d.coord[2] = rank / (parts[0] * parts[1]);
    for (int a = 0; a < 3; ++a) {
        d.parts[a] = parts[a];
        size_t base = global[a] / parts[a], remainder = global[a] % parts[a];
        d.n[a] = base + (static_cast<size_t>(d.coord[a]) < remainder);
        d.start[a] = size_t(d.coord[a]) * base + std::min(size_t(d.coord[a]), remainder);
    }
    d.pitch = d.n[0] + 2;
    d.plane = d.pitch * (d.n[1] + 2);
    int stride[3] = {1, parts[0], parts[0] * parts[1]};
    for (int a = 0; a < 3; ++a) {
        d.neighbor[2*a] = d.coord[a] > 0 ? rank - stride[a] : MPI_PROC_NULL;
        d.neighbor[2*a+1] = d.coord[a] + 1 < parts[a] ? rank + stride[a] : MPI_PROC_NULL;
    }
    // Face types include neither edges nor corners: the stencil needs only faces.
    MPI_Datatype row;
    MPI_Type_vector(int(d.n[1]), 1, int(d.pitch), MPI_DOUBLE, &row);
    MPI_Type_create_hvector(int(d.n[2]), 1, MPI_Aint(d.plane * sizeof(double)), row, &d.face[0]);
    MPI_Type_commit(&d.face[0]);
    MPI_Type_free(&row);
    MPI_Type_create_hvector(int(d.n[2]), int(d.n[0]), MPI_Aint(d.plane * sizeof(double)), MPI_DOUBLE, &d.face[1]);
    MPI_Type_commit(&d.face[1]);
    MPI_Type_vector(int(d.n[1]), int(d.n[0]), int(d.pitch), MPI_DOUBLE, &d.face[2]);
    MPI_Type_commit(&d.face[2]);
}

// Start communication and fill the faces on the physical domain boundary with
// the adjacent cell, implementing the original clamped boundary conditions.
static int beginExchange(std::vector<double>& v, const Domain& d, MPI_Request req[12]) {
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    if (d.neighbor[0] == MPI_PROC_NULL || d.neighbor[1] == MPI_PROC_NULL)
        for (size_t z = 1; z <= nz; ++z) for (size_t y = 1; y <= ny; ++y) {
            if (d.neighbor[0] == MPI_PROC_NULL) v[d.index(0,y,z)] = v[d.index(1,y,z)];
            if (d.neighbor[1] == MPI_PROC_NULL) v[d.index(nx+1,y,z)] = v[d.index(nx,y,z)];
        }
    if (d.neighbor[2] == MPI_PROC_NULL || d.neighbor[3] == MPI_PROC_NULL)
        for (size_t z = 1; z <= nz; ++z) for (size_t x = 1; x <= nx; ++x) {
            if (d.neighbor[2] == MPI_PROC_NULL) v[d.index(x,0,z)] = v[d.index(x,1,z)];
            if (d.neighbor[3] == MPI_PROC_NULL) v[d.index(x,ny+1,z)] = v[d.index(x,ny,z)];
        }
    if (d.neighbor[4] == MPI_PROC_NULL || d.neighbor[5] == MPI_PROC_NULL)
        for (size_t y = 1; y <= ny; ++y) for (size_t x = 1; x <= nx; ++x) {
            if (d.neighbor[4] == MPI_PROC_NULL) v[d.index(x,y,0)] = v[d.index(x,y,1)];
            if (d.neighbor[5] == MPI_PROC_NULL) v[d.index(x,y,nz+1)] = v[d.index(x,y,nz)];
        }
    size_t lowGhost[3] = {d.index(0,1,1), d.index(1,0,1), d.index(1,1,0)};
    size_t highGhost[3] = {d.index(nx+1,1,1), d.index(1,ny+1,1), d.index(1,1,nz+1)};
    size_t lowFace[3] = {d.index(1,1,1), d.index(1,1,1), d.index(1,1,1)};
    size_t highFace[3] = {d.index(nx,1,1), d.index(1,ny,1), d.index(1,1,nz)};
    int count = 0;
    for (int a = 0; a < 3; ++a) {
        if (d.neighbor[2*a] != MPI_PROC_NULL) {
            MPI_Irecv(v.data()+lowGhost[a], 1, d.face[a], d.neighbor[2*a], 2*a+1, d.comm, &req[count++]);
            MPI_Isend(v.data()+lowFace[a], 1, d.face[a], d.neighbor[2*a], 2*a, d.comm, &req[count++]);
        }
        if (d.neighbor[2*a+1] != MPI_PROC_NULL) {
            MPI_Irecv(v.data()+highGhost[a], 1, d.face[a], d.neighbor[2*a+1], 2*a, d.comm, &req[count++]);
            MPI_Isend(v.data()+highFace[a], 1, d.face[a], d.neighbor[2*a+1], 2*a+1, d.comm, &req[count++]);
        }
    }
    return count;
}

// Preserve the arithmetic and its order from the original two stencil passes.
template<bool Chemical>
static inline void computeRow(const std::vector<double>& input, const std::vector<double>& cold,
                              std::vector<double>& output, const Domain& d,
                              size_t xbegin, size_t xend, size_t y, size_t z) {
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    size_t i = d.index(xbegin, y, z);
    for (size_t x = xbegin; x < xend; ++x, ++i) {
        double center = input[i];
        double cxx = (input[i+1] + input[i-1] - 2.0 * center) / (1.0 * 1.0);
        double cyy = (input[i+d.pitch] + input[i-d.pitch] - 2.0 * center) / (1.0 * 1.0);
        double czz = (input[i+d.plane] + input[i-d.plane] - 2.0 * center) / (1.0 * 1.0);
        double lap = cxx + cyy + czz;
        if constexpr (Chemical)
            output[i] = 4.5 * ((center + 1.0) * e_AA + (center - 1.0) * e_BB - 2.0 * center * e_AB)
                        + 3.0 * center + center * center * center - 0.5 * lap;
        else
            output[i] = cold[i] + 0.01 * 1.0 * lap;
    }
}

template<bool Chemical>
static void stencil(std::vector<double>& input, const std::vector<double>& cold,
                    std::vector<double>& output, const Domain& d) {
    MPI_Request req[12];
    int count = beginExchange(input, d, req);
    const size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    for (size_t z = 2; z < nz; ++z)
        for (size_t y = 2; y < ny; ++y)
            if (nx > 2) computeRow<Chemical>(input, cold, output, d, 2, nx, y, z);
    MPI_Waitall(count, req, MPI_STATUSES_IGNORE);
    for (size_t z = 1; z <= nz; ++z) for (size_t y = 1; y <= ny; ++y) {
        if (z == 1 || z == nz || y == 1 || y == ny || nx <= 2) {
            computeRow<Chemical>(input, cold, output, d, 1, nx+1, y, z);
        } else {
            computeRow<Chemical>(input, cold, output, d, 1, 2, y, z);
            computeRow<Chemical>(input, cold, output, d, nx, nx+1, y, z);
        }
    }
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Pack in x-fastest order. Results are reconstructed at rank zero because the
// existing results helper computes an order-dependent hash and Kahan sum.
static std::vector<double> pack(const std::vector<double>& field, const Domain& d) {
    std::vector<double> data(d.n[0] * d.n[1] * d.n[2]);
    size_t at = 0;
    for (size_t z = 1; z <= d.n[2]; ++z) for (size_t y = 1; y <= d.n[1]; ++y) {
        std::copy_n(field.data() + d.index(1,y,z), d.n[0], data.data() + at);
        at += d.n[0];
    }
    return data;
}

static void transfer(double* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        int chunk = int(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_DOUBLE, peer, 0, comm);
        else MPI_Recv(data, chunk, MPI_DOUBLE, peer, 0, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

static void printGlobalResults(const std::vector<double>& field, const Domain& d,
                               const size_t global[3], int rank, int used) {
    std::vector<double> local = pack(field, d);
    if (rank != 0) {
        transfer(local.data(), local.size(), 0, true, d.comm);
        return;
    }
    std::vector<double> all(global[0] * global[1] * global[2]);
    for (int r = 0; r < used; ++r) {
        int coord[3] = {r % d.parts[0], (r / d.parts[0]) % d.parts[1], r / (d.parts[0] * d.parts[1])};
        size_t n[3], start[3];
        for (int a = 0; a < 3; ++a) {
            size_t base = global[a] / d.parts[a], rem = global[a] % d.parts[a];
            n[a] = base + (size_t(coord[a]) < rem);
            start[a] = size_t(coord[a]) * base + std::min(size_t(coord[a]), rem);
        }
        std::vector<double> received;
        const double* data = local.data();
        if (r != 0) {
            received.resize(n[0] * n[1] * n[2]);
            transfer(received.data(), received.size(), r, false, d.comm);
            data = received.data();
        }
        size_t at = 0;
        for (size_t z = 0; z < n[2]; ++z) for (size_t y = 0; y < n[1]; ++y) {
            size_t dest = ((z + start[2]) * global[1] + y + start[1]) * global[0] + start[0];
            std::copy_n(data + at, n[0], all.data() + dest);
            at += n[0];
        }
    }
    print_results(all, "Concentration");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20, exitCode = 0;
    bool validate = false, printResults = false, done = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); done = true; break; }
        else { if (worldRank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } exitCode = 1; done = true; break; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    // MPI vector datatype counts and strides are int, and all sizes must fit.
    if (!done && (nx == 0 || nx > size_t(INT_MAX-2) || ny > size_t(INT_MAX-2) || nz > size_t(INT_MAX-2)
        || nx > std::numeric_limits<size_t>::max() / ny || nx * ny > std::numeric_limits<size_t>::max() / nz)) {
        if (worldRank == 0) fprintf(stderr, "Invalid or unsupported grid size\n");
        exitCode = 1; done = true;
    }
    if (done) { MPI_Finalize(); return exitCode; }
    const size_t global[3] = {nx, ny, nz};
    const size_t gridSize = nx * ny * nz;
    int parts[3];
    int used = chooseDecomposition(global, worldSize, parts);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < used ? 0 : MPI_UNDEFINED, worldRank, &active);
    if (worldRank >= used) { MPI_Finalize(); return 0; }
    Domain d;
    setupDomain(d, global, parts, worldRank, active);
    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    size_t allocated = d.plane * (d.n[2] + 2);
    std::vector<double> cold(allocated), cnew(allocated), mu(allocated);
    for (size_t z = 1; z <= d.n[2]; ++z) for (size_t y = 1; y <= d.n[1]; ++y) for (size_t x = 1; x <= d.n[0]; ++x) {
        size_t linear_id = (z - 1 + d.start[2]) * (nx * ny) + (y - 1 + d.start[1]) * nx + x - 1 + d.start[0];
        double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[d.index(x,y,z)] = -1.0 + 2.0 * pseudo;
    }
    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(active);
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        stencil<true>(cold, cold, mu, d);
        stencil<false>(mu, cold, cnew, d);
        std::swap(cold, cnew);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count(), maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    if (worldRank == 0) {
        long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        double cellUpdates = double(gridSize) * iterations;
        double mcups = cellUpdates / maxElapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) printGlobalResults(cold, d, global, worldRank, used);
    if (validate) {
        int finite = 1, allFinite;
        double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
        for (size_t z = 1; z <= d.n[2]; ++z) for (size_t y = 1; y <= d.n[1]; ++y) for (size_t x = 1; x <= d.n[0]; ++x) {
            double v = cold[d.index(x,y,z)];
            if (!std::isfinite(v)) finite = 0;
            minimum = std::min(minimum, v); maximum = std::max(maximum, v);
        }
        double globalMin, globalMax;
        MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, active);
        MPI_Reduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, active);
        MPI_Reduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, active);
        if (worldRank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0) printf("Validation failed: values out of expected range\n");
            }
            bool valid = allFinite && globalMax <= 10.0 && globalMin >= -10.0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    for (auto& type : d.face) MPI_Type_free(&type);
    MPI_Comm_free(&active);
    MPI_Finalize();
    return exitCode;
}
