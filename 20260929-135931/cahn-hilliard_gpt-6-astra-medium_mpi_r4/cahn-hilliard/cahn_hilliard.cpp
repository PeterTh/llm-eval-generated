#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Select a Cartesian grid that uses as many ranks as the domain permits, then
// minimizes the face area exchanged. This also handles thin and uneven grids.
std::array<int, 3> processGrid(const std::array<int, 3>& global, int ranks) {
    std::array<int, 3> best{1, 1, 1};
    int used = 0;
    double bestArea = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= std::min(global[0], ranks); ++x) {
        for (int y = 1; y <= std::min(global[1], ranks / x); ++y) {
            int z = std::min(global[2], ranks / (x * y));
            int count = x * y * z;
            double area = double(x - 1) * global[1] * global[2]
                        + double(y - 1) * global[0] * global[2]
                        + double(z - 1) * global[0] * global[1];
            if (count > used || (count == used && area < bestArea)) {
                best = {x, y, z};
                used = count;
                bestArea = area;
            }
        }
    }
    return best;
}

struct Domain {
    MPI_Comm comm;
    std::array<int, 3> global, dims, n, start;
    std::array<int, 6> neighbor;
    std::array<MPI_Datatype, 3> face;
    size_t row, plane, volume;

    Domain(MPI_Comm active, std::array<int, 3> globalSize,
           std::array<int, 3> processDims) : global(globalSize), dims(processDims) {
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims.data(), periods, 0, &comm);
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coords[a] < global[a] % dims[a]);
            start[a] = coords[a] * (global[a] / dims[a])
                     + std::min(coords[a], global[a] % dims[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        row = size_t(n[0]) + 2;
        plane = row * (size_t(n[1]) + 2);
        if (plane > std::numeric_limits<size_t>::max() / (size_t(n[2]) + 2)
            / sizeof(double)) throw std::runtime_error("Local grid is too large");
        volume = plane * (size_t(n[2]) + 2);
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        int origin[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) {
            int subsizes[3] = {n[2], n[1], n[0]};
            subsizes[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, origin, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
    }
    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Comm_free(&comm);
    }
    size_t index(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * row + size_t(x);
    }

    // Faces exclude edges and corners: both operators use a seven-point stencil.
    int exchange(std::vector<double>& field, MPI_Request* requests) const {
        int count = 0;
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                int peer = neighbor[2*a+side];
                if (peer == MPI_PROC_NULL) continue;
                int recv[3] = {1, 1, 1}, send[3] = {1, 1, 1};
                recv[a] = side ? n[a]+1 : 0;
                send[a] = side ? n[a] : 1;
                MPI_Irecv(field.data() + index(recv[0], recv[1], recv[2]),
                          1, face[a], peer, 2*a+1-side, comm, &requests[count++]);
                MPI_Isend(field.data() + index(send[0], send[1], send[2]),
                          1, face[a], peer, 2*a+side, comm, &requests[count++]);
            }
        }
        return count;
    }
};

// Unit grid spacings and physical constants are those of the original benchmark.
// Keep the expression order intact; no reassociation/fast-math is required.
template<bool chemical>
void computeBox(const Domain& d, const std::vector<double>& input,
                const std::vector<double>& cold, std::vector<double>& output,
                int xb, int xe, int yb, int ye, int zb, int ze) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (int z = zb; z < ze; ++z) {
        size_t zm = (z == 1 && d.neighbor[4] == MPI_PROC_NULL) ? 0 : d.plane;
        size_t zp = (z == d.n[2] && d.neighbor[5] == MPI_PROC_NULL) ? 0 : d.plane;
        for (int y = yb; y < ye; ++y) {
            size_t ym = (y == 1 && d.neighbor[2] == MPI_PROC_NULL) ? 0 : d.row;
            size_t yp = (y == d.n[1] && d.neighbor[3] == MPI_PROC_NULL) ? 0 : d.row;
            size_t base = d.index(0, y, z);
            auto cell = [&](int x, size_t xm, size_t xp) {
                size_t i = base + x;
                double cv = input[i];
                double cxx = input[i+xp] + input[i-xm] - 2.0 * cv;
                double cyy = input[i+yp] + input[i-ym] - 2.0 * cv;
                double czz = input[i+zp] + input[i-zm] - 2.0 * cv;
                double lap = cxx + cyy + czz;
                if constexpr (chemical) {
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                } else {
                    output[i] = cold[i] + 0.01 * lap;
                }
            };
            // Separate the two physical endpoints so the long X loop uses
            // contiguous vector loads, with no per-lane clamps or gathers.
            int begin = xb, end = xe;
            bool rightClamped = d.neighbor[1] == MPI_PROC_NULL;
            if (begin < end && begin == 1 && d.neighbor[0] == MPI_PROC_NULL) {
                cell(begin, 0, (begin == d.n[0] && rightClamped) ? 0 : 1);
                ++begin;
            }
            if (begin < end && end == d.n[0]+1 && rightClamped) {
                --end;
                cell(end, 1, 0);
            }
            for (int x = begin; x < end; ++x) cell(x, 1, 1);
        }
    }
}

template<bool chemical>
void step(const Domain& d, std::vector<double>& input,
          const std::vector<double>& cold, std::vector<double>& output) {
    MPI_Request requests[12];
    int count = d.exchange(input, requests);
    computeBox<chemical>(d, input, cold, output, 2, d.n[0], 2, d.n[1], 2, d.n[2]);
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    // Disjoint boundary boxes avoid both duplicate work and per-cell tests.
    int nx = d.n[0], ny = d.n[1], nz = d.n[2];
    computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, ny+1, 1, 2);
    if (nz > 1) computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, ny+1, nz, nz+1);
    computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, 2, 2, nz);
    if (ny > 1) computeBox<chemical>(d, input, cold, output, 1, nx+1, ny, ny+1, 2, nz);
    computeBox<chemical>(d, input, cold, output, 1, 2, 2, ny, 2, nz);
    if (nx > 1) computeBox<chemical>(d, input, cold, output, nx, nx+1, 2, ny, 2, nz);
}

// Gathering is only needed for the optional order-sensitive hash and Kahan sum.
// Chunked transfers avoid MPI's int count limit and need only one block of scratch.
void printDistributedResults(const Domain& d, const std::vector<double>& c) {
    int rank, ranks;
    MPI_Comm_rank(d.comm, &rank);
    MPI_Comm_size(d.comm, &ranks);
    std::vector<double> packed(size_t(d.n[0]) * d.n[1] * d.n[2]);
    size_t pos = 0;
    for (int z = 1; z <= d.n[2]; ++z)
        for (int y = 1; y <= d.n[1]; ++y) {
            std::copy_n(c.data() + d.index(1, y, z), d.n[0], packed.data() + pos);
            pos += d.n[0];
        }
    constexpr size_t chunk = std::numeric_limits<int>::max();
    if (rank != 0) {
        for (size_t offset = 0; offset < packed.size(); offset += chunk)
            MPI_Send(packed.data()+offset, int(std::min(chunk, packed.size()-offset)),
                     MPI_DOUBLE, 0, 20, d.comm);
        return;
    }
    std::vector<double> full(size_t(d.global[0]) * d.global[1] * d.global[2]);
    for (int peer = 0; peer < ranks; ++peer) {
        int coords[3], n[3], start[3];
        MPI_Cart_coords(d.comm, peer, 3, coords);
        for (int a = 0; a < 3; ++a) {
            n[a] = d.global[a] / d.dims[a] + (coords[a] < d.global[a] % d.dims[a]);
            start[a] = coords[a] * (d.global[a] / d.dims[a])
                     + std::min(coords[a], d.global[a] % d.dims[a]);
        }
        packed.resize(size_t(n[0]) * n[1] * n[2]);
        if (peer != 0)
            for (size_t offset = 0; offset < packed.size(); offset += chunk)
                MPI_Recv(packed.data()+offset, int(std::min(chunk, packed.size()-offset)),
                         MPI_DOUBLE, peer, 20, d.comm, MPI_STATUS_IGNORE);
        pos = 0;
        for (int z = 0; z < n[2]; ++z)
            for (int y = 0; y < n[1]; ++y) {
                size_t dest = (size_t(z+start[2]) * d.global[1] + y+start[1]) * d.global[0] + start[0];
                std::copy_n(packed.data()+pos, n[0], full.data()+dest);
                pos += n[0];
            }
    }
    print_results(full, "Concentration");
}

bool validateResult(const Domain& d, const std::vector<double>& c) {
    double minimum = std::numeric_limits<double>::infinity(), maximum = -minimum;
    int invalid = 0;
    for (int z = 1; z <= d.n[2]; ++z)
        for (int y = 1; y <= d.n[1]; ++y)
            for (int x = 1; x <= d.n[0]; ++x) {
                double v = c[d.index(x, y, z)];
                invalid |= !std::isfinite(v);
                minimum = std::min(minimum, v);
                maximum = std::max(maximum, v);
            }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, d.comm);
    double bounds[2] = {-minimum, maximum};
    MPI_Allreduce(MPI_IN_PLACE, bounds, 2, MPI_DOUBLE, MPI_MAX, d.comm);
    int rank;
    MPI_Comm_rank(d.comm, &rank);
    bool valid = !invalid && bounds[0] <= 10.0 && bounds[1] <= 10.0;
    if (rank == 0) {
        if (invalid) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", -bounds[0], bounds[1]);
            if (!valid) printf("Validation failed: values out of expected range\n");
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

void printUsage(const char* progName) {
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

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int nx = 64, ny = 0, nz = 0, iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-x") || !strcmp(argv[i], "-y") ||
             !strcmp(argv[i], "-z") || !strcmp(argv[i], "-i")) && i+1 < argc) {
            char option = argv[i][1];
            char* end = nullptr;
            long value = std::strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < 0 || value > std::numeric_limits<int>::max()-2 ||
                (option == 'x' && value == 0)) {
                if (rank == 0) fprintf(stderr, "Invalid value for -%c: %s\n", option, argv[i]);
                return 1;
            }
            if (option == 'x') nx = int(value);
            if (option == 'y') ny = int(value);
            if (option == 'z') nz = int(value);
            if (option == 'i') iterations = int(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (size_t(nx) > std::numeric_limits<size_t>::max() / size_t(ny) / size_t(nz) / sizeof(double))
        throw std::runtime_error("Global grid is too large");
    size_t gridSize = size_t(nx) * ny * nz;
    auto dims = processGrid({nx, ny, nz}, ranks);
    int activeRanks = dims[0] * dims[1] * dims[2];
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL) return 0;
    Domain d(active, {nx, ny, nz}, dims);
    MPI_Comm_free(&active);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %d x %d x %d\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold(d.volume), cnew(d.volume), mu(d.volume);
    for (int z = 1; z <= d.n[2]; ++z)
        for (int y = 1; y <= d.n[1]; ++y)
            for (int x = 1; x <= d.n[0]; ++x) {
                size_t linear = (size_t(d.start[2]+z-1) * ny + d.start[1]+y-1) * nx + d.start[0]+x-1;
                double pseudo = (((linear + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
                cold[d.index(x, y, z)] = -1.0 + 2.0 * pseudo;
            }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(d.comm);
    double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        step<true>(d, cold, cold, mu);
        step<false>(d, mu, cold, cnew);
        cold.swap(cnew);
    }
    double elapsed = MPI_Wtime() - start, duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000.0));
        double mcups = duration > 0 ? double(gridSize) * iterations / duration / 1e6 : 0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) printDistributedResults(d, cold);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        return validateResult(d, cold) ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 0;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Idle ranks participate here too, including in a failed validation status.
    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
