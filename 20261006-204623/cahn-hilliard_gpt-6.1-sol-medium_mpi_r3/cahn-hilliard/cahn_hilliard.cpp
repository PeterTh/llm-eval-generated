#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <exception>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Each rank stores a rectangular block and one ghost cell on every face.
struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int dims[3] = {1, 1, 1}, coords[3] = {}, neighbor[6];
    size_t global[3], n[3], offset[3], stride[3], storage;
    MPI_Datatype face[3];

    Domain(size_t nx, size_t ny, size_t nz, int worldSize, int worldRank)
        : global{nx, ny, nz} {
        // Use as many ranks as the grid permits, then minimize the surface
        // to volume ratio. This also handles thin and non-cubic grids.
        int active = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= static_cast<int>(std::min(nx, size_t(worldSize))); ++x) {
            for (int y = 1; y <= static_cast<int>(std::min(ny, size_t(worldSize / x))); ++y) {
                int z = static_cast<int>(std::min(nz, size_t(worldSize / x / y)));
                int used = x * y * z;
                double cost = double(x) / nx + double(y) / ny + double(z) / nz;
                if (used > active || (used == active && cost < best)) {
                    active = used;
                    best = cost;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        MPI_Comm participating;
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED,
                       worldRank, &participating);
        if (participating == MPI_COMM_NULL) return;
        int periodic[3] = {};
        MPI_Cart_create(participating, 3, dims, periodic, 0, &comm);
        MPI_Comm_free(&participating);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) {
            size_t base = global[a] / dims[a], extra = global[a] % dims[a];
            n[a] = base + (size_t(coords[a]) < extra);
            offset[a] = size_t(coords[a]) * base + std::min(size_t(coords[a]), extra);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
            if (n[a] > size_t(INT_MAX - 2)) {
                if (rank == 0) fprintf(stderr, "Local grid dimensions exceed MPI datatype limits\n");
                MPI_Abort(comm, 1);
            }
        }
        const size_t limit = std::vector<double>().max_size();
        if (n[0]+2 > limit/(n[1]+2) ||
            (n[0]+2)*(n[1]+2) > limit/(n[2]+2)) {
            fprintf(stderr, "Rank %d: local grid storage is too large\n", worldRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        stride[0] = 1; stride[1] = n[0] + 2;
        stride[2] = stride[1] * (n[1] + 2);
        storage = stride[2] * (n[2] + 2);
        int sizes[3] = {int(n[2]+2), int(n[1]+2), int(n[0]+2)};
        int starts[3] = {};
        for (int a = 0; a < 3; ++a) {
            int subsizes[3] = {int(n[2]), int(n[1]), int(n[0])};
            subsizes[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
    }

    size_t index(size_t x, size_t y, size_t z) const {
        return z * stride[2] + y * stride[1] + x;
    }

    // Only faces are needed by the seven-point stencil, never edges/corners.
    int exchange(std::vector<double>& field, MPI_Request requests[12]) const {
        int count = 0;
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                size_t pos[3] = {1, 1, 1};
                pos[a] = side ? n[a] : 1;
                size_t send = index(pos[0], pos[1], pos[2]);
                pos[a] = side ? n[a]+1 : 0;
                size_t recv = index(pos[0], pos[1], pos[2]);
                int peer = neighbor[2*a+side];
                if (peer == MPI_PROC_NULL) {
                    // Replicate the boundary value for the original clamping.
                    size_t extent[3] = {n[0], n[1], n[2]};
                    extent[a] = 1;
                    for (size_t z = 0; z < extent[2]; ++z)
                        for (size_t y = 0; y < extent[1]; ++y)
                            for (size_t x = 0; x < extent[0]; ++x) {
                                size_t delta = index(x, y, z);
                                field[recv+delta] = field[send+delta];
                            }
                } else {
                    MPI_Irecv(field.data()+recv, 1, face[a], peer, 2*a+1-side,
                              comm, &requests[count++]);
                    MPI_Isend(field.data()+send, 1, face[a], peer, 2*a+side,
                              comm, &requests[count++]);
                }
            }
        }
        return count;
    }

    void close() {
        if (comm != MPI_COMM_NULL) {
            for (auto& type : face) MPI_Type_free(&type);
            MPI_Comm_free(&comm);
        }
    }
};

// Preserve the original expression order for both stencil passes.
template<bool chemical>
void computeBox(const Domain& d, const std::vector<double>& input,
                const std::vector<double>& cold, std::vector<double>& output,
                size_t x0, size_t x1, size_t y0, size_t y1, size_t z0, size_t z1) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const size_t sy = d.stride[1], sz = d.stride[2];
    const double* in = input.data();
    const double* old = cold.data();
    double* out = output.data();
    for (size_t z = z0; z < z1; ++z) {
        for (size_t y = y0; y < y1; ++y) {
            size_t row = d.index(0, y, z);
            for (size_t x = x0; x < x1; ++x) {
                size_t i = row + x;
                double cv = in[i];
                double cxx = in[i+1] + in[i-1] - 2.0 * cv;
                double cyy = in[i+sy] + in[i-sy] - 2.0 * cv;
                double czz = in[i+sz] + in[i-sz] - 2.0 * cv;
                double lap = cxx + cyy + czz;
                if constexpr (chemical)
                    out[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                             + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                else
                    out[i] = old[i] + 0.01 * 1.0 * lap;
            }
        }
    }
}

template<bool chemical>
void stencil(const Domain& d, std::vector<double>& input,
             const std::vector<double>& cold, std::vector<double>& output) {
    MPI_Request requests[12];
    int count = d.exchange(input, requests);
    size_t nx = d.n[0], ny = d.n[1], nz = d.n[2];
    // Strict interior work can proceed while the halo transfers are in flight.
    if (nx > 2 && ny > 2 && nz > 2)
        computeBox<chemical>(d, input, cold, output, 2, nx, 2, ny, 2, nz);
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    if (nx <= 2 || ny <= 2 || nz <= 2) {
        computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, ny+1, 1, nz+1);
    } else {
        // Six disjoint shells avoid branching in the innermost loop.
        computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, ny+1, 1, 2);
        computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, ny+1, nz, nz+1);
        computeBox<chemical>(d, input, cold, output, 1, nx+1, 1, 2, 2, nz);
        computeBox<chemical>(d, input, cold, output, 1, nx+1, ny, ny+1, 2, nz);
        computeBox<chemical>(d, input, cold, output, 1, 2, 2, ny, 2, nz);
        computeBox<chemical>(d, input, cold, output, nx, nx+1, 2, ny, 2, nz);
    }
}

// Gathering is only performed for -r, outside the timed simulation. Chunked
// transfers avoid MPI's int count limit and preserve the serial result order.
void printDistributedResults(const Domain& d, const std::vector<double>& c) {
    std::vector<double> packed(d.n[0]*d.n[1]*d.n[2]);
    size_t pos = 0;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y) {
            std::copy_n(c.data()+d.index(1,y,z), d.n[0], packed.data()+pos);
            pos += d.n[0];
        }
    constexpr size_t chunk = 1 << 20;
    if (d.rank != 0) {
        for (size_t i = 0; i < packed.size(); i += chunk)
            MPI_Send(packed.data()+i, int(std::min(chunk, packed.size()-i)), MPI_DOUBLE, 0, 20, d.comm);
        return;
    }
    std::vector<double> global(d.global[0]*d.global[1]*d.global[2]);
    int ranks;
    MPI_Comm_size(d.comm, &ranks);
    for (int r = 0; r < ranks; ++r) {
        int coord[3];
        MPI_Cart_coords(d.comm, r, 3, coord);
        size_t n[3], offset[3];
        for (int a = 0; a < 3; ++a) {
            size_t base = d.global[a]/d.dims[a], extra = d.global[a]%d.dims[a];
            n[a] = base + (size_t(coord[a]) < extra);
            offset[a] = size_t(coord[a])*base + std::min(size_t(coord[a]), extra);
        }
        if (r != 0) {
            packed.resize(n[0]*n[1]*n[2]);
            for (size_t i = 0; i < packed.size(); i += chunk)
                MPI_Recv(packed.data()+i, int(std::min(chunk, packed.size()-i)), MPI_DOUBLE,
                         r, 20, d.comm, MPI_STATUS_IGNORE);
        }
        size_t p = 0;
        for (size_t z = 0; z < n[2]; ++z)
            for (size_t y = 0; y < n[1]; ++y) {
                size_t i = ((z+offset[2])*d.global[1]+y+offset[1])*d.global[0]+offset[0];
                std::copy_n(packed.data()+p, n[0], global.data()+i);
                p += n[0];
            }
    }
    print_results(global, "Concentration");
}

bool validateDistributed(const Domain& d, const std::vector<double>& c) {
    int bad = 0, anyBad = 0;
    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -minVal;
    for (size_t z = 1; z <= d.n[2]; ++z)
        for (size_t y = 1; y <= d.n[1]; ++y)
            for (size_t x = 1; x <= d.n[0]; ++x) {
                double val = c[d.index(x,y,z)];
                bad |= !std::isfinite(val);
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, d.comm);
    double globalMin, globalMax;
    MPI_Allreduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    bool valid = !anyBad && globalMin >= -10.0 && globalMax <= 10.0;
    if (d.rank == 0) {
        if (anyBad) printf("Validation failed: found NaN or Inf value\n");
        else {
            printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > size_t(INT_MAX-2) || ny > size_t(INT_MAX-2) || nz > size_t(INT_MAX-2) ||
        nx > std::numeric_limits<size_t>::max()/ny ||
        nx*ny > std::numeric_limits<size_t>::max()/nz) {
        if (worldRank == 0) fprintf(stderr, "Invalid or oversized grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Domain d(nx, ny, nz, worldSize, worldRank);
    int status = 0;
    if (d.comm != MPI_COMM_NULL) {
        try {
            const size_t gridSize = nx * ny * nz;
            std::vector<double> cold(d.storage), cnew(d.storage), mu(d.storage);
            if (d.rank == 0) printf("Initializing concentration field...\n");
            for (size_t z = 1; z <= d.n[2]; ++z)
                for (size_t y = 1; y <= d.n[1]; ++y)
                    for (size_t x = 1; x <= d.n[0]; ++x) {
                        size_t id = ((z-1+d.offset[2])*ny+y-1+d.offset[1])*nx+x-1+d.offset[0];
                        double pseudo = (((id+1)*1299709)%gridSize)/static_cast<double>(gridSize);
                        cold[d.index(x,y,z)] = -1.0 + 2.0*pseudo;
                    }
            if (d.rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(d.comm);
            double start = MPI_Wtime();
            for (int t = 0; t < iterations; ++t) {
                stencil<true>(d, cold, cold, mu);
                stencil<false>(d, mu, cold, cnew);
                cold.swap(cnew);
            }
            double elapsed = MPI_Wtime() - start, duration;
            MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
            if (d.rank == 0) {
                printf("Computation time: %ld ms\n", static_cast<long>(duration*1000.0));
                double mcups = double(gridSize)*iterations/duration/1e6;
                printf("Performance: %.3f MCellUpdates/s\n", mcups);
            }
            if (printResults) printDistributedResults(d, cold);
            if (validate) {
                if (d.rank == 0) printf("Validating result...\n");
                status = validateDistributed(d, cold) ? 0 : 1;
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        d.close();
    }
    // Include surplus ranks in the final status and cleanly finalize MPI.
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
