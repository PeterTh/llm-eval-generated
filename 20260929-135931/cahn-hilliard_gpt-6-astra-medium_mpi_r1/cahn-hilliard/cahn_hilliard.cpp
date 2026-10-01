#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// MPI dimensions are in z,y,x order; x remains contiguous in memory.
struct Domain {
    MPI_Comm comm;
    int rank, dims[3], coord[3], n[3], start[3], neighbor[6];
    size_t row, plane, volume;
    MPI_Datatype sendFace[6], recvFace[6], interior;

    static MPI_Datatype region(const int* sizes, const int* counts, const int* offsets) {
        MPI_Datatype type;
        MPI_Type_create_subarray(3, sizes, counts, offsets, MPI_ORDER_C,
                                 MPI_DOUBLE, &type);
        MPI_Type_commit(&type);
        return type;
    }

    Domain(MPI_Comm active, const int* global, const int* decomposition) {
        std::copy(decomposition, decomposition + 3, dims);
        int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coord);
        int padded[3];
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coord[a] < global[a] % dims[a]);
            start[a] = coord[a] * (global[a] / dims[a]) +
                       std::min(coord[a], global[a] % dims[a]);
            padded[a] = n[a] + 2;
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        row = padded[2];
        plane = row * padded[1];
        volume = plane * padded[0];
        int origin[3] = {1, 1, 1};
        interior = region(padded, n, origin);
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                int count[3] = {n[0], n[1], n[2]};
                int offset[3] = {1, 1, 1};
                count[a] = 1;
                offset[a] = side ? n[a] : 1;
                sendFace[2*a+side] = region(padded, count, offset);
                offset[a] = side ? n[a]+1 : 0;
                recvFace[2*a+side] = region(padded, count, offset);
            }
        }
    }

    ~Domain() {
        for (int f = 0; f < 6; ++f) {
            MPI_Type_free(&sendFace[f]);
            MPI_Type_free(&recvFace[f]);
        }
        MPI_Type_free(&interior);
        MPI_Comm_free(&comm);
    }

    size_t index(int z, int y, int x) const { return z * plane + y * row + x; }

    void exchange(std::vector<double>& field, MPI_Request* requests) const {
        for (int f = 0; f < 6; ++f) {
            MPI_Irecv(field.data(), 1, recvFace[f], neighbor[f], f ^ 1,
                      comm, &requests[f]);
        }
        for (int f = 0; f < 6; ++f) {
            MPI_Isend(field.data(), 1, sendFace[f], neighbor[f], f,
                      comm, &requests[6+f]);
        }
        // Only faces are needed by the seven-point stencil, never halo edges.
        if (neighbor[0] == MPI_PROC_NULL || neighbor[1] == MPI_PROC_NULL)
            for (int y = 1; y <= n[1]; ++y)
                for (int x = 1; x <= n[2]; ++x) {
                    if (neighbor[0] == MPI_PROC_NULL)
                        field[index(0,y,x)] = field[index(1,y,x)];
                    if (neighbor[1] == MPI_PROC_NULL)
                        field[index(n[0]+1,y,x)] = field[index(n[0],y,x)];
                }
        if (neighbor[2] == MPI_PROC_NULL || neighbor[3] == MPI_PROC_NULL)
            for (int z = 1; z <= n[0]; ++z)
                for (int x = 1; x <= n[2]; ++x) {
                    if (neighbor[2] == MPI_PROC_NULL)
                        field[index(z,0,x)] = field[index(z,1,x)];
                    if (neighbor[3] == MPI_PROC_NULL)
                        field[index(z,n[1]+1,x)] = field[index(z,n[1],x)];
                }
        if (neighbor[4] == MPI_PROC_NULL || neighbor[5] == MPI_PROC_NULL)
            for (int z = 1; z <= n[0]; ++z)
                for (int y = 1; y <= n[1]; ++y) {
                    if (neighbor[4] == MPI_PROC_NULL)
                        field[index(z,y,0)] = field[index(z,y,1)];
                    if (neighbor[5] == MPI_PROC_NULL)
                        field[index(z,y,n[2]+1)] = field[index(z,y,n[2])];
                }
    }
};

// Keep the original expression ordering and physical parameters; no fast-math.
template<bool chemical>
void computeBox(const Domain& d, const std::vector<double>& input,
                const std::vector<double>& cold, std::vector<double>& output,
                int z0, int z1, int y0, int y1, int x0, int x1) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const size_t i = d.index(z,y,x);
                const double cv = input[i];
                const double cxx = input[i+1] + input[i-1] - 2.0 * cv;
                const double cyy = input[i+d.row] + input[i-d.row] - 2.0 * cv;
                const double czz = input[i+d.plane] + input[i-d.plane] - 2.0 * cv;
                const double lap = cxx + cyy + czz;
                if constexpr (chemical)
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - 0.5 * lap;
                else
                    output[i] = cold[i] + 0.01 * lap;
            }
}

template<bool chemical>
void step(const Domain& d, std::vector<double>& input,
          const std::vector<double>& cold, std::vector<double>& output) {
    MPI_Request requests[12];
    d.exchange(input, requests);
    const int z = d.n[0], y = d.n[1], x = d.n[2];
    if (z > 2 && y > 2 && x > 2)
        computeBox<chemical>(d,input,cold,output,2,z,2,y,2,x);
    MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);
    if (z <= 2 || y <= 2 || x <= 2) {
        computeBox<chemical>(d,input,cold,output,1,z+1,1,y+1,1,x+1);
    } else {
        // Six disjoint boundary boxes complete the pass after halo arrival.
        computeBox<chemical>(d,input,cold,output,1,2,1,y+1,1,x+1);
        computeBox<chemical>(d,input,cold,output,z,z+1,1,y+1,1,x+1);
        computeBox<chemical>(d,input,cold,output,2,z,1,2,1,x+1);
        computeBox<chemical>(d,input,cold,output,2,z,y,y+1,1,x+1);
        computeBox<chemical>(d,input,cold,output,2,z,2,y,1,2);
        computeBox<chemical>(d,input,cold,output,2,z,2,y,x,x+1);
    }
}

// Maximize participating ranks, then minimize total inter-rank face area.
int chooseDecomposition(const int* global, int ranks, int* dims) {
    int used = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int z = 1; z <= std::min(global[0], ranks); ++z)
        for (int y = 1; y <= std::min(global[1], ranks / z); ++y) {
            const int x = std::min(global[2], ranks / (z*y));
            const int count = z*y*x;
            const double area = double(z-1)*global[1]*global[2] +
                                double(y-1)*global[0]*global[2] +
                                double(x-1)*global[0]*global[1];
            if (count > used || (count == used && area < best)) {
                used = count;
                best = area;
                dims[0] = z; dims[1] = y; dims[2] = x;
            }
        }
    return used;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > std::numeric_limits<size_t>::max() / ny / nz / sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Invalid or oversized grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const int global[3] = {static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx)};
    int dims[3];
    const int used = chooseDecomposition(global, ranks, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    if (rank < used) {
        Domain d(active, global, dims);
        const size_t gridSize = nx * ny * nz;
        std::vector<double> cold(d.volume), cnew(d.volume), mu(d.volume);
        if (rank == 0) {
            printf("Cahn-Hilliard Phase Separation Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Time steps: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing concentration field...\n");
        }
        for (int z = 1; z <= d.n[0]; ++z)
            for (int y = 1; y <= d.n[1]; ++y)
                for (int x = 1; x <= d.n[2]; ++x) {
                    const size_t id = (size_t(d.start[0]+z-1)*ny + d.start[1]+y-1)*nx + d.start[2]+x-1;
                    const double pseudo = (((id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
                    cold[d.index(z,y,x)] = -1.0 + 2.0 * pseudo;
                }
        if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
        MPI_Barrier(d.comm);
        const double start = MPI_Wtime();
        for (int t = 0; t < iterations; ++t) {
            step<true>(d, cold, cold, mu);
            step<false>(d, mu, cold, cnew);
            cold.swap(cnew);
        }
        const double elapsed = MPI_Wtime() - start;
        double duration;
        MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
        if (rank == 0) {
            printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
            printf("Performance: %.3f MCellUpdates/s\n", double(gridSize)*iterations / duration / 1e6);
        }
        if (printResults) {
            if (rank == 0) {
                std::vector<double> result(gridSize);
                for (int z = 1; z <= d.n[0]; ++z)
                    for (int y = 1; y <= d.n[1]; ++y)
                        std::copy_n(&cold[d.index(z,y,1)], d.n[2],
                                    &result[(size_t(d.start[0]+z-1)*ny+d.start[1]+y-1)*nx+d.start[2]]);
                for (int source = 1; source < used; ++source) {
                    int coord[3], count[3], offset[3];
                    MPI_Cart_coords(d.comm, source, 3, coord);
                    for (int a = 0; a < 3; ++a) {
                        count[a] = global[a]/dims[a] + (coord[a] < global[a]%dims[a]);
                        offset[a] = coord[a]*(global[a]/dims[a]) + std::min(coord[a],global[a]%dims[a]);
                    }
                    MPI_Datatype block = Domain::region(global, count, offset);
                    MPI_Recv(result.data(), 1, block, source, 20, d.comm, MPI_STATUS_IGNORE);
                    MPI_Type_free(&block);
                }
                print_results(result, "Concentration");
            } else {
                MPI_Send(cold.data(), 1, d.interior, 0, 20, d.comm);
            }
        }
        if (validate) {
            double lo = std::numeric_limits<double>::infinity(), hi = -lo;
            int bad = 0, anyBad;
            for (int z = 1; z <= d.n[0]; ++z)
                for (int y = 1; y <= d.n[1]; ++y)
                    for (int x = 1; x <= d.n[2]; ++x) {
                        const double value = cold[d.index(z,y,x)];
                        bad |= !std::isfinite(value);
                        lo = std::min(lo,value); hi = std::max(hi,value);
                    }
            double globalLo, globalHi;
            MPI_Reduce(&bad,&anyBad,1,MPI_INT,MPI_MAX,0,d.comm);
            MPI_Reduce(&lo,&globalLo,1,MPI_DOUBLE,MPI_MIN,0,d.comm);
            MPI_Reduce(&hi,&globalHi,1,MPI_DOUBLE,MPI_MAX,0,d.comm);
            if (rank == 0) {
                printf("Validating result...\n");
                if (anyBad) {
                    printf("Validation failed: found NaN or Inf value\n");
                    status = 1;
                } else {
                    printf("Concentration range: [%.6f, %.6f]\n",globalLo,globalHi);
                    if (globalHi > 10.0 || globalLo < -10.0) {
                        printf("Validation failed: values out of expected range\n");
                        status = 1;
                    }
                }
                printf("Validation: %s\n",status ? "FAILED" : "PASSED");
            }
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
