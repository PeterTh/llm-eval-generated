#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>
// Use the MPI C interface; deprecated MPI C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

struct Block {
    int n[3], offset[3];
    size_t row, plane, size;
    Block(const int* global, const int* dims, const int* coords) {
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coords[a] < global[a] % dims[a]);
            offset[a] = coords[a] * (global[a] / dims[a]) +
                        std::min(coords[a], global[a] % dims[a]);
        }
        row = size_t(n[0]) + 2;
        plane = row * (size_t(n[1]) + 2);
        if (plane > std::numeric_limits<size_t>::max() / (size_t(n[2]) + 2) / sizeof(Real))
            throw std::runtime_error("Local grid is too large");
        size = plane * (size_t(n[2]) + 2);
    }
    size_t index(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * row + x;
    }
};

// Minimize total interface area, taking the actual grid aspect ratio into account.
// Prefer fewer X cuts on ties to retain long contiguous rows.
int chooseDimensions(const int* global, int ranks, int* dims) {
    const size_t cells = size_t(global[0]) * global[1] * global[2];
    for (int active = int(std::min(size_t(ranks), cells)); active > 0; --active) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(global[0], active); ++x) {
            if (active % x) continue;
            const int rest = active / x;
            for (int y = 1; y <= std::min(global[1], rest); ++y) {
                if (rest % y) continue;
                const int z = rest / y;
                if (z > global[2]) continue;
                const double area = double(x - 1) * global[1] * global[2] +
                                    double(y - 1) * global[0] * global[2] +
                                    double(z - 1) * global[0] * global[1];
                if (area < best) {
                    best = area;
                    dims[0] = x; dims[1] = y; dims[2] = z;
                }
            }
        }
        if (std::isfinite(best)) return active;
    }
    return 1;
}

// Half-open box; preserve the original floating-point operation order.
void updateBox(const Real* __restrict__ in, Real* __restrict__ out, const Block& b,
               int x0, int x1, int y0, int y1, int z0, int z1) {
    for (int z = z0; z < z1; ++z)
        for (int y = y0; y < y1; ++y) {
            const size_t base = b.index(0, y, z);
            for (int x = x0; x < x1; ++x) {
                const size_t i = base + x;
                out[i] = (in[i] + in[i-1] + in[i+1] + in[i-b.row] +
                          in[i+b.row] + in[i-b.plane] + in[i+b.plane]) / 7.0;
            }
        }
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int global[3] = {128, 0, 0}, iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        const char* option = argv[i];
        if ((!strcmp(option, "-x") || !strcmp(option, "-y") ||
             !strcmp(option, "-z") || !strcmp(option, "-i")) && i+1 < argc) {
            char* end = nullptr;
            const long value = strtol(argv[++i], &end, 10);
            if (!*argv[i] || *end || value < 0 || value > INT_MAX-2) {
                if (!rank) fprintf(stderr, "Invalid value for %s\n", option);
                return 1;
            }
            if (!strcmp(option, "-i")) iterations = int(value);
            else global[option[1] - 'x'] = int(value);
        } else if (!strcmp(option, "-v")) validate = true;
        else if (!strcmp(option, "-r")) printResults = true;
        else if (!strcmp(option, "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", option); printUsage(argv[0]); }
            return 1;
        }
    }
    if (!global[1]) global[1] = global[0];
    if (!global[2]) global[2] = global[0];
    const size_t limit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (global[0] == 0 || size_t(global[0]) > limit / global[1] / global[2]) {
        if (!rank) fprintf(stderr, "Invalid or oversized grid\n");
        return 1;
    }
    int dims[3];
    const int active = chooseDimensions(global, ranks, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &workers);
    int status = 0;
    if (rank < active) {
        MPI_Comm cart;
        int periodic[3] = {0, 0, 0}, coords[3];
        MPI_Cart_create(workers, 3, dims, periodic, 0, &cart);
        MPI_Cart_coords(cart, rank, 3, coords);
        const Block b(global, dims, coords);
        std::vector<Real> input(b.size), output(b.size);
        for (int z = 1; z <= b.n[2]; ++z)
            for (int y = 1; y <= b.n[1]; ++y)
                for (int x = 1; x <= b.n[0]; ++x) {
                    const size_t index = (size_t(z-1+b.offset[2]) * global[1] +
                                          y-1+b.offset[1]) * global[0] + x-1+b.offset[0];
                    input[b.index(x,y,z)] = Real(index % 19);
                }
        // Both buffers retain fixed physical boundaries for every iteration.
        output = input;
        int neighbor[6];
        MPI_Datatype face[3];
        for (int a = 0; a < 3; ++a) {
            MPI_Cart_shift(cart, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
            int sizes[3] = {b.n[2]+2, b.n[1]+2, b.n[0]+2};
            int subsizes[3] = {b.n[2], b.n[1], b.n[0]};
            int starts[3] = {0, 0, 0};
            subsizes[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
        int lo[3], hi[3], innerLo[3], innerHi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = 1 + (b.offset[a] == 0);
            hi[a] = b.n[a]+1 - (b.offset[a]+b.n[a] == global[a]);
            hi[a] = std::max(lo[a], hi[a]);
            innerLo[a] = std::min(hi[a], std::max(lo[a], 2));
            innerHi[a] = std::max(innerLo[a], std::min(hi[a], b.n[a]));
        }
        if (!rank) {
            printf("3D Stencil Benchmark\nGrid size: %d x %d x %d\n", global[0],global[1],global[2]);
            printf("Iterations: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
            printf("Initializing grid...\nRunning stencil computation...\n");
        }
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        for (int iter = 0; iter < iterations; ++iter) {
            MPI_Request requests[12];
            int count = 0;
            // Only faces are needed by the seven-point stencil, never edges/corners.
            for (int a = 0; a < 3; ++a)
                for (int side = 0; side < 2; ++side) {
                    const int peer = neighbor[2*a+side];
                    if (peer == MPI_PROC_NULL) continue;
                    int recv[3] = {1,1,1}, send[3] = {1,1,1};
                    recv[a] = side ? b.n[a]+1 : 0;
                    send[a] = side ? b.n[a] : 1;
                    MPI_Irecv(input.data()+b.index(recv[0],recv[1],recv[2]), 1, face[a],
                              peer, 2*a+1-side, cart, &requests[count++]);
                    MPI_Isend(input.data()+b.index(send[0],send[1],send[2]), 1, face[a],
                              peer, 2*a+side, cart, &requests[count++]);
                }
            updateBox(input.data(), output.data(), b, innerLo[0],innerHi[0],
                      innerLo[1],innerHi[1],innerLo[2],innerHi[2]);
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
            // Six disjoint boxes cover the remaining cells, including thin blocks.
            updateBox(input.data(),output.data(),b,lo[0],hi[0],lo[1],hi[1],lo[2],innerLo[2]);
            updateBox(input.data(),output.data(),b,lo[0],hi[0],lo[1],hi[1],innerHi[2],hi[2]);
            updateBox(input.data(),output.data(),b,lo[0],hi[0],lo[1],innerLo[1],innerLo[2],innerHi[2]);
            updateBox(input.data(),output.data(),b,lo[0],hi[0],innerHi[1],hi[1],innerLo[2],innerHi[2]);
            updateBox(input.data(),output.data(),b,lo[0],innerLo[0],innerLo[1],innerHi[1],innerLo[2],innerHi[2]);
            updateBox(input.data(),output.data(),b,innerHi[0],hi[0],innerLo[1],innerHi[1],innerLo[2],innerHi[2]);
            input.swap(output);
        }
        const double elapsed = MPI_Wtime()-start;
        double seconds;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (!rank) {
            const double updates = double(std::max(0,global[0]-2)) * std::max(0,global[1]-2) *
                                   std::max(0,global[2]-2) * iterations;
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds*1000));
            printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? updates/seconds/1e6 : 0);
        }
        if (printResults) {
            // Gather only on request, restoring the original global X-major order.
            std::vector<Real> packed(size_t(b.n[0])*b.n[1]*b.n[2]);
            size_t pos = 0;
            for (int z = 1; z <= b.n[2]; ++z)
                for (int y = 1; y <= b.n[1]; ++y) {
                    std::copy_n(input.data()+b.index(1,y,z), b.n[0], packed.data()+pos);
                    pos += b.n[0];
                }
            if (!rank) {
                std::vector<Real> full(size_t(global[0])*global[1]*global[2]);
                for (int source = 0; source < active; ++source) {
                    int c[3];
                    MPI_Cart_coords(cart, source, 3, c);
                    const Block other(global,dims,c);
                    const size_t length = size_t(other.n[0])*other.n[1]*other.n[2];
                    if (source) {
                        packed.resize(length);
                        for (size_t offset = 0; offset < length;) {
                            const int chunk = int(std::min(length-offset, size_t(INT_MAX)));
                            MPI_Recv(packed.data()+offset, chunk, MPI_DOUBLE, source, 6, cart, MPI_STATUS_IGNORE);
                            offset += chunk;
                        }
                    }
                    size_t p = 0;
                    for (int z = 0; z < other.n[2]; ++z)
                        for (int y = 0; y < other.n[1]; ++y) {
                            const size_t dest = (size_t(z+other.offset[2])*global[1]+y+other.offset[1]) * global[0]+other.offset[0];
                            std::copy_n(packed.data()+p,other.n[0],full.data()+dest);
                            p += other.n[0];
                        }
                }
                print_results(full,"Grid");
            } else {
                for (size_t offset = 0; offset < packed.size();) {
                    const int chunk = int(std::min(packed.size()-offset,size_t(INT_MAX)));
                    MPI_Send(packed.data()+offset,chunk,MPI_DOUBLE,0,6,cart);
                    offset += chunk;
                }
            }
        }
        if (validate) {
            double localMin = std::numeric_limits<double>::infinity(), localMax = -localMin;
            int bad = 0;
            for (int z = 1; z <= b.n[2]; ++z)
                for (int y = 1; y <= b.n[1]; ++y)
                    for (int x = 1; x <= b.n[0]; ++x) {
                        const Real value = input[b.index(x,y,z)];
                        bad |= !std::isfinite(value);
                        localMin = std::min(localMin,value);
                        localMax = std::max(localMax,value);
                    }
            double minValue, maxValue;
            MPI_Allreduce(&bad,&status,1,MPI_INT,MPI_MAX,cart);
            MPI_Reduce(&localMin,&minValue,1,MPI_DOUBLE,MPI_MIN,0,cart);
            MPI_Reduce(&localMax,&maxValue,1,MPI_DOUBLE,MPI_MAX,0,cart);
            if (!rank) {
                printf("Validating result...\n");
                if (status) printf("Validation failed: found NaN or Inf value\n");
                else {
                    printf("Value range: [%.6f, %.6f]\n",minValue,maxValue);
                    if (minValue < -1e6 || maxValue > 1e6) {
                        status = 1;
                        printf("Validation failed: values out of expected range\n");
                    }
                }
                printf("Validation: %s\n",status ? "FAILED" : "PASSED");
            }
        }
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int status = 1;
    try { status = run(argc,argv); }
    catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        fprintf(stderr,"Rank %d: %s\n",rank,error.what());
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();
    return status;
}
