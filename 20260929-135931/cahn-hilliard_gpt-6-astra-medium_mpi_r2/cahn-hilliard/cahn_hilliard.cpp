#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
// Use the portable MPI C interface; legacy MPI C++ bindings are unnecessary.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

// Each rank owns a rectangular block and one ghost cell on each face.
struct Domain {
    MPI_Comm comm;
    int n[3], offset[3], neighbor[6];
    size_t sy, sz, volume;
    MPI_Datatype face[3];
    MPI_Request requests[12];
    int pending = 0;

    Domain(MPI_Comm cart, const int global[3], const int dims[3]) : comm(cart) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coords[a] < global[a] % dims[a]);
            offset[a] = coords[a] * (global[a] / dims[a]) +
                        std::min(coords[a], global[a] % dims[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        sy = size_t(n[0]) + 2;
        sz = sy * (size_t(n[1]) + 2);
        volume = sz * (size_t(n[2]) + 2);
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        int starts[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) {
            int subs[3] = {n[2], n[1], n[0]};
            subs[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
    }
    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
    }
    size_t index(int x, int y, int z) const {
        return size_t(z)*sz + size_t(y)*sy + x;
    }
    void begin(std::vector<double>& field) {
        pending = 0;
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                int f = 2*a + side;
                int src[3] = {1,1,1}, dst[3] = {1,1,1};
                src[a] = side ? n[a] : 1;
                dst[a] = side ? n[a]+1 : 0;
                if (neighbor[f] == MPI_PROC_NULL) {
                    // Clamp only physical boundaries, for both c and mu.
                    int extent[3] = {n[0],n[1],n[2]};
                    extent[a] = 1;
                    for (int z = 0; z < extent[2]; ++z)
                        for (int y = 0; y < extent[1]; ++y)
                            for (int x = 0; x < extent[0]; ++x)
                                field[index(dst[0]+x,dst[1]+y,dst[2]+z)] =
                                    field[index(src[0]+x,src[1]+y,src[2]+z)];
                } else {
                    MPI_Irecv(field.data()+index(dst[0],dst[1],dst[2]), 1,
                              face[a], neighbor[f], f^1, comm, &requests[pending++]);
                    MPI_Isend(field.data()+index(src[0],src[1],src[2]), 1,
                              face[a], neighbor[f], f, comm, &requests[pending++]);
                }
            }
        }
    }

    // Unit grid spacing and physical constants match the original benchmark.
    template<bool chemical>
    void box(const std::vector<double>& in, const std::vector<double>& cold,
             std::vector<double>& out, int x0, int x1, int y0, int y1,
             int z0, int z1) const {
        for (int z = z0; z <= z1; ++z) {
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    size_t i = index(x,y,z);
                    double v = in[i];
                    double lap = (in[i+1] + in[i-1] - 2.0*v) +
                                 (in[i+sy] + in[i-sy] - 2.0*v) +
                                 (in[i+sz] + in[i-sz] - 2.0*v);
                    if constexpr (chemical) {
                        out[i] = 4.5 * ((v+1.0)*(-(2.0/9.0)) +
                                       (v-1.0)*(-(2.0/9.0)) - 2.0*v*(2.0/9.0)) +
                                 3.0*v + v*v*v - 0.5*lap;
                    } else {
                        out[i] = cold[i] + 0.01*lap;
                    }
                }
            }
        }
    }
    template<bool chemical>
    void step(std::vector<double>& in, const std::vector<double>& cold,
              std::vector<double>& out) {
        begin(in);
        box<chemical>(in,cold,out,2,n[0]-1,2,n[1]-1,2,n[2]-1);
        MPI_Waitall(pending, requests, MPI_STATUSES_IGNORE);
        // Six disjoint slabs cover the remaining cells, including thin blocks.
        box<chemical>(in,cold,out,1,n[0],1,n[1],1,1);
        if (n[2] > 1) box<chemical>(in,cold,out,1,n[0],1,n[1],n[2],n[2]);
        box<chemical>(in,cold,out,1,n[0],1,1,2,n[2]-1);
        if (n[1] > 1) box<chemical>(in,cold,out,1,n[0],n[1],n[1],2,n[2]-1);
        box<chemical>(in,cold,out,1,1,2,n[1]-1,2,n[2]-1);
        if (n[0] > 1) box<chemical>(in,cold,out,n[0],n[0],2,n[1]-1,2,n[2]-1);
    }
};

// Choose a geometry-aware factorization minimizing total internal face area.
// Surplus ranks participate in finalization when the grid is too small.
int partition(const int global[3], int ranks, int dims[3]) {
    for (int used = ranks; used >= 1; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int x = 1; x <= std::min(global[0],used); ++x) {
            if (used % x) continue;
            int rest = used/x;
            for (int y = 1; y <= std::min(global[1],rest); ++y) {
                if (rest % y) continue;
                int z = rest/y;
                if (z > global[2]) continue;
                double area = double(x-1)*global[1]*global[2] +
                              double(y-1)*global[0]*global[2] +
                              double(z-1)*global[0]*global[1];
                if (area < best) {
                    best = area;
                    dims[0]=x; dims[1]=y; dims[2]=z;
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
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
    
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    if (nx == 0 || nx > INT_MAX-2 || ny > INT_MAX-2 || nz > INT_MAX-2 ||
        nx > std::numeric_limits<size_t>::max()/ny/nz/sizeof(double)) {
        if (rank == 0) fprintf(stderr, "Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const int global[3] = {int(nx),int(ny),int(nz)};
    int dims[3];
    int used = partition(global, int(std::min(size_t(ranks),nx*ny*nz)), dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    if (rank < used) {
        int periods[3] = {0,0,0};
        MPI_Comm cart;
        MPI_Cart_create(active, 3, dims, periods, 0, &cart);
        {
            Domain d(cart, global, dims);
            std::vector<double> cold(d.volume), cnew(d.volume), mu(d.volume);
            size_t gridSize = nx*ny*nz;
            if (rank == 0) printf("Initializing concentration field...\n");
            for (int z=1; z<=d.n[2]; ++z)
                for (int y=1; y<=d.n[1]; ++y)
                    for (int x=1; x<=d.n[0]; ++x) {
                        size_t id = size_t(z-1+d.offset[2])*nx*ny +
                                    size_t(y-1+d.offset[1])*nx + x-1+d.offset[0];
                        double pseudo = (((id+1)*1299709)%gridSize)/double(gridSize);
                        cold[d.index(x,y,z)] = -1.0 + 2.0*pseudo;
                    }
            if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            double start = MPI_Wtime();
            for (int t=0; t<iterations; ++t) {
                d.step<true>(cold,cold,mu);
                d.step<false>(mu,cold,cnew);
                cold.swap(cnew);
            }
            double elapsed = MPI_Wtime()-start, seconds;
            MPI_Reduce(&elapsed,&seconds,1,MPI_DOUBLE,MPI_MAX,0,cart);
            if (rank == 0) {
                printf("Computation time: %lld ms\n", (long long)(seconds*1000));
                printf("Performance: %.3f MCellUpdates/s\n",
                       double(gridSize)*iterations/seconds/1e6);
            }
            if (printResults) {
                // Receive each block directly into its original global ordering.
                // Derived types avoid MPI int count limits on block volumes.
                std::vector<double> result;
                if (rank == 0) result.resize(gridSize);
                int sizes[3] = {d.n[2]+2,d.n[1]+2,d.n[0]+2};
                int subs[3] = {d.n[2],d.n[1],d.n[0]}, starts[3] = {1,1,1};
                MPI_Datatype block;
                MPI_Type_create_subarray(3,sizes,subs,starts,MPI_ORDER_C,MPI_DOUBLE,&block);
                MPI_Type_commit(&block);
                MPI_Request send;
                MPI_Isend(cold.data(),1,block,0,10,cart,&send);
                if (rank == 0) {
                    int gs[3] = {int(nz),int(ny),int(nx)};
                    for (int r=0; r<used; ++r) {
                        int coords[3], sub[3], off[3];
                        MPI_Cart_coords(cart,r,3,coords);
                        for (int a=0; a<3; ++a) {
                            sub[2-a] = global[a]/dims[a] + (coords[a]<global[a]%dims[a]);
                            off[2-a] = coords[a]*(global[a]/dims[a]) +
                                       std::min(coords[a],global[a]%dims[a]);
                        }
                        MPI_Datatype target;
                        MPI_Type_create_subarray(3,gs,sub,off,MPI_ORDER_C,MPI_DOUBLE,&target);
                        MPI_Type_commit(&target);
                        MPI_Recv(result.data(),1,target,r,10,cart,MPI_STATUS_IGNORE);
                        MPI_Type_free(&target);
                    }
                    print_results(result,"Concentration");
                }
                MPI_Wait(&send,MPI_STATUS_IGNORE);
                MPI_Type_free(&block);
            }
            if (validate) {
                double localMin = std::numeric_limits<double>::infinity();
                double localMax = -localMin;
                int bad = 0, anyBad;
                for (int z=1; z<=d.n[2]; ++z)
                    for (int y=1; y<=d.n[1]; ++y)
                        for (int x=1; x<=d.n[0]; ++x) {
                            double v = cold[d.index(x,y,z)];
                            bad |= !std::isfinite(v);
                            localMin = std::min(localMin,v);
                            localMax = std::max(localMax,v);
                        }
                double minimum, maximum;
                MPI_Reduce(&bad,&anyBad,1,MPI_INT,MPI_MAX,0,cart);
                MPI_Reduce(&localMin,&minimum,1,MPI_DOUBLE,MPI_MIN,0,cart);
                MPI_Reduce(&localMax,&maximum,1,MPI_DOUBLE,MPI_MAX,0,cart);
                if (rank == 0) {
                    printf("Validating result...\n");
                    if (anyBad) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n",minimum,maximum);
                        if (minimum < -10.0 || maximum > 10.0)
                            printf("Validation failed: values out of expected range\n");
                    }
                    status = anyBad || minimum < -10.0 || maximum > 10.0;
                    printf("Validation: %s\n",status ? "FAILED" : "PASSED");
                }
            }
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
