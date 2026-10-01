#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// All fields contain one ghost cell on each face. Edges and corners are unused.
struct Domain {
    MPI_Comm comm;
    int n[3], start[3], neighbor[6];
    size_t sy, sz, volume;
    MPI_Datatype face[3], owned;

    size_t index(int x, int y, int z) const {
        return size_t(z) * sz + size_t(y) * sy + x;
    }

    Domain(MPI_Comm cart, const int global[3], const int dims[3]) : comm(cart) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        for (int a = 0; a < 3; ++a) {
            n[a] = global[a] / dims[a] + (coords[a] < global[a] % dims[a]);
            start[a] = coords[a] * (global[a] / dims[a]) +
                       std::min(coords[a], global[a] % dims[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        sy = size_t(n[0]) + 2;
        sz = sy * (size_t(n[1]) + 2);
        volume = sz * (size_t(n[2]) + 2);
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2};
        int begins[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a) {
            int subs[3] = {n[2], n[1], n[0]};
            subs[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subs, begins, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
        int subs[3] = {n[2], n[1], n[0]}, first[3] = {1, 1, 1};
        MPI_Type_create_subarray(3, sizes, subs, first, MPI_ORDER_C, MPI_DOUBLE, &owned);
        MPI_Type_commit(&owned);
    }

    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Type_free(&owned);
    }

    int exchange(std::vector<double>& field, MPI_Request requests[12]) const {
        int count = 0;
        // Physical boundaries copy the adjacent cell, preserving clamping.
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                int from[3] = {1,1,1}, to[3] = {1,1,1};
                from[a] = side ? n[a] : 1;
                to[a] = side ? n[a]+1 : 0;
                const int peer = neighbor[2*a+side];
                if (peer != MPI_PROC_NULL) {
                    MPI_Irecv(field.data()+index(to[0],to[1],to[2]), 1, face[a],
                              peer, 2*a+1-side, comm, &requests[count++]);
                    MPI_Isend(field.data()+index(from[0],from[1],from[2]), 1, face[a],
                              peer, 2*a+side, comm, &requests[count++]);
                } else {
                    int hi[3] = {n[0],n[1],n[2]};
                    hi[a] = 1;
                    const size_t offset = index(to[0],to[1],to[2]);
                    const size_t source = index(from[0],from[1],from[2]);
                    for (int z=0; z<hi[2]; ++z)
                        for (int y=0; y<hi[1]; ++y)
                            for (int x=0; x<hi[0]; ++x) {
                                size_t i = index(x,y,z);
                                field[offset+i] = field[source+i];
                            }
                }
            }
        }
        return count;
    }
};

// Unit grid spacings and physical constants are those of the original benchmark.
// Separate boxes keep the inner loop branch-free and permit vectorization.
template<bool chemical>
void box(const Domain& d, const std::vector<double>& input,
         const std::vector<double>& cold, std::vector<double>& output,
         int x0, int x1, int y0, int y1, int z0, int z1) {
    const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
    for (int z=z0; z<z1; ++z)
        for (int y=y0; y<y1; ++y) {
            const size_t row = d.index(0,y,z);
            for (int x=x0; x<x1; ++x) {
                const size_t i = row+x;
                const double cv = input[i];
                const double cxx = input[i+1] + input[i-1] - 2.0*cv;
                const double cyy = input[i+d.sy] + input[i-d.sy] - 2.0*cv;
                const double czz = input[i+d.sz] + input[i-d.sz] - 2.0*cv;
                const double lap = cxx + cyy + czz;
                if constexpr (chemical)
                    output[i] = 4.5 * ((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
                              + 3.0*cv + cv*cv*cv - 0.5*lap;
                else
                    output[i] = cold[i] + 0.01*lap;
            }
        }
}

template<bool chemical>
void step(const Domain& d, std::vector<double>& input,
          const std::vector<double>& cold, std::vector<double>& output) {
    MPI_Request requests[12];
    const int count = d.exchange(input, requests);
    const int x=d.n[0], y=d.n[1], z=d.n[2];
    box<chemical>(d,input,cold,output,2,x,2,y,2,z);
    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    // Disjoint boundary boxes also handle local dimensions of one or two.
    box<chemical>(d,input,cold,output,1,x+1,1,y+1,1,2);
    if (z>1) box<chemical>(d,input,cold,output,1,x+1,1,y+1,z,z+1);
    if (z>2) {
        box<chemical>(d,input,cold,output,1,x+1,1,2,2,z);
        if (y>1) box<chemical>(d,input,cold,output,1,x+1,y,y+1,2,z);
        if (y>2) {
            box<chemical>(d,input,cold,output,1,2,2,y,2,z);
            if (x>1) box<chemical>(d,input,cold,output,x,x+1,2,y,2,z);
        }
    }
}

// Maximize the number of nonempty ranks, then minimize halo surface area.
int chooseGrid(int ranks, const int global[3], int dims[3]) {
    const size_t cells = size_t(global[0])*global[1]*global[2];
    for (int p = int(std::min(size_t(ranks),cells)); p>0; --p) {
        double best = std::numeric_limits<double>::infinity();
        for (int x=1; x<=std::min(p,global[0]); ++x) {
            if (p%x) continue;
            const int rest=p/x;
            for (int y=1; y<=std::min(rest,global[1]); ++y) {
                if (rest%y) continue;
                const int z=rest/y;
                if (z>global[2]) continue;
                const double surface = double(x-1)/global[0] +
                    double(y-1)/global[1] + double(z-1)/global[2];
                if (surface<best) {
                    best=surface; dims[0]=x; dims[1]=y; dims[2]=z;
                }
            }
        }
        if (std::isfinite(best)) return p;
    }
    return 1;
}

// Gather only for requested output, using subarrays to retain global x-fastest
// order (including the original sequential Kahan sum and bytewise hash).
void printDistributed(const Domain& d, const std::vector<double>& c,
                      const int global[3], const int dims[3], int rank, int ranks) {
    if (rank != 0) {
        MPI_Send(c.data(), 1, d.owned, 0, 20, d.comm);
        return;
    }
    std::vector<double> full(size_t(global[0])*global[1]*global[2]);
    for (int z=1; z<=d.n[2]; ++z)
        for (int y=1; y<=d.n[1]; ++y)
            std::copy_n(c.data()+d.index(1,y,z), d.n[0],
                full.data()+(size_t(d.start[2]+z-1)*global[1]+d.start[1]+y-1)*global[0]+d.start[0]);
    for (int r=1; r<ranks; ++r) {
        int coords[3], sizes[3], subs[3], starts[3];
        MPI_Cart_coords(d.comm,r,3,coords);
        for (int a=0; a<3; ++a) {
            sizes[2-a]=global[a];
            subs[2-a]=global[a]/dims[a]+(coords[a]<global[a]%dims[a]);
            starts[2-a]=coords[a]*(global[a]/dims[a])+std::min(coords[a],global[a]%dims[a]);
        }
        MPI_Datatype block;
        MPI_Type_create_subarray(3,sizes,subs,starts,MPI_ORDER_C,MPI_DOUBLE,&block);
        MPI_Type_commit(&block);
        MPI_Recv(full.data(),1,block,r,20,d.comm,MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(full,"Concentration");
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
    MPI_Init(&argc,&argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
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
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || nx > size_t(std::numeric_limits<int>::max()-2) ||
        ny > size_t(std::numeric_limits<int>::max()-2) ||
        nz > size_t(std::numeric_limits<int>::max()-2) ||
        nx > std::numeric_limits<size_t>::max()/ny/nz/sizeof(double)) {
        if (rank == 0) fprintf(stderr,"Invalid or oversized grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const int global[3] = {int(nx),int(ny),int(nz)};
    int dims[3];
    const int active = chooseGrid(worldSize,global,dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD,rank<active ? 0 : MPI_UNDEFINED,rank,&workers);
    int status=0;
    if (rank<active) {
        int periods[3] = {0,0,0};
        MPI_Comm cart;
        MPI_Cart_create(workers,3,dims,periods,0,&cart);
        {
            Domain d(cart,global,dims);
            std::vector<double> cold(d.volume), cnew(d.volume), mu(d.volume);
            const size_t gridSize=nx*ny*nz;
            if (rank==0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n",nx,ny,nz);
                printf("Time steps: %d\n",iterations);
                printf("Validation: %s\n",validate ? "enabled" : "disabled");
                printf("Initializing concentration field...\n");
            }
            for (int z=1; z<=d.n[2]; ++z)
                for (int y=1; y<=d.n[1]; ++y)
                    for (int x=1; x<=d.n[0]; ++x) {
                        const size_t id=(size_t(d.start[2]+z-1)*ny+d.start[1]+y-1)*nx+d.start[0]+x-1;
                        const double pseudo=(((id+1)*1299709)%gridSize)/static_cast<double>(gridSize);
                        cold[d.index(x,y,z)]=-1.0+2.0*pseudo;
                    }
            if (rank==0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(cart);
            const double begin=MPI_Wtime();
            for (int t=0; t<iterations; ++t) {
                step<true>(d,cold,cold,mu);
                step<false>(d,mu,cold,cnew);
                cold.swap(cnew);
            }
            double elapsed=MPI_Wtime()-begin, duration=0;
            MPI_Reduce(&elapsed,&duration,1,MPI_DOUBLE,MPI_MAX,0,cart);
            if (rank==0) {
                printf("Computation time: %ld ms\n",static_cast<long>(duration*1000));
                printf("Performance: %.3f MCellUpdates/s\n",double(gridSize)*iterations/duration/1e6);
            }
            if (printResults) printDistributed(d,cold,global,dims,rank,active);
            if (validate) {
                double low=std::numeric_limits<double>::infinity(), high=-low;
                int bad=0, anyBad=0;
                for (int z=1; z<=d.n[2]; ++z)
                    for (int y=1; y<=d.n[1]; ++y)
                        for (int x=1; x<=d.n[0]; ++x) {
                            const double v=cold[d.index(x,y,z)];
                            bad |= !std::isfinite(v);
                            low=std::min(low,v); high=std::max(high,v);
                        }
                double minVal,maxVal;
                MPI_Reduce(&bad,&anyBad,1,MPI_INT,MPI_MAX,0,cart);
                MPI_Reduce(&low,&minVal,1,MPI_DOUBLE,MPI_MIN,0,cart);
                MPI_Reduce(&high,&maxVal,1,MPI_DOUBLE,MPI_MAX,0,cart);
                if (rank==0) {
                    printf("Validating result...\n");
                    if (anyBad) printf("Validation failed: found NaN or Inf value\n");
                    else {
                        printf("Concentration range: [%.6f, %.6f]\n",minVal,maxVal);
                        if (maxVal>10.0 || minVal< -10.0)
                            printf("Validation failed: values out of expected range\n");
                    }
                    status=anyBad || maxVal>10.0 || minVal< -10.0;
                    printf("Validation: %s\n",status ? "FAILED" : "PASSED");
                }
            }
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
