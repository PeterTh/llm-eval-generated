#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Each rank owns a rectangular block with one ghost cell on every face.
struct Domain {
    MPI_Comm comm;
    int dims[3], coord[3], n[3], neighbor[6];
    size_t begin[3], row, plane;
    MPI_Datatype face[3];
    size_t at(int x, int y, int z) const {
        return size_t(z) * plane + size_t(y) * row + x;
    }
    Domain(MPI_Comm active, const int* divisions, const size_t* global) {
        std::copy(divisions, divisions + 3, dims);
        int periods[3] = {0, 0, 0}, rank;
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coord);
        for (int a = 0; a < 3; ++a) {
            n[a] = int(global[a] / dims[a] + (size_t(coord[a]) < global[a] % dims[a]));
            begin[a] = size_t(coord[a]) * (global[a] / dims[a]) +
                       std::min(size_t(coord[a]), global[a] % dims[a]);
            MPI_Cart_shift(comm, a, 1, &neighbor[2*a], &neighbor[2*a+1]);
        }
        row = size_t(n[0]) + 2;
        plane = row * (size_t(n[1]) + 2);
        int sizes[3] = {n[2]+2, n[1]+2, n[0]+2}, starts[3] = {0,0,0};
        for (int a = 0; a < 3; ++a) {
            int subs[3] = {n[2], n[1], n[0]};
            subs[2-a] = 1;
            MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[a]);
            MPI_Type_commit(&face[a]);
        }
    }
    ~Domain() {
        for (auto& t : face) MPI_Type_free(&t);
        MPI_Comm_free(&comm);
    }
    void exchange(std::vector<double>& field, MPI_Request* requests) const {
        // Physical boundaries implement the original clamped stencil.
        for (int z=1; z<=n[2]; ++z)
            for (int y=1; y<=n[1]; ++y) {
                if (neighbor[0] == MPI_PROC_NULL) field[at(0,y,z)] = field[at(1,y,z)];
                if (neighbor[1] == MPI_PROC_NULL) field[at(n[0]+1,y,z)] = field[at(n[0],y,z)];
            }
        for (int z=1; z<=n[2]; ++z)
            for (int x=1; x<=n[0]; ++x) {
                if (neighbor[2] == MPI_PROC_NULL) field[at(x,0,z)] = field[at(x,1,z)];
                if (neighbor[3] == MPI_PROC_NULL) field[at(x,n[1]+1,z)] = field[at(x,n[1],z)];
            }
        for (int y=1; y<=n[1]; ++y)
            for (int x=1; x<=n[0]; ++x) {
                if (neighbor[4] == MPI_PROC_NULL) field[at(x,y,0)] = field[at(x,y,1)];
                if (neighbor[5] == MPI_PROC_NULL) field[at(x,y,n[2]+1)] = field[at(x,y,n[2])];
            }
        for (int a=0; a<3; ++a) {
            for (int side=0; side<2; ++side) {
                int send[3]={1,1,1}, recv[3]={1,1,1};
                send[a] = side ? n[a] : 1;
                recv[a] = side ? n[a]+1 : 0;
                const int f=2*a+side;
                MPI_Irecv(field.data()+at(recv[0],recv[1],recv[2]), 1, face[a],
                          neighbor[f], 2*a+1-side, comm, &requests[f]);
                MPI_Isend(field.data()+at(send[0],send[1],send[2]), 1, face[a],
                          neighbor[f], f, comm, &requests[6+f]);
            }
        }
    }
};

// Keep the arithmetic order of the serial stencil (unit grid spacing).
template<bool chemical>
void compute(const Domain& d, const std::vector<double>& input,
             const std::vector<double>& cold, std::vector<double>& output, bool interior) {
    const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = 2.0/9.0;
    auto row = [&](int y, int z, int first, int last) {
        for (int x=first; x<=last; ++x) {
            size_t i=d.at(x,y,z);
            const double v=input[i];
            const double cxx=input[i+1]+input[i-1]-2.0*v;
            const double cyy=input[i+d.row]+input[i-d.row]-2.0*v;
            const double czz=input[i+d.plane]+input[i-d.plane]-2.0*v;
            const double lap=cxx+cyy+czz;
            if constexpr (chemical)
                output[i]=4.5*((v+1.0)*e_AA+(v-1.0)*e_BB-2.0*v*e_AB)
                          +3.0*v+v*v*v-0.5*lap;
            else
                output[i]=cold[i]+0.01*lap;
        }
    };
    if (interior) {
        for (int z=2; z<d.n[2]; ++z)
            for (int y=2; y<d.n[1]; ++y) row(y,z,2,d.n[0]-1);
    } else {
        for (int z=1; z<=d.n[2]; ++z)
            for (int y=1; y<=d.n[1]; ++y) {
                if (z==1 || z==d.n[2] || y==1 || y==d.n[1]) row(y,z,1,d.n[0]);
                else {
                    row(y,z,1,1);
                    if (d.n[0]>1) row(y,z,d.n[0],d.n[0]);
                }
            }
    }
}

// Gather only for the optional order-sensitive result hash and Kahan sum.
void printGlobal(const Domain& d, const std::vector<double>& c, const size_t* global) {
    int rank, ranks;
    MPI_Comm_rank(d.comm,&rank);
    MPI_Comm_size(d.comm,&ranks);
    std::vector<double> packed(size_t(d.n[0])*d.n[1]*d.n[2]);
    size_t pos=0;
    for (int z=1; z<=d.n[2]; ++z)
        for (int y=1; y<=d.n[1]; ++y) {
            std::copy_n(c.data()+d.at(1,y,z),d.n[0],packed.data()+pos);
            pos+=d.n[0];
        }
    if (rank != 0) {
        for (size_t off=0; off<packed.size();) {
            int count=int(std::min(size_t(INT_MAX),packed.size()-off));
            MPI_Send(packed.data()+off,count,MPI_DOUBLE,0,0,d.comm);
            off+=count;
        }
    } else {
        std::vector<double> all(global[0]*global[1]*global[2]);
        for (int r=0; r<ranks; ++r) {
            int coord[3]; size_t start[3], n[3];
            MPI_Cart_coords(d.comm,r,3,coord);
            for (int a=0; a<3; ++a) {
                n[a]=global[a]/d.dims[a]+(size_t(coord[a])<global[a]%d.dims[a]);
                start[a]=size_t(coord[a])*(global[a]/d.dims[a])+
                         std::min(size_t(coord[a]),global[a]%d.dims[a]);
            }
            packed.resize(n[0]*n[1]*n[2]);
            if (r!=0) {
                for (size_t off=0; off<packed.size();) {
                    int count=int(std::min(size_t(INT_MAX),packed.size()-off));
                    MPI_Recv(packed.data()+off,count,MPI_DOUBLE,r,0,d.comm,MPI_STATUS_IGNORE);
                    off+=count;
                }
            }
            size_t off=0;
            for (size_t z=0; z<n[2]; ++z)
                for (size_t y=0; y<n[1]; ++y) {
                    size_t dst=((start[2]+z)*global[1]+start[1]+y)*global[0]+start[0];
                    std::copy_n(packed.data()+off,n[0],all.data()+dst);
                    off+=n[0];
                }
        }
        print_results(all,"Concentration");
    }
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
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

    const size_t global[3]={nx,ny,nz};
    if (nx==0 || nx>INT_MAX-2 || ny>INT_MAX-2 || nz>INT_MAX-2 ||
        nx>std::numeric_limits<size_t>::max()/ny ||
        nx*ny>std::numeric_limits<size_t>::max()/nz/sizeof(double)) {
        if (rank==0) fprintf(stderr,"Invalid or excessively large grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    // Minimize exchanged face area, keeping every selected rank nonempty.
    int divisions[3]={1,1,1};
    int activeRanks=int(std::min(size_t(ranks),nx*ny*nz));
    bool found=false;
    for (; activeRanks>0; --activeRanks) {
        double best=std::numeric_limits<double>::infinity();
        for (int px=1; px<=activeRanks && size_t(px)<=nx; ++px) {
            if (activeRanks%px) continue;
            int rest=activeRanks/px;
            for (int py=1; py<=rest && size_t(py)<=ny; ++py) {
                if (rest%py) continue;
                int pz=rest/py;
                if (size_t(pz)>nz) continue;
                double area=double(px-1)*ny*nz+double(py-1)*nx*nz+double(pz-1)*nx*ny;
                if (area<best) {
                    best=area;
                    divisions[0]=px; divisions[1]=py; divisions[2]=pz;
                    found=true;
                }
            }
        }
        if (found) break;
    }
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank<activeRanks ? 0 : MPI_UNDEFINED,rank,&active);
    int status=0;
    if (active!=MPI_COMM_NULL) {
        {
            Domain d(active,divisions,global);
            const size_t volume=nx*ny*nz;
            std::vector<double> cold(d.plane*(size_t(d.n[2])+2));
            std::vector<double> cnew(cold.size()), mu(cold.size());
            if (rank==0) printf("Initializing concentration field...\n");
            for (int z=1; z<=d.n[2]; ++z)
                for (int y=1; y<=d.n[1]; ++y)
                    for (int x=1; x<=d.n[0]; ++x) {
                        size_t id=((d.begin[2]+z-1)*ny+d.begin[1]+y-1)*nx+d.begin[0]+x-1;
                        double pseudo=(((id+1)*1299709)%volume)/static_cast<double>(volume);
                        cold[d.at(x,y,z)]=-1.0+2.0*pseudo;
                    }
            if (rank==0) printf("Running Cahn-Hilliard simulation...\n");
            MPI_Barrier(d.comm);
            double start=MPI_Wtime();
            for (int t=0; t<iterations; ++t) {
                MPI_Request requests[12];
                d.exchange(cold,requests);
                compute<true>(d,cold,cold,mu,true);
                MPI_Waitall(12,requests,MPI_STATUSES_IGNORE);
                compute<true>(d,cold,cold,mu,false);
                d.exchange(mu,requests);
                compute<false>(d,mu,cold,cnew,true);
                MPI_Waitall(12,requests,MPI_STATUSES_IGNORE);
                compute<false>(d,mu,cold,cnew,false);
                cold.swap(cnew);
            }
            double elapsed=MPI_Wtime()-start, seconds;
            MPI_Reduce(&elapsed,&seconds,1,MPI_DOUBLE,MPI_MAX,0,d.comm);
            if (rank==0) {
                printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000.0));
                printf("Performance: %.3f MCellUpdates/s\n",double(volume)*iterations/seconds/1e6);
            }
            if (printResults) printGlobal(d,cold,global);
            if (validate) {
                double lo=std::numeric_limits<double>::infinity(), hi=-lo;
                int bad=0, anyBad;
                for (int z=1; z<=d.n[2]; ++z)
                    for (int y=1; y<=d.n[1]; ++y)
                        for (int x=1; x<=d.n[0]; ++x) {
                            double v=cold[d.at(x,y,z)];
                            bad |= !std::isfinite(v);
                            lo=std::min(lo,v); hi=std::max(hi,v);
                        }
                double minVal,maxVal;
                MPI_Reduce(&bad,&anyBad,1,MPI_INT,MPI_MAX,0,d.comm);
                MPI_Reduce(&lo,&minVal,1,MPI_DOUBLE,MPI_MIN,0,d.comm);
                MPI_Reduce(&hi,&maxVal,1,MPI_DOUBLE,MPI_MAX,0,d.comm);
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
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
