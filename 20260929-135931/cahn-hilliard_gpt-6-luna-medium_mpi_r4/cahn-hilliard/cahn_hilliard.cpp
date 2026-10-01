#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x,size_t y,size_t z,size_t nx,size_t ny) { return z*nx*ny+y*nx+x; }

static void exchange(std::vector<double>& a, size_t nx, size_t ny, size_t nzLocal,
                     int rank, int ranks, MPI_Comm comm) {
    const int plane = static_cast<int>(nx*ny);
    MPI_Sendrecv(a.data()+idx3(0,0,1,nx,ny), plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 10,
                 a.data()+idx3(0,0,nzLocal+1,nx,ny), plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 10, comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data()+idx3(0,0,nzLocal,nx,ny), plane, MPI_DOUBLE, rank+1<ranks ? rank+1 : MPI_PROC_NULL, 11,
                 a.data(), plane, MPI_DOUBLE, rank ? rank-1 : MPI_PROC_NULL, 11, comm, MPI_STATUS_IGNORE);
    if(rank==0) std::copy_n(a.data()+idx3(0,0,1,nx,ny),plane,a.data());
    if(rank==ranks-1) std::copy_n(a.data()+idx3(0,0,nzLocal,nx,ny),plane,a.data()+idx3(0,0,nzLocal+1,nx,ny));
}

static double lap(const std::vector<double>& a,size_t nx,size_t ny,size_t z,size_t x,size_t y) {
    size_t xp=std::min(x+1,nx-1), yp=std::min(y+1,ny-1);
    const size_t xn=x?x-1:0, yn=y?y-1:0;
    const double center=a[idx3(x,y,z,nx,ny)];
    const double cxx=(a[idx3(xp,y,z,nx,ny)]+a[idx3(xn,y,z,nx,ny)]-2.0*center)/1.0;
    const double cyy=(a[idx3(x,yp,z,nx,ny)]+a[idx3(x,yn,z,nx,ny)]-2.0*center)/1.0;
    const double czz=(a[idx3(x,y,z+1,nx,ny)]+a[idx3(x,y,z-1,nx,ny)]-2.0*center)/1.0;
    return cxx+cyy+czz;
}

void printUsage(const char* p) { printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int worldRank,worldSize; MPI_Comm_rank(MPI_COMM_WORLD,&worldRank); MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
    size_t nx=64,ny=0,nz=0; int iterations=20; bool validate=false,printResults=false,help=false,bad=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) help=true;
        else { if(worldRank==0) printf("Unknown option: %s\n",argv[i]); bad=true; }
    }
    int flags[2]={bad,help}; MPI_Bcast(flags,2,MPI_INT,0,MPI_COMM_WORLD); bad=flags[0]; help=flags[1];
    if(help||bad) { if(worldRank==0) printUsage(argv[0]); MPI_Finalize(); return bad?1:0; }
    if(!ny) ny=nx; if(!nz) nz=nx;
    if(!nx||!ny||!nz) { if(worldRank==0) printf("Grid dimensions must be positive\n"); MPI_Finalize(); return 1; }
    const int activeSize=static_cast<int>(std::min<size_t>(worldSize,nz));
    MPI_Comm comm; MPI_Comm_split(MPI_COMM_WORLD,worldRank<activeSize?0:MPI_UNDEFINED,worldRank,&comm);
    if(worldRank>=activeSize) { MPI_Finalize(); return 0; }
    int rank,ranks; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&ranks);
    size_t base=nz/ranks,rem=nz%ranks,localNz=base+(size_t(rank)<rem),zStart=size_t(rank)*base+std::min<size_t>(rank,rem);
    size_t plane=nx*ny, n=localNz*plane;
    // One ghost plane on each side; boundary ranks replicate their edge plane.
    std::vector<double> c((localNz+2)*plane), next((localNz+2)*plane),mu((localNz+2)*plane);
    for(size_t z=0;z<localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t gid=(zStart+z)*plane+y*nx+x;
        c[idx3(x,y,z+1,nx,ny)]=-1.0+2.0*((((gid+1)*1299709)% (nx*ny*nz))/double(nx*ny*nz));
    }
    const double factor=0.01;
    if(worldRank==0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\n",nx,ny,nz,iterations); }
    MPI_Barrier(comm); auto start=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t) {
        exchange(c,nx,ny,localNz,rank,ranks,comm);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            auto q=idx3(x,y,z,nx,ny); double cv=c[q];
            mu[q]=4.5*((cv+1)*(-2.0/9.0)+(cv-1)*(-2.0/9.0)-2*cv*(2.0/9.0))+3*cv+cv*cv*cv-0.5*lap(c,nx,ny,z,x,y);
        }
        exchange(mu,nx,ny,localNz,rank,ranks,comm);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            auto q=idx3(x,y,z,nx,ny); next[q]=c[q]+factor*lap(mu,nx,ny,z,x,y);
        }
        c.swap(next);
    }
    MPI_Barrier(comm); auto end=std::chrono::high_resolution_clock::now();
    double secs=std::chrono::duration<double>(end-start).count(),maxSecs; MPI_Reduce(&secs,&maxSecs,1,MPI_DOUBLE,MPI_MAX,0,comm);
    if(rank==0) printf("Computation time: %.3f ms\nPerformance: %.3f MCellUpdates/s\n",maxSecs*1000,(double(nx)*ny*nz*iterations)/maxSecs/1e6);
    std::vector<int> counts(ranks),displs(ranks); for(int r=0;r<ranks;++r) { size_t rn=(nz/ranks+(size_t(r)<nz%ranks))*plane; size_t off=(size_t(r)*(nz/ranks)+std::min<size_t>(r,nz%ranks))*plane; counts[r]=int(rn); displs[r]=int(off); }
    std::vector<double> packed(n); for(size_t z=0;z<localNz;++z) std::copy_n(c.data()+idx3(0,0,z+1,nx,ny),plane,packed.data()+z*plane);
    std::vector<double> full; if(rank==0) full.resize(nx*ny*nz);
    MPI_Gatherv(packed.data(),int(n),MPI_DOUBLE,full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,comm);
    int status=0;
    if(rank==0 && printResults) print_results(full,"Concentration");
    if(rank==0 && validate) {
        double lo=full[0],hi=full[0]; bool valid=true;
        for(double v:full) { if(!std::isfinite(v)) valid=false; lo=std::min(lo,v); hi=std::max(hi,v); }
        printf("Concentration range: [%.6f, %.6f]\n",lo,hi);
        status=(!valid||hi>10||lo< -10)?1:0; printf("Validation: %s\n",status?"FAILED":"PASSED");
    }
    MPI_Bcast(&status,1,MPI_INT,0,comm); MPI_Comm_free(&comm); MPI_Finalize(); return status;
}
