#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

inline size_t at(size_t x, size_t y, size_t z, size_t nx, size_t ny) { return (z * ny + y) * nx + x; }

static void exchange(std::vector<double>& a, size_t nx, size_t ny, size_t nz,
                     int rank, int nr, MPI_Comm comm) {
    const int plane = static_cast<int>(nx * ny);
    if (rank > 0) MPI_Sendrecv(a.data() + plane, plane, MPI_DOUBLE, rank - 1, 0,
                               a.data(), plane, MPI_DOUBLE, rank - 1, 1, comm, MPI_STATUS_IGNORE);
    else std::copy_n(a.data() + plane, plane, a.data());
    if (rank + 1 < nr) MPI_Sendrecv(a.data() + nz * plane, plane, MPI_DOUBLE, rank + 1, 1,
                                    a.data() + (nz + 1) * plane, plane, MPI_DOUBLE, rank + 1, 0, comm, MPI_STATUS_IGNORE);
    else std::copy_n(a.data() + nz * plane, plane, a.data() + (nz + 1) * plane);
}

static inline double lap(const std::vector<double>& a, size_t x, size_t y, size_t z,
                          size_t nx, size_t ny, [[maybe_unused]] size_t nz, double dx, double dy, double dz) {
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t zp = z + 1, zn = z - 1;
    const double v = a[at(x,y,z,nx,ny)];
    return (a[at(xp,y,z,nx,ny)] + a[at(xn,y,z,nx,ny)] - 2*v)/(dx*dx)
         + (a[at(x,yp,z,nx,ny)] + a[at(x,yn,z,nx,ny)] - 2*v)/(dy*dy)
         + (a[at(x,y,zp,nx,ny)] + a[at(x,y,zn,nx,ny)] - 2*v)/(dz*dz);
}

static void usage(const char* p) {
    printf("Usage: %s [-x n] [-y n] [-z n] [-i steps] [-v] [-r] [-h]\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nr; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nr);
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, results=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true; else if (!strcmp(argv[i],"-r")) results=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (!nx || !ny || !nz || iterations < 0 || static_cast<size_t>(nr)>nz) {
        if(rank==0) fprintf(stderr,"Invalid grid or too many MPI ranks (ranks must not exceed z)\n");
        MPI_Finalize(); return 1;
    }
    const size_t base=nz/nr, rem=nz%nr, local_nz=base+(static_cast<size_t>(rank)<rem);
    const size_t z0=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem), plane=nx*ny;
    std::vector<double> cold((local_nz+2)*plane), cnew(cold.size()), mu(cold.size());
    for(size_t z=1; z<=local_nz; ++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        const size_t gid=at(x,y,z0+z-1,nx,ny);
        cold[at(x,y,z,nx,ny)] = -1.0 + 2.0 * ((((gid+1)*1299709)% (nx*ny*nz))/static_cast<double>(nx*ny*nz));
    }
    if(rank==0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); printf("Running MPI simulation on %d ranks...\n",nr); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::steady_clock::now();
    for(int t=0;t<iterations;++t) {
        exchange(cold,nx,ny,local_nz,rank,nr,MPI_COMM_WORLD);
        for(size_t z=1;z<=local_nz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t q=at(x,y,z,nx,ny); const double v=cold[q];
            mu[q]=4.5*((v+1)*(-2.0/9.0)+(v-1)*(-2.0/9.0)-2*v*(2.0/9.0))+3*v+v*v*v-0.5*lap(cold,x,y,z,nx,ny,local_nz,1,1,1);
        }
        exchange(mu,nx,ny,local_nz,rank,nr,MPI_COMM_WORLD);
        for(size_t z=1;z<=local_nz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) { const size_t q=at(x,y,z,nx,ny); cnew[q]=cold[q]+0.01*lap(mu,x,y,z,nx,ny,local_nz,1,1,1); }
        cold.swap(cnew);
    }
    const double local_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(), seconds= [&]{double v; MPI_Reduce(&local_seconds,&v,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); return v;}();
    if(rank==0) { printf("Computation time: %ld ms\n",(long)(seconds*1000)); printf("Performance: %.3f MCellUpdates/s\n",(nx*ny*nz*(double)iterations/seconds/1e6)); }
    std::vector<double> full; if(rank==0) full.resize(nx*ny*nz);
    std::vector<int> counts(nr), displs(nr); for(int r=0;r<nr;++r){size_t n=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(n*plane); displs[r]=static_cast<int>((r*base+std::min(static_cast<size_t>(r),rem))*plane);}
    MPI_Gatherv(cold.data()+plane,static_cast<int>(local_nz*plane),MPI_DOUBLE,rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int ok=1; if(rank==0 && results) print_results(full,"Concentration");
    if(validate){ double lo=0,hi=0, local_lo=1e300,local_hi=-1e300; for(size_t z=1;z<=local_nz;++z) for(size_t q=z*plane;q<(z+1)*plane;++q){local_lo=std::min(local_lo,cold[q]);local_hi=std::max(local_hi,cold[q]);} MPI_Reduce(&local_lo,&lo,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(&local_hi,&hi,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); if(rank==0){printf("Concentration range: [%.6f, %.6f]\n",lo,hi); ok=(lo>=-10&&hi<=10); printf("Validation: %s\n",ok?"PASSED":"FAILED");} MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);}
    MPI_Finalize(); return ok?0:1;
}
