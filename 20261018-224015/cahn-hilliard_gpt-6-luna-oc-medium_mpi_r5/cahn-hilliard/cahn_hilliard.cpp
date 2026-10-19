#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * nx * ny + y * nx + x;
}

static void exchangeHalos(std::vector<double>& a, size_t plane, size_t localZ,
                          int rank, int ranks, MPI_Comm comm) {
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                 a.data() + (localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + localZ * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                 a.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                 comm, MPI_STATUS_IGNORE);
    if (rank == 0) std::copy_n(a.data() + plane, plane, a.data());
    if (rank + 1 == ranks) std::copy_n(a.data() + localZ * plane, plane, a.data() + (localZ + 1) * plane);
}

static double laplacian(const std::vector<double>& a, size_t x, size_t y, size_t z,
                        size_t nx, size_t ny, size_t nz, size_t firstZ) {
    const size_t p = nx * ny, i = idx3(x, y, z, nx, ny);
    const size_t xp = x + 1 < nx ? x + 1 : x, xn = x ? x - 1 : 0;
    const size_t yp = y + 1 < ny ? y + 1 : y, yn = y ? y - 1 : 0;
    const size_t zp = firstZ + z - 1 < nz - 1 ? z + 1 : z;
    const size_t zn = firstZ + z - 1 > 0 ? z - 1 : z;
    return a[idx3(xp,y,z,nx,ny)] + a[idx3(xn,y,z,nx,ny)] - 2*a[i]
         + a[idx3(x,yp,z,nx,ny)] + a[idx3(x,yn,z,nx,ny)] - 2*a[i]
         + a[zp * p + (i - z * p)] + a[zn * p + (i - z * p)] - 2*a[i];
}

static void initialize(std::vector<double>& a, size_t nx, size_t ny, size_t nz,
                       size_t firstZ, size_t localZ) {
    const size_t vol = nx * ny * nz, plane = nx * ny;
    for (size_t z=1; z<=localZ; ++z) for (size_t y=0; y<ny; ++y) for (size_t x=0; x<nx; ++x) {
        const size_t id = (firstZ + z - 1) * plane + y * nx + x;
        a[z*plane+y*nx+x] = -1.0 + 2.0 * ((((id+1)*1299709) % vol) / static_cast<double>(vol));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, printResults=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(rank==0) { printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]); } MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (nz < static_cast<size_t>(ranks) || nx*ny > static_cast<size_t>(INT_MAX)) {
        if(rank==0) fprintf(stderr,"Grid must have at least one Z plane per MPI rank and plane size must fit MPI counts.\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t localZ=base+(static_cast<size_t>(rank)<rem), firstZ=rank*base+std::min<size_t>(rank,rem), plane=nx*ny;
    std::vector<double> cold((localZ+2)*plane), cnew((localZ+2)*plane), mu((localZ+2)*plane);
    initialize(cold,nx,ny,nz,firstZ,localZ);
    const double dt=.01, gamma=.5, eAA=-2.0/9.0, eBB=-2.0/9.0, eAB=2.0/9.0;
    if(rank==0) { printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::steady_clock::now();
    for(int t=0;t<iterations;++t) {
        exchangeHalos(cold,plane,localZ,rank,ranks,MPI_COMM_WORLD);
        for(size_t z=1;z<=localZ;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t i=z*plane+y*nx+x; const double c=cold[i];
            const double lap=laplacian(cold,x,y,z,nx,ny,nz,firstZ);
            mu[i]=4.5*((c+1)*eAA+(c-1)*eBB-2*c*eAB)+3*c+c*c*c-gamma*lap;
        }
        exchangeHalos(mu,plane,localZ,rank,ranks,MPI_COMM_WORLD);
        for(size_t z=1;z<=localZ;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x)
            cnew[z*plane+y*nx+x]=cold[z*plane+y*nx+x]+dt*laplacian(mu,x,y,z,nx,ny,nz,firstZ);
        cold.swap(cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD); const auto end=std::chrono::steady_clock::now();
    const double seconds=std::chrono::duration<double>(end-start).count();
    double maxSeconds; MPI_Reduce(&seconds,&maxSeconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",maxSeconds*1000,(nx*ny*nz*static_cast<double>(iterations))/maxSeconds/1e6);

    if(printResults || validate) {
        std::vector<int> counts(ranks), displs(ranks);
        for(int r=0;r<ranks;++r) { const size_t zc=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(zc*plane); displs[r]=static_cast<int>((r*base+std::min<size_t>(r,rem))*plane); }
        std::vector<double> full(rank==0?nx*ny*nz:0);
        MPI_Gatherv(cold.data()+plane,static_cast<int>(localZ*plane),MPI_DOUBLE,full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        int valid=1;
        if(rank==0 && printResults) print_results(full,"Concentration");
        if(rank==0 && validate) {
            double lo=*std::min_element(full.begin(),full.end()), hi=*std::max_element(full.begin(),full.end());
            printf("Concentration range: [%.6f, %.6f]\n",lo,hi);
            valid=std::all_of(full.begin(),full.end(),[](double v){return std::isfinite(v);}) && lo>=-10 && hi<=10;
            printf("Validation: %s\n",valid?"PASSED":"FAILED");
        }
        MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Finalize(); return validate && !valid ? 1 : 0;
    }
    MPI_Finalize(); return 0;
}
