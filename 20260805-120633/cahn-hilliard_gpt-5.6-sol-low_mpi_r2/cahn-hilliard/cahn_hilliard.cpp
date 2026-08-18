#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

struct Decomposition {
    size_t local_nz;
    size_t z0;
};

Decomposition decompose(size_t nz, int rank, int ranks) {
    const size_t q = nz / static_cast<size_t>(ranks);
    const size_t r = nz % static_cast<size_t>(ranks);
    const size_t extra = static_cast<size_t>(rank) < r ? 1 : 0;
    return {q + extra, static_cast<size_t>(rank) * q + std::min(static_cast<size_t>(rank), r)};
}

// Exchange the two Z ghost planes. MPI_PROC_NULL boundaries are filled by
// copying the boundary plane, exactly matching the original clamped stencil.
void exchangeHalos(std::vector<double>& a, size_t local_nz, size_t plane,
                   int rank, int ranks, MPI_Comm comm) {
    const int lower = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int upper = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    const int count = static_cast<int>(plane);

    MPI_Sendrecv(a.data() + plane, count, MPI_DOUBLE, lower, 0,
                 a.data() + (local_nz + 1) * plane, count, MPI_DOUBLE, upper, 0,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + local_nz * plane, count, MPI_DOUBLE, upper, 1,
                 a.data(), count, MPI_DOUBLE, lower, 1, comm, MPI_STATUS_IGNORE);

    if (lower == MPI_PROC_NULL)
        std::copy_n(a.data() + plane, plane, a.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(a.data() + local_nz * plane, plane,
                    a.data() + (local_nz + 1) * plane);
}

void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              size_t nx, size_t ny, size_t local_nz,
                              double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t ym = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = idx3(x, y, z, nx, ny);
                const double cv = c[i];
                const double cxx = c[idx3(xp,y,z,nx,ny)] + c[idx3(xm,y,z,nx,ny)] - 2.0*cv;
                const double cyy = c[idx3(x,yp,z,nx,ny)] + c[idx3(x,ym,z,nx,ny)] - 2.0*cv;
                const double czz = c[i+plane] + c[i-plane] - 2.0*cv;
                const double lap = cxx + cyy + czz;
                mu[i] = 4.5*((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
                        + 3.0*cv + cv*cv*cv - gamma*lap;
            }
        }
    }
}

void update(std::vector<double>& out, const std::vector<double>& in,
            const std::vector<double>& mu, size_t nx, size_t ny, size_t local_nz,
            double dtD) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t ym = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = idx3(x,y,z,nx,ny);
                const double mv = mu[i];
                const double mxx = mu[idx3(xp,y,z,nx,ny)] + mu[idx3(xm,y,z,nx,ny)] - 2.0*mv;
                const double myy = mu[idx3(x,yp,z,nx,ny)] + mu[idx3(x,ym,z,nx,ny)] - 2.0*mv;
                const double mzz = mu[i+plane] + mu[i-plane] - 2.0*mv;
                const double lap = mxx + myy + mzz;
                out[i] = in[i] + dtD*lap;
            }
        }
    }
}

void initialize(std::vector<double>& c, size_t nx, size_t ny, size_t nz,
                size_t local_nz, size_t z0) {
    const size_t vol = nx*ny*nz;
    for (size_t z = 0; z < local_nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_id = (z0+z)*nx*ny + y*nx + x;
                const double pseudo = ((global_id+1)*1299709 % vol) / static_cast<double>(vol);
                c[idx3(x,y,z+1,nx,ny)] = -1.0 + 2.0*pseudo;
            }
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n  -y <num>  Grid Y (default X)\n  -z <num>  Grid Z (default X)\n  -i <num>  Time steps (default 20)\n  -v        Validate\n  -r        Print results\n  -h        Help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx=64, ny=0, nz=0; int iterations=20; bool validate=false, printResults=false;
    bool help=false, bad=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else bad=true;
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (help || bad || !nx || !ny || !nz || iterations < 0) {
        if (world_rank==0) { if (bad) std::printf("Invalid option or argument\n"); printUsage(argv[0]); }
        MPI_Finalize(); return bad ? 1 : 0;
    }
    if (nx*ny > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (world_rank==0) std::fprintf(stderr,"A single XY plane exceeds the MPI count limit\n");
        MPI_Finalize(); return 1;
    }

    // Empty ranks are excluded when more processes than Z planes are launched.
    const int active_size = std::min(world_size, static_cast<int>(nz));
    MPI_Comm comm=MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank<active_size ? 0 : MPI_UNDEFINED, world_rank, &comm);
    if (world_rank>=active_size) { MPI_Finalize(); return 0; }
    int rank, ranks; MPI_Comm_rank(comm,&rank); MPI_Comm_size(comm,&ranks);
    const auto d=decompose(nz,rank,ranks); const size_t plane=nx*ny;
    std::vector<double> cold((d.local_nz+2)*plane), cnew((d.local_nz+2)*plane), mu((d.local_nz+2)*plane);
    initialize(cold,nx,ny,nz,d.local_nz,d.z0);

    if (rank==0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\nMPI processes: %d\nInitializing concentration field...\nRunning Cahn-Hilliard simulation...\n",nx,ny,nz,iterations,validate?"enabled":"disabled",ranks);
    }
    MPI_Barrier(comm); const double start=MPI_Wtime();
    constexpr double gamma=.5, e_AA=-(2.0/9.0), e_BB=-(2.0/9.0), e_AB=2.0/9.0, dtD=.01;
    for (int t=0; t<iterations; ++t) {
        exchangeHalos(cold,d.local_nz,plane,rank,ranks,comm);
        computeChemicalPotential(cold,mu,nx,ny,d.local_nz,gamma,e_AA,e_BB,e_AB);
        exchangeHalos(mu,d.local_nz,plane,rank,ranks,comm);
        update(cnew,cold,mu,nx,ny,d.local_nz,dtD);
        cold.swap(cnew);
    }
    const double local_elapsed=MPI_Wtime()-start; double elapsed=0;
    MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,comm);
    if (rank==0) {
        std::printf("Computation time: %ld ms\n",static_cast<long>(elapsed*1000.0));
        const double updates=static_cast<double>(nx)*ny*nz*iterations;
        std::printf("Performance: %.3f MCellUpdates/s\n",elapsed>0 ? updates/elapsed/1e6 : 0.0);
    }

    std::vector<double> global;
    if (printResults) {
        std::vector<int> counts(ranks), displs(ranks);
        for (int r=0;r<ranks;++r) { auto rd=decompose(nz,r,ranks); counts[r]=static_cast<int>(rd.local_nz*plane); displs[r]=static_cast<int>(rd.z0*plane); }
        if (rank==0) global.resize(nx*ny*nz);
        MPI_Gatherv(cold.data()+plane,static_cast<int>(d.local_nz*plane),MPI_DOUBLE,
                    rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,comm);
        if (rank==0) print_results(global,"Concentration");
    }

    int local_ok=1;
    double local_min=std::numeric_limits<double>::infinity(), local_max=-local_min;
    if (validate) for (size_t i=plane;i<(d.local_nz+1)*plane;++i) {
        if (!std::isfinite(cold[i])) local_ok=0;
        local_min=std::min(local_min,cold[i]); local_max=std::max(local_max,cold[i]);
    }
    int ok=1; double global_min=0,global_max=0;
    if (validate) {
        MPI_Allreduce(&local_ok,&ok,1,MPI_INT,MPI_MIN,comm);
        MPI_Reduce(&local_min,&global_min,1,MPI_DOUBLE,MPI_MIN,0,comm);
        MPI_Reduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,0,comm);
        if (rank==0) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",global_min,global_max);
            ok = ok && global_max<=10.0 && global_min>=-10.0;
            std::printf("Validation: %s\n",ok?"PASSED":"FAILED");
        }
        MPI_Bcast(&ok,1,MPI_INT,0,comm);
    }
    MPI_Comm_free(&comm); MPI_Finalize(); return ok?0:1;
}
