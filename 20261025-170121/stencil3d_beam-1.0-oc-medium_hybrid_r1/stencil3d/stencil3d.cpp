#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void stencilKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz_local, size_t plane) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1 || k == 0 || k >= nz_local+1) return;
    size_t idx = k*(nx*ny) + j*nx + i;
    Real center = input[idx];
    Real left = input[idx-1];
    Real right = input[idx+1];
    Real front = input[idx-nx];
    Real back = input[idx+nx];
    Real bottom = input[idx-nx*ny];
    Real top = input[idx+nx*ny];
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(3)
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
            else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
            else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
            else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
            else if (strcmp(argv[i], "-v") == 0) validate = true;
            else if (strcmp(argv[i], "-r") == 0) printResults = true;
        }
        if (ny == 0) ny = nx;
        if (nz == 0) nz = nx;
    }
    MPI_Bcast(&nx, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // broadcast bools as int
    int v = validate, p = printResults;
    MPI_Bcast(&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = v; printResults = p;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // partition interior z planes [1, nz-1)
    size_t interior = (nz > 2) ? (nz - 2) : 0;
    size_t base = interior / size;
    size_t rem = interior % size;
    size_t myInterior = base + (rank < (int)rem ? 1 : 0);
    size_t offset = rank * base + (rank < (int)rem ? rank : rem);
    // global z of first interior plane for this rank
    size_t z0 = 1 + offset;

    size_t planes = myInterior + 2; // +2 halo
    size_t planeSize = nx * ny;
    size_t localSize = planes * planeSize;

    std::vector<Real> grid1(localSize), grid2(localSize);
    // initialize local including halo
    for (size_t li = 0; li < planes; ++li) {
        size_t gz = (li == 0) ? (z0 > 1 ? z0 - 1 : 0) : (li == planes-1 ? z0 + myInterior : z0 + li - 1);
        if (gz >= nz) gz = nz - 1;
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                size_t gidx = idx3(x, y, gz, nx, ny);
                grid1[li*planeSize + y*nx + x] = (gidx % 19) * 1.0;
                grid2[li*planeSize + y*nx + x] = grid1[li*planeSize + y*nx + x];
            }
    }

    // device buffers
    Real *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, localSize * sizeof(Real));
    cudaMalloc(&d_out, localSize * sizeof(Real));

    auto runIter = [&](std::vector<Real>& in, std::vector<Real>& out) {
        cudaMemcpy(d_in, in.data(), localSize * sizeof(Real), cudaMemcpyHostToDevice);
        dim3 block(8,8,4);
        dim3 grid((nx+7)/8, (ny+7)/8, (planes+3)/4);
        stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, myInterior, planeSize);
        cudaDeviceSynchronize();
        cudaMemcpy(out.data(), d_out, localSize * sizeof(Real), cudaMemcpyDeviceToHost);

        for (size_t li = 0; li < planes; ++li)
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x)
                    if (x==0||x==nx-1||y==0||y==ny-1) {
                        size_t idx = li*planeSize + y*nx + x;
                        out[idx] = in[idx];
                    }
        for (size_t li = 0; li < planes; ++li)
            if (li==0 || li==planes-1)
                memcpy(out.data()+li*planeSize, in.data()+li*planeSize, planeSize*sizeof(Real));
        // halo exchange
        // bottom halo (li=0) <-> prev rank's top interior (li=planes-2?) 
        // We'll exchange plane li=1 (first interior) with prev, and li=myInterior (last interior) with next.
        if (size > 1) {
            // send first interior plane to prev
            if (rank > 0) {
                MPI_Sendrecv(out.data() + 1*planeSize, planeSize, MPI_DOUBLE, rank-1, 0,
                             out.data(), planeSize, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) {
                MPI_Sendrecv(out.data() + myInterior*planeSize, planeSize, MPI_DOUBLE, rank+1, 1,
                             out.data() + (myInterior+1)*planeSize, planeSize, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // boundary halo fixed already
    };

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) runIter(grid1, grid2);
        else runIter(grid2, grid1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates = (double)interior * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // gather final grid to rank0
    const std::vector<Real>& finalLocal = (iterations % 2 == 0) ? grid1 : grid2;
    std::vector<Real> globalGrid;
    if (rank == 0) globalGrid.resize(nx*ny*nz);
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        // recompute interior count
        size_t ic = base + (r < (int)rem ? 1 : 0);
        size_t off = r * base + (r < (int)rem ? r : rem);
        counts[r] = ic * planeSize;
        displs[r] = (1 + off) * planeSize;
    }
    MPI_Gatherv(finalLocal.data() + 1*planeSize, myInterior*planeSize, MPI_DOUBLE,
                globalGrid.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // set boundary values initial
        for (size_t z = 0; z < nz; ++z)
            for (size_t y = 0; y < ny; ++y)
                for (size_t x = 0; x < nx; ++x)
                    if (x==0||x==nx-1||y==0||y==ny-1||z==0||z==nz-1) {
                        size_t idx = idx3(x,y,z,nx,ny);
                        globalGrid[idx] = (idx % 19) * 1.0;
                    }
        if (printResults) print_results(globalGrid, "Grid");
        if (validate) {
            printf("Validating result...\n");
            bool valid = true;
            for (auto& v : globalGrid) if (std::isnan(v)||std::isinf(v)) valid=false;
            Real mn=globalGrid[0], mx=globalGrid[0];
            for (auto& v:globalGrid){mn=std::min(mn,v);mx=std::max(mx,v);}
            printf("Value range: [%.6f, %.6f]\n", mn, mx);
            if (mx > 1e6 || mn < -1e6) valid=false;
            printf("Validation: %s\n", valid?"PASSED":"FAILED");
        }
    }

    cudaFree(d_in); cudaFree(d_out);
    MPI_Finalize();
    return 0;
}
