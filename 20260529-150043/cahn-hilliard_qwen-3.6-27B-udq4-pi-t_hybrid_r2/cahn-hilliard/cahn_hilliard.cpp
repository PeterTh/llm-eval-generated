#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Kernels
// ============================================================================

// 3D index calculation (device)
__device__ __forceinline__ size_t idx3d(const size_t x, const size_t y, const size_t z,
                                        const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Device Laplacian: clamped boundaries for x/y, ghost-cell for z
// Ghost cells in z direction are maintained by host-side MPI exchange
__device__ double computeLaplacianDev(const double* __restrict__ c,
                                      const size_t nx, const size_t ny, const size_t nz,
                                      const double dx, const double dy, const double dz,
                                      const size_t x, const size_t y, const size_t z) {
    // Clamped boundaries for x and y (matches original code)
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    // For z: ghost cells are maintained externally (MPI exchange + clamped BCs)
    // so we just use z-1 and z+1 directly (always valid within local domain)
    const size_t zp = z + 1;
    const size_t zn = z - 1;

    const double cxx = (c[idx3d(xp, y, z, nx, ny)] + c[idx3d(xn, y, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3d(x, yp, z, nx, ny)] + c[idx3d(x, yn, z, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3d(x, y, zp, nx, ny)] + c[idx3d(x, y, zn, nx, ny)] -
                        2.0 * c[idx3d(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Initialize concentration field kernel (data cells only)
__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                              const size_t nx, const size_t ny, const size_t nz_data,
                                              const size_t global_nx, const size_t global_ny,
                                              const size_t global_nz,
                                              const size_t z_offset,
                                              const size_t nz_ghost) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz_data) {
        const size_t local_z = z + nz_ghost;
        const size_t idx = idx3d(x, y, local_z, nx, ny);
        const size_t global_linear_id = (z + z_offset) * (global_nx * global_ny) + y * global_nx + x;
        const size_t vol = global_nx * global_ny * global_nz;
        const double pseudo = ((((global_linear_id + 1) * 1299709ULL) % vol) /
                               static_cast<double>(vol));
        c[idx] = -1.0 + 2.0 * pseudo;
    }
}

// Compute chemical potential kernel (data cells only)
__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const size_t nx, const size_t ny,
                                               const size_t nz_data, const size_t nz_local,
                                               const double dx, const double dy, const double dz,
                                               const double gamma, const double e_AA,
                                               const double e_BB, const double e_AB,
                                               const size_t nz_ghost) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz_data) {
        const size_t local_z = z + nz_ghost;
        const size_t idx = idx3d(x, y, local_z, nx, ny);
        const double cv = c[idx];

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * computeLaplacianDev(c, nx, ny, nz_local, dx, dy, dz,
                                               x, y, local_z);
    }
}

// Cahn-Hilliard update kernel (data cells only)
__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny,
                                         const size_t nz_data, const size_t nz_local,
                                         const double D, const double dt,
                                         const double dx, const double dy, const double dz,
                                         const size_t nz_ghost) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz_data) {
        const size_t local_z = z + nz_ghost;
        const size_t idx = idx3d(x, y, local_z, nx, ny);
        cnew[idx] = cold[idx] + dt * D *
                    computeLaplacianDev(mu, nx, ny, nz_local, dx, dy, dz,
                                        x, y, local_z);
    }
}

// ============================================================================
// Host-side helpers
// ============================================================================

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct LocalDomain {
    size_t local_nx;
    size_t local_ny;
    size_t local_nz;
    size_t nz_data;
    size_t nz_ghost;
    size_t z_offset;
    size_t global_nx;
    size_t global_ny;
    size_t global_nz;
    int rank;
    int num_ranks;
};

LocalDomain computeLocalDomain(size_t gnx, size_t gny, size_t gnz,
                               int rank, int nranks) {
    LocalDomain d;
    d.global_nx = gnx; d.global_ny = gny; d.global_nz = gnz;
    d.rank = rank; d.num_ranks = nranks;
    d.local_nx = gnx; d.local_ny = gny; d.nz_ghost = 1;

    size_t base = gnz / nranks;
    size_t rem = gnz % nranks;
    d.nz_data = base + (rank < static_cast<int>(rem) ? 1 : 0);

    size_t offset = 0;
    for (int r = 0; r < rank; ++r)
        offset += base + (r < static_cast<int>(rem) ? 1 : 0);
    d.z_offset = offset;

    d.local_nz = d.nz_data + 2 * d.nz_ghost;
    return d;
}

void exchangeGhostCells(double* h_data, const LocalDomain& dom) {
    const size_t gs = dom.local_nx * dom.local_ny;
    const size_t nx = dom.local_nx;
    const size_t ny = dom.local_ny;
    const size_t nzl = dom.local_nz;
    const size_t nzd = dom.nz_data;
    const size_t ng = dom.nz_ghost;

    std::vector<double> send_top(gs), send_bottom(gs);
    std::vector<double> recv_top(gs), recv_bottom(gs);

    // Top data layer
    const size_t tidx = idx3(0, 0, ng, nx, ny);
    std::memcpy(send_top.data(), &h_data[tidx], gs * sizeof(double));

    // Bottom data layer
    const size_t bidx = idx3(0, 0, ng + nzd - 1, nx, ny);
    std::memcpy(send_bottom.data(), &h_data[bidx], gs * sizeof(double));

    MPI_Request reqs[4];
    int cnt = 0;

    if (dom.rank > 0)
        MPI_Isend(send_top.data(), gs, MPI_DOUBLE, dom.rank - 1, 0,
                  MPI_COMM_WORLD, &reqs[cnt++]);
    if (dom.rank < dom.num_ranks - 1)
        MPI_Irecv(recv_top.data(), gs, MPI_DOUBLE, dom.rank + 1, 0,
                  MPI_COMM_WORLD, &reqs[cnt++]);

    if (dom.rank < dom.num_ranks - 1)
        MPI_Isend(send_bottom.data(), gs, MPI_DOUBLE, dom.rank + 1, 1,
                  MPI_COMM_WORLD, &reqs[cnt++]);
    if (dom.rank > 0)
        MPI_Irecv(recv_bottom.data(), gs, MPI_DOUBLE, dom.rank - 1, 1,
                  MPI_COMM_WORLD, &reqs[cnt++]);

    if (cnt > 0)
        MPI_Waitall(cnt, reqs, MPI_STATUSES_IGNORE);

    // Top ghost cells
    if (dom.rank == 0) {
        // Clamped BC: top ghost = first data layer
        const size_t tgidx = idx3(0, 0, 0, nx, ny);
        for (size_t g = 0; g < ng; ++g)
            std::memcpy(&h_data[tgidx + g * nx * ny],
                        send_top.data(), gs * sizeof(double));
    } else {
        const size_t tgidx = idx3(0, 0, 0, nx, ny);
        for (size_t g = 0; g < ng; ++g)
            std::memcpy(&h_data[tgidx + g * nx * ny],
                        recv_top.data(), gs * sizeof(double));
    }

    // Bottom ghost cells
    if (dom.rank == dom.num_ranks - 1) {
        // Clamped BC: bottom ghost = last data layer
        const size_t bgidx = idx3(0, 0, nzl - ng, nx, ny);
        for (size_t g = 0; g < ng; ++g)
            std::memcpy(&h_data[bgidx + g * nx * ny],
                        send_bottom.data(), gs * sizeof(double));
    } else {
        const size_t bgidx = idx3(0, 0, nzl - ng, nx, ny);
        for (size_t g = 0; g < ng; ++g)
            std::memcpy(&h_data[bgidx + g * nx * ny],
                        recv_bottom.data(), gs * sizeof(double));
    }
}

void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    LocalDomain dom = computeLocalDomain(nx, ny, nz, rank, num_ranks);
    const size_t lnx = dom.local_nx, lny = dom.local_ny, lnz = dom.local_nz;
    const size_t nzd = dom.nz_data, ng = dom.nz_ghost;
    const size_t local_size = lnx * lny * lnz;
    const size_t global_size = nx * ny * nz;

    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0/9.0), e_BB = -(2.0/9.0), e_AB = (2.0/9.0);
    const double gamma = 0.5, D = 1.0;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("MPI ranks: %d\n", num_ranks);
        printf("CUDA hybrid parallelization enabled\n");
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> h_cold(local_size, 0.0);
    std::vector<double> h_cnew(local_size, 0.0);
    std::vector<double> h_mu(local_size, 0.0);

    double* d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cudaMalloc(&d_cold, local_size * sizeof(double));
    cudaMalloc(&d_cnew, local_size * sizeof(double));
    cudaMalloc(&d_mu, local_size * sizeof(double));

    // Initialize on GPU
    {
        dim3 bs(8, 8, 8);
        dim3 gs((lnx+bs.x-1)/bs.x, (lny+bs.y-1)/bs.y, (nzd+bs.z-1)/bs.z);
        initializeConcentrationKernel<<<gs, bs>>>(
            d_cold, lnx, lny, nzd, nx, ny, nz, dom.z_offset, ng);
        cudaDeviceSynchronize();
    }

    // Set ghost cells and copy to device
    cudaMemcpy(h_cold.data(), d_cold, local_size * sizeof(double), cudaMemcpyDeviceToHost);
    exchangeGhostCells(h_cold.data(), dom);
    cudaMemcpy(d_cold, h_cold.data(), local_size * sizeof(double), cudaMemcpyHostToDevice);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        dim3 bs(8, 8, 8);
        dim3 gs((lnx+bs.x-1)/bs.x, (lny+bs.y-1)/bs.y, (nzd+bs.z-1)/bs.z);

        computeChemicalPotentialKernel<<<gs, bs>>>(
            d_cold, d_mu, lnx, lny, nzd, lnz, dx, dy, dz,
            gamma, e_AA, e_BB, e_AB, ng);

        cahnHilliardUpdateKernel<<<gs, bs>>>(
            d_cnew, d_cold, d_mu, lnx, lny, nzd, lnz,
            D, dt, dx, dy, dz, ng);

        cudaDeviceSynchronize();

        cudaMemcpy(h_cold.data(), d_cnew, local_size * sizeof(double),
                   cudaMemcpyDeviceToHost);
        exchangeGhostCells(h_cold.data(), dom);
        cudaMemcpy(d_cold, h_cold.data(), local_size * sizeof(double),
                   cudaMemcpyHostToDevice);

        std::swap(d_cold, d_cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    cudaMemcpy(h_cold.data(), d_cold, local_size * sizeof(double), cudaMemcpyDeviceToHost);
    MPI_Barrier(MPI_COMM_WORLD);

    double local_cellUpdates = static_cast<double>(local_size) * iterations;
    double global_cellUpdates;
    MPI_Reduce(&local_cellUpdates, &global_cellUpdates, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double mcups = duration.count() > 0 ?
        global_cellUpdates / (duration.count() / 1000.0) / 1e6 : 0.0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full solution on rank 0
    std::vector<double> global_cold;
    if (rank == 0) global_cold.resize(global_size);

    {
        size_t gather_count = lnx * lny * nzd;
        std::vector<double> h_data(gather_count);
        const size_t doff = idx3(0, 0, ng, lnx, lny);
        #pragma omp parallel for collapse(2)
        for (size_t z = 0; z < nzd; ++z)
            for (size_t i = 0; i < lnx * lny; ++i)
                h_data[z * lnx * lny + i] = h_cold[doff + z * lnx * lny + i];

        std::vector<int> displs(num_ranks), counts(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
            LocalDomain rd = computeLocalDomain(nx, ny, nz, r, num_ranks);
            counts[r] = static_cast<int>(rd.local_nx * rd.local_ny * rd.nz_data);
            size_t disp = 0;
            for (int rr = 0; rr < r; ++rr) {
                LocalDomain rrd = computeLocalDomain(nx, ny, nz, rr, num_ranks);
                disp += rrd.local_nx * rrd.local_ny * rrd.nz_data;
            }
            displs[r] = static_cast<int>(disp);
        }

        MPI_Gatherv(h_data.data(), gather_count, MPI_DOUBLE,
                    global_cold.data(), counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0)
        print_results(global_cold, "Concentration");

    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = true;
        for (const auto& val : global_cold) {
            if (std::isnan(val) || std::isinf(val)) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = false; break;
            }
        }
        if (valid) {
            double minVal = global_cold[0], maxVal = global_cold[0];
            for (const auto& val : global_cold) {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu);
    MPI_Finalize();
    return 0;
}
