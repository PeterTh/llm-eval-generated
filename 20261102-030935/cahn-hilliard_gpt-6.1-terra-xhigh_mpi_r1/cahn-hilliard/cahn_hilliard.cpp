#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Each rank owns complete XY planes, with one ghost plane on either side.
// Physical boundaries use the same clamped (zero normal derivative) values as
// the original implementation.
class HaloExchange {
public:
    HaloExchange(size_t plane, int rank, int ranks, MPI_Comm comm)
        : plane_(plane), lower_(rank == 0 ? MPI_PROC_NULL : rank - 1),
          upper_(rank + 1 == ranks ? MPI_PROC_NULL : rank + 1), comm_(comm) {}

    void start(std::vector<double>& field, size_t planes) {
        double* data = field.data();
        if (lower_ == MPI_PROC_NULL)
            std::copy_n(data + plane_, plane_, data);
        if (upper_ == MPI_PROC_NULL)
            std::copy_n(data + planes * plane_, plane_, data + (planes + 1) * plane_);
        requests_.clear();
        // Chunk transfers to remain valid even when a plane exceeds MPI's int count.
        for (size_t offset = 0; offset < plane_;) {
            const int count = static_cast<int>(std::min(plane_ - offset, static_cast<size_t>(INT_MAX)));
            MPI_Request req;
            MPI_Irecv(data + offset, count, MPI_DOUBLE, lower_, 1, comm_, &req);
            requests_.push_back(req);
            MPI_Irecv(data + (planes + 1) * plane_ + offset, count, MPI_DOUBLE, upper_, 0, comm_, &req);
            requests_.push_back(req);
            MPI_Isend(data + plane_ + offset, count, MPI_DOUBLE, lower_, 0, comm_, &req);
            requests_.push_back(req);
            MPI_Isend(data + planes * plane_ + offset, count, MPI_DOUBLE, upper_, 1, comm_, &req);
            requests_.push_back(req);
            offset += static_cast<size_t>(count);
        }
    }

    void finish() {
        MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
    }

private:
    size_t plane_;
    int lower_, upper_;
    MPI_Comm comm_;
    std::vector<MPI_Request> requests_;
};

// z is a local plane index, including the leading ghost plane.
inline double laplacian(const double* a, size_t i, size_t x, size_t y,
                        size_t nx, size_t ny, size_t plane) {
    const size_t xp = x + 1 < nx ? i + 1 : i;
    const size_t xn = x > 0 ? i - 1 : i;
    const size_t yp = y + 1 < ny ? i + nx : i;
    const size_t yn = y > 0 ? i - nx : i;
    const double center = a[i];
    const double cxx = a[xp] + a[xn] - 2.0 * center;
    const double cyy = a[yp] + a[yn] - 2.0 * center;
    const double czz = a[i + plane] + a[i - plane] - 2.0 * center;
    return cxx + cyy + czz;
}

void chemical_planes(const std::vector<double>& c, std::vector<double>& mu,
                     size_t nx, size_t ny, size_t first, size_t end) {
    const size_t plane = nx * ny;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double* src = c.data();
    double* dst = mu.data();
    for (size_t z = first; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = z * plane + y * nx + x;
                const double cv = src[i];
                dst[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - 0.5 * laplacian(src, i, x, y, nx, ny, plane);
            }
}

void update_planes(std::vector<double>& next, const std::vector<double>& c,
                   const std::vector<double>& mu, size_t nx, size_t ny,
                   size_t first, size_t end) {
    const size_t plane = nx * ny;
    const double* src = c.data();
    const double* potential = mu.data();
    double* dst = next.data();
    for (size_t z = first; z < end; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t i = z * plane + y * nx + x;
                dst[i] = src[i] + 0.01 * laplacian(potential, i, x, y, nx, ny, plane);
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
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions\n");
        MPI_Finalize();
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) {
        MPI_Finalize();
        return 0;
    }
    const size_t localPlanes = nz / active + (static_cast<size_t>(rank) < nz % active);
    const size_t zStart = static_cast<size_t>(rank) * (nz / active) + std::min(static_cast<size_t>(rank), nz % active);
    const size_t localSize = localPlanes * plane;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }
    std::vector<double> cold((localPlanes + 2) * plane), cnew((localPlanes + 2) * plane), mu((localPlanes + 2) * plane);
    for (size_t i = 0; i < localSize; ++i) {
        const size_t linear_id = zStart * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[plane + i] = -1.0 + 2.0 * pseudo;
    }
    HaloExchange halo(plane, rank, active, comm);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        halo.start(cold, localPlanes);
        if (localPlanes > 2) chemical_planes(cold, mu, nx, ny, 2, localPlanes);
        halo.finish();
        chemical_planes(cold, mu, nx, ny, 1, 2);
        if (localPlanes > 1) chemical_planes(cold, mu, nx, ny, localPlanes, localPlanes + 1);
        halo.start(mu, localPlanes);
        if (localPlanes > 2) update_planes(cnew, cold, mu, nx, ny, 2, localPlanes);
        halo.finish();
        update_planes(cnew, cold, mu, nx, ny, 1, 2);
        if (localPlanes > 1) update_planes(cnew, cold, mu, nx, ny, localPlanes, localPlanes + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double mcups = static_cast<double>(gridSize) * iterations / (milliseconds / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    if (printResults) {
        // Preserve the original global order, Kahan sum and hash, without MPI's int
        // count/displacement limit. This collection is outside the timed region.
        std::vector<double> result;
        if (rank == 0) {
            result.resize(gridSize);
            std::copy_n(cold.data() + plane, localSize, result.data());
            for (int r = 1; r < active; ++r) {
                const size_t nplanes = nz / active + (static_cast<size_t>(r) < nz % active);
                size_t offset = (static_cast<size_t>(r) * (nz / active) + std::min(static_cast<size_t>(r), nz % active)) * plane;
                size_t remaining = nplanes * plane;
                while (remaining) {
                    const int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset, count, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    offset += count; remaining -= count;
                }
            }
            print_results(result, "Concentration");
        } else {
            size_t offset = 0;
            while (offset < localSize) {
                const int count = static_cast<int>(std::min(localSize - offset, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + plane + offset, count, MPI_DOUBLE, 0, 2, comm);
                offset += count;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int localFinite = 1;
        double localMin = std::numeric_limits<double>::infinity();
        double localMax = -std::numeric_limits<double>::infinity();
        for (size_t i = plane; i < plane + localSize; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        int finite = 0;
        double minVal = 0, maxVal = 0;
        MPI_Allreduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, comm);
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) {
                printf("Validation failed: found NaN or Inf value\n");
                valid = 0;
            } else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
