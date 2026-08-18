// QT clustering benchmark: MPI distributes candidate seeds, CUDA evaluates
// candidate-cluster cardinalities, and OpenMP reconstructs the selected cluster.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH = 20.0;
static constexpr double MAX_HEIGHT = 20.0;
static constexpr int CUDA_THREADS = 256;
static constexpr int MAX_CANDIDATE_BATCH = 4096;

struct Point {
    double x;
    double y;
};

struct Cluster {
    std::vector<int> members;
    int seed_point;
};

struct CandidateBest {
    int cardinality;
    int seed;
};

static void cudaCheck(cudaError_t status, const char* expression,
                      const char* file, int line) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s:%d: CUDA call %s failed: %s",
                      file, line, expression, cudaGetErrorString(status));
        throw std::runtime_error(message);
    }
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

// The generated data is kept bit-for-bit identical to the original benchmark.
static void generateSyntheticData(std::vector<Point>& points, int pointCount,
                                  unsigned int seed = 42) {
    auto frand = [&seed]() mutable {
        return rand_r(&seed) / static_cast<double>(RAND_MAX);
    };

    const double minDim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    while (count < pointCount) {
        const double centerX = frand() * MAX_WIDTH;
        const double centerY = frand() * MAX_HEIGHT;
        const double radius = frand() * minDim / 2.0;
        int groupCount = static_cast<int>(frand() * (pointCount / 30.0));
        groupCount = std::min(groupCount, pointCount - count);

        while (groupCount > 0) {
            const double sign = frand() < 0.5 ? -1.0 : 1.0;
            const double radiusOffset = frand() * radius;
            const double dx = (2.0 * frand() - 1.0) * radiusOffset;
            const double dy = std::sqrt(radiusOffset * radiusOffset - dx * dx) * sign;
            const double x = centerX + dx;
            const double y = centerY + dy;
            if (x < 0.0 || x > MAX_WIDTH || y < 0.0 || y > MAX_HEIGHT) {
                continue;
            }
            points[count++] = {x, y};
            --groupCount;
        }
    }
}

// QT only compares distances, so squared distance avoids an expensive square root
// without changing either the ordering or the threshold predicate.
inline double squaredDistance(const Point& first, const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return dx * dx + dy * dy;
}

inline double distance(const Point& first, const Point& second) {
    return std::sqrt(squaredDistance(first, second));
}

static bool isBetterCandidate(const CandidateBest& candidate,
                              const CandidateBest& current) {
    return candidate.cardinality > current.cardinality ||
           (candidate.cardinality == current.cardinality &&
            candidate.cardinality >= 0 && candidate.seed < current.seed);
}

// MPI user operation for the deterministic QT tie-break: maximum cardinality,
// then the lowest input index.  It makes the result independent of rank count.
extern "C" void reduceCandidateBest(void* incoming, void* inoutgoing, int* length,
                                    MPI_Datatype*) {
    auto* in = static_cast<CandidateBest*>(incoming);
    auto* inout = static_cast<CandidateBest*>(inoutgoing);
    for (int i = 0; i < *length; ++i) {
        if (isBetterCandidate(in[i], inout[i])) {
            inout[i] = in[i];
        }
    }
}

__device__ __forceinline__ double deviceSquaredDistance(const Point& first,
                                                         const Point& second) {
    const double dx = first.x - second.x;
    const double dy = first.y - second.y;
    return dx * dx + dy * dy;
}

// One CUDA block owns one candidate seed.  The full greedy loop lives in the
// block so a candidate requires one launch rather than one host launch for
// every selected member.  Its row of state is private to that block.
__global__ void evaluateCandidateRows(const Point* points, const int* seeds,
                                      const unsigned char* clustered,
                                      int batchCount, int pointCount,
                                      double thresholdSquared,
                                      double* maxDistances,
                                      unsigned char* inCluster,
                                      int* cardinalities) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= batchCount) {
        return;
    }

    __shared__ double bestDistances[CUDA_THREADS];
    __shared__ int bestPoints[CUDA_THREADS];

    const int seed = seeds[row];
    const size_t rowOffset = static_cast<size_t>(row) * pointCount;
    for (int point = thread; point < pointCount; point += blockDim.x) {
        const size_t offset = rowOffset + point;
        maxDistances[offset] = deviceSquaredDistance(points[seed], points[point]);
        inCluster[offset] = static_cast<unsigned char>(point == seed);
    }
    if (thread == 0) {
        cardinalities[row] = 1;
    }
    __syncthreads();

    for (int iteration = 1; iteration < pointCount; ++iteration) {
        double localDistance = DBL_MAX;
        int localPoint = INT_MAX;
        for (int point = thread; point < pointCount; point += blockDim.x) {
            const double candidateDistance = maxDistances[rowOffset + point];
            if (!clustered[point] && !inCluster[rowOffset + point] &&
                candidateDistance < thresholdSquared &&
                (candidateDistance < localDistance ||
                 (candidateDistance == localDistance && point < localPoint))) {
                localDistance = candidateDistance;
                localPoint = point;
            }
        }
        bestDistances[thread] = localDistance;
        bestPoints[thread] = localPoint;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (thread < stride) {
                const double otherDistance = bestDistances[thread + stride];
                const int otherPoint = bestPoints[thread + stride];
                if (otherDistance < bestDistances[thread] ||
                    (otherDistance == bestDistances[thread] &&
                     otherPoint < bestPoints[thread])) {
                    bestDistances[thread] = otherDistance;
                    bestPoints[thread] = otherPoint;
                }
            }
            __syncthreads();
        }

        const int selected = bestPoints[0];
        if (selected == INT_MAX) {
            break;
        }
        for (int point = thread; point < pointCount; point += blockDim.x) {
            const size_t offset = rowOffset + point;
            const double addedDistance = deviceSquaredDistance(points[selected], points[point]);
            if (addedDistance > maxDistances[offset]) {
                maxDistances[offset] = addedDistance;
            }
            if (point == selected) {
                inCluster[offset] = 1;
            }
        }
        if (thread == 0) {
            ++cardinalities[row];
        }
        __syncthreads();
    }
}

class GpuCandidateEvaluator {
  public:
    explicit GpuCandidateEvaluator(const std::vector<Point>& points)
        : pointCount_(static_cast<int>(points.size())) {
        if (pointCount_ <= 0) {
            throw std::runtime_error("CUDA evaluator requires at least one point");
        }

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&devicePoints_, static_cast<size_t>(pointCount_) * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&deviceClustered_,
                              static_cast<size_t>(pointCount_) * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpyAsync(devicePoints_, points.data(),
                                   static_cast<size_t>(pointCount_) * sizeof(Point),
                                   cudaMemcpyHostToDevice, stream_));

        size_t freeBytes = 0;
        size_t totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        const size_t bytesPerRow = static_cast<size_t>(pointCount_) *
                                   (sizeof(double) + sizeof(unsigned char));
        // Leave half of currently free device memory available for MPI peers and
        // CUDA runtime allocations.  The hard cap keeps batches latency-friendly.
        const size_t memoryLimitedRows = (freeBytes / 2) / bytesPerRow;
        batchCapacity_ = static_cast<int>(std::min<size_t>(
            {static_cast<size_t>(pointCount_), static_cast<size_t>(MAX_CANDIDATE_BATCH),
             std::max<size_t>(1, memoryLimitedRows)}));

        const size_t stateElements = static_cast<size_t>(batchCapacity_) * pointCount_;
        CUDA_CHECK(cudaMalloc(&deviceSeeds_, static_cast<size_t>(batchCapacity_) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&deviceMaxDistances_, stateElements * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&deviceInCluster_, stateElements * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&deviceCardinalities_,
                              static_cast<size_t>(batchCapacity_) * sizeof(int)));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    GpuCandidateEvaluator(const GpuCandidateEvaluator&) = delete;
    GpuCandidateEvaluator& operator=(const GpuCandidateEvaluator&) = delete;

    ~GpuCandidateEvaluator() {
        cudaFree(deviceCardinalities_);
        cudaFree(deviceInCluster_);
        cudaFree(deviceMaxDistances_);
        cudaFree(deviceSeeds_);
        cudaFree(deviceClustered_);
        cudaFree(devicePoints_);
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void uploadClustered(const std::vector<unsigned char>& clustered) {
        CUDA_CHECK(cudaMemcpyAsync(deviceClustered_, clustered.data(),
                                   static_cast<size_t>(pointCount_) * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice, stream_));
    }

    std::vector<int> evaluate(const int* seeds, int seedCount, double thresholdSquared) {
        if (seedCount <= 0 || seedCount > batchCapacity_) {
            throw std::runtime_error("invalid CUDA candidate batch size");
        }

        CUDA_CHECK(cudaMemcpyAsync(deviceSeeds_, seeds,
                                   static_cast<size_t>(seedCount) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream_));
        evaluateCandidateRows<<<seedCount, CUDA_THREADS, 0, stream_>>>(
            devicePoints_, deviceSeeds_, deviceClustered_, seedCount, pointCount_,
            thresholdSquared, deviceMaxDistances_, deviceInCluster_, deviceCardinalities_);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> cardinalities(seedCount);
        CUDA_CHECK(cudaMemcpyAsync(cardinalities.data(), deviceCardinalities_,
                                   static_cast<size_t>(seedCount) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        return cardinalities;
    }

    int batchCapacity() const { return batchCapacity_; }

  private:
    int pointCount_ = 0;
    int batchCapacity_ = 0;
    cudaStream_t stream_ = nullptr;
    Point* devicePoints_ = nullptr;
    unsigned char* deviceClustered_ = nullptr;
    int* deviceSeeds_ = nullptr;
    double* deviceMaxDistances_ = nullptr;
    unsigned char* deviceInCluster_ = nullptr;
    int* deviceCardinalities_ = nullptr;
};

// This is the only membership reconstruction needed in a round.  It retains
// the exact greedy ordering while using OpenMP for both the arg-min scan and
// the incremental diameter update.
static std::vector<int> reconstructSelectedCluster(
    int seed, const std::vector<unsigned char>& clustered,
    const std::vector<Point>& points, double thresholdSquared) {
    const int pointCount = static_cast<int>(points.size());
    std::vector<unsigned char> inCluster(pointCount, 0);
    std::vector<double> maxDistances(pointCount);
    std::vector<int> members;
    members.reserve(pointCount);

    // Keep one OpenMP team alive for this entire greedy reconstruction.  QT
    // clusters can contain many points, so this avoids paying team creation
    // and teardown for every arg-min/update pair.
    int selected = seed;
    int bestPoint = INT_MAX;
    double bestDistance = std::numeric_limits<double>::max();
    bool complete = false;
#pragma omp parallel shared(selected, bestPoint, bestDistance, complete)
    {
#pragma omp for schedule(static)
        for (int point = 0; point < pointCount; ++point) {
            maxDistances[point] = squaredDistance(points[seed], points[point]);
        }
#pragma omp single
        {
            inCluster[seed] = 1;
            members.push_back(seed);
        }

        while (true) {
#pragma omp single
            {
                bestPoint = INT_MAX;
                bestDistance = std::numeric_limits<double>::max();
            }

            int localPoint = INT_MAX;
            double localDistance = std::numeric_limits<double>::max();
#pragma omp for nowait schedule(static)
            for (int point = 0; point < pointCount; ++point) {
                const double candidateDistance = maxDistances[point];
                if (!clustered[point] && !inCluster[point] &&
                    candidateDistance < thresholdSquared &&
                    (candidateDistance < localDistance ||
                     (candidateDistance == localDistance && point < localPoint))) {
                    localDistance = candidateDistance;
                    localPoint = point;
                }
            }
#pragma omp critical(qt_cluster_argmin)
            {
                if (localDistance < bestDistance ||
                    (localDistance == bestDistance && localPoint < bestPoint)) {
                    bestDistance = localDistance;
                    bestPoint = localPoint;
                }
            }
#pragma omp barrier
#pragma omp single
            {
                complete = bestPoint == INT_MAX;
                if (!complete) {
                    selected = bestPoint;
                    inCluster[selected] = 1;
                    members.push_back(selected);
                }
            }
#pragma omp barrier
            if (complete) {
                break;
            }

#pragma omp for schedule(static)
            for (int point = 0; point < pointCount; ++point) {
                const double addedDistance = squaredDistance(points[selected], points[point]);
                if (addedDistance > maxDistances[point]) {
                    maxDistances[point] = addedDistance;
                }
            }
        }
    }
    return members;
}

static CandidateBest evaluateLocalSeeds(
    const std::vector<int>& localSeeds, const std::vector<unsigned char>& clustered,
    double thresholdSquared, GpuCandidateEvaluator& evaluator) {
    CandidateBest localBest{-1, INT_MAX};
    evaluator.uploadClustered(clustered);

    for (size_t begin = 0; begin < localSeeds.size();
         begin += static_cast<size_t>(evaluator.batchCapacity())) {
        const int count = static_cast<int>(std::min(
            static_cast<size_t>(evaluator.batchCapacity()), localSeeds.size() - begin));
        const std::vector<int> cardinalities =
            evaluator.evaluate(localSeeds.data() + begin, count, thresholdSquared);

#pragma omp parallel
        {
            CandidateBest threadBest{-1, INT_MAX};
#pragma omp for nowait schedule(static)
            for (int index = 0; index < count; ++index) {
                const CandidateBest candidate{cardinalities[index],
                                               localSeeds[begin + index]};
                if (isBetterCandidate(candidate, threadBest)) {
                    threadBest = candidate;
                }
            }
#pragma omp critical(qt_seed_argmax)
            {
                if (isBetterCandidate(threadBest, localBest)) {
                    localBest = threadBest;
                }
            }
        }
    }
    return localBest;
}

static std::vector<Cluster> qtClusteringDistributed(
    const std::vector<Point>& points, double threshold, int rank, int worldSize,
    MPI_Datatype candidateType, MPI_Op candidateOp) {
    const int pointCount = static_cast<int>(points.size());
    const double thresholdSquared = threshold * threshold;
    std::vector<unsigned char> clustered(pointCount, 0);
    std::vector<Cluster> clusters;
    GpuCandidateEvaluator evaluator(points);
    int remaining = pointCount;

    while (remaining > 0) {
        std::vector<int> localSeeds;
        localSeeds.reserve((remaining + worldSize - 1) / worldSize);
        for (int point = rank; point < pointCount; point += worldSize) {
            if (!clustered[point]) {
                localSeeds.push_back(point);
            }
        }

        const CandidateBest localBest =
            evaluateLocalSeeds(localSeeds, clustered, thresholdSquared, evaluator);
        CandidateBest globalBest{-1, INT_MAX};
        MPI_Allreduce(&localBest, &globalBest, 1, candidateType, candidateOp,
                      MPI_COMM_WORLD);
        if (globalBest.cardinality <= 0 || globalBest.seed == INT_MAX) {
            throw std::runtime_error("MPI candidate reduction found no remaining seed");
        }

        std::vector<int> members;
        if (rank == 0) {
            members = reconstructSelectedCluster(globalBest.seed, clustered, points,
                                                 thresholdSquared);
            if (static_cast<int>(members.size()) != globalBest.cardinality) {
                throw std::runtime_error("CUDA and OpenMP candidate cardinalities disagree");
            }
        }
        int memberCount = static_cast<int>(members.size());
        MPI_Bcast(&memberCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            members.resize(memberCount);
        }
        MPI_Bcast(members.data(), memberCount, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static)
        for (int index = 0; index < memberCount; ++index) {
            clustered[members[index]] = 1;
        }
        remaining -= memberCount;

        if (rank == 0) {
            clusters.push_back({std::move(members), globalBest.seed});
        }
    }
    return clusters;
}

static bool validateClusters(const std::vector<Cluster>& clusters,
                             const std::vector<Point>& points,
                             double threshold) {
    bool valid = true;
    std::printf("Validating clusters:\n");
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        const Cluster& cluster = clusters[clusterIndex];
        double maxDiameter = 0.0;
        for (size_t first = 0; first < cluster.members.size(); ++first) {
            for (size_t second = first + 1; second < cluster.members.size(); ++second) {
                maxDiameter = std::max(maxDiameter,
                    distance(points[cluster.members[first]], points[cluster.members[second]]));
            }
        }
        if (clusterIndex < 10) {
            std::printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", clusterIndex,
                        cluster.members.size(), cluster.seed_point, maxDiameter);
        }
        if (maxDiameter > threshold * 1.001) {
            std::printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                        clusterIndex, maxDiameter, threshold);
            valid = false;
        }
    }

    std::vector<int> membership(points.size(), -1);
    for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
        for (int member : clusters[clusterIndex].members) {
            if (membership[member] >= 0) {
                std::printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                            member, membership[member], clusterIndex);
                valid = false;
            }
            membership[member] = static_cast<int>(clusterIndex);
        }
    }
    const int clusteredCount = static_cast<int>(std::count_if(
        membership.begin(), membership.end(), [](int value) { return value >= 0; }));
    std::printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
                clusteredCount, points.size() - clusteredCount);
    return valid;
}

static void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of points (default: 1000)\n");
    std::printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

static void selectLocalCudaDevice() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("no CUDA accelerator is available");
    }

    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &nodeCommunicator);
    int nodeRank = 0;
    MPI_Comm_rank(nodeCommunicator, &nodeRank);
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr)); // create the selected device context now
    MPI_Comm_free(&nodeCommunicator);
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int exitCode = 0;
    MPI_Datatype candidateType = MPI_DATATYPE_NULL;
    MPI_Op candidateOp = MPI_OP_NULL;
    try {
        if (providedThreadLevel < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED support");
        }
        omp_set_dynamic(0);
        selectLocalCudaDevice();

        int numPoints = 1000;
        double threshold = 2.0;
        bool validate = false;
        bool printResults = false;
        for (int index = 1; index < argc; ++index) {
            if (std::strcmp(argv[index], "-n") == 0 && index + 1 < argc) {
                numPoints = std::atoi(argv[++index]);
            } else if (std::strcmp(argv[index], "-t") == 0 && index + 1 < argc) {
                threshold = std::atof(argv[++index]);
            } else if (std::strcmp(argv[index], "-v") == 0) {
                validate = true;
            } else if (std::strcmp(argv[index], "-r") == 0) {
                printResults = true;
            } else if (std::strcmp(argv[index], "-h") == 0) {
                if (rank == 0) {
                    printUsage(argv[0]);
                }
                MPI_Finalize();
                return 0;
            } else {
                if (rank == 0) {
                    std::printf("Unknown option: %s\n", argv[index]);
                    printUsage(argv[0]);
                }
                MPI_Finalize();
                return 1;
            }
        }
        if (numPoints <= 0 || threshold <= 0.0) {
            if (rank == 0) {
                std::printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                            numPoints, threshold);
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            std::printf("QT Clustering Benchmark\n");
            std::printf("Number of points: %d\n", numPoints);
            std::printf("Distance threshold: %.2f\n", threshold);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("MPI ranks: %d, OpenMP threads/rank: %d\n", worldSize,
                        omp_get_max_threads());
        }

        std::vector<Point> points(numPoints);
        if (rank == 0) {
            generateSyntheticData(points, numPoints);
        }
        MPI_Bcast(points.data(), static_cast<int>(points.size() * sizeof(Point)), MPI_BYTE,
                  0, MPI_COMM_WORLD);

        MPI_Type_contiguous(2, MPI_INT, &candidateType);
        MPI_Type_commit(&candidateType);
        MPI_Op_create(reduceCandidateBest, 1, &candidateOp);

        MPI_Barrier(MPI_COMM_WORLD);
        const double startTime = MPI_Wtime();
        const std::vector<Cluster> clusters = qtClusteringDistributed(
            points, threshold, rank, worldSize, candidateType, candidateOp);
        const double localElapsed = MPI_Wtime() - startTime;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const long milliseconds = static_cast<long>(elapsed * 1000.0);
            std::printf("Clustering time: %ld ms\n", milliseconds);
            std::printf("Clusters found: %zu\n", clusters.size());
            int totalClustered = 0;
            int maxClusterSize = 0;
            for (const Cluster& cluster : clusters) {
                const int clusterSize = static_cast<int>(cluster.members.size());
                totalClustered += clusterSize;
                maxClusterSize = std::max(maxClusterSize, clusterSize);
            }
            const double averageClusterSize = clusters.empty() ? 0.0 :
                static_cast<double>(totalClustered) / clusters.size();
            std::printf("Points clustered: %d / %d (%.1f%%)\n", totalClustered, numPoints,
                        100.0 * totalClustered / numPoints);
            std::printf("Average cluster size: %.2f\n", averageClusterSize);
            std::printf("Maximum cluster size: %d\n", maxClusterSize);
            const double safeElapsed = std::max(elapsed, std::numeric_limits<double>::min());
            std::printf("Performance: %.1f clusters/s, %.1f points/s\n",
                        clusters.size() / safeElapsed, numPoints / safeElapsed);

            if (printResults) {
                std::vector<double> membershipData;
                membershipData.reserve(numPoints);
                std::vector<int> membership(numPoints, -1);
                for (size_t clusterIndex = 0; clusterIndex < clusters.size(); ++clusterIndex) {
                    for (int member : clusters[clusterIndex].members) {
                        membership[member] = static_cast<int>(clusterIndex);
                    }
                }
                for (int membershipIndex : membership) {
                    membershipData.push_back(static_cast<double>(membershipIndex));
                }
                print_results(membershipData, "ClusterMembership");
            }
            if (validate && !validateClusters(clusters, points, threshold)) {
                exitCode = 1;
            } else if (validate) {
                std::printf("Validation: PASSED\n");
            }
        }

        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Op_free(&candidateOp);
        MPI_Type_free(&candidateType);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        if (candidateOp != MPI_OP_NULL) {
            MPI_Op_free(&candidateOp);
        }
        if (candidateType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&candidateType);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return exitCode;
}
