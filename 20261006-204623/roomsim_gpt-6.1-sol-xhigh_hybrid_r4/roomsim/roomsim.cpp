/**
 * Room Response Simulation Benchmark
 *
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <exception>
#include <climits>
#include <mpi.h>
#if defined(OMPI_MAJOR_VERSION) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
// The radius-10 mesh has delays <= 40. Pack the sample ordinal with the delay
// so preparation does not require a third distributed N-by-N matrix.
constexpr int DELAY_BITS = 8;
constexpr int DELAY_MASK = (1 << DELAY_BITS) - 1;

// ============================================================================
// Vector and Triangle Types
// ============================================================================

#define HD __host__ __device__

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    HD Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                #pragma omp task firstprivate(i, childMin, childMax) shared(childIndices) if(indices.size() > 256)
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
    }

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Hybrid runtime and storage
// ============================================================================

void mpiCheck(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "%s: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

struct MpiRuntime {
    int rank = 0, size = 1, localRank = 0, localSize = 1;
    bool cudaAware = false;
    MpiRuntime(int& argc, char**& argv) {
        int provided = 0;
        MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        MPI_Comm node;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                     MPI_INFO_NULL, &node));
        MPI_CHECK(MPI_Comm_rank(node, &localRank));
        MPI_CHECK(MPI_Comm_size(node, &localSize));
        MPI_CHECK(MPI_Comm_free(&node));
        int deviceCollectives = 0;
        #if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceCollectives = MPIX_Query_cuda_support();
        #endif
        int allDeviceCollectives = 0;
        MPI_CHECK(MPI_Allreduce(&deviceCollectives, &allDeviceCollectives, 1,
                                MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        cudaAware = allDeviceCollectives != 0;
    }
    ~MpiRuntime() { MPI_Finalize(); }
    void selectDevice() const {
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: a CUDA device is required.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        if (!std::getenv("OMP_NUM_THREADS")) {
            omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
        }
    }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t size = 0;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t count) {
        size = count;
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    void release() {
        if (data) CUDA_CHECK(cudaFree(data));
        data = nullptr;
        size = 0;
    }
    void upload(const T* host, size_t count, cudaStream_t stream) {
        if (count) CUDA_CHECK(cudaMemcpyAsync(data, host, count * sizeof(T),
                                            cudaMemcpyHostToDevice, stream));
    }
};

template<class T> struct PinnedBuffer {
    T* data = nullptr;
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    void allocate(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data),
                                  std::max(size_t(1), count) * sizeof(T)));
    }
};

// A portable representation of the original mt19937(42) stream. Checkpoints
// contain untempered state, so advancing the stream needs no sample generation.
// A receiver row always consumes 64 numbers for each non-culled emitter.
struct MtState {
    uint32_t words[624];
    int index;
    void seed(uint32_t value) {
        words[0] = value;
        for (uint32_t i = 1; i < 624; ++i)
            words[i] = 1812433253U * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        index = 624;
    }
    void twist() {
        for (int i = 0; i < 227; ++i) {
            uint32_t bits = (words[i] & 0x80000000U) | (words[i + 1] & 0x7fffffffU);
            words[i] = words[i + 397] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
        }
        for (int i = 227; i < 623; ++i) {
            uint32_t bits = (words[i] & 0x80000000U) | (words[i + 1] & 0x7fffffffU);
            words[i] = words[i - 227] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
        }
        uint32_t bits = (words[623] & 0x80000000U) | (words[0] & 0x7fffffffU);
        words[623] = words[396] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
        index = 0;
    }
    void discard(uint64_t count) {
        while (count) {
            if (index == 624) twist();
            uint64_t step = std::min(count, static_cast<uint64_t>(624 - index));
            index += static_cast<int>(step);
            count -= step;
        }
    }
};

struct FlatNode {
    Vec3 center, halfExtent;
    int escape;                  // First node after this subtree (stackless traversal)
    int firstTriangle, triangleCount;
};

void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                   std::vector<idx_t>& indices) {
    int id = static_cast<int>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<int>(indices.size()), static_cast<int>(tree.triangleIndices.size())});
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<idx_t>(i));
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[id].escape = static_cast<int>(nodes.size());
}

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    size_t rowBegin = 0, localRows = 0, historyLength = 0, batchSize = 1;
    int rank = 0, ranks = 1, minDelay = INT_MAX, maxDelay = 0;
    val_t reflectivity = 0;
    bool validation = false, cudaAware = false;
    uint64_t nonZeroKij = 0, receivedEnergy = 0;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, distances;
    std::vector<int> counts, displacements, activePairs;
    std::vector<int> pairMetadata; // Owned rows: delay and (sample ordinal + 1)
    Octree octree;
    cudaStream_t stream = nullptr;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<val_t> deviceAreas, weights, history, localSignals, sourceSignal;
    DeviceBuffer<val_t> randomSamples, correlations, deviceDistances, gathered;
    DeviceBuffer<int> deviceMetadata, deviceActivePairs;
    DeviceBuffer<MtState> generators;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<idx_t> leafTriangles;
    DeviceBuffer<unsigned long long> factorCount;
    PinnedBuffer<val_t> sendBuffer, receiveBuffer;
    size_t tileRows = 0;
    ~SimulationState() { if (stream) cudaStreamDestroy(stream); }
};

void initializeSimulation(SimulationState& state, const MpiRuntime& runtime,
                          int subdivisions, size_t timesteps, size_t sourceIdx,
                          val_t reflectivity, bool validation) {
    state.rank = runtime.rank;
    state.ranks = runtime.size;
    state.cudaAware = runtime.cudaAware;
    state.numTimesteps = timesteps;
    state.reflectivity = reflectivity;
    state.validation = validation;
    // Limit team size on small meshes while respecting OMP_NUM_THREADS.
    size_t expectedTriangles = size_t(20) << (2 * subdivisions);
    omp_set_num_threads(std::min(omp_get_max_threads(),
                                std::max(1, static_cast<int>((expectedTriangles + 63) / 64))));
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.counts.resize(state.ranks);
    state.displacements.resize(state.ranks);
    for (int r = 0; r < state.ranks; ++r) {
        state.counts[r] = static_cast<int>(state.numTriangles / state.ranks +
                                         (static_cast<size_t>(r) < state.numTriangles % state.ranks));
        state.displacements[r] = r == 0 ? 0 : state.displacements[r - 1] + state.counts[r - 1];
    }
    state.rowBegin = state.displacements[state.rank];
    state.localRows = state.counts[state.rank];
    if (state.rank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("MPI ranks: %d; OpenMP threads per rank: %d\n", state.ranks, omp_get_max_threads());
        printf("Building octree...\n");
    }
    #pragma omp parallel
    {
        #pragma omp single
        state.octree.build(state.triangles);
    }
    std::vector<FlatNode> nodes;
    std::vector<idx_t> leafTriangles;
    flattenOctree(state.octree, nodes, leafTriangles);
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) state.areas[i] = state.triangles[i].area();

    size_t pairs = state.localRows * state.numTriangles;
    state.activePairs.resize(state.localRows);
    state.pairMetadata.resize(pairs);
    CUDA_CHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
    state.deviceTriangles.allocate(state.numTriangles);
    state.deviceAreas.allocate(state.numTriangles);
    state.nodes.allocate(nodes.size());
    state.leafTriangles.allocate(leafTriangles.size());
    state.deviceTriangles.upload(state.triangles.data(), state.numTriangles, state.stream);
    state.deviceAreas.upload(state.areas.data(), state.numTriangles, state.stream);
    state.nodes.upload(nodes.data(), nodes.size(), state.stream);
    state.leafTriangles.upload(leafTriangles.data(), leafTriangles.size(), state.stream);
    state.weights.allocate(pairs);
    state.deviceMetadata.allocate(pairs);
    state.deviceActivePairs.allocate(state.localRows);
    state.generators.allocate(state.localRows);
    state.factorCount.allocate(1);
    CUDA_CHECK(cudaMemsetAsync(state.factorCount.data, 0, sizeof(unsigned long long), state.stream));
    // Bound temporary random samples independently of the full matrix size.
    state.tileRows = std::min(state.localRows, std::max(size_t(1),
                              (64 * 1024 * 1024) / (state.numTriangles * NUM_RAYS * 4 * sizeof(val_t))));
    state.randomSamples.allocate(state.tileRows * state.numTriangles * NUM_RAYS * 4);
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
}

// ============================================================================
// CUDA visibility, sampling and form factors
// ============================================================================

__device__ bool intersectsBox(const FlatNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool rayBlocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                           const FlatNode* nodes, const idx_t* indices, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    int node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& box = nodes[node];
        // The original traversal does not test the root's bounding box.
        if (node != 0 && !intersectsBox(box, from, to)) {
            node = box.escape;
            continue;
        }
        for (int k = 0; k < box.triangleCount; ++k) {
            idx_t idx = indices[box.firstTriangle + k];
            if (idx == src || idx == dst) continue;
            const Triangle& tri = triangles[idx];
            Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
            Vec3 pvec = dir.cross(e2);
            val_t det = e1.dot(pvec);
            if (fabsf(det) < EPSILON) continue;
            val_t invDet = ONE / det;
            Vec3 tvec = from - tri.a;
            val_t u = tvec.dot(pvec) * invDet;
            if (u < ZERO || u > ONE) continue;
            Vec3 qvec = tvec.cross(e1);
            val_t v = dir.dot(qvec) * invDet;
            if (v < ZERO || u + v > ONE) continue;
            val_t dist = e2.dot(qvec) * invDet;
            if (dist > EPSILON && dist < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__device__ val_t uniformSample(uint32_t bits) {
    bits ^= bits >> 11;
    bits ^= (bits << 7) & 0x9d2c5680U;
    bits ^= (bits << 15) & 0xefc60000U;
    bits ^= bits >> 18;
    // std::uniform_real_distribution<float> uses one MT word and clamps 1.0.
    return fminf(static_cast<val_t>(bits) * 0x1p-32f, 0x1.fffffep-1f);
}

__global__ void generateSamples(const MtState* checkpoints, const int* activePairs,
                                val_t* samples, size_t firstRow, size_t n) {
    size_t row = firstRow + blockIdx.x;
    __shared__ uint32_t oldState[624], newState[624];
    for (int k = threadIdx.x; k < 624; k += blockDim.x) oldState[k] = checkpoints[row].words[k];
    int index = checkpoints[row].index;
    size_t remaining = static_cast<size_t>(activePairs[row]) * NUM_RAYS * 4;
    size_t output = 0;
    val_t* dest = samples + static_cast<size_t>(blockIdx.x) * n * NUM_RAYS * 4;
    __syncthreads();
    while (remaining) {
        if (index == 624) {
            // The in-place MT recurrence has three dependency bands.
            for (int k = threadIdx.x; k < 227; k += blockDim.x) {
                uint32_t bits = (oldState[k] & 0x80000000U) | (oldState[k + 1] & 0x7fffffffU);
                newState[k] = oldState[k + 397] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
            }
            __syncthreads();
            for (int k = 227 + threadIdx.x; k < 454; k += blockDim.x) {
                uint32_t bits = (oldState[k] & 0x80000000U) | (oldState[k + 1] & 0x7fffffffU);
                newState[k] = newState[k - 227] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
            }
            __syncthreads();
            for (int k = 454 + threadIdx.x; k < 623; k += blockDim.x) {
                uint32_t bits = (oldState[k] & 0x80000000U) | (oldState[k + 1] & 0x7fffffffU);
                newState[k] = newState[k - 227] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
            }
            if (threadIdx.x == 0) {
                uint32_t bits = (oldState[623] & 0x80000000U) | (newState[0] & 0x7fffffffU);
                newState[623] = newState[396] ^ (bits >> 1) ^ ((bits & 1U) ? 0x9908b0dfU : 0U);
            }
            __syncthreads();
            for (int k = threadIdx.x; k < 624; k += blockDim.x) oldState[k] = newState[k];
            __syncthreads();
            index = 0;
        }
        int take = static_cast<int>(remaining < static_cast<size_t>(624 - index) ? remaining : 624 - index);
        for (int k = threadIdx.x; k < take; k += blockDim.x)
            dest[output + k] = uniformSample(oldState[index + k]);
        __syncthreads();
        index += take;
        output += take;
        remaining -= take;
    }
}

__global__ void formFactors(const Triangle* triangles, const FlatNode* nodes,
                            const idx_t* indices, const val_t* areas, const int* metadata,
                            const val_t* samples, val_t* weights, size_t rowBegin,
                            size_t firstRow, size_t tileRows, size_t n,
                            unsigned long long* nonZeroCount) {
    // Sixteen adjacent lanes evaluate the sixteen independent visibility rays.
    // Each subgroup's first lane sums in sample order, as in the serial version.
    size_t pair = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / NUM_RAYS;
    unsigned int ray = threadIdx.x % NUM_RAYS;
    val_t contribution = ZERO, kij = ZERO;
    if (pair < tileRows * n) {
        size_t tileRow = pair / n, j = pair % n;
        size_t row = firstRow + tileRow, i = rowBegin + row;
        int offset = (metadata[row * n + j] >> DELAY_BITS) - 1;
        if (offset >= 0) {
            const Triangle& triI = triangles[i];
            const Triangle& triJ = triangles[j];
            const val_t* values = samples + (tileRow * n + static_cast<size_t>(offset)) * NUM_RAYS * 4 + ray * 4;
            Vec3 pI = samplePoint(triI, values[0], values[1]);
            Vec3 pJ = samplePoint(triJ, values[2], values[3]);
            if (!rayBlocked(pI, pJ, triangles, nodes, indices, i, j)) {
                Vec3 v = pJ - pI;
                val_t distSqr = v.squaredNorm();
                if (distSqr >= EPSILON) {
                    val_t norm = v.norm();
                    val_t cosI = norm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI.normal()) / norm);
                    val_t cosJ = norm <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(triJ.normal()) / norm);
                    if (cosI > ZERO && cosJ > ZERO)
                        contribution = (cosI * cosJ) / (PI * distSqr);
                }
            }
        }
    }
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t value = __shfl_sync(0xffffffffU, contribution, r, NUM_RAYS);
        if (ray == 0) kij += value;
    }
    kij *= INV_NUM_RAYS;
    if (ray == 0 && pair < tileRows * n) {
        size_t row = firstRow + pair / n, j = pair % n;
        weights[row * n + j] = fminf(kij * areas[j], ONE);
    }
    // Count the original Kij values, before area scaling, for validation.
    __shared__ unsigned int counts[256];
    counts[threadIdx.x] = ray == 0 && kij > EPSILON;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(nonZeroCount, static_cast<unsigned long long>(counts[0]));
}

void computeTimeDelays(SimulationState& state) {
    if (state.rank == 0) printf("Computing time delays (Tau)...\n");
    std::vector<Vec3> centers(state.numTriangles), normals(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        centers[i] = state.triangles[i].center();
        normals[i] = state.triangles[i].normal();
    }
    int minimum = INT_MAX, maximum = 0;
    #pragma omp parallel for schedule(static) reduction(min:minimum) reduction(max:maximum)
    for (size_t row = 0; row < state.localRows; ++row) {
        size_t i = state.rowBegin + row;
        int active = 0;
        for (size_t j = 0; j < state.numTriangles; ++j) {
            int delay = 0, ordinal = 0;
            if (i != j) {
                val_t dist = (centers[i] - centers[j]).norm();
                delay = static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
                minimum = std::min(minimum, delay);
                maximum = std::max(maximum, delay);
                if (normals[i].dot(normals[j]) <= 0.99f)
                    ordinal = ++active;
            }
            state.pairMetadata[row * state.numTriangles + j] = (ordinal << DELAY_BITS) | delay;
        }
        state.activePairs[row] = active;
    }
    MPI_CHECK(MPI_Allreduce(&minimum, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&maximum, &state.maxDelay, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    // Distinct icosphere faces have strictly positive delays. Batching never
    // crosses a dependency: t - tau is earlier than the first step of a batch.
    if (state.minDelay <= 0) {
        fprintf(stderr, "Zero-delay coupling between distinct faces is unsupported.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    state.batchSize = std::max(size_t(1), std::min(state.numTimesteps, static_cast<size_t>(state.minDelay)));
    state.historyLength = std::min(state.numTimesteps, static_cast<size_t>(state.maxDelay) + state.batchSize);
    state.deviceMetadata.upload(state.pairMetadata.data(), state.pairMetadata.size(), state.stream);
    state.deviceActivePairs.upload(state.activePairs.data(), state.activePairs.size(), state.stream);
    state.history.allocate(state.historyLength * state.numTriangles);
    state.localSignals.allocate(state.numTimesteps * state.localRows);
    state.sourceSignal.allocate(state.numTimesteps);
    if (state.ranks > 1) {
        state.gathered.allocate(state.batchSize * state.numTriangles);
        if (!state.cudaAware) {
            state.sendBuffer.allocate(state.batchSize * state.localRows);
            state.receiveBuffer.allocate(state.batchSize * state.numTriangles);
        }
    }
    if (state.history.size) CUDA_CHECK(cudaMemsetAsync(state.history.data, 0,
                              state.historyLength * state.numTriangles * sizeof(val_t), state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    // Host metadata is no longer needed after the upload.
    std::vector<int>().swap(state.pairMetadata);
}

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij) on CUDA devices...\n");
    std::vector<int> allActive;
    std::vector<MtState> allCheckpoints;
    if (state.rank == 0) {
        allActive.resize(state.numTriangles);
        allCheckpoints.resize(state.numTriangles);
    }
    MPI_CHECK(MPI_Gatherv(state.activePairs.data(), static_cast<int>(state.localRows), MPI_INT,
                         allActive.data(), state.counts.data(), state.displacements.data(),
                         MPI_INT, 0, MPI_COMM_WORLD));
    if (state.rank == 0) {
        MtState rng;
        rng.seed(42);
        for (size_t i = 0; i < state.numTriangles; ++i) {
            allCheckpoints[i] = rng;
            rng.discard(static_cast<uint64_t>(allActive[i]) * NUM_RAYS * 4);
        }
    }
    std::vector<MtState> checkpoints(state.localRows);
    MPI_Datatype checkpointType;
    MPI_CHECK(MPI_Type_contiguous(sizeof(MtState), MPI_BYTE, &checkpointType));
    MPI_CHECK(MPI_Type_commit(&checkpointType));
    MPI_CHECK(MPI_Scatterv(allCheckpoints.data(), state.counts.data(), state.displacements.data(),
                          checkpointType, checkpoints.data(), static_cast<int>(state.localRows),
                          checkpointType, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&checkpointType));
    state.generators.upload(checkpoints.data(), checkpoints.size(), state.stream);
    for (size_t first = 0; first < state.localRows; first += state.tileRows) {
        size_t rows = std::min(state.tileRows, state.localRows - first);
        generateSamples<<<static_cast<unsigned int>(rows), 256, 0, state.stream>>>(
            state.generators.data, state.deviceActivePairs.data, state.randomSamples.data,
            first, state.numTriangles);
        CUDA_CHECK(cudaGetLastError());
        formFactors<<<static_cast<unsigned int>((rows * state.numTriangles * NUM_RAYS + 255) / 256), 256, 0, state.stream>>>(
            state.deviceTriangles.data, state.nodes.data, state.leafTriangles.data,
            state.deviceAreas.data, state.deviceMetadata.data, state.randomSamples.data,
            state.weights.data, state.rowBegin, first, rows,
            state.numTriangles, state.factorCount.data);
        CUDA_CHECK(cudaGetLastError());
    }
    uint64_t localCount = 0;
    CUDA_CHECK(cudaMemcpyAsync(&localCount, state.factorCount.data, sizeof(localCount),
                              cudaMemcpyDeviceToHost, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    MPI_CHECK(MPI_Allreduce(&localCount, &state.nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    state.randomSamples.release();
    state.generators.release();
    state.deviceActivePairs.release();
    state.deviceTriangles.release();
    state.deviceAreas.release();
    state.nodes.release();
    state.leafTriangles.release();
    state.factorCount.release();
    state.octree = Octree();
    std::vector<Triangle>().swap(state.triangles);
    std::vector<val_t>().swap(state.areas);
    std::vector<int>().swap(state.activePairs);
}

// ============================================================================
// Distributed, delay-batched wave propagation
// ============================================================================

__global__ void propagate(const val_t* weights, const int* metadata, const val_t* history,
                          val_t* signals, size_t rowBegin, size_t localRows, size_t n,
                          size_t firstTime, size_t steps, size_t historyLength,
                          size_t totalSteps, size_t source, val_t reflectivity) {
    // A warp loads contributions cooperatively, then adds them in the original
    // emitter order. This preserves ordered float sums and avoids atomics.
    size_t work = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    unsigned int lane = threadIdx.x & 31;
    if (work >= localRows * steps) return;
    size_t row = work % localRows, t = firstTime + work / localRows;
    val_t sum = ZERO;
    for (size_t firstJ = 0; firstJ < n; firstJ += 32) {
        size_t j = firstJ + lane;
        val_t contribution = ZERO;
        if (j < n && j != rowBegin + row) {
            int delay = metadata[row * n + j] & DELAY_MASK;
            val_t weight = weights[row * n + j];
            if (t >= static_cast<size_t>(delay) && weight > ZERO) {
                val_t value = history[((t - delay) % historyLength) * n + j];
                if (!(value <= ZERO)) contribution = weight * value;
            }
        }
        for (int k = 0; k < 32 && firstJ + k < n; ++k) {
            val_t value = __shfl_sync(0xffffffffU, contribution, k);
            if (lane == 0) sum += value;
        }
    }
    if (lane == 0) {
        val_t emission = rowBegin + row == source && t < totalSteps / 2 ? ONE : ZERO;
        signals[t * localRows + row] = reflectivity * sum + emission;
    }
}

__global__ void installHistory(const val_t* packed, val_t* history, val_t* sourceSignal,
                               size_t n, int ranks, size_t firstTime, size_t steps,
                               size_t historyLength, size_t source) {
    size_t work = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (work >= n * steps) return;
    size_t j = work % n, k = work / n;
    size_t small = n / ranks, extra = n % ranks, cutoff = extra * (small + 1);
    size_t owner, offset, rows, begin;
    if (j < cutoff) {
        owner = j / (small + 1);
        rows = small + 1;
        begin = owner * rows;
        offset = j - begin;
    } else {
        owner = extra + (j - cutoff) / small;
        rows = small;
        begin = cutoff + (owner - extra) * rows;
        offset = j - begin;
    }
    val_t value = packed[begin * steps + k * rows + offset];
    history[((firstTime + k) % historyLength) * n + j] = value;
    if (j == source) sourceSignal[firstTime + k] = value;
}

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    std::vector<int> counts(state.ranks), displacements(state.ranks);
    for (size_t t = 0; t < state.numTimesteps; t += state.batchSize) {
        size_t steps = std::min(state.batchSize, state.numTimesteps - t);
        if (state.localRows) {
            propagate<<<static_cast<unsigned int>((state.localRows * steps * 32 + 127) / 128), 128, 0, state.stream>>>(
                state.weights.data, state.deviceMetadata.data, state.history.data,
                state.localSignals.data, state.rowBegin, state.localRows, state.numTriangles,
                t, steps, state.historyLength, state.numTimesteps, state.sourceIndex, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
            if (state.ranks > 1 && !state.cudaAware)
                CUDA_CHECK(cudaMemcpyAsync(state.sendBuffer.data, state.localSignals.data + t * state.localRows,
                                          steps * state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost, state.stream));
        }
        const val_t* packed;
        if (state.ranks == 1) {
            // The CUDA stream orders propagation and installation without a
            // host round trip when the whole mesh belongs to this device.
            packed = state.localSignals.data + t * state.localRows;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(state.stream));
            for (int r = 0; r < state.ranks; ++r) {
                counts[r] = static_cast<int>(steps * state.counts[r]);
                displacements[r] = static_cast<int>(steps * state.displacements[r]);
            }
            if (state.cudaAware) {
                const val_t* send = state.localRows ? state.localSignals.data + t * state.localRows : nullptr;
                MPI_CHECK(MPI_Allgatherv(send, counts[state.rank], MPI_FLOAT,
                                        state.gathered.data, counts.data(), displacements.data(),
                                        MPI_FLOAT, MPI_COMM_WORLD));
            } else {
                // Pinned staging also supports MPI builds without device-buffer
                // collectives. Both transports exchange only new time slices.
                MPI_CHECK(MPI_Allgatherv(state.sendBuffer.data, counts[state.rank], MPI_FLOAT,
                                        state.receiveBuffer.data, counts.data(), displacements.data(),
                                        MPI_FLOAT, MPI_COMM_WORLD));
                state.gathered.upload(state.receiveBuffer.data, steps * state.numTriangles, state.stream);
            }
            packed = state.gathered.data;
        }
        installHistory<<<static_cast<unsigned int>((steps * state.numTriangles + 255) / 256), 256, 0, state.stream>>>(
            packed, state.history.data, state.sourceSignal.data,
            state.numTriangles, state.ranks, t, steps, state.historyLength, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
        if (state.rank == 0 && ((t / 10) != ((t + steps) / 10) || t + steps == state.numTimesteps))
            printf("  Timestep %zu/%zu\n", t + steps, state.numTimesteps);
    }
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
}

// ============================================================================
// Distributed CUDA cross-correlation, with the original first-maximum tie rule
// ============================================================================

__global__ void crossCorrelations(const val_t* signals, const val_t* source,
                                 val_t* correlations, size_t rows, size_t timesteps) {
    size_t work = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (work >= rows * timesteps) return;
    size_t row = work % rows, lag = work / rows;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < timesteps; ++tt)
        sum += source[tt - lag] * signals[tt * rows + row];
    correlations[work] = sum;
}

__global__ void selectDistances(const val_t* correlations, val_t* distances,
                               size_t rows, size_t timesteps) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    val_t maximum = ZERO;
    size_t best = 0;
    for (size_t lag = 0; lag < timesteps; ++lag) {
        val_t value = correlations[lag * rows + row];
        if (value > maximum) { maximum = value; best = lag; }
    }
    distances[row] = WAVE_SPEED * static_cast<val_t>(best);
}

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    state.correlations.allocate(state.localRows * state.numTimesteps);
    state.deviceDistances.allocate(state.localRows);
    std::vector<val_t> localDistances(state.localRows);
    if (state.localRows) {
        if (state.numTimesteps) {
            crossCorrelations<<<static_cast<unsigned int>((state.localRows * state.numTimesteps + 255) / 256), 256, 0, state.stream>>>(
                state.localSignals.data, state.sourceSignal.data, state.correlations.data,
                state.localRows, state.numTimesteps);
            CUDA_CHECK(cudaGetLastError());
        }
        selectDistances<<<static_cast<unsigned int>((state.localRows + 255) / 256), 256, 0, state.stream>>>(
            state.correlations.data, state.deviceDistances.data, state.localRows, state.numTimesteps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(localDistances.data(), state.deviceDistances.data,
                                  state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost, state.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    if (state.rank == 0) state.distances.resize(state.numTriangles);
    MPI_CHECK(MPI_Gatherv(localDistances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                         state.distances.data(), state.counts.data(), state.displacements.data(),
                         MPI_FLOAT, 0, MPI_COMM_WORLD));
    if (state.validation) {
        std::vector<val_t> signals(state.numTimesteps * state.localRows);
        if (!signals.empty()) CUDA_CHECK(cudaMemcpy(signals.data(), state.localSignals.data,
                                                   signals.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
        uint64_t received = 0;
        #pragma omp parallel for schedule(static) reduction(+:received)
        for (size_t row = 0; row < state.localRows; ++row) {
            for (size_t t = 0; t < state.numTimesteps; ++t) {
                if (signals[t * state.localRows + row] > EPSILON) { ++received; break; }
            }
        }
        MPI_CHECK(MPI_Allreduce(&received, &state.receivedEnergy, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    uint64_t receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %llu/%zu\n",
           static_cast<unsigned long long>(receivedEnergy), state.numTriangles);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    uint64_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv, const MpiRuntime& runtime) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (runtime.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (runtime.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (runtime.rank == 0) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    runtime.selectDevice();

    if (runtime.rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }
    // Initialize
    SimulationState state;
    initializeSimulation(state, runtime, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, validate);

    if (runtime.rank == 0) printf("\n");

    // Precomputation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startPre = MPI_Wtime();

    computeTimeDelays(state);
    computeFormFactors(state);

    double preElapsed = MPI_Wtime() - startPre, preMaximum = 0;
    MPI_CHECK(MPI_Allreduce(&preElapsed, &preMaximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long preDuration = static_cast<long>(preMaximum * 1000);
    if (runtime.rank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    // Simulation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startSim = MPI_Wtime();

    runSimulation(state);

    double simElapsed = MPI_Wtime() - startSim, simMaximum = 0;
    MPI_CHECK(MPI_Allreduce(&simElapsed, &simMaximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long simDuration = static_cast<long>(simMaximum * 1000);
    if (runtime.rank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    // Distance computation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startDist = MPI_Wtime();

    computeDistances(state);

    double distElapsed = MPI_Wtime() - startDist, distMaximum = 0;
    MPI_CHECK(MPI_Allreduce(&distElapsed, &distMaximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long distDuration = static_cast<long>(distMaximum * 1000);
    int status = 0;
    if (runtime.rank == 0) {
        printf("Distance computation time: %ld ms\n\n", distDuration);

        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * t);
        double distOps = static_cast<double>(n * t * t);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = state.localRows * n * sizeof(val_t);
        size_t memTau = state.localRows * n * sizeof(int);
        size_t memRad = (2 * t * state.localRows + state.historyLength * n + t) * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Matrix and signal memory per rank (rank 0): %.2f MB\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        if (validate && !validateResults(state)) status = 1;
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}

int main(int argc, char** argv) {
    MpiRuntime runtime(argc, argv);
    try {
        return runBenchmark(argc, argv, runtime);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", runtime.rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
