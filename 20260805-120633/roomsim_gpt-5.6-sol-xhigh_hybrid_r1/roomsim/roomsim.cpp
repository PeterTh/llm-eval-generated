/**
 * Room Response Simulation Benchmark
 * 
 * This hybrid MPI/OpenMP/CUDA implementation simulates room impulse response
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
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
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

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

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
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
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Hybrid MPI/OpenMP/CUDA implementation
// ============================================================================

void mpiCheck(int result, const char* expression, int line) {
    if (result == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(result, message, &length);
    fprintf(stderr, "MPI error at line %d (%s): %.*s\n", line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, result);
}

void cudaCheck(cudaError_t result, const char* expression, int line) {
    if (result == cudaSuccess) return;
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d CUDA error at line %d (%s): %s\n",
            rank, line, expression, cudaGetErrorString(result));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));
}

#define MPI_CHECK(call) mpiCheck((call), #call, __LINE__)
#define CUDA_CHECK(call) cudaCheck((call), #call, __LINE__)

struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    int triangleOffset;
    int triangleCount;
};

int flattenOctree(const Octree& source, std::vector<FlatOctreeNode>& nodes,
                  std::vector<int>& triangleIndices) {
    const int nodeIndex = static_cast<int>(nodes.size());
    FlatOctreeNode flat{};
    flat.center = source.center;
    flat.halfExtent = source.halfExtent;
    std::fill(std::begin(flat.children), std::end(flat.children), -1);
    flat.triangleOffset = static_cast<int>(triangleIndices.size());
    flat.triangleCount = static_cast<int>(source.triangleIndices.size());
    for (size_t index : source.triangleIndices) {
        triangleIndices.push_back(static_cast<int>(index));
    }
    nodes.push_back(flat);

    for (int child = 0; child < 8; ++child) {
        if (source.children[child]) {
            nodes[nodeIndex].children[child] =
                flattenOctree(*source.children[child], nodes, triangleIndices);
        }
    }
    return nodeIndex;
}

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    val_t reflectivity = ZERO;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> distances;
    Octree octree;
    std::vector<FlatOctreeNode> flatNodes;
    std::vector<int> flatTriangleIndices;
};

struct Partition {
    int rank = 0;
    int ranks = 1;
    int localRank = 0;
    int localRanks = 1;
    int device = 0;
    int deviceCount = 0;
    bool cudaAwareMPI = false;
    size_t rowBegin = 0;
    size_t localRows = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
};

struct DeviceState {
    Triangle* triangles = nullptr;
    FlatOctreeNode* nodes = nullptr;
    int* triangleIndices = nullptr;
    val_t* areas = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;
    val_t* globalRow = nullptr;  // Pinned staging buffer when MPI is not CUDA-aware.
};

Partition initializePartition(size_t numTriangles) {
    Partition partition;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &partition.rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &partition.ranks));

    MPI_Comm localCommunicator;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                  partition.rank, MPI_INFO_NULL, &localCommunicator));
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &partition.localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &partition.localRanks));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    CUDA_CHECK(cudaGetDeviceCount(&partition.deviceCount));
    if (partition.deviceCount == 0) {
        if (partition.rank == 0) fprintf(stderr, "roomsim requires at least one CUDA device per node.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    partition.device = partition.localRank % partition.deviceCount;
    CUDA_CHECK(cudaSetDevice(partition.device));
    CUDA_CHECK(cudaFree(nullptr));  // Initialize the rank's CUDA context.

#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    partition.cudaAwareMPI = MPIX_Query_cuda_support() != 0;
#endif
    // MPI stacks without a query extension can be enabled explicitly when the
    // cluster administrator guarantees CUDA-buffer support.
    const char* forceCudaAware = getenv("ROOMSIM_CUDA_AWARE_MPI");
    if (forceCudaAware != nullptr) {
        partition.cudaAwareMPI = strcmp(forceCudaAware, "0") != 0;
    }

    partition.rowCounts.resize(partition.ranks);
    partition.rowDisplacements.resize(partition.ranks);
    const size_t base = numTriangles / static_cast<size_t>(partition.ranks);
    const size_t remainder = numTriangles % static_cast<size_t>(partition.ranks);
    size_t displacement = 0;
    for (int rank = 0; rank < partition.ranks; ++rank) {
        const size_t count = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        partition.rowCounts[rank] = static_cast<int>(count);
        partition.rowDisplacements[rank] = static_cast<int>(displacement);
        displacement += count;
    }
    partition.rowBegin = static_cast<size_t>(partition.rowDisplacements[partition.rank]);
    partition.localRows = static_cast<size_t>(partition.rowCounts[partition.rank]);
    return partition;
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool printProgress) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    if (printProgress) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }
    state.octree.build(state.triangles);
    flattenOctree(state.octree, state.flatNodes, state.flatTriangleIndices);

    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();
    }
}

__device__ __forceinline__ Vec3 deviceAdd(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ Vec3 deviceSub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ Vec3 deviceMul(const Vec3& a, val_t scale) {
    return {a.x * scale, a.y * scale, a.z * scale};
}

__device__ __forceinline__ val_t deviceDot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 deviceCross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ val_t deviceSquaredNorm(const Vec3& a) {
    return deviceDot(a, a);
}

__device__ __forceinline__ uint64_t mixBits(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ __forceinline__ val_t counterRandom(uint64_t pair, int ray, int component) {
    const uint64_t counter = pair * static_cast<uint64_t>(NUM_RAYS * 4) +
                             static_cast<uint64_t>(ray * 4 + component);
    const uint32_t mantissa = static_cast<uint32_t>(mixBits(counter ^ 42ULL) >> 40);
    return static_cast<val_t>(mantissa) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ Vec3 deviceRandomPoint(const Triangle& triangle,
                                                   uint64_t pair, int ray, int component) {
    val_t u = counterRandom(pair, ray, component);
    val_t v = counterRandom(pair, ray, component + 1);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return deviceAdd(triangle.a,
                     deviceAdd(deviceMul(deviceSub(triangle.b, triangle.a), u),
                               deviceMul(deviceSub(triangle.c, triangle.a), v)));
}

__device__ __forceinline__ bool deviceRayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                                        const FlatOctreeNode& node) {
    const Vec3 d = deviceMul(deviceSub(p2, p1), 0.5f);
    const Vec3 c = deviceSub(deviceAdd(p1, d), node.center);
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ val_t deviceRayTriangleIntersect(const Vec3& origin,
                                                             const Vec3& direction,
                                                             const Triangle& triangle) {
    const Vec3 e1 = deviceSub(triangle.b, triangle.a);
    const Vec3 e2 = deviceSub(triangle.c, triangle.a);
    const Vec3 pvec = deviceCross(direction, e2);
    const val_t determinant = deviceDot(e1, pvec);
    if (fabsf(determinant) < EPSILON) return 3.402823466e+38F;
    const val_t inverseDeterminant = ONE / determinant;
    const Vec3 tvec = deviceSub(origin, triangle.a);
    const val_t u = deviceDot(tvec, pvec) * inverseDeterminant;
    if (u < ZERO || u > ONE) return 3.402823466e+38F;
    const Vec3 qvec = deviceCross(tvec, e1);
    const val_t v = deviceDot(direction, qvec) * inverseDeterminant;
    if (v < ZERO || u + v > ONE) return 3.402823466e+38F;
    return deviceDot(e2, qvec) * inverseDeterminant;
}

__device__ bool deviceRayBlocked(const Vec3& from, const Vec3& to,
                                 const Triangle* __restrict__ triangles,
                                 const FlatOctreeNode* __restrict__ nodes,
                                 const int* __restrict__ triangleIndices,
                                 size_t sourceIndex, size_t destinationIndex) {
    const Vec3 direction = deviceSub(to, from);
    const val_t rayLengthSquared = deviceSquaredNorm(direction);
    if (rayLengthSquared < EPSILON * EPSILON) return true;
    const val_t rayLength = sqrtf(rayLengthSquared);
    const Vec3 normalizedDirection = deviceMul(direction, ONE / rayLength);

    // The construction limits depth to roughly log2(room diameter / leaf size).
    // A depth-first traversal needs at most seven pending siblings per level.
    int stack[64];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize != 0) {
        const FlatOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount != 0) {
            for (int local = 0; local < node.triangleCount; ++local) {
                const int triangleIndex = triangleIndices[node.triangleOffset + local];
                if (static_cast<size_t>(triangleIndex) == sourceIndex ||
                    static_cast<size_t>(triangleIndex) == destinationIndex) continue;
                const val_t distance = deviceRayTriangleIntersect(
                    from, normalizedDirection, triangles[triangleIndex]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
        } else {
            for (int child = 7; child >= 0; --child) {
                const int childIndex = node.children[child];
                if (childIndex >= 0 && deviceRayIntersectsBox(from, to, nodes[childIndex])) {
                    if (stackSize < 64) stack[stackSize++] = childIndex;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ int deviceTau(const Triangle& a, const Triangle& b) {
    const Vec3 centerA = deviceMul(deviceAdd(deviceAdd(a.a, a.b), a.c), 1.0f / 3.0f);
    const Vec3 centerB = deviceMul(deviceAdd(deviceAdd(b.a, b.b), b.c), 1.0f / 3.0f);
    return static_cast<int>(ceilf(sqrtf(deviceSquaredNorm(deviceSub(centerA, centerB))) *
                                  INV_WAVE_SPEED));
}

__global__ void precomputeKernel(const Triangle* __restrict__ triangles,
                                 const FlatOctreeNode* __restrict__ nodes,
                                 const int* __restrict__ triangleIndices,
                                 val_t* __restrict__ kij, int* __restrict__ tau,
                                 size_t numTriangles, size_t rowBegin,
                                 size_t localRows) {
    const size_t total = localRows * numTriangles;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t localPair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         localPair < total; localPair += stride) {
        const size_t localRow = localPair / numTriangles;
        const size_t j = localPair - localRow * numTriangles;
        const size_t i = rowBegin + localRow;
        if (i == j) {
            tau[localPair] = 0;
            kij[localPair] = ZERO;
            continue;
        }

        const Triangle& triangleI = triangles[i];
        const Triangle& triangleJ = triangles[j];
        tau[localPair] = deviceTau(triangleI, triangleJ);
        if (deviceDot(triangleI._normal, triangleJ._normal) > 0.99f) {
            kij[localPair] = ZERO;
            continue;
        }

        const uint64_t globalPair = static_cast<uint64_t>(i) * numTriangles + j;
        val_t value = ZERO;
        for (int ray = 0; ray < NUM_RAYS; ++ray) {
            const Vec3 pointI = deviceRandomPoint(triangleI, globalPair, ray, 0);
            const Vec3 pointJ = deviceRandomPoint(triangleJ, globalPair, ray, 2);
            if (deviceRayBlocked(pointI, pointJ, triangles, nodes, triangleIndices, i, j)) continue;
            const Vec3 direction = deviceSub(pointJ, pointI);
            const val_t distanceSquared = deviceSquaredNorm(direction);
            if (distanceSquared < EPSILON) continue;
            const val_t inverseDistance = rsqrtf(distanceSquared);
            const val_t cosineI = fmaxf(ZERO, deviceDot(direction, triangleI._normal) * inverseDistance);
            const val_t cosineJ = fmaxf(ZERO, -deviceDot(direction, triangleJ._normal) * inverseDistance);
            if (cosineI > ZERO && cosineJ > ZERO) {
                value += (cosineI * cosineJ) / (PI * distanceSquared);
            }
        }
        kij[localPair] = value * INV_NUM_RAYS;
    }
}

__global__ void simulationTimestepKernel(const val_t* __restrict__ kij,
                                         const int* __restrict__ tau,
                                         const val_t* __restrict__ areas,
                                         val_t* __restrict__ radB,
                                         size_t numTriangles, size_t timestep,
                                         size_t numTimesteps, size_t sourceIndex,
                                         size_t rowBegin, size_t localRows,
                                         val_t reflectivity) {
    __shared__ val_t partialSums[256];
    for (size_t localRow = blockIdx.x; localRow < localRows; localRow += gridDim.x) {
        const size_t i = rowBegin + localRow;
        const size_t matrixRow = localRow * numTriangles;
        val_t sum = ZERO;
        for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
            if (i == j) continue;
            const int delay = tau[matrixRow + j];
            if (static_cast<int>(timestep) < delay) continue;
            const val_t formFactor = kij[matrixRow + j];
            if (formFactor <= ZERO) continue;
            const size_t sourceTime = timestep - static_cast<size_t>(delay);
            const val_t sourceRadiosity = radB[sourceTime * numTriangles + j];
            if (sourceRadiosity <= ZERO) continue;
            sum += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
        }
        partialSums[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2) {
            if (threadIdx.x < offset) partialSums[threadIdx.x] += partialSums[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const val_t emission =
                (i == sourceIndex && timestep < numTimesteps / 2) ? ONE : ZERO;
            radB[timestep * numTriangles + i] = reflectivity * partialSums[0] + emission;
        }
        __syncthreads();
    }
}

__global__ void distanceKernel(const val_t* __restrict__ radB,
                               val_t* __restrict__ distances,
                               size_t numTriangles, size_t numTimesteps,
                               size_t sourceIndex, size_t rowBegin,
                               size_t localRows) {
    __shared__ val_t maximums[256];
    __shared__ size_t bestTimes[256];
    for (size_t localRow = blockIdx.x; localRow < localRows; localRow += gridDim.x) {
        const size_t i = rowBegin + localRow;
        val_t maximumCorrelation = ZERO;
        size_t bestTime = 0;
        for (size_t delay = threadIdx.x; delay < numTimesteps; delay += blockDim.x) {
            val_t sum = ZERO;
            for (size_t t = delay; t < numTimesteps; ++t) {
                sum += radB[(t - delay) * numTriangles + sourceIndex] *
                       radB[t * numTriangles + i];
            }
            if (sum > maximumCorrelation) {
                maximumCorrelation = sum;
                bestTime = delay;
            }
        }
        maximums[threadIdx.x] = maximumCorrelation;
        bestTimes[threadIdx.x] = bestTime;
        __syncthreads();
        for (unsigned int offset = blockDim.x / 2; offset != 0; offset /= 2) {
            if (threadIdx.x < offset) {
                const val_t otherMaximum = maximums[threadIdx.x + offset];
                const size_t otherTime = bestTimes[threadIdx.x + offset];
                if (otherMaximum > maximums[threadIdx.x] ||
                    (otherMaximum == maximums[threadIdx.x] &&
                     otherTime < bestTimes[threadIdx.x])) {
                    maximums[threadIdx.x] = otherMaximum;
                    bestTimes[threadIdx.x] = otherTime;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            distances[localRow] = WAVE_SPEED * static_cast<val_t>(bestTimes[0]);
        }
        __syncthreads();
    }
}

__global__ void countFormFactorsKernel(const val_t* values, size_t count,
                                       unsigned long long* result) {
    unsigned long long local = 0;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        local += values[i] > EPSILON;
    }
    if (local != 0) atomicAdd(result, local);
}

__global__ void countReceivedKernel(const val_t* radB, size_t numTriangles,
                                    size_t numTimesteps, size_t rowBegin,
                                    size_t localRows, unsigned long long* result) {
    unsigned long long local = 0;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < localRows; row += stride) {
        const size_t i = rowBegin + row;
        for (size_t t = 0; t < numTimesteps; ++t) {
            if (radB[t * numTriangles + i] > EPSILON) {
                ++local;
                break;
            }
        }
    }
    if (local != 0) atomicAdd(result, local);
}

int launchBlocks(size_t workItems, int threads = 256) {
    if (workItems == 0) return 0;
    return static_cast<int>(std::min<size_t>((workItems + threads - 1) / threads, 65535));
}

void allocateDeviceState(const SimulationState& state, const Partition& partition,
                         DeviceState& device) {
    const size_t n = state.numTriangles;
    const size_t localMatrixElements = partition.localRows * n;
    CUDA_CHECK(cudaMalloc(&device.triangles, state.triangles.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&device.nodes, state.flatNodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&device.triangleIndices,
                          state.flatTriangleIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device.areas, state.areas.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&device.kij, std::max<size_t>(localMatrixElements, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&device.tau, std::max<size_t>(localMatrixElements, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device.radB, state.numTimesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&device.distances,
                          std::max<size_t>(partition.localRows, 1) * sizeof(val_t)));
    if (!partition.cudaAwareMPI) {
        CUDA_CHECK(cudaHostAlloc(&device.globalRow, n * sizeof(val_t), cudaHostAllocPortable));
    }

    CUDA_CHECK(cudaMemcpy(device.triangles, state.triangles.data(),
                          state.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.nodes, state.flatNodes.data(),
                          state.flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.triangleIndices, state.flatTriangleIndices.data(),
                          state.flatTriangleIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.areas, state.areas.data(),
                          state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device.radB, 0, state.numTimesteps * n * sizeof(val_t)));
}

void releaseDeviceState(DeviceState& device) {
    if (device.globalRow != nullptr) CUDA_CHECK(cudaFreeHost(device.globalRow));
    CUDA_CHECK(cudaFree(device.distances));
    CUDA_CHECK(cudaFree(device.radB));
    CUDA_CHECK(cudaFree(device.tau));
    CUDA_CHECK(cudaFree(device.kij));
    CUDA_CHECK(cudaFree(device.areas));
    CUDA_CHECK(cudaFree(device.triangleIndices));
    CUDA_CHECK(cudaFree(device.nodes));
    CUDA_CHECK(cudaFree(device.triangles));
}

void computeFormFactorsAndDelays(const SimulationState& state, const Partition& partition,
                                 DeviceState& device) {
    const size_t workItems = partition.localRows * state.numTriangles;
    const int blocks = launchBlocks(workItems);
    if (blocks != 0) {
        precomputeKernel<<<blocks, 256>>>(
            device.triangles, device.nodes, device.triangleIndices, device.kij, device.tau,
            state.numTriangles, partition.rowBegin, partition.localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runSimulation(const SimulationState& state, const Partition& partition,
                   DeviceState& device, bool printProgress) {
    const int blocks = static_cast<int>(std::min<size_t>(partition.localRows, 65535));
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        if (blocks != 0) {
            simulationTimestepKernel<<<blocks, 256>>>(
                device.kij, device.tau, device.areas, device.radB,
                state.numTriangles, timestep, state.numTimesteps, state.sourceIndex,
                partition.rowBegin, partition.localRows, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
        }
        val_t* deviceRow = device.radB + timestep * state.numTriangles;
        if (partition.cudaAwareMPI) {
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, deviceRow,
                                     partition.rowCounts.data(), partition.rowDisplacements.data(),
                                     MPI_FLOAT, MPI_COMM_WORLD));
        } else {
            if (partition.localRows != 0) {
                CUDA_CHECK(cudaMemcpy(device.globalRow + partition.rowBegin,
                                      deviceRow + partition.rowBegin,
                                      partition.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
            }
            MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, device.globalRow,
                                     partition.rowCounts.data(), partition.rowDisplacements.data(),
                                     MPI_FLOAT, MPI_COMM_WORLD));
            CUDA_CHECK(cudaMemcpy(deviceRow, device.globalRow,
                                  state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
        }
        if (printProgress && ((timestep + 1) % 10 == 0 || timestep + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state, const Partition& partition,
                      DeviceState& device) {
    const int blocks = static_cast<int>(std::min<size_t>(partition.localRows, 65535));
    if (blocks != 0) {
        distanceKernel<<<blocks, 256>>>(device.radB, device.distances, state.numTriangles,
                                        state.numTimesteps, state.sourceIndex,
                                        partition.rowBegin, partition.localRows);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<val_t> localDistances(partition.localRows);
    if (partition.localRows != 0) {
        CUDA_CHECK(cudaMemcpy(localDistances.data(), device.distances,
                              partition.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    if (partition.rank == 0) state.distances.resize(state.numTriangles);
    MPI_CHECK(MPI_Gatherv(localDistances.data(), static_cast<int>(partition.localRows), MPI_FLOAT,
                          partition.rank == 0 ? state.distances.data() : nullptr,
                          partition.rowCounts.data(), partition.rowDisplacements.data(), MPI_FLOAT,
                          0, MPI_COMM_WORLD));
}

void distributedValidationCounts(const SimulationState& state, const Partition& partition,
                                 const DeviceState& device,
                                 unsigned long long& globalFormFactors,
                                 unsigned long long& globalReceived) {
    unsigned long long* deviceCounts = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceCounts, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(deviceCounts, 0, 2 * sizeof(unsigned long long)));
    const size_t localElements = partition.localRows * state.numTriangles;
    int blocks = launchBlocks(localElements);
    if (blocks != 0) {
        countFormFactorsKernel<<<blocks, 256>>>(device.kij, localElements, deviceCounts);
    }
    blocks = launchBlocks(partition.localRows);
    if (blocks != 0) {
        countReceivedKernel<<<blocks, 256>>>(device.radB, state.numTriangles,
                                             state.numTimesteps, partition.rowBegin,
                                             partition.localRows, deviceCounts + 1);
    }
    CUDA_CHECK(cudaGetLastError());
    unsigned long long localCounts[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(localCounts, deviceCounts, sizeof(localCounts), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceCounts));
    unsigned long long globalCounts[2] = {0, 0};
    MPI_CHECK(MPI_Reduce(localCounts, globalCounts, 2, MPI_UNSIGNED_LONG_LONG,
                         MPI_SUM, 0, MPI_COMM_WORLD));
    globalFormFactors = globalCounts[0];
    globalReceived = globalCounts[1];
}

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij,
                     unsigned long long receivedEnergy) {
    printf("\nValidation:\n");
    int invalidCount = 0;
    int negativeCount = 0;
    int nonZeroCount = 0;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    double sumDist = 0.0;
#pragma omp parallel for reduction(+:invalidCount,negativeCount,nonZeroCount,sumDist) \
    reduction(min:minDist) reduction(max:maxDist) schedule(static)
    for (long long index = 0; index < static_cast<long long>(state.numTriangles); ++index) {
        const val_t distance = state.distances[static_cast<size_t>(index)];
        invalidCount += !std::isfinite(distance);
        negativeCount += distance < ZERO;
        nonZeroCount += distance > EPSILON;
        if (std::isfinite(distance)) {
            minDist = std::min(minDist, distance);
            maxDist = std::max(maxDist, distance);
            sumDist += distance;
        }
    }
    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<double>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
    if (state.distances[state.sourceIndex] > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n",
               state.distances[state.sourceIndex]);
    }
    printf("  Triangles receiving energy: %llu/%zu\n", receivedEnergy, state.numTriangles);
    const size_t matrixElements = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij, matrixElements,
           100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(matrixElements));

    if (invalidCount != 0) printf("  ERROR: %d non-finite distances\n", invalidCount);
    if (negativeCount != 0) printf("  ERROR: %d negative distances\n", negativeCount);
    if (receivedEnergy == 0) printf("  ERROR: No triangles received energy - simulation failed\n");
    if (nonZeroKij == 0) printf("  ERROR: All form factors are zero - visibility computation failed\n");
    const bool valid = invalidCount == 0 && negativeCount == 0 &&
                       receivedEnergy != 0 && nonZeroKij != 0;
    printf("  Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid;
}

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
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120,
    //               5->20480, 6->81920
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
    printf("               20, 80, 320, 1280, 5120, 20480, 81920\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    const int initResult = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                                           &providedThreadLevel);
    if (initResult != MPI_SUCCESS) {
        fprintf(stderr, "Unable to initialize MPI.\n");
        return 1;
    }
    int rank = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide the required FUNNELED thread support.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

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
            if (rank == 0) printUsage(argv[0]);
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (targetTriangles <= 0 || timesteps <= 0 || sourceIdx < 0 ||
        !std::isfinite(reflectivity) || reflectivity < ZERO || reflectivity > ONE) {
        if (rank == 0) {
            fprintf(stderr, "Invalid arguments: -n and -t must be positive, -s must be "
                            "non-negative, and -r must be in [0,1].\n");
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI + OpenMP + CUDA\n\n");
    }

    size_t actualTriangles = 20;
    for (int level = 0; level < subdivisions; ++level) actualTriangles *= 4;
    Partition partition = initializePartition(actualTriangles);
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / partition.localRanks));
    }
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank == 0);
    if (state.numTriangles != actualTriangles) {
        if (rank == 0) fprintf(stderr, "Internal mesh partition size mismatch.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    if (rank == 0) {
        printf("Hybrid resources: %d MPI rank(s), %d OpenMP thread(s)/rank, "
               "%d CUDA device(s)/node\n",
               partition.ranks, omp_get_max_threads(), partition.deviceCount);
        printf("MPI device buffers: %s\n",
               partition.cudaAwareMPI ? "CUDA-aware direct collectives" : "pinned-host staging");
        printf("Distributed matrix rows: %zu on rank 0 (contiguous balanced partition)\n\n",
               partition.localRows);
    }

    DeviceState device;
    allocateDeviceState(state, partition, device);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computing time delays (Tau) and form factors (Kij) on CUDA devices...\n");
    }
    double phaseStart = MPI_Wtime();
    computeFormFactorsAndDelays(state, partition, device);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double localDuration = MPI_Wtime() - phaseStart;
    double preSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localDuration, &preSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) printf("Precomputation time: %ld ms\n\n", static_cast<long>(preSeconds * 1000.0));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) printf("Running distributed wave propagation simulation...\n");
    phaseStart = MPI_Wtime();
    runSimulation(state, partition, device, rank == 0);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    localDuration = MPI_Wtime() - phaseStart;
    double simulationSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localDuration, &simulationSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Simulation time: %ld ms\n\n", static_cast<long>(simulationSeconds * 1000.0));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    phaseStart = MPI_Wtime();
    computeDistances(state, partition, device);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    localDuration = MPI_Wtime() - phaseStart;
    double distanceSeconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localDuration, &distanceSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Distance computation time: %ld ms\n\n", static_cast<long>(distanceSeconds * 1000.0));
    }

    unsigned long long nonZeroKij = 0;
    unsigned long long receivedEnergy = 0;
    if (validate) {
        distributedValidationCounts(state, partition, device, nonZeroKij, receivedEnergy);
    }

    int validationStatus = 0;
    if (rank == 0) {
        const long preDuration = static_cast<long>(preSeconds * 1000.0);
        const long simulationDuration = static_cast<long>(simulationSeconds * 1000.0);
        const long distanceDuration = static_cast<long>(distanceSeconds * 1000.0);
        const long totalTime = preDuration + simulationDuration + distanceDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        const size_t n = state.numTriangles;
        const size_t t = state.numTimesteps;
        const double kijOps = static_cast<double>(n) * static_cast<double>(n);
        const double simOps = kijOps * static_cast<double>(t);
        const double distOps = static_cast<double>(n) * static_cast<double>(t) * static_cast<double>(t);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        const size_t memKij = n * n * sizeof(val_t);
        const size_t memTau = n * n * sizeof(int);
        const size_t memRad = 2 * t * n * sizeof(val_t);
        const size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
        printf("  Matrix storage per rank: up to %.2f MB\n",
               (static_cast<double>(partition.localRows) * n *
                (sizeof(val_t) + sizeof(int))) / (1024.0 * 1024.0));

        const uint64_t hash = computeHash(state);
        printf("  Result hash: %016llX\n\n", static_cast<unsigned long long>(hash));

        if (printResults) {
            std::vector<double> distanceData(state.distances.size());
#pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(state.distances.size()); ++i) {
                distanceData[static_cast<size_t>(i)] = state.distances[static_cast<size_t>(i)];
            }
            print_results(distanceData, "Distances");
        }
        if (validate && !validateResults(state, nonZeroKij, receivedEnergy)) {
            validationStatus = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));
    releaseDeviceState(device);
    MPI_CHECK(MPI_Finalize());
    return validationStatus;
}
