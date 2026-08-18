/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
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
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

#ifdef __CUDACC__
#define ROOMSIM_HD __host__ __device__
#else
#define ROOMSIM_HD
#endif

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    ROOMSIM_HD constexpr Vec3() : x(0), y(0), z(0) {}
    ROOMSIM_HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    ROOMSIM_HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    ROOMSIM_HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    ROOMSIM_HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    ROOMSIM_HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    ROOMSIM_HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    ROOMSIM_HD Vec3 operator-() const { return {-x, -y, -z}; }

    ROOMSIM_HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    ROOMSIM_HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    ROOMSIM_HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    ROOMSIM_HD val_t norm() const { return std::sqrt(squaredNorm()); }
    ROOMSIM_HD Vec3 normalized() const {
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
// Flattened scene representation and CUDA kernels
// ============================================================================

// The host octree uses unique_ptrs, which are convenient while building the
// tree but cannot be copied to a device.  A compact indexed representation is
// built once per MPI rank for CUDA traversal.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    uint32_t firstTriangle;
    uint32_t triangleCount;
};

struct DeviceTriangle {
    Vec3 a, b, c, normal;
};

int flattenOctreeNode(const Octree& node,
                      std::vector<FlatOctreeNode>& nodes,
                      std::vector<uint32_t>& triangleIndices) {
    const int nodeIndex = static_cast<int>(nodes.size());
    FlatOctreeNode flat{};
    flat.center = node.center;
    flat.halfExtent = node.halfExtent;
    flat.firstTriangle = 0;
    flat.triangleCount = 0;
    for (int i = 0; i < 8; ++i) flat.children[i] = -1;
    nodes.push_back(flat);

    if (!node.triangleIndices.empty()) {
        nodes[nodeIndex].firstTriangle = static_cast<uint32_t>(triangleIndices.size());
        nodes[nodeIndex].triangleCount = static_cast<uint32_t>(node.triangleIndices.size());
        for (size_t index : node.triangleIndices) {
            triangleIndices.push_back(static_cast<uint32_t>(index));
        }
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                nodes[nodeIndex].children[i] = flattenOctreeNode(
                    *node.children[i], nodes, triangleIndices);
            }
        }
    }
    return nodeIndex;
}

void checkCuda(cudaError_t result, const char* operation, int rank) {
    if (result == cudaSuccess) return;
    if (rank == 0) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

#define CUDA_CHECK(call) checkCuda((call), #call, mpiRank)

struct GpuScene {
    DeviceTriangle* triangles = nullptr;
    FlatOctreeNode* nodes = nullptr;
    uint32_t* nodeTriangles = nullptr;
    size_t triangleCount = 0;
    size_t nodeCount = 0;
    size_t nodeTriangleCount = 0;

    void upload(const std::vector<Triangle>& hostTriangles, const Octree& octree, int mpiRank) {
        std::vector<DeviceTriangle> deviceTriangles(hostTriangles.size());
        for (size_t i = 0; i < hostTriangles.size(); ++i) {
            deviceTriangles[i] = {hostTriangles[i].a, hostTriangles[i].b,
                                  hostTriangles[i].c, hostTriangles[i].normal()};
        }

        std::vector<FlatOctreeNode> hostNodes;
        std::vector<uint32_t> hostNodeTriangles;
        hostNodes.reserve(hostTriangles.size() * 2);
        hostNodeTriangles.reserve(hostTriangles.size() * 2);
        flattenOctreeNode(octree, hostNodes, hostNodeTriangles);

        triangleCount = deviceTriangles.size();
        nodeCount = hostNodes.size();
        nodeTriangleCount = hostNodeTriangles.size();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&triangles),
                              triangleCount * sizeof(DeviceTriangle)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nodes),
                              nodeCount * sizeof(FlatOctreeNode)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nodeTriangles),
                              nodeTriangleCount * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(triangles, deviceTriangles.data(),
                              triangleCount * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(nodes, hostNodes.data(),
                              nodeCount * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(nodeTriangles, hostNodeTriangles.data(),
                              nodeTriangleCount * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    void release() {
        cudaFree(triangles);
        cudaFree(nodes);
        cudaFree(nodeTriangles);
        triangles = nullptr;
        nodes = nullptr;
        nodeTriangles = nullptr;
    }

    ~GpuScene() { release(); }
};

__device__ inline uint32_t nextRandom(uint32_t& state) {
    // A cheap, independent stream per triangle pair avoids serialized RNG
    // state and gives identical samples for the same pair on every MPI rank.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline val_t uniformRandom(uint32_t& state) {
    return static_cast<val_t>(nextRandom(state) >> 8) * 0x1.0p-24f;
}

__device__ inline Vec3 randomPointInDeviceTriangle(const DeviceTriangle& tri,
                                                   uint32_t& state) {
    val_t u = uniformRandom(state);
    val_t v = uniformRandom(state);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__device__ inline bool rayIntersectsDeviceBox(const Vec3& p1, const Vec3& p2,
                                              const FlatOctreeNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ inline val_t rayTriangleIntersectDevice(const Vec3& orig, const Vec3& dir,
                                                   const DeviceTriangle& tri) {
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = ONE / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return FLT_MAX;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                                   const FlatOctreeNode* nodes,
                                   const uint32_t* nodeTriangles,
                                   const DeviceTriangle* triangles,
                                   int triangleCount,
                                   uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 direction = to - from;
    val_t rayLength = sqrtf(direction.squaredNorm());
    if (rayLength < EPSILON) return true;
    Vec3 directionNormalized = direction / rayLength;

    // The generated octree is shallow (the leaf size is fixed), so a small
    // private traversal stack is substantially cheaper than global recursion.
    int stack[128];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize > 0) {
        const int nodeIndex = stack[--stackSize];
        const FlatOctreeNode& node = nodes[nodeIndex];
        if (!rayIntersectsDeviceBox(from, to, node)) continue;

        if (node.triangleCount != 0) {
            for (uint32_t k = 0; k < node.triangleCount; ++k) {
                const uint32_t idx = nodeTriangles[node.firstTriangle + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                val_t distance = rayTriangleIntersectDevice(from, directionNormalized,
                                                             triangles[idx]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
        } else {
            for (int child = 7; child >= 0; --child) {
                const int childIndex = node.children[child];
                if (childIndex >= 0 && stackSize < 128) stack[stackSize++] = childIndex;
            }
        }
    }
    (void)triangleCount;
    return false;
}

__global__ void formFactorKernel(const DeviceTriangle* triangles,
                                 const FlatOctreeNode* nodes,
                                 const uint32_t* nodeTriangles,
                                 int triangleCount,
                                 int rowBegin, int localRows, val_t* output) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = static_cast<size_t>(localRows) * triangleCount;
    if (pair >= pairCount) return;

    const int localI = static_cast<int>(pair / triangleCount);
    const int j = static_cast<int>(pair % triangleCount);
    const int i = rowBegin + localI;
    val_t result = ZERO;
    if (i != j) {
        const DeviceTriangle& triI = triangles[i];
        const DeviceTriangle& triJ = triangles[j];
        if (triI.normal.dot(triJ.normal) <= 0.99f) {
            uint32_t randomState = 42u ^
                (static_cast<uint32_t>(i + 1) * 0x9e3779b9u) ^
                (static_cast<uint32_t>(j + 1) * 0x85ebca6bu);
            if (randomState == 0) randomState = 1;
            for (int ray = 0; ray < NUM_RAYS; ++ray) {
                Vec3 pI = randomPointInDeviceTriangle(triI, randomState);
                Vec3 pJ = randomPointInDeviceTriangle(triJ, randomState);
                if (isRayBlockedDevice(pI, pJ, nodes, nodeTriangles, triangles,
                                       triangleCount, static_cast<uint32_t>(i),
                                       static_cast<uint32_t>(j))) continue;

                Vec3 v = pJ - pI;
                val_t distanceSquared = v.squaredNorm();
                if (distanceSquared < EPSILON) continue;
                val_t length = sqrtf(distanceSquared);
                val_t cosPhiI = fmaxf(ZERO, v.dot(triI.normal) / length);
                val_t cosPhiJ = fmaxf(ZERO, (-v).dot(triJ.normal) / length);
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    result += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
                }
            }
            result *= INV_NUM_RAYS;
        }
    }
    output[pair] = result;
}

__global__ void timeDelayKernel(const DeviceTriangle* triangles, int triangleCount,
                                int rowBegin, int localRows, int* output) {
    const size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = static_cast<size_t>(localRows) * triangleCount;
    if (pair >= pairCount) return;
    const int localI = static_cast<int>(pair / triangleCount);
    const int j = static_cast<int>(pair % triangleCount);
    const int i = rowBegin + localI;
    if (i == j) {
        output[pair] = 0;
        return;
    }
    Vec3 delta = (triangles[i].a + triangles[i].b + triangles[i].c) / 3.0f -
                 (triangles[j].a + triangles[j].b + triangles[j].c) / 3.0f;
    output[pair] = static_cast<int>(ceilf(sqrtf(delta.squaredNorm()) * INV_WAVE_SPEED));
}

__global__ void propagationKernel(const val_t* kij, const int* tau,
                                  const val_t* areas, const val_t* history,
                                  int triangleCount, int rowBegin, int localRows,
                                  int time, int sourceIndex, int timeOff,
                                  val_t reflectivity, val_t* current) {
    const int localI = static_cast<int>(blockIdx.x);
    if (localI >= localRows) return;
    const int globalI = rowBegin + localI;
    val_t sum = ZERO;
    for (int j = static_cast<int>(threadIdx.x); j < triangleCount; j += blockDim.x) {
        if (globalI == j) continue;
        const size_t pair = static_cast<size_t>(localI) * triangleCount + j;
        const int delay = tau[pair];
        if (time < delay) continue;
        const val_t formFactor = kij[pair];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = history[
            (static_cast<size_t>(time - delay) * triangleCount) + j];
        if (sourceRadiosity <= ZERO) continue;
        sum += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }

    extern __shared__ val_t reduction[];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (globalI == sourceIndex && time < timeOff) ? ONE : ZERO;
        current[localI] = reflectivity * reduction[0] + emission;
    }
}

__global__ void distanceKernel(const val_t* history, int triangleCount, int timesteps,
                               int rowBegin, int localRows, int sourceIndex,
                               val_t* localDistances) {
    const int localI = static_cast<int>(blockIdx.x);
    if (localI >= localRows) return;
    const int globalI = rowBegin + localI;
    int bestTime = 0;
    val_t bestCorrelation = ZERO;
    for (int lag = static_cast<int>(threadIdx.x); lag < timesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (int time = lag; time < timesteps; ++time) {
            val_t receiver = history[static_cast<size_t>(time) * triangleCount + globalI];
            val_t source = history[static_cast<size_t>(time - lag) * triangleCount + sourceIndex];
            sum += source * receiver;
        }
        if (sum > bestCorrelation ||
            (sum == bestCorrelation && lag < bestTime)) {
            bestCorrelation = sum;
            bestTime = lag;
        }
    }

    extern __shared__ unsigned char shared[];
    val_t* correlations = reinterpret_cast<val_t*>(shared);
    int* times = reinterpret_cast<int*>(correlations + blockDim.x);
    correlations[threadIdx.x] = bestCorrelation;
    times[threadIdx.x] = bestTime;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t otherCorrelation = correlations[threadIdx.x + stride];
            const int otherTime = times[threadIdx.x + stride];
            if (otherCorrelation > correlations[threadIdx.x] ||
                (otherCorrelation == correlations[threadIdx.x] &&
                 otherTime < times[threadIdx.x])) {
                correlations[threadIdx.x] = otherCorrelation;
                times[threadIdx.x] = otherTime;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        localDistances[localI] = WAVE_SPEED * static_cast<val_t>(times[0]);
    }
}

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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t rowBegin = 0;
    size_t localRows = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Local rows of the form-factor matrix
    std::vector<int> tau;           // Local rows of the time-delay matrix
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const {
        return (i - rowBegin) * numTriangles + j;
    }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

void makeRowPartition(size_t n, int mpiRank, int mpiSize,
                      size_t& rowBegin, size_t& localRows,
                      std::vector<int>& counts, std::vector<int>& displacements) {
    const size_t base = n / static_cast<size_t>(mpiSize);
    const size_t remainder = n % static_cast<size_t>(mpiSize);
    rowBegin = static_cast<size_t>(mpiRank) * base +
               std::min(static_cast<size_t>(mpiRank), remainder);
    localRows = base + (static_cast<size_t>(mpiRank) < remainder ? 1 : 0);

    counts.resize(mpiSize);
    displacements.resize(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        const size_t rankRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        const size_t rankBegin = static_cast<size_t>(rank) * base +
                                 std::min(static_cast<size_t>(rank), remainder);
        if (rankRows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            rankBegin > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (mpiRank == 0) {
                fprintf(stderr, "MPI row partition exceeds MPI int count limits\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[rank] = static_cast<int>(rankRows);
        displacements[rank] = static_cast<int>(rankBegin);
    }
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          size_t rowBegin, size_t localRows, int mpiRank) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.rowBegin = rowBegin;
    state.localRows = localRows;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(localRows * state.numTriangles, ZERO);
    state.tau.resize(localRows * state.numTriangles, 0);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, const GpuScene& scene, int mpiRank) {
    if (mpiRank == 0) printf("Computing form factors (Kij) on CUDA...\n");
    if (state.localRows == 0) return;

    const size_t pairCount = state.localRows * state.numTriangles;
    val_t* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput),
                          pairCount * sizeof(val_t)));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pairCount + threads - 1) / threads);
    formFactorKernel<<<blocks, threads>>>(
        scene.triangles, scene.nodes, scene.nodeTriangles,
        static_cast<int>(state.numTriangles), static_cast<int>(state.rowBegin),
        static_cast<int>(state.localRows), deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.kij.data(), deviceOutput,
                          pairCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
}

void computeTimeDelays(SimulationState& state, const GpuScene& scene, int mpiRank) {
    if (mpiRank == 0) printf("Computing time delays (Tau) on CUDA...\n");
    if (state.localRows == 0) return;

    const size_t pairCount = state.localRows * state.numTriangles;
    int* deviceOutput = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput),
                          pairCount * sizeof(int)));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((pairCount + threads - 1) / threads);
    timeDelayKernel<<<blocks, threads>>>(
        scene.triangles, static_cast<int>(state.numTriangles),
        static_cast<int>(state.rowBegin), static_cast<int>(state.localRows), deviceOutput);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.tau.data(), deviceOutput,
                          pairCount * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, const GpuScene& scene,
                   const std::vector<int>& rowCounts,
                   const std::vector<int>& rowDisplacements,
                   int mpiRank, int mpiSize, val_t reflectivity) {
    if (mpiRank == 0) printf("Running wave propagation simulation on CUDA...\n");

    val_t* deviceKij = nullptr;
    int* deviceTau = nullptr;
    val_t* deviceAreas = nullptr;
    val_t* deviceHistory = nullptr;
    val_t* deviceCurrent = nullptr;
    if (state.localRows != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceKij),
                              state.kij.size() * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTau),
                              state.tau.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceAreas),
                              state.areas.size() * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(deviceKij, state.kij.data(),
                              state.kij.size() * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceTau, state.tau.data(),
                              state.tau.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceAreas, state.areas.data(),
                              state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (!state.radB.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceHistory),
                              state.radB.size() * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(deviceHistory, 0, state.radB.size() * sizeof(val_t)));
    }
    if (state.localRows != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCurrent),
                              state.localRows * sizeof(val_t)));
    }

    std::vector<val_t> localCurrent(state.localRows, ZERO);
    const int timeOff = static_cast<int>(state.numTimesteps / 2);
    constexpr int threads = 256;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows != 0) {
            propagationKernel<<<static_cast<int>(state.localRows), threads,
                                threads * sizeof(val_t)>>>(
                deviceKij, deviceTau, deviceAreas, deviceHistory,
                static_cast<int>(state.numTriangles), static_cast<int>(state.rowBegin),
                static_cast<int>(state.localRows), static_cast<int>(t),
                static_cast<int>(state.sourceIndex), timeOff, reflectivity, deviceCurrent);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localCurrent.data(), deviceCurrent,
                                  state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(localCurrent.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                       state.radB.data() + t * state.numTriangles,
                       rowCounts.data(), rowDisplacements.data(), MPI_FLOAT,
                       MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(deviceHistory + t * state.numTriangles,
                              state.radB.data() + t * state.numTriangles,
                              state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    if (deviceAreas) CUDA_CHECK(cudaFree(deviceAreas));
    if (deviceTau) CUDA_CHECK(cudaFree(deviceTau));
    if (deviceKij) CUDA_CHECK(cudaFree(deviceKij));
    if (deviceCurrent) CUDA_CHECK(cudaFree(deviceCurrent));
    if (deviceHistory) CUDA_CHECK(cudaFree(deviceHistory));
    (void)scene;
    (void)mpiSize;
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, int mpiRank, int mpiSize,
                      const std::vector<int>& rowCounts,
                      const std::vector<int>& rowDisplacements) {
    if (mpiRank == 0) printf("Computing distances via cross-correlation on CUDA...\n");
    std::vector<val_t> localDistances(state.localRows, ZERO);
    if (state.localRows != 0 && state.numTimesteps != 0) {
        val_t* deviceHistory = nullptr;
        val_t* deviceDistances = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceHistory),
                              state.radB.size() * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDistances),
                              state.localRows * sizeof(val_t)));
        CUDA_CHECK(cudaMemcpy(deviceHistory, state.radB.data(),
                              state.radB.size() * sizeof(val_t), cudaMemcpyHostToDevice));
        constexpr int threads = 256;
        distanceKernel<<<static_cast<int>(state.localRows), threads,
                         threads * (sizeof(val_t) + sizeof(int))>>>(
            deviceHistory, static_cast<int>(state.numTriangles),
            static_cast<int>(state.numTimesteps), static_cast<int>(state.rowBegin),
            static_cast<int>(state.localRows), static_cast<int>(state.sourceIndex),
            deviceDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDistances.data(), deviceDistances,
                              state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceDistances));
        CUDA_CHECK(cudaFree(deviceHistory));
    }

    if (mpiRank == 0) state.distances.assign(state.numTriangles, ZERO);
    MPI_Gatherv(localDistances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
                mpiRank == 0 ? state.distances.data() : nullptr,
                rowCounts.data(), rowDisplacements.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    (void)mpiSize;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, int mpiRank, int mpiSize) {
    int localReceivedEnergy = 0;
    #pragma omp parallel for reduction(+:localReceivedEnergy) schedule(static)
    for (long long local = 0; local < static_cast<long long>(state.localRows); ++local) {
        const size_t i = state.rowBegin + static_cast<size_t>(local);
        bool received = false;
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                received = true;
                break;
            }
        }
        if (received) ++localReceivedEnergy;
    }

    int receivedEnergy = 0;
    MPI_Reduce(&localReceivedEnergy, &receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int localNonZeroKij = 0;
    #pragma omp parallel for reduction(+:localNonZeroKij) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.kij.size()); ++i) {
        if (state.kij[static_cast<size_t>(i)] > EPSILON) ++localNonZeroKij;
    }
    int nonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (mpiRank == 0) {
        printf("\nValidation:\n");
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;
        int allNonNegative = 1;
        int allFinite = 1;

        #pragma omp parallel for reduction(min:minDist) reduction(max:maxDist) \
            reduction(+:sumDist,nonZeroCount) reduction(&:allNonNegative,allFinite) schedule(static)
        for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
            const val_t d = state.distances[static_cast<size_t>(i)];
            if (d < ZERO) allNonNegative = 0;
            if (!std::isfinite(d)) allFinite = 0;
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) ++nonZeroCount;
        }

        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n",
               sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

        const val_t srcDist = state.distances[state.sourceIndex];
        if (srcDist > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
        }

        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        if (receivedEnergy == 0) {
            printf("  ERROR: No triangles received energy - simulation failed\n");
            valid = 0;
        }

        const size_t totalPairs = state.numTriangles * state.numTriangles;
        printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
               nonZeroKij, totalPairs,
               100.0f * nonZeroKij / static_cast<val_t>(totalPairs));
        if (nonZeroKij == 0) {
            printf("  ERROR: All form factors are zero - visibility computation failed\n");
            valid = 0;
        }
        if (!allNonNegative || !allFinite) valid = 0;
        if (valid) printf("  Validation: PASSED\n");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    (void)mpiSize;
    return valid != 0;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (mpiRank == 0) fprintf(stderr, "MPI implementation lacks MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    size_t actualTriangles = 20;
    for (int level = 0; level < subdivisions; ++level) actualTriangles *= 4;
    size_t rowBegin = 0;
    size_t localRows = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    makeRowPartition(actualTriangles, mpiRank, mpiSize, rowBegin, localRows,
                     rowCounts, rowDisplacements);

    // CUDA is deliberately mandatory: every MPI rank owns a GPU context and
    // all dense kernels execute on that rank's assigned accelerator.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (mpiRank == 0) fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Hybrid execution: MPI ranks=%d, OpenMP threads=%d, CUDA devices=%d\n",
               mpiSize, omp_get_max_threads(), deviceCount);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity,
                         rowBegin, localRows, mpiRank);
    GpuScene scene;
    scene.upload(state.triangles, state.octree, mpiRank);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startPre = MPI_Wtime();

    computeTimeDelays(state, scene, mpiRank);
    computeFormFactors(state, scene, mpiRank);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localPreDuration = MPI_Wtime() - startPre;
    double maxPreDuration = 0.0;
    MPI_Reduce(&localPreDuration, &maxPreDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long preDuration = mpiRank == 0 ? static_cast<long>(maxPreDuration * 1000.0) : 0;

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startSim = MPI_Wtime();

    runSimulation(state, scene, rowCounts, rowDisplacements,
                  mpiRank, mpiSize, reflectivity);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localSimDuration = MPI_Wtime() - startSim;
    double maxSimDuration = 0.0;
    MPI_Reduce(&localSimDuration, &maxSimDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long simDuration = mpiRank == 0 ? static_cast<long>(maxSimDuration * 1000.0) : 0;

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    const double startDist = MPI_Wtime();

    computeDistances(state, mpiRank, mpiSize, rowCounts, rowDisplacements);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localDistDuration = MPI_Wtime() - startDist;
    double maxDistDuration = 0.0;
    MPI_Reduce(&localDistDuration, &maxDistDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long distDuration = mpiRank == 0 ? static_cast<long>(maxDistDuration * 1000.0) : 0;

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    const long totalTime = preDuration + simDuration + distDuration;
    if (mpiRank == 0) printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    if (mpiRank == 0) {
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Kij and Tau are row-distributed; radiosity is replicated because
        // every destination row needs delayed values from every source row.
        size_t memKij = state.kij.size() * sizeof(val_t);
        size_t memTau = state.tau.size() * sizeof(int);
        size_t memRad = t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage per rank: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t resultHash = computeHash(state);
        printf("  Result hash: %016llX\n", static_cast<unsigned long long>(resultHash));
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }
    }

    // Validation
    bool validationPassed = true;
    if (validate) {
        validationPassed = validateResults(state, mpiRank, mpiSize);
    }

    scene.release();
    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
