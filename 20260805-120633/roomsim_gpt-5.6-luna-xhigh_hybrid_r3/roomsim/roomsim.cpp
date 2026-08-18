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
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#define ROOMSIM_HD __host__ __device__
#else
#define ROOMSIM_HD
#endif

int g_mpiRank = 0;

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

    ROOMSIM_HD constexpr Vec3() : x(0), y(0), z(0) {}
    ROOMSIM_HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    ROOMSIM_HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

// A contiguous representation of the host octree for GPU traversal.  The
// original tree is retained for construction and the flat form is copied to
// every rank's accelerator once, avoiding an O(N) visibility search in the
// form-factor kernel.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    int triangleOffset;
    int triangleCount;
};

int flattenOctreeNode(const Octree& source,
                      std::vector<FlatOctreeNode>& nodes,
                      std::vector<idx_t>& triangleIndices) {
    const int nodeIndex = static_cast<int>(nodes.size());
    FlatOctreeNode flat{};
    flat.center = source.center;
    flat.halfExtent = source.halfExtent;
    flat.triangleOffset = static_cast<int>(triangleIndices.size());
    flat.triangleCount = static_cast<int>(source.triangleIndices.size());
    for (int i = 0; i < 8; ++i) flat.children[i] = -1;

    nodes.push_back(flat);
    triangleIndices.reserve(triangleIndices.size() + source.triangleIndices.size());
    for (size_t triangleIndex : source.triangleIndices) {
        triangleIndices.push_back(static_cast<idx_t>(triangleIndex));
    }

    for (int i = 0; i < 8; ++i) {
        if (source.children[i]) {
            nodes[nodeIndex].children[i] = flattenOctreeNode(
                *source.children[i], nodes, triangleIndices);
        }
    }
    return nodeIndex;
}

void flattenOctree(const Octree& source,
                   std::vector<FlatOctreeNode>& nodes,
                   std::vector<idx_t>& triangleIndices) {
    nodes.clear();
    triangleIndices.clear();
    if (source.allTriangles != nullptr) {
        flattenOctreeNode(source, nodes, triangleIndices);
    }
}

// ============================================================================
// CUDA device routines and kernels
// ============================================================================

__device__ inline Vec3 deviceSub(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline Vec3 deviceAdd(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline Vec3 deviceScale(const Vec3& a, val_t scale) {
    return Vec3(a.x * scale, a.y * scale, a.z * scale);
}

__device__ inline val_t deviceDot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline Vec3 deviceCross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__device__ inline val_t deviceSquaredNorm(const Vec3& v) {
    return deviceDot(v, v);
}

__device__ inline val_t deviceNorm(const Vec3& v) {
    return sqrtf(deviceSquaredNorm(v));
}

__device__ inline Vec3 deviceTriangleCenter(const Triangle& triangle) {
    return deviceScale(deviceAdd(deviceAdd(triangle.a, triangle.b), triangle.c), 1.0f / 3.0f);
}

__device__ inline bool deviceRayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                               const FlatOctreeNode& node) {
    const Vec3 d = deviceScale(deviceSub(p2, p1), 0.5f);
    const Vec3 c = deviceSub(deviceAdd(p1, d), node.center);
    const Vec3 ad = Vec3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

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

__device__ inline val_t deviceRayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                    const Triangle& triangle) {
    const Vec3 e1 = deviceSub(triangle.b, triangle.a);
    const Vec3 e2 = deviceSub(triangle.c, triangle.a);
    const Vec3 pvec = deviceCross(dir, e2);
    const val_t det = deviceDot(e1, pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;

    const val_t invDet = 1.0f / det;
    const Vec3 tvec = deviceSub(orig, triangle.a);
    const val_t u = deviceDot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    const Vec3 qvec = deviceCross(tvec, e1);
    const val_t v = deviceDot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return deviceDot(e2, qvec) * invDet;
}

__device__ bool deviceRayBlocked(const Vec3& from, const Vec3& to,
                                 const Triangle* triangles,
                                 const FlatOctreeNode* nodes,
                                 const idx_t* octreeTriangles,
                                 idx_t sourceIndex, idx_t destinationIndex) {
    const Vec3 direction = deviceSub(to, from);
    const val_t rayLength = deviceNorm(direction);
    if (rayLength < EPSILON) return true;
    const Vec3 directionNorm = deviceScale(direction, 1.0f / rayLength);

    // The maximum depth of the generated tree is small.  An explicit stack
    // avoids device recursion while retaining the host traversal order.
    int stack[128];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize > 0) {
        const FlatOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount > 0) {
            for (int k = 0; k < node.triangleCount; ++k) {
                const idx_t triangleIndex = octreeTriangles[node.triangleOffset + k];
                if (triangleIndex == sourceIndex || triangleIndex == destinationIndex) continue;
                const val_t distance = deviceRayTriangleIntersect(
                    from, directionNorm, triangles[triangleIndex]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
            continue;
        }

        // Push in reverse so that children are visited in 0..7 order.
        for (int child = 7; child >= 0; --child) {
            const int childIndex = node.children[child];
            if (childIndex >= 0 && deviceRayIntersectsBox(from, to, nodes[childIndex])) {
                if (stackSize < static_cast<int>(sizeof(stack) / sizeof(stack[0]))) {
                    stack[stackSize++] = childIndex;
                }
            }
        }
    }
    return false;
}

__device__ inline uint32_t deviceMix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

__device__ inline val_t deviceRandom(uint64_t pairIndex, int ray, int component) {
    const uint64_t key = pairIndex * 0x9e3779b97f4a7c15ULL +
                         static_cast<uint64_t>(ray * 4 + component + 42);
    const uint32_t mixed = deviceMix(static_cast<uint32_t>(key) ^
                                      static_cast<uint32_t>(key >> 32));
    return static_cast<val_t>(mixed & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ inline Vec3 deviceRandomPoint(const Triangle& triangle,
                                          uint64_t pairIndex, int ray, int offset) {
    val_t u = deviceRandom(pairIndex, ray, offset);
    val_t v = deviceRandom(pairIndex, ray, offset + 1);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return deviceAdd(triangle.a,
                     deviceAdd(deviceScale(deviceSub(triangle.b, triangle.a), u),
                               deviceScale(deviceSub(triangle.c, triangle.a), v)));
}

__device__ inline val_t deviceCosPhi(const Vec3& vector, const Vec3& normal) {
    const val_t vectorNorm = deviceNorm(vector);
    if (vectorNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, deviceDot(vector, normal) / vectorNorm);
}

__global__ void computeFormFactorsKernel(const Triangle* triangles,
                                         const FlatOctreeNode* nodes,
                                         const idx_t* octreeTriangles,
                                         idx_t triangleCount, idx_t rowStart,
                                         idx_t localRowCount, val_t* kij) {
    const uint64_t localPairCount = static_cast<uint64_t>(localRowCount) * triangleCount;
    const uint64_t threadStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t threadStride = static_cast<uint64_t>(gridDim.x) * blockDim.x;

    for (uint64_t localPair = threadStart; localPair < localPairCount; localPair += threadStride) {
        const idx_t localI = static_cast<idx_t>(localPair / triangleCount);
        const idx_t i = rowStart + localI;
        const idx_t j = static_cast<idx_t>(localPair % triangleCount);
        const size_t matrixIndex = static_cast<size_t>(i) * triangleCount + j;
        if (i == j) {
            kij[matrixIndex] = ZERO;
            continue;
        }

        const Triangle& triangleI = triangles[i];
        const Triangle& triangleJ = triangles[j];
        if (deviceDot(triangleI._normal, triangleJ._normal) > 0.99f) {
            kij[matrixIndex] = ZERO;
            continue;
        }

        val_t value = ZERO;
        const uint64_t pairIndex = static_cast<uint64_t>(i) * triangleCount + j;
        for (int ray = 0; ray < NUM_RAYS; ++ray) {
            const Vec3 pI = deviceRandomPoint(triangleI, pairIndex, ray, 0);
            const Vec3 pJ = deviceRandomPoint(triangleJ, pairIndex, ray, 2);
            if (deviceRayBlocked(pI, pJ, triangles, nodes, octreeTriangles, i, j)) continue;

            const Vec3 vector = deviceSub(pJ, pI);
            const val_t distanceSquared = deviceSquaredNorm(vector);
            if (distanceSquared < EPSILON) continue;
            const val_t cosPhiI = deviceCosPhi(vector, triangleI._normal);
            const val_t cosPhiJ = deviceCosPhi(deviceScale(vector, -1.0f), triangleJ._normal);
            if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;
            value += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
        }
        kij[matrixIndex] = value * INV_NUM_RAYS;
    }
}

__global__ void computeTimeDelaysKernel(const Vec3* centers,
                                        idx_t triangleCount, idx_t rowStart,
                                        idx_t localRowCount, int* tau) {
    const uint64_t localPairCount = static_cast<uint64_t>(localRowCount) * triangleCount;
    const uint64_t threadStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t threadStride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t localPair = threadStart; localPair < localPairCount; localPair += threadStride) {
        const idx_t i = rowStart + static_cast<idx_t>(localPair / triangleCount);
        const idx_t j = static_cast<idx_t>(localPair % triangleCount);
        if (i == j) {
            tau[static_cast<size_t>(i) * triangleCount + j] = 0;
            continue;
        }
        const Vec3 delta = deviceSub(centers[i], centers[j]);
        tau[static_cast<size_t>(i) * triangleCount + j] =
            static_cast<int>(ceilf(deviceNorm(delta) * INV_WAVE_SPEED));
    }
}

__global__ void propagateKernel(uint32_t timestep, uint32_t timesteps,
                                uint32_t triangleCount, uint32_t rowStart,
                                uint32_t localRowCount, uint32_t sourceIndex,
                                val_t reflectivity, const val_t* areas,
                                const val_t* kij, const int* tau,
                                val_t* radB) {
    const uint32_t localI = blockIdx.x;
    if (localI >= localRowCount) return;
    const uint32_t i = rowStart + localI;
    const uint32_t tid = threadIdx.x;
    __shared__ val_t partial[256];
    val_t sumB = ZERO;

    for (uint32_t j = tid; j < triangleCount; j += blockDim.x) {
        if (i == j) continue;
        const size_t matrixIndex = static_cast<size_t>(i) * triangleCount + j;
        const int delay = tau[matrixIndex];
        if (static_cast<int>(timestep) < delay) continue;
        const val_t formFactor = kij[matrixIndex];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radB[
            static_cast<size_t>(timestep - static_cast<uint32_t>(delay)) * triangleCount + j];
        if (sourceRadiosity <= ZERO) continue;
        sumB += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }

    partial[tid] = sumB;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const val_t emission = (timestep < timesteps / 2 && i == sourceIndex) ? ONE : ZERO;
        radB[static_cast<size_t>(timestep) * triangleCount + i] =
            reflectivity * partial[0] + emission;
    }
}

__global__ void distanceCorrelationKernel(uint32_t triangleCount,
                                           uint32_t timesteps,
                                           uint32_t rowStart,
                                           uint32_t localRowCount,
                                           uint32_t sourceIndex,
                                           const val_t* radB,
                                           val_t* correlations) {
    const uint64_t work = static_cast<uint64_t>(localRowCount) * timesteps;
    const uint64_t threadStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t threadStride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t item = threadStart; item < work; item += threadStride) {
        const uint32_t localI = static_cast<uint32_t>(item / timesteps);
        const uint32_t lag = static_cast<uint32_t>(item % timesteps);
        const uint32_t i = rowStart + localI;
        val_t sum = ZERO;
        for (uint32_t tt = lag; tt < timesteps; ++tt) {
            const val_t pB = radB[static_cast<size_t>(tt) * triangleCount + i];
            const val_t pS = radB[static_cast<size_t>(tt - lag) * triangleCount + sourceIndex];
            sum += pS * pB;
        }
        correlations[static_cast<size_t>(localI) * timesteps + lag] = sum;
    }
}

__global__ void distanceReduceKernel(uint32_t timesteps, uint32_t localRowCount,
                                     val_t* correlations, val_t* distances) {
    const uint32_t localI = blockIdx.x * blockDim.x + threadIdx.x;
    if (localI >= localRowCount) return;
    val_t maxCorrelation = ZERO;
    uint32_t bestT = 0;
    const size_t base = static_cast<size_t>(localI) * timesteps;
    for (uint32_t lag = 0; lag < timesteps; ++lag) {
        const val_t correlation = correlations[base + lag];
        if (correlation > maxCorrelation) {
            maxCorrelation = correlation;
            bestT = lag;
        }
    }
    distances[localI] = WAVE_SPEED * static_cast<val_t>(bestT);
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
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

struct MpiPartition {
    int rank = 0;
    int size = 1;
    size_t rowStart = 0;
    size_t rowCount = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;
    std::vector<int> counts;
    std::vector<int> displacements;
};

void cudaFailure(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: CUDA failure at %s:%d (%s): %s\n",
                 g_mpiRank, file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(expression) cudaFailure((expression), #expression, __FILE__, __LINE__)

template <typename T>
void freeDeviceArray(T*& pointer) {
    if (pointer != nullptr) CUDA_CHECK(cudaFree(pointer));
    pointer = nullptr;
}

MpiPartition makeMpiPartition(size_t triangleCount) {
    MpiPartition partition;
    MPI_Comm_rank(MPI_COMM_WORLD, &partition.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &partition.size);
    partition.rowCounts.resize(partition.size);
    partition.rowDisplacements.resize(partition.size);
    partition.counts.resize(partition.size);
    partition.displacements.resize(partition.size);

    const size_t baseRows = triangleCount / static_cast<size_t>(partition.size);
    const size_t remainder = triangleCount % static_cast<size_t>(partition.size);
    size_t displacement = 0;
    for (int rank = 0; rank < partition.size; ++rank) {
        const size_t rows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        const size_t elements = rows * triangleCount;
        const size_t elementDisplacement = displacement * triangleCount;
        if (elements > static_cast<size_t>(INT_MAX) ||
            elementDisplacement > static_cast<size_t>(INT_MAX)) {
            if (partition.rank == 0) {
                std::fprintf(stderr, "MPI row partition exceeds MPI int count range\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (rows > static_cast<size_t>(INT_MAX) || displacement > static_cast<size_t>(INT_MAX)) {
            if (partition.rank == 0) {
                std::fprintf(stderr, "MPI row partition exceeds MPI int row range\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        partition.rowCounts[rank] = static_cast<int>(rows);
        partition.rowDisplacements[rank] = static_cast<int>(displacement);
        partition.counts[rank] = static_cast<int>(elements);
        partition.displacements[rank] = static_cast<int>(elementDisplacement);
        if (rank == partition.rank) {
            partition.rowStart = displacement;
            partition.rowCount = rows;
        }
        displacement += rows;
    }
    return partition;
}

uint32_t asDeviceIndex(size_t value, const char* name) {
    if (value > static_cast<size_t>(UINT32_MAX)) {
        std::fprintf(stderr, "Rank %d: %s does not fit in CUDA index type\n", g_mpiRank, name);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<uint32_t>(value);
}

uint32_t gridBlocksFor(uint64_t work, uint32_t threads) {
    if (work == 0) return 0;
    const uint64_t blocks = (work + threads - 1) / threads;
    return static_cast<uint32_t>(std::min<uint64_t>(blocks, 2147483647ULL));
}

struct ParallelDeviceState {
    uint32_t triangleCount = 0;
    uint32_t timesteps = 0;
    uint32_t localRowCount = 0;

    Triangle* triangles = nullptr;
    Vec3* centers = nullptr;
    FlatOctreeNode* octreeNodes = nullptr;
    idx_t* octreeTriangles = nullptr;
    val_t* areas = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* correlations = nullptr;
    val_t* distances = nullptr;

    void release() {
        freeDeviceArray(distances);
        freeDeviceArray(correlations);
        freeDeviceArray(radB);
        freeDeviceArray(tau);
        freeDeviceArray(kij);
        freeDeviceArray(areas);
        freeDeviceArray(octreeTriangles);
        freeDeviceArray(octreeNodes);
        freeDeviceArray(centers);
        freeDeviceArray(triangles);
    }
};

void allocateDeviceArray(void** pointer, size_t bytes) {
    if (bytes == 0) {
        *pointer = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(pointer, bytes));
}

void initializeParallelDevice(ParallelDeviceState& device,
                               const SimulationState& state,
                               const MpiPartition& partition) {
    device.triangleCount = asDeviceIndex(state.numTriangles, "triangle count");
    device.timesteps = asDeviceIndex(state.numTimesteps, "timestep count");
    device.localRowCount = asDeviceIndex(partition.rowCount, "local row count");

    std::vector<FlatOctreeNode> flatNodes;
    std::vector<idx_t> flatTriangles;
    flattenOctree(state.octree, flatNodes, flatTriangles);
    if (flatNodes.empty() || flatTriangles.empty()) {
        std::fprintf(stderr, "Rank %d: generated octree is empty\n", g_mpiRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<Vec3> centers(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        centers[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].center();
    }

    const size_t n = state.numTriangles;
    const size_t matrixElements = n * n;
    const size_t historyElements = state.numTimesteps * n;
    allocateDeviceArray(reinterpret_cast<void**>(&device.triangles),
                        n * sizeof(Triangle));
    allocateDeviceArray(reinterpret_cast<void**>(&device.centers),
                        n * sizeof(Vec3));
    allocateDeviceArray(reinterpret_cast<void**>(&device.octreeNodes),
                        flatNodes.size() * sizeof(FlatOctreeNode));
    allocateDeviceArray(reinterpret_cast<void**>(&device.octreeTriangles),
                        flatTriangles.size() * sizeof(idx_t));
    allocateDeviceArray(reinterpret_cast<void**>(&device.areas), n * sizeof(val_t));
    allocateDeviceArray(reinterpret_cast<void**>(&device.kij), matrixElements * sizeof(val_t));
    allocateDeviceArray(reinterpret_cast<void**>(&device.tau), matrixElements * sizeof(int));
    allocateDeviceArray(reinterpret_cast<void**>(&device.radB), historyElements * sizeof(val_t));
    allocateDeviceArray(reinterpret_cast<void**>(&device.correlations),
                        partition.rowCount * state.numTimesteps * sizeof(val_t));
    allocateDeviceArray(reinterpret_cast<void**>(&device.distances),
                        partition.rowCount * sizeof(val_t));

    CUDA_CHECK(cudaMemcpy(device.triangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.centers, centers.data(),
                          n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.octreeNodes, flatNodes.data(),
                          flatNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.octreeTriangles, flatTriangles.data(),
                          flatTriangles.size() * sizeof(idx_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device.areas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    if (historyElements > 0) {
        CUDA_CHECK(cudaMemset(device.radB, 0, historyElements * sizeof(val_t)));
    }
}

void selectCudaDevice() {
    int deviceCount = 0;
    const cudaError_t result = cudaGetDeviceCount(&deviceCount);
    if (result != cudaSuccess || deviceCount == 0) {
        std::fprintf(stderr, "Rank %d: an accelerator is required (CUDA devices found: %d)\n",
                     g_mpiRank, deviceCount);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_free(&localCommunicator);

    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    if (g_mpiRank == 0) {
        int worldSize = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, localRank % deviceCount));
        std::printf("Hybrid execution: MPI ranks=%d, OpenMP threads/rank=%d, CUDA device=%d (%s)\n",
                    worldSize, omp_get_max_threads(), localRank % deviceCount, properties.name);
    }
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (g_mpiRank == 0) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }

    // Build octree for spatial acceleration
    if (g_mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[static_cast<size_t>(i)] = state.triangles[static_cast<size_t>(i)].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
#pragma omp parallel for schedule(static)
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state,
                         ParallelDeviceState& device,
                         const MpiPartition& partition) {
    if (g_mpiRank == 0) printf("Computing form factors (Kij) on CUDA...\n");
    constexpr uint32_t threads = 256;
    const uint64_t work = static_cast<uint64_t>(partition.rowCount) * state.numTriangles;
    const uint32_t blocks = gridBlocksFor(work, threads);
    if (blocks > 0) {
        computeFormFactorsKernel<<<blocks, threads>>>(
            device.triangles, device.octreeNodes, device.octreeTriangles,
            device.triangleCount, asDeviceIndex(partition.rowStart, "row start"),
            device.localRowCount, device.kij);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t localElements = partition.rowCount * state.numTriangles;
    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(state.kij.data() + partition.rowStart * state.numTriangles,
                              device.kij + partition.rowStart * state.numTriangles,
                              localElements * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.kij.data(),
                   partition.counts.data(), partition.displacements.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(device.kij, state.kij.data(),
                          state.numTriangles * state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    if (g_mpiRank == 0) {
        printf("  Form-factor rows complete: %zu/%zu\n",
               state.numTriangles, state.numTriangles);
    }
}

void computeTimeDelays(SimulationState& state,
                       ParallelDeviceState& device,
                       const MpiPartition& partition) {
    if (g_mpiRank == 0) printf("Computing time delays (Tau) on CUDA...\n");
    constexpr uint32_t threads = 256;
    const uint64_t work = static_cast<uint64_t>(partition.rowCount) * state.numTriangles;
    const uint32_t blocks = gridBlocksFor(work, threads);
    if (blocks > 0) {
        computeTimeDelaysKernel<<<blocks, threads>>>(
            device.centers, device.triangleCount,
            asDeviceIndex(partition.rowStart, "row start"), device.localRowCount, device.tau);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t localElements = partition.rowCount * state.numTriangles;
    if (localElements > 0) {
        CUDA_CHECK(cudaMemcpy(state.tau.data() + partition.rowStart * state.numTriangles,
                              device.tau + partition.rowStart * state.numTriangles,
                              localElements * sizeof(int), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.tau.data(),
                   partition.counts.data(), partition.displacements.data(),
                   MPI_INT, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(device.tau, state.tau.data(),
                          state.numTriangles * state.numTriangles * sizeof(int),
                          cudaMemcpyHostToDevice));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state,
                   ParallelDeviceState& device,
                   const MpiPartition& partition) {
    if (g_mpiRank == 0) printf("Running wave propagation simulation on CUDA...\n");
    constexpr uint32_t threads = 256;
    std::vector<val_t> localRow(partition.rowCount);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (partition.rowCount > 0) {
            propagateKernel<<<static_cast<unsigned int>(partition.rowCount), threads>>>(
                static_cast<uint32_t>(t), device.timesteps, device.triangleCount,
                asDeviceIndex(partition.rowStart, "row start"), device.localRowCount,
                asDeviceIndex(state.sourceIndex, "source index"),
                state.rho.empty() ? ZERO : state.rho[0], device.areas,
                device.kij, device.tau, device.radB);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localRow.data(),
                                  device.radB + t * state.numTriangles + partition.rowStart,
                                  partition.rowCount * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(localRow.data(), partition.rowCounts[partition.rank], MPI_FLOAT,
                       state.radB.data() + t * state.numTriangles,
                       partition.rowCounts.data(), partition.rowDisplacements.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(device.radB + t * state.numTriangles,
                              state.radB.data() + t * state.numTriangles,
                              state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));

        if (g_mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state,
                      ParallelDeviceState& device,
                      const MpiPartition& partition) {
    if (g_mpiRank == 0) printf("Computing distances via CUDA cross-correlation...\n");
    constexpr uint32_t threads = 256;
    const uint64_t work = static_cast<uint64_t>(partition.rowCount) * state.numTimesteps;
    const uint32_t blocks = gridBlocksFor(work, threads);
    if (blocks > 0) {
        distanceCorrelationKernel<<<blocks, threads>>>(
            device.triangleCount, device.timesteps,
            asDeviceIndex(partition.rowStart, "row start"), device.localRowCount,
            asDeviceIndex(state.sourceIndex, "source index"), device.radB,
            device.correlations);
        CUDA_CHECK(cudaGetLastError());
    }

    if (partition.rowCount > 0) {
        const uint32_t reduceBlocks = (device.localRowCount + threads - 1) / threads;
        distanceReduceKernel<<<reduceBlocks, threads>>>(
            device.timesteps, device.localRowCount, device.correlations, device.distances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + partition.rowStart, device.distances,
                              partition.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   partition.rowCounts.data(), partition.rowDisplacements.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
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
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpiRank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (g_mpiRank == 0) {
            std::fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        }
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
            if (g_mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (g_mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Every rank owns a row block and is bound to a CUDA device.  MPI,
    // OpenMP, and CUDA are all required execution backends; there is no
    // serial fallback path.
    selectCudaDevice();

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    const MpiPartition partition = makeMpiPartition(state.numTriangles);
    ParallelDeviceState device;
    initializeParallelDevice(device, state, partition);

    if (g_mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, device, partition);
    computeFormFactors(state, device, partition);

    auto endPre = std::chrono::high_resolution_clock::now();
    long localPreDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endPre - startPre).count();
    long preDuration = 0;
    MPI_Reduce(&localPreDuration, &preDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, device, partition);

    auto endSim = std::chrono::high_resolution_clock::now();
    long localSimDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endSim - startSim).count();
    long simDuration = 0;
    MPI_Reduce(&localSimDuration, &simDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, device, partition);

    auto endDist = std::chrono::high_resolution_clock::now();
    long localDistDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endDist - startDist).count();
    long distDuration = 0;
    MPI_Reduce(&localDistDuration, &distDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (g_mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    if (g_mpiRank == 0) {
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
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
    }

    // Validation
    int valid = 1;
    if (validate && g_mpiRank == 0) {
        valid = validateResults(state) ? 1 : 0;
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    device.release();
    MPI_Finalize();

    return valid ? 0 : 1;
}
