/**
 * Room Response Simulation Benchmark
 * 
 * This CUDA implementation performs room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light
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
#include <cinttypes>
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

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;
using delay_t = uint8_t;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr val_t ROOM_RADIUS = 10.0f;
constexpr int MAX_TIME_DELAY = 40;        // ceil(2 * ROOM_RADIUS / WAVE_SPEED)
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

static_assert(MAX_TIME_DELAY <= std::numeric_limits<delay_t>::max(),
              "delay_t must represent every possible room chord delay");

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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
// CUDA data layout, random sampling, and kernels
// ============================================================================

namespace cuda_detail {

constexpr int PAIR_TILE = 16;
constexpr int REDUCTION_THREADS = 256;
constexpr val_t INV_PI = 1.0f / PI;

// Four aligned float4 loads provide all geometry for one triangle.  The first
// w component stores area, which is consumed while form factors are generated.
struct DeviceTriangle {
    float4 a;
    float4 b;
    float4 c;
    float4 normal;
};

struct Uint4 {
    uint32_t x, y, z, w;
};

__device__ __forceinline__ uint32_t mulHi(uint32_t a, uint32_t b) {
    return __umulhi(a, b);
}

// Counter-based Philox4x32-10 gives every unordered triangle/ray pair its own
// deterministic random values without serialized RNG state or a giant random
// number array.  This preserves the original Monte Carlo sampling semantics and
// makes results independent of CUDA scheduling.
__device__ __forceinline__ Uint4 philox(uint64_t pair, uint32_t ray) {
    Uint4 counter{static_cast<uint32_t>(pair), static_cast<uint32_t>(pair >> 32), ray, 0u};
    uint32_t key0 = 42u;
    uint32_t key1 = 0xdecafbadU;

#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const uint32_t hi0 = mulHi(0xD2511F53U, counter.x);
        const uint32_t lo0 = 0xD2511F53U * counter.x;
        const uint32_t hi1 = mulHi(0xCD9E8D57U, counter.z);
        const uint32_t lo1 = 0xCD9E8D57U * counter.z;
        counter = {hi1 ^ counter.y ^ key0, lo1,
                   hi0 ^ counter.w ^ key1, lo0};
        key0 += 0x9E3779B9U;
        key1 += 0xBB67AE85U;
    }
    return counter;
}

__device__ __forceinline__ val_t uniform01(uint32_t bits) {
    return static_cast<val_t>(bits) * 0x1.0p-32f;
}

__device__ __forceinline__ float3 pointInTriangle(
        const DeviceTriangle& triangle, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return make_float3(
        triangle.a.x + (triangle.b.x - triangle.a.x) * u +
                           (triangle.c.x - triangle.a.x) * v,
        triangle.a.y + (triangle.b.y - triangle.a.y) * u +
                           (triangle.c.y - triangle.a.y) * v,
        triangle.a.z + (triangle.b.z - triangle.a.z) * u +
                           (triangle.c.z - triangle.a.z) * v);
}

__device__ __forceinline__ val_t dot3(const float3& a, const float3& b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ __forceinline__ float3 centerOf(const DeviceTriangle& triangle) {
    constexpr val_t oneThird = 1.0f / 3.0f;
    return make_float3((triangle.a.x + triangle.b.x + triangle.c.x) * oneThird,
                       (triangle.a.y + triangle.b.y + triangle.c.y) * oneThird,
                       (triangle.a.z + triangle.b.z + triangle.c.z) * oneThird);
}

__global__ void timeDelayKernel(const DeviceTriangle* __restrict__ triangles,
                                delay_t* __restrict__ tau, int n) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t entries = static_cast<uint64_t>(n) * n;
    if (index >= entries) return;

    const int i = static_cast<int>(index / n);
    const int j = static_cast<int>(index - static_cast<uint64_t>(i) * n);
    if (i == j) {
        tau[index] = 0;
        return;
    }

    const float3 ci = centerOf(triangles[i]);
    const float3 cj = centerOf(triangles[j]);
    const val_t dx = ci.x - cj.x;
    const val_t dy = ci.y - cj.y;
    const val_t dz = ci.z - cj.z;
    tau[index] = static_cast<delay_t>(
        ceilf(sqrtf(fmaf(dx, dx, fmaf(dy, dy, dz * dz))) * INV_WAVE_SPEED));
}

__global__ void formFactorKernel(const DeviceTriangle* __restrict__ triangles,
                                 val_t* __restrict__ weights, int n) {
    // Only the upper block triangle is launched into useful work.  Kij is a
    // reciprocal geometric integral, so one shared sample set computes both
    // directions while the emitter-area weighting is applied independently.
    if (blockIdx.y > blockIdx.x) return;

    const int i = static_cast<int>(blockIdx.y) * PAIR_TILE + threadIdx.y;
    const int j = static_cast<int>(blockIdx.x) * PAIR_TILE + threadIdx.x;
    if (i >= n || j >= n) return;

    const uint64_t ij = static_cast<uint64_t>(i) * n + j;
    if (i == j) {
        weights[ij] = ZERO;
        return;
    }
    if (i > j) return;

    const DeviceTriangle triangleI = triangles[i];
    const DeviceTriangle triangleJ = triangles[j];
    const float3 normalI = make_float3(triangleI.normal.x, triangleI.normal.y,
                                       triangleI.normal.z);
    const float3 normalJ = make_float3(triangleJ.normal.x, triangleJ.normal.y,
                                       triangleJ.normal.z);
    const uint64_t ji = static_cast<uint64_t>(j) * n + i;

    if (dot3(normalI, normalJ) > 0.99f) {
        weights[ij] = ZERO;
        weights[ji] = ZERO;
        return;
    }

    val_t kij = ZERO;
#pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const Uint4 random = philox(ij, static_cast<uint32_t>(ray));
        const float3 pointI = pointInTriangle(
            triangleI, uniform01(random.x), uniform01(random.y));
        const float3 pointJ = pointInTriangle(
            triangleJ, uniform01(random.z), uniform01(random.w));
        const float3 direction = make_float3(pointJ.x - pointI.x,
                                             pointJ.y - pointI.y,
                                             pointJ.z - pointI.z);
        const val_t distanceSquared = dot3(direction, direction);
        if (distanceSquared < EPSILON) continue;

        const val_t inverseDistance = rsqrtf(distanceSquared);
        const val_t cosineI = fmaxf(ZERO, dot3(direction, normalI) * inverseDistance);
        const val_t cosineJ = fmaxf(
            ZERO, -dot3(direction, normalJ) * inverseDistance);
        kij = fmaf(cosineI * cosineJ, INV_PI / distanceSquared, kij);
    }
    kij *= INV_NUM_RAYS;

    // The generated icosphere is a convex closed polyhedron.  A chord between
    // points on two faces lies in its interior, so (apart from its filtered
    // endpoints) it cannot intersect a third face.  This is exactly the result
    // of the original octree visibility walk, with no divergent traversal.
    weights[ij] = fminf(kij * triangleJ.a.w, ONE);
    weights[ji] = fminf(kij * triangleI.a.w, ONE);
}

__device__ __forceinline__ val_t warpSum(val_t value) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void wavePropagationKernel(const val_t* __restrict__ weights,
                                      const delay_t* __restrict__ tau,
                                      val_t* __restrict__ radB,
                                      int n, int timestep, int source,
                                      int timeOff, val_t reflectivity) {
    const int receiver = static_cast<int>(blockIdx.x);
    if (receiver >= n) return;

    val_t partial = ZERO;
    const uint64_t row = static_cast<uint64_t>(receiver) * n;
    for (int emitter = threadIdx.x; emitter < n; emitter += blockDim.x) {
        if (receiver == emitter) continue;
        const int delay = tau[row + emitter];
        if (timestep < delay) continue;

        const val_t weight = weights[row + emitter];
        if (weight <= ZERO) continue;
        const val_t emitterRadiosity =
            radB[static_cast<uint64_t>(timestep - delay) * n + emitter];
        if (emitterRadiosity > ZERO) {
            partial = fmaf(weight, emitterRadiosity, partial);
        }
    }

    partial = warpSum(partial);
    __shared__ val_t warpSums[REDUCTION_THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = partial;
    __syncthreads();

    if (warp == 0) {
        val_t total = lane < REDUCTION_THREADS / 32 ? warpSums[lane] : ZERO;
        total = warpSum(total);
        if (lane == 0) {
            const val_t emission =
                (receiver == source && timestep < timeOff) ? ONE : ZERO;
            radB[static_cast<uint64_t>(timestep) * n + receiver] =
                fmaf(reflectivity, total, emission);
        }
    }
}

__device__ __forceinline__ void selectBetter(val_t& correlation, int& delay,
                                             val_t otherCorrelation, int otherDelay) {
    if (otherCorrelation > correlation ||
        (otherCorrelation == correlation && otherDelay < delay)) {
        correlation = otherCorrelation;
        delay = otherDelay;
    }
}

__global__ void distanceKernel(const val_t* __restrict__ radB,
                               val_t* __restrict__ distances,
                               int n, int timesteps, int source,
                               bool cacheSignals) {
    const int receiver = static_cast<int>(blockIdx.x);
    if (receiver >= n) return;

    extern __shared__ val_t signals[];
    val_t* receiverSignal = signals;
    val_t* sourceSignal = signals + timesteps;
    if (cacheSignals) {
        for (int t = threadIdx.x; t < timesteps; t += blockDim.x) {
            receiverSignal[t] = radB[static_cast<uint64_t>(t) * n + receiver];
            sourceSignal[t] = radB[static_cast<uint64_t>(t) * n + source];
        }
        __syncthreads();
    }

    val_t bestCorrelation = ZERO;
    int bestDelay = 0;
    for (int delay = threadIdx.x; delay < timesteps; delay += blockDim.x) {
        val_t correlation = ZERO;
        for (int t = delay; t < timesteps; ++t) {
            const val_t receiverValue = cacheSignals
                ? receiverSignal[t]
                : radB[static_cast<uint64_t>(t) * n + receiver];
            const val_t sourceValue = cacheSignals
                ? sourceSignal[t - delay]
                : radB[static_cast<uint64_t>(t - delay) * n + source];
            correlation = fmaf(sourceValue, receiverValue, correlation);
        }
        selectBetter(bestCorrelation, bestDelay, correlation, delay);
    }

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        const val_t otherCorrelation =
            __shfl_down_sync(0xffffffffU, bestCorrelation, offset);
        const int otherDelay = __shfl_down_sync(0xffffffffU, bestDelay, offset);
        selectBetter(bestCorrelation, bestDelay, otherCorrelation, otherDelay);
    }

    __shared__ val_t warpCorrelations[REDUCTION_THREADS / 32];
    __shared__ int warpDelays[REDUCTION_THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warpCorrelations[warp] = bestCorrelation;
        warpDelays[warp] = bestDelay;
    }
    __syncthreads();

    if (warp == 0) {
        bestCorrelation = lane < REDUCTION_THREADS / 32
            ? warpCorrelations[lane] : ZERO;
        bestDelay = lane < REDUCTION_THREADS / 32 ? warpDelays[lane] : 0;
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            const val_t otherCorrelation =
                __shfl_down_sync(0xffffffffU, bestCorrelation, offset);
            const int otherDelay = __shfl_down_sync(0xffffffffU, bestDelay, offset);
            selectBetter(bestCorrelation, bestDelay, otherCorrelation, otherDelay);
        }
        if (lane == 0) {
            distances[receiver] = WAVE_SPEED * static_cast<val_t>(bestDelay);
        }
    }
}

__global__ void countNonZeroKernel(const val_t* __restrict__ values,
                                   uint64_t entries,
                                   unsigned long long* count) {
    uint64_t local = 0;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < entries; index += stride) {
        local += values[index] > EPSILON;
    }

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local += __shfl_down_sync(0xffffffffU, local, offset);
    }
    __shared__ uint64_t warpCounts[REDUCTION_THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warpCounts[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < REDUCTION_THREADS / 32 ? warpCounts[lane] : 0;
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local += __shfl_down_sync(0xffffffffU, local, offset);
        }
        if (lane == 0) atomicAdd(count, static_cast<unsigned long long>(local));
    }
}

}  // namespace cuda_detail

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cudaCheckError = (expression);                          \
        if (cudaCheckError != cudaSuccess) {                                      \
            cudaFailure(cudaCheckError, #expression, __FILE__, __LINE__);         \
        }                                                                        \
    } while (false)

struct DeviceBuffers {
    cuda_detail::DeviceTriangle* triangles = nullptr;
    val_t* weights = nullptr;
    delay_t* tau = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    DeviceBuffers() = default;
    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        if (distances) cudaFree(distances);
        if (radB) cudaFree(radB);
        if (tau) cudaFree(tau);
        if (weights) cudaFree(weights);
        if (triangles) cudaFree(triangles);
    }
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    DeviceBuffers device;

    size_t sourceIndex;
    val_t reflectivity;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    int deviceIndex = 0;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex));
    printf("Using CUDA device %d: %s (%d SMs)\n",
           deviceIndex, properties.name, properties.multiProcessorCount);

    // Force lazy CUDA module loading into initialization, alongside the original
    // mesh setup, so phase timings contain kernel work rather than driver setup.
    cudaFuncAttributes kernelAttributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes, cuda_detail::timeDelayKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes, cuda_detail::formFactorKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes,
                                     cuda_detail::wavePropagationKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes, cuda_detail::distanceKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&kernelAttributes,
                                     cuda_detail::countNonZeroKernel));

    const size_t matrixEntries = state.numTriangles * state.numTriangles;
    const size_t radiosityEntries = timesteps * state.numTriangles;
    std::vector<cuda_detail::DeviceTriangle> deviceTriangles(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const Triangle& triangle = state.triangles[i];
        deviceTriangles[i] = {
            make_float4(triangle.a.x, triangle.a.y, triangle.a.z, state.areas[i]),
            make_float4(triangle.b.x, triangle.b.y, triangle.b.z, ZERO),
            make_float4(triangle.c.x, triangle.c.y, triangle.c.z, ZERO),
            make_float4(triangle._normal.x, triangle._normal.y, triangle._normal.z, ZERO)};
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.device.triangles),
                          state.numTriangles * sizeof(cuda_detail::DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.device.weights),
                          matrixEntries * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.device.tau),
                          matrixEntries * sizeof(delay_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.device.radB),
                          radiosityEntries * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.device.distances),
                          state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.device.triangles, deviceTriangles.data(),
                          state.numTriangles * sizeof(cuda_detail::DeviceTriangle),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.device.radB, 0, radiosityEntries * sizeof(val_t)));

    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const dim3 block(cuda_detail::PAIR_TILE, cuda_detail::PAIR_TILE);
    const dim3 grid((n + cuda_detail::PAIR_TILE - 1) / cuda_detail::PAIR_TILE,
                    (n + cuda_detail::PAIR_TILE - 1) / cuda_detail::PAIR_TILE);
    cuda_detail::formFactorKernel<<<grid, block>>>(
        state.device.triangles, state.device.weights, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed %zu triangle pairs on GPU\n",
           state.numTriangles * state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const uint64_t entries = state.numTriangles * state.numTriangles;
    constexpr int threads = cuda_detail::REDUCTION_THREADS;
    const unsigned int blocks = static_cast<unsigned int>((entries + threads - 1) / threads);
    cuda_detail::timeDelayKernel<<<blocks, threads>>>(
        state.device.triangles, state.device.tau,
        static_cast<int>(state.numTriangles));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int timesteps = static_cast<int>(state.numTimesteps);
    for (int t = 0; t < timesteps; ++t) {
        cuda_detail::wavePropagationKernel<<<n, cuda_detail::REDUCTION_THREADS>>>(
            state.device.weights, state.device.tau, state.device.radB,
            n, t, static_cast<int>(state.sourceIndex), timesteps / 2,
            state.reflectivity);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Completed %zu/%zu timesteps on GPU\n",
           state.numTimesteps, state.numTimesteps);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int timesteps = static_cast<int>(state.numTimesteps);
    int deviceIndex = 0;
    int sharedMemoryLimit = 0;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemoryLimit,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      deviceIndex));
    const size_t signalBytes = 2 * state.numTimesteps * sizeof(val_t);
    const bool cacheSignals = signalBytes <= static_cast<size_t>(sharedMemoryLimit);
    const size_t sharedBytes = cacheSignals ? signalBytes : 0;
    cuda_detail::distanceKernel<<<n, cuda_detail::REDUCTION_THREADS, sharedBytes>>>(
        state.device.radB, state.device.distances, n, timesteps,
        static_cast<int>(state.sourceIndex), cacheSignals);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.device.distances,
                          state.numTriangles * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
    state.radB.resize(state.numTimesteps * state.numTriangles);
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.device.radB,
                          state.radB.size() * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

uint64_t countNonZeroFormFactors(const SimulationState& state) {
    unsigned long long* deviceCount = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCount), sizeof(*deviceCount)));
    CUDA_CHECK(cudaMemset(deviceCount, 0, sizeof(*deviceCount)));
    const uint64_t entries = state.numTriangles * state.numTriangles;
    constexpr uint64_t threads = cuda_detail::REDUCTION_THREADS;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<uint64_t>((entries + threads - 1) / threads, 65535));
    cuda_detail::countNonZeroKernel<<<blocks, threads>>>(
        state.device.weights, entries, deviceCount);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long hostCount = 0;
    CUDA_CHECK(cudaMemcpy(&hostCount, deviceCount, sizeof(hostCount),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceCount));
    return static_cast<uint64_t>(hostCount);
}

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
    const uint64_t nonZeroKij = countNonZeroFormFactors(state);
    const size_t matrixEntries = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           nonZeroKij, matrixEntries,
           100.0 * static_cast<double>(nonZeroKij) /
               static_cast<double>(matrixEntries));

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (targetTriangles <= 0 || timesteps <= 0 || sourceIdx < 0) {
        printf("Triangle count and timesteps must be positive; source index must be non-negative.\n");
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

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
    size_t memTau = n * n * sizeof(delay_t);
    size_t memRad = t * n * sizeof(val_t);
    size_t memGeometry = n * sizeof(cuda_detail::DeviceTriangle);
    size_t totalMem = memKij + memTau + memRad + memGeometry + n * sizeof(val_t);
    printf("  GPU memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016" PRIX64 "\n", hash);
    printf("\n");
    
    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
