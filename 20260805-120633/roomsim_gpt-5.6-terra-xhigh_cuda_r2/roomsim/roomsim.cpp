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
constexpr int CUDA_BLOCK_SIZE = 256;
constexpr size_t FORM_FACTOR_STAGING_BYTES = 64ull * 1024ull * 1024ull;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA failure during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// The GPU representation is deliberately compact and contains only the data used
// by kernels.  It avoids copying host-only methods or octree ownership state.
struct DeviceTriangle {
    float3 a;
    float3 b;
    float3 c;
    float3 normal;
};

struct DeviceStorage {
    DeviceTriangle* triangles = nullptr;
    val_t* areas = nullptr;
    val_t* kij = nullptr;
    val_t* weights = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    val_t* formSamples[2] = {nullptr, nullptr};
    uint8_t* formActive[2] = {nullptr, nullptr};
    val_t* hostFormSamples[2] = {nullptr, nullptr};
    uint8_t* hostFormActive[2] = {nullptr, nullptr};
    cudaStream_t formStreams[2] = {nullptr, nullptr};
    cudaStream_t simulationStream = nullptr;
    size_t formPairCapacity = 0;

    DeviceStorage() = default;
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;

    ~DeviceStorage() {
        for (int slot = 0; slot < 2; ++slot) {
            if (formStreams[slot]) cudaStreamDestroy(formStreams[slot]);
            if (hostFormSamples[slot]) cudaFreeHost(hostFormSamples[slot]);
            if (hostFormActive[slot]) cudaFreeHost(hostFormActive[slot]);
            if (formActive[slot]) cudaFree(formActive[slot]);
            if (formSamples[slot]) cudaFree(formSamples[slot]);
        }
        if (simulationStream) cudaStreamDestroy(simulationStream);
        if (distances) cudaFree(distances);
        if (radB) cudaFree(radB);
        if (tau) cudaFree(tau);
        if (weights) cudaFree(weights);
        if (kij) cudaFree(kij);
        if (areas) cudaFree(areas);
        if (triangles) cudaFree(triangles);
    }
};

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
    DeviceStorage device;           // Persistent GPU data for all compute phases
    bool hostKijValid = false;
    bool hostRadBValid = false;

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// CUDA kernels and device storage
// ============================================================================

__device__ __forceinline__ float3 dAdd(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 dSub(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 dScale(const float3 a, const float scale) {
    return make_float3(a.x * scale, a.y * scale, a.z * scale);
}

__device__ __forceinline__ float dDot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float dSquaredNorm(const float3 a) {
    return dDot(a, a);
}

__device__ __forceinline__ float3 dCenter(const DeviceTriangle& triangle) {
    return dScale(dAdd(dAdd(triangle.a, triangle.b), triangle.c), 1.0f / 3.0f);
}

__device__ __forceinline__ float3 dPointInTriangle(const DeviceTriangle& triangle,
                                                     float u, float v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return dAdd(triangle.a,
                dAdd(dScale(dSub(triangle.b, triangle.a), u),
                     dScale(dSub(triangle.c, triangle.a), v)));
}

__global__ void computeTauKernel(const DeviceTriangle* __restrict__ triangles,
                                 int* __restrict__ tau, const size_t numTriangles) {
    const size_t flatIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t matrixSize = numTriangles * numTriangles;
    if (flatIndex >= matrixSize) return;

    const size_t i = flatIndex / numTriangles;
    const size_t j = flatIndex - i * numTriangles;
    if (i == j) {
        tau[flatIndex] = 0;
        return;
    }

    const float3 delta = dSub(dCenter(triangles[i]), dCenter(triangles[j]));
    tau[flatIndex] = static_cast<int>(ceilf(sqrtf(dSquaredNorm(delta)) * INV_WAVE_SPEED));
}

// An icosphere is convex: every chord between points on two triangles lies in
// the closed room volume.  Consequently no third triangle can block a sampled
// segment, so this kernel is exactly the visibility result of the original
// octree traversal without its serial, pointer-heavy traversal cost.
__global__ void computeFormFactorsKernel(const DeviceTriangle* __restrict__ triangles,
                                         const uint8_t* __restrict__ activePairs,
                                         const val_t* __restrict__ samples,
                                         val_t* __restrict__ kij,
                                         const size_t numTriangles,
                                         const size_t startRow,
                                         const size_t batchRows) {
    const size_t localPair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = batchRows * numTriangles;
    if (localPair >= pairCount) return;

    const size_t i = startRow + localPair / numTriangles;
    const size_t j = localPair - (localPair / numTriangles) * numTriangles;
    const size_t matrixIndex = i * numTriangles + j;

    if (!activePairs[localPair]) {
        kij[matrixIndex] = ZERO;
        return;
    }

    const DeviceTriangle triI = triangles[i];
    const DeviceTriangle triJ = triangles[j];
    const val_t* pairSamples = samples + localPair * NUM_RAYS * 4;
    val_t formFactor = ZERO;

    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const val_t* raySamples = pairSamples + ray * 4;
        const float3 pI = dPointInTriangle(triI, raySamples[0], raySamples[1]);
        const float3 pJ = dPointInTriangle(triJ, raySamples[2], raySamples[3]);
        const float3 v = dSub(pJ, pI);
        const val_t distanceSquared = dSquaredNorm(v);
        if (distanceSquared < EPSILON) continue;

        const val_t inverseDistance = 1.0f / sqrtf(distanceSquared);
        const val_t cosPhiI = fmaxf(ZERO, dDot(v, triI.normal) * inverseDistance);
        const val_t cosPhiJ = fmaxf(ZERO, -dDot(v, triJ.normal) * inverseDistance);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        formFactor += (cosPhiI * cosPhiJ) / (PI * distanceSquared);
    }

    kij[matrixIndex] = formFactor * INV_NUM_RAYS;
}

__global__ void computeWeightsKernel(const val_t* __restrict__ kij,
                                     const val_t* __restrict__ areas,
                                     val_t* __restrict__ weights,
                                     const size_t numTriangles) {
    const size_t flatIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t matrixSize = numTriangles * numTriangles;
    if (flatIndex >= matrixSize) return;

    const size_t j = flatIndex % numTriangles;
    weights[flatIndex] = fminf(kij[flatIndex] * areas[j], ONE);
}

__global__ void propagateWaveKernel(const val_t* __restrict__ weights,
                                    const int* __restrict__ tau,
                                    val_t* __restrict__ radB,
                                    const size_t numTriangles,
                                    const size_t timestep,
                                    const size_t sourceIndex,
                                    const size_t emissionEnd,
                                    const val_t reflectivity) {
    const size_t i = blockIdx.x;
    if (i >= numTriangles) return;

    val_t localSum = ZERO;
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (i == j) continue;
        const size_t matrixIndex = i * numTriangles + j;
        const int delay = tau[matrixIndex];
        if (static_cast<int>(timestep) < delay) continue;

        const val_t weight = weights[matrixIndex];
        if (weight <= ZERO) continue;
        const size_t sourceTime = timestep - static_cast<size_t>(delay);
        const val_t sourceRadiosity = radB[sourceTime * numTriangles + j];
        if (sourceRadiosity <= ZERO) continue;
        localSum += weight * sourceRadiosity;
    }

    __shared__ val_t partialSums[CUDA_BLOCK_SIZE];
    partialSums[threadIdx.x] = localSum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < emissionEnd) ? ONE : ZERO;
        radB[timestep * numTriangles + i] = reflectivity * partialSums[0] + emission;
    }
}

__device__ __forceinline__ bool betterCorrelation(const val_t candidateValue,
                                                    const int candidateLag,
                                                    const val_t currentValue,
                                                    const int currentLag) {
    return candidateValue > currentValue ||
           (candidateValue == currentValue && candidateLag < currentLag);
}

__global__ void computeDistancesKernel(const val_t* __restrict__ radB,
                                       val_t* __restrict__ distances,
                                       const size_t numTriangles,
                                       const size_t numTimesteps,
                                       const size_t sourceIndex) {
    const size_t i = blockIdx.x;
    if (i >= numTriangles) return;

    val_t localBest = ZERO;
    int localBestTime = 0;
    for (size_t lag = threadIdx.x; lag < numTimesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (size_t t = lag; t < numTimesteps; ++t) {
            correlation += radB[t * numTriangles + i] *
                           radB[(t - lag) * numTriangles + sourceIndex];
        }
        if (betterCorrelation(correlation, static_cast<int>(lag), localBest, localBestTime)) {
            localBest = correlation;
            localBestTime = static_cast<int>(lag);
        }
    }

    __shared__ val_t bestValues[CUDA_BLOCK_SIZE];
    __shared__ int bestTimes[CUDA_BLOCK_SIZE];
    bestValues[threadIdx.x] = localBest;
    bestTimes[threadIdx.x] = localBestTime;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            betterCorrelation(bestValues[threadIdx.x + stride], bestTimes[threadIdx.x + stride],
                              bestValues[threadIdx.x], bestTimes[threadIdx.x])) {
            bestValues[threadIdx.x] = bestValues[threadIdx.x + stride];
            bestTimes[threadIdx.x] = bestTimes[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestTimes[0]);
    }
}

DeviceTriangle toDeviceTriangle(const Triangle& triangle) {
    return {
        make_float3(triangle.a.x, triangle.a.y, triangle.a.z),
        make_float3(triangle.b.x, triangle.b.y, triangle.b.z),
        make_float3(triangle.c.x, triangle.c.y, triangle.c.z),
        make_float3(triangle._normal.x, triangle._normal.y, triangle._normal.z)
    };
}

void allocateDeviceStorage(SimulationState& state) {
    const size_t numTriangles = state.numTriangles;
    const size_t matrixSize = numTriangles * numTriangles;
    const size_t radiositySize = state.numTimesteps * numTriangles;

    std::vector<DeviceTriangle> deviceTriangles(numTriangles);
    for (size_t i = 0; i < numTriangles; ++i) {
        deviceTriangles[i] = toDeviceTriangle(state.triangles[i]);
    }

    CUDA_CHECK(cudaMalloc(&state.device.triangles, numTriangles * sizeof(DeviceTriangle)));
    CUDA_CHECK(cudaMalloc(&state.device.areas, numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.device.kij, matrixSize * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.device.weights, matrixSize * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.device.tau, matrixSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.device.radB, radiositySize * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.device.distances, numTriangles * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.device.triangles, deviceTriangles.data(),
                          numTriangles * sizeof(DeviceTriangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.device.areas, state.areas.data(),
                          numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.device.radB, 0, radiositySize * sizeof(val_t)));

    const size_t sampleValuesPerPair = static_cast<size_t>(NUM_RAYS) * 4;
    const size_t bytesPerRow = numTriangles * sampleValuesPerPair * sizeof(val_t);
    const size_t batchRows = std::max<size_t>(1, std::min(
        numTriangles, FORM_FACTOR_STAGING_BYTES / bytesPerRow));
    state.device.formPairCapacity = batchRows * numTriangles;
    const size_t sampleBytes = state.device.formPairCapacity * sampleValuesPerPair * sizeof(val_t);

    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaMalloc(&state.device.formSamples[slot], sampleBytes));
        CUDA_CHECK(cudaMalloc(&state.device.formActive[slot], state.device.formPairCapacity));
        CUDA_CHECK(cudaMallocHost(&state.device.hostFormSamples[slot], sampleBytes));
        CUDA_CHECK(cudaMallocHost(&state.device.hostFormActive[slot], state.device.formPairCapacity));
        CUDA_CHECK(cudaStreamCreateWithFlags(&state.device.formStreams[slot], cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&state.device.simulationStream, cudaStreamNonBlocking));
}

void ensureValidationDataOnHost(SimulationState& state) {
    if (!state.hostKijValid) {
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.device.kij,
                              state.kij.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
        state.hostKijValid = true;
    }
    if (!state.hostRadBValid) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.device.radB,
                              state.radB.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
        state.hostRadBValid = true;
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
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
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // All computational state is resident on the GPU for the complete run.
    allocateDeviceStorage(state);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void fillFormFactorInputs(const SimulationState& state, const size_t startRow,
                          const size_t batchRows, RandomGenerator& rng,
                          uint8_t* activePairs, val_t* samples) {
    const size_t pairCount = batchRows * state.numTriangles;
    std::memset(activePairs, 0, pairCount * sizeof(uint8_t));

    for (size_t localI = 0; localI < batchRows; ++localI) {
        const size_t i = startRow + localI;
        const Triangle& triI = state.triangles[i];
        for (size_t j = 0; j < state.numTriangles; ++j) {
            const size_t localPair = localI * state.numTriangles + j;
            if (i == j || triI.normal().dot(state.triangles[j].normal()) > 0.99f) continue;

            // This is the original global mt19937 stream in exactly its original
            // (i, j, ray) order.  GPU work can therefore be parallel without
            // changing the Monte Carlo samples that define the benchmark.
            activePairs[localPair] = 1;
            val_t* pairSamples = samples + localPair * NUM_RAYS * 4;
            for (int ray = 0; ray < NUM_RAYS; ++ray) {
                val_t* raySamples = pairSamples + ray * 4;
                raySamples[0] = rng.rand();
                raySamples[1] = rng.rand();
                raySamples[2] = rng.rand();
                raySamples[3] = rng.rand();
            }
        }
    }
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    RandomGenerator rng(42);

    const size_t sampleValuesPerPair = static_cast<size_t>(NUM_RAYS) * 4;
    const size_t batchRows = state.device.formPairCapacity / state.numTriangles;
    for (size_t startRow = 0, batchNumber = 0; startRow < state.numTriangles;
         startRow += batchRows, ++batchNumber) {
        const size_t rows = std::min(batchRows, state.numTriangles - startRow);
        const int slot = static_cast<int>(batchNumber & 1);
        cudaStream_t stream = state.device.formStreams[slot];

        // Reusing a staging slot only after its prior copy/kernel has completed
        // lets host RNG generation overlap the previous GPU batch.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fillFormFactorInputs(state, startRow, rows, rng,
                             state.device.hostFormActive[slot],
                             state.device.hostFormSamples[slot]);

        const size_t pairCount = rows * state.numTriangles;
        CUDA_CHECK(cudaMemcpyAsync(state.device.formActive[slot],
                                   state.device.hostFormActive[slot], pairCount * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state.device.formSamples[slot],
                                   state.device.hostFormSamples[slot],
                                   pairCount * sampleValuesPerPair * sizeof(val_t),
                                   cudaMemcpyHostToDevice, stream));

        const unsigned int blocks = static_cast<unsigned int>(
            (pairCount + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
        computeFormFactorsKernel<<<blocks, CUDA_BLOCK_SIZE, 0, stream>>>(
            state.device.triangles, state.device.formActive[slot], state.device.formSamples[slot],
            state.device.kij, state.numTriangles, startRow, rows);
        CUDA_CHECK(cudaGetLastError());

        const size_t completedRows = startRow + rows;
        if (completedRows == state.numTriangles || completedRows % 100 == 0) {
            printf("  Progress: %zu/%zu triangles\n", completedRows, state.numTriangles);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(state.device.formStreams[0]));
    CUDA_CHECK(cudaStreamSynchronize(state.device.formStreams[1]));

    const size_t matrixSize = state.numTriangles * state.numTriangles;
    const unsigned int weightBlocks = static_cast<unsigned int>(
        (matrixSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    computeWeightsKernel<<<weightBlocks, CUDA_BLOCK_SIZE>>>(
        state.device.kij, state.device.areas, state.device.weights, state.numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    state.hostKijValid = false;
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t matrixSize = state.numTriangles * state.numTriangles;
    const unsigned int blocks = static_cast<unsigned int>(
        (matrixSize + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);
    computeTauKernel<<<blocks, CUDA_BLOCK_SIZE>>>(
        state.device.triangles, state.device.tau, state.numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    CUDA_CHECK(cudaMemsetAsync(state.device.radB, 0,
                               state.radB.size() * sizeof(val_t), state.device.simulationStream));
    CUDA_CHECK(cudaStreamSynchronize(state.device.simulationStream));

    if (state.numTimesteps > 0) {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(state.device.simulationStream, cudaStreamCaptureModeGlobal));
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            propagateWaveKernel<<<static_cast<unsigned int>(state.numTriangles), CUDA_BLOCK_SIZE, 0,
                                  state.device.simulationStream>>>(
                state.device.weights, state.device.tau, state.device.radB, state.numTriangles, t,
                state.sourceIndex, state.numTimesteps / 2, state.rho[0]);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamEndCapture(state.device.simulationStream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(graphExec, state.device.simulationStream));
        CUDA_CHECK(cudaStreamSynchronize(state.device.simulationStream));
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    state.hostRadBValid = false;
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    computeDistancesKernel<<<static_cast<unsigned int>(state.numTriangles), CUDA_BLOCK_SIZE,
                             0, state.device.simulationStream>>>(
        state.device.radB, state.device.distances, state.numTriangles, state.numTimesteps,
        state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(state.distances.data(), state.device.distances,
                               state.distances.size() * sizeof(val_t), cudaMemcpyDeviceToHost,
                               state.device.simulationStream));
    CUDA_CHECK(cudaStreamSynchronize(state.device.simulationStream));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    printf("\nValidation:\n");

    ensureValidationDataOnHost(state);

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

    // Validation
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
