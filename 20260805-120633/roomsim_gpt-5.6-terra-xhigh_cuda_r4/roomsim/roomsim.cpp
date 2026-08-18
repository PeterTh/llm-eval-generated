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

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HOST_DEVICE constexpr Vec3() : x(0), y(0), z(0) {}
    HOST_DEVICE constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HOST_DEVICE explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

// The host octree is convenient to construct, while the GPU needs a compact,
// pointer-free representation.  Leaves reference the same (possibly repeated)
// triangle-index lists as the host tree, so traversal has identical visibility
// semantics without any device-side allocation.
struct DeviceOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    uint32_t children[8];
    uint32_t triangleStart;
    uint32_t triangleCount;
};

struct FlatOctree {
    std::vector<DeviceOctreeNode> nodes;
    std::vector<uint32_t> triangleIndices;
};

uint32_t flattenOctreeNode(const Octree& source, FlatOctree& destination) {
    const uint32_t nodeIndex = static_cast<uint32_t>(destination.nodes.size());
    DeviceOctreeNode node{};
    node.center = source.center;
    node.halfExtent = source.halfExtent;
    std::fill(std::begin(node.children), std::end(node.children), std::numeric_limits<uint32_t>::max());

    // Reserve the slot before descending because child indices refer to this
    // parent node.  Reallocation would invalidate a reference, hence assign it
    // again once all children have been flattened.
    destination.nodes.push_back(node);
    if (!source.triangleIndices.empty()) {
        node.triangleStart = static_cast<uint32_t>(destination.triangleIndices.size());
        node.triangleCount = static_cast<uint32_t>(source.triangleIndices.size());
        for (size_t triangleIndex : source.triangleIndices) {
            destination.triangleIndices.push_back(static_cast<uint32_t>(triangleIndex));
        }
    } else {
        for (int child = 0; child < 8; ++child) {
            if (source.children[child]) {
                node.children[child] = flattenOctreeNode(*source.children[child], destination);
            }
        }
    }
    destination.nodes[nodeIndex] = node;
    return nodeIndex;
}

FlatOctree flattenOctree(const Octree& source) {
    FlatOctree result;
    flattenOctreeNode(source, result);
    return result;
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

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Dense simulation arrays reside on the GPU.  Host copies are materialized
    // only for validation, keeping the normal benchmark path device-resident.
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void cudaCheck(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

struct DeviceTriangle {
    Vec3 a;
    Vec3 b;
    Vec3 c;
    Vec3 normal;
};

__device__ __forceinline__ val_t deviceDot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 deviceSubtract(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 deviceCross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ val_t deviceSquaredNorm(const Vec3& v) {
    return deviceDot(v, v);
}

__device__ __forceinline__ bool deviceRayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                                        const DeviceOctreeNode& node) {
    const Vec3 d((p2.x - p1.x) * 0.5f, (p2.y - p1.y) * 0.5f, (p2.z - p1.z) * 0.5f);
    const Vec3 c(p1.x + d.x - node.center.x, p1.y + d.y - node.center.y,
                 p1.z + d.z - node.center.z);
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ bool deviceSegmentIntersectsTriangle(const Vec3& origin,
                                                                  const Vec3& direction,
                                                                  val_t rayLength,
                                                                  const DeviceTriangle& triangle) {
    const Vec3 e1 = deviceSubtract(triangle.b, triangle.a);
    const Vec3 e2 = deviceSubtract(triangle.c, triangle.a);
    const Vec3 pvec = deviceCross(direction, e2);
    const val_t determinant = deviceDot(e1, pvec);
    if (fabsf(determinant) < EPSILON) return false;

    const val_t inverseDeterminant = 1.0f / determinant;
    const Vec3 tvec = deviceSubtract(origin, triangle.a);
    const val_t u = deviceDot(tvec, pvec) * inverseDeterminant;
    if (u < ZERO || u > ONE) return false;

    const Vec3 qvec = deviceCross(tvec, e1);
    const val_t v = deviceDot(direction, qvec) * inverseDeterminant;
    if (v < ZERO || u + v > ONE) return false;

    const val_t distance = deviceDot(e2, qvec) * inverseDeterminant;
    return distance > EPSILON && distance < rayLength - EPSILON;
}

__device__ bool deviceRayBlockedBruteForce(const Vec3& from, const Vec3& direction,
                                           val_t rayLength, const DeviceTriangle* triangles,
                                           uint32_t triangleCount, uint32_t sourceIndex,
                                           uint32_t destinationIndex) {
    for (uint32_t triangleIndex = 0; triangleIndex < triangleCount; ++triangleIndex) {
        if (triangleIndex != sourceIndex && triangleIndex != destinationIndex &&
            deviceSegmentIntersectsTriangle(from, direction, rayLength, triangles[triangleIndex])) {
            return true;
        }
    }
    return false;
}

// The generated icospheres are shallow, but retain a conservative stack and a
// brute-force overflow path so no visibility test can be silently dropped.
__device__ bool deviceRayBlocked(const Vec3& from, const Vec3& to,
                                 const DeviceOctreeNode* nodes,
                                 const uint32_t* leafTriangleIndices,
                                 const DeviceTriangle* triangles, uint32_t triangleCount,
                                 uint32_t sourceIndex, uint32_t destinationIndex) {
    const Vec3 delta = deviceSubtract(to, from);
    const val_t rayLength = sqrtf(deviceSquaredNorm(delta));
    if (rayLength < EPSILON) return true;
    const Vec3 direction(delta.x / rayLength, delta.y / rayLength, delta.z / rayLength);

    constexpr int MAX_OCTREE_STACK = 64;
    uint32_t stack[MAX_OCTREE_STACK];
    int stackSize = 1;
    stack[0] = 0;

    while (stackSize > 0) {
        const DeviceOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount != 0) {
            for (uint32_t offset = 0; offset < node.triangleCount; ++offset) {
                const uint32_t triangleIndex = leafTriangleIndices[node.triangleStart + offset];
                if (triangleIndex != sourceIndex && triangleIndex != destinationIndex &&
                    deviceSegmentIntersectsTriangle(from, direction, rayLength, triangles[triangleIndex])) {
                    return true;
                }
            }
            continue;
        }

        // Reverse push order preserves the host traversal's child order.
        for (int child = 7; child >= 0; --child) {
            const uint32_t childIndex = node.children[child];
            if (childIndex != 0xffffffffu &&
                deviceRayIntersectsBox(from, to, nodes[childIndex])) {
                if (stackSize == MAX_OCTREE_STACK) {
                    return deviceRayBlockedBruteForce(from, direction, rayLength, triangles,
                                                      triangleCount, sourceIndex, destinationIndex);
                }
                stack[stackSize++] = childIndex;
            }
        }
    }
    return false;
}

__device__ __forceinline__ Vec3 deviceRandomPointInTriangle(const DeviceTriangle& triangle,
                                                              val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return Vec3(triangle.a.x + (triangle.b.x - triangle.a.x) * u + (triangle.c.x - triangle.a.x) * v,
                triangle.a.y + (triangle.b.y - triangle.a.y) * u + (triangle.c.y - triangle.a.y) * v,
                triangle.a.z + (triangle.b.z - triangle.a.z) * u + (triangle.c.z - triangle.a.z) * v);
}

__global__ void computeTimeDelaysKernel(int* tau, const DeviceTriangle* triangles, size_t numTriangles) {
    const size_t pairIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pairCount = numTriangles * numTriangles;
    if (pairIndex >= pairCount) return;

    const size_t i = pairIndex / numTriangles;
    const size_t j = pairIndex - i * numTriangles;
    if (i == j) {
        tau[pairIndex] = 0;
        return;
    }

    const DeviceTriangle& triangleI = triangles[i];
    const DeviceTriangle& triangleJ = triangles[j];
    const Vec3 centerI((triangleI.a.x + triangleI.b.x + triangleI.c.x) / 3.0f,
                       (triangleI.a.y + triangleI.b.y + triangleI.c.y) / 3.0f,
                       (triangleI.a.z + triangleI.b.z + triangleI.c.z) / 3.0f);
    const Vec3 centerJ((triangleJ.a.x + triangleJ.b.x + triangleJ.c.x) / 3.0f,
                       (triangleJ.a.y + triangleJ.b.y + triangleJ.c.y) / 3.0f,
                       (triangleJ.a.z + triangleJ.b.z + triangleJ.c.z) / 3.0f);
    const Vec3 difference = deviceSubtract(centerI, centerJ);
    tau[pairIndex] = static_cast<int>(ceilf(sqrtf(deviceSquaredNorm(difference)) * INV_WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(val_t* kij, const DeviceTriangle* triangles,
                                          const DeviceOctreeNode* nodes,
                                          const uint32_t* leafTriangleIndices,
                                          const val_t* samples, size_t pairBegin,
                                          size_t pairCount, size_t numTriangles) {
    const size_t localPairIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localPairIndex >= pairCount) return;
    const size_t pairIndex = pairBegin + localPairIndex;
    const size_t i = pairIndex / numTriangles;
    const size_t j = pairIndex - i * numTriangles;
    if (i == j) {
        kij[pairIndex] = ZERO;
        return;
    }

    const DeviceTriangle& triangleI = triangles[i];
    const DeviceTriangle& triangleJ = triangles[j];
    if (deviceDot(triangleI.normal, triangleJ.normal) > 0.99f) {
        kij[pairIndex] = ZERO;
        return;
    }

    const val_t* pairSamples = samples + localPairIndex * NUM_RAYS * 4;
    val_t value = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        const val_t* raySamples = pairSamples + ray * 4;
        const Vec3 pointI = deviceRandomPointInTriangle(triangleI, raySamples[0], raySamples[1]);
        const Vec3 pointJ = deviceRandomPointInTriangle(triangleJ, raySamples[2], raySamples[3]);
        if (deviceRayBlocked(pointI, pointJ, nodes, leafTriangleIndices, triangles,
                             static_cast<uint32_t>(numTriangles), static_cast<uint32_t>(i),
                             static_cast<uint32_t>(j))) {
            continue;
        }

        const Vec3 vector = deviceSubtract(pointJ, pointI);
        const val_t distanceSquared = deviceSquaredNorm(vector);
        if (distanceSquared < EPSILON) continue;
        const val_t inverseDistance = rsqrtf(distanceSquared);
        const val_t cosI = fmaxf(ZERO, deviceDot(vector, triangleI.normal) * inverseDistance);
        const Vec3 reverseVector(-vector.x, -vector.y, -vector.z);
        const val_t cosJ = fmaxf(ZERO, deviceDot(reverseVector, triangleJ.normal) * inverseDistance);
        if (cosI > ZERO && cosJ > ZERO) {
            value += (cosI * cosJ) / (PI * distanceSquared);
        }
    }
    kij[pairIndex] = value * INV_NUM_RAYS;
}

__global__ void initializeEmissionKernel(val_t* radE, size_t numTriangles, size_t numTimesteps,
                                         size_t sourceIndex) {
    const size_t timestep = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (timestep < numTimesteps / 2) {
        radE[timestep * numTriangles + sourceIndex] = ONE;
    }
}

__global__ void propagateWaveKernel(val_t* radB, const val_t* radE, const val_t* kij,
                                    const int* tau, const val_t* areas, const val_t* rho,
                                    size_t numTriangles, size_t timestep) {
    const size_t receiver = blockIdx.x;
    if (receiver >= numTriangles) return;

    val_t partial = ZERO;
    const size_t rowOffset = receiver * numTriangles;
    for (size_t emitter = threadIdx.x; emitter < numTriangles; emitter += blockDim.x) {
        const int delay = tau[rowOffset + emitter];
        if (static_cast<int>(timestep) < delay) continue;
        const val_t formFactor = kij[rowOffset + emitter];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radB[(timestep - static_cast<size_t>(delay)) * numTriangles + emitter];
        if (sourceRadiosity <= ZERO) continue;
        partial += fminf(formFactor * areas[emitter], ONE) * sourceRadiosity;
    }

    __shared__ val_t partialSums[256];
    partialSums[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) partialSums[threadIdx.x] += partialSums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        radB[timestep * numTriangles + receiver] = rho[receiver] * partialSums[0] +
                                                    radE[timestep * numTriangles + receiver];
    }
}

__device__ __forceinline__ bool deviceBetterCorrelation(val_t candidateValue, int candidateTime,
                                                         val_t currentValue, int currentTime) {
    return candidateValue > currentValue ||
           (candidateValue == currentValue && candidateTime < currentTime);
}

__global__ void computeDistancesKernel(val_t* distances, const val_t* radB, size_t numTriangles,
                                       size_t numTimesteps, size_t sourceIndex) {
    const size_t triangleIndex = blockIdx.x;
    if (triangleIndex >= numTriangles) return;

    val_t bestValue = ZERO;
    int bestTime = 0;
    for (size_t delay = threadIdx.x; delay < numTimesteps; delay += blockDim.x) {
        val_t sum = ZERO;
        for (size_t timestep = delay; timestep < numTimesteps; ++timestep) {
            const val_t response = radB[timestep * numTriangles + triangleIndex];
            const val_t source = radB[(timestep - delay) * numTriangles + sourceIndex];
            sum += source * response;
        }
        if (deviceBetterCorrelation(sum, static_cast<int>(delay), bestValue, bestTime)) {
            bestValue = sum;
            bestTime = static_cast<int>(delay);
        }
    }

    __shared__ val_t correlationValues[128];
    __shared__ int correlationTimes[128];
    correlationValues[threadIdx.x] = bestValue;
    correlationTimes[threadIdx.x] = bestTime;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset &&
            deviceBetterCorrelation(correlationValues[threadIdx.x + offset],
                                    correlationTimes[threadIdx.x + offset],
                                    correlationValues[threadIdx.x], correlationTimes[threadIdx.x])) {
            correlationValues[threadIdx.x] = correlationValues[threadIdx.x + offset];
            correlationTimes[threadIdx.x] = correlationTimes[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[triangleIndex] = WAVE_SPEED * static_cast<val_t>(correlationTimes[0]);
    }
}

class DeviceSimulation {
public:
    explicit DeviceSimulation(const SimulationState& state)
        : numTriangles(state.numTriangles), numTimesteps(state.numTimesteps) {
        std::vector<DeviceTriangle> hostTriangles(numTriangles);
        for (size_t i = 0; i < numTriangles; ++i) {
            hostTriangles[i] = {state.triangles[i].a, state.triangles[i].b, state.triangles[i].c,
                                state.triangles[i].normal()};
        }
        const FlatOctree flatOctree = flattenOctree(state.octree);
        const size_t matrixElements = numTriangles * numTriangles;
        const size_t timeElements = numTriangles * numTimesteps;

        CUDA_CHECK(cudaMalloc(&triangles, numTriangles * sizeof(DeviceTriangle)));
        CUDA_CHECK(cudaMalloc(&areas, numTriangles * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&rho, numTriangles * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&kij, matrixElements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&tau, matrixElements * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&radE, timeElements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&radB, timeElements * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&distances, numTriangles * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&octreeNodes, flatOctree.nodes.size() * sizeof(DeviceOctreeNode)));
        CUDA_CHECK(cudaMalloc(&leafTriangleIndices,
                              flatOctree.triangleIndices.size() * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(triangles, hostTriangles.data(), numTriangles * sizeof(DeviceTriangle),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(areas, state.areas.data(), numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(rho, state.rho.data(), numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(octreeNodes, flatOctree.nodes.data(),
                              flatOctree.nodes.size() * sizeof(DeviceOctreeNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(leafTriangleIndices, flatOctree.triangleIndices.data(),
                              flatOctree.triangleIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(kij, 0, matrixElements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(tau, 0, matrixElements * sizeof(int)));
        CUDA_CHECK(cudaMemset(radE, 0, timeElements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(radB, 0, timeElements * sizeof(val_t)));
        CUDA_CHECK(cudaMemset(distances, 0, numTriangles * sizeof(val_t)));

        const unsigned int blocks = static_cast<unsigned int>((numTimesteps + 255) / 256);
        initializeEmissionKernel<<<blocks, 256>>>(radE, numTriangles, numTimesteps, state.sourceIndex);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    DeviceSimulation(const DeviceSimulation&) = delete;
    DeviceSimulation& operator=(const DeviceSimulation&) = delete;

    ~DeviceSimulation() {
        cudaFree(leafTriangleIndices);
        cudaFree(octreeNodes);
        cudaFree(distances);
        cudaFree(radB);
        cudaFree(radE);
        cudaFree(tau);
        cudaFree(kij);
        cudaFree(rho);
        cudaFree(areas);
        cudaFree(triangles);
    }

    void copyResultsToHost(SimulationState& state, bool copyValidationData) const {
        CUDA_CHECK(cudaMemcpy(state.distances.data(), distances, numTriangles * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        if (copyValidationData) {
            const size_t matrixElements = numTriangles * numTriangles;
            const size_t timeElements = numTriangles * numTimesteps;
            state.kij.resize(matrixElements);
            state.radB.resize(timeElements);
            CUDA_CHECK(cudaMemcpy(state.kij.data(), kij, matrixElements * sizeof(val_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(state.radB.data(), radB, timeElements * sizeof(val_t), cudaMemcpyDeviceToHost));
        }
    }

    size_t numTriangles;
    size_t numTimesteps;
    DeviceTriangle* triangles = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;
    DeviceOctreeNode* octreeNodes = nullptr;
    uint32_t* leafTriangleIndices = nullptr;
};

constexpr size_t RANDOM_VALUES_PER_PAIR = static_cast<size_t>(NUM_RAYS) * 4;
constexpr size_t FORM_FACTOR_TILE_PAIRS = 1u << 18;

void populateRandomSamples(val_t* samples, size_t pairBegin, size_t pairCount,
                           const SimulationState& state, RandomGenerator& rng) {
    for (size_t localPair = 0; localPair < pairCount; ++localPair) {
        const size_t pairIndex = pairBegin + localPair;
        const size_t i = pairIndex / state.numTriangles;
        const size_t j = pairIndex - i * state.numTriangles;
        if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;

        val_t* pairSamples = samples + localPair * RANDOM_VALUES_PER_PAIR;
        for (int ray = 0; ray < NUM_RAYS; ++ray) {
            // This reproduces the original single mt19937 stream exactly while
            // allowing every pair's geometric work to run independently on GPU.
            *pairSamples++ = rng.rand();
            *pairSamples++ = rng.rand();
            *pairSamples++ = rng.rand();
            *pairSamples++ = rng.rand();
        }
    }
}

void computeFormFactors(const SimulationState& state, DeviceSimulation& device) {
    printf("Computing form factors (Kij) on CUDA GPU...\n");
    const size_t totalPairs = state.numTriangles * state.numTriangles;
    const size_t tileCapacity = std::min(totalPairs, FORM_FACTOR_TILE_PAIRS);
    val_t* sampleBuffers[2] = {nullptr, nullptr};
    val_t* deviceSampleBuffers[2] = {nullptr, nullptr};
    cudaStream_t streams[2] = {nullptr, nullptr};
    cudaEvent_t finished[2] = {nullptr, nullptr};
    bool inFlight[2] = {false, false};

    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaMallocHost(&sampleBuffers[slot], tileCapacity * RANDOM_VALUES_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&deviceSampleBuffers[slot],
                              tileCapacity * RANDOM_VALUES_PER_PAIR * sizeof(val_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&finished[slot], cudaEventDisableTiming));
    }

    RandomGenerator rng(42);
    for (size_t pairBegin = 0, tileNumber = 0; pairBegin < totalPairs; ++tileNumber) {
        const size_t pairCount = std::min(tileCapacity, totalPairs - pairBegin);
        const int slot = static_cast<int>(tileNumber & 1);
        if (inFlight[slot]) CUDA_CHECK(cudaEventSynchronize(finished[slot]));

        populateRandomSamples(sampleBuffers[slot], pairBegin, pairCount, state, rng);
        CUDA_CHECK(cudaMemcpyAsync(deviceSampleBuffers[slot], sampleBuffers[slot],
                                   pairCount * RANDOM_VALUES_PER_PAIR * sizeof(val_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        computeFormFactorsKernel<<<static_cast<unsigned int>((pairCount + 255) / 256), 256, 0, streams[slot]>>>(
            device.kij, device.triangles, device.octreeNodes, device.leafTriangleIndices,
            deviceSampleBuffers[slot], pairBegin, pairCount, state.numTriangles);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(finished[slot], streams[slot]));
        pairBegin += pairCount;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaEventDestroy(finished[slot]));
        CUDA_CHECK(cudaStreamDestroy(streams[slot]));
        CUDA_CHECK(cudaFree(deviceSampleBuffers[slot]));
        CUDA_CHECK(cudaFreeHost(sampleBuffers[slot]));
    }
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(const SimulationState& state, DeviceSimulation& device) {
    printf("Computing time delays (Tau) on CUDA GPU...\n");
    const size_t pairCount = state.numTriangles * state.numTriangles;
    computeTimeDelaysKernel<<<static_cast<unsigned int>((pairCount + 255) / 256), 256>>>(
        device.tau, device.triangles, state.numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(const SimulationState& state, DeviceSimulation& device) {
    printf("Running wave propagation simulation on CUDA GPU...\n");
    for (size_t timestep = 0; timestep < state.numTimesteps; ++timestep) {
        propagateWaveKernel<<<static_cast<unsigned int>(state.numTriangles), 256>>>(
            device.radB, device.radE, device.kij, device.tau, device.areas, device.rho,
            state.numTriangles, timestep);
        CUDA_CHECK(cudaGetLastError());
        if ((timestep + 1) % 10 == 0 || timestep + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", timestep + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(const SimulationState& state, DeviceSimulation& device) {
    printf("Computing distances via cross-correlation on CUDA GPU...\n");
    computeDistancesKernel<<<static_cast<unsigned int>(state.numTriangles), 128>>>(
        device.distances, device.radB, state.numTriangles, state.numTimesteps, state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    DeviceSimulation deviceState(state);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state, deviceState);
    computeFormFactors(state, deviceState);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, deviceState);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, deviceState);
    deviceState.copyResultsToHost(state, validate);

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
