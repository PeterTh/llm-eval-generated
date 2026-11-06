/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified CUDA implementation of room impulse response
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

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

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

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

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
}


// A compact, stackless representation of the CPU octree.  `escape` points
// past the entire subtree, so each CUDA thread can traverse it without a
// per-thread stack or device-side allocation.
struct DeviceOctreeNode {
    float cx, cy, cz, hx, hy, hz;
    int first, count, escape;
};

struct DeviceTriangle {
    float ax, ay, az, bx, by, bz, cx, cy, cz, nx, ny, nz;
};

static int flattenFormFactorOctree(const Octree& tree,
                                   std::vector<DeviceOctreeNode>& nodes,
                                   std::vector<unsigned>& indices) {
    int index = static_cast<int>(nodes.size());
    nodes.push_back({tree.center.x, tree.center.y, tree.center.z,
                     tree.halfExtent.x, tree.halfExtent.y, tree.halfExtent.z,
                     static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size()), 0});
    for (size_t triangle : tree.triangleIndices)
        indices.push_back(static_cast<unsigned>(triangle));
    for (int c = 0; c < 8; ++c)
        if (tree.children[c]) flattenFormFactorOctree(*tree.children[c], nodes, indices);
    nodes[index].escape = static_cast<int>(nodes.size());
    return index;
}

struct FFVec { float x, y, z; };
__device__ inline FFVec ffadd(FFVec a, FFVec b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ inline FFVec ffsub(FFVec a, FFVec b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ inline FFVec ffmul(FFVec a, float b) { return {a.x*b,a.y*b,a.z*b}; }
__device__ inline FFVec ffdiv(FFVec a, float b) { return {a.x/b,a.y/b,a.z/b}; }
__device__ inline float ffdot(FFVec a, FFVec b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ inline FFVec ffcross(FFVec a, FFVec b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
__device__ inline float ffnorm(FFVec a) { return sqrtf(ffdot(a,a)); }
__device__ inline FFVec ffa(const DeviceTriangle& t) { return {t.ax,t.ay,t.az}; }
__device__ inline FFVec ffb(const DeviceTriangle& t) { return {t.bx,t.by,t.bz}; }
__device__ inline FFVec ffc(const DeviceTriangle& t) { return {t.cx,t.cy,t.cz}; }
__device__ inline FFVec ffn(const DeviceTriangle& t) { return {t.nx,t.ny,t.nz}; }

__device__ inline FFVec ffpoint(const DeviceTriangle& t, float u, float v) {
    if (u+v > 1.0f) { u=1.0f-u; v=1.0f-v; }
    FFVec a=ffa(t);
    return ffadd(ffadd(a,ffmul(ffsub(ffb(t),a),u)),ffmul(ffsub(ffc(t),a),v));
}

__device__ inline bool ffbox(const DeviceOctreeNode& node, FFVec p1, FFVec p2) {
    FFVec d=ffmul(ffsub(p2,p1),0.5f);
    FFVec c=ffsub(ffadd(p1,d),{node.cx,node.cy,node.cz});
    FFVec ad={fabsf(d.x),fabsf(d.y),fabsf(d.z)};
    if (fabsf(c.x)>node.hx+ad.x || fabsf(c.y)>node.hy+ad.y || fabsf(c.z)>node.hz+ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y)>node.hy*ad.z+node.hz*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z)>node.hz*ad.x+node.hx*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x)>node.hx*ad.y+node.hy*ad.x+EPSILON) return false;
    return true;
}

__device__ inline bool ffintersect(FFVec orig, FFVec dir, float rayLen, const DeviceTriangle& tri) {
    FFVec v0=ffa(tri), e1=ffsub(ffb(tri),v0), e2=ffsub(ffc(tri),v0);
    FFVec pvec=ffcross(dir,e2);
    float det=ffdot(e1,pvec);
    if (fabsf(det)<EPSILON) return false;
    float invDet=1.0f/det;
    FFVec tvec=ffsub(orig,v0);
    float u=ffdot(tvec,pvec)*invDet;
    if (u<0.0f || u>1.0f) return false;
    FFVec qvec=ffcross(tvec,e1);
    float v=ffdot(dir,qvec)*invDet;
    if (v<0.0f || u+v>1.0f) return false;
    float dist=ffdot(e2,qvec)*invDet;
    return dist>EPSILON && dist<rayLen-EPSILON;
}

__device__ inline bool ffblocked(FFVec from, FFVec to, unsigned src, unsigned dst,
                                 const DeviceTriangle* triangles, const DeviceOctreeNode* nodes,
                                 const unsigned* indices, int nodeCount) {
    FFVec dir=ffsub(to,from);
    float len=ffnorm(dir);
    if (len<EPSILON) return true;
    dir=ffdiv(dir,len);
    int node=0;
    while (node<nodeCount) {
        DeviceOctreeNode n=nodes[node];
        // The CPU visits the root unconditionally; child boxes are tested.
        if (node!=0 && !ffbox(n,from,to)) { node=n.escape; continue; }
        if (n.count) {
            for (int k=0; k<n.count; ++k) {
                unsigned idx=indices[n.first+k];
                if (idx!=src && idx!=dst && ffintersect(from,dir,len,triangles[idx])) return true;
            }
        }
        ++node;
    }
    return false;
}

__global__ void formFactorKernel(const DeviceTriangle* triangles, int n,
                                  const DeviceOctreeNode* nodes, int nodeCount,
                                  const unsigned* indices, const unsigned* offsets,
                                  const float* randoms, float* output, int firstRow, int pairs) {
    // A half-warp handles a triangle pair, with one ray per lane.  Summation
    // below is ordered to match the sequential reference.
    int thread=blockIdx.x*blockDim.x+threadIdx.x;
    int pair=thread/NUM_RAYS, ray=thread%NUM_RAYS;
    if (pair>=pairs) return;
    unsigned offset=offsets[pair];
    if (offset==UINT32_MAX) { if (ray==0) output[pair]=0.0f; return; }
    unsigned i=firstRow+pair/n, j=pair%n;
    DeviceTriangle a=triangles[i], b=triangles[j];
    const float* random=randoms+offset+4*ray;
    FFVec pi=ffpoint(a,random[0],random[1]);
    FFVec pj=ffpoint(b,random[2],random[3]);
    float result=0.0f;
    if (!ffblocked(pi,pj,i,j,triangles,nodes,indices,nodeCount)) {
        FFVec v=ffsub(pj,pi);
        float distSqr=ffdot(v,v);
        if (distSqr>=EPSILON) {
            float vNorm=ffnorm(v);
            float cosI=vNorm<=EPSILON ? 0.0f : fmaxf(0.0f,ffdot(v,ffn(a))/vNorm);
            FFVec neg={-v.x,-v.y,-v.z};
            float cosJ=vNorm<=EPSILON ? 0.0f : fmaxf(0.0f,ffdot(neg,ffn(b))/vNorm);
            if (cosI>0.0f && cosJ>0.0f) result=(cosI*cosJ)/(PI*distSqr);
        }
    }
    unsigned mask=__activemask();
    int base=(threadIdx.x & 31) & ~15;
    float sum=0.0f;
    for (int r=0; r<NUM_RAYS; ++r) sum+=__shfl_sync(mask,result,base+r);
    if (ray==0) output[pair]=sum*INV_NUM_RAYS;
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t n=state.numTriangles;
    std::vector<DeviceTriangle> triangles;
    triangles.reserve(n);
    for (const Triangle& t : state.triangles)
        triangles.push_back({t.a.x,t.a.y,t.a.z,t.b.x,t.b.y,t.b.z,t.c.x,t.c.y,t.c.z,
                             t._normal.x,t._normal.y,t._normal.z});
    std::vector<DeviceOctreeNode> nodes;
    std::vector<unsigned> indices;
    flattenFormFactorOctree(state.octree,nodes,indices);

    DeviceTriangle* dTriangles=nullptr;
    DeviceOctreeNode* dNodes=nullptr;
    unsigned *dIndices=nullptr, *dOffsets=nullptr;
    float *dRandoms=nullptr, *dOutput=nullptr;
    CUDA_CHECK(cudaMalloc(&dTriangles,triangles.size()*sizeof(*dTriangles)));
    CUDA_CHECK(cudaMalloc(&dNodes,nodes.size()*sizeof(*dNodes)));
    CUDA_CHECK(cudaMalloc(&dIndices,indices.size()*sizeof(*dIndices)));
    CUDA_CHECK(cudaMemcpy(dTriangles,triangles.data(),triangles.size()*sizeof(*dTriangles),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNodes,nodes.data(),nodes.size()*sizeof(*dNodes),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dIndices,indices.data(),indices.size()*sizeof(*dIndices),cudaMemcpyHostToDevice));

    // The original generator is a single sequential mt19937 stream, and
    // facing-pair culling consumes no samples. Generate its stream in row
    // order and transfer bounded batches to preserve exactly that mapping.
    const size_t rowsPerBatch=std::max<size_t>(1,(1u<<18)/n);
    const size_t maxPairs=std::min(n,rowsPerBatch)*n;
    std::vector<unsigned> offsets(maxPairs);
    std::vector<float> randoms;
    randoms.reserve(maxPairs*NUM_RAYS*4);
    CUDA_CHECK(cudaMalloc(&dOffsets,maxPairs*sizeof(*dOffsets)));
    CUDA_CHECK(cudaMalloc(&dRandoms,maxPairs*NUM_RAYS*4*sizeof(*dRandoms)));
    CUDA_CHECK(cudaMalloc(&dOutput,maxPairs*sizeof(*dOutput)));
    RandomGenerator rng(42);
    for (size_t first=0; first<n; first+=rowsPerBatch) {
        size_t rows=std::min(rowsPerBatch,n-first), pairs=rows*n;
        randoms.clear();
        for (size_t k=0; k<pairs; ++k) {
            size_t i=first+k/n, j=k%n;
            if (i==j || state.triangles[i].normal().dot(state.triangles[j].normal())>0.99f) {
                offsets[k]=UINT32_MAX;
                continue;
            }
            offsets[k]=static_cast<unsigned>(randoms.size());
            for (int r=0; r<NUM_RAYS*4; ++r) randoms.push_back(rng.rand());
        }
        CUDA_CHECK(cudaMemcpy(dOffsets,offsets.data(),pairs*sizeof(*dOffsets),cudaMemcpyHostToDevice));
        if (!randoms.empty()) CUDA_CHECK(cudaMemcpy(dRandoms,randoms.data(),randoms.size()*sizeof(*dRandoms),cudaMemcpyHostToDevice));
        formFactorKernel<<<(pairs*NUM_RAYS+127)/128,128>>>(dTriangles,static_cast<int>(n),dNodes,
            static_cast<int>(nodes.size()),dIndices,dOffsets,dRandoms,dOutput,static_cast<int>(first),static_cast<int>(pairs));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.kij.data()+first*n,dOutput,pairs*sizeof(*dOutput),cudaMemcpyDeviceToHost));
        printf("  Progress: %zu/%zu triangles\n",first+rows,n);
    }
    CUDA_CHECK(cudaFree(dOutput));
    CUDA_CHECK(cudaFree(dRandoms));
    CUDA_CHECK(cudaFree(dOffsets));
    CUDA_CHECK(cudaFree(dIndices));
    CUDA_CHECK(cudaFree(dNodes));
    CUDA_CHECK(cudaFree(dTriangles));
}

__global__ void timeDelayKernel(const Vec3* centers, int* tau, size_t n) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    size_t i = index / n;
    size_t j = index % n;
    if (i == j) {
        tau[index] = 0;
        return;
    }
    Vec3 delta = centers[i] - centers[j];
    tau[index] = static_cast<int>(ceilf(delta.norm() * INV_WAVE_SPEED));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    size_t n = state.numTriangles;
    std::vector<Vec3> centers(n);
    for (size_t i = 0; i < n; ++i) centers[i] = state.triangles[i].center();
    Vec3* deviceCenters;
    int* deviceTau;
    CUDA_CHECK(cudaMalloc(&deviceCenters, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&deviceTau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(deviceCenters, centers.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    timeDelayKernel<<<(n * n + 255) / 256, 256>>>(deviceCenters, deviceTau, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.tau.data(), deviceTau, n * n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceTau));
    CUDA_CHECK(cudaFree(deviceCenters));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Transpose the pair matrices so adjacent destination threads read adjacent data.
__global__ void transposePairs(const val_t* kij, const int* tau,
                               val_t* kijT, int* tauT, size_t n) {
    __shared__ val_t kTile[32][33];
    __shared__ int tTile[32][33];
    size_t x = blockIdx.x * 32 + threadIdx.x;
    size_t y = blockIdx.y * 32 + threadIdx.y;
    for (int row = 0; row < 32; row += 8) {
        if (x < n && y + row < n) {
            kTile[threadIdx.y + row][threadIdx.x] = kij[(y + row) * n + x];
            tTile[threadIdx.y + row][threadIdx.x] = tau[(y + row) * n + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    for (int row = 0; row < 32; row += 8) {
        if (x < n && y + row < n) {
            kijT[(y + row) * n + x] = kTile[threadIdx.x][threadIdx.y + row];
            tauT[(y + row) * n + x] = tTile[threadIdx.x][threadIdx.y + row];
        }
    }
}

__global__ void propagateStep(const val_t* kij, const int* tau,
                              const val_t* areas, const val_t* rho,
                              const val_t* radE, val_t* radB,
                              size_t n, size_t t) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t sumB = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t pair = j * n + i;
        int delay = tau[pair];
        if (static_cast<int>(t) < delay) continue;
        val_t k = kij[pair];
        if (k <= ZERO) continue;
        val_t radJ = radB[(t - static_cast<size_t>(delay)) * n + j];
        if (radJ <= ZERO) continue;
        sumB += fminf(k * areas[j], ONE) * radJ;
    }
    radB[t * n + i] = rho[i] * sumB + radE[t * n + i];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles, nt = state.numTimesteps;
    const size_t pairs = n * n;
    const size_t samples = n * nt;
    val_t *kij, *kijT, *areas, *rho, *radE, *radB;
    int *tau, *tauT;
    CUDA_CHECK(cudaMalloc(&kij, pairs * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&kijT, pairs * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&tau, pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tauT, pairs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&radE, std::max<size_t>(samples, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&radB, std::max<size_t>(samples, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(kij, state.kij.data(), pairs * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tau, state.tau.data(), pairs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    if (samples) {
        CUDA_CHECK(cudaMemcpy(radE, state.radE.data(), samples * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(radB, 0, samples * sizeof(val_t)));
    }
    dim3 block(32, 8), grid((n + 31) / 32, (n + 31) / 32);
    transposePairs<<<grid, block>>>(kij, tau, kijT, tauT, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(kij));
    CUDA_CHECK(cudaFree(tau));
    for (size_t t = 0; t < nt; ++t) {
        propagateStep<<<(n + 127) / 128, 128>>>(kijT, tauT, areas, rho, radE, radB, n, t);
        CUDA_CHECK(cudaGetLastError());
        if ((t + 1) % 10 == 0 || t + 1 == nt)
            printf("  Timestep %zu/%zu\n", t + 1, nt);
    }
    if (samples) CUDA_CHECK(cudaMemcpy(state.radB.data(), radB, samples * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(kijT));
    CUDA_CHECK(cudaFree(tauT));
    CUDA_CHECK(cudaFree(areas));
    CUDA_CHECK(cudaFree(rho));
    CUDA_CHECK(cudaFree(radE));
    CUDA_CHECK(cudaFree(radB));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void correlateDistances(const val_t* radB, val_t* distances,
                                   size_t n, size_t nt, size_t source) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < nt; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < nt; ++tt) {
            val_t pB = radB[tt * n + i];
            val_t pS = radB[(tt - t) * n + source];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = state.numTriangles, samples = n * state.numTimesteps;
    val_t *radB, *distances;
    CUDA_CHECK(cudaMalloc(&radB, std::max<size_t>(samples, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&distances, n * sizeof(val_t)));
    if (samples) CUDA_CHECK(cudaMemcpy(radB, state.radB.data(), samples * sizeof(val_t), cudaMemcpyHostToDevice));
    correlateDistances<<<(n + 127) / 128, 128>>>(radB, distances, n, state.numTimesteps, state.sourceIndex);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), distances, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(radB));
    CUDA_CHECK(cudaFree(distances));
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
