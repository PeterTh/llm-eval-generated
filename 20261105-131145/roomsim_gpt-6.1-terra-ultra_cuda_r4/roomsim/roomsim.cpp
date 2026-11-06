/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
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
    // A float uniform distribution consumes one mt19937 result per sample.
    void discard(size_t count) { rng.discard(count); }
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

// ============================================================================
// Precomputation Phase
// ============================================================================

// A compact copy of the CPU octree, with indices instead of pointers, can be
// traversed by every GPU thread independently.
struct GpuOctreeNode {
    Vec3 center, halfExtent;
    int children[8];
    int first, count;
};

static void gpuPreCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) gpuPreCheck((call), #call)

static int flattenOctree(const Octree& tree, std::vector<GpuOctreeNode>& nodes,
                         std::vector<unsigned int>& indices) {
    const int current = static_cast<int>(nodes.size());
    GpuOctreeNode node{};
    node.center = tree.center;
    node.halfExtent = tree.halfExtent;
    node.first = static_cast<int>(indices.size());
    node.count = static_cast<int>(tree.triangleIndices.size());
    for (int k = 0; k < 8; ++k) node.children[k] = -1;
    nodes.push_back(node);
    for (size_t idx : tree.triangleIndices) indices.push_back(static_cast<unsigned int>(idx));
    for (int k = 0; k < 8; ++k) {
        if (tree.children[k]) {
            int child = flattenOctree(*tree.children[k], nodes, indices);
            nodes[current].children[k] = child;
        }
    }
    return current;
}

__device__ static float gpuNorm(Vec3 v) { return sqrtf(v.squaredNorm()); }

__device__ static bool gpuBox(const GpuOctreeNode& node, Vec3 p1, Vec3 p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ static bool gpuHit(Vec3 orig, Vec3 dir, const Triangle& tri, float rayLen) {
    Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    float det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return false;
    float invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 qvec = tvec.cross(e1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    float dist = e2.dot(qvec) * invDet;
    return dist > EPSILON && dist < rayLen - EPSILON;
}

__device__ static bool gpuBlocked(Vec3 from, Vec3 to, const Triangle* tris,
                                   const GpuOctreeNode* nodes, const unsigned int* indices,
                                   unsigned int src, unsigned int dst) {
    Vec3 direction = to - from;
    float length = gpuNorm(direction);
    if (length < EPSILON) return true;
    direction = direction / length;
    // The octree terminates when a node diagonal is below 0.5. The room
    // diameter is 20, so its maximum depth is seven; 64 entries are ample.
    int stack[64];
    int top = 0;
    stack[top++] = 0;
    while (top) {
        const GpuOctreeNode& node = nodes[stack[--top]];
        if (node.count) {
            for (int k = 0; k < node.count; ++k) {
                unsigned int idx = indices[node.first + k];
                if (idx != src && idx != dst && gpuHit(from, direction, tris[idx], length)) return true;
            }
        } else {
            for (int k = 7; k >= 0; --k) {
                int child = node.children[k];
                if (child >= 0 && gpuBox(nodes[child], from, to)) stack[top++] = child;
            }
        }
    }
    return false;
}

__device__ static Vec3 gpuPoint(const Triangle& t, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
}

__global__ static void gpuFormFactorKernel(const Triangle* tris, const GpuOctreeNode* nodes,
                                            const unsigned int* indices, const float* random,
                                            const unsigned char* active, float* result,
                                            size_t n, size_t firstRow, size_t pairCount) {
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= pairCount) return;
    if (!active[local]) { result[local] = 0.0f; return; }
    size_t i = firstRow + local / n, j = local % n;
    const Triangle& ti = tris[i];
    const Triangle& tj = tris[j];
    const float* values = random + local * 64;
    float sum = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pi = gpuPoint(ti, values[4*r], values[4*r+1]);
        Vec3 pj = gpuPoint(tj, values[4*r+2], values[4*r+3]);
        if (gpuBlocked(pi, pj, tris, nodes, indices, static_cast<unsigned int>(i), static_cast<unsigned int>(j))) continue;
        Vec3 v = pj - pi;
        float distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;
        float length = gpuNorm(v);
        float ci = length <= EPSILON ? 0.0f : fmaxf(0.0f, v.dot(ti._normal) / length);
        Vec3 neg = -v;
        float cj = length <= EPSILON ? 0.0f : fmaxf(0.0f, neg.dot(tj._normal) / length);
        if (ci <= 0.0f || cj <= 0.0f) continue;
        sum += (ci * cj) / (PI * distSqr);
    }
    result[local] = sum * INV_NUM_RAYS;
}

__global__ static void gpuTauKernel(const Triangle* tris, int* tau, size_t n, size_t total) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t i = idx / n, j = idx % n;
    if (i == j) { tau[idx] = 0; return; }
    Vec3 ci = (tris[i].a + tris[i].b + tris[i].c) / 3.0f;
    Vec3 cj = (tris[j].a + tris[j].b + tris[j].c) / 3.0f;
    tau[idx] = static_cast<int>(ceilf(gpuNorm(ci - cj) * INV_WAVE_SPEED));
}

// Each chunk starts from a copy of the original generator state. Advancing
// the master by the number of samples in the chunk preserves its single
// row-major stream while allowing the expensive sample generation to run in
// parallel. Workers persist across tiles so their startup cost is paid once.
class RandomSampleWorkers {
    static constexpr size_t chunkSize = 256;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable ready, done;
    std::atomic<size_t> next{0};
    size_t generation = 0, remaining = 0, count = 0;
    bool stopping = false;
    float* samples = nullptr;
    const unsigned char* active = nullptr;
    const std::vector<RandomGenerator>* checkpoints = nullptr;

    void work() {
        size_t observed = 0;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            ready.wait(lock, [&] { return stopping || generation != observed; });
            if (stopping) return;
            observed = generation;
            lock.unlock();
            for (;;) {
                size_t chunk = next.fetch_add(1, std::memory_order_relaxed);
                if (chunk >= checkpoints->size()) break;
                RandomGenerator rng = (*checkpoints)[chunk];
                size_t end = std::min(count, (chunk + 1) * chunkSize);
                for (size_t k = chunk * chunkSize; k < end; ++k) {
                    if (active[k]) {
                        for (int r = 0; r < 64; ++r) samples[k * 64 + r] = rng.rand();
                    }
                }
            }
            lock.lock();
            if (--remaining == 0) done.notify_one();
        }
    }
public:
    explicit RandomSampleWorkers(size_t capacity) {
        unsigned available = std::thread::hardware_concurrency();
        size_t count = std::min<size_t>({16, available ? available : 1,
                                         (capacity + chunkSize - 1) / chunkSize});
        for (size_t i = 0; i < count; ++i) workers.emplace_back(&RandomSampleWorkers::work, this);
    }
    ~RandomSampleWorkers() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto& worker : workers) worker.join();
    }
    void generate(RandomGenerator& master, const unsigned char* flags, float* output, size_t n) {
        std::vector<RandomGenerator> states;
        states.reserve((n + chunkSize - 1) / chunkSize);
        for (size_t start = 0; start < n; start += chunkSize) {
            states.push_back(master);
            size_t used = 0;
            for (size_t k = start; k < std::min(start + chunkSize, n); ++k) used += flags[k];
            master.discard(used * 64);
        }
        std::unique_lock<std::mutex> lock(mutex);
        active = flags;
        samples = output;
        count = n;
        checkpoints = &states;
        next.store(0, std::memory_order_relaxed);
        remaining = workers.size();
        ++generation;
        ready.notify_all();
        done.wait(lock, [&] { return remaining == 0; });
    }
};

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    std::vector<GpuOctreeNode> nodes;
    std::vector<unsigned int> indices;
    flattenOctree(state.octree, nodes, indices);
    Triangle* dTris;
    GpuOctreeNode* dNodes;
    unsigned int* dIndices;
    const size_t rows = std::min<size_t>(n, 16);
    const size_t capacity = rows * n;
    gpuPreCheck(cudaMalloc(&dTris, n * sizeof(Triangle)), "allocating triangles");
    gpuPreCheck(cudaMalloc(&dNodes, nodes.size() * sizeof(GpuOctreeNode)), "allocating octree");
    gpuPreCheck(cudaMalloc(&dIndices, indices.size() * sizeof(unsigned int)), "allocating octree indices");
    gpuPreCheck(cudaMemcpy(dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "copying triangles");
    gpuPreCheck(cudaMemcpy(dNodes, nodes.data(), nodes.size() * sizeof(GpuOctreeNode), cudaMemcpyHostToDevice), "copying octree");
    gpuPreCheck(cudaMemcpy(dIndices, indices.data(), indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copying octree indices");

    // Two independent streams and pinned buffers let CPU sample generation
    // proceed while the GPU processes the preceding tile. The RNG remains
    // strictly ordered across tiles, just as in the original serial loop.
    struct Tile {
        float *random, *result, *dRandom, *dResult;
        unsigned char *active, *dActive;
        cudaStream_t stream;
        size_t row = 0, count = 0;
    } tiles[2]{};
    for (auto& tile : tiles) {
        gpuPreCheck(cudaStreamCreateWithFlags(&tile.stream, cudaStreamNonBlocking), "creating form factor stream");
        gpuPreCheck(cudaMallocHost(&tile.random, capacity * 64 * sizeof(float)), "allocating pinned random samples");
        gpuPreCheck(cudaMallocHost(&tile.active, capacity), "allocating pinned active flags");
        gpuPreCheck(cudaMallocHost(&tile.result, capacity * sizeof(float)), "allocating pinned form factors");
        gpuPreCheck(cudaMalloc(&tile.dRandom, capacity * 64 * sizeof(float)), "allocating random samples");
        gpuPreCheck(cudaMalloc(&tile.dResult, capacity * sizeof(float)), "allocating form factors");
        gpuPreCheck(cudaMalloc(&tile.dActive, capacity), "allocating active flags");
    }
    RandomGenerator rng(42);
    RandomSampleWorkers sampleWorkers(capacity);
    size_t tileNumber = 0;
    for (size_t row = 0; row < n; row += rows, ++tileNumber) {
        Tile& tile = tiles[tileNumber % 2];
        if (tile.count) {
            gpuPreCheck(cudaStreamSynchronize(tile.stream), "waiting for form factor tile");
            std::memcpy(state.kij.data() + tile.row * n, tile.result, tile.count * sizeof(float));
        }
        tile.row = row;
        tile.count = std::min(rows, n - row) * n;
        for (size_t k = 0; k < tile.count; ++k) {
            size_t i = row + k / n, j = k % n;
            bool use = i != j && !(state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f);
            tile.active[k] = static_cast<unsigned char>(use);
        }
        sampleWorkers.generate(rng, tile.active, tile.random, tile.count);
        gpuPreCheck(cudaMemcpyAsync(tile.dRandom, tile.random, tile.count * 64 * sizeof(float), cudaMemcpyHostToDevice, tile.stream), "copying random samples");
        gpuPreCheck(cudaMemcpyAsync(tile.dActive, tile.active, tile.count, cudaMemcpyHostToDevice, tile.stream), "copying active flags");
        gpuFormFactorKernel<<<static_cast<unsigned int>((tile.count + 127) / 128), 128, 0, tile.stream>>>(dTris, dNodes, dIndices, tile.dRandom, tile.dActive, tile.dResult, n, row, tile.count);
        gpuPreCheck(cudaGetLastError(), "launching form factor kernel");
        gpuPreCheck(cudaMemcpyAsync(tile.result, tile.dResult, tile.count * sizeof(float), cudaMemcpyDeviceToHost, tile.stream), "copying form factors");
    }
    for (auto& tile : tiles) {
        if (tile.count) {
            gpuPreCheck(cudaStreamSynchronize(tile.stream), "waiting for form factor tile");
            std::memcpy(state.kij.data() + tile.row * n, tile.result, tile.count * sizeof(float));
        }
        gpuPreCheck(cudaFree(tile.dRandom), "freeing random samples");
        gpuPreCheck(cudaFree(tile.dResult), "freeing form factors");
        gpuPreCheck(cudaFree(tile.dActive), "freeing active flags");
        gpuPreCheck(cudaFreeHost(tile.random), "freeing pinned random samples");
        gpuPreCheck(cudaFreeHost(tile.result), "freeing pinned form factors");
        gpuPreCheck(cudaFreeHost(tile.active), "freeing pinned active flags");
        gpuPreCheck(cudaStreamDestroy(tile.stream), "destroying form factor stream");
    }
    printf("  Progress: %zu/%zu triangles\n", n, n);
    cudaFree(dTris); cudaFree(dNodes); cudaFree(dIndices);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const size_t n = state.numTriangles, total = n * n;
    Triangle* dTris;
    int* dTau;
    gpuPreCheck(cudaMalloc(&dTris, n * sizeof(Triangle)), "allocating triangles");
    gpuPreCheck(cudaMalloc(&dTau, total * sizeof(int)), "allocating time delays");
    gpuPreCheck(cudaMemcpy(dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "copying triangles");
    gpuTauKernel<<<static_cast<unsigned int>((total + 255) / 256), 256>>>(dTris, dTau, n, total);
    gpuPreCheck(cudaGetLastError(), "launching time delay kernel");
    gpuPreCheck(cudaMemcpy(state.tau.data(), dTau, total * sizeof(int), cudaMemcpyDeviceToHost), "copying time delays");
    cudaFree(dTris); cudaFree(dTau);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One warp computes a receiver. Lanes fetch adjacent emitters, while lane 0
// sums their contributions in the original order to preserve rounding.
__global__ void propagationKernel(size_t n, size_t t, const val_t* kij,
                                  const int* tau, const val_t* areas,
                                  const val_t* rho, const val_t* emission,
                                  val_t* radiosity) {
    __shared__ val_t contributions[128];
    const unsigned lane = threadIdx.x & 31;
    const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (i >= n) return;
    val_t sum = 0.0f;
    for (size_t base = 0; base < n; base += 32) {
        size_t j = base + lane;
        val_t contribution = 0.0f;
        if (j < n && i != j) {
            const size_t pos = i * n + j;
            int delay = tau[pos];
            if (static_cast<int>(t) >= delay) {
                val_t factor = kij[pos];
                if (factor > 0.0f) {
                    val_t rad = radiosity[(t - static_cast<size_t>(delay)) * n + j];
                    if (rad > 0.0f) {
                        val_t weight = factor * areas[j];
                        contribution = (weight < 1.0f ? weight : 1.0f) * rad;
                    }
                }
            }
        }
        contributions[threadIdx.x] = contribution;
        __syncwarp();
        if (lane == 0) {
            const unsigned count = static_cast<unsigned>((n - base < 32) ? n - base : 32);
            for (unsigned k = 0; k < count; ++k) sum += contributions[threadIdx.x + k];
        }
        __syncwarp();
    }
    if (lane == 0) radiosity[t * n + i] = rho[i] * sum + emission[t * n + i];
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles, steps = state.numTimesteps;
    val_t *dKij, *dAreas, *dRho, *dEmission, *dRad;
    int* dTau;
    CUDA_CHECK(cudaMalloc(&dKij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dTau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dEmission, std::max<size_t>(1, n * steps) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dRad, std::max<size_t>(1, n * steps) * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(dKij, state.kij.data(), n * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTau, state.tau.data(), n * n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    if (steps) {
        CUDA_CHECK(cudaMemcpy(dEmission, state.radE.data(), n * steps * sizeof(val_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dRad, 0, n * steps * sizeof(val_t)));
    }
    for (size_t t = 0; t < steps; ++t) {
        propagationKernel<<<(n + 3) / 4, 128>>>(n, t, dKij, dTau, dAreas, dRho, dEmission, dRad);
        CUDA_CHECK(cudaGetLastError());
    }
    if (steps) CUDA_CHECK(cudaMemcpy(state.radB.data(), dRad, n * steps * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dKij)); CUDA_CHECK(cudaFree(dTau));
    CUDA_CHECK(cudaFree(dAreas)); CUDA_CHECK(cudaFree(dRho));
    CUDA_CHECK(cudaFree(dEmission)); CUDA_CHECK(cudaFree(dRad));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void distanceKernel(size_t n, size_t steps, size_t source,
                               const val_t* rad, val_t* distances) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < steps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < steps; ++tt) {
                val_t pB = rad[tt * n + i];
                val_t pS = rad[(tt - t) * n + source];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles, count = n * state.numTimesteps;
    val_t *dRad, *dDistances;
    CUDA_CHECK(cudaMalloc(&dRad, std::max<size_t>(1, count) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&dDistances, n * sizeof(val_t)));
    if (count) CUDA_CHECK(cudaMemcpy(dRad, state.radB.data(), count * sizeof(val_t), cudaMemcpyHostToDevice));
    distanceKernel<<<(n + 127) / 128, 128>>>(n, state.numTimesteps, state.sourceIndex, dRad, dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), dDistances, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dRad)); CUDA_CHECK(cudaFree(dDistances));
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
