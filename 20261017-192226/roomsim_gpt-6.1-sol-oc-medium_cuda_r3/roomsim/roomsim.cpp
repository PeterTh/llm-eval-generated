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

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
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

#define HD __host__ __device__

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    void allocate(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

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

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

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
    std::vector<val_t> kij;         // Validation copy of GPU form factors (column-major)
    std::vector<val_t> radB;        // Validation copy of radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    bool validationEnabled = false;

    DeviceBuffer<Triangle> gpuTriangles;
    DeviceBuffer<val_t> gpuKij, gpuRadB, gpuDistances, gpuAreas, gpuRho;
    DeviceBuffer<int> gpuTau;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Initialize the required CUDA runtime outside the computation phase timers.
    cudaCheck(cudaFree(nullptr));
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

    // Simulation matrices stay on the GPU; only validation needs host copies.
    if (state.validationEnabled) {
        state.kij.resize(state.numTriangles * state.numTriangles);
        state.radB.resize(timesteps * state.numTriangles);
    }
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Preorder octree with escape links: traversal needs no per-ray stack.
struct GpuNode {
    Vec3 center, halfExtent;
    size_t begin, count, escape;
};

void flattenOctree(const Octree& tree, std::vector<GpuNode>& nodes,
                   std::vector<size_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(), tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[index].escape = nodes.size();
}

__device__ bool intersectsNode(const GpuNode& node, Vec3 from, Vec3 to) {
    Vec3 d = (to - from) * 0.5f;
    Vec3 c = from + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(Vec3 from, Vec3 to, const Triangle* triangles,
                        const GpuNode* nodes, size_t nodeCount, const size_t* indices,
                        size_t i, size_t j) {
    Vec3 direction = to - from;
    float length = direction.norm();
    if (length < EPSILON) return true;
    direction = direction / length;
    for (size_t n = 0; n < nodeCount;) {
        const GpuNode node = nodes[n];
        if (n != 0 && !intersectsNode(node, from, to)) {
            n = node.escape;
            continue;
        }
        for (size_t k = 0; k < node.count; ++k) {
            size_t index = indices[node.begin + k];
            if (index == i || index == j) continue;
            const Triangle tri = triangles[index];
            float distance = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++n;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}

// Count only uncullable pairs: the serial implementation consumes no random
// numbers for the diagonal or for similarly oriented triangles.
__global__ void sampleOffsetsKernel(const Triangle* triangles, size_t n, size_t first,
                                     size_t count, unsigned* offsets) {
    __shared__ unsigned scan[256];
    __shared__ unsigned total;
    unsigned lane = threadIdx.x;
    if (lane == 0) total = 0;
    __syncthreads();
    for (size_t base = 0; base < count; base += 256) {
        size_t p = base+lane;
        unsigned draws = 0;
        if (p < count) {
            size_t i = (first+p)/n, j = (first+p)%n;
            if (i != j && triangles[i].normal().dot(triangles[j].normal()) <= 0.99f)
                draws = NUM_RAYS*4;
        }
        scan[lane] = draws;
        __syncthreads();
        for (unsigned step = 1; step < 256; step *= 2) {
            unsigned add = lane >= step ? scan[lane-step] : 0;
            __syncthreads();
            scan[lane] += add;
            __syncthreads();
        }
        if (p < count) offsets[p] = total+scan[lane]-draws;
        __syncthreads();
        if (lane == 0) total += scan[255];
        __syncthreads();
    }
    if (lane == 0) offsets[count] = total;
}

// MT19937's recurrence has a minimum dependency distance of 227 words.
// Generate those words in parallel, retaining its exact state between batches.
// Tempering and uint32-to-uniform conversion match std::mt19937 and libstdc++'s
// uniform_real_distribution<float>, rather than substituting a different RNG.
__global__ void randomSamplesKernel(unsigned* state, unsigned* cursor,
                                     const unsigned* offsets, size_t count, float* samples) {
    __shared__ unsigned ring[624];
    for (unsigned k = threadIdx.x; k < 624; k += blockDim.x) ring[k] = state[k];
    unsigned start = *cursor;
    unsigned draws = offsets[count];
    __syncthreads();
    for (unsigned base = 0; base < draws; base += 227) {
        unsigned lane = threadIdx.x;
        unsigned index = (start+base+lane)%624;
        unsigned value = 0;
        bool active = lane < 227 && base+lane < draws;
        if (active) {
            unsigned y = (ring[index]&0x80000000u) | (ring[(index+1)%624]&0x7fffffffu);
            value = ring[(index+397)%624] ^ (y>>1) ^ ((y&1) ? 0x9908b0dfu : 0u);
        }
        __syncthreads();
        if (active) {
            ring[index] = value;
            value ^= value>>11;
            value ^= (value<<7)&0x9d2c5680u;
            value ^= (value<<15)&0xefc60000u;
            value ^= value>>18;
            float uniform = float(value)*0x1p-32f;
            samples[base+lane] = fminf(uniform, 0x1.fffffep-1f);
        }
        __syncthreads();
    }
    for (unsigned k = threadIdx.x; k < 624; k += blockDim.x) state[k] = ring[k];
    if (threadIdx.x == 0) *cursor = (start+draws)%624;
}

// Sixteen adjacent lanes trace one pair's rays. Sum in the original order,
// avoiding floating-point atomics and nondeterministic reduction trees.
__global__ void formFactorKernel(const Triangle* triangles, const GpuNode* nodes,
                                 size_t nodeCount, const size_t* indices,
                                 const float4* samples, const unsigned* offsets, size_t first, size_t count,
                                 size_t n, float* kij) {
    __shared__ float contributions[128];
    size_t localPair = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_RAYS;
    int ray = threadIdx.x % NUM_RAYS;
    float value = 0;
    size_t i = 0, j = 0;
    if (localPair < count) {
        size_t pair = first + localPair;
        i = pair / n; j = pair % n;
        Triangle a = triangles[i], b = triangles[j];
        if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
            float4 sample = samples[offsets[localPair]/4 + ray];
            Vec3 from = sampledPoint(a, sample.x, sample.y);
            Vec3 to = sampledPoint(b, sample.z, sample.w);
            if (!blocked(from, to, triangles, nodes, nodeCount, indices, i, j)) {
                Vec3 v = to-from;
                float squared = v.squaredNorm();
                if (squared >= EPSILON) {
                    float length = v.norm();
                    float ca = fmaxf(0.0f, v.dot(a.normal()) / length);
                    float cb = fmaxf(0.0f, (-v).dot(b.normal()) / length);
                    if (ca > 0 && cb > 0) value = (ca*cb)/(PI*squared);
                }
            }
        }
    }
    contributions[threadIdx.x] = value;
    __syncthreads();
    if (ray == 0 && localPair < count) {
        float sum = 0;
        for (int r = 0; r < NUM_RAYS; ++r) sum += contributions[threadIdx.x+r];
        // Transposed matrices coalesce accesses across receivers during propagation.
        kij[j*n+i] = sum * INV_NUM_RAYS;
    }
}

__global__ void timeDelayKernel(const Triangle* triangles, size_t n, int* tau) {
    size_t pair = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair >= n*n) return;
    size_t i = pair % n, j = pair / n;
    tau[pair] = i == j ? 0 : int(ceilf((triangles[i].center()-triangles[j].center()).norm()*INV_WAVE_SPEED));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    std::vector<GpuNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    DeviceBuffer<GpuNode> gpuNodes;
    DeviceBuffer<size_t> gpuIndices;
    gpuNodes.allocate(nodes.size()); gpuIndices.allocate(indices.size());
    cudaCheck(cudaMemcpy(gpuNodes.data, nodes.data(), nodes.size()*sizeof(GpuNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpuIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));

    // Bound temporary storage independently of mesh size. All random samples
    // and geometry remain on the GPU throughout precomputation.
    constexpr size_t batchSize = 8192;
    DeviceBuffer<float4> gpuSamples;
    DeviceBuffer<unsigned> offsets, randomState, cursor;
    gpuSamples.allocate(batchSize*NUM_RAYS);
    offsets.allocate(batchSize+1); randomState.allocate(624); cursor.allocate(1);
    unsigned seed[624];
    seed[0] = 42;
    for (unsigned k = 1; k < 624; ++k)
        seed[k] = 1812433253u*(seed[k-1]^(seed[k-1]>>30))+k;
    cudaCheck(cudaMemcpy(randomState.data, seed, sizeof(seed), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(cursor.data, 0, sizeof(unsigned)));
    const size_t n = state.numTriangles;
    for (size_t first = 0; first < n*n; first += batchSize) {
        size_t count = std::min(batchSize, n*n-first);
        sampleOffsetsKernel<<<1, 256>>>(state.gpuTriangles.data, n, first, count, offsets.data);
        cudaCheck(cudaGetLastError());
        randomSamplesKernel<<<1, 256>>>(randomState.data, cursor.data, offsets.data, count,
                                      reinterpret_cast<float*>(gpuSamples.data));
        cudaCheck(cudaGetLastError());
        formFactorKernel<<<(count*NUM_RAYS+127)/128, 128>>>(
            state.gpuTriangles.data, gpuNodes.data, nodes.size(), gpuIndices.data,
            gpuSamples.data, offsets.data, first, count, n, state.gpuKij.data);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles, t = state.numTimesteps;
    state.gpuTriangles.allocate(n);
    state.gpuKij.allocate(n*n); state.gpuTau.allocate(n*n);
    state.gpuRadB.allocate(t*n); state.gpuDistances.allocate(n);
    state.gpuAreas.allocate(n); state.gpuRho.allocate(n);
    cudaCheck(cudaMemcpy(state.gpuTriangles.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(state.gpuAreas.data, state.areas.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(state.gpuRho.data, state.rho.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(state.gpuRadB.data, 0, t*n*sizeof(float)));
    timeDelayKernel<<<(n*n+255)/256, 256>>>(state.gpuTriangles.data, n, state.gpuTau.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

__global__ void propagationKernel(size_t n, size_t t, size_t timeOff, size_t source,
                                   const int* tau, const float* kij, const float* areas,
                                   const float* rho, float* radB) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float sum = 0;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j*n+i];
        if (t < size_t(delay)) continue;
        float k = kij[j*n+i];
        if (k <= ZERO) continue;
        float rad = radB[(t-size_t(delay))*n+j];
        if (rad <= ZERO) continue;
        sum += fminf(k*areas[j], ONE)*rad;
    }
    radB[t*n+i] = rho[i]*sum + (i == source && t < timeOff ? ONE : ZERO);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<(state.numTriangles+127)/128, 128>>>(state.numTriangles, t,
            state.numTimesteps/2, state.sourceIndex, state.gpuTau.data, state.gpuKij.data,
            state.gpuAreas.data, state.gpuRho.data, state.gpuRadB.data);
        cudaCheck(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void correlationKernel(size_t n, size_t timesteps, size_t source,
                                   const float* radB, float* correlations) {
    size_t p = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p >= n*timesteps) return;
    size_t i = p % n, lag = p / n;
    float sum = 0;
    for (size_t tt = lag; tt < timesteps; ++tt)
        sum += radB[(tt-lag)*n+source]*radB[tt*n+i];
    correlations[p] = sum;
}

__global__ void distanceKernel(size_t n, size_t timesteps, const float* correlations,
                               float* distances) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float maxCorr = 0;
    size_t best = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        float value = correlations[t*n+i];
        if (value > maxCorr) { maxCorr = value; best = t; }
    }
    distances[i] = WAVE_SPEED*float(best);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = state.numTriangles, t = state.numTimesteps;
    DeviceBuffer<float> correlations;
    correlations.allocate(n*t);
    if (t) {
        correlationKernel<<<(n*t+255)/256, 256>>>(n, t, state.sourceIndex, state.gpuRadB.data, correlations.data);
        cudaCheck(cudaGetLastError());
    }
    distanceKernel<<<(n+255)/256, 256>>>(n, t, correlations.data, state.gpuDistances.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(state.distances.data(), state.gpuDistances.data, n*sizeof(float), cudaMemcpyDeviceToHost));
    if (state.validationEnabled) {
        cudaCheck(cudaMemcpy(state.radB.data(), state.gpuRadB.data, n*t*sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(state.kij.data(), state.gpuKij.data, n*n*sizeof(float), cudaMemcpyDeviceToHost));
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
    state.validationEnabled = validate;
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
