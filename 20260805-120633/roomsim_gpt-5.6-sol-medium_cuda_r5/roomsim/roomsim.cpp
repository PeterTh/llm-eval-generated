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

#define HD __host__ __device__

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
    HD Vec3 normalized() const {
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

struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    uint32_t firstTriangle;
    uint32_t triangleCount;
};

struct GpuStorage {
    Triangle* triangles = nullptr;
    FlatOctreeNode* nodes = nullptr;
    uint32_t* nodeTriangles = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    ~GpuStorage() {
        cudaFree(triangles);
        cudaFree(nodes);
        cudaFree(nodeTriangles);
        cudaFree(areas);
        cudaFree(rho);
        cudaFree(kij);
        cudaFree(tau);
        cudaFree(radB);
        cudaFree(distances);
    }
};

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
    std::unique_ptr<GpuStorage> gpu;

    size_t sourceIndex;
    bool validationEnabled = false;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

static int flattenOctree(const Octree& tree, std::vector<FlatOctreeNode>& nodes,
                         std::vector<uint32_t>& triangleIndices) {
    const int nodeIndex = static_cast<int>(nodes.size());
    FlatOctreeNode node{};
    node.center = tree.center;
    node.halfExtent = tree.halfExtent;
    std::fill(std::begin(node.children), std::end(node.children), -1);
    node.firstTriangle = static_cast<uint32_t>(triangleIndices.size());
    node.triangleCount = static_cast<uint32_t>(tree.triangleIndices.size());
    for (size_t index : tree.triangleIndices) {
        triangleIndices.push_back(static_cast<uint32_t>(index));
    }
    nodes.push_back(node);
    for (int child = 0; child < 8; ++child) {
        if (tree.children[child]) {
            nodes[nodeIndex].children[child] =
                flattenOctree(*tree.children[child], nodes, triangleIndices);
        }
    }
    return nodeIndex;
}

static void prepareGpu(SimulationState& state) {
    if (state.gpu) return;

    state.gpu = std::make_unique<GpuStorage>();
    GpuStorage& gpu = *state.gpu;
    std::vector<FlatOctreeNode> nodes;
    std::vector<uint32_t> nodeTriangles;
    flattenOctree(state.octree, nodes, nodeTriangles);

    const size_t n = state.numTriangles;
    const size_t matrixBytes = n * n * sizeof(val_t);
    cudaCheck(cudaMalloc(&gpu.triangles, n * sizeof(Triangle)), "triangle allocation");
    cudaCheck(cudaMalloc(&gpu.nodes, nodes.size() * sizeof(FlatOctreeNode)), "octree allocation");
    cudaCheck(cudaMalloc(&gpu.nodeTriangles, nodeTriangles.size() * sizeof(uint32_t)), "octree index allocation");
    cudaCheck(cudaMalloc(&gpu.areas, n * sizeof(val_t)), "area allocation");
    cudaCheck(cudaMalloc(&gpu.rho, n * sizeof(val_t)), "reflectivity allocation");
    cudaCheck(cudaMalloc(&gpu.kij, matrixBytes), "form-factor allocation");
    cudaCheck(cudaMalloc(&gpu.tau, n * n * sizeof(int)), "delay allocation");
    cudaCheck(cudaMalloc(&gpu.radB, state.numTimesteps * n * sizeof(val_t)), "radiosity allocation");
    cudaCheck(cudaMalloc(&gpu.distances, n * sizeof(val_t)), "distance allocation");

    cudaCheck(cudaMemcpy(gpu.triangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice), "triangle upload");
    cudaCheck(cudaMemcpy(gpu.nodes, nodes.data(), nodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice), "octree upload");
    cudaCheck(cudaMemcpy(gpu.nodeTriangles, nodeTriangles.data(), nodeTriangles.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "octree index upload");
    cudaCheck(cudaMemcpy(gpu.areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "area upload");
    cudaCheck(cudaMemcpy(gpu.rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "reflectivity upload");
    cudaCheck(cudaMemset(gpu.radB, 0, state.numTimesteps * n * sizeof(val_t)), "radiosity initialization");
}

__device__ __forceinline__ uint32_t randomBits(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return static_cast<uint32_t>((state * 2685821657736338717ULL) >> 32);
}

__device__ __forceinline__ val_t randomUnit(uint64_t& state) {
    return static_cast<val_t>(randomBits(state) >> 8) * 0x1.0p-24f;
}

__device__ __forceinline__ Vec3 randomPoint(const Triangle& triangle, uint64_t& rng) {
    val_t u = randomUnit(rng);
    val_t v = randomUnit(rng);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v;
}

__device__ __forceinline__ bool rayIntersectsBoxGpu(const Vec3& p1, const Vec3& p2,
                                                     const FlatOctreeNode& node) {
    const Vec3 d = (p2 - p1) * 0.5f;
    const Vec3 c = p1 + d - node.center;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ val_t rayTriangleIntersectGpu(const Vec3& origin, const Vec3& direction,
                                                          const Triangle& triangle) {
    const Vec3 e1 = triangle.b - triangle.a;
    const Vec3 e2 = triangle.c - triangle.a;
    const Vec3 p = direction.cross(e2);
    const val_t determinant = e1.dot(p);
    if (fabsf(determinant) < EPSILON) return FLT_MAX;
    const val_t inverse = 1.0f / determinant;
    const Vec3 tv = origin - triangle.a;
    const val_t u = tv.dot(p) * inverse;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    const Vec3 q = tv.cross(e1);
    const val_t v = direction.dot(q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return e2.dot(q) * inverse;
}

__device__ bool rayBlockedGpu(const Vec3& from, const Vec3& to, const Triangle* triangles,
                              const FlatOctreeNode* nodes, const uint32_t* nodeTriangles,
                              uint32_t source, uint32_t destination) {
    const Vec3 ray = to - from;
    const val_t rayLength = ray.norm();
    if (rayLength < EPSILON) return true;
    const Vec3 direction = ray / rayLength;
    int stack[64];
    int stackSize = 1;
    stack[0] = 0;
    while (stackSize) {
        const FlatOctreeNode& node = nodes[stack[--stackSize]];
        if (node.triangleCount) {
            for (uint32_t k = 0; k < node.triangleCount; ++k) {
                const uint32_t index = nodeTriangles[node.firstTriangle + k];
                if (index == source || index == destination) continue;
                const val_t distance = rayTriangleIntersectGpu(from, direction, triangles[index]);
                if (distance > EPSILON && distance < rayLength - EPSILON) return true;
            }
        } else {
            for (int child = 7; child >= 0; --child) {
                const int childIndex = node.children[child];
                if (childIndex >= 0 && rayIntersectsBoxGpu(from, to, nodes[childIndex])) {
                    if (stackSize < 64) stack[stackSize++] = childIndex;
                }
            }
        }
    }
    return false;
}

__global__ void timeDelayKernel(const Triangle* triangles, int* tau, size_t n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    const size_t i = index / n;
    const size_t j = index - i * n;
    tau[index] = i == j ? 0 : static_cast<int>(ceilf((triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
}

__global__ void formFactorKernel(const Triangle* triangles, const FlatOctreeNode* nodes,
                                 const uint32_t* nodeTriangles, val_t* kij, size_t n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    const uint32_t i = static_cast<uint32_t>(index / n);
    const uint32_t j = static_cast<uint32_t>(index - static_cast<size_t>(i) * n);
    if (i == j || triangles[i].normal().dot(triangles[j].normal()) > 0.99f) {
        kij[index] = ZERO;
        return;
    }

    uint64_t rng = (static_cast<uint64_t>(index) + 0x9e3779b97f4a7c15ULL) ^ 42ULL;
    val_t result = ZERO;
    const Triangle triangleI = triangles[i];
    const Triangle triangleJ = triangles[j];
    for (int rayIndex = 0; rayIndex < NUM_RAYS; ++rayIndex) {
        const Vec3 pointI = randomPoint(triangleI, rng);
        const Vec3 pointJ = randomPoint(triangleJ, rng);
        if (rayBlockedGpu(pointI, pointJ, triangles, nodes, nodeTriangles, i, j)) continue;
        const Vec3 v = pointJ - pointI;
        const val_t distanceSquared = v.squaredNorm();
        if (distanceSquared < EPSILON) continue;
        const val_t inverseLength = rsqrtf(distanceSquared);
        const val_t cosineI = fmaxf(ZERO, v.dot(triangleI.normal()) * inverseLength);
        const val_t cosineJ = fmaxf(ZERO, (-v).dot(triangleJ.normal()) * inverseLength);
        if (cosineI > ZERO && cosineJ > ZERO) result += cosineI * cosineJ / (PI * distanceSquared);
    }
    kij[index] = result * INV_NUM_RAYS;
}

__global__ void simulationKernel(size_t timestep, size_t timesteps, size_t sourceIndex,
                                 size_t n, const val_t* kij, const int* tau,
                                 const val_t* areas, const val_t* rho, val_t* radB) {
    const size_t i = blockIdx.x;
    if (i >= n) return;
    val_t partial = ZERO;
    const size_t row = i * n;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (i == j) continue;
        const int delay = tau[row + j];
        if (timestep < static_cast<size_t>(delay)) continue;
        const val_t factor = kij[row + j];
        if (factor <= ZERO) continue;
        const val_t radiosity = radB[(timestep - static_cast<size_t>(delay)) * n + j];
        if (radiosity > ZERO) partial += fminf(factor * areas[j], ONE) * radiosity;
    }
    __shared__ val_t sums[256];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < timesteps / 2) ? ONE : ZERO;
        radB[timestep * n + i] = rho[i] * sums[0] + emission;
    }
}

__global__ void distanceKernel(const val_t* radB, val_t* distances, size_t n,
                               size_t timesteps, size_t sourceIndex) {
    const size_t i = blockIdx.x;
    if (i >= n) return;
    val_t localMaximum = ZERO;
    int localBest = 0;
    for (size_t lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (size_t time = lag; time < timesteps; ++time) {
            correlation += radB[(time - lag) * n + sourceIndex] * radB[time * n + i];
        }
        if (correlation > localMaximum || (correlation == localMaximum && static_cast<int>(lag) < localBest)) {
            localMaximum = correlation;
            localBest = static_cast<int>(lag);
        }
    }
    __shared__ val_t maxima[256];
    __shared__ int bestLags[256];
    maxima[threadIdx.x] = localMaximum;
    bestLags[threadIdx.x] = localBest;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) {
            const val_t other = maxima[threadIdx.x + offset];
            const int otherLag = bestLags[threadIdx.x + offset];
            if (other > maxima[threadIdx.x] || (other == maxima[threadIdx.x] && otherLag < bestLags[threadIdx.x])) {
                maxima[threadIdx.x] = other;
                bestLags[threadIdx.x] = otherLag;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(bestLags[0]);
}

__global__ void countPositiveKernel(const val_t* values, size_t count,
                                    unsigned long long* result) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned localCount = 0;
    for (; index < count; index += stride) localCount += values[index] > EPSILON;
    __shared__ unsigned blockCounts[256];
    blockCounts[threadIdx.x] = localCount;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) blockCounts[threadIdx.x] += blockCounts[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, static_cast<unsigned long long>(blockCounts[0]));
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool validationEnabled) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.validationEnabled = validationEnabled;

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
    if (validationEnabled) state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    prepareGpu(state);
    constexpr unsigned threads = 256;
    const size_t count = state.numTriangles * state.numTriangles;
    formFactorKernel<<<static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        state.gpu->triangles, state.gpu->nodes, state.gpu->nodeTriangles,
        state.gpu->kij, state.numTriangles);
    cudaCheck(cudaGetLastError(), "form-factor kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "form-factor computation");
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    prepareGpu(state);
    constexpr unsigned threads = 256;
    const size_t count = state.numTriangles * state.numTriangles;
    timeDelayKernel<<<static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        state.gpu->triangles, state.gpu->tau, state.numTriangles);
    cudaCheck(cudaGetLastError(), "time-delay kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "time-delay computation");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    prepareGpu(state);
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<static_cast<unsigned>(state.numTriangles), 256>>>(
            t, state.numTimesteps, state.sourceIndex, state.numTriangles,
            state.gpu->kij, state.gpu->tau, state.gpu->areas, state.gpu->rho,
            state.gpu->radB);
        cudaCheck(cudaGetLastError(), "simulation kernel launch");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "wave propagation");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    prepareGpu(state);
    distanceKernel<<<static_cast<unsigned>(state.numTriangles), 256>>>(
        state.gpu->radB, state.gpu->distances, state.numTriangles,
        state.numTimesteps, state.sourceIndex);
    cudaCheck(cudaGetLastError(), "distance kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "distance computation");
    cudaCheck(cudaMemcpy(state.distances.data(), state.gpu->distances,
                         state.numTriangles * sizeof(val_t), cudaMemcpyDeviceToHost),
              "distance download");
    if (state.validationEnabled) {
        cudaCheck(cudaMemcpy(state.radB.data(), state.gpu->radB,
                             state.numTimesteps * state.numTriangles * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "radiosity download");
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
    unsigned long long* deviceCount = nullptr;
    unsigned long long nonZeroKij = 0;
    cudaCheck(cudaMalloc(&deviceCount, sizeof(unsigned long long)), "validation counter allocation");
    cudaCheck(cudaMemset(deviceCount, 0, sizeof(unsigned long long)), "validation counter initialization");
    const size_t matrixCount = state.numTriangles * state.numTriangles;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((matrixCount + 255) / 256, 4096));
    countPositiveKernel<<<blocks, 256>>>(state.gpu->kij, matrixCount, deviceCount);
    cudaCheck(cudaGetLastError(), "validation kernel launch");
    cudaCheck(cudaMemcpy(&nonZeroKij, deviceCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
              "validation count download");
    cudaFree(deviceCount);
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
                         static_cast<size_t>(sourceIdx), reflectivity, validate);

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
