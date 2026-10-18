/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI/OpenMP/CUDA implementation of room impulse response
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
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>
#include <sstream>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

int worldRank = 0, worldSize = 1;
void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", worldRank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
// Only rank zero emits the benchmark's public output.
template<class... Args> int rootPrintf(const char* format, Args... args) {
    return worldRank == 0 ? std::printf(format, args...) : 0;
}
int rootPrintf(const char* text) {
    return worldRank == 0 ? std::fputs(text, stdout) : 0;
}
#define printf rootPrintf
#define HD __host__ __device__

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
                #pragma omp task firstprivate(i, childMin, childMax) shared(childIndices) if(indices.size() > 512)
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
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

    if (fabsf(det) < EPSILON) return FLT_MAX;

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
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(0.0f, v.dot(normal) / vNorm);
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
    int minimumDelay = std::numeric_limits<int>::max();
    size_t rowBegin = 0, rowCount = 0;
    std::vector<int> counts, offsets;
    Triangle* deviceTriangles = nullptr;
    val_t *deviceKij = nullptr, *deviceRad = nullptr, *deviceAreas = nullptr;
    int* deviceTau = nullptr;
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
    #pragma omp parallel
    {
        #pragma omp single
        state.octree.build(state.triangles);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.counts.resize(worldSize);
    state.offsets.resize(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        state.offsets[r] = state.numTriangles * r / worldSize;
        state.counts[r] = state.numTriangles * (r + 1) / worldSize - state.offsets[r];
    }
    state.rowBegin = state.offsets[worldRank];
    state.rowCount = state.counts[worldRank];
    state.kij.resize(state.rowCount * state.numTriangles, ZERO);
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

template<class T> void deviceAllocate(T*& p, size_t count) {
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(T)));
}

// Preorder octree with escape links: stackless traversal retains the original
// leaf tests and child order, including triangles duplicated across leaves.
struct DeviceNode {
    Vec3 center, half;
    int escape, first, count;
    HD bool intersects(Vec3 p1, Vec3 p2) const {
        Vec3 d = (p2 - p1) * 0.5f, c = p1 + d - center;
        Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
        if (fabsf(c.x) > half.x + ad.x || fabsf(c.y) > half.y + ad.y ||
            fabsf(c.z) > half.z + ad.z) return false;
        if (fabsf(d.y*c.z-d.z*c.y) > half.y*ad.z+half.z*ad.y+EPSILON) return false;
        if (fabsf(d.z*c.x-d.x*c.z) > half.z*ad.x+half.x*ad.z+EPSILON) return false;
        if (fabsf(d.x*c.y-d.y*c.x) > half.x*ad.y+half.y*ad.x+EPSILON) return false;
        return true;
    }
};
void flattenTree(const Octree& tree, std::vector<DeviceNode>& nodes, std::vector<idx_t>& indices) {
    int index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    for (size_t i : tree.triangleIndices) indices.push_back(i);
    for (const auto& child : tree.children) if (child) flattenTree(*child, nodes, indices);
    nodes[index].escape = nodes.size();
}
__device__ bool blocked(Vec3 from, Vec3 to, size_t i, size_t j,
                       const Triangle* triangles, const DeviceNode* nodes, const idx_t* indices) {
    Vec3 dir = to - from;
    float length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    for (int node = 0; node < nodes[0].escape;) {
        DeviceNode n = nodes[node];
        if (node != 0 && !n.intersects(from, to)) { node = n.escape; continue; }
        for (int k = 0; k < n.count; ++k) {
            idx_t tri = indices[n.first + k];
            if (tri == i || tri == j) continue;
            const Triangle& t = triangles[tri];
            float distance = rayTriangleIntersect(from, dir, t.a, t.b, t.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

// Cooperative, exact MT19937 twist. The three phases preserve the in-place
// recurrence used by std::mt19937, rather than changing the sampling stream.
__device__ void twist(uint32_t* mt) {
    __shared__ uint32_t old[624];
    int lane = threadIdx.x;
    for (int k = lane; k < 624; k += blockDim.x) old[k] = mt[k];
    __syncthreads();
    for (int base = 0; base < 623; base += 227) {
        int end = min(base + 227, 623);
        for (int k = base + lane; k < end; k += blockDim.x) {
            uint32_t y = (old[k] & 0x80000000U) | (old[k+1] & 0x7fffffffU);
            mt[k] = mt[(k+397)%624] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
        }
        __syncthreads();
    }
    if (lane == 0) {
        uint32_t y = (old[623] & 0x80000000U) | (mt[0] & 0x7fffffffU);
        mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfU : 0U);
    }
    __syncthreads();
}
__device__ Vec3 sampledPoint(const Triangle& tri, const float* samples) {
    float u = samples[0], v = samples[1];
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}
__global__ void formFactorsKernel(const Triangle* tris, const DeviceNode* nodes,
                                 const idx_t* indices, const uint32_t* checkpoints,
                                 float* kij, int* tau, int* minimumDelay, size_t n, size_t begin) {
    constexpr int tile = 64;
    __shared__ uint32_t mt[624];
    __shared__ float samples[tile*64];
    __shared__ int offset[tile], count, position;
    int lane = threadIdx.x;
    size_t row = blockIdx.x, i = begin + row;
    int localMinimum = 0x7fffffff;
    const Triangle a = tris[i];
    for (int k = lane; k < 624; k += blockDim.x) mt[k] = checkpoints[row*625+k];
    if (lane == 0) position = checkpoints[row*625+624];
    __syncthreads();
    for (size_t base = 0; base < n; base += tile) {
        if (lane == 0) {
            count = 0;
            for (int k = 0; k < tile; ++k) {
                size_t j = base+k;
                offset[k] = -1;
                if (j < n && i != j && a.normal().dot(tris[j].normal()) <= 0.99f) {
                    offset[k] = count; count += 64;
                }
            }
        }
        __syncthreads();
        int done = 0;
        while (done < count) {
            if (position == 624) {
                twist(mt);
                if (lane == 0) position = 0;
                __syncthreads();
            }
            int batch = min(624-position, count-done);
            for (int k = lane; k < batch; k += blockDim.x) {
                uint32_t y = mt[position+k];
                y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680U;
                y ^= (y << 15) & 0xefc60000U; y ^= y >> 18;
                samples[done+k] = fminf(static_cast<float>(y)*0x1p-32f, 0x1.fffffep-1f);
            }
            __syncthreads();
            if (lane == 0) position += batch;
            done += batch;
            __syncthreads();
        }
        size_t j = base + lane;
        if (j < n) {
            const Triangle b = tris[j];
            tau[j*gridDim.x+row] = i == j ? 0 : static_cast<int>(ceilf((a.center()-b.center()).norm()*INV_WAVE_SPEED));
            if (i != j) localMinimum = min(localMinimum, tau[j*gridDim.x+row]);
            float value = 0;
            if (offset[lane] >= 0) {
                for (int r = 0; r < NUM_RAYS; ++r) {
                    const float* random = samples + offset[lane] + r*4;
                    Vec3 p = sampledPoint(a, random), q = sampledPoint(b, random+2);
                    if (blocked(p, q, i, j, tris, nodes, indices)) continue;
                    Vec3 v = q-p;
                    float d = v.squaredNorm();
                    if (d < EPSILON) continue;
                    float ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
                    if (ci > 0 && cj > 0) value += (ci*cj)/(PI*d);
                }
            }
            kij[j*gridDim.x+row] = value * INV_NUM_RAYS;
        }
        __syncthreads();
    }
    offset[lane] = localMinimum;
    __syncthreads();
    for (int stride = 32; stride; stride /= 2) {
        if (lane < stride) offset[lane] = min(offset[lane], offset[lane+stride]);
        __syncthreads();
    }
    if (lane == 0) atomicMin(minimumDelay, offset[0]);
}

void computeTimeDelays(SimulationState&) {
    printf("Computing time delays and form factors on CUDA devices...\n");
}
void computeFormFactors(SimulationState& state) {
    size_t n = state.numTriangles;
    std::vector<size_t> draws(n);
    std::vector<uint32_t> checkpoints(worldRank == 0 ? n*625 : 0);
    if (worldRank == 0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t active = 0;
            for (size_t j = 0; j < n; ++j)
                if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) ++active;
            draws[i] = active * NUM_RAYS * 4;
        }
        std::mt19937 rng(42);
        for (size_t i = 0; i < n; ++i) {
            std::stringstream stream;
            stream << rng;
            for (int k = 0; k < 625; ++k) stream >> checkpoints[i*625+k];
            rng.discard(draws[i]);
        }
    }
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = state.counts[r]*625; offsets[r] = state.offsets[r]*625;
    }
    std::vector<uint32_t> local(state.rowCount*625);
    MPI_Scatterv(checkpoints.data(), counts.data(), offsets.data(), MPI_UINT32_T,
                 local.data(), local.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);
    std::vector<DeviceNode> nodes;
    std::vector<idx_t> indices;
    flattenTree(state.octree, nodes, indices);
    DeviceNode* dn; idx_t* di; uint32_t* dr; int* minimumDelay;
    deviceAllocate(minimumDelay, 1);
    cudaCheck(cudaMemcpy(minimumDelay, &state.minimumDelay, sizeof(int), cudaMemcpyHostToDevice));
    deviceAllocate(dn, nodes.size()); deviceAllocate(di, indices.size());
    deviceAllocate(dr, local.size());
    deviceAllocate(state.deviceTriangles, n);
    deviceAllocate(state.deviceKij, state.rowCount*n);
    deviceAllocate(state.deviceTau, state.rowCount*n);
    deviceAllocate(state.deviceRad, state.numTimesteps*n);
    deviceAllocate(state.deviceAreas, n);
    cudaCheck(cudaMemcpy(dn, nodes.data(), nodes.size()*sizeof(DeviceNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(di, indices.data(), indices.size()*sizeof(idx_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dr, local.data(), local.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(state.deviceTriangles, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(state.deviceAreas, state.areas.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(state.deviceRad, 0, state.numTimesteps*n*sizeof(float)));
    if (state.rowCount) formFactorsKernel<<<state.rowCount,64>>>(state.deviceTriangles, dn, di, dr,
        state.deviceKij, state.deviceTau, minimumDelay, n, state.rowBegin);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(state.kij.data(), state.deviceKij, state.kij.size()*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(&state.minimumDelay, minimumDelay, sizeof(int), cudaMemcpyDeviceToHost));
    MPI_Allreduce(MPI_IN_PLACE, &state.minimumDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    cudaCheck(cudaFree(minimumDelay));
    cudaCheck(cudaFree(dn)); cudaCheck(cudaFree(di)); cudaCheck(cudaFree(dr));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    // Implemented below after the CUDA propagation kernel.
    extern void propagate(SimulationState&);
    propagate(state);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

__global__ void propagationKernel(const float* kij, const int* tau, const float* areas,
                                  float* rad, size_t n, size_t rows, size_t begin,
                                  size_t t, size_t timesteps, size_t source, float rho) {
    size_t row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    size_t i = begin+row;
    float sum = 0;
    // j order and arithmetic match the reference; neighboring threads read
    // neighboring receivers in the transposed local matrix.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j*rows+row];
        if (t < static_cast<size_t>(delay)) continue;
        float k = kij[j*rows+row];
        if (k <= 0) continue;
        float value = rad[(t-delay)*n+j];
        if (value <= 0) continue;
        sum += fminf(k*areas[j], ONE)*value;
    }
    rad[t*n+i] = rho*sum + (i == source && t < timesteps/2 ? ONE : ZERO);
}
void propagate(SimulationState& s) {
    size_t n = s.numTriangles;
    // No receiver depends on any timestep in the same minimum-delay window.
    // Exchange a whole window to amortize network and PCIe latency.
    size_t window = std::min(s.numTimesteps, static_cast<size_t>(std::max(1, s.minimumDelay)));
    float* staging = nullptr;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&staging),
        std::max(size_t(1), worldSize > 1 ? 2*n*window : 0)*sizeof(float)));
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t first = 0; first < s.numTimesteps; first += window) {
        size_t batch = std::min(window, s.numTimesteps-first);
        for (size_t t = first; t < first+batch; ++t) {
            if (s.rowCount) propagationKernel<<<(s.rowCount+127)/128,128>>>(s.deviceKij, s.deviceTau,
                s.deviceAreas, s.deviceRad, n, s.rowCount, s.rowBegin, t, s.numTimesteps,
                s.sourceIndex, s.rho[0]);
            cudaCheck(cudaGetLastError());
        }
        if (worldSize > 1) {
            if (s.rowCount) cudaCheck(cudaMemcpy2D(staging+s.rowBegin*batch, s.rowCount*sizeof(float),
                s.deviceRad+first*n+s.rowBegin, n*sizeof(float), s.rowCount*sizeof(float), batch,
                cudaMemcpyDeviceToHost));
            for (int r = 0; r < worldSize; ++r) {
                counts[r] = s.counts[r]*batch; offsets[r] = s.offsets[r]*batch;
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, staging, counts.data(),
                           offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            for (int r = 0; r < worldSize; ++r)
                for (size_t t = 0; t < batch; ++t)
                    std::memcpy(staging+n*window+t*n+s.offsets[r],
                        staging+offsets[r]+t*s.counts[r], s.counts[r]*sizeof(float));
            cudaCheck(cudaMemcpy(s.deviceRad+first*n, staging+n*window,
                                batch*n*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    cudaCheck(cudaMemcpy(s.radB.data(), s.deviceRad, s.radB.size()*sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFreeHost(staging));
}
__global__ void distancesKernel(const float* rad, float* distances, size_t n, size_t timesteps,
                                size_t begin, size_t rows, size_t source) {
    size_t row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= rows) return;
    float maximum = 0;
    size_t best = 0;
    for (size_t lag = 0; lag < timesteps; ++lag) {
        float sum = 0;
        for (size_t t = lag; t < timesteps; ++t)
            sum += rad[(t-lag)*n+source]*rad[t*n+begin+row];
        if (sum > maximum) { maximum = sum; best = lag; }
    }
    distances[row] = WAVE_SPEED*static_cast<float>(best);
}
void computeDistances(SimulationState& state) {
    printf("Computing distances via CUDA cross-correlation...\n");
    float* distances;
    deviceAllocate(distances, state.rowCount);
    if (state.rowCount) distancesKernel<<<(state.rowCount+127)/128,128>>>(state.deviceRad, distances,
        state.numTriangles, state.numTimesteps, state.rowBegin, state.rowCount, state.sourceIndex);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(state.distances.data()+state.rowBegin, distances,
                        state.rowCount*sizeof(float), cudaMemcpyDeviceToHost));
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.counts.data(),
                   state.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
    cudaCheck(cudaFree(distances));
    cudaCheck(cudaFree(state.deviceTriangles)); cudaCheck(cudaFree(state.deviceKij));
    cudaCheck(cudaFree(state.deviceTau)); cudaCheck(cudaFree(state.deviceRad));
    cudaCheck(cudaFree(state.deviceAreas));
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
    for (size_t i = 0; i < state.kij.size(); ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    MPI_Allreduce(MPI_IN_PLACE, &nonZeroKij, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct FinalizeMPI { ~FinalizeMPI() { MPI_Finalize(); } } finalizeMPI;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
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
    if (timesteps < 0) {
        printf("Number of timesteps must be non-negative\n");
        return 1;
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices));

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
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);
    MPI_Barrier(MPI_COMM_WORLD);

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
    if (printResults && worldRank == 0) {
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
