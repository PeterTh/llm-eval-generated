/**
 * Room Response Simulation Benchmark
 * 
 * This is a parallel implementation of room impulse response
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
#include <type_traits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

// Every rank owns a contiguous range of destination triangles.
int mpiRank = 0;
int mpiSize = 1;
size_t rowStart(size_t n) { return n * static_cast<size_t>(mpiRank) / static_cast<size_t>(mpiSize); }
size_t rowEnd(size_t n) { return n * static_cast<size_t>(mpiRank + 1) / static_cast<size_t>(mpiSize); }

struct MpiSession {
    MpiSession(int& argc, char**& argv) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        if (provided < MPI_THREAD_FUNNELED) {
            if (mpiRank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm localComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank,
                            MPI_INFO_NULL, &localComm);
        int localRank = 0;
        MPI_Comm_rank(localComm, &localRank);
        MPI_Comm_free(&localComm);
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            fprintf(stderr, "Rank %d: no usable CUDA device: %s\n", mpiRank,
                    cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        error = cudaSetDevice(localRank % deviceCount);
        if (error != cudaSuccess) {
            fprintf(stderr, "Rank %d: cannot select CUDA device: %s\n", mpiRank,
                    cudaGetErrorString(error));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    ~MpiSession() { MPI_Finalize(); }
};

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
    std::vector<val_t> kij;         // Form factors for local rows, row-major
    std::vector<int> tau;           // Time delays for local rows, row-major
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

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

    if (mpiRank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    const size_t localRows = rowEnd(state.numTriangles) - rowStart(state.numTriangles);
    state.kij.resize(localRows * state.numTriangles, ZERO);
    state.tau.resize(localRows * state.numTriangles, 0);
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

// A compact, read-only representation of the host octree for device traversal.
struct DeviceOctreeNode {
    float3 center, halfExtent;
    int child[8];
    int first, count;
};

static int flattenOctree(const Octree& tree, std::vector<DeviceOctreeNode>& nodes,
                         std::vector<idx_t>& indices) {
    const int index = static_cast<int>(nodes.size());
    DeviceOctreeNode node{};
    node.center = make_float3(tree.center.x, tree.center.y, tree.center.z);
    node.halfExtent = make_float3(tree.halfExtent.x, tree.halfExtent.y, tree.halfExtent.z);
    std::fill(node.child, node.child + 8, -1);
    node.first = static_cast<int>(indices.size());
    node.count = static_cast<int>(tree.triangleIndices.size());
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<idx_t>(i));
    nodes.push_back(node);
    for (int i = 0; i < 8; ++i) {
        if (tree.children[i]) {
            int child = flattenOctree(*tree.children[i], nodes, indices);
            nodes[index].child[i] = child;
        }
    }
    return index;
}

static void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

__device__ static float3 dadd(float3 a, float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ static float3 dsub(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ static float3 dmul(float3 a, float s) { return make_float3(a.x*s,a.y*s,a.z*s); }
__device__ static float ddot(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ static float3 dcross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ static float3 dv(const Vec3& v) { return make_float3(v.x,v.y,v.z); }
__device__ static float3 dcenter(const Triangle& t) {
    float3 x = dadd(dadd(dv(t.a), dv(t.b)), dv(t.c));
    return make_float3(x.x / 3.0f, x.y / 3.0f, x.z / 3.0f);
}
__device__ static float3 dpoint(const Triangle& t, const float* r) {
    float u=r[0], v=r[1];
    if (u+v > 1.0f) { u=1.0f-u; v=1.0f-v; }
    return dadd(dadd(dv(t.a),dmul(dsub(dv(t.b),dv(t.a)),u)),dmul(dsub(dv(t.c),dv(t.a)),v));
}
__device__ static bool dbox(float3 p1, float3 p2, const DeviceOctreeNode& node) {
    float3 d=dmul(dsub(p2,p1),0.5f), c=dsub(dadd(p1,d),node.center);
    float3 ad=make_float3(fabsf(d.x),fabsf(d.y),fabsf(d.z)), h=node.halfExtent;
    if (fabsf(c.x)>h.x+ad.x || fabsf(c.y)>h.y+ad.y || fabsf(c.z)>h.z+ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y)>h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z)>h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x)>h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}
__device__ static bool dhit(float3 orig, float3 dir, const Triangle& tri, float rayLen) {
    float3 e1=dsub(dv(tri.b),dv(tri.a)), e2=dsub(dv(tri.c),dv(tri.a));
    float3 p=dcross(dir,e2);
    float det=ddot(e1,p);
    if (fabsf(det)<EPSILON) return false;
    float invDet=1.0f/det;
    float3 tv=dsub(orig,dv(tri.a));
    float u=ddot(tv,p)*invDet;
    if (u<0.0f || u>1.0f) return false;
    float3 q=dcross(tv,e1);
    float v=ddot(dir,q)*invDet;
    if (v<0.0f || u+v>1.0f) return false;
    float distance=ddot(e2,q)*invDet;
    return distance>EPSILON && distance<rayLen-EPSILON;
}
__device__ static bool dblocked(float3 from, float3 to, const DeviceOctreeNode* nodes,
                                const idx_t* indices, const Triangle* triangles,
                                size_t src, size_t dst) {
    float3 dir=dsub(to,from);
    float len=sqrtf(ddot(dir,dir));
    if (len<EPSILON) return true;
    dir=make_float3(dir.x/len,dir.y/len,dir.z/len);
    // The room's root has diameter 20 and subdivision stops below 0.5,
    // so at most 7 siblings per level are pending (depth at most 7).
    int stack[64];
    int top=0;
    stack[top++]=0;
    while (top) {
        const DeviceOctreeNode& node=nodes[stack[--top]];
        if (node.count) {
            for (int k=0;k<node.count;++k) {
                idx_t index=indices[node.first+k];
                if (index!=src && index!=dst && dhit(from,dir,triangles[index],len)) return true;
            }
        } else {
            for (int k=7;k>=0;--k) {
                int child=node.child[k];
                if (child>=0 && dbox(from,to,nodes[child])) stack[top++]=child;
            }
        }
    }
    return false;
}

__global__ static void delayKernel(const Triangle* triangles, int* delays, size_t n,
                                   size_t first, size_t rows) {
    size_t p=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p>=rows*n) return;
    size_t i=first+p/n, j=p%n;
    if (i==j) { delays[p]=0; return; }
    float3 diff=dsub(dcenter(triangles[i]),dcenter(triangles[j]));
    delays[p]=static_cast<int>(ceilf(sqrtf(ddot(diff,diff))*INV_WAVE_SPEED));
}

__global__ static void factorKernel(const Triangle* triangles, const DeviceOctreeNode* nodes,
                                    const idx_t* indices, const float* samples, float* factors,
                                    size_t n, size_t first, size_t rows) {
    size_t p=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p>=rows*n) return;
    size_t i=first+p/n, j=p%n;
    const Triangle& ti=triangles[i];
    const Triangle& tj=triangles[j];
    float3 ni=dv(ti._normal), nj=dv(tj._normal);
    if (i==j || ddot(ni,nj)>0.99f) { factors[p]=0.0f; return; }
    float sum=0.0f;
    const float* r=samples+p*(NUM_RAYS*4);
    for (int k=0;k<NUM_RAYS;++k) {
        float3 pi=dpoint(ti,r+4*k), pj=dpoint(tj,r+4*k+2);
        if (dblocked(pi,pj,nodes,indices,triangles,i,j)) continue;
        float3 v=dsub(pj,pi);
        float dist2=ddot(v,v);
        if (dist2<EPSILON) continue;
        float len=sqrtf(dist2);
        float ci=len<=EPSILON ? 0.0f : fmaxf(0.0f,ddot(v,ni)/len);
        float3 nv=dmul(v,-1.0f);
        float cj=len<=EPSILON ? 0.0f : fmaxf(0.0f,ddot(nv,nj)/len);
        if (ci<=0.0f || cj<=0.0f) continue;
        sum+=(ci*cj)/(PI*dist2);
    }
    factors[p]=sum*INV_NUM_RAYS;
}

void computeTimeDelays(SimulationState& state) {
    if (mpiRank==0) printf("Computing time delays (Tau)...\n");
    size_t n=state.numTriangles, first=rowStart(n), rows=rowEnd(n)-first;
    Triangle* dt=nullptr; int* out=nullptr;
    cudaCheck(cudaMalloc(&dt,n*sizeof(Triangle)));
    cudaCheck(cudaMemcpy(dt,state.triangles.data(),n*sizeof(Triangle),cudaMemcpyHostToDevice));
    if (rows) {
        cudaCheck(cudaMalloc(&out,rows*n*sizeof(int)));
        delayKernel<<<static_cast<unsigned>((rows*n+255)/256),256>>>(dt,out,n,first,rows);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(state.tau.data(),out,rows*n*sizeof(int),cudaMemcpyDeviceToHost));
        cudaCheck(cudaFree(out));
    }
    cudaCheck(cudaFree(dt));
}

void computeFormFactors(SimulationState& state) {
    if (mpiRank==0) printf("Computing form factors (Kij)...\n");
    const size_t n=state.numTriangles, first=rowStart(n), last=rowEnd(n);
    std::vector<DeviceOctreeNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(state.octree,nodes,indices);
    Triangle* dt=nullptr; DeviceOctreeNode* dn=nullptr; idx_t* di=nullptr;
    cudaCheck(cudaMalloc(&dt,n*sizeof(Triangle)));
    cudaCheck(cudaMalloc(&dn,nodes.size()*sizeof(DeviceOctreeNode)));
    cudaCheck(cudaMalloc(&di,indices.size()*sizeof(idx_t)));
    cudaCheck(cudaMemcpy(dt,state.triangles.data(),n*sizeof(Triangle),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dn,nodes.data(),nodes.size()*sizeof(DeviceOctreeNode),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(di,indices.data(),indices.size()*sizeof(idx_t),cudaMemcpyHostToDevice));
    // Build one checkpoint per row of the original serial generator stream.
    // The count pass is parallel; only rank zero advances the generator, then
    // scatters the checkpoints. Each row can subsequently generate independently.
    static_assert(std::is_trivially_copyable<RandomGenerator>::value,
                  "RNG checkpoints must be transferable between MPI ranks");
    std::vector<RandomGenerator> checkpoints;
    std::vector<int> counts(mpiSize), displacements(mpiSize);
    for (int r=0;r<mpiSize;++r) {
        size_t begin=n*static_cast<size_t>(r)/mpiSize;
        size_t end=n*static_cast<size_t>(r+1)/mpiSize;
        counts[r]=static_cast<int>((end-begin)*sizeof(RandomGenerator));
        displacements[r]=static_cast<int>(begin*sizeof(RandomGenerator));
    }
    if (mpiRank==0) {
        std::vector<size_t> draws(n);
        #pragma omp parallel for schedule(static)
        for (size_t i=0;i<n;++i) {
            size_t count=0;
            for (size_t j=0;j<n;++j)
                if (i!=j && state.triangles[i].normal().dot(state.triangles[j].normal())<=0.99f)
                    ++count;
            draws[i]=count*(NUM_RAYS*4);
        }
        checkpoints.resize(n);
        RandomGenerator rng(42);
        for (size_t i=0;i<n;++i) {
            checkpoints[i]=rng;
            rng.discard(draws[i]);
        }
    }
    std::vector<RandomGenerator> localCheckpoints(last-first);
    MPI_Scatterv(checkpoints.data(),counts.data(),displacements.data(),MPI_BYTE,
                 localCheckpoints.data(),counts[mpiRank],MPI_BYTE,0,MPI_COMM_WORLD);
    // Limit temporary sample storage while exposing many independent pairs per launch.
    const size_t chunkRows=std::max<size_t>(1,std::min<size_t>(32, (16*1024*1024)/(n*NUM_RAYS*4*sizeof(float))));
    const size_t capacity=std::min(chunkRows,last-first)*n;
    float *ds=nullptr, *df=nullptr;
    std::vector<float> samples(capacity*NUM_RAYS*4);
    if (capacity) {
        cudaCheck(cudaMalloc(&ds,samples.size()*sizeof(float)));
        cudaCheck(cudaMalloc(&df,capacity*sizeof(float)));
    }
    for (size_t begin=first;begin<last;begin+=chunkRows) {
        size_t rows=std::min(chunkRows,last-begin), pairs=rows*n;
        #pragma omp parallel for schedule(static)
        for (size_t row=0;row<rows;++row) {
            const size_t i=begin+row;
            RandomGenerator rng=localCheckpoints[i-first];
            for (size_t j=0;j<n;++j) {
                if (i!=j && state.triangles[i].normal().dot(state.triangles[j].normal())<=0.99f) {
                    float* sample=samples.data()+(row*n+j)*NUM_RAYS*4;
                    for (int k=0;k<NUM_RAYS*4;++k) sample[k]=rng.rand();
                }
            }
        }
        cudaCheck(cudaMemcpy(ds,samples.data(),pairs*NUM_RAYS*4*sizeof(float),cudaMemcpyHostToDevice));
        factorKernel<<<static_cast<unsigned>((pairs+127)/128),128>>>(dt,dn,di,ds,df,n,begin,rows);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(state.kij.data()+(begin-first)*n,df,pairs*sizeof(float),cudaMemcpyDeviceToHost));
    }
    if (capacity) { cudaCheck(cudaFree(ds)); cudaCheck(cudaFree(df)); }
    cudaCheck(cudaFree(dt)); cudaCheck(cudaFree(dn)); cudaCheck(cudaFree(di));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Each CUDA thread owns one receiving triangle, retaining the original
// accumulation order.  Transposing the rank's rows makes accesses coalesced
// when adjacent threads read the same emitting triangle.
static void simulationCudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", mpiRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void transposeSimulationInputs(const val_t* kij, const int* tau,
                                           val_t* weightsT, int* tauT,
                                           const val_t* areas, size_t n,
                                           size_t localRows) {
    size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n * localRows) return;
    size_t i = k / n, j = k % n;
    size_t out = j * localRows + i;
    weightsT[out] = fminf(kij[k] * areas[j], ONE);
    tauT[out] = tau[k];
}

__global__ void propagationStep(const val_t* weightsT, const int* tauT,
                                const val_t* rho, const val_t* radE,
                                val_t* radB, size_t n, size_t first,
                                size_t localRows, size_t t) {
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= localRows) return;
    size_t i = first + local;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tauT[j * localRows + local];
        if (t < static_cast<size_t>(delay)) continue;
        val_t weight = weightsT[j * localRows + local];
        if (weight <= ZERO) continue;
        val_t rad = radB[(t - static_cast<size_t>(delay)) * n + j];
        if (rad > ZERO) sum += weight * rad;
    }
    radB[t * n + i] = rho[i] * sum + radE[t * n + i];
}

__global__ void correlationKernel(const val_t* radB, val_t* distances,
                                  size_t n, size_t timesteps, size_t source,
                                  size_t first, size_t localRows) {
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= localRows) return;
    size_t i = first + local;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt)
            sum += radB[(tt - t) * n + source] * radB[tt * n + i];
        if (sum > maxCorr) { maxCorr = sum; bestT = static_cast<int>(t); }
    }
    distances[local] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// Retain the complete history on each device: each new timestep needs earlier
// values from all ranks. MPI exchanges just the newly computed timestep.
void runSimulation(SimulationState& state) {
    if (mpiRank == 0) printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles, nt = state.numTimesteps;
    const size_t first = rowStart(n), rows = rowEnd(n) - first;
    const size_t matrixBytes = rows * n * sizeof(val_t);
    val_t *dKij, *dWeights, *dAreas, *dRho, *dEmission, *dRad;
    int *dTau, *dTauT;
    auto alloc = [](void** ptr, size_t bytes) {
        simulationCudaCheck(cudaMalloc(ptr, std::max<size_t>(bytes, 1)), "cudaMalloc");
    };
    alloc(reinterpret_cast<void**>(&dKij), matrixBytes);
    alloc(reinterpret_cast<void**>(&dWeights), matrixBytes);
    alloc(reinterpret_cast<void**>(&dTau), rows * n * sizeof(int));
    alloc(reinterpret_cast<void**>(&dTauT), rows * n * sizeof(int));
    alloc(reinterpret_cast<void**>(&dAreas), n * sizeof(val_t));
    alloc(reinterpret_cast<void**>(&dRho), n * sizeof(val_t));
    alloc(reinterpret_cast<void**>(&dEmission), nt * n * sizeof(val_t));
    alloc(reinterpret_cast<void**>(&dRad), nt * n * sizeof(val_t));
    if (matrixBytes) {
        simulationCudaCheck(cudaMemcpy(dKij, state.kij.data(), matrixBytes, cudaMemcpyHostToDevice), "copy form factors");
        simulationCudaCheck(cudaMemcpy(dTau, state.tau.data(), rows * n * sizeof(int), cudaMemcpyHostToDevice), "copy delays");
    }
    simulationCudaCheck(cudaMemcpy(dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "copy areas");
    simulationCudaCheck(cudaMemcpy(dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice), "copy reflectivity");
    if (nt) {
        simulationCudaCheck(cudaMemcpy(dEmission, state.radE.data(), nt * n * sizeof(val_t), cudaMemcpyHostToDevice), "copy emission");
        simulationCudaCheck(cudaMemset(dRad, 0, nt * n * sizeof(val_t)), "initialize radiosity");
    }
    constexpr int block = 256;
    if (rows) {
        transposeSimulationInputs<<<(rows * n + block - 1) / block, block>>>(dKij, dTau, dWeights, dTauT, dAreas, n, rows);
        simulationCudaCheck(cudaGetLastError(), "transpose simulation inputs");
    }
    cudaFree(dKij); cudaFree(dTau); cudaFree(dAreas);

    std::vector<int> counts(mpiSize), displacements(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        size_t begin = n * static_cast<size_t>(rank) / mpiSize;
        size_t end = n * static_cast<size_t>(rank + 1) / mpiSize;
        counts[rank] = static_cast<int>(end - begin);
        displacements[rank] = static_cast<int>(begin);
    }
    for (size_t t = 0; t < nt; ++t) {
        if (rows) {
            propagationStep<<<(rows + block - 1) / block, block>>>(dWeights, dTauT, dRho, dEmission, dRad, n, first, rows, t);
            simulationCudaCheck(cudaGetLastError(), "propagation kernel");
        }
        if (mpiSize > 1) {
            if (rows) simulationCudaCheck(cudaMemcpy(state.radB.data() + t * n + first, dRad + t * n + first, rows * sizeof(val_t), cudaMemcpyDeviceToHost), "copy local radiosity");
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.radB.data() + t * n,
                           counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
            simulationCudaCheck(cudaMemcpy(dRad + t * n, state.radB.data() + t * n, n * sizeof(val_t), cudaMemcpyHostToDevice), "copy global radiosity");
        }
        if (mpiRank == 0 && ((t + 1) % 10 == 0 || t + 1 == nt))
            printf("  Timestep %zu/%zu\n", t + 1, nt);
    }
    if (mpiSize == 1 && nt)
        simulationCudaCheck(cudaMemcpy(state.radB.data(), dRad, nt * n * sizeof(val_t), cudaMemcpyDeviceToHost), "copy radiosity history");
    cudaFree(dWeights); cudaFree(dTauT); cudaFree(dRho); cudaFree(dEmission); cudaFree(dRad);
}

void computeDistances(SimulationState& state) {
    if (mpiRank == 0) printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles, nt = state.numTimesteps;
    const size_t first = rowStart(n), rows = rowEnd(n) - first;
    val_t *dRad, *dDistances;
    simulationCudaCheck(cudaMalloc(reinterpret_cast<void**>(&dRad), std::max<size_t>(nt * n * sizeof(val_t), 1)), "allocate radiosity");
    simulationCudaCheck(cudaMalloc(reinterpret_cast<void**>(&dDistances), std::max<size_t>(rows * sizeof(val_t), 1)), "allocate distances");
    if (nt) simulationCudaCheck(cudaMemcpy(dRad, state.radB.data(), nt * n * sizeof(val_t), cudaMemcpyHostToDevice), "copy radiosity history");
    if (rows) {
        constexpr int block = 256;
        correlationKernel<<<(rows + block - 1) / block, block>>>(dRad, dDistances, n, nt, state.sourceIndex, first, rows);
        simulationCudaCheck(cudaGetLastError(), "correlation kernel");
        simulationCudaCheck(cudaMemcpy(state.distances.data() + first, dDistances, rows * sizeof(val_t), cudaMemcpyDeviceToHost), "copy distances");
    }
    std::vector<int> counts(mpiSize), displacements(mpiSize);
    for (int rank = 0; rank < mpiSize; ++rank) {
        size_t begin = n * static_cast<size_t>(rank) / mpiSize;
        size_t end = n * static_cast<size_t>(rank + 1) / mpiSize;
        counts[rank] = static_cast<int>(end - begin);
        displacements[rank] = static_cast<int>(begin);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(), counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    cudaFree(dRad); cudaFree(dDistances);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
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
    MpiSession mpi(argc, argv);
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
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
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
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);
    MPI_Barrier(MPI_COMM_WORLD);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    unsigned long long nonZeroKij = 0;
    if (validate) {
        unsigned long long localNonZeroKij = 0;
        #pragma omp parallel for reduction(+:localNonZeroKij)
        for (size_t i = 0; i < state.kij.size(); ++i) {
            if (state.kij[i] > EPSILON) ++localNonZeroKij;
        }
        MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Only the root rank reports and validates the assembled result.
    if (mpiRank != 0) return 0;

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
        if (!validateResults(state, nonZeroKij)) {
            return 1;
        }
    }

    return 0;
}
