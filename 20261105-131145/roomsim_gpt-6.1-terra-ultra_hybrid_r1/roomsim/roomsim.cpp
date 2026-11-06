/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed room impulse response simulation using MPI, OpenMP,
 * and CUDA for radiosity-based wave propagation. It models how sound/light
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

#include <mpi.h>
#include <omp.h>

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
    void discard(uint64_t count) { rng.discard(count); }
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
    int rank = 0, ranks = 1;
    size_t rowBegin = 0, rowEnd = 0;
    std::vector<int> counts, displacements;

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

    size_t idx2d(size_t i, size_t j) const { return (i - rowBegin) * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// CUDA keeps the local matrix rows transposed so that adjacent threads read
// adjacent elements while each thread accumulates its row in original j order.
static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__global__ void transposeMatrices(const val_t* inK, const int* inTau,
                                  val_t* outK, int* outTau,
                                  size_t rows, size_t n) {
    __shared__ val_t tileK[32][33];
    __shared__ int tileTau[32][33];
    size_t col = static_cast<size_t>(blockIdx.x) * 32 + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (col < n && row + offset < rows) {
            size_t index = (row + offset) * n + col;
            tileK[threadIdx.y + offset][threadIdx.x] = inK[index];
            tileTau[threadIdx.y + offset][threadIdx.x] = inTau[index];
        }
    }
    __syncthreads();
    size_t outRow = static_cast<size_t>(blockIdx.y) * 32 + threadIdx.x;
    size_t outCol = static_cast<size_t>(blockIdx.x) * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (outRow < rows && outCol + offset < n) {
            size_t index = (outCol + offset) * rows + outRow;
            outK[index] = tileK[threadIdx.x][threadIdx.y + offset];
            outTau[index] = tileTau[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void propagateRows(size_t n, size_t rows, size_t begin, size_t t,
                              const val_t* kij, const int* tau,
                              const val_t* areas, const val_t* rho,
                              const val_t* radE, val_t* radB) {
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= rows) return;
    size_t i = begin + local;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t matrixIndex = j * rows + local;
        int delay = tau[matrixIndex];
        if (t < static_cast<size_t>(delay)) continue;
        val_t k = kij[matrixIndex];
        if (k <= ZERO) continue;
        val_t previous = radB[(t - static_cast<size_t>(delay)) * n + j];
        if (previous <= ZERO) continue;
        val_t factor = k * areas[j];
        sum += (factor < ONE ? factor : ONE) * previous;
    }
    radB[t * n + i] = rho[i] * sum + radE[t * n + i];
}

__global__ void correlateRows(size_t n, size_t timesteps, size_t begin,
                              size_t rows, size_t source, const val_t* radB,
                              val_t* distances) {
    size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local >= rows) return;
    size_t i = begin + local;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt) {
            sum += radB[(tt - t) * n + source] * radB[tt * n + i];
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    distances[local] = WAVE_SPEED * static_cast<val_t>(bestT);
}

struct FlatOctreeNode {
    Vec3 center, halfExtent;
    size_t next, first, count;
};

static size_t flattenOctree(const Octree& tree, std::vector<FlatOctreeNode>& nodes,
                            std::vector<size_t>& indices) {
    size_t current = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, indices.size(), tree.triangleIndices.size()});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[current].next = nodes.size();
    return current;
}

__device__ bool gpuBoxOverlap(const FlatOctreeNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

__device__ val_t gpuRayTriangle(const Vec3& orig, const Vec3& dir, const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool gpuBlocked(const Vec3& from, const Vec3& to, size_t src, size_t dst,
                           const Triangle* triangles, const FlatOctreeNode* nodes,
                           size_t nodeCount, const size_t* indices) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;
    size_t node = 0;
    while (node < nodeCount) {
        const FlatOctreeNode& current = nodes[node];
        if (node != 0 && !gpuBoxOverlap(current, from, to)) {
            node = current.next;
        } else if (current.count != 0) {
            for (size_t k = 0; k < current.count; ++k) {
                size_t idx = indices[current.first + k];
                if (idx == src || idx == dst) continue;
                val_t distance = gpuRayTriangle(from, dirNorm, triangles[idx]);
                if (distance > EPSILON && distance < rayLen - EPSILON) return true;
            }
            node = current.next;
        } else {
            ++node;
        }
    }
    return false;
}

__device__ Vec3 gpuRandomPoint(const Triangle& tri, const val_t* random) {
    val_t u = random[0], v = random[1];
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    Vec3 ab = tri.b - tri.a, ac = tri.c - tri.a;
    return tri.a + ab * u + ac * v;
}

__device__ val_t gpuCosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t result = v.dot(normal) / vNorm;
    return result > ZERO ? result : ZERO;
}

__global__ void formFactorPairs(size_t n, size_t firstPair, size_t pairs,
                                const Triangle* triangles, const FlatOctreeNode* nodes,
                                size_t nodeCount, const size_t* indices,
                                const val_t* random, val_t* output) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    size_t pair = firstPair + p, i = pair / n, j = pair % n;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    if (random[p * (4 * NUM_RAYS)] < ZERO) { output[p] = ZERO; return; }
    val_t value = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* samples = random + p * (4 * NUM_RAYS) + r * 4;
        Vec3 pI = gpuRandomPoint(a, samples), pJ = gpuRandomPoint(b, samples + 2);
        if (gpuBlocked(pI, pJ, i, j, triangles, nodes, nodeCount, indices)) continue;
        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;
        val_t cosI = gpuCosPhi(v, a._normal), cosJ = gpuCosPhi(-v, b._normal);
        if (cosI <= ZERO || cosJ <= ZERO) continue;
        value += (cosI * cosJ) / (PI * distSqr);
    }
    output[p] = value * INV_NUM_RAYS;
}

static void selectCudaDevice(int rank) {
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

struct CudaSimulation {
    val_t *kij = nullptr, *areas = nullptr, *rho = nullptr;
    val_t *radE = nullptr, *radB = nullptr, *distances = nullptr;
    int* tau = nullptr;
    size_t n, rows, begin, timesteps;

    explicit CudaSimulation(const SimulationState& state)
        : n(state.numTriangles), rows(state.rowEnd - state.rowBegin),
          begin(state.rowBegin), timesteps(state.numTimesteps) {
        selectCudaDevice(state.rank);

        auto allocate = [](auto** ptr, size_t count) {
            if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(**ptr)));
        };
        allocate(&areas, n);
        allocate(&rho, n);
        allocate(&radE, timesteps * n);
        allocate(&radB, timesteps * n);
        allocate(&distances, rows);
        allocate(&kij, rows * n);
        allocate(&tau, rows * n);
        if (n) {
            CUDA_CHECK(cudaMemcpy(areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        }
        if (timesteps * n) {
            CUDA_CHECK(cudaMemcpy(radE, state.radE.data(), timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(radB, state.radB.data(), timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
        }
        if (rows) {
            val_t* rowK = nullptr;
            int* rowTau = nullptr;
            allocate(&rowK, rows * n);
            allocate(&rowTau, rows * n);
            CUDA_CHECK(cudaMemcpy(rowK, state.kij.data(), rows * n * sizeof(val_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(rowTau, state.tau.data(), rows * n * sizeof(int), cudaMemcpyHostToDevice));
            dim3 block(32, 8);
            dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>((rows + 31) / 32));
            transposeMatrices<<<grid, block>>>(rowK, rowTau, kij, tau, rows, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(rowK));
            CUDA_CHECK(cudaFree(rowTau));
        }
    }
    ~CudaSimulation() {
        cudaFree(kij); cudaFree(tau); cudaFree(areas); cudaFree(rho);
        cudaFree(radE); cudaFree(radB); cudaFree(distances);
    }
};

static std::unique_ptr<CudaSimulation> gpuSimulation;

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
    state.counts.resize(state.ranks);
    state.displacements.resize(state.ranks);
    for (int r = 0; r < state.ranks; ++r) {
        size_t begin = state.numTriangles * static_cast<size_t>(r) / state.ranks;
        size_t end = state.numTriangles * static_cast<size_t>(r + 1) / state.ranks;
        state.counts[r] = static_cast<int>(end - begin);
        state.displacements[r] = static_cast<int>(begin);
    }
    state.rowBegin = static_cast<size_t>(state.displacements[state.rank]);
    state.rowEnd = state.rowBegin + static_cast<size_t>(state.counts[state.rank]);
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
    state.kij.resize((state.rowEnd - state.rowBegin) * state.numTriangles, ZERO);
    state.tau.resize((state.rowEnd - state.rowBegin) * state.numTriangles, 0);
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

void computeFormFactors(SimulationState& state) {
    if (state.rank == 0) printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    selectCudaDevice(state.rank);
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // Preserve the original single MT19937 stream: a culled pair consumes no
    // samples, and every other pair consumes exactly four per ray.
    std::vector<uint64_t> rowDraws(state.rowEnd + 1, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.rowEnd; ++i) {
        uint64_t pairs = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) ++pairs;
        }
        rowDraws[i + 1] = pairs * (4 * NUM_RAYS);
    }
    for (size_t i = 1; i <= state.rowEnd; ++i) rowDraws[i] += rowDraws[i - 1];

    std::vector<FlatOctreeNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    Triangle* deviceTriangles = nullptr;
    FlatOctreeNode* deviceNodes = nullptr;
    size_t* deviceIndices = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceTriangles), n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceNodes), nodes.size() * sizeof(FlatOctreeNode)));
    if (!indices.empty()) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceIndices), indices.size() * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(deviceTriangles, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceNodes, nodes.data(), nodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice));
    if (!indices.empty()) CUDA_CHECK(cudaMemcpy(deviceIndices, indices.data(), indices.size() * sizeof(size_t), cudaMemcpyHostToDevice));

    const int workers = std::max(1, std::min({8, omp_get_max_threads(), static_cast<int>(state.rowEnd - state.rowBegin)}));
    std::vector<RandomGenerator> generators(static_cast<size_t>(workers), RandomGenerator(42));
    RandomGenerator stream(42);
    stream.discard(rowDraws[state.rowBegin]);
    size_t previous = state.rowBegin;
    for (int thread = 0; thread < workers; ++thread) {
        size_t begin = state.rowBegin + (state.rowEnd - state.rowBegin) * static_cast<size_t>(thread) / workers;
        stream.discard(rowDraws[begin] - rowDraws[previous]);
        generators[thread] = stream;
        previous = begin;
    }

    // Each worker owns a stream and a bounded pair buffer. RNG generation and
    // GPU traversal can overlap across workers without changing the sequence.
    #pragma omp parallel num_threads(workers)
    {
        int thread = omp_get_thread_num(), threads = omp_get_num_threads();
        CUDA_CHECK(cudaSetDevice(device));
        size_t begin = state.rowBegin + (state.rowEnd - state.rowBegin) * static_cast<size_t>(thread) / threads;
        size_t end = state.rowBegin + (state.rowEnd - state.rowBegin) * static_cast<size_t>(thread + 1) / threads;
        RandomGenerator rng = generators[thread];
        if (threads != workers) { rng = RandomGenerator(42); rng.discard(rowDraws[begin]); }
        constexpr size_t batchPairs = 8192;
        val_t* random = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&random), batchPairs * 4 * NUM_RAYS * sizeof(val_t)));
        val_t *deviceRandom = nullptr, *deviceOutput = nullptr;
        cudaStream_t cudaStream;
        CUDA_CHECK(cudaStreamCreate(&cudaStream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceRandom), batchPairs * 4 * NUM_RAYS * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), batchPairs * sizeof(val_t)));
        for (size_t first = begin * n; first < end * n; first += batchPairs) {
            size_t count = std::min(batchPairs, end * n - first);
            for (size_t p = 0; p < count; ++p) {
                size_t pair = first + p, i = pair / n, j = pair % n;
                if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) {
                    val_t* samples = random + p * 4 * NUM_RAYS;
                    for (int r = 0; r < 4 * NUM_RAYS; ++r) samples[r] = rng.rand();
                } else {
                    random[p * 4 * NUM_RAYS] = -1.0f;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(deviceRandom, random, count * 4 * NUM_RAYS * sizeof(val_t), cudaMemcpyHostToDevice, cudaStream));
            formFactorPairs<<<static_cast<unsigned>((count + 127) / 128), 128, 0, cudaStream>>>(
                n, first, count, deviceTriangles, deviceNodes, nodes.size(), deviceIndices, deviceRandom, deviceOutput);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(state.kij.data() + first - state.rowBegin * n, deviceOutput,
                                       count * sizeof(val_t), cudaMemcpyDeviceToHost, cudaStream));
            CUDA_CHECK(cudaStreamSynchronize(cudaStream));
        }
        CUDA_CHECK(cudaFree(deviceRandom));
        CUDA_CHECK(cudaFree(deviceOutput));
        CUDA_CHECK(cudaFreeHost(random));
        CUDA_CHECK(cudaStreamDestroy(cudaStream));
    }
    CUDA_CHECK(cudaFree(deviceTriangles));
    CUDA_CHECK(cudaFree(deviceNodes));
    CUDA_CHECK(cudaFree(deviceIndices));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    #pragma omp parallel for schedule(static)
    for (size_t i = state.rowBegin; i < state.rowEnd; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i != j) state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    if (state.rank == 0) printf("Running wave propagation simulation...\n");
    gpuSimulation = std::make_unique<CudaSimulation>(state);
    CudaSimulation& gpu = *gpuSimulation;
    constexpr unsigned threads = 128;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (gpu.rows) {
            propagateRows<<<static_cast<unsigned>((gpu.rows + threads - 1) / threads), threads>>>(
                gpu.n, gpu.rows, gpu.begin, t, gpu.kij, gpu.tau, gpu.areas,
                gpu.rho, gpu.radE, gpu.radB);
            CUDA_CHECK(cudaGetLastError());
            if (state.ranks > 1) {
                CUDA_CHECK(cudaMemcpy(state.radB.data() + t * gpu.n + gpu.begin,
                                      gpu.radB + t * gpu.n + gpu.begin,
                                      gpu.rows * sizeof(val_t), cudaMemcpyDeviceToHost));
            }
        }
        if (state.ranks > 1) {
            val_t* step = state.radB.data() + t * gpu.n;
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, step, state.counts.data(),
                           state.displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(gpu.radB + t * gpu.n, step, gpu.n * sizeof(val_t), cudaMemcpyHostToDevice));
        }
        if (state.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    if (state.ranks == 1 && gpu.timesteps) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), gpu.radB,
                              gpu.timesteps * gpu.n * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rank == 0) printf("Computing distances via cross-correlation...\n");
    CudaSimulation& gpu = *gpuSimulation;
    constexpr unsigned threads = 128;
    if (gpu.rows) {
        correlateRows<<<static_cast<unsigned>((gpu.rows + threads - 1) / threads), threads>>>(
            gpu.n, gpu.timesteps, gpu.begin, gpu.rows, state.sourceIndex,
            gpu.radB, gpu.distances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + gpu.begin, gpu.distances,
                              gpu.rows * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.counts.data(),
                   state.displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
    gpuSimulation.reset();
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
    uint64_t localNonZeroKij = 0, nonZeroKij = 0;
    for (val_t kij : state.kij) {
        if (kij > EPSILON) ++localNonZeroKij;
    }
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    struct MpiFinalizer { ~MpiFinalizer() { MPI_Finalize(); } } mpiFinalizer;
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (rank != 0 && !freopen("/dev/null", "w", stdout)) {
        fprintf(stderr, "Could not redirect output for rank %d\n", rank);
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
    state.rank = rank;
    state.ranks = ranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Time each distributed phase from a common starting point.
    MPI_Barrier(MPI_COMM_WORLD);

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
