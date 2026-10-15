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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define HD __host__ __device__

// Only rank 0 produces output
static int g_rank = 0;
#define rprintf(...) do { if (g_rank == 0) std::printf(__VA_ARGS__); } while (0)

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

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
    explicit HD constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
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

        // Overlap tests are independent per triangle: evaluate in parallel,
        // then gather in the original order
        const long nIdx = static_cast<long>(indices.size());
        std::vector<uint8_t> masks(indices.size());
        #pragma omp parallel for schedule(static) if (nIdx > 512)
        for (long k = 0; k < nIdx; ++k) {
            const Triangle& tri = (*allTriangles)[indices[k]];
            uint8_t m = 0;
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) m |= (1u << i);
            }
            masks[k] = m;
        }
        for (long k = 0; k < nIdx; ++k) {
            for (int i = 0; i < 8; ++i) {
                if (masks[k] & (1u << i)) childIndices[i].push_back(indices[k]);
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
// Device-side geometry helpers
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                              const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    const val_t NO_HIT = 3.402823466e+38f;
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Flattened octree node for GPU traversal
struct GpuNode {
    Vec3 center, half;
    int child[8];
    int triBegin, triCount;   // triCount > 0 -> leaf
};

HD bool rayIntersectsNode(const GpuNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = n.half;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

static int flattenOctree(const Octree& node, std::vector<GpuNode>& nodes, std::vector<int>& tris) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    GpuNode g;
    g.center = node.center;
    g.half = node.halfExtent;
    g.triBegin = static_cast<int>(tris.size());
    g.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t t : node.triangleIndices) tris.push_back(static_cast<int>(t));
    for (int i = 0; i < 8; ++i) {
        g.child[i] = node.children[i] ? flattenOctree(*node.children[i], nodes, tris) : -1;
    }
    nodes[id] = g;
    return id;
}

// ============================================================================
// Random numbers (counter based: one independent stream per (i, j) pair)
// ============================================================================

struct DevRng {
    uint64_t s;
    __device__ explicit DevRng(uint64_t seed) : s(seed) {}
    __device__ uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    __device__ val_t rand() { return static_cast<val_t>(next() >> 40) * (1.0f / 16777216.0f); }
};

__device__ Vec3 randomPointInTriangle(const Triangle& t, DevRng& rng) {
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
// Visibility Testing (octree-accelerated, iterative traversal on the GPU)
// ============================================================================

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const GpuNode* nodes, const int* nodeTris, const Triangle* tris,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const GpuNode& n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (int k = 0; k < n.triCount; ++k) {
                int idx = nodeTris[n.triBegin + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                int c = n.child[i];
                if (c >= 0 && rayIntersectsNode(nodes[c], from, to)) stack[sp++] = c;
            }
        }
    }
    return false;
}

// ============================================================================
// Kernels
// ============================================================================

constexpr int BLOCK = 128;

// Time delays Tau for the rows owned by this rank
__global__ void tauKernel(size_t N, size_t i0, size_t rows, size_t base,
                          const Triangle* tris, uint8_t* tau) {
    size_t tid = base + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= rows * N) return;
    size_t li = tid / N, j = tid % N, i = i0 + li;
    int v = 0;
    if (i != j) {
        val_t dist = (tris[i].center() - tris[j].center()).norm();
        v = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tau[tid] = static_cast<uint8_t>(min(v, 255));
}

// Form factors; stores W = min(Kij * area_j, 1) (0 where Kij <= 0)
__global__ void formFactorKernel(size_t N, size_t i0, size_t rows, size_t base,
                                 const Triangle* tris, const val_t* areas,
                                 const GpuNode* nodes, const int* nodeTris,
                                 val_t* W, unsigned long long* nzCount) {
    size_t tid = base + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool nonZero = false;
    if (tid < rows * N) {
        size_t li = tid / N, j = tid % N, i = i0 + li;
        val_t w = ZERO;
        if (i != j) {
            const Triangle& triI = tris[i];
            const Triangle& triJ = tris[j];
            if (!(triI.normal().dot(triJ.normal()) > 0.99f)) {
                DevRng rng(0x2545F4914F6CDD1DULL ^ (static_cast<uint64_t>(i) * N + j) * 0x9e3779b97f4a7c15ULL);
                val_t kij = ZERO;
                for (int r = 0; r < NUM_RAYS; ++r) {
                    Vec3 pI = randomPointInTriangle(triI, rng);
                    Vec3 pJ = randomPointInTriangle(triJ, rng);

                    if (isRayBlocked(pI, pJ, nodes, nodeTris, tris, (int)i, (int)j)) continue;

                    Vec3 v = pJ - pI;
                    val_t distSqr = v.squaredNorm();
                    if (distSqr < EPSILON) continue;

                    val_t cosPhiI = cosPhi(v, triI.normal());
                    val_t cosPhiJ = cosPhi(-v, triJ.normal());
                    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                    kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
                kij *= INV_NUM_RAYS;
                nonZero = kij > EPSILON;
                if (kij > ZERO) w = fminf(kij * areas[j], ONE);
            }
        }
        W[tid] = w;
    }
    // Block-level count of non-zero form factors
    int cnt = __syncthreads_count(nonZero);
    if (threadIdx.x == 0 && cnt) atomicAdd(nzCount, static_cast<unsigned long long>(cnt));
}

// One timestep of wave propagation: one warp per owned row
__global__ void simStepKernel(size_t N, size_t i0, size_t rows, size_t t, size_t emitUntil,
                              size_t sourceIdx, val_t rho,
                              const val_t* W, const uint8_t* tau, val_t* radB) {
    size_t warp = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    size_t i = i0 + warp;
    const val_t* wRow = W + warp * N;
    const uint8_t* tRow = tau + warp * N;

    val_t sum = ZERO;
    #pragma unroll 4
    for (size_t j = lane; j < N; j += 32) {
        size_t tauij = tRow[j];
        if (t < tauij || j == i) continue;
        val_t w = wRow[j];
        if (w <= ZERO) continue;
        val_t radJ = radB[(t - tauij) * N + j];
        if (radJ <= ZERO) continue;
        sum += w * radJ;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);

    if (lane == 0) {
        val_t emit = (i == sourceIdx && t < emitUntil) ? ONE : ZERO;
        radB[t * N + i] = rho * sum + emit;
    }
}

__global__ void transposeKernel(size_t N, size_t T, const val_t* radB, val_t* radBT) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= N * T) return;
    size_t t = tid / N, n = tid % N;
    radBT[n * T + t] = radB[tid];
}

struct Best { val_t sum; int t; };
__device__ inline bool better(const Best& a, const Best& b) {
    return a.sum > b.sum || (a.sum == b.sum && a.t < b.t);
}

// Cross-correlation: one block per owned row
__global__ void distanceKernel(size_t N, size_t i0, size_t rows, size_t T, size_t sourceIdx,
                               const val_t* radBT, val_t* dist) {
    size_t li = blockIdx.x;
    if (li >= rows) return;
    const val_t* pB = radBT + (i0 + li) * T;
    const val_t* pS = radBT + sourceIdx * T;

    Best best{ZERO, 0x7fffffff};
    for (size_t t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < T; ++tt) sum += pS[tt - t] * pB[tt];
        if (sum > best.sum) { best.sum = sum; best.t = static_cast<int>(t); }
    }
    __shared__ Best sh[BLOCK];
    sh[threadIdx.x] = best;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && better(sh[threadIdx.x + s], sh[threadIdx.x])) sh[threadIdx.x] = sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int bt = sh[0].sum > ZERO ? sh[0].t : 0;
        dist[li] = WAVE_SPEED * static_cast<val_t>(bt);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    val_t reflectivity;             // Reflectivity (0.0 to 1.0), same for all triangles
    std::vector<val_t> distances;   // Computed distances from source (all triangles)
    std::vector<val_t> radB;        // Reflected radiosity (T x N), only gathered for validation

    Octree octree;                  // Host octree (flattened for the GPU)
    size_t sourceIndex;

    // MPI decomposition: this rank owns rows [rowBegin, rowBegin + rowCount)
    int rank = 0, size = 1;
    size_t rowBegin = 0, rowCount = 0;
    std::vector<int> counts, displs;

    // Device data
    Triangle* dTris = nullptr;
    val_t* dAreas = nullptr;
    GpuNode* dNodes = nullptr;
    int* dNodeTris = nullptr;
    val_t* dW = nullptr;            // rowCount x N weights min(Kij*A_j, 1)
    uint8_t* dTau = nullptr;        // rowCount x N time delays
    val_t* dRadB = nullptr;         // T x N (full, replicated)
    val_t* dRadBT = nullptr;        // N x T (transposed copy)
    val_t* dDist = nullptr;
    unsigned long long* dNz = nullptr;
    val_t* hRow = nullptr;          // pinned staging row for MPI exchange

    unsigned long long nonZeroKij = 0;   // global count (valid on rank 0)
};

template <typename T>
static T* devAlloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    return p;
}

static void releaseDevice(SimulationState& s) {
    cudaFree(s.dTris); cudaFree(s.dAreas); cudaFree(s.dNodes); cudaFree(s.dNodeTris);
    cudaFree(s.dW); cudaFree(s.dTau); cudaFree(s.dRadB); cudaFree(s.dRadBT);
    cudaFree(s.dDist); cudaFree(s.dNz); cudaFreeHost(s.hRow);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh (replicated on every rank; cheap compared to the simulation)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;

    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    const size_t N = state.numTriangles;

    // Initialize areas
    state.areas.resize(N);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        state.areas[i] = state.triangles[i].area();
    }
    state.distances.assign(N, ZERO);

    // Row decomposition over MPI ranks
    state.counts.resize(state.size);
    state.displs.resize(state.size);
    size_t base = N / state.size, rem = N % state.size, off = 0;
    for (int r = 0; r < state.size; ++r) {
        size_t c = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        state.counts[r] = static_cast<int>(c);
        state.displs[r] = static_cast<int>(off);
        off += c;
    }
    state.rowBegin = state.displs[state.rank];
    state.rowCount = state.counts[state.rank];

    // Upload to the device
    std::vector<GpuNode> nodes;
    std::vector<int> nodeTris;
    flattenOctree(state.octree, nodes, nodeTris);

    state.dTris = devAlloc<Triangle>(N);
    state.dAreas = devAlloc<val_t>(N);
    state.dNodes = devAlloc<GpuNode>(nodes.size());
    state.dNodeTris = devAlloc<int>(nodeTris.size());
    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, nodes.data(), nodes.size() * sizeof(GpuNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodeTris, nodeTris.data(), nodeTris.size() * sizeof(int), cudaMemcpyHostToDevice));

    state.dW = devAlloc<val_t>(state.rowCount * N);
    state.dTau = devAlloc<uint8_t>(state.rowCount * N);
    state.dRadB = devAlloc<val_t>(timesteps * N);
    state.dRadBT = devAlloc<val_t>(timesteps * N);
    state.dDist = devAlloc<val_t>(state.rowCount);
    state.dNz = devAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(timesteps * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dNz, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocHost(&state.hRow, N * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

constexpr size_t MAX_THREADS_PER_LAUNCH = size_t(1) << 28;

void computeFormFactors(SimulationState& state) {
    rprintf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const size_t total = state.rowCount * N;

    for (size_t base = 0; base < total; base += MAX_THREADS_PER_LAUNCH) {
        size_t n = std::min(MAX_THREADS_PER_LAUNCH, total - base);
        formFactorKernel<<<static_cast<unsigned>((n + BLOCK - 1) / BLOCK), BLOCK>>>(
            N, state.rowBegin, state.rowCount, base, state.dTris, state.dAreas,
            state.dNodes, state.dNodeTris, state.dW, state.dNz);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long local = 0;
    CUDA_CHECK(cudaMemcpy(&local, state.dNz, sizeof(local), cudaMemcpyDeviceToHost));
    MPI_Reduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    rprintf("  Progress: %zu/%zu triangles\n", N, N);
}

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");
    const size_t N = state.numTriangles;
    const size_t total = state.rowCount * N;

    for (size_t base = 0; base < total; base += MAX_THREADS_PER_LAUNCH) {
        size_t n = std::min(MAX_THREADS_PER_LAUNCH, total - base);
        tauKernel<<<static_cast<unsigned>((n + BLOCK - 1) / BLOCK), BLOCK>>>(
            N, state.rowBegin, state.rowCount, base, state.dTris, state.dTau);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t emitUntil = T / 2;

    for (size_t t = 0; t < T; ++t) {
        if (state.rowCount > 0) {
            unsigned blocks = static_cast<unsigned>((state.rowCount * 32 + BLOCK - 1) / BLOCK);
            simStepKernel<<<blocks, BLOCK>>>(N, state.rowBegin, state.rowCount, t, emitUntil,
                                             state.sourceIndex, state.reflectivity,
                                             state.dW, state.dTau, state.dRadB);
        }
        CUDA_CHECK(cudaGetLastError());

        if (state.size > 1) {
            // Exchange this timestep's radiosity row between ranks
            val_t* dRow = state.dRadB + t * N;
            CUDA_CHECK(cudaMemcpy(state.hRow + state.rowBegin, dRow + state.rowBegin,
                                  state.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.hRow,
                           state.counts.data(), state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(dRow, state.hRow, N * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            rprintf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    rprintf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    if (N * T > 0) {
        transposeKernel<<<static_cast<unsigned>((N * T + BLOCK - 1) / BLOCK), BLOCK>>>(
            N, T, state.dRadB, state.dRadBT);
    }
    if (state.rowCount > 0) {
        distanceKernel<<<static_cast<unsigned>(state.rowCount), BLOCK>>>(
            N, state.rowBegin, state.rowCount, T, state.sourceIndex, state.dRadBT, state.dDist);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowBegin, state.dDist,
                          state.rowCount * sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(),
                   state.counts.data(), state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (rank 0 only)
// ============================================================================

bool validateResults(SimulationState& state) {
    rprintf("\nValidation:\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Bring radiosity history back to the host
    state.radB.resize(T * N);
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, T * N * sizeof(val_t), cudaMemcpyDeviceToHost));

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < N; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            rprintf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            rprintf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    rprintf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    rprintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(N));
    rprintf("  Non-zero distances: %d/%zu\n", nonZeroCount, N);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rprintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
    for (long i = 0; i < static_cast<long>(N); ++i) {
        for (size_t t = 0; t < T; ++t) {
            if (state.radB[t * N + i] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    rprintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, N);

    if (receivedEnergy == 0) {
        rprintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
    rprintf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
            nonZeroKij, N * N,
            100.0f * nonZeroKij / static_cast<val_t>(N * N));

    if (nonZeroKij == 0) {
        rprintf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    rprintf("  Validation: PASSED\n");
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
    std::printf("Usage: %s [options]\n", progName);
    rprintf("Options:\n");
    rprintf("  -n <num>     Target number of triangles (default: 320)\n");
    rprintf("               Actual count will be rounded to nearest icosphere level:\n");
    rprintf("               20, 80, 320, 1280, 5120, 20480\n");
    rprintf("  -t <num>     Number of timesteps (default: 50)\n");
    rprintf("  -s <num>     Source triangle index (default: 0)\n");
    rprintf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    rprintf("  -v           Enable validation\n");
    rprintf("  -o           Print results for external validation\n");
    rprintf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    g_rank = rank;

    // One GPU per rank (round-robin over the GPUs visible on the node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
        CUDA_CHECK(cudaFree(0));
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
            MPI_Finalize();
            return 0;
        } else {
            rprintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rprintf("Room Response Simulation Benchmark\n");
    rprintf("===================================\n");
    rprintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rprintf("Timesteps: %d\n", timesteps);
    rprintf("Source triangle: %d\n", sourceIdx);
    rprintf("Reflectivity: %.2f\n", reflectivity);
    rprintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rprintf("\n");

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.size = nranks;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    rprintf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    rprintf("Precomputation time: %ld ms\n", preDuration);
    rprintf("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    rprintf("Distance computation time: %ld ms\n", distDuration);
    rprintf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    rprintf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    rprintf("\nPerformance:\n");
    rprintf("  Triangles: %zu\n", n);
    rprintf("  Timesteps: %zu\n", t);
    rprintf("  Form factor computations: %.2e\n", kijOps);
    rprintf("  Simulation operations: %.2e\n", simOps);
    rprintf("  Distance computations: %.2e\n", distOps);
    rprintf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    rprintf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    rprintf("  Result hash: %016lX\n", hash);
    rprintf("\n");
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate && rank == 0) {
        if (!validateResults(state)) status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    releaseDevice(state);
    MPI_Finalize();
    return status;
}
