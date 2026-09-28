/**
 * Room Response Simulation Benchmark
 *
 * Hybrid MPI + OpenMP + CUDA implementation of a radiosity-based room impulse
 * response simulation. It models how sound/light waves propagate between
 * surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy
 * ------------------------
 *  - MPI     : the N x N pair space is distributed by rows of the Kij/Tau
 *              matrices; one rank per GPU. Only the small per-timestep
 *              radiosity vector (N floats) and the final distance vector are
 *              communicated.
 *  - CUDA    : all heavy kernels (form factors / octree ray casting, time
 *              delays, wave propagation, cross-correlation) run on the GPU of
 *              the owning rank. Every kernel keeps the original per-output
 *              summation order (one thread per output element, sequential
 *              reduction inside the thread) so results stay equivalent to the
 *              sequential code.
 *  - OpenMP  : host side setup that stays on the CPU (octree construction via
 *              tasks, mesh post-processing, validation sweeps).
 *
 * The Monte-Carlo visibility sampling uses one deterministic, statistically
 * independent counter-based random stream per triangle pair instead of a single
 * global sequential stream. This is required to make the sampling
 * order-independent (and hence parallel), and it is the standard practice for
 * parallel Monte-Carlo; the estimator that is computed is unchanged.
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#define HD __host__ __device__

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr val_t VAL_MAX = std::numeric_limits<val_t>::max();

// ============================================================================
// MPI / CUDA infrastructure
// ============================================================================

static int g_rank = 0;
static int g_nranks = 1;

// printf that only produces output on rank 0
static void rprintf(const char* fmt, ...) {
    if (g_rank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err_ = (call);                                                              \
        if (err_ != cudaSuccess) {                                                              \
            fprintf(stderr, "[rank %d] CUDA error at %s:%d: %s\n", g_rank, __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

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

// Sub-trees below this size are built by the calling thread instead of an
// OpenMP task (task overhead would dominate).
constexpr size_t OCTREE_TASK_CUTOFF = 512;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<uint32_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box (reduction over all vertices)
        val_t loX = triangles[0].a.x, loY = triangles[0].a.y, loZ = triangles[0].a.z;
        val_t hiX = loX, hiY = loY, hiZ = loZ;
        #pragma omp parallel for schedule(static) \
            reduction(min : loX, loY, loZ) reduction(max : hiX, hiY, hiZ)
        for (size_t i = 0; i < triangles.size(); ++i) {
            const Triangle& tri = triangles[i];
            for (const Vec3* v : {&tri.a, &tri.b, &tri.c}) {
                loX = std::min(loX, v->x);
                loY = std::min(loY, v->y);
                loZ = std::min(loZ, v->z);
                hiX = std::max(hiX, v->x);
                hiY = std::max(hiY, v->y);
                hiZ = std::max(hiZ, v->z);
            }
        }
        minBound = Vec3(loX, loY, loZ);
        maxBound = Vec3(hiX, hiY, hiZ);

        // Collect all indices
        std::vector<uint32_t> allIndices(triangles.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = static_cast<uint32_t>(i);

        // The tree is built in parallel: every sub-tree is independent, so the
        // resulting structure is identical to the sequential construction.
        #pragma omp parallel
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
    }

private:
    void buildNode(const std::vector<uint32_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
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
        Vec3 childCenters[8];
        for (int i = 0; i < 8; ++i) {
            Vec3 cc = center;
            cc.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
            cc.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
            cc.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
            childCenters[i] = cc;
        }

        std::vector<uint32_t> childIndices[8];

        if (indices.size() >= OCTREE_TASK_CUTOFF) {
            // Classify the triangles for the 8 children concurrently; each child
            // list keeps the original (ascending) index order.
            #pragma omp taskloop grainsize(1) default(shared)
            for (int i = 0; i < 8; ++i) {
                for (uint32_t idx : indices) {
                    if (triangleBoxOverlap(childCenters[i], childHalfSize, (*allTriangles)[idx])) {
                        childIndices[i].push_back(idx);
                    }
                }
            }
        } else {
            for (uint32_t idx : indices) {
                const Triangle& tri = (*allTriangles)[idx];
                for (int i = 0; i < 8; ++i) {
                    if (triangleBoxOverlap(childCenters[i], childHalfSize, tri)) {
                        childIndices[i].push_back(idx);
                    }
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

                Octree* child = children[i].get();
                // childIndices lives until the taskwait below, so it can be shared.
                #pragma omp task default(shared) firstprivate(child, i, childMin, childMax) \
                    if (childIndices[i].size() >= OCTREE_TASK_CUTOFF)
                child->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
    }
};

// ============================================================================
// Flattened Octree (GPU representation)
// ============================================================================

struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];    // -1 if absent
    int triStart;    // offset into the flat triangle index list
    int triCount;    // 0 for interior nodes
};

// Maximum traversal stack depth supported by the device traversal. A node pushes
// at most 8 children and pops itself, so depth D needs 7*D+1 slots.
constexpr int TRAVERSAL_STACK_SIZE = 64;

// Writes the octree into nodes[myIdx]. The children of a node are placed in
// consecutive slots, so the box tests of one traversal step read a contiguous
// run of nodes instead of scattered ones.
static void flattenOctree(const Octree* node, int myIdx, std::vector<FlatNode>& nodes,
                          std::vector<uint32_t>& triList, int depth, int& maxDepth) {
    if (depth > maxDepth) maxDepth = depth;

    FlatNode n{};
    n.center = node->center;
    n.halfExtent = node->halfExtent;
    for (int i = 0; i < 8; ++i) n.child[i] = -1;
    n.triStart = 0;
    n.triCount = 0;

    if (!node->triangleIndices.empty()) {
        n.triStart = static_cast<int>(triList.size());
        n.triCount = static_cast<int>(node->triangleIndices.size());
        triList.insert(triList.end(), node->triangleIndices.begin(), node->triangleIndices.end());
        nodes[myIdx] = n;
        return;
    }

    int present = 0;
    for (int i = 0; i < 8; ++i) present += node->children[i] ? 1 : 0;

    const int base = static_cast<int>(nodes.size());
    nodes.resize(static_cast<size_t>(base) + present);
    int k = 0;
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) n.child[i] = base + k++;
    }
    nodes[myIdx] = n;

    k = 0;
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            flattenOctree(node->children[i].get(), base + k++, nodes, triList, depth + 1, maxDepth);
        }
    }
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
            newFaces.reserve(faces.size() * 4);
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t f = 0; f < faces.size(); ++f) {
            const auto& face = faces[f];
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation (counter based, one stream per triangle pair)
// ============================================================================

// SplitMix64: a well tested counter based generator. Every triangle pair gets its
// own independent stream, so the Monte-Carlo sampling does not depend on the
// order in which the pairs are evaluated.
struct RandomGenerator {
    uint64_t s;

    HD explicit RandomGenerator(uint64_t seed) : s(seed) {}

    HD uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1) with 24 bit resolution (same granularity as the
    // float uniform_real_distribution of the reference implementation)
    HD val_t rand() { return static_cast<val_t>(next() >> 40) * 0x1.0p-24f; }
};

// Turns the (unique) pair index into a well separated starting state
HD inline uint64_t pairSeed(uint32_t i, uint32_t j) {
    uint64_t k = (static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(j);
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDull;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ull;
    k ^= k >> 33;
    return k ^ 42ull;
}

// Generate a random point inside a triangle using barycentric coordinates
HD inline Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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

HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return VAL_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return VAL_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return VAL_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, device side)
// ============================================================================

// Ray/box overlap test of the original octree node test, with the ray dependent
// terms (half direction, midpoint) hoisted out of the traversal loop.
__device__ inline bool rayIntersectsBox(const FlatNode& node, const Vec3& mid,
                                        const Vec3& d, const Vec3& ad) {
    Vec3 c = mid - node.center;
    const Vec3& h = node.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Traverses the flattened octree with an explicit stack; visits exactly the same
// set of triangles as the recursive host traversal of the reference code.
__device__ inline bool isRayBlocked(const Vec3& from, const Vec3& to,
                                    const FlatNode* __restrict__ nodes,
                                    const uint32_t* __restrict__ triList,
                                    const Triangle* __restrict__ triangles,
                                    uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    const Vec3 d = dir * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    int stack[TRAVERSAL_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root is entered unconditionally

    while (sp > 0) {
        const FlatNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                const uint32_t idx = triList[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 7; i >= 0; --i) {
                const int ci = node.child[i];
                if (ci >= 0 && rayIntersectsBox(nodes[ci], mid, d, ad)) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor / Time Delay kernels
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

HD inline int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Form factor weight and time delay of one triangle pair. They are always used
// together by the propagation kernel, so they are interleaved into a single
// 8 byte record; the row block is stored column major (N x localRows) so that
// the propagation kernel reads them fully coalesced.
struct WTau {
    val_t w;    // min(kij * area_j, 1)
    int tau;    // time delay
};

__global__ void tauKernel(int rowBegin, int rowCount, int liBase, int localRows, int n,
                          const Triangle* __restrict__ triangles, WTau* __restrict__ wtau) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<long long>(rowCount) * n) return;

    const int r = static_cast<int>(tid % rowCount);
    const int j = static_cast<int>(tid / rowCount);
    const int i = rowBegin + r;

    int value = 0;
    if (i != j) value = computeTau(triangles[i], triangles[j]);
    wtau[static_cast<size_t>(j) * localRows + (liBase + r)].tau = value;
}

// Form factors. One thread per triangle pair; the NUM_RAYS samples of a pair are
// accumulated sequentially by that thread, exactly as in the reference code.
// Stored pre-multiplied as min(kij * area_j, 1), which is the only form in which
// the propagation phase uses it.
__global__ void kijKernel(int rowBegin, int rowCount, int liBase, int localRows, int n,
                          const Triangle* __restrict__ triangles,
                          const FlatNode* __restrict__ nodes,
                          const uint32_t* __restrict__ triList,
                          const val_t* __restrict__ areas,
                          WTau* __restrict__ wtau,
                          unsigned long long* __restrict__ nonZeroCount) {
    __shared__ unsigned int blockNonZero;
    if (threadIdx.x == 0) blockNonZero = 0;
    __syncthreads();

    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < static_cast<long long>(rowCount) * n) {
        const int r = static_cast<int>(tid % rowCount);
        const int j = static_cast<int>(tid / rowCount);
        const int i = rowBegin + r;

        val_t kij = ZERO;

        if (i != j) {
            const Triangle triI = triangles[i];
            const Triangle triJ = triangles[j];

            // Cull triangles facing the same direction
            if (triI.normal().dot(triJ.normal()) <= 0.99f) {
                RandomGenerator rng(pairSeed(i, j));

                for (int ray = 0; ray < NUM_RAYS; ++ray) {
                    Vec3 pI = randomPointInTriangle(triI, rng);
                    Vec3 pJ = randomPointInTriangle(triJ, rng);

                    if (isRayBlocked(pI, pJ, nodes, triList, triangles, i, j)) continue;

                    Vec3 v = pJ - pI;
                    val_t distSqr = v.squaredNorm();
                    if (distSqr < EPSILON) continue;

                    val_t cosPhiI = cosPhi(v, triI.normal());
                    val_t cosPhiJ = cosPhi(-v, triJ.normal());

                    if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                    kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
                kij *= INV_NUM_RAYS;
            }
        }

        wtau[static_cast<size_t>(j) * localRows + (liBase + r)].w = fminf(kij * areas[j], ONE);

        // Counted for the validation report
        if (kij > EPSILON) atomicAdd(&blockNonZero, 1u);
    }

    __syncthreads();
    if (threadIdx.x == 0 && blockNonZero != 0) {
        atomicAdd(nonZeroCount, static_cast<unsigned long long>(blockNonZero));
    }
}

// ============================================================================
// Wave Propagation kernel
// ============================================================================

// One thread per owned triangle; the reduction over j runs sequentially inside
// the thread, preserving the summation order of the reference implementation.
// The (w, tau) pairs are streamed coalesced, and the delayed radiosity is read
// from the transposed history so that a warp (32 consecutive i, same j) touches
// only a short contiguous slice of it.
constexpr int PROP_UNROLL = 16;

__global__ void propagateKernel(int t, int rowBegin, int localRows, int n,
                                const WTau* __restrict__ wtau,
                                const val_t* __restrict__ rho,
                                const val_t* __restrict__ radE,
                                const val_t* __restrict__ radBT, int timesteps,
                                val_t* __restrict__ radB) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= localRows) return;
    const int i = rowBegin + li;

    val_t sumB = ZERO;
    int j = 0;
    for (; j + PROP_UNROLL <= n; j += PROP_UNROLL) {
        // Independent loads first (memory level parallelism), then the
        // accumulation strictly in increasing j order.
        WTau e[PROP_UNROLL];
        val_t r[PROP_UNROLL];
        #pragma unroll
        for (int u = 0; u < PROP_UNROLL; ++u) {
            e[u] = wtau[static_cast<size_t>(j + u) * localRows + li];
        }
        #pragma unroll
        for (int u = 0; u < PROP_UNROLL; ++u) {
            const int src = t - e[u].tau;
            r[u] = radBT[static_cast<size_t>(j + u) * timesteps + (src < 0 ? 0 : src)];
        }
        #pragma unroll
        for (int u = 0; u < PROP_UNROLL; ++u) {
            const int jj = j + u;
            if (i == jj) continue;
            if (t < e[u].tau) continue;  // wave has not yet propagated from j to i
            if (e[u].w <= ZERO) continue;
            if (r[u] <= ZERO) continue;
            sumB += e[u].w * r[u];
        }
    }
    for (; j < n; ++j) {
        if (i == j) continue;
        const WTau e = wtau[static_cast<size_t>(j) * localRows + li];
        if (t < e.tau) continue;
        if (e.w <= ZERO) continue;
        const val_t radJ = radBT[static_cast<size_t>(j) * timesteps + (t - e.tau)];
        if (radJ <= ZERO) continue;
        sumB += e.w * radJ;
    }

    const size_t out = static_cast<size_t>(t) * n + i;
    const val_t value = rho[i] * sumB + radE[out];
    radB[out] = value;
}

// Mirror one timestep of the radiosity into the transposed history
__global__ void transposeRowKernel(int t, int n, int timesteps,
                                   const val_t* __restrict__ radB, val_t* __restrict__ radBT) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    radBT[static_cast<size_t>(i) * timesteps + t] = radB[static_cast<size_t>(t) * n + i];
}

// ============================================================================
// Cross-Correlation kernels
// ============================================================================

// corr[t][li] = sum_{tt >= t} radB[tt][i] * radB[tt - t][source]
__global__ void correlateKernel(int rowBegin, int localRows, int n, int timesteps,
                                int sourceIndex, const val_t* __restrict__ radB,
                                val_t* __restrict__ corr) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= static_cast<long long>(localRows) * timesteps) return;

    const int li = static_cast<int>(tid % localRows);
    const int t = static_cast<int>(tid / localRows);
    const int i = rowBegin + li;

    val_t sum = ZERO;
    for (int tt = t; tt < timesteps; ++tt) {
        const val_t pB = radB[static_cast<size_t>(tt) * n + i];
        const val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
        sum += pS * pB;
    }
    corr[static_cast<size_t>(t) * localRows + li] = sum;
}

// Pick the first (smallest) t with the maximal correlation, as the sequential
// "sum > maxCorr" scan does.
__global__ void argmaxKernel(int localRows, int timesteps, const val_t* __restrict__ corr,
                             val_t* __restrict__ distances) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= localRows) return;

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < timesteps; ++t) {
        const val_t sum = corr[static_cast<size_t>(t) * localRows + li];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex = 0;
    uint64_t nonZeroKij = 0;        // number of form factors > EPSILON (all ranks)

    // MPI row decomposition of the N x N matrices
    int rowBegin = 0;
    int localRows = 0;
    std::vector<int> rowCounts;
    std::vector<int> rowOffsets;

    // Device resources
    Triangle* d_triangles = nullptr;
    FlatNode* d_nodes = nullptr;
    uint32_t* d_triList = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    WTau* d_wtau = nullptr;         // (weight, tau) pairs, column major (N x localRows)
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;        // radiosity history, row major (T x N)
    val_t* d_radBT = nullptr;       // radiosity history, column major (N x T)
    val_t* d_corr = nullptr;
    val_t* d_distances = nullptr;
    unsigned long long* d_counter = nullptr;

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

    rprintf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration (OpenMP task parallel)
    rprintf("Building octree...\n");
    state.octree.build(state.triangles);

    const int n = static_cast<int>(state.numTriangles);

    // Row decomposition of the N x N matrices over the MPI ranks
    state.rowCounts.resize(g_nranks);
    state.rowOffsets.resize(g_nranks);
    const int base = n / g_nranks;
    const int rem = n % g_nranks;
    int off = 0;
    for (int r = 0; r < g_nranks; ++r) {
        state.rowCounts[r] = base + (r < rem ? 1 : 0);
        state.rowOffsets[r] = off;
        off += state.rowCounts[r];
    }
    state.rowBegin = state.rowOffsets[g_rank];
    state.localRows = state.rowCounts[g_rank];

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(state.numTriangles, reflectivity);

    // Initialize radiosity matrices
    state.radE.assign(timesteps * state.numTriangles, ZERO);
    state.radB.assign(timesteps * state.numTriangles, ZERO);
    state.distances.assign(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---- Device setup -------------------------------------------------------
    std::vector<FlatNode> nodes;
    std::vector<uint32_t> triList;
    nodes.reserve(state.numTriangles);
    triList.reserve(state.numTriangles * 4);
    int maxDepth = 0;
    nodes.resize(1);
    flattenOctree(&state.octree, 0, nodes, triList, 1, maxDepth);
    if (7 * maxDepth + 1 > TRAVERSAL_STACK_SIZE) {
        fprintf(stderr, "octree too deep (%d levels) for the device traversal stack\n", maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const size_t lr = static_cast<size_t>(state.localRows);
    CUDA_CHECK(cudaMalloc(&state.d_triangles, state.numTriangles * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMalloc(&state.d_triList, triList.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, state.numTriangles * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, state.radE.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, state.radB.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radBT, state.radB.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_counter, sizeof(unsigned long long)));
    if (lr > 0) {
        CUDA_CHECK(cudaMalloc(&state.d_wtau, lr * state.numTriangles * sizeof(WTau)));
        CUDA_CHECK(cudaMalloc(&state.d_corr, lr * timesteps * sizeof(val_t)));
        CUDA_CHECK(cudaMalloc(&state.d_distances, lr * sizeof(val_t)));
    }

    CUDA_CHECK(cudaMemcpy(state.d_triangles, state.triangles.data(),
                          state.numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(FlatNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triList, triList.data(), triList.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), state.numTriangles * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), state.radE.size() * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, state.radB.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_radBT, 0, state.radB.size() * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_counter, 0, sizeof(unsigned long long)));
}

void releaseSimulation(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_nodes);
    cudaFree(state.d_triList);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_wtau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_radBT);
    cudaFree(state.d_corr);
    cudaFree(state.d_distances);
    cudaFree(state.d_counter);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

constexpr int BLOCK = 128;
constexpr int KIJ_BLOCK = 64;   // small blocks schedule the divergent ray casts better

void computeTimeDelays(SimulationState& state) {
    rprintf("Computing time delays (Tau)...\n");
    if (state.localRows == 0) return;

    const int n = static_cast<int>(state.numTriangles);
    const long long total = static_cast<long long>(state.localRows) * n;
    tauKernel<<<static_cast<unsigned>((total + BLOCK - 1) / BLOCK), BLOCK>>>(
        state.rowBegin, state.localRows, 0, state.localRows, n, state.d_triangles, state.d_wtau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    rprintf("Computing form factors (Kij)...\n");

    const int n = static_cast<int>(state.numTriangles);

    // Row chunking: keeps every launch large enough to saturate the GPU while
    // still allowing progress reporting.
    int rowsPerLaunch = std::max(1, static_cast<int>(4000000 / std::max(1, n)));
    rowsPerLaunch = std::min(rowsPerLaunch, std::max(1, state.localRows));

    size_t nextProgress = 100;
    for (int done = 0; done < state.localRows; done += rowsPerLaunch) {
        const int rows = std::min(rowsPerLaunch, state.localRows - done);
        const long long total = static_cast<long long>(rows) * n;
        kijKernel<<<static_cast<unsigned>((total + KIJ_BLOCK - 1) / KIJ_BLOCK), KIJ_BLOCK>>>(
            state.rowBegin + done, rows, done, state.localRows, n, state.d_triangles,
            state.d_nodes, state.d_triList, state.d_areas, state.d_wtau, state.d_counter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Progress of rank 0 extrapolated to the global triangle count
        const size_t reached = static_cast<size_t>(
            (static_cast<double>(done + rows) / state.localRows) * static_cast<double>(n));
        while (nextProgress <= reached && nextProgress < state.numTriangles) {
            rprintf("  Progress: %zu/%zu triangles\n", nextProgress, state.numTriangles);
            nextProgress += 100;
        }
    }
    while (nextProgress < state.numTriangles) {
        rprintf("  Progress: %zu/%zu triangles\n", nextProgress, state.numTriangles);
        nextProgress += 100;
    }
    rprintf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);

    // Number of non-zero form factors (needed by the validation report)
    unsigned long long localCount = 0;
    CUDA_CHECK(cudaMemcpy(&localCount, state.d_counter, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    unsigned long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    state.nonZeroKij = globalCount;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    rprintf("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int timesteps = static_cast<int>(state.numTimesteps);
    // Small blocks: only localRows threads exist, they should spread over all SMs
    constexpr int PROP_BLOCK = 64;
    const unsigned grid = static_cast<unsigned>((std::max(state.localRows, 1) + PROP_BLOCK - 1) / PROP_BLOCK);
    const unsigned tgrid = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.localRows > 0) {
            propagateKernel<<<grid, PROP_BLOCK>>>(static_cast<int>(t), state.rowBegin,
                                                  state.localRows, n, state.d_wtau, state.d_rho,
                                                  state.d_radE, state.d_radBT, timesteps,
                                                  state.d_radB);
            CUDA_CHECK(cudaGetLastError());
        }

        // Exchange the freshly computed radiosity row: every rank needs the full
        // history of all triangles for the delayed contributions.
        if (g_nranks > 1) {
            val_t* hostRow = &state.radB[state.idxTN(t, 0)];
            if (state.localRows > 0) {
                CUDA_CHECK(cudaMemcpy(hostRow + state.rowBegin,
                                      state.d_radB + state.idxTN(t, 0) + state.rowBegin,
                                      static_cast<size_t>(state.localRows) * sizeof(val_t),
                                      cudaMemcpyDeviceToHost));
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, hostRow, state.rowCounts.data(),
                           state.rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(state.d_radB + state.idxTN(t, 0), hostRow,
                                  state.numTriangles * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        // Mirror the completed row into the transposed history used by the
        // delayed lookups of the following timesteps.
        transposeRowKernel<<<tgrid, BLOCK>>>(static_cast<int>(t), n, timesteps, state.d_radB,
                                             state.d_radBT);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            rprintf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (g_nranks == 1) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, state.radB.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    rprintf("Computing distances via cross-correlation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int timesteps = static_cast<int>(state.numTimesteps);

    if (state.localRows > 0) {
        const long long total = static_cast<long long>(state.localRows) * timesteps;
        correlateKernel<<<static_cast<unsigned>((total + BLOCK - 1) / BLOCK), BLOCK>>>(
            state.rowBegin, state.localRows, n, timesteps, static_cast<int>(state.sourceIndex),
            state.d_radB, state.d_corr);
        CUDA_CHECK(cudaGetLastError());

        argmaxKernel<<<static_cast<unsigned>((state.localRows + BLOCK - 1) / BLOCK), BLOCK>>>(
            state.localRows, timesteps, state.d_corr, state.d_distances);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowBegin, state.d_distances,
                              static_cast<size_t>(state.localRows) * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (g_nranks > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.distances.data(), state.rowCounts.data(),
                       state.rowOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    rprintf("\nValidation:\n");

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
    rprintf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    rprintf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        rprintf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    #pragma omp parallel for reduction(+ : receivedEnergy) schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    rprintf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        rprintf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    const uint64_t nonZeroKij = state.nonZeroKij;
    rprintf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
            static_cast<unsigned long long>(nonZeroKij), state.numTriangles * state.numTriangles,
            100.0f * static_cast<val_t>(nonZeroKij) /
                static_cast<val_t>(state.numTriangles * state.numTriangles));

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
    #pragma omp parallel for reduction(^ : hash) schedule(static)
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
    rprintf("Usage: %s [options]\n", progName);
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

// Bind this rank to one of the GPUs of its node
static void selectDevice() {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "[rank %d] no CUDA device available\n", g_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation now

    // Share the node's cores between the ranks that run on it
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int hw = omp_get_num_procs();
        omp_set_num_threads(std::max(1, hw / std::max(1, localSize)));
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

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

    selectDevice();

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    rprintf("Room Response Simulation Benchmark\n");
    rprintf("===================================\n");
    rprintf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    rprintf("Timesteps: %d\n", timesteps);
    rprintf("Source triangle: %d\n", sourceIdx);
    rprintf("Reflectivity: %.2f\n", reflectivity);
    rprintf("Validation: %s\n", validate ? "enabled" : "disabled");
    rprintf("MPI ranks: %d, OpenMP threads/rank: %d\n", g_nranks, omp_get_max_threads());
    rprintf("\n");

    // Initialize
    SimulationState state;
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
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    rprintf("Simulation time: %ld ms\n", simDuration);
    rprintf("\n");

    // Distance computation
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
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (!validateResults(state)) status = 1;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    releaseSimulation(state);
    MPI_Finalize();
    return status;
}
