/**
 * Room Response Simulation Benchmark -- hybrid MPI + OpenMP + CUDA implementation
 *
 * This models how sound/light waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy
 * ------------------------
 *  * MPI    : the triangle rows of the Kij matrix (and therefore the radiosity update
 *             and the cross-correlation) are distributed over the ranks in a
 *             block-cyclic fashion.  Only the (tiny) per-timestep radiosity vector and
 *             the final distance vector are exchanged (MPI_Allreduce).  Each rank only
 *             stores its own slice of the O(N^2) form factor matrix, so the problem
 *             size scales with the number of ranks.
 *  * CUDA   : the two dominant kernels (form factor / visibility ray casting and the
 *             radiosity propagation) run on the GPU.  Every rank drives one GPU.
 *  * OpenMP : the CPU cores of a rank cooperate with the GPU on the form factor
 *             computation (dynamic work stealing from a shared tile queue), generate
 *             the random numbers that feed the GPU kernels, and parallelize the
 *             remaining host side work (mesh setup, octree flattening, validation).
 *
 * Determinism / semantics
 * -----------------------
 * The original code draws all random numbers from a single std::mt19937 stream that is
 * consumed in (i,j) order.  The exact number of draws consumed by a pair only depends on
 * the (purely geometric) back-face cull test, so the stream offset of every pair can be
 * computed in advance.  A bit-identical MT19937 implementation with fast skip-ahead is
 * used to seed every work tile, hence every pair sees exactly the random numbers the
 * sequential code would have given it.  Floating point contraction is disabled on both
 * the host and the device so that a pair yields bit-identical results no matter whether
 * it was computed by a CPU thread or by the GPU.
 */

#include <algorithm>
#include <array>
#include <atomic>
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
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

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
constexpr val_t VAL_MAX = 3.40282347e+38f;   // std::numeric_limits<float>::max()

// Random numbers consumed by one (non culled) triangle pair: 2 points * 2 draws * rays
constexpr int RND_PER_PAIR = 4 * NUM_RAYS;

#define HD __host__ __device__

#define CUDA_CHECK(expr)                                                                      \
    do {                                                                                      \
        cudaError_t _err = (expr);                                                            \
        if(_err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(_err), __FILE__, \
                    __LINE__);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while(0)

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
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
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

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Bit-identical MT19937 with fast skip-ahead
// ============================================================================
//
// Produces exactly the same sequence of values as
// std::uniform_real_distribution<float>(0,1) driven by std::mt19937 (which boils down to
// std::generate_canonical<float,24>, i.e. one engine word scaled by 2^-32), but supports
// O(1) state copies and cheap skipping, which is what makes the parallel decomposition of
// the single global random stream possible.
class MT19937 {
  public:
    static constexpr int NN = 624;
    static constexpr int MM = 397;

    void seed(uint32_t s) {
        mt_[0] = s;
        for(int i = 1; i < NN; ++i) {
            mt_[i] = static_cast<uint32_t>(1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i));
        }
        idx_ = NN;
    }

    // Skip n generated values.
    void advance(uint64_t n) {
        uint64_t avail = static_cast<uint64_t>(NN - idx_);
        if(n < avail) {
            idx_ += static_cast<int>(n);
            return;
        }
        n -= avail;
        idx_ = NN;
        uint64_t blocks = n / NN;
        for(uint64_t b = 0; b < blocks; ++b) twist();
        idx_ = NN;  // the blocks generated above are entirely skipped
        n -= blocks * NN;
        if(n > 0) {
            twist();
            idx_ = static_cast<int>(n);
        }
    }

    // Generate n floats in [0,1)
    void genFloats(float* out, size_t n) {
        size_t k = 0;
        while(k < n) {
            if(idx_ >= NN) twist();
            size_t take = std::min(static_cast<size_t>(NN - idx_), n - k);
            const uint32_t* src = mt_ + idx_;
            for(size_t q = 0; q < take; ++q) {
                float v = static_cast<float>(temper(src[q])) * (1.0f / 4294967296.0f);
                if(v >= 1.0f) v = 0x1.fffffep-1f;  // nextafter(1,0), as in generate_canonical
                out[k + q] = v;
            }
            idx_ += static_cast<int>(take);
            k += take;
        }
    }

    float nextFloat() {
        if(idx_ >= NN) twist();
        float v = static_cast<float>(temper(mt_[idx_++])) * (1.0f / 4294967296.0f);
        return v >= 1.0f ? 0x1.fffffep-1f : v;
    }

  private:
    static uint32_t temper(uint32_t y) {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return y;
    }

    // The three vectorizable phases of the Mersenne twister state update.
    void twist() {
        constexpr uint32_t UPPER = 0x80000000u;
        constexpr uint32_t LOWER = 0x7fffffffu;
        constexpr uint32_t MATRIX_A = 0x9908b0dfu;
        for(int i = 0; i < NN - MM; ++i) {
            uint32_t y = (mt_[i] & UPPER) | (mt_[i + 1] & LOWER);
            mt_[i] = mt_[i + MM] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
        }
        for(int i = NN - MM; i < NN - 1; ++i) {
            uint32_t y = (mt_[i] & UPPER) | (mt_[i + 1] & LOWER);
            mt_[i] = mt_[i + (MM - NN)] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
        }
        uint32_t y = (mt_[NN - 1] & UPPER) | (mt_[0] & LOWER);
        mt_[NN - 1] = mt_[MM - 1] ^ (y >> 1) ^ ((0u - (y & 1u)) & MATRIX_A);
        idx_ = 0;
    }

    uint32_t mt_[NN];
    int idx_ = NN;
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

        // The overlap tests of the 8 children are independent; for the upper levels of
        // the tree (which hold most of the triangles) this is worth threading.
        if(indices.size() >= 4096) {
            #pragma omp parallel for schedule(static) num_threads(8)
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
                for (size_t idx : indices) {
                    if (triangleBoxOverlap(childCenter, childHalfSize, (*allTriangles)[idx])) {
                        childIndices[i].push_back(idx);
                    }
                }
            }
        } else {
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
};

// ============================================================================
// Flattened octree (used by both the CPU workers and the CUDA kernels)
// ============================================================================

struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

// Triangle payload stored inline in the leaves: avoids a second indirection during
// traversal and keeps the hot data contiguous.
struct LeafTri {
    Vec3 a, b, c;
    uint32_t idx;
    uint32_t pad[2];
};

struct FlatOctree {
    std::vector<FlatNode> nodes;
    std::vector<LeafTri> leaves;
};

static int flattenOctree(const Octree* node, const std::vector<Triangle>& tris, FlatOctree& out) {
    int id = static_cast<int>(out.nodes.size());
    out.nodes.emplace_back();
    FlatNode fn{};
    fn.center = node->center;
    fn.halfExtent = node->halfExtent;
    for(int i = 0; i < 8; ++i) fn.child[i] = -1;
    fn.triStart = 0;
    fn.triCount = 0;

    if(!node->triangleIndices.empty()) {
        fn.triStart = static_cast<int>(out.leaves.size());
        fn.triCount = static_cast<int>(node->triangleIndices.size());
        for(size_t k : node->triangleIndices) {
            LeafTri lt{};
            lt.a = tris[k].a;
            lt.b = tris[k].b;
            lt.c = tris[k].c;
            lt.idx = static_cast<uint32_t>(k);
            out.leaves.push_back(lt);
        }
        out.nodes[id] = fn;
    } else {
        out.nodes[id] = fn;
        for(int i = 0; i < 8; ++i) {
            if(node->children[i]) {
                int c = flattenOctree(node->children[i].get(), tris, out);
                out.nodes[id].child[i] = c;
            }
        }
    }
    return id;
}

// ============================================================================
// Ray casting (shared host/device code)
// ============================================================================

HD inline bool rayIntersectsBoxFlat(const FlatNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

    if (std::abs(c.x) > n.halfExtent.x + ad.x) return false;
    if (std::abs(c.y) > n.halfExtent.y + ad.y) return false;
    if (std::abs(c.z) > n.halfExtent.z + ad.z) return false;

    if (std::abs(d.y * c.z - d.z * c.y) > n.halfExtent.y * ad.z + n.halfExtent.z * ad.y + EPSILON) return false;
    if (std::abs(d.z * c.x - d.x * c.z) > n.halfExtent.z * ad.x + n.halfExtent.x * ad.z + EPSILON) return false;
    if (std::abs(d.x * c.y - d.y * c.x) > n.halfExtent.x * ad.y + n.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Ray-Triangle Intersection (Möller-Trumbore algorithm)
HD inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                     const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return VAL_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return VAL_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return VAL_MAX;

    return e2.dot(qvec) * invDet;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal; the result is independent of the visiting order.
HD inline bool isRayBlocked(const Vec3& from, const Vec3& to, const FlatNode* nodes,
                            const LeafTri* leaves, uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;
    while(sp > 0) {
        const FlatNode& n = nodes[stack[--sp]];
        if(n.triCount > 0) {
            for(int k = 0; k < n.triCount; ++k) {
                const LeafTri lt = leaves[n.triStart + k];
                if(lt.idx == srcTriIdx || lt.idx == dstTriIdx) continue;
                val_t dist = rayTriangleIntersect(from, dirNorm, lt.a, lt.b, lt.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for(int i = 7; i >= 0; --i) {
                int c = n.child[i];
                if(c >= 0 && rayIntersectsBoxFlat(nodes[c], from, to)) stack[sp++] = c;
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Generate a random point inside a triangle using barycentric coordinates
HD inline Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// True if the pair is back-face culled, i.e. consumes no random numbers at all.
HD inline bool pairCulled(const Vec3& nI, const Vec3& nJ) { return nI.dot(nJ) > 0.99f; }

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter).
// 'rnd' holds the RND_PER_PAIR random values this pair would have drawn sequentially.
HD inline val_t computeKij(uint32_t idxI, uint32_t idxJ, const Triangle* triangles,
                           const FlatNode* nodes, const LeafTri* leaves, const float* rnd) {
    const Triangle triI = triangles[idxI];
    const Triangle triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (pairCulled(triI._normal, triJ._normal)) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rnd[4 * r + 0], rnd[4 * r + 1]);
        Vec3 pJ = randomPointInTriangle(triJ, rnd[4 * r + 2], rnd[4 * r + 3]);

        if (isRayBlocked(pI, pJ, nodes, leaves, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI._normal);
        val_t cosPhiJ = cosPhi(-v, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
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

            newFaces.reserve(faces.size() * 4);
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
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[faces[f][2]], vertices[faces[f][1]], vertices[faces[f][0]]);
        }
    }
};

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD inline int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// CUDA kernels
// ============================================================================

// Form factors: one thread per ray, i.e. NUM_RAYS lanes cooperate on one triangle pair.
// A pair of the current row window is identified by its linear index p: it belongs to row
// p/N and column p%N, and its random numbers live at a fixed stride in the window buffer
// (at exactly the offsets the sequential stream would have used).  Using one thread per
// ray gives NUM_RAYS times more threads - the octree traversal is heavily latency bound -
// and the rays of one pair follow very similar paths, so the lanes of a warp stay
// coherent.  The per-ray contributions are summed strictly in ray order, exactly like the
// sequential loop does.
__global__ void kijKernel(size_t firstPair, int numPairs, uint32_t N, int rowBase,
                          const uint32_t* __restrict__ myRows, const Triangle* __restrict__ tris,
                          const FlatNode* __restrict__ nodes, const LeafTri* __restrict__ leaves,
                          const float* __restrict__ rnd, float* __restrict__ out) {
    extern __shared__ val_t sTerm[];
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & (NUM_RAYS - 1);
    const int pairSlot = gid / NUM_RAYS;
    if(pairSlot >= numPairs) return;

    const size_t p = firstPair + static_cast<size_t>(pairSlot);
    const uint32_t row = static_cast<uint32_t>(p / N);
    const uint32_t j = static_cast<uint32_t>(p - static_cast<size_t>(row) * N);
    const uint32_t i = myRows[rowBase + row];

    val_t term = ZERO;
    bool active = i != j;
    if(active) {
        const Triangle triI = tris[i];
        const Triangle triJ = tris[j];
        if(pairCulled(triI._normal, triJ._normal)) {
            active = false;
        } else {
            const float* r4 = rnd + static_cast<size_t>(pairSlot) * RND_PER_PAIR + 4 * lane;
            Vec3 pI = randomPointInTriangle(triI, r4[0], r4[1]);
            Vec3 pJ = randomPointInTriangle(triJ, r4[2], r4[3]);
            if(!isRayBlocked(pI, pJ, nodes, leaves, i, j)) {
                Vec3 v = pJ - pI;
                val_t distSqr = v.squaredNorm();
                if(distSqr >= EPSILON) {
                    val_t cosPhiI = cosPhi(v, triI._normal);
                    val_t cosPhiJ = cosPhi(-v, triJ._normal);
                    if(cosPhiI > ZERO && cosPhiJ > ZERO) term = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }
    }
    sTerm[threadIdx.x] = term;
    __syncwarp();
    if(lane == 0) {
        val_t kij = ZERO;
        if(active) {
            #pragma unroll
            for(int r = 0; r < NUM_RAYS; ++r) kij += sTerm[threadIdx.x + r];
            kij *= INV_NUM_RAYS;
        }
        out[pairSlot] = kij;
    }
}

// Transpose a chunk of locally owned Kij rows into the column-major layout used by the
// propagation kernel (so that consecutive threads read consecutive addresses).
__global__ void transposeKij(const float* __restrict__ src, float* __restrict__ dst, int rows,
                             uint32_t N, int nLocal, int rowOffset) {
    __shared__ float tile[32][33];
    uint32_t j0 = blockIdx.x * 32;
    int r0 = blockIdx.y * 32;
    uint32_t j = j0 + threadIdx.x;
    for(int k = 0; k < 32; k += 8) {
        int r = r0 + threadIdx.y + k;
        if(r < rows && j < N) tile[threadIdx.y + k][threadIdx.x] = src[static_cast<size_t>(r) * N + j];
    }
    __syncthreads();
    int r = r0 + threadIdx.x;
    for(int k = 0; k < 32; k += 8) {
        uint32_t jj = j0 + threadIdx.y + k;
        if(r < rows && jj < N)
            dst[static_cast<size_t>(jj) * nLocal + rowOffset + r] = tile[threadIdx.x][threadIdx.y + k];
    }
}

// Radiosity propagation for one timestep; one thread per locally owned triangle.
// The summation order over j is identical to the sequential version.
__global__ void simKernel(int rowCount, int nLocal, uint32_t N, int t,
                          const uint32_t* __restrict__ myRows,
                          const float* __restrict__ kijT, const Vec3* __restrict__ centers,
                          const float* __restrict__ areas, const float* __restrict__ rho,
                          const float* __restrict__ radB, const float* __restrict__ radE,
                          float* __restrict__ outRow) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if(li >= rowCount) return;
    uint32_t i = myRows[li];
    const Vec3 ci = centers[i];
    val_t sumB = ZERO;

    // The number of rows is small, so occupancy is low: the emitter loop is unrolled
    // and software pipelined to keep several independent loads in flight.  The
    // accumulation itself stays strictly in j order, as in the sequential code.
    constexpr int U = 8;
    uint32_t j = 0;
    for(; j + U <= N; j += U) {
        val_t kv[U];
        val_t rv[U];
        #pragma unroll
        for(int u = 0; u < U; ++u) kv[u] = kijT[static_cast<size_t>(j + u) * nLocal + li];
        #pragma unroll
        for(int u = 0; u < U; ++u) {
            int tauij = computeTau(ci, centers[j + u]);
            rv[u] = (kv[u] > ZERO && t >= tauij)
                        ? radB[static_cast<size_t>(t - tauij) * N + j + u]
                        : ZERO;
        }
        #pragma unroll
        for(int u = 0; u < U; ++u) {
            uint32_t jj = j + u;
            if(i == jj || kv[u] <= ZERO || rv[u] <= ZERO) continue;
            sumB += fminf(kv[u] * areas[jj], ONE) * rv[u];
        }
    }
    for(; j < N; ++j) {
        if(i == j) continue;
        val_t kij = kijT[static_cast<size_t>(j) * nLocal + li];
        if(kij <= ZERO) continue;
        int tauij = computeTau(ci, centers[j]);
        if(t < tauij) continue;
        val_t radJ = radB[static_cast<size_t>(t - tauij) * N + j];
        if(radJ <= ZERO) continue;
        sumB += fminf(kij * areas[j], ONE) * radJ;
    }
    outRow[li] = rho[i] * sumB + radE[static_cast<size_t>(t) * N + i];
}

// Cross-correlation: one block per locally owned triangle, threads split the lag range.
__global__ void distKernel(uint32_t N, uint32_t T, const uint32_t* __restrict__ myRows,
                           const float* __restrict__ radB, uint32_t srcIndex,
                           float* __restrict__ distOut) {
    extern __shared__ char smem[];
    float* sVal = reinterpret_cast<float*>(smem);
    int* sT = reinterpret_cast<int*>(sVal + blockDim.x);

    int li = blockIdx.x;
    uint32_t i = myRows[li];

    val_t bestVal = ZERO;
    int bestT = 0;
    for(uint32_t t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for(uint32_t tt = t; tt < T; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * N + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * N + srcIndex];
            sum += pS * pB;
        }
        if(sum > bestVal) {
            bestVal = sum;
            bestT = static_cast<int>(t);
        }
    }
    sVal[threadIdx.x] = bestVal;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    // argmax reduction, ties resolved towards the smaller lag (as in the serial scan)
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(static_cast<int>(threadIdx.x) < s) {
            float ov = sVal[threadIdx.x + s];
            int ot = sT[threadIdx.x + s];
            if(ov > sVal[threadIdx.x] || (ov == sVal[threadIdx.x] && ot < sT[threadIdx.x])) {
                sVal[threadIdx.x] = ov;
                sT[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) distOut[li] = WAVE_SPEED * static_cast<val_t>(sT[0]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<Vec3> centers;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flat;                // Flattened octree for CPU workers / CUDA

    // --- distributed data ---
    std::vector<uint32_t> myRows;   // globally owned triangle rows of this rank
    std::vector<val_t> kijLocal;    // form factors of the owned rows (nLocal x N)

    size_t sourceIndex = 0;

    // --- MPI / device context ---
    int rank = 0, nranks = 1;
    int device = 0;

    // --- device buffers ---
    Triangle* d_tris = nullptr;
    FlatNode* d_nodes = nullptr;
    LeafTri* d_leaves = nullptr;
    Vec3* d_centers = nullptr;
    float* d_areas = nullptr;
    float* d_rho = nullptr;
    float* d_radB = nullptr;
    float* d_radE = nullptr;
    float* d_kijT = nullptr;
    uint32_t* d_myRows = nullptr;
    float* d_rowOut = nullptr;
    float* d_dist = nullptr;

    size_t nLocal() const { return myRows.size(); }
};

// ============================================================================
// Initialization
// ============================================================================

// Rows are handed out to the ranks in small cyclic blocks so that geometric cost
// variations average out between the ranks.
constexpr int ROW_BLOCK = 4;

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.rank == 0;

    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if(root) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if(root) printf("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctree(&state.octree, state.triangles, state.flat);

    const size_t N = state.numTriangles;

    // Initialize areas / centers
    state.areas.resize(N);
    state.centers.resize(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
        state.centers[i] = state.triangles[i].center();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Distribute the rows (cyclic blocks)
    for(size_t b = 0; b < N; b += ROW_BLOCK) {
        if(static_cast<int>((b / ROW_BLOCK) % static_cast<size_t>(state.nranks)) == state.rank) {
            for(size_t i = b; i < std::min(b + ROW_BLOCK, N); ++i)
                state.myRows.push_back(static_cast<uint32_t>(i));
        }
    }

    // Initialize matrices
    state.kijLocal.assign(state.nLocal() * N, ZERO);
    state.radE.assign(timesteps * N, ZERO);
    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[t * N + state.sourceIndex] = 1.0f;
    }

    // ---- device setup ----
    CUDA_CHECK(cudaSetDevice(state.device));
    CUDA_CHECK(cudaMalloc(&state.d_tris, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(state.d_tris, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, state.flat.nodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, state.flat.nodes.data(),
                          state.flat.nodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_leaves, std::max<size_t>(1, state.flat.leaves.size()) * sizeof(LeafTri)));
    if(!state.flat.leaves.empty())
        CUDA_CHECK(cudaMemcpy(state.d_leaves, state.flat.leaves.data(),
                              state.flat.leaves.size() * sizeof(LeafTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_centers, N * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(state.d_centers, state.centers.data(), N * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_radB, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, std::max<size_t>(1, timesteps * N) * sizeof(float)));
    if(timesteps * N > 0)
        CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), timesteps * N * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_myRows, std::max<size_t>(1, state.nLocal()) * sizeof(uint32_t)));
    if(state.nLocal() > 0)
        CUDA_CHECK(cudaMemcpy(state.d_myRows, state.myRows.data(), state.nLocal() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.d_rowOut, std::max<size_t>(1, state.nLocal()) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_dist, std::max<size_t>(1, state.nLocal()) * sizeof(float)));
}

// ============================================================================
// Precomputation Phase: form factors
// ============================================================================
//
// The rows owned by a rank are processed in windows of a few rows.  For a window, the
// random numbers of the whole window are generated up front (in parallel, one row per
// thread, walking the row exactly like the sequential code would).  The pairs of the
// window are then consumed from one shared atomic counter by the CPU threads (small
// chunks) and by the GPU (large contiguous batches), so the CPU/GPU load balance adapts
// itself to the machine at runtime.

constexpr int CPU_CHUNK_PAIRS = 64;                // pairs a CPU worker grabs at a time
constexpr size_t GPU_BATCH_PAIRS = 1 << 16;        // pairs per CUDA kernel launch
constexpr size_t RND_WINDOW_BYTES = 256ull << 20;  // host memory budget for the window

void computeFormFactors(SimulationState& state) {
    const bool root = state.rank == 0;
    if(root) printf("Computing form factors (Kij)...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const int nLocal = static_cast<int>(state.nLocal());

    // ---- random stream bookkeeping -----------------------------------------
    // Number of random draws consumed by every row of the global matrix.  Only the
    // (purely geometric) back-face cull decides whether a pair draws numbers at all.
    std::vector<uint64_t> rowDraws(N, 0);
    #pragma omp parallel for schedule(static)
    for(uint32_t i = 0; i < N; ++i) {
        const Vec3 nI = state.triangles[i]._normal;
        uint32_t cnt = 0;
        for(uint32_t j = 0; j < N; ++j) {
            if(i == j) continue;
            if(!pairCulled(nI, state.triangles[j]._normal)) ++cnt;
        }
        rowDraws[i] = static_cast<uint64_t>(cnt) * RND_PER_PAIR;
    }

    // RNG state at the start of each owned row (owned rows are in increasing order).
    // Walking the global stream is strictly sequential, so it runs in the background and
    // the row windows below simply wait until their rows have been reached.
    std::vector<MT19937> rowState(std::max(1, nLocal));
    std::atomic<int> rowsReady{0};
    std::thread rngWalker([&] {
        MT19937 eng;
        eng.seed(42);
        uint64_t pos = 0, prefix = 0;
        uint32_t nextRow = 0;
        for(int li = 0; li < nLocal; ++li) {
            uint32_t row = state.myRows[li];
            for(; nextRow < row; ++nextRow) prefix += rowDraws[nextRow];
            eng.advance(prefix - pos);
            pos = prefix;
            rowState[li] = eng;
            rowsReady.store(li + 1, std::memory_order_release);
        }
    });

    // ---- buffers -----------------------------------------------------------
    const size_t rndPerRow = static_cast<size_t>(N) * RND_PER_PAIR;
    int windowRows = static_cast<int>(RND_WINDOW_BYTES / (rndPerRow * sizeof(float)));
    windowRows = std::max(1, std::min(windowRows, std::max(1, nLocal)));

    int nThreads = omp_get_max_threads();
    if(nThreads < 2) nThreads = 2;

    CUDA_CHECK(cudaSetDevice(state.device));

    // The window buffer is ordinary memory (page locking hundreds of MB would cost more
    // than it saves); batches are staged through a small pinned buffer instead.
    std::unique_ptr<float[]> rndWindow;
    if(nLocal > 0) rndWindow.reset(new float[static_cast<size_t>(windowRows) * rndPerRow]);
    float* h_rnd = rndWindow.get();

    constexpr int NBUF = 2;
    cudaStream_t stream[NBUF];
    cudaEvent_t done[NBUF];
    float* h_out[NBUF] = {nullptr, nullptr};
    float* h_stage[NBUF] = {nullptr, nullptr};
    float* d_rnd[NBUF] = {nullptr, nullptr};
    float* d_out[NBUF] = {nullptr, nullptr};
    size_t batchFirst[NBUF] = {0, 0};   // first pair index (within window) of the batch
    int batchCount[NBUF] = {0, 0};
    int batchRowBase[NBUF] = {0, 0};
    bool inFlight[NBUF] = {false, false};
    if(nLocal > 0) {
        for(int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaStreamCreate(&stream[b]));
            CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaMallocHost(&h_out[b], GPU_BATCH_PAIRS * sizeof(float)));
            CUDA_CHECK(cudaMallocHost(&h_stage[b], GPU_BATCH_PAIRS * RND_PER_PAIR * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_rnd[b], GPU_BATCH_PAIRS * RND_PER_PAIR * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out[b], GPU_BATCH_PAIRS * sizeof(float)));
        }
    }

    // ---- window loop -------------------------------------------------------
    for(int winStart = 0; winStart < nLocal; winStart += windowRows) {
        const int rows = std::min(windowRows, nLocal - winStart);
        const size_t windowPairs = static_cast<size_t>(rows) * N;

        // Phase 1: generate the random numbers of the window (one row per thread).
        while(rowsReady.load(std::memory_order_acquire) < winStart + rows) std::this_thread::yield();
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nThreads)
        for(int r = 0; r < rows; ++r) {
            const uint32_t i = state.myRows[winStart + r];
            const Vec3 nI = state.triangles[i]._normal;
            MT19937 eng = rowState[winStart + r];
            float* base = h_rnd + static_cast<size_t>(r) * rndPerRow;
            for(uint32_t j = 0; j < N; ++j) {
                if(j == i || pairCulled(nI, state.triangles[j]._normal)) continue;
                eng.genFloats(base + static_cast<size_t>(j) * RND_PER_PAIR, RND_PER_PAIR);
            }
        }

        // Phase 2: hybrid CPU/GPU processing of the window's pairs.
        std::atomic<size_t> nextPair{0};

        #pragma omp parallel num_threads(nThreads)
        {
            const int tid = omp_get_thread_num();
            if(tid == 0) {
                // ---------------- GPU driver ----------------
                CUDA_CHECK(cudaSetDevice(state.device));

                auto drain = [&](int b) {
                    if(!inFlight[b]) return;
                    CUDA_CHECK(cudaEventSynchronize(done[b]));
                    std::memcpy(&state.kijLocal[static_cast<size_t>(batchRowBase[b]) * N + batchFirst[b]],
                                h_out[b], static_cast<size_t>(batchCount[b]) * sizeof(float));
                    inFlight[b] = false;
                };

                int cur = 0;
                while(true) {
                    size_t taken = nextPair.load(std::memory_order_relaxed);
                    if(taken >= windowPairs) break;
                    size_t remaining = windowPairs - taken;
                    // Taper the batch size towards the end so that the GPU does not
                    // swallow the tail of the window.
                    size_t want = std::min(GPU_BATCH_PAIRS, std::max<size_t>(remaining / 3, 1));
                    size_t start = nextPair.fetch_add(want, std::memory_order_relaxed);
                    if(start >= windowPairs) break;
                    int nP = static_cast<int>(std::min(want, windowPairs - start));

                    drain(cur);

                    batchFirst[cur] = start;
                    batchCount[cur] = nP;
                    batchRowBase[cur] = winStart;

                    const size_t rndBytes = static_cast<size_t>(nP) * RND_PER_PAIR * sizeof(float);
                    std::memcpy(h_stage[cur], h_rnd + start * RND_PER_PAIR, rndBytes);
                    CUDA_CHECK(cudaMemcpyAsync(d_rnd[cur], h_stage[cur], rndBytes,
                                               cudaMemcpyHostToDevice, stream[cur]));
                    const int threads = 128;
                    const long long lanes = static_cast<long long>(nP) * NUM_RAYS;
                    const int blocks = static_cast<int>((lanes + threads - 1) / threads);
                    kijKernel<<<blocks, threads, threads * sizeof(val_t), stream[cur]>>>(
                        start, nP, N, winStart, state.d_myRows, state.d_tris, state.d_nodes,
                        state.d_leaves, d_rnd[cur], d_out[cur]);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpyAsync(h_out[cur], d_out[cur],
                                               static_cast<size_t>(nP) * sizeof(float),
                                               cudaMemcpyDeviceToHost, stream[cur]));
                    CUDA_CHECK(cudaEventRecord(done[cur], stream[cur]));
                    inFlight[cur] = true;
                    cur = (cur + 1) % NBUF;
                }
                for(int b = 0; b < NBUF; ++b) drain(b);
            } else {
                // ---------------- CPU workers ----------------
                while(true) {
                    size_t start = nextPair.fetch_add(CPU_CHUNK_PAIRS, std::memory_order_relaxed);
                    if(start >= windowPairs) break;
                    const size_t end = std::min(start + CPU_CHUNK_PAIRS, windowPairs);
                    for(size_t p = start; p < end; ++p) {
                        const uint32_t r = static_cast<uint32_t>(p / N);
                        const uint32_t j = static_cast<uint32_t>(p - static_cast<size_t>(r) * N);
                        const uint32_t i = state.myRows[winStart + r];
                        val_t v = ZERO;
                        if(i != j) {
                            v = computeKij(i, j, state.triangles.data(), state.flat.nodes.data(),
                                           state.flat.leaves.data(), h_rnd + p * RND_PER_PAIR);
                        }
                        state.kijLocal[static_cast<size_t>(winStart + r) * N + j] = v;
                    }
                }
            }
        }
    }

    rngWalker.join();
    if(nLocal > 0) {
        for(int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaStreamDestroy(stream[b]));
            CUDA_CHECK(cudaEventDestroy(done[b]));
            CUDA_CHECK(cudaFreeHost(h_out[b]));
            CUDA_CHECK(cudaFreeHost(h_stage[b]));
            CUDA_CHECK(cudaFree(d_rnd[b]));
            CUDA_CHECK(cudaFree(d_out[b]));
        }
    }

    // Upload the form factors in the transposed layout needed by the propagation kernel.
    if(nLocal > 0) {
        CUDA_CHECK(cudaMalloc(&state.d_kijT, static_cast<size_t>(nLocal) * N * sizeof(float)));
        const size_t rowBytes = static_cast<size_t>(N) * sizeof(float);
        const int chunkRows = std::max(1, static_cast<int>(std::min<size_t>(nLocal, (64ull << 20) / rowBytes)));
        float* d_tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_tmp, static_cast<size_t>(chunkRows) * N * sizeof(float)));
        for(int r0 = 0; r0 < nLocal; r0 += chunkRows) {
            int rows = std::min(chunkRows, nLocal - r0);
            CUDA_CHECK(cudaMemcpy(d_tmp, &state.kijLocal[static_cast<size_t>(r0) * N],
                                  static_cast<size_t>(rows) * rowBytes, cudaMemcpyHostToDevice));
            dim3 block(32, 8);
            dim3 grid((N + 31) / 32, (rows + 31) / 32);
            transposeKij<<<grid, block>>>(d_tmp, state.d_kijT, rows, N, nLocal, r0);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_tmp));
    }

    // Progress reporting: the rows are processed out of order and in parallel, so the
    // original per-row progress lines are emitted once the phase is complete.
    if(root) {
        for(size_t i = 0; i < state.numTriangles; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == state.numTriangles) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, state.numTriangles);
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    // The time delays are a pure function of the triangle centers and are evaluated
    // directly inside the propagation kernel; materializing the full N x N matrix would
    // only add O(N^2) memory traffic.
    if(state.rank == 0) printf("Computing time delays (Tau)...\n");
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = state.rank == 0;
    if(root) printf("Running wave propagation simulation...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const int nLocal = static_cast<int>(state.nLocal());
    CUDA_CHECK(cudaSetDevice(state.device));

    float* hostLocal = nullptr;   // pinned, so that the D2H copy is truly asynchronous
    CUDA_CHECK(cudaMallocHost(&hostLocal, std::max(1, nLocal) * sizeof(float)));
    std::memset(hostLocal, 0, std::max(1, nLocal) * sizeof(float));
    std::vector<float> fullRow(N);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // The number of rows per timestep is only O(N/ranks), which is too little work to
    // saturate a GPU while the CPU cores would idle.  The rows are therefore split
    // between the GPU and the OpenMP threads, with the split point re-tuned every
    // timestep from the measured throughput of both sides.
    int gpuRows = nLocal;   // start GPU heavy, the controller below takes over
    int gr = 0;
    std::chrono::high_resolution_clock::time_point c0;
    const int threads = 128;

    // One single parallel region for the whole time loop: the per-timestep work is far
    // too short to pay for repeated OpenMP fork/join.
    // Leave a little CPU headroom: the master thread also has to drive the GPU and the
    // MPI reduction while the others wait on the barrier.
    const int simThreads = std::max(1, (omp_get_max_threads() * 3) / 4);
    #pragma omp parallel num_threads(simThreads)
    {
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        #pragma omp master
        {
            gr = std::min(gpuRows, nLocal);
            if(gr > 0) {
                simKernel<<<(gr + threads - 1) / threads, threads, 0, stream>>>(
                    gr, nLocal, N, static_cast<int>(t), state.d_myRows, state.d_kijT,
                    state.d_centers, state.d_areas, state.d_rho, state.d_radB, state.d_radE,
                    state.d_rowOut);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(hostLocal, state.d_rowOut, gr * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
            }
            c0 = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        const int it = static_cast<int>(t);
        #pragma omp for schedule(static)
        for(int li = gr; li < nLocal; ++li) {
            const uint32_t i = state.myRows[li];
            const Vec3 ci = state.centers[i];
            const val_t* kijRow = &state.kijLocal[static_cast<size_t>(li) * N];
            val_t sumB = ZERO;
            for(uint32_t j = 0; j < N; ++j) {
                if(i == j) continue;
                const val_t kij = kijRow[j];
                if(kij <= ZERO) continue;

                const int tauij = computeTau(ci, state.centers[j]);
                // Skip if wave hasn't yet propagated from j to i
                if(it < tauij) continue;

                const val_t radJ = state.radB[static_cast<size_t>(it - tauij) * N + j];
                if(radJ <= ZERO) continue;

                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }
            hostLocal[li] = state.rho[i] * sumB + state.radE[static_cast<size_t>(it) * N + i];
        }

        #pragma omp master
        {
            const auto c1 = std::chrono::high_resolution_clock::now();
            if(gr > 0) CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto c2 = std::chrono::high_resolution_clock::now();

            // Re-balance: aim for both sides finishing at the same time.  A direct
            // feedback controller on the time difference is used instead of a throughput
            // model, because the GPU time is dominated by launch latency for small row
            // counts.
            const double tCpu = std::chrono::duration<double>(c1 - c0).count();
            const double tGpu = std::chrono::duration<double>(c2 - c0).count();
            if(gr > 0 && gr < nLocal && tCpu + tGpu > 1e-9) {
                const double imbalance = (tCpu - tGpu) / (tCpu + tGpu);
                int step = static_cast<int>(0.25 * imbalance * nLocal);
                if(step == 0) step = imbalance > 0 ? 1 : (imbalance < 0 ? -1 : 0);
                gpuRows = std::max(0, std::min(nLocal, gr + step));
            } else if(gr == 0) {
                gpuRows = std::max(1, nLocal / 32);           // probe the GPU again
            } else if(gr == nLocal && nLocal > 1) {
                gpuRows = nLocal - std::max(1, nLocal / 32);  // probe the CPU again
            }

            std::fill(fullRow.begin(), fullRow.end(), ZERO);
            for(int li = 0; li < nLocal; ++li) fullRow[state.myRows[li]] = hostLocal[li];
            if(state.nranks > 1)
                MPI_Allreduce(MPI_IN_PLACE, fullRow.data(), static_cast<int>(N), MPI_FLOAT,
                              MPI_SUM, MPI_COMM_WORLD);

            std::memcpy(&state.radB[t * N], fullRow.data(), N * sizeof(float));
            CUDA_CHECK(cudaMemcpy(state.d_radB + static_cast<size_t>(t) * N, fullRow.data(),
                                  N * sizeof(float), cudaMemcpyHostToDevice));

            if(root && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
            }
        }
        #pragma omp barrier
    }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hostLocal));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if(state.rank == 0) printf("Computing distances via cross-correlation...\n");

    const uint32_t N = static_cast<uint32_t>(state.numTriangles);
    const int nLocal = static_cast<int>(state.nLocal());
    CUDA_CHECK(cudaSetDevice(state.device));

    std::vector<float> hostLocal(std::max(1, nLocal), ZERO);
    if(nLocal > 0 && state.numTimesteps > 0) {
        const int threads = 128;
        const size_t shmem = threads * (sizeof(float) + sizeof(int));
        distKernel<<<nLocal, threads, shmem>>>(N, static_cast<uint32_t>(state.numTimesteps),
                                               state.d_myRows, state.d_radB,
                                               static_cast<uint32_t>(state.sourceIndex), state.d_dist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(hostLocal.data(), state.d_dist, nLocal * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    std::fill(state.distances.begin(), state.distances.end(), ZERO);
    for(int li = 0; li < nLocal; ++li) state.distances[state.myRows[li]] = hostLocal[li];
    if(state.nranks > 1)
        MPI_Allreduce(MPI_IN_PLACE, state.distances.data(), static_cast<int>(N), MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    // Kij is distributed, so its statistics need a global reduction first.
    long long localNonZeroKij = 0;
    #pragma omp parallel for schedule(static) reduction(+ : localNonZeroKij)
    for (size_t i = 0; i < state.kijLocal.size(); ++i) {
        if (state.kijLocal[i] > EPSILON) localNonZeroKij++;
    }
    long long nonZeroKijGlobal = localNonZeroKij;
    if(state.nranks > 1)
        MPI_Allreduce(MPI_IN_PLACE, &nonZeroKijGlobal, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    if(state.rank != 0) return true;

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
            if (state.radB[t * state.numTriangles + i] > EPSILON) {
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
    const size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
           nonZeroKijGlobal, totalKij,
           100.0f * static_cast<val_t>(nonZeroKijGlobal) / static_cast<val_t>(totalKij));

    if (nonZeroKijGlobal == 0) {
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = rank == 0;

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
            if(root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if(root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // ---- MPI/GPU/OpenMP topology ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);

    int deviceCount = 0;
    if(cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        if(root) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // establish the context up front

    if(getenv("OMP_NUM_THREADS") == nullptr) {
        // Use the CPUs this rank is actually allowed to run on, divided by the number of
        // ranks that share the very same affinity mask (works both for bound and for
        // unbound MPI launches).
        const int totalCpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        cpu_set_t mask;
        int avail = omp_get_num_procs();
        int firstCpu = -1;
        if(sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            avail = CPU_COUNT(&mask);
            for(int c = 0; c < CPU_SETSIZE; ++c) {
                if(CPU_ISSET(c, &mask)) { firstCpu = c; break; }
            }
        }
        // A launcher that binds every rank to a single core (the default of several MPI
        // implementations) would leave the node almost idle for a hybrid code: widen the
        // mask again in that case.
        if(static_cast<long>(avail) * localSize < totalCpus) {
            cpu_set_t full;
            CPU_ZERO(&full);
            for(int c = 0; c < totalCpus && c < CPU_SETSIZE; ++c) CPU_SET(c, &full);
            if(sched_setaffinity(0, sizeof(full), &full) == 0) {
                avail = totalCpus;
                firstCpu = 0;
            }
        }
        std::vector<int> firsts(localSize, -1);
        MPI_Allgather(&firstCpu, 1, MPI_INT, firsts.data(), 1, MPI_INT, nodeComm);
        int sharing = 0;
        for(int c : firsts) if(c == firstCpu) ++sharing;
        omp_set_num_threads(std::max(1, avail / std::max(1, sharing)));
    }
    omp_set_max_active_levels(2);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if(root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nranks,
               omp_get_max_threads(), deviceCount);
        printf("\n");
    }

    // Initialize
    SimulationState state;
    state.rank = rank;
    state.nranks = nranks;
    state.device = device;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if(root) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if(root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if(root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if(root) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    if(root) {
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
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
    }

    // Validation
    int failed = 0;
    if (validate) {
        if (!validateResults(state)) failed = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return failed;
}
