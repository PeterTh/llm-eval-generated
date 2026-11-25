/**
 * Room Response Simulation Benchmark
 *
 * This benchmark implements distributed room impulse response
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
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
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
constexpr val_t ROOM_RADIUS = 10.0f;
// Every triangle center lies within the room radius. The ring can hold the
// maximum propagation delay and the largest independent batch simultaneously.
constexpr int HISTORY_SIZE = 128;
static_assert(HISTORY_SIZE > 2 * (2 * ROOM_RADIUS * INV_WAVE_SPEED + 1));
static_assert((HISTORY_SIZE & (HISTORY_SIZE - 1)) == 0);

#define HOST_DEVICE __host__ __device__

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HOST_DEVICE constexpr Vec3() : x(0), y(0), z(0) {}
    HOST_DEVICE constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}

    HOST_DEVICE Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HOST_DEVICE Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HOST_DEVICE Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HOST_DEVICE Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HOST_DEVICE Vec3 operator-() const { return {-x, -y, -z}; }

    HOST_DEVICE val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HOST_DEVICE Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HOST_DEVICE val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
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

int hostThreads(size_t work, size_t itemsPerThread) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                           std::max<size_t>(1, work / itemsPerThread)));
}

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

        #pragma omp parallel num_threads(hostThreads(triangles.size(), 512))
        {
            #pragma omp single
            buildNode(allIndices, minBound, maxBound);
        }
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
                Octree* child = children[i].get();
                if (childIndices[i].size() >= 256 && omp_get_num_threads() > 1) {
                    #pragma omp task shared(childIndices) firstprivate(i, child, childMin, childMax)
                    child->buildNode(childIndices[i], childMin, childMax);
                } else {
                    child->buildNode(childIndices[i], childMin, childMax);
                }
            }
        }
        #pragma omp taskwait
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
            // Reverse winding to make normals point inward.
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};


// CUDA failures must terminate all ranks, including peers inside collectives.
void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA(operation) checkCuda((operation), #operation)

template <typename T, bool Pinned = false>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(size_t count) { allocate(count); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    ~Buffer() {
        if (!data) return;
        if constexpr (Pinned) cudaFreeHost(data);
        else cudaFree(data);
    }

    void allocate(size_t count) {
        if (!count) return;
        if constexpr (Pinned) CUDA(cudaMallocHost(&data, count * sizeof(T)));
        else CUDA(cudaMalloc(&data, count * sizeof(T)));
    }

    T* data = nullptr;
};

template <typename T>
void upload(T* device, const T* host, size_t count) {
    if (count) CUDA(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice));
}

// Depth-first layout. An internal node's first child follows it, and "end"
// skips to the next sibling or to the end of the enclosing subtree.
struct RayNode {
    Vec3 center, halfExtent;
    int end, firstTriangle, triangleCount;
};

void flattenOctree(const Octree& tree, std::vector<RayNode>& nodes, std::vector<idx_t>& indices) {
    size_t node = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[node].end = static_cast<int>(nodes.size());
}

struct FaceGeometry {
    Vec3 origin, edge1, edge2, normal, center;
    val_t area;
};

__device__ bool intersectsBox(const RayNode& node, const Vec3& midpoint,
                              const Vec3& d, const Vec3& ad) {
    Vec3 c = midpoint - node.center;
    Vec3 half = node.halfExtent;
    if (fabsf(c.x) > half.x + ad.x || fabsf(c.y) > half.y + ad.y ||
        fabsf(c.z) > half.z + ad.z) return false;

    return fabsf(d.y * c.z - d.z * c.y) <= half.y * ad.z + half.z * ad.y + EPSILON &&
           fabsf(d.z * c.x - d.x * c.z) <= half.z * ad.x + half.x * ad.z + EPSILON &&
           fabsf(d.x * c.y - d.y * c.x) <= half.x * ad.y + half.y * ad.x + EPSILON;
}

__device__ bool intersectsTriangle(const Vec3& origin, const Vec3& direction, val_t rayLen,
                                   const FaceGeometry& face) {
    Vec3 pvec = direction.cross(face.edge2);
    val_t det = face.edge1.dot(pvec);
    if (fabsf(det) < EPSILON) return false;

    val_t invDet = ONE / det;
    Vec3 tvec = origin - face.origin;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return false;

    Vec3 qvec = tvec.cross(face.edge1);
    val_t v = direction.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    val_t dist = face.edge2.dot(qvec) * invDet;
    return dist > EPSILON && dist < rayLen - EPSILON;
}

struct RoomGeometry {
    const FaceGeometry* faces;
    const RayNode* nodes;
    const idx_t* indices;
};

__device__ bool isRayBlocked(const Vec3& from, const Vec3& v, val_t rayLen,
                             const RoomGeometry& room, idx_t i, idx_t j) {
    Vec3 direction = v / rayLen;
    Vec3 d = v * 0.5f;
    Vec3 midpoint = from + d;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    int end = room.nodes[0].end;
    for (int n = 0; n < end;) {
        const RayNode& node = room.nodes[n];
        if (n && !intersectsBox(node, midpoint, d, ad)) {
            n = node.end;
            continue;
        }
        for (int k = 0; k < node.triangleCount; ++k) {
            idx_t index = room.indices[node.firstTriangle + k];
            if (index != i && index != j &&
                intersectsTriangle(from, direction, rayLen, room.faces[index])) return true;
        }
        ++n;
    }
    return false;
}

// MT19937 with the same initialization and recurrence as std::mt19937.
// Only pairs that pass the original normal test consume random numbers.
constexpr int MT_SIZE = 624;
constexpr int MT_SHIFT = 397;
constexpr int MT_STEP = MT_SIZE - MT_SHIFT;
constexpr int PAIRS_PER_TILE = 64;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr int RAY_THREADS = 256;
constexpr int WARP_SIZE = 32;
static_assert(PAIRS_PER_TILE == 64 && 2 * NUM_RAYS == WARP_SIZE);

HOST_DEVICE uint32_t twist(uint32_t left, uint32_t right) {
    uint32_t joined = (left & 0x80000000u) | (right & 0x7fffffffu);
    return (joined >> 1) ^ ((0u - (joined & 1u)) & 0x9908b0dfu);
}

// Non-leading exponents of the MT19937 characteristic polynomial over GF(2).
// Reducing x^offset by this polynomial represents that many stream advances.
constexpr int MT_DEGREE = 19937;
constexpr int MT_POWERS[] = {
    0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681,
    3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752,
    7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147,
    11157, 11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717, 11838,
    11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509, 12620, 12673,
    12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243,
    13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093,
    14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774,
    14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455,
    15513, 15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252,
    16363, 16368, 16421, 16537, 16590, 16595, 16817, 16822, 16875, 16933,
    17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783, 17841, 17952,
    18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314
};
static_assert(MT_DEGREE - MT_POWERS[sizeof(MT_POWERS) / sizeof(int) - 1] >= 64);

constexpr int MT_WORDS = (MT_DEGREE + 63) / 64;
using Polynomial = std::array<uint64_t, MT_WORDS>;

uint64_t spreadBits(uint32_t bits) {
    // Squaring in GF(2) doubles each term's exponent.
    uint64_t x = bits;
    x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
    x = (x | (x << 8)) & 0x00ff00ff00ff00ffULL;
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x << 2)) & 0x3333333333333333ULL;
    return (x | (x << 1)) & 0x5555555555555555ULL;
}

Polynomial jumpPolynomial(uint64_t draws) {
    Polynomial polynomial{};
    polynomial[0] = 1;
    for (int bit = 63 - __builtin_clzll(draws); bit >= 0; --bit) {
        int shift = (draws >> bit) & 1;
        std::array<uint64_t, 2 * MT_WORDS> squared;
        for (int i = 0; i < MT_WORDS; ++i) {
            squared[2 * i] = spreadBits(static_cast<uint32_t>(polynomial[i])) << shift;
            squared[2 * i + 1] = spreadBits(polynomial[i] >> 32) << shift;
        }

        // Fold 64 high terms at a time. The gap below the leading term
        // makes the terms within this window independent.
        constexpr int word = MT_DEGREE / 64;
        constexpr int offset = MT_DEGREE % 64;
        for (int i = MT_WORDS - 1; i >= 0; --i) {
            uint64_t high = (squared[i + word] >> offset) |
                            (squared[i + word + 1] << (64 - offset));
            if (!high) continue;
            for (int power : MT_POWERS) {
                int target = i + power / 64;
                int shift = power % 64;
                squared[target] ^= high << shift;
                if (shift) squared[target + 1] ^= high >> (64 - shift);
            }
        }
        std::copy_n(squared.begin(), MT_WORDS, polynomial.begin());
        polynomial.back() &= (uint64_t{1} << offset) - 1;
    }
    return polynomial;
}

struct RandomTile {
    uint32_t state[MT_SIZE];
    uint32_t offset;
    uint64_t pairs;
};

class RandomStream {
public:
    RandomStream() {
        state[0] = 42;
        for (uint32_t i = 1; i < MT_SIZE; ++i) {
            state[i] = 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
        }
        refresh();
    }

    void advance(uint64_t draws) {
        draws += cursor;
        while (draws >= MT_SIZE) {
            refresh();
            draws -= MT_SIZE;
        }
        cursor = static_cast<uint32_t>(draws);
    }

    // Seek from the initial state. Each rank can find its own starting
    // window without waiting for its predecessor to generate any draws.
    void seek(uint64_t draws) {
        if (draws < (uint64_t{1} << 24)) {
            advance(draws);
            return;
        }

        Polynomial polynomial = jumpPolynomial(draws);
        std::array<uint32_t, MT_SIZE + MT_DEGREE - 1> window;
        std::copy_n(state.begin(), MT_SIZE, window.begin());
        for (int i = MT_SIZE; i < static_cast<int>(window.size()); ++i) {
            window[i] = window[i - MT_STEP] ^ twist(window[i - MT_SIZE], window[i - MT_SIZE + 1]);
        }

        state.fill(0);
        cursor = 0;
        for (int word = 0; word < MT_WORDS; ++word) {
            uint64_t terms = polynomial[word];
            while (terms) {
                int start = 64 * word + __builtin_ctzll(terms);
                #pragma omp simd
                for (int i = 0; i < MT_SIZE; ++i) state[i] ^= window[start + i];
                terms &= terms - 1;
            }
        }
    }

    void checkpoint(RandomTile& tile, uint64_t pairs) const {
        std::copy_n(state.data(), MT_SIZE, tile.state);
        tile.offset = cursor;
        tile.pairs = pairs;
    }

private:
    std::array<uint32_t, MT_SIZE> state;
    uint32_t cursor = 0;

    void refresh() {
        for (int i = 0; i < MT_STEP; ++i) {
            state[i] = state[i + MT_SHIFT] ^ twist(state[i], state[i + 1]);
        }
        for (int i = MT_STEP; i < MT_SIZE - 1; ++i) {
            state[i] = state[i - MT_STEP] ^ twist(state[i], state[i + 1]);
        }
        state[MT_SIZE - 1] = state[MT_SHIFT - 1] ^ twist(state[MT_SIZE - 1], state[0]);
        cursor = 0;
    }
};

__device__ val_t uniformDraw(uint32_t value) {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680u;
    value ^= (value << 15) & 0xefc60000u;
    value ^= value >> 18;
    return fminf(__uint2float_rn(value) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 sampleFace(const FaceGeometry& face, const uint32_t* draws) {
    val_t u = uniformDraw(draws[0]);
    val_t v = uniformDraw(draws[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {
        fmaf(face.edge2.x, v, fmaf(face.edge1.x, u, face.origin.x)),
        fmaf(face.edge2.y, v, fmaf(face.edge1.y, u, face.origin.y)),
        fmaf(face.edge2.z, v, fmaf(face.edge1.z, u, face.origin.z))
    };
}

__device__ val_t rayContribution(const uint32_t* draws, const RoomGeometry& room,
                                 idx_t i, idx_t j) {
    const FaceGeometry& a = room.faces[i];
    const FaceGeometry& b = room.faces[j];
    Vec3 from = sampleFace(a, draws);
    Vec3 v = sampleFace(b, draws + 2) - from;
    val_t distSqr = v.squaredNorm();
    if (distSqr < EPSILON) return ZERO;

    val_t distance = sqrtf(distSqr);
    val_t cosI = v.dot(a.normal) / distance;
    val_t cosJ = (-v).dot(b.normal) / distance;
    if (cosI <= ZERO || cosJ <= ZERO || isRayBlocked(from, v, distance, room, i, j)) return ZERO;
    return (cosI * cosJ) / (PI * distSqr);
}

__global__ void precompute(const RandomTile* tiles, idx_t firstTile, idx_t tilesPerRow,
                           idx_t n, idx_t firstRow, RoomGeometry room, val_t* weights,
                           uint8_t* delays, int* minDelay, unsigned long long* nonzero) {
    __shared__ uint32_t draws[MT_SIZE + DRAWS_PER_PAIR * PAIRS_PER_TILE];
    __shared__ val_t contributions[RAY_THREADS];
    __shared__ int tileMinDelay;
    __shared__ unsigned tileNonzero;

    const RandomTile& tile = tiles[blockIdx.x];
    idx_t tileIndex = firstTile + blockIdx.x;
    idx_t row = tileIndex / tilesPerRow;
    idx_t i = firstRow + row;
    idx_t firstColumn = (tileIndex % tilesPerRow) * PAIRS_PER_TILE;
    size_t rowOffset = static_cast<size_t>(row) * n;
    uint64_t mask = tile.pairs;
    int lane = threadIdx.x;
    int localMin = HISTORY_SIZE / 2;
    unsigned localNonzero = 0;
    int used = tile.offset + DRAWS_PER_PAIR * __popcll(mask);
    if (lane == 0) {
        tileMinDelay = HISTORY_SIZE / 2;
        tileNonzero = 0;
    }
    for (int k = lane; k < min(used, MT_SIZE); k += RAY_THREADS) draws[k] = tile.state[k];
    __syncthreads();

    // A window of MT_STEP words depends only on words from earlier windows.
    for (int base = MT_SIZE; base < used; base += MT_STEP) {
        int k = base + lane;
        if (lane < MT_STEP && k < used) {
            draws[k] = draws[k - MT_STEP] ^ twist(draws[k - MT_SIZE], draws[k - MT_SIZE + 1]);
        }
        __syncthreads();
    }

    int ray = lane % NUM_RAYS;
    for (int pair = lane / NUM_RAYS; pair < PAIRS_PER_TILE; pair += RAY_THREADS / NUM_RAYS) {
        idx_t j = firstColumn + pair;
        uint64_t bit = uint64_t{1} << pair;
        if (mask & bit) {
            int ordinal = __popcll(mask & (bit - 1));
            contributions[lane] = rayContribution(draws + tile.offset +
                                                   ordinal * DRAWS_PER_PAIR + 4 * ray, room, i, j);
        } else {
            contributions[lane] = ZERO;
        }
        __syncwarp();

        if (ray == 0 && j < n) {
            val_t weight = ZERO;
            int delay = 0;
            if (mask & bit) {
                val_t kij = ZERO;
                for (int r = 0; r < NUM_RAYS; ++r) kij += contributions[lane + r];
                kij *= INV_NUM_RAYS;
                weight = fminf(kij * room.faces[j].area, ONE);
                if (weight > ZERO) {
                    Vec3 d = room.faces[i].center - room.faces[j].center;
                    val_t distance = sqrtf(fmaf(d.z, d.z, fmaf(d.x, d.x, d.y * d.y)));
                    delay = __float2int_ru(distance * INV_WAVE_SPEED);
                    localMin = min(localMin, delay);
                }
                localNonzero += kij > EPSILON;
            }
            delays[rowOffset + j] = delay;
            weights[rowOffset + j] = weight;
        }
        __syncwarp();
    }
    int warpMin = min(localMin, __shfl_xor_sync(0xffffffffu, localMin, NUM_RAYS));
    unsigned warpNonzero = localNonzero +
                           __shfl_xor_sync(0xffffffffu, localNonzero, NUM_RAYS);
    if (lane % WARP_SIZE == 0) {
        atomicMin(&tileMinDelay, warpMin);
        if (nonzero) atomicAdd(&tileNonzero, warpNonzero);
    }
    __syncthreads();
    if (lane == 0) {
        atomicMin(minDelay, tileMinDelay);
        if (nonzero && tileNonzero) atomicAdd(nonzero, tileNonzero);
    }
}

struct Partition {
    std::vector<int> offsets;
    std::vector<int> widths;
    int rank;

    Partition(int n, int rank, int ranks) : offsets(ranks), widths(ranks), rank(rank) {
        for (int i = 0; i < ranks; ++i) {
            offsets[i] = i * (n / ranks) + std::min(i, n % ranks);
            widths[i] = n / ranks + (i < n % ranks);
        }
    }

    int begin() const { return offsets[rank]; }
    int size() const { return widths[rank]; }
    int ranks() const { return static_cast<int>(widths.size()); }
};

struct Simulation {
    const Partition partition;
    const int n, timesteps;
    const int source;
    const val_t reflectivity;
    const size_t pairCount;
    Buffer<val_t> weights, localHistory, ring;
    Buffer<uint8_t> delays;
    int minDelay = HISTORY_SIZE / 2;
    unsigned long long nonzeroFactors = 0, receivingEnergy = 0;
    std::vector<val_t> distances;

    Simulation(int n, int timesteps, size_t sourceIndex, val_t reflectivity, int rank, int ranks)
        : partition(n, rank, ranks), n(n), timesteps(timesteps), source(sourceIndex % n),
          reflectivity(reflectivity), pairCount(static_cast<size_t>(partition.size()) * n),
          weights(pairCount), localHistory(static_cast<size_t>(timesteps) * partition.size()),
          ring(partition.size() ? static_cast<size_t>(HISTORY_SIZE) * n : 0), delays(pairCount) {}
};

void computeFormFactors(Simulation& sim, const std::vector<Triangle>& triangles,
                        const Octree& octree, bool validate) {
    const Partition& part = sim.partition;
    int tilesPerRow = (sim.n + PAIRS_PER_TILE - 1) / PAIRS_PER_TILE;
    size_t tileCount = static_cast<size_t>(part.size()) * tilesPerRow;
    std::vector<uint64_t> masks(tileCount);
    std::vector<Vec3> normals(sim.n);
    #pragma omp parallel for num_threads(hostThreads(normals.size(), 128))
    for (int i = 0; i < sim.n; ++i) normals[i] = triangles[i].normal();

    // Classify on the host with the original floating-point normal test. The
    // masks also assign every eligible pair its position in the serial RNG stream.
    uint64_t draws = 0;
    #pragma omp parallel for reduction(+:draws) num_threads(hostThreads(tileCount, 256))
    for (size_t tile = 0; tile < tileCount; ++tile) {
        int i = part.begin() + tile / tilesPerRow;
        int first = (tile % tilesPerRow) * PAIRS_PER_TILE;
        int count = std::min(PAIRS_PER_TILE, sim.n - first);
        Vec3 normal = normals[i];
        uint64_t mask = 0;
        #pragma omp simd reduction(|:mask)
        for (int bit = 0; bit < count; ++bit) {
            int j = first + bit;
            const Vec3& other = normals[j];
            val_t dot = std::fma(normal.z, other.z,
                                 std::fma(normal.x, other.x, normal.y * other.y));
            if (i != j && dot <= 0.99f) mask |= uint64_t{1} << bit;
        }
        masks[tile] = mask;
        draws += DRAWS_PER_PAIR * __builtin_popcountll(mask);
    }

    uint64_t precedingDraws = 0;
    MPI_Exscan(&draws, &precedingDraws, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    RandomStream stream;
    if (part.rank > 0 && tileCount) stream.seek(precedingDraws);

    int localMinDelay = sim.minDelay;
    unsigned long long localFactors = 0;
    if (tileCount) {
        std::vector<FaceGeometry> faces(sim.n);
        #pragma omp parallel for num_threads(hostThreads(faces.size(), 128))
        for (int i = 0; i < sim.n; ++i) {
            const Triangle& t = triangles[i];
            faces[i] = {t.a, t.b - t.a, t.c - t.a, t.normal(), t.center(), t.area()};
        }
        std::vector<RayNode> nodes;
        std::vector<idx_t> indices;
        flattenOctree(octree, nodes, indices);

        Buffer<FaceGeometry> deviceFaces(faces.size());
        Buffer<RayNode> deviceNodes(nodes.size());
        Buffer<idx_t> deviceIndices(indices.size());
        Buffer<int> deviceMin(1);
        Buffer<unsigned long long> deviceNonzero(validate ? 1 : 0);
        upload(deviceFaces.data, faces.data(), faces.size());
        upload(deviceNodes.data, nodes.data(), nodes.size());
        upload(deviceIndices.data, indices.data(), indices.size());
        upload(deviceMin.data, &localMinDelay, 1);
        upload(deviceNonzero.data, &localFactors, validate ? 1 : 0);
        RoomGeometry room{deviceFaces.data, deviceNodes.data, deviceIndices.data};

        // Pipeline host checkpoint generation, PCIe transfers, and ray tracing.
        size_t capacity = std::min<size_t>(tileCount, 4096);
        int slots = tileCount > capacity ? 2 : 1;
        Buffer<RandomTile, true> hostTiles[2];
        Buffer<RandomTile> deviceTiles[2];
        cudaEvent_t transferred[2], consumed[2];
        cudaStream_t copyStream;
        CUDA(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
        for (int i = 0; i < slots; ++i) {
            hostTiles[i].allocate(capacity);
            deviceTiles[i].allocate(capacity);
            CUDA(cudaEventCreateWithFlags(&transferred[i], cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&consumed[i], cudaEventDisableTiming));
        }
        int slot = 0;
        for (size_t first = 0; first < tileCount; first += capacity, slot ^= 1) {
            if (first >= 2 * capacity) {
                CUDA(cudaEventSynchronize(transferred[slot]));
                CUDA(cudaStreamWaitEvent(copyStream, consumed[slot]));
            }
            size_t count = std::min(capacity, tileCount - first);
            for (size_t i = 0; i < count; ++i) {
                uint64_t pairs = masks[first + i];
                stream.checkpoint(hostTiles[slot].data[i], pairs);
                stream.advance(DRAWS_PER_PAIR * __builtin_popcountll(pairs));
            }
            CUDA(cudaMemcpyAsync(deviceTiles[slot].data, hostTiles[slot].data,
                                 count * sizeof(RandomTile), cudaMemcpyHostToDevice, copyStream));
            CUDA(cudaEventRecord(transferred[slot], copyStream));
            CUDA(cudaStreamWaitEvent(nullptr, transferred[slot]));
            precompute<<<count, RAY_THREADS>>>(deviceTiles[slot].data, first, tilesPerRow,
                                             sim.n, part.begin(), room, sim.weights.data,
                                             sim.delays.data, deviceMin.data, deviceNonzero.data);
            CUDA(cudaGetLastError());
            CUDA(cudaEventRecord(consumed[slot]));
        }
        CUDA(cudaMemcpy(&localMinDelay, deviceMin.data, sizeof(int), cudaMemcpyDeviceToHost));
        if (validate) {
            CUDA(cudaMemcpy(&localFactors, deviceNonzero.data, sizeof(localFactors), cudaMemcpyDeviceToHost));
        }
        for (int i = 0; i < slots; ++i) {
            CUDA(cudaEventDestroy(transferred[i]));
            CUDA(cudaEventDestroy(consumed[i]));
        }
        CUDA(cudaStreamDestroy(copyStream));
    }
    MPI_Allreduce(&localMinDelay, &sim.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (validate) {
        MPI_Reduce(&localFactors, &sim.nonzeroFactors, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

constexpr int PROPAGATION_THREADS = 128;

template <bool SingleRank>
__global__ void propagate(const val_t* weights, const uint8_t* delays, val_t* history,
                          val_t* ring, const val_t reflectivity, int n, int rows,
                          int firstTime, int firstRow, int source, int timeOff, int minDelay) {
    __shared__ val_t contributions[PROPAGATION_THREADS];
    int row = blockIdx.x * (PROPAGATION_THREADS / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    if (row >= rows) return;

    int t = firstTime + blockIdx.y;
    int lane = threadIdx.x % WARP_SIZE;
    const size_t rowOffset = static_cast<size_t>(row) * n;
    val_t sum = ZERO;

    // Read each group coalesced and add in the original emitter order.
    if (t >= minDelay) {
        for (int first = 0; first < n; first += WARP_SIZE) {
            int j = first + lane;
            val_t value = ZERO;
            if (j < n) {
                int delay = delays[rowOffset + j];
                if (delay && t >= delay) {
                    val_t rad = ring[((t - delay) & (HISTORY_SIZE - 1)) * n + j];
                    if (!(rad <= ZERO)) value = weights[rowOffset + j] * rad;
                }
            }
            if (__any_sync(0xffffffffu, value != ZERO)) {
                contributions[threadIdx.x] = value;
                __syncwarp();
                if (lane == 0) {
                    for (int k = 0; k < WARP_SIZE; ++k) sum += contributions[threadIdx.x + k];
                }
                __syncwarp();
            }
        }
    }
    if (lane == 0) {
        int i = firstRow + row;
        val_t emission = i == source && t < timeOff ? ONE : ZERO;
        val_t value = fmaf(reflectivity, sum, emission);
        history[static_cast<size_t>(t) * rows + row] = value;
        if constexpr (SingleRank) ring[(t & (HISTORY_SIZE - 1)) * n + i] = value;
    }
}

__global__ void unpackHistory(const val_t* packed, val_t* ring, int n, int ranks,
                              int firstTime, int count) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int wideRanks = n % ranks;
    int width = n / ranks;
    int pivot = wideRanks * (width + 1);
    int offset;
    if (j < pivot) {
        ++width;
        offset = (j / width) * width;
    } else {
        offset = pivot + ((j - pivot) / width) * width;
    }
    int t = blockIdx.y;
    ring[((firstTime + t) & (HISTORY_SIZE - 1)) * n + j] =
        packed[count * offset + t * width + j - offset];
}

void runSimulation(Simulation& sim) {
    const Partition& part = sim.partition;
    if (part.rank == 0) printf("Running wave propagation simulation...\n");
    if (!sim.timesteps) return;

    int rows = part.size();
    if (rows) CUDA(cudaMemsetAsync(sim.ring.data, 0, HISTORY_SIZE * sim.n * sizeof(val_t)));
    bool distributed = part.ranks() > 1;
    // An active edge always connects distinct triangle centers. No result in
    // this batch can depend on another result produced in the same batch.
    int batch = std::min(sim.minDelay, sim.timesteps);
    Buffer<val_t, true> hostBatch(distributed ? batch * sim.n : 0);
    Buffer<val_t> deviceBatch(distributed && rows ? batch * sim.n : 0);
    std::vector<int> counts(part.ranks()), offsets(part.ranks());
    for (int rank = 0; rank < part.ranks(); ++rank) {
        counts[rank] = batch * part.widths[rank];
        offsets[rank] = batch * part.offsets[rank];
    }
    for (int t = 0; t < sim.timesteps;) {
        int steps = std::min(batch, sim.timesteps - t);
        if (rows) {
            dim3 grid((rows + PROPAGATION_THREADS / WARP_SIZE - 1) /
                       (PROPAGATION_THREADS / WARP_SIZE), steps);
            if (distributed) {
                propagate<false><<<grid, PROPAGATION_THREADS>>>
                    (sim.weights.data, sim.delays.data, sim.localHistory.data, sim.ring.data,
                     sim.reflectivity, sim.n, rows, t, part.begin(), sim.source,
                     sim.timesteps / 2, sim.minDelay);
            } else {
                propagate<true><<<grid, PROPAGATION_THREADS>>>
                    (sim.weights.data, sim.delays.data, sim.localHistory.data, sim.ring.data,
                     sim.reflectivity, sim.n, rows, t, part.begin(), sim.source,
                     sim.timesteps / 2, sim.minDelay);
            }
            CUDA(cudaGetLastError());
        }
        if (distributed && steps < sim.timesteps - t) {
            if (rows) {
                CUDA(cudaMemcpyAsync(hostBatch.data + steps * part.begin(),
                                     sim.localHistory.data + static_cast<size_t>(t) * rows,
                                     steps * rows * sizeof(val_t), cudaMemcpyDeviceToHost));
                CUDA(cudaStreamSynchronize(nullptr));
            }
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, hostBatch.data, counts.data(),
                           offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            if (rows) {
                CUDA(cudaMemcpyAsync(deviceBatch.data, hostBatch.data, steps * sim.n * sizeof(val_t),
                                     cudaMemcpyHostToDevice));
                dim3 grid((sim.n + 127) / 128, steps);
                unpackHistory<<<grid, 128>>>(deviceBatch.data, sim.ring.data, sim.n,
                                            part.ranks(), t, steps);
                CUDA(cudaGetLastError());
            }
        }
        if (part.rank == 0) {
            for (int step = 1; step <= steps; ++step) {
                if ((t + step) % 10 == 0 || t + step == sim.timesteps)
                    printf("  Timestep %d/%d\n", t + step, sim.timesteps);
            }
        }
        t += steps;
    }
    if (rows) CUDA(cudaStreamSynchronize(nullptr));
}

constexpr int LAG_LANES = 8;

struct CorrelationPeak {
    val_t value;
    int lag;

    __device__ void consider(CorrelationPeak other) {
        if (other.value > value || (other.value == value && other.lag < lag)) *this = other;
    }
};

__global__ void correlate(const val_t* history, const val_t* source, val_t* distances,
                          CorrelationPeak* partial, int rows, int timesteps, unsigned* received) {
    __shared__ CorrelationPeak peaks[LAG_LANES][WARP_SIZE];

    int row = blockIdx.x * WARP_SIZE + threadIdx.x;
    int lane = threadIdx.y;
    int lagStride = LAG_LANES * gridDim.y;
    CorrelationPeak best{ZERO, 0};
    if (row < rows) {
        for (int lag = blockIdx.y * LAG_LANES + lane; lag < timesteps;) {
            val_t sum = ZERO;
            for (int t = lag; t < timesteps; ++t) {
                sum += source[t - lag] * history[static_cast<size_t>(t) * rows + row];
            }
            best.consider({sum, lag});
            if (timesteps - lag <= lagStride) break;
            lag += lagStride;
        }
    }
    peaks[lane][threadIdx.x] = best;
    __syncthreads();
    if (lane == 0 && row < rows) {
        for (int k = 1; k < LAG_LANES; ++k) best.consider(peaks[k][threadIdx.x]);
        if (partial) partial[blockIdx.y * rows + row] = best;
        else distances[row] = WAVE_SPEED * static_cast<val_t>(best.lag);

        if (received && blockIdx.y == 0) {
            for (int t = 0; t < timesteps; ++t) {
                if (history[static_cast<size_t>(t) * rows + row] > EPSILON) {
                    atomicAdd(received, 1u);
                    break;
                }
            }
        }
    }
}

__global__ void finishCorrelation(const CorrelationPeak* partial, val_t* distances,
                                  int rows, int groups) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    CorrelationPeak best{ZERO, 0};
    for (int group = 0; group < groups; ++group) best.consider(partial[group * rows + row]);
    distances[row] = WAVE_SPEED * static_cast<val_t>(best.lag);
}

void computeDistances(Simulation& sim, bool validate) {
    const Partition& part = sim.partition;
    if (part.rank == 0) printf("Computing distances via cross-correlation...\n");
    int rows = part.size();
    std::vector<val_t> localDistances(rows);
    unsigned long long localReceived = 0;

    if (sim.timesteps) {
        int sourceRank = std::upper_bound(part.offsets.begin(), part.offsets.end(), sim.source) -
                         part.offsets.begin() - 1;
        Buffer<val_t, true> source(sim.timesteps);
        Buffer<val_t> deviceSource(rows ? sim.timesteps : 0);
        if (part.rank == sourceRank) {
            const val_t* origin = sim.localHistory.data + (sim.source - part.begin());
            CUDA(cudaMemcpy2D(source.data, sizeof(val_t), origin, rows * sizeof(val_t),
                              sizeof(val_t), sim.timesteps, cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(source.data, sim.timesteps, MPI_FLOAT, sourceRank, MPI_COMM_WORLD);
        if (rows) {
            int rowBlocks = (rows + WARP_SIZE - 1) / WARP_SIZE;
            int groups = 1;
            // Long signals benefit from distributing lags across blocks,
            // especially when this rank owns only a small set of receivers.
            if (sim.timesteps >= 256) {
                int device, multiprocessors;
                CUDA(cudaGetDevice(&device));
                CUDA(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
                groups = std::min(sim.timesteps / LAG_LANES,
                                  (2 * multiprocessors + rowBlocks - 1) / rowBlocks);
            }
            Buffer<val_t> deviceDistances(rows);
            Buffer<CorrelationPeak> partial(groups > 1 ? static_cast<size_t>(groups) * rows : 0);
            Buffer<unsigned> deviceReceived(validate ? 1 : 0);
            unsigned received = 0;
            upload(deviceSource.data, source.data, sim.timesteps);
            upload(deviceReceived.data, &received, validate ? 1 : 0);
            correlate<<<dim3(rowBlocks, groups), dim3(WARP_SIZE, LAG_LANES)>>>
                (sim.localHistory.data, deviceSource.data, deviceDistances.data, partial.data,
                 rows, sim.timesteps, deviceReceived.data);
            CUDA(cudaGetLastError());
            if (groups > 1) {
                finishCorrelation<<<(rows + 127) / 128, 128>>>(partial.data, deviceDistances.data,
                                                              rows, groups);
                CUDA(cudaGetLastError());
            }
            CUDA(cudaMemcpy(localDistances.data(), deviceDistances.data, rows * sizeof(val_t),
                            cudaMemcpyDeviceToHost));
            if (validate) {
                CUDA(cudaMemcpy(&received, deviceReceived.data, sizeof(received), cudaMemcpyDeviceToHost));
                localReceived = received;
            }
        }
    }

    if (part.rank == 0) sim.distances.resize(sim.n);
    MPI_Gatherv(localDistances.data(), rows, MPI_FLOAT, sim.distances.data(), part.widths.data(),
                part.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (validate) {
        MPI_Reduce(&localReceived, &sim.receivingEnergy, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    }
}

bool validateResults(const Simulation& sim) {
    printf("\nValidation:\n");
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    bool allNonNegative = true;
    for (int i = 0; i < sim.n; ++i) {
        val_t d = sim.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %d: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %d: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) ++nonZeroCount;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(sim.n));
    printf("  Non-zero distances: %d/%d\n", nonZeroCount, sim.n);
    if (sim.distances[sim.source] > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", sim.distances[sim.source]);
    }
    printf("  Triangles receiving energy: %llu/%d\n", sim.receivingEnergy, sim.n);
    if (!sim.receivingEnergy) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }
    size_t pairs = static_cast<size_t>(sim.n) * sim.n;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", sim.nonzeroFactors, pairs,
           100.0f * sim.nonzeroFactors / static_cast<val_t>(pairs));
    if (!sim.nonzeroFactors) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }
    if (!allNonNegative) return false;
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (timesteps < 0) {
        if (rank == 0) fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
    }

    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, localRanks;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_size(node, &localRanks);
    MPI_Comm_free(&node);
    if (!std::getenv("OMP_NUM_THREADS")) {
        // Bound the default team on nodes that expose every CPU to every rank.
        int available = omp_get_num_procs();
        if (available > 8) available = std::max(1, available / localRanks);
        omp_set_num_threads(std::min(8, available));
    }
    int devices;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: no CUDA devices visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    Simulation sim(mesh.triangles.size(), timesteps, static_cast<size_t>(sourceIdx),
                   reflectivity, rank, ranks);
    if (rank == 0) {
        printf("Generated icosphere mesh with %d triangles\n", sim.n);
        printf("Building octree...\n\n");
    }
    Octree octree;
    if (sim.partition.size()) octree.build(mesh.triangles);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rank == 0) {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");
    }
    computeFormFactors(sim, mesh.triangles, octree, validate);
    long preDuration = static_cast<long>((MPI_Wtime() - start) * 1000);
    if (rank == 0) {
        printf("  Progress: %d/%d triangles\n", sim.n, sim.n);
        printf("Precomputation time: %ld ms\n\n", preDuration);
    }

    start = MPI_Wtime();
    runSimulation(sim);
    MPI_Barrier(MPI_COMM_WORLD);
    long simDuration = static_cast<long>((MPI_Wtime() - start) * 1000);
    if (rank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    start = MPI_Wtime();
    computeDistances(sim, validate);
    long distDuration = static_cast<long>((MPI_Wtime() - start) * 1000);
    int valid = 1;
    if (rank == 0) {
        printf("Distance computation time: %ld ms\n\n", distDuration);
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        size_t n = sim.n;
        size_t t = sim.timesteps;
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Persistent simulation buffers summed over the ranks.
        double memory = static_cast<double>(n) * n * (sizeof(val_t) + sizeof(uint8_t)) +
                        (static_cast<double>(t) +
                         static_cast<double>(HISTORY_SIZE) * std::min(ranks, sim.n)) *
                            n * sizeof(val_t);
        printf("  Memory usage: %.2f MB\n", memory / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", computeHash(sim.distances));
        if (printResults) {
            std::vector<double> distData(sim.distances.begin(), sim.distances.end());
            print_results(distData, "Distances");
        }
        if (validate) valid = validateResults(sim);
    }
    if (validate) MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
