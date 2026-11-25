/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI, OpenMP, and CUDA implementation of room impulse response
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
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
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

struct RayBox {
    Vec3 center, halfExtent;
    idx_t skip, first, count;
};

int hostThreads(size_t tasks) {
    return static_cast<int>(std::min(tasks, static_cast<size_t>(omp_get_max_threads())));
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

        buildNode(allIndices, minBound, maxBound, true);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin,
                   const Vec3& nodeMax, bool parallel = false) {
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

        auto partitionChild = [&](int i) {
            const Vec3 childCenter = center + Vec3(
                (i & 1) ? childHalfSize.x : -childHalfSize.x,
                (i & 2) ? childHalfSize.y : -childHalfSize.y,
                (i & 4) ? childHalfSize.z : -childHalfSize.z);
            for (size_t idx : indices) {
                if (triangleBoxOverlap(childCenter, childHalfSize, (*allTriangles)[idx])) {
                    childIndices[i].push_back(idx);
                }
            }
        };
        if (parallel) {
            #pragma omp parallel for schedule(static) num_threads(hostThreads(8))
            for (int i = 0; i < 8; ++i) partitionChild(i);
        } else {
            for (int i = 0; i < 8; ++i) partitionChild(i);
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

        // Each subtree owns its output and reads the same immutable mesh.
        auto buildChild = [&](int i) {
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
        };
        if (parallel) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(hostThreads(8))
            for (int i = 0; i < 8; ++i) buildChild(i);
        } else {
            for (int i = 0; i < 8; ++i) buildChild(i);
        }
    }

public:
    // Preorder layout lets a device thread skip a whole subtree without a stack.
    void flatten(std::vector<RayBox>& boxes, std::vector<idx_t>& entries) const {
        const size_t index = boxes.size();
        boxes.push_back({center, halfExtent, 0, static_cast<idx_t>(entries.size()),
                         static_cast<idx_t>(triangleIndices.size())});
        for (size_t triangle : triangleIndices) entries.push_back(static_cast<idx_t>(triangle));
        for (const auto& child : children) {
            if (child) child->flatten(boxes, entries);
        }
        boxes[index].skip = static_cast<idx_t>(boxes.size());
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
            newFaces.reserve(4 * faces.size());
            std::unordered_map<uint64_t, idx_t> midpointCache;
            midpointCache.reserve(faces.size() * 3 / 2);

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                const uint64_t key = (static_cast<uint64_t>(std::min(i1, i2)) << 32) | std::max(i1, i2);
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache.emplace(key, idx);
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


// One tile covers consecutive emitters in one receiver row. The mask both
// excludes culled pairs and indexes into the original row-major random stream.
constexpr int PAIRS_PER_TILE = 64;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr int MT_WORDS = 624;
constexpr int MT_SHIFT = 397;
constexpr int MT_STEP = MT_WORDS - MT_SHIFT;
constexpr int MT_PERIOD_BITS = 19937;
constexpr int TILE_THREADS = 256;
// The longest possible delay is 40 timesteps for the radius-10 room.
constexpr int HISTORY_SLOTS = 64;

__host__ __device__ uint32_t twistWord(uint32_t first, uint32_t next, uint32_t middle) {
    const uint32_t bits = (first & 0x80000000u) | (next & 0x7fffffffu);
    return middle ^ (bits >> 1) ^ ((bits & 1) ? 0x9908b0dfu : 0);
}

struct RandomState {
    uint32_t words[MT_WORDS];
    int position;
};

// x^offset modulo the characteristic polynomial of the MT19937 transition.
// These are the nonzero exponents below the leading term x^19937.
auto jumpPolynomial(uint64_t offset) {
    constexpr int degree = MT_PERIOD_BITS;
    constexpr int terms[] = {
        0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681,
        3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752,
        7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147, 11157,
        11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717, 11838, 11881, 11944, 11997,
        12277, 12335, 12393, 12504, 12509, 12620, 12673, 12731, 12736, 12789, 12905, 12958,
        12963, 13137, 13185, 13190, 13243, 13301, 13412, 13528, 13533, 13639, 13697, 13760,
        13813, 13866, 14093, 14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721,
        14774, 14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513,
        15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421,
        16537, 16590, 16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445,
        17498, 17725, 17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860,
        19087, 19314
    };

    std::array<uint64_t, (degree + 63) / 64> polynomial{};
    polynomial[0] = 1;
    if (offset == 0) return polynomial;
    for (int bit = 63 - __builtin_clzll(offset); bit >= 0; --bit) {
        std::array<uint64_t, polynomial.size() * 2> squared{};
        const int shift = static_cast<int>((offset >> bit) & 1);
        for (size_t word = 0; word < polynomial.size(); ++word) {
            for (uint64_t bits = polynomial[word]; bits != 0; bits &= bits - 1) {
                const int power = 2 * (64 * word + __builtin_ctzll(bits)) + shift;
                squared[power / 64] |= uint64_t{1} << (power % 64);
            }
        }
        for (int power = 2 * degree - 1; power >= degree; --power) {
            const uint64_t mask = uint64_t{1} << (power % 64);
            if (!(squared[power / 64] & mask)) continue;
            squared[power / 64] ^= mask;
            for (int term : terms) {
                const int index = power - degree + term;
                squared[index / 64] ^= uint64_t{1} << (index % 64);
            }
        }
        std::copy_n(squared.begin(), polynomial.size(), polynomial.begin());
    }
    return polynomial;
}

class RandomStream {
public:
    RandomState state{};

    explicit RandomStream(uint64_t offset) {
        state.words[0] = 42;
        for (int i = 1; i < MT_WORDS; ++i) {
            const uint32_t prev = state.words[i - 1];
            state.words[i] = 1812433253u * (prev ^ (prev >> 30)) + i;
        }
        twist();
        // A blockwise discard is faster for short offsets. Large offsets jump
        // independently, so a late MPI rank never replays the entire prefix.
        if (offset < (1u << 22)) advance(offset);
        else jump(offset);
    }

    void advance(uint64_t draws) {
        draws += state.position;
        while (draws >= MT_WORDS) {
            twist();
            draws -= MT_WORDS;
        }
        state.position = static_cast<int>(draws);
    }

private:
    void twist() {
        auto* words = state.words;
        for (int i = 0; i < MT_STEP; ++i) {
            words[i] = twistWord(words[i], words[i + 1], words[i + MT_SHIFT]);
        }
        for (int i = MT_STEP; i < MT_WORDS - 1; ++i) {
            words[i] = twistWord(words[i], words[i + 1], words[i - MT_STEP]);
        }
        words[MT_WORDS - 1] = twistWord(words[MT_WORDS - 1], words[0], words[MT_SHIFT - 1]);
    }

    void jump(uint64_t offset) {
        const auto polynomial = jumpPolynomial(offset);
        RandomState sum{};
        int first = 0;
        for (int power = 0; power < MT_PERIOD_BITS; ++power) {
            if ((polynomial[power / 64] >> (power % 64)) & 1) {
                for (int i = 0; i < MT_WORDS - first; ++i) sum.words[i] ^= state.words[first + i];
                for (int i = 0; i < first; ++i) sum.words[MT_WORDS - first + i] ^= state.words[i];
            }
            const int next = (first + 1) % MT_WORDS;
            state.words[first] = twistWord(state.words[first], state.words[next],
                                           state.words[(first + MT_SHIFT) % MT_WORDS]);
            first = next;
        }
        state = sum;
    }
};

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<typename T, bool pinned = false>
class Buffer {
public:
    T* data = nullptr;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (!data) return;
        if constexpr (pinned) cudaFreeHost(data);
        else cudaFree(data);
    }

    void allocate(size_t count) {
        if (count == 0) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflow");
        }
        if constexpr (pinned) cudaCheck(cudaMallocHost(&data, count * sizeof(T)));
        else cudaCheck(cudaMalloc(&data, count * sizeof(T)));
    }

    void upload(const std::vector<T>& source) {
        allocate(source.size());
        cudaCheck(cudaMemcpy(data, source.data(), source.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
    }
};

class Stream {
public:
    cudaStream_t value = nullptr;
    Stream() { cudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() { cudaStreamDestroy(value); }
};

struct Patch {
    Vec3 origin, edge1, edge2, center, normal;
    val_t area;
};

struct Ray {
    Vec3 delta, center, absDelta;

    __device__ Ray(const Vec3& from, const Vec3& to)
        : delta((to - from) * 0.5f), center(from + delta),
          absDelta(fabsf(delta.x), fabsf(delta.y), fabsf(delta.z)) {}

    __device__ bool intersects(const RayBox& box) const {
        const Vec3 c = center - box.center;
        const Vec3 h = box.halfExtent;
        if (fabsf(c.x) > h.x + absDelta.x ||
            fabsf(c.y) > h.y + absDelta.y ||
            fabsf(c.z) > h.z + absDelta.z) return false;
        return fabsf(delta.y * c.z - delta.z * c.y) <= h.y * absDelta.z + h.z * absDelta.y + EPSILON &&
               fabsf(delta.z * c.x - delta.x * c.z) <= h.z * absDelta.x + h.x * absDelta.z + EPSILON &&
               fabsf(delta.x * c.y - delta.y * c.x) <= h.x * absDelta.y + h.y * absDelta.x + EPSILON;
    }
};

__device__ bool intersectsPatch(const Vec3& origin, const Vec3& direction,
                                val_t length, const Patch& patch) {
    const Vec3 p = direction.cross(patch.edge2);
    const val_t det = patch.edge1.dot(p);
    if (fabsf(det) < EPSILON) return false;
    const val_t invDet = 1.0f / det;
    const Vec3 relative = origin - patch.origin;
    const val_t u = relative.dot(p) * invDet;
    if (u < ZERO || u > ONE) return false;
    const Vec3 q = relative.cross(patch.edge1);
    const val_t v = direction.dot(q) * invDet;
    if (v < ZERO || u + v > ONE) return false;
    const val_t distance = patch.edge2.dot(q) * invDet;
    return distance > EPSILON && distance < length - EPSILON;
}

struct Geometry {
    const Patch* patches;
    const RayBox* boxes;
    const idx_t* entries;

    __device__ bool blocked(const Vec3& from, const Vec3& to, const Vec3& v,
                            val_t length, int receiver, int emitter) const {
        const Ray ray(from, to);
        const Vec3 direction = v / length;
        for (idx_t node = 0; node < boxes[0].skip;) {
            const RayBox& box = boxes[node];
            if (node != 0 && !ray.intersects(box)) {
                node = box.skip;
                continue;
            }
            ++node;
            for (idx_t i = box.first; i < box.first + box.count; ++i) {
                const idx_t other = entries[i];
                if (other != static_cast<idx_t>(receiver) && other != static_cast<idx_t>(emitter) &&
                    intersectsPatch(from, direction, length, patches[other])) return true;
            }
        }
        return false;
    }
};

__device__ val_t uniformFloat(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    return fminf(static_cast<val_t>(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 sample(const Patch& patch, const uint32_t* draws) {
    val_t u = uniformFloat(draws[0]);
    val_t v = uniformFloat(draws[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return patch.origin + patch.edge1 * u + patch.edge2 * v;
}

__device__ val_t sampleFactor(const Geometry& geometry, const Patch& a, const Patch& b,
                               int receiver, int emitter, const uint32_t* draws) {
    const Vec3 from = sample(a, draws);
    const Vec3 to = sample(b, draws + 2);
    const Vec3 v = to - from;
    const val_t distSqr = v.squaredNorm();
    if (distSqr < EPSILON) return ZERO;
    const val_t length = sqrtf(distSqr);
    const val_t cosA = v.dot(a.normal) / length;
    const val_t cosB = (-v).dot(b.normal) / length;
    if (cosA <= ZERO || cosB <= ZERO || geometry.blocked(from, to, v, length, receiver, emitter)) {
        return ZERO;
    }
    return (cosA * cosB) / (PI * distSqr);
}

struct Matrices {
    val_t* weights;
    uint8_t* delays;
    int stride;
};

struct FormTile {
    uint64_t mask;
    RandomState rng;
};

// One half warp per pair, with one lane per sampled ray. Products are reduced
// in original ray order to retain the sequential floating-point accumulation.
__global__ void formFactors(Geometry geometry, const FormTile* tiles, size_t offset,
                            int firstRow, int columns, int tilesPerRow, Matrices matrix,
                            unsigned long long* nonzero) {
    __shared__ uint32_t draws[MT_WORDS + PAIRS_PER_TILE * DRAWS_PER_PAIR];
    const FormTile& tile = tiles[blockIdx.x];
    const int lane = threadIdx.x % NUM_RAYS;
    const int pairInGroup = threadIdx.x / NUM_RAYS;
    const int groupSize = TILE_THREADS / NUM_RAYS;
    const size_t index = offset + blockIdx.x;
    const int row = index / tilesPerRow;
    const int firstColumn = (index % tilesPerRow) * PAIRS_PER_TILE;
    const int receiver = firstRow + row;

    const uint64_t mask = tile.mask;
    if (mask != 0) {
        for (int i = threadIdx.x; i < MT_WORDS; i += blockDim.x) draws[i] = tile.rng.words[i];
        __syncthreads();
        const int needed = tile.rng.position + __popcll(mask) * DRAWS_PER_PAIR;
        for (int base = MT_WORDS; base < needed; base += MT_STEP) {
            const int i = base + threadIdx.x;
            if (threadIdx.x < MT_STEP && i < needed) {
                draws[i] = twistWord(draws[i - MT_WORDS], draws[i - MT_WORDS + 1],
                                     draws[i - MT_STEP]);
            }
            __syncthreads();
        }
    }

    unsigned int count = 0;
    for (int pair = pairInGroup; pair < PAIRS_PER_TILE; pair += groupSize) {
        const int emitter = firstColumn + pair;
        if (emitter >= columns) continue;
        const size_t element = static_cast<size_t>(row) * matrix.stride + emitter;
        val_t kij = ZERO;
        if ((mask >> pair) & 1) {
            const int activeIndex = __popcll(mask & ((uint64_t{1} << pair) - 1));
            const uint32_t* random = draws + tile.rng.position + activeIndex * DRAWS_PER_PAIR;
            const val_t contribution = sampleFactor(geometry, geometry.patches[receiver],
                geometry.patches[emitter], receiver, emitter, random + 4 * lane);

            const unsigned int halfWarp = threadIdx.x & 16 ? 0xffff0000u : 0x0000ffffu;
            #pragma unroll
            for (int ray = 0; ray < NUM_RAYS; ++ray) {
                const val_t contributionInOrder = __shfl_sync(halfWarp, contribution, ray, NUM_RAYS);
                if (lane == 0) kij += contributionInOrder;
            }
            kij *= INV_NUM_RAYS;
        }

        if (lane == 0) {
            count += kij > EPSILON;
            matrix.weights[element] = fminf(kij * geometry.patches[emitter].area, ONE);
            const Vec3 separation = geometry.patches[receiver].center - geometry.patches[emitter].center;
            // All centroids are within radius 10 and delays are at most 40.
            matrix.delays[element] = static_cast<uint8_t>(__float2int_ru(separation.norm() * INV_WAVE_SPEED));
        }
    }
    if (nonzero) {
        for (int delta = 16; delta > 0; delta /= 2) count += __shfl_down_sync(0xffffffff, count, delta);
        if (threadIdx.x % 32 == 0 && count) atomicAdd(nonzero, count);
    }
}

constexpr int PROPAGATION_THREADS = 128;

// A warp loads consecutive emitters for a receiver, then accumulates their
// products in emitter order to preserve the original floating-point sum.
__global__ void propagateSteps(Matrices matrix, const val_t* history, val_t* output,
                               int outputStride, int start, int columns, int rows,
                               int historyMask, int firstRow, int source, int timeOff,
                               val_t reflectivity) {
    const int lane = threadIdx.x % warpSize;
    const int row = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;
    if (row >= rows) return;
    const int time = start + blockIdx.y;
    const size_t base = static_cast<size_t>(row) * matrix.stride;
    __shared__ val_t terms[PROPAGATION_THREADS];
    val_t* const warpTerms = terms + threadIdx.x - lane;
    val_t sum = ZERO;
    for (int startColumn = 0; startColumn < columns; startColumn += warpSize) {
        const int column = startColumn + lane;
        val_t term = ZERO;
        if (column < columns) {
            const size_t element = base + column;
            const int delay = matrix.delays[element];
            const val_t weight = matrix.weights[element];
            if (time >= delay && weight > ZERO) {
                const int past = (time - delay) & historyMask;
                const val_t incoming = history[static_cast<size_t>(past) * columns + column];
                if (!(incoming <= ZERO)) term = weight * incoming;
            }
        }
        warpTerms[lane] = term;
        __syncwarp();
        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) sum += warpTerms[i];
        }
        __syncwarp();
    }
    if (lane == 0) {
        const val_t emission = time < timeOff && firstRow + row == source ? ONE : ZERO;
        output[static_cast<size_t>(blockIdx.y) * outputStride + row] = fmaf(reflectivity, sum, emission);
    }
}

// The MPI receive buffer is rank-major so each rank can send one contiguous
// range. Transpose its timestep slices into the time-major device history.
__global__ void unpackHistory(const val_t* gathered, val_t* history, int columns,
                              int ranks, int batch, int start) {
    const int column = blockDim.x * blockIdx.x + threadIdx.x;
    if (column >= columns) return;
    const int small = columns / ranks;
    const int extra = columns % ranks;
    const int split = extra * (small + 1);
    const int owner = column < split ? column / (small + 1) : extra + (column - split) / small;
    const int first = owner * small + min(owner, extra);
    const int count = small + (owner < extra);
    const int slot = (start + blockIdx.y) & (HISTORY_SLOTS - 1);
    history[static_cast<size_t>(slot) * columns + column] =
        gathered[static_cast<size_t>(batch - 1) * first + column + blockIdx.y * count];
}

constexpr int CORRELATION_THREADS = 128;

__global__ void correlateRows(const val_t* history, const val_t* sourceHistory,
                              val_t* distances, int timesteps, int stride,
                              int sourceStride, unsigned long long* received) {
    __shared__ val_t scores[CORRELATION_THREADS];
    __shared__ int lags[CORRELATION_THREADS];
    const int row = blockIdx.x;
    val_t best = ZERO;
    int bestLag = 0;
    for (int lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (int t = lag; t < timesteps; ++t) {
            sum += history[static_cast<size_t>(t) * stride + row] *
                   sourceHistory[static_cast<size_t>(t - lag) * sourceStride];
        }
        if (sum > best) {
            best = sum;
            bestLag = lag;
        }
    }
    scores[threadIdx.x] = best;
    lags[threadIdx.x] = bestLag;
    __syncthreads();
    for (int delta = blockDim.x / 2; delta > 0; delta /= 2) {
        if (threadIdx.x < delta) {
            const int other = threadIdx.x + delta;
            if (scores[other] > scores[threadIdx.x] ||
                (scores[other] == scores[threadIdx.x] && lags[other] < lags[threadIdx.x])) {
                scores[threadIdx.x] = scores[other];
                lags[threadIdx.x] = lags[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[blockIdx.x] = WAVE_SPEED * lags[0];
        if (received) {
            for (int t = 0; t < timesteps; ++t) {
                if (history[static_cast<size_t>(t) * stride + row] > EPSILON) {
                    atomicAdd(received, 1ULL);
                    break;
                }
            }
        }
    }
}


void selectDevice(bool active) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    int localRank;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    if (!active) return;
    int devices;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("No visible CUDA devices");
    cudaCheck(cudaSetDevice(localRank % devices));
}

long phaseTime(double start) {
    const double elapsed = MPI_Wtime() - start;
    double maximum = 0;
    MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return static_cast<long>(maximum * 1000);
}

struct Results {
    std::vector<val_t> distances;
    unsigned long long nonzero[2]{}; // Form factors and patches receiving energy.
};

class Simulation {
public:
    int triangles = 0;
    int source = 0;

    Simulation(int subdivisions, int timesteps, size_t sourceIndex, int rank, int ranks, bool root)
        : timesteps(timesteps), counts(ranks), offsets(ranks) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        triangles = static_cast<int>(mesh.triangles.size());
        source = static_cast<int>(sourceIndex % triangles);

        for (int i = 0; i < ranks; ++i) {
            counts[i] = triangles / ranks + (i < triangles % ranks);
            offsets[i] = i * (triangles / ranks) + std::min(i, triangles % ranks);
        }
        first = offsets[rank];
        rows = counts[rank];
        selectDevice(rows != 0);
        if (root) {
            printf("Generated icosphere mesh with %d triangles\n", triangles);
            printf("Building octree...\n");
        }

        if (rows != 0) {
            Octree octree;
            octree.build(mesh.triangles);
            std::vector<RayBox> flatBoxes;
            std::vector<idx_t> flatEntries;
            octree.flatten(flatBoxes, flatEntries);
            boxes.upload(flatBoxes);
            entries.upload(flatEntries);

            patches.resize(triangles);
            #pragma omp parallel for schedule(static) num_threads(hostThreads((triangles + 511) / 512))
            for (int i = 0; i < triangles; ++i) {
                const Triangle& tri = mesh.triangles[i];
                patches[i] = {tri.a, tri.b - tri.a, tri.c - tri.a, tri.center(),
                              tri.normal(), tri.area()};
            }
            devicePatches.upload(patches);
        }
        const size_t elements = static_cast<size_t>(rows) * matrixStride();
        weights.allocate(elements);
        delays.allocate(elements);
        if (rows != 0) {
            const int slots = ranks > 1 ? std::min(timesteps, HISTORY_SLOTS) : timesteps;
            history.allocate(static_cast<size_t>(slots) * triangles);
            if (ranks > 1) localHistory.allocate(static_cast<size_t>(timesteps) * rows);
        }
    }

    void precompute(bool validate) {
        const int tilesPerRow = (triangles + PAIRS_PER_TILE - 1) / PAIRS_PER_TILE;
        const size_t tileCount = static_cast<size_t>(rows) * tilesPerRow;
        std::vector<uint64_t> masks(tileCount);
        unsigned long long localDraws = 0;
        const int threads = hostThreads(std::max<size_t>(1, (tileCount + 1023) / 1024));
        #pragma omp parallel for schedule(static) num_threads(threads) reduction(+: localDraws)
        for (size_t tile = 0; tile < tileCount; ++tile) {
            const int receiver = first + tile / tilesPerRow;
            const int start = (tile % tilesPerRow) * PAIRS_PER_TILE;
            const int end = std::min(start + PAIRS_PER_TILE, triangles);
            const Vec3& normal = patches[receiver].normal;
            uint64_t mask = 0;
            for (int j = start; j < end; ++j) {
                if (j != receiver && normal.dot(patches[j].normal) <= 0.99f) {
                    mask |= uint64_t{1} << (j - start);
                }
            }
            masks[tile] = mask;
            localDraws += __builtin_popcountll(mask) * DRAWS_PER_PAIR;
        }

        unsigned long long firstDraw = 0;
        if (counts.size() > 1) {
            MPI_Exscan(&localDraws, &firstDraw, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (first == 0) firstDraw = 0; // MPI leaves the first rank's result undefined.
        }
        if (tileCount == 0) return;

        if (validate) {
            nonzeroFactors.allocate(1);
            cudaCheck(cudaMemset(nonzeroFactors.data, 0, sizeof(unsigned long long)));
            cudaCheck(cudaDeviceSynchronize());
        }

        // Overlap CPU checkpoint generation, PCIe transfers and ray processing.
        constexpr size_t batchSize = 2048;
        struct Batch {
            Buffer<FormTile, true> host;
            Buffer<FormTile> device;
            Stream stream;
        };
        Batch batches[2];
        for (auto& batch : batches) {
            batch.host.allocate(std::min(batchSize, tileCount));
            batch.device.allocate(std::min(batchSize, tileCount));
        }

        const Geometry geometry{devicePatches.data, boxes.data, entries.data};
        const Matrices matrix{weights.data, delays.data, matrixStride()};
        RandomStream rng(firstDraw);
        for (size_t start = 0, batch = 0; start < tileCount; start += batchSize, batch ^= 1) {
            Batch& slot = batches[batch];
            cudaCheck(cudaStreamSynchronize(slot.stream.value));
            const size_t count = std::min(batchSize, tileCount - start);
            for (size_t i = 0; i < count; ++i) {
                const uint64_t mask = masks[start + i];
                slot.host.data[i].mask = mask;
                slot.host.data[i].rng = rng.state;
                rng.advance(__builtin_popcountll(mask) * DRAWS_PER_PAIR);
            }
            cudaCheck(cudaMemcpyAsync(slot.device.data, slot.host.data, count * sizeof(FormTile),
                                      cudaMemcpyHostToDevice, slot.stream.value));
            formFactors<<<count, TILE_THREADS, 0, slot.stream.value>>>(
                geometry, slot.device.data, start, first, triangles, tilesPerRow, matrix,
                nonzeroFactors.data);
            cudaCheck(cudaGetLastError());
        }
        for (auto& batch : batches) cudaCheck(cudaStreamSynchronize(batch.stream.value));
    }

    void propagate(val_t reflectivity, bool root) {
        if (timesteps == 0) return;
        const bool distributed = counts.size() > 1;
        const int width = exchangeWidth = independentSteps();
        Buffer<val_t, true> timestep;
        Buffer<val_t> gathered;
        std::vector<val_t> idleTimestep, sourceTrace;
        std::vector<int> packedCounts(counts.size()), packedOffsets(offsets.size());
        int sourceOwner = 0;
        if (distributed) {
            const size_t exchangeSize = static_cast<size_t>(width) * triangles;
            if (rows != 0) {
                timestep.allocate(exchangeSize);
                if (width > 1) gathered.allocate(exchangeSize);
                sourceTrace.resize(timesteps);
                sourceOwner = std::upper_bound(offsets.begin(), offsets.end(), source) - offsets.begin() - 1;
            } else {
                idleTimestep.resize(exchangeSize);
            }
        }
        val_t* const hostExchange = rows != 0 ? timestep.data : idleTimestep.data();
        const Matrices matrix{weights.data, delays.data, matrixStride()};
        constexpr int rowsPerBlock = PROPAGATION_THREADS / 32;
        int previousBatch = 0;
        const int historyMask = distributed ? HISTORY_SLOTS - 1 : -1;
        for (int t = 0; t < timesteps;) {
            const int batch = std::min(width, timesteps - t);
            val_t* historyRow = rows ? history.data + static_cast<size_t>(t & historyMask) * triangles : nullptr;
            val_t* destination = distributed && rows != 0
                ? localHistory.data + static_cast<size_t>(t) * rows : historyRow;
            if (rows != 0) {
                const dim3 grid((rows + rowsPerBlock - 1) / rowsPerBlock, batch);
                propagateSteps<<<grid, PROPAGATION_THREADS>>>(
                    matrix, history.data, destination,
                    distributed ? rows : triangles, t, triangles, rows, historyMask, first, source,
                    timesteps / 2, reflectivity);
                cudaCheck(cudaGetLastError());
            }
            if (distributed) {
                if (batch != previousBatch) {
                    for (size_t i = 0; i < counts.size(); ++i) {
                        packedCounts[i] = batch * counts[i];
                        packedOffsets[i] = batch * offsets[i];
                    }
                    previousBatch = batch;
                }
                if (rows != 0) {
                    cudaCheck(cudaMemcpy(hostExchange + batch * first, destination,
                                         static_cast<size_t>(batch) * rows * sizeof(val_t),
                                         cudaMemcpyDeviceToHost));
                }
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, hostExchange, packedCounts.data(),
                               packedOffsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
                if (rows != 0) {
                    const int sourceOffset = (batch - 1) * offsets[sourceOwner] + source;
                    for (int step = 0; step < batch; ++step) {
                        sourceTrace[t + step] = hostExchange[sourceOffset + step * counts[sourceOwner]];
                    }
                    cudaCheck(cudaMemcpy(batch == 1 ? historyRow : gathered.data, hostExchange,
                                         static_cast<size_t>(batch) * triangles * sizeof(val_t),
                                         cudaMemcpyHostToDevice));
                    if (batch > 1) {
                        const dim3 grid((triangles + 127) / 128, batch);
                        unpackHistory<<<grid, 128>>>(gathered.data, history.data, triangles,
                                                     static_cast<int>(counts.size()), batch, t);
                        cudaCheck(cudaGetLastError());
                    }
                }
            }
            if (root) {
                for (int step = 0; step < batch; ++step) {
                    const int completed = t + step + 1;
                    if (completed % 10 == 0 || completed == timesteps) {
                        printf("  Timestep %d/%d\n", completed, timesteps);
                    }
                }
            }
            t += batch;
        }
        if (rows != 0) cudaCheck(cudaDeviceSynchronize());
        if (distributed && rows != 0) sourceHistory.upload(sourceTrace);
    }

    Results correlate(bool root, bool validate) const {
        std::vector<val_t> local(rows);
        unsigned long long localNonzero[2]{};
        if (rows != 0) {
            const bool distributed = counts.size() > 1;
            const val_t* radiosity = distributed ? localHistory.data : history.data;
            const val_t* sourceTrace = distributed ? sourceHistory.data : (timesteps ? history.data + source : nullptr);
            Buffer<val_t> output;
            output.allocate(rows);
            Buffer<unsigned long long> received;
            if (validate) {
                received.allocate(1);
                cudaCheck(cudaMemset(received.data, 0, sizeof(unsigned long long)));
            }
            correlateRows<<<rows, CORRELATION_THREADS>>>(
                radiosity, sourceTrace, output.data, timesteps, rows, distributed ? 1 : triangles, received.data);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(local.data(), output.data, rows * sizeof(val_t), cudaMemcpyDeviceToHost));
            if (validate) {
                cudaCheck(cudaMemcpy(&localNonzero[0], nonzeroFactors.data,
                                     sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                cudaCheck(cudaMemcpy(&localNonzero[1], received.data,
                                     sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            }
        }
        Results result;
        if (root) result.distances.resize(triangles);
        MPI_Gatherv(local.data(), rows, MPI_FLOAT, result.distances.data(), counts.data(),
                    offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        if (validate) {
            MPI_Reduce(localNonzero, result.nonzero, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                       0, MPI_COMM_WORLD);
        }
        return result;
    }

    size_t matrixMemory() const {
        size_t bytes = 0;
        const bool distributed = counts.size() > 1;
        for (int count : counts) {
            if (distributed && timesteps != 0) bytes += exchangeWidth * triangles * sizeof(val_t);
            if (count == 0) continue;
            bytes += static_cast<size_t>(count) * matrixStride() * (sizeof(val_t) + sizeof(uint8_t));
            if (distributed) {
                bytes += static_cast<size_t>(std::min(timesteps, HISTORY_SLOTS)) * triangles * sizeof(val_t);
                bytes += static_cast<size_t>(timesteps) * (count + 1) * sizeof(val_t);
                if (exchangeWidth > 1) bytes += exchangeWidth * triangles * sizeof(val_t);
            } else {
                bytes += static_cast<size_t>(timesteps) * triangles * sizeof(val_t);
            }
        }
        return bytes;
    }

private:
    int timesteps;
    int exchangeWidth = 1;
    std::vector<int> counts, offsets;
    int first, rows;
    std::vector<Patch> patches;
    Buffer<Patch> devicePatches;
    Buffer<RayBox> boxes;
    Buffer<idx_t> entries;
    Buffer<val_t> weights, history, localHistory, sourceHistory;
    Buffer<uint8_t> delays;
    Buffer<unsigned long long> nonzeroFactors;

    int matrixStride() const { return (triangles + 31) / 32 * 32; }

    int independentSteps() const {
        // A coarse mesh can advance several timesteps before any newly
        // reflected wave reaches another patch. Fine meshes use one step.
        if (triangles > 320 || timesteps < 2) return 1;
        int width = std::min(8, timesteps);
        for (int i = first; i < first + rows; ++i) {
            for (int j = 0; j < triangles; ++j) {
                if (i == j) continue;
                const val_t delay = (patches[i].center - patches[j].center).norm() * INV_WAVE_SPEED;
                // Round down with margin to bound the device's rounded-up delay.
                width = std::min(width, std::max(1, static_cast<int>(delay - EPSILON)));
            }
        }
        if (counts.size() > 1) MPI_Allreduce(MPI_IN_PLACE, &width, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return width;
    }
};

bool validateResults(const Results& results, int source) {
    printf("\nValidation:\n");
    const size_t triangles = results.distances.size();
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;
    for (size_t i = 0; i < triangles; ++i) {
        const val_t d = results.distances[i];
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
        nonZeroCount += d > EPSILON;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(triangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, triangles);
    const val_t srcDist = results.distances[source];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }
    printf("  Triangles receiving energy: %llu/%zu\n", results.nonzero[1], triangles);
    if (results.nonzero[1] == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const size_t pairs = triangles * triangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", results.nonzero[0], pairs,
           100.0f * results.nonzero[0] / static_cast<val_t>(pairs));
    if (results.nonzero[0] == 0) {
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
        memcpy(&bits, &distances[i], sizeof(bits));
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

int benchmark(int argc, char** argv, int rank, int ranks) {
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
        if (rank == 0) fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
    }
    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const bool root = rank == 0;

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    Simulation simulation(subdivisions, timesteps, static_cast<size_t>(sourceIdx),
                          rank, ranks, root);
    if (root) printf("\nComputing time delays (Tau)...\nComputing form factors (Kij)...\n");
    const double startPre = MPI_Wtime();
    simulation.precompute(validate);
    const long preDuration = phaseTime(startPre);
    if (root) printf("Precomputation time: %ld ms\n\n", preDuration);

    if (root) printf("Running wave propagation simulation...\n");
    const double startSim = MPI_Wtime();
    simulation.propagate(reflectivity, root);
    const long simDuration = phaseTime(startSim);
    if (root) printf("Simulation time: %ld ms\n\n", simDuration);

    if (root) printf("Computing distances via cross-correlation...\n");
    const double startDist = MPI_Wtime();
    const Results results = simulation.correlate(root, validate);
    const long distDuration = phaseTime(startDist);
    if (!root) {
        int status;
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return status;
    }
    printf("Distance computation time: %ld ms\n\n", distDuration);

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = simulation.triangles;
    size_t t = timesteps;
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

    printf("  Memory usage: %.2f MB\n", simulation.matrixMemory() / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(results.distances);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(results.distances.begin(), results.distances.end());
        print_results(distData, "Distances");
    }

    int status = validate && !validateResults(results, simulation.source) ? 1 : 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        status = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
