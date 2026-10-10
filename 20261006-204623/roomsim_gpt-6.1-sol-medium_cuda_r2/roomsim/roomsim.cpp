/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallel implementation of room impulse response
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
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

__host__ __device__ int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================


static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    void allocate(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Preorder nodes with escape links allow stackless traversal, preserving the
// original child order and triangle-box membership (including duplicates).
struct FlatNode {
    Vec3 center, halfExtent;
    size_t begin, count, escape;
};

static void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes,
                          std::vector<size_t>& indices) {
    size_t id = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(), tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[id].escape = nodes.size();
}

__device__ bool intersectsBox(const Vec3& p1, const Vec3& p2, const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x ||
        fabsf(c.y) > node.halfExtent.y + ad.y ||
        fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > node.halfExtent.y*ad.z+node.halfExtent.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > node.halfExtent.z*ad.x+node.halfExtent.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > node.halfExtent.x*ad.y+node.halfExtent.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(const Vec3& from, const Vec3& to, size_t i, size_t j,
                       const Triangle* triangles, const FlatNode* nodes,
                       const size_t* indices, size_t nodeCount) {
    Vec3 dir = to - from;
    float length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    for (size_t k = 0; k < nodeCount;) {
        FlatNode node = nodes[k];
        if (k && !intersectsBox(from, to, node)) { k = node.escape; continue; }
        for (size_t q = 0; q < node.count; ++q) {
            size_t index = indices[node.begin + q];
            if (index == i || index == j) continue;
            const Triangle& tri = triangles[index];
            float distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

__device__ Vec3 sampledPoint(const Triangle& tri, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}

// Sixteen lanes cooperate on a pair, one visibility ray per lane. The final
// summation follows ray order instead of a reassociated tree reduction.
__global__ void formFactorKernel(const Triangle* triangles, const FlatNode* nodes,
                                const size_t* indices, size_t nodeCount,
                                const size_t* pairs, const float* random,
                                size_t count, size_t n, float* kij) {
    size_t pair = (size_t(blockIdx.x)*blockDim.x + threadIdx.x) / NUM_RAYS;
    int ray = threadIdx.x % NUM_RAYS;
    if (pair >= count) return;
    size_t index = pairs[pair], i = index / n, j = index % n;
    Triangle ti = triangles[i], tj = triangles[j];
    const float* rnd = random + pair * (4*NUM_RAYS) + ray*4;
    Vec3 pi = sampledPoint(ti, rnd[0], rnd[1]);
    Vec3 pj = sampledPoint(tj, rnd[2], rnd[3]);
    float value = ZERO;
    if (!blocked(pi, pj, i, j, triangles, nodes, indices, nodeCount)) {
        Vec3 v = pj-pi;
        float dist = v.squaredNorm();
        if (dist >= EPSILON) {
            float ci = cosPhi(v, ti.normal()), cj = cosPhi(-v, tj.normal());
            if (ci > ZERO && cj > ZERO) value = (ci*cj)/(PI*dist);
        }
    }
    unsigned mask = __activemask();
    float sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r)
        sum += __shfl_sync(mask, value, r, NUM_RAYS);
    if (ray == 0) kij[i*n+j] = sum*INV_NUM_RAYS;
}

__global__ void delayKernel(const Triangle* triangles, size_t n, int* tau) {
    for (size_t index = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
         index < n*n; index += size_t(gridDim.x)*blockDim.x) {
        size_t i = index/n, j = index%n;
        tau[index] = i == j ? 0 : computeTau(triangles[i], triangles[j]);
    }
}

// A warp owns one receiver. Load contributions in parallel, then accumulate
// them in ascending emitter order to preserve sequential floating-point results.
__global__ void propagationKernel(size_t n, size_t t, size_t timeOff, size_t source,
                                  const float* kij, const int* tau, const float* areas,
                                  const float* rho, float* rad) {
    size_t i = (size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    int lane = threadIdx.x%32;
    if (i >= n) return;
    float sum = ZERO;
    for (size_t base = 0; base < n; base += 32) {
        size_t j = base+lane;
        float contribution = ZERO;
        bool active = false;
        if (j < n && i != j) {
            int delay = tau[i*n+j];
            if (static_cast<int>(t) >= delay) {
                float k = kij[i*n+j];
                if (k > ZERO) {
                    float b = rad[(t-size_t(delay))*n+j];
                    if (!(b <= ZERO)) {
                        contribution = fminf(k*areas[j], ONE)*b;
                        active = true;
                    }
                }
            }
        }
        unsigned activeLanes = __ballot_sync(0xffffffffu, active);
        // Skipping absent contributions also matches the original NaN/zero rules.
        while (activeLanes) {
            int emitter = __ffs(activeLanes)-1;
            float value = __shfl_sync(0xffffffffu, contribution, emitter);
            if (lane == 0) sum += value;
            activeLanes &= activeLanes-1;
        }
    }
    if (lane == 0)
        rad[t*n+i] = rho[i]*sum + (i == source && t < timeOff ? ONE : ZERO);
}

__global__ void transposeRadKernel(const float* rad, float* transposed, size_t n, size_t t) {
    __shared__ float tile[32][33];
    size_t x = size_t(blockIdx.x)*32+threadIdx.x;
    size_t y = size_t(blockIdx.y)*32+threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < n && y+k < t) tile[threadIdx.y+k][threadIdx.x] = rad[(y+k)*n+x];
    __syncthreads();
    x = size_t(blockIdx.y)*32+threadIdx.x;
    y = size_t(blockIdx.x)*32+threadIdx.y;
    for (int k = 0; k < 32; k += 8)
        if (x < t && y+k < n) transposed[(y+k)*t+x] = tile[threadIdx.x][threadIdx.y+k];
}

__global__ void distanceKernel(const float* rad, size_t t, size_t source, float* distances) {
    size_t i = blockIdx.x;
    float best = ZERO;
    size_t bestT = 0;
    for (size_t lag = threadIdx.x; lag < t; lag += blockDim.x) {
        float sum = ZERO;
        for (size_t tt = lag; tt < t; ++tt)
            sum += rad[source*t+tt-lag]*rad[i*t+tt];
        if (sum > best) { best = sum; bestT = lag; }
    }
    __shared__ float values[128];
    __shared__ size_t lags[128];
    values[threadIdx.x] = best; lags[threadIdx.x] = bestT;
    __syncthreads();
    for (int stride = 64; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            int other = threadIdx.x+stride;
            if (values[other] > values[threadIdx.x] ||
                (values[other] == values[threadIdx.x] && lags[other] < lags[threadIdx.x])) {
                values[threadIdx.x] = values[other]; lags[threadIdx.x] = lags[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[i] = WAVE_SPEED*static_cast<float>(lags[0]);
}

struct GPUState {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<size_t> indices;
    DeviceBuffer<float> kij, rad, areas, rho, distances;
    DeviceBuffer<int> tau;
    size_t nodeCount = 0;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    GPUState gpu;
    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool validate) {
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

    // Only validation needs host copies of the GPU matrices.
    if (validate) {
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.radB.resize(timesteps * state.numTriangles, ZERO);
    }
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================


void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on CUDA...\n");
    auto& gpu = state.gpu;
    size_t n = state.numTriangles;
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    gpu.nodeCount = nodes.size();
    gpu.triangles.allocate(n); gpu.nodes.allocate(nodes.size()); gpu.indices.allocate(indices.size());
    gpu.kij.allocate(n*n); gpu.tau.allocate(n*n); gpu.areas.allocate(n); gpu.rho.allocate(n);
    gpu.rad.allocate(n*state.numTimesteps); gpu.distances.allocate(n);
    cudaCheck(cudaMemcpy(gpu.triangles.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.nodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.indices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.areas.data, state.areas.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpu.rho.data, state.rho.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(gpu.kij.data, 0, n*n*sizeof(float)));
    if (state.numTimesteps) cudaCheck(cudaMemset(gpu.rad.data, 0, n*state.numTimesteps*sizeof(float)));
    delayKernel<<<static_cast<unsigned>(std::min<size_t>((n*n+255)/256, 65535)), 256>>>(gpu.triangles.data, n, gpu.tau.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on CUDA...\n");
    constexpr size_t batchSize = 8192;
    // Double-buffer pinned staging to overlap the exact original mt19937 stream
    // generation with asynchronous transfers and GPU visibility calculations.
    struct Batch {
        float* random = nullptr;
        size_t* pairs = nullptr;
        DeviceBuffer<float> deviceRandom{batchSize*NUM_RAYS*4};
        DeviceBuffer<size_t> devicePairs{batchSize};
        cudaStream_t stream;
        Batch() {
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&random), batchSize*NUM_RAYS*4*sizeof(float)));
            cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&pairs), batchSize*sizeof(size_t)));
            cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        ~Batch() { cudaStreamDestroy(stream); cudaFreeHost(random); cudaFreeHost(pairs); }
    } batches[2];
    RandomGenerator rng(42);
    size_t count = 0, current = 0, n = state.numTriangles;
    auto submit = [&]() {
        Batch& b = batches[current];
        cudaCheck(cudaMemcpyAsync(b.devicePairs.data, b.pairs, count*sizeof(size_t), cudaMemcpyHostToDevice, b.stream));
        cudaCheck(cudaMemcpyAsync(b.deviceRandom.data, b.random, count*NUM_RAYS*4*sizeof(float), cudaMemcpyHostToDevice, b.stream));
        auto& g = state.gpu;
        formFactorKernel<<<static_cast<unsigned>((count*NUM_RAYS+127)/128), 128, 0, b.stream>>>(
            g.triangles.data, g.nodes.data, g.indices.data, g.nodeCount,
            b.devicePairs.data, b.deviceRandom.data, count, n, g.kij.data);
        cudaCheck(cudaGetLastError());
        current ^= 1;
        cudaCheck(cudaStreamSynchronize(batches[current].stream));
        count = 0;
    };
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // These pairs consumed no random draws in the original algorithm.
            if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
            Batch& b = batches[current];
            b.pairs[count] = i*n+j;
            for (int r = 0; r < NUM_RAYS*4; ++r)
                b.random[count*NUM_RAYS*4+r] = rng.rand();
            if (++count == batchSize) submit();
        }
    }
    if (count) submit();
    for (auto& b : batches) cudaCheck(cudaStreamSynchronize(b.stream));
    if (!state.kij.empty())
        cudaCheck(cudaMemcpy(state.kij.data(), state.gpu.kij.data, n*n*sizeof(float), cudaMemcpyDeviceToHost));
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on CUDA...\n");
    size_t n = state.numTriangles;
    auto& g = state.gpu;
    // Every pair of distinct mesh centers has a strictly positive delay. Separate
    // launches in one stream provide the timestep barrier without host waits.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<static_cast<unsigned>((n+3)/4), 128>>>(n, t, state.numTimesteps/2, state.sourceIndex,
            g.kij.data, g.tau.data, g.areas.data, g.rho.data, g.rad.data);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via CUDA cross-correlation...\n");
    size_t n = state.numTriangles, t = state.numTimesteps;
    auto& g = state.gpu;
    DeviceBuffer<float> transposed(n*t);
    if (t) {
        transposeRadKernel<<<dim3(static_cast<unsigned>((n+31)/32), static_cast<unsigned>((t+31)/32)), dim3(32,8)>>>(g.rad.data, transposed.data, n, t);
        cudaCheck(cudaGetLastError());
    }
    distanceKernel<<<static_cast<unsigned>(n), 128>>>(transposed.data, t, state.sourceIndex, g.distances.data);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(state.distances.data(), g.distances.data, n*sizeof(float), cudaMemcpyDeviceToHost));
    if (!state.radB.empty()) cudaCheck(cudaMemcpy(state.radB.data(), g.rad.data, n*t*sizeof(float), cudaMemcpyDeviceToHost));
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
