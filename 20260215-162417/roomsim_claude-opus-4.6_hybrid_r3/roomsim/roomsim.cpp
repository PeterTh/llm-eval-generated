/**
 * Room Response Simulation Benchmark
 * Hybrid MPI + OpenMP + CUDA parallelization
 *
 * MPI: distributes triangle rows across ranks
 * CUDA: GPU kernels for form factors, simulation, distance computation
 * OpenMP: CPU-side parallelism for time delays and data preparation
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
#include <queue>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types (host + device)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(val_t x_, val_t y_, val_t z_) : x(x_), y(y_), z(z_) {}
    __host__ __device__ explicit Vec3(val_t v) : x(v), y(v), z(v) {}

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
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a_, const Vec3& b_, const Vec3& c_)
        : a(a_), b(b_), c(c_), _normal((b_ - a_).cross(c_ - a_).normalized()) {}

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
// GPU Octree Node (flattened for GPU traversal)
// ============================================================================

struct GPUOctreeNode {
    float cx, cy, cz;       // center
    float hx, hy, hz;       // half extent
    int children[8];         // child indices, -1 = none
    int triStart, triCount;  // leaf triangle data
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host only)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * fabsf(triNormal.x) +
              boxHalfSize.y * fabsf(triNormal.y) +
              boxHalfSize.z * fabsf(triNormal.z);
    if (fabsf(d) > r) return false;

    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * fabsf(axis.x) +
                  boxHalfSize.y * fabsf(axis.y) +
                  boxHalfSize.z * fabsf(axis.z);
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
// Octree for Spatial Acceleration (host only, flattened for GPU)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

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

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];
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

        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center, childMax = center;
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
// Flatten Octree for GPU
// ============================================================================

int flattenOctreeRecursive(const Octree* node,
                           std::vector<GPUOctreeNode>& nodes,
                           std::vector<int>& leafTriIndices) {
    int myIdx = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // Recurse children first (before accessing nodes[myIdx] which may be invalidated)
    int childIdx[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    if (node->triangleIndices.empty()) {
        for (int i = 0; i < 8; i++) {
            if (node->children[i]) {
                childIdx[i] = flattenOctreeRecursive(
                    node->children[i].get(), nodes, leafTriIndices);
            }
        }
    }

    // Now safe to write to nodes[myIdx]
    GPUOctreeNode& gn = nodes[myIdx];
    gn.cx = node->center.x;
    gn.cy = node->center.y;
    gn.cz = node->center.z;
    gn.hx = node->halfExtent.x;
    gn.hy = node->halfExtent.y;
    gn.hz = node->halfExtent.z;
    for (int i = 0; i < 8; i++) gn.children[i] = childIdx[i];

    if (!node->triangleIndices.empty()) {
        gn.triStart = static_cast<int>(leafTriIndices.size());
        gn.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) {
            leafTriIndices.push_back(static_cast<int>(idx));
        }
    } else {
        gn.triStart = 0;
        gn.triCount = 0;
    }

    return myIdx;
}

void flattenOctree(const Octree& root,
                   std::vector<GPUOctreeNode>& nodes,
                   std::vector<int>& leafTriIndices) {
    nodes.clear();
    leafTriIndices.clear();
    flattenOctreeRecursive(&root, nodes, leafTriIndices);
}

// ============================================================================
// Mesh Generation: Icosphere (host only)
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t = (1.0f + sqrtf(5.0f)) / 2.0f;

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

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

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

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// CUDA Device Functions
// ============================================================================

__device__ unsigned int xorshift32(unsigned int& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ float randFloatGPU(unsigned int& state) {
    return (float)(xorshift32(state) & 0x7FFFFF) / (float)0x800000;
}

__device__ float rayTriIntersectDevice(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    const float* tv)  // 9 floats: ax,ay,az, bx,by,bz, cx,cy,cz
{
    float e1x = tv[3]-tv[0], e1y = tv[4]-tv[1], e1z = tv[5]-tv[2];
    float e2x = tv[6]-tv[0], e2y = tv[7]-tv[1], e2z = tv[8]-tv[2];
    float pvx = dy*e2z - dz*e2y;
    float pvy = dz*e2x - dx*e2z;
    float pvz = dx*e2y - dy*e2x;
    float det = e1x*pvx + e1y*pvy + e1z*pvz;
    if (fabsf(det) < 1e-6f) return 1e30f;
    float invDet = 1.0f / det;
    float tvx = ox-tv[0], tvy = oy-tv[1], tvz = oz-tv[2];
    float u = (tvx*pvx + tvy*pvy + tvz*pvz) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;
    float qvx = tvy*e1z - tvz*e1y;
    float qvy = tvz*e1x - tvx*e1z;
    float qvz = tvx*e1y - tvy*e1x;
    float v = (dx*qvx + dy*qvy + dz*qvz) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;
    return (e2x*qvx + e2y*qvy + e2z*qvz) * invDet;
}

__device__ bool isRayBlockedDevice(
    float fx, float fy, float fz,
    float tx, float ty, float tz,
    const GPUOctreeNode* nodes,
    const int* leafTris,
    const float* triVerts,
    int srcIdx, int dstIdx)
{
    float dx = tx - fx, dy = ty - fy, dz = tz - fz;
    float rayLen = sqrtf(dx*dx + dy*dy + dz*dz);
    if (rayLen < 1e-6f) return true;
    float invLen = 1.0f / rayLen;
    float dirx = dx*invLen, diry = dy*invLen, dirz = dz*invLen;

    // Stack-based iterative octree traversal
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nidx = stack[--sp];
        const GPUOctreeNode& nd = nodes[nidx];

        // Ray-box intersection (midpoint method)
        float mdx = (fx + tx) * 0.5f - nd.cx;
        float mdy = (fy + ty) * 0.5f - nd.cy;
        float mdz = (fz + tz) * 0.5f - nd.cz;
        float hdx = dx * 0.5f, hdy = dy * 0.5f, hdz = dz * 0.5f;
        float adx = fabsf(hdx), ady = fabsf(hdy), adz = fabsf(hdz);

        if (fabsf(mdx) > nd.hx + adx) continue;
        if (fabsf(mdy) > nd.hy + ady) continue;
        if (fabsf(mdz) > nd.hz + adz) continue;

        float eps = 1e-6f;
        if (fabsf(hdy*mdz - hdz*mdy) > nd.hy*adz + nd.hz*ady + eps) continue;
        if (fabsf(hdz*mdx - hdx*mdz) > nd.hz*adx + nd.hx*adz + eps) continue;
        if (fabsf(hdx*mdy - hdy*mdx) > nd.hx*ady + nd.hy*adx + eps) continue;

        if (nd.triCount > 0) {
            // Leaf node
            for (int t = nd.triStart; t < nd.triStart + nd.triCount; t++) {
                int idx = leafTris[t];
                if (idx == srcIdx || idx == dstIdx) continue;
                float dist = rayTriIntersectDevice(fx, fy, fz, dirx, diry, dirz,
                                                    triVerts + idx * 9);
                if (dist > 1e-6f && dist < rayLen - 1e-6f) return true;
            }
        } else {
            for (int c = 0; c < 8; c++) {
                if (nd.children[c] >= 0 && sp < 63) {
                    stack[sp++] = nd.children[c];
                }
            }
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeFormFactorsKernel(
    int N, int startRow, int numRows,
    const float* __restrict__ triVerts,
    const float* __restrict__ triNormals,
    const GPUOctreeNode* __restrict__ octreeNodes,
    const int* __restrict__ leafTris,
    float* __restrict__ kij)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numPairs = numRows * N;
    if (tid >= numPairs) return;

    int localI = tid / N;
    int j = tid % N;
    int i = startRow + localI;

    if (i == j) { kij[localI * N + j] = 0.0f; return; }

    float nix = triNormals[i*3], niy = triNormals[i*3+1], niz = triNormals[i*3+2];
    float njx = triNormals[j*3], njy = triNormals[j*3+1], njz = triNormals[j*3+2];
    if (nix*njx + niy*njy + niz*njz > 0.99f) { kij[localI * N + j] = 0.0f; return; }

    // Per-(i,j) deterministic RNG
    unsigned int rng = (unsigned int)(i * 1000003u + j * 999983u + 42u);
    rng = rng * 2654435761u + 1u;

    const float* vi = triVerts + i * 9;
    const float* vj = triVerts + j * 9;
    float aix=vi[0], aiy=vi[1], aiz=vi[2];
    float abix=vi[3]-vi[0], abiy=vi[4]-vi[1], abiz=vi[5]-vi[2];
    float acix=vi[6]-vi[0], aciy=vi[7]-vi[1], aciz=vi[8]-vi[2];
    float ajx=vj[0], ajy=vj[1], ajz=vj[2];
    float abjx=vj[3]-vj[0], abjy=vj[4]-vj[1], abjz=vj[5]-vj[2];
    float acjx=vj[6]-vj[0], acjy=vj[7]-vj[1], acjz=vj[8]-vj[2];

    float result = 0.0f;
    for (int r = 0; r < 16; r++) {
        float u1 = randFloatGPU(rng), v1 = randFloatGPU(rng);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        float pix = aix + abix*u1 + acix*v1;
        float piy = aiy + abiy*u1 + aciy*v1;
        float piz = aiz + abiz*u1 + aciz*v1;

        float u2 = randFloatGPU(rng), v2 = randFloatGPU(rng);
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
        float pjx = ajx + abjx*u2 + acjx*v2;
        float pjy = ajy + abjy*u2 + acjy*v2;
        float pjz = ajz + abjz*u2 + acjz*v2;

        if (isRayBlockedDevice(pix,piy,piz, pjx,pjy,pjz,
                               octreeNodes, leafTris, triVerts, i, j))
            continue;

        float vx = pjx-pix, vy = pjy-piy, vz = pjz-piz;
        float distSqr = vx*vx + vy*vy + vz*vz;
        if (distSqr < 1e-6f) continue;
        float vNorm = sqrtf(distSqr);

        float cosI = fmaxf(0.0f, (vx*nix + vy*niy + vz*niz) / vNorm);
        float cosJ = fmaxf(0.0f, (-vx*njx + -vy*njy + -vz*njz) / vNorm);
        if (cosI <= 0.0f || cosJ <= 0.0f) continue;

        result += (cosI * cosJ) / (3.14159265358979323846f * distSqr);
    }

    kij[localI * N + j] = result / 16.0f;
}

__global__ void simulationTimestepKernel(
    int t, int N, int startRow, int numRows,
    const float* __restrict__ kij,
    const int* __restrict__ tau,
    const float* __restrict__ areas,
    const float* __restrict__ rho,
    const float* __restrict__ radE,
    float* __restrict__ radB)
{
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= numRows) return;
    int i = startRow + localIdx;

    float sumB = 0.0f;
    for (int j = 0; j < N; j++) {
        if (i == j) continue;
        int tauij = tau[i * N + j];
        if (t < tauij) continue;
        float k = kij[i * N + j];
        if (k <= 0.0f) continue;
        int srcTime = t - tauij;
        float radJ = radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;
        sumB += fminf(k * areas[j], 1.0f) * radJ;
    }
    radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
}

__global__ void computeDistancesKernel(
    int N, int T, int startRow, int numRows, int sourceIndex,
    const float* __restrict__ radB,
    float* __restrict__ distances)
{
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= numRows) return;
    int i = startRow + localIdx;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < T; t++) {
        float sum = 0.0f;
        for (int tt = t; tt < T; tt++) {
            sum += radB[(tt - t) * N + sourceIndex] * radB[tt * N + i];
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[localIdx] = 0.5f * (float)bestT;
}

// ============================================================================
// Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    // Flattened octree for GPU
    std::vector<GPUOctreeNode> gpuOctreeNodes;
    std::vector<int> gpuLeafTriIndices;

    // SoA triangle data for GPU
    std::vector<float> triVerts;    // N*9
    std::vector<float> triNormals;  // N*3

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// MPI Row Distribution Helpers
// ============================================================================

void computeRowDistribution(int N, int numRanks, int rank,
                            int& startRow, int& localRows) {
    int baseRows = N / numRanks;
    int remainder = N % numRanks;
    localRows = baseRows + (rank < remainder ? 1 : 0);
    startRow = rank * baseRows + std::min(rank, remainder);
}

void computeAllRowDistributions(int N, int numRanks,
                                std::vector<int>& recvcounts,
                                std::vector<int>& displs) {
    recvcounts.resize(numRanks);
    displs.resize(numRanks);
    for (int r = 0; r < numRanks; r++) {
        int sr, lr;
        computeRowDistribution(N, numRanks, r, sr, lr);
        recvcounts[r] = lr * N;
        displs[r] = sr * N;
    }
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (rank == 0)
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Flatten octree for GPU
    flattenOctree(state.octree, state.gpuOctreeNodes, state.gpuLeafTriIndices);
    if (rank == 0)
        printf("Flattened octree: %zu nodes, %zu leaf tri refs\n",
               state.gpuOctreeNodes.size(), state.gpuLeafTriIndices.size());

    // Prepare SoA triangle data for GPU
    size_t N = state.numTriangles;
    state.triVerts.resize(N * 9);
    state.triNormals.resize(N * 3);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tri = state.triangles[i];
        state.triVerts[i*9+0] = tri.a.x; state.triVerts[i*9+1] = tri.a.y; state.triVerts[i*9+2] = tri.a.z;
        state.triVerts[i*9+3] = tri.b.x; state.triVerts[i*9+4] = tri.b.y; state.triVerts[i*9+5] = tri.b.z;
        state.triVerts[i*9+6] = tri.c.x; state.triVerts[i*9+7] = tri.c.y; state.triVerts[i*9+8] = tri.c.z;
        Vec3 n = tri.normal();
        state.triNormals[i*3+0] = n.x; state.triNormals[i*3+1] = n.y; state.triNormals[i*3+2] = n.z;
    }

    // Initialize areas with OpenMP
    state.areas.resize(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(N, reflectivity);

    state.kij.resize(N * N, ZERO);
    state.tau.resize(N * N, 0);
    state.radE.resize(timesteps * N, ZERO);
    state.radB.resize(timesteps * N, ZERO);
    state.distances.resize(N, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation: Time Delays (OpenMP on all ranks independently)
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    size_t N = state.numTriangles;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (i == j) {
                state.tau[i * N + j] = 0;
            } else {
                val_t dist = (state.triangles[i].center() - state.triangles[j].center()).norm();
                state.tau[i * N + j] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
            }
        }
    }
}

// ============================================================================
// Precomputation: Form Factors (MPI + CUDA)
// ============================================================================

void computeFormFactors(SimulationState& state, int rank, int numRanks) {
    int N = static_cast<int>(state.numTriangles);
    int startRow, localRows;
    computeRowDistribution(N, numRanks, rank, startRow, localRows);

    if (rank == 0) printf("Computing form factors (Kij) with %d MPI ranks + CUDA...\n", numRanks);

    if (localRows == 0) {
        // This rank has no work; participate in Allgatherv with zero contribution
        std::vector<int> recvcounts, displs;
        computeAllRowDistributions(N, numRanks, recvcounts, displs);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                       state.kij.data(), recvcounts.data(), displs.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
        return;
    }

    // Allocate GPU memory
    float *d_triVerts, *d_triNormals, *d_kij;
    GPUOctreeNode *d_octreeNodes;
    int *d_leafTris;

    CUDA_CHECK(cudaMalloc(&d_triVerts, N * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_triNormals, N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kij, (size_t)localRows * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_octreeNodes, state.gpuOctreeNodes.size() * sizeof(GPUOctreeNode)));
    CUDA_CHECK(cudaMalloc(&d_leafTris, state.gpuLeafTriIndices.size() * sizeof(int)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_triVerts, state.triVerts.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_triNormals, state.triNormals.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_octreeNodes, state.gpuOctreeNodes.data(),
                          state.gpuOctreeNodes.size() * sizeof(GPUOctreeNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_leafTris, state.gpuLeafTriIndices.data(),
                          state.gpuLeafTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    long long numPairs = (long long)localRows * N;
    int gridSize = static_cast<int>((numPairs + blockSize - 1) / blockSize);

    computeFormFactorsKernel<<<gridSize, blockSize>>>(
        N, startRow, localRows,
        d_triVerts, d_triNormals,
        d_octreeNodes, d_leafTris,
        d_kij);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(state.kij.data() + (size_t)startRow * N, d_kij,
                          (size_t)localRows * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory for form factors
    CUDA_CHECK(cudaFree(d_triVerts));
    CUDA_CHECK(cudaFree(d_triNormals));
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_octreeNodes));
    CUDA_CHECK(cudaFree(d_leafTris));

    // MPI Allgatherv to distribute full kij
    std::vector<int> recvcounts, displs;
    computeAllRowDistributions(N, numRanks, recvcounts, displs);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                   state.kij.data(), recvcounts.data(), displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);

    if (rank == 0) printf("  Form factors computed.\n");
}

// ============================================================================
// Simulation Phase (MPI + CUDA, per-timestep sync)
// ============================================================================

void runSimulation(SimulationState& state, int rank, int numRanks) {
    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int startRow, localRows;
    computeRowDistribution(N, numRanks, rank, startRow, localRows);

    if (rank == 0) printf("Running wave propagation simulation with MPI+CUDA...\n");

    // Allocate GPU memory for simulation
    float *d_kij, *d_areas, *d_rho, *d_radE, *d_radB;
    int *d_tau;

    CUDA_CHECK(cudaMalloc(&d_kij, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tau, (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_areas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radE, (size_t)T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_radB, (size_t)T * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_kij, state.kij.data(), (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tau, state.tau.data(), (size_t)N * N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), (size_t)T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), (size_t)T * N * sizeof(float), cudaMemcpyHostToDevice));

    // Per-timestep row distribution for Allgatherv
    std::vector<int> recvcounts_row(numRanks), displs_row(numRanks);
    for (int r = 0; r < numRanks; r++) {
        int sr, lr;
        computeRowDistribution(N, numRanks, r, sr, lr);
        recvcounts_row[r] = lr;
        displs_row[r] = sr;
    }

    std::vector<float> h_radB_row(N);

    int blockSize = 256;
    int gridSize = localRows > 0 ? (localRows + blockSize - 1) / blockSize : 0;

    for (int t = 0; t < T; ++t) {
        if (localRows > 0) {
            simulationTimestepKernel<<<gridSize, blockSize>>>(
                t, N, startRow, localRows,
                d_kij, d_tau, d_areas, d_rho, d_radE, d_radB);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy local results to host
            CUDA_CHECK(cudaMemcpy(h_radB_row.data() + startRow, d_radB + (size_t)t * N + startRow,
                                  localRows * sizeof(float), cudaMemcpyDeviceToHost));
        }

        // Allgatherv this timestep's radB row
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                       h_radB_row.data(), recvcounts_row.data(), displs_row.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);

        // Copy full row back to GPU
        CUDA_CHECK(cudaMemcpy(d_radB + (size_t)t * N, h_radB_row.data(),
                              N * sizeof(float), cudaMemcpyHostToDevice));

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }

    // Copy full radB back to host for distance computation
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, (size_t)T * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
}

// ============================================================================
// Distance Computation (MPI + CUDA)
// ============================================================================

void computeDistances(SimulationState& state, int rank, int numRanks) {
    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    int startRow, localRows;
    computeRowDistribution(N, numRanks, rank, startRow, localRows);

    if (rank == 0) printf("Computing distances via cross-correlation with MPI+CUDA...\n");

    if (localRows > 0) {
        float *d_radB, *d_distances;
        CUDA_CHECK(cudaMalloc(&d_radB, (size_t)T * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_distances, localRows * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_radB, state.radB.data(), (size_t)T * N * sizeof(float), cudaMemcpyHostToDevice));

        int blockSize = 256;
        int gridSize = (localRows + blockSize - 1) / blockSize;

        computeDistancesKernel<<<gridSize, blockSize>>>(
            N, T, startRow, localRows, static_cast<int>(state.sourceIndex),
            d_radB, d_distances);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(state.distances.data() + startRow, d_distances,
                              localRows * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_radB));
        CUDA_CHECK(cudaFree(d_distances));
    }

    // Gather all distances to all ranks (needed for validation/output)
    std::vector<int> recvcounts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; r++) {
        int sr, lr;
        computeRowDistribution(N, numRanks, r, sr, lr);
        recvcounts[r] = lr;
        displs[r] = sr;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT,
                   state.distances.data(), recvcounts.data(), displs.data(),
                   MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (rank 0 only)
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

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

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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
    MPI_Init(&argc, &argv);

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Set CUDA device (round-robin across available GPUs)
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % numDevices));
    } else {
        if (rank == 0) fprintf(stderr, "No CUDA devices available!\n");
        MPI_Finalize();
        return 1;
    }

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResultsFlag = false;

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
            printResultsFlag = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark (MPI+OpenMP+CUDA)\n");
        printf("=====================================================\n");
        printf("MPI ranks: %d\n", numRanks);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", numDevices);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize (all ranks independently)
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);  // OpenMP on all ranks
    computeFormFactors(state, rank, numRanks);  // MPI + CUDA

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (rank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Output results (rank 0 only)
    if (rank == 0) {
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

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

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResultsFlag) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        if (validate) {
            if (!validateResults(state)) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
