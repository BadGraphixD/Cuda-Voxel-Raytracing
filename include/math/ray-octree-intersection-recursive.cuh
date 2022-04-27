#pragma once

#include <cuda_runtime.h>

#include "math/rt-math.cuh"
#include "util/buffer.cuh"
#include "voxel/chunk.h"
#include "voxel/material.h"

/*

    Code in this file is depricated, slow and should not be used :)

*/

struct Octree {
    float3 min, max, center;

    Octree() = default;
    Octree(const Chunk& chunk) {
        min = {
            (float)chunk.posx,
            (float)chunk.posy,
            (float)chunk.posz
        };

        center = {
            (float)(chunk.posx + 0.5f),
            (float)(chunk.posy + 0.5f),
            (float)(chunk.posz + 0.5f)
        };

        max = {
            (float)(chunk.posx + 1.0f),
            (float)(chunk.posy + 1.0f),
            (float)(chunk.posz + 1.0f)
        };
    }

    __device__ int traverse(Ray ray, const Buffer<Node>& nodes, const Buffer<Material>& materials) {
        unsigned char a = 0;
        
        if (ray.dir.x < 0.0f) {
            ray.org.x = center.x * 2.0f - ray.org.x;
            ray.invDir.x = -ray.invDir.x;
            a |= 4;
        }
        if (ray.dir.y < 0.0f) {
            ray.org.y = center.y * 2.0f - ray.org.y;
            ray.invDir.y = -ray.invDir.y;
            a |= 2;
        }
        if (ray.dir.z < 0.0f) {
            ray.org.z = center.z * 2.0f - ray.org.z;
            ray.invDir.z = -ray.invDir.z;
            a |= 1;
        }

        // {
        //     bool invertX = ray.dir.x < 0.0f;
        //     ray.org.x = invertX ? center.x * 2.0f - ray.org.x : ray.org.x;
        //     ray.invDir.x = invertX ? -ray.invDir.x : ray.invDir.x;
        //     a |= (4 * invertX);
        // }
        
        // {
        //     bool invertY = ray.dir.y < 0.0f;
        //     ray.org.y = invertY ? center.y * 2.0f - ray.org.y : ray.org.y;
        //     ray.invDir.y = invertY ? -ray.invDir.y : ray.invDir.y;
        //     a |= (2 * invertY);
        // }
        
        // {
        //     bool invertZ = ray.dir.z < 0.0f;
        //     ray.org.z = invertZ ? center.z * 2.0f - ray.org.z : ray.org.z;
        //     ray.invDir.z = invertZ ? -ray.invDir.z : ray.invDir.z;
        //     a |= (1 * invertZ);
        // }

        const float tx0 = (min.x - ray.org.x) * ray.invDir.x;
        const float tx1 = (max.x - ray.org.x) * ray.invDir.x;
        const float ty0 = (min.y - ray.org.y) * ray.invDir.y;
        const float ty1 = (max.y - ray.org.y) * ray.invDir.y;
        const float tz0 = (min.z - ray.org.z) * ray.invDir.z;
        const float tz1 = (max.z - ray.org.z) * ray.invDir.z;

        if (fmaxf(fmaxf(tx0, ty0), tz0) < fminf(fminf(tx1, ty1), tz1)) {
            return proc_subtree(tx0, ty0, tz0, tx1, ty1, tz1, 0, a, nodes, materials);
        }

        return -1;
    }

    __device__ int proc_subtree(const float tx0, const float ty0, const float tz0, const float tx1, const float ty1, const float tz1, const int node, const unsigned char& a, const Buffer<Node>& nodes, const Buffer<Material>& materials) {

        if (tx1 < 0.0f || ty1 < 0.0f || tz1 < 0.0f) return -1;

        if (nodes.get(node).isVoxel) {
            return materials.get(nodes.get(node).materialIdx).transparent ? -1 : node;
        }

        const float txm = 0.5f * (tx0 + tx1);
        const float tym = 0.5f * (ty0 + ty1);
        const float tzm = 0.5f * (tz0 + tz1);
        
        int currentNode = getFirstQuadrant(tx0, ty0, tz0, txm, tym, tzm);
        
        do {
            int foundNode = -1;
            switch (currentNode) {
            case 0:
                foundNode = proc_subtree(tx0, ty0, tz0, txm, tym, tzm, nodes.get(node).firstChildIdx + a, a, nodes, materials);
                currentNode = getNextQuadrant(txm, 4, tym, 2, tzm, 1);
                break;
            case 1:
                foundNode = proc_subtree(tx0, ty0, tzm, txm, tym, tz1, nodes.get(node).firstChildIdx + (1 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(txm, 5, tym, 3, tz1, 8);
                break;
            case 2:
                foundNode = proc_subtree(tx0, tym, tz0, txm, ty1, tzm, nodes.get(node).firstChildIdx + (2 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(txm, 6, ty1, 8, tzm, 3);
                break;
            case 3:
                foundNode = proc_subtree(tx0, tym, tzm, txm, ty1, tz1, nodes.get(node).firstChildIdx + (3 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(txm, 7, ty1, 8, tz1, 8);
                break;
            case 4:
                foundNode = proc_subtree(txm, ty0, tz0, tx1, tym, tzm, nodes.get(node).firstChildIdx + (4 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(tx1, 8, tym, 6, tzm, 5);
                break;
            case 5:
                foundNode = proc_subtree(txm, ty0, tzm, tx1, tym, tz1, nodes.get(node).firstChildIdx + (5 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(tx1, 8, tym, 7, tz1, 8);
                break;
            case 6:
                foundNode = proc_subtree(txm, tym, tz0, tx1, ty1, tzm, nodes.get(node).firstChildIdx + (6 ^ a), a, nodes, materials);
                currentNode = getNextQuadrant(tx1, 8, ty1, 8, tzm, 7);
                break;
            case 7:
                foundNode = proc_subtree(txm, tym, tzm, tx1, ty1, tz1, nodes.get(node).firstChildIdx + (7 ^ a), a, nodes, materials);
                currentNode = 8;
                break;
            }

            if (foundNode != -1) {
                return foundNode;
            }
        }
        while (currentNode < 8);
        
        return -1;
    }

    __device__ int getFirstQuadrant(const float& tx0, const float& ty0, const float& tz0, const float& txm, const float& tym, const float& tzm) {
        unsigned char a = 0;

        if (tx0 > ty0) {
            if (tx0 > tz0){
                if (tym < tx0) a |= 2;
                if (tzm < tx0) a |= 1;
                return (int) a;
            }
        }

        else {
            if (ty0 > tz0) {
                if (txm < ty0) a |= 4;
                if (tzm < ty0) a |= 1;
                return (int) a;
            }
        }

        if (txm < tz0) a |= 4;
        if (tym < tz0) a |= 2;
        return (int) a;

        // if (tx0 > ty0 && tx0 > tz0 && tym < tx0) a |= 2;
        // if (tx0 > ty0 && tx0 > tz0 && tzm < tx0) a |= 1;
        // if (tx0 > ty0 && tx0 > tz0) return (int)a;

        // if (tx0 <= ty0 && ty0 > tz0 && txm < ty0) a |= 4;
        // if (tx0 <= ty0 && ty0 > tz0 && tzm < ty0) a |= 1;
        // if (tx0 <= ty0 && ty0 > tz0) return (int) a;

        // if (txm < tz0) a |= 4;
        // if (tym < tz0) a |= 2;
        // return (int) a;
    }

    __device__ int getNextQuadrant(const float& txm, const int& x, const float& tym, const int& y, const float& tzm, const int& z) {

        if (txm < tym) {
            if (txm < tzm) return x;
        }

        else {
            if (tym < tzm) return y;
        }
        
        return z;

        // bool txm_tym = (txm < tym);
        // bool txm_tzm = (txm < tzm);
        // bool tym_tzm = (tym < tzm);

        // return (txm_tym && txm_tzm) * x + !(txm_tym && txm_tzm) * ((!txm_tym && tym_tzm) * y + (txm_tym || !tym_tzm) * z);
    }

};
