#pragma once

#include <vector>
#include <stack>

#include <cuda_runtime.h>

struct Node {
    Node() = default;
    Node(int parentIdx, int firstChildIdx, int materialIdx, bool isVoxel = true, float3 color = make_float3(0, 0, 0));

    // Used for updating the parent and finding neighbour nodes/voxels
    int parentIdx;

    // Used for updating and finding child nodes/voxels
    int firstChildIdx;

    // Material of the voxel (stored elsewhere)
    int materialIdx;

    // Determines if this is a node or voxel (in case of a node the algorithm must traverse further)
    bool isVoxel;

    // Lighting of the voxel
    float3 color;
};

class Chunk {
public:
    std::vector<Node> nodes;

    int posx, posy, posz;

    Chunk() = default;
    Chunk(int posx, int posy, int posz, int materialIdx);

    int splitNode(int nodeIdx);
    int combineNode(int nodeIdx, bool ignoreChildMaterials = false, int materialIdx = -1);

    inline void setNodeMaterial(int nodeIdx, int materialIdx) { nodes[nodeIdx].materialIdx = materialIdx; };
    inline int getNodeAmount() { return (int)nodes.size(); }
    
private:
    std::stack<int> unusedStorageIndices;
    
    int addChildNodes(int parentNodeIdx, int materialIdx, float3 color);
};
