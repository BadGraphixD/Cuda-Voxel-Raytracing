#pragma once

#include <vector>

#include "voxel/material.h"

class Node {
    // Used for updating the parent and finding neighbour nodes/voxels
    unsigned int parentPtr;

    // Used for updating and finding child nodes/voxels
    unsigned int firstChildPtr;

    // In case of a node: used to display distant voxels (replace 8 children with 1 averaged parent)
    Material material;

    // Determines if this is a node or voxel (in case of a node the algorithm must traverse further)
    bool isVoxel;
}

class Chunk {
    std::vector<Node> nodes;
}