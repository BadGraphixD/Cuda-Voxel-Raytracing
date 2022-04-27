#pragma once

#include "application/application.h"
#include "util/render-data-displayer.h"
#include "util/grid-layout.h"
#include "util/buffer.cuh"

#include "graphics/raytracing/ray-generation.cuh"
#include "graphics/raytracing/ray-tracing.cuh"
#include "graphics/camera.h"

/*

    Code in this file is for testing purposes only. It is written in a horrible way and is not permanent :)

*/

class CVRApp : public Application {
private:
    BufferManager bufferManager;
    RenderDataDisplayer displayer;

    GridLayout layout;

    CameraController cameraController;

    MaterialList materialList;
    Chunk chunk;

    Octree octree;
    
    Buffer<Payload> d_payload;
    Buffer<Material> d_materials;
    Buffer<Node> d_nodes;
    Buffer<Ray> d_rays;
    Buffer<float3> d_frame;
    Buffer<uchar3> d_result;

    float deltaTimeSum = 0.0f;

protected:
    void initialize() override {
        cudaDeviceSetLimit(cudaLimit::cudaLimitStackSize, 1024);

        chunk = Chunk(0, 0, 0, 0);
        buildTerrain0(chunk);

        bufferManager.addBuffer(&d_payload, BufferLocation::DEVICE);
        bufferManager.addBuffer(&d_materials, BufferLocation::DEVICE, materialList.getAmount());
        bufferManager.addBuffer(&d_nodes, BufferLocation::DEVICE, chunk.getNodeAmount());
        bufferManager.addBuffer(&d_rays, BufferLocation::DEVICE, window.getPixelAmount());
        bufferManager.addBuffer(&d_frame, BufferLocation::DEVICE, window.getPixelAmount());
        bufferManager.addBuffer(&d_result, BufferLocation::DEVICE, window.getPixelAmount());

        d_materials.copyFrom(materialList.getData(), BufferLocation::HOST);
        d_nodes.copyFrom(chunk.nodes.data(), BufferLocation::HOST);

        layout = GridLayout(2, 16, window);
        cameraController = CameraController(90, Camera(), 2.0f, 100.0f);

        cameraController.tempInitializeBecauseOfBug(&window, deltaTime);

        octree = Octree(chunk);
    }

    void update() override {

        cameraController.updateCamera(&window, deltaTime);

        // if (window.input.getButton(GLFW_MOUSE_BUTTON_1)) {
        //     printf("Splitting Node");
        //     window.input.buttons[GLFW_MOUSE_BUTTON_1] = false;

        //     float3 rot = cameraController.getCamera().rotation;
        //     Ray viewRay = { cameraController.getCamera().position, directionFromAngles(rot.x, rot.y) };
        //     viewRay.calcInvDir();

        //     int voxelIdx = octree.traverse(viewRay, chunk.nodes.data(), materialList.getData());

        //     if (voxelIdx != -1 && chunk.nodes[voxelIdx].isVoxel) {
        //         int childIdx = chunk.splitNode(voxelIdx);

        //         for (int i = 0; i < 8; i++) {
        //             chunk.nodes[childIdx + i].materialIdx = rand() % materialList.getAmount();
        //         }

        //         d_nodes.destroy();
        //         d_nodes.init(BufferLocation::DEVICE, chunk.getNodeAmount());
        //         d_nodes.copyFrom(chunk.nodes.data(), BufferLocation::HOST);
        //     }
        // }

        Payload payload = {
            window.getWidth(), window.getHeight(),
            cameraController.getCamera()
        };

        if (frame > 100) {
            deltaTimeSum += deltaTime;
            printf(
                "Average FPS: %f | Average ms per frame: %f | FPS: %f | ms per frame: %f\n",
                (frame - 100) / deltaTimeSum, (deltaTimeSum / (frame - 100)) * 1000.0f, 1.0f / deltaTime, deltaTime * 1000.0f
            );
        }
        
		d_payload.copyFrom(&payload, BufferLocation::HOST);

        // debugKernelLaunch;
        generateRays<<<layout.blocks, layout.threads>>>(d_rays, d_payload);
        // debugKernelLaunch;
        traceRays<<<layout.blocks, layout.threads>>>(d_rays, octree, d_nodes, d_materials, d_frame, d_payload);
        // debugKernelLaunch;
        frameToUChar3<<<layout.blocks, layout.threads>>>(d_frame, d_result, d_payload);
        // debugKernelLaunch;

        displayer.display(d_result.getPtr(), &window);
    }
    
    void terminate() override {
        bufferManager.destroyBuffers();
    }

private:
    void buildTerrain0(Chunk& chunk) {
        std::vector<int> normalMaterials;

        int airMat = materialList.registerMaterial(Material(make_float3(0, 0, 0), true));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.1f, .2f, .8f), false)));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.1f, .8f, .2f), false)));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.8f, .1f, .2f), false)));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.8f, .8f, .2f), false)));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.2f, .8f, .8f), false)));
        normalMaterials.push_back(materialList.registerMaterial(Material(make_float3(.8f, .2f, .8f), false)));

        int foo1 = chunk.splitNode(0);
        int voxelAmount = 8;

        int splittingRatio = 1000000000000;

        for (int i = 0; i < 8; i++) {
            int foo2 = chunk.splitNode(foo1 + i);
            voxelAmount += 7;

            if (rand() % splittingRatio != 0) for (int j = 0; j < 8; j++) {
                int foo3 = chunk.splitNode(foo2 + j);
                voxelAmount += 7;
                
                if (rand() % splittingRatio != 0) for (int k = 0; k < 8; k++) {
                    int foo4 = chunk.splitNode(foo3 + k);
                    voxelAmount += 7;

                    if (rand() % splittingRatio != 0) for (int l = 0; l < 8; l++) {
                        int foo5 = chunk.splitNode(foo4 + l);
                        voxelAmount += 7;
                            
                        if (rand() % splittingRatio != 0) for (int m = 0; m < 8; m++) {
                            int foo6 = chunk.splitNode(foo5 + m);
                            voxelAmount += 7;
                            
                            if (rand() % splittingRatio != 0) for (int n = 0; n < 8; n++) {
                                int foo7 = chunk.splitNode(foo6 + n);
                                voxelAmount += 7;
                                
                                if (rand() % splittingRatio != 0) for (int o = 0; o < 8; o++) {
                                    int foo8 = chunk.splitNode(foo7 + o);
                                    voxelAmount += 7;
                                
                                    if (rand() % splittingRatio != 0) for (int p = 0; p < 8; p++) {
                                        int foo9 = chunk.splitNode(foo8 + p);
                                        voxelAmount += 7;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        printf("Number of voxels: %d\n", voxelAmount);
        printf("Chunk Byte Size in Mb: %d\n", (int)(chunk.getNodeAmount() * sizeof(Node) / 1000000));

        for (int i = 0; i < chunk.nodes.size(); i++) {
            chunk.nodes[i].materialIdx = normalMaterials[rand() % normalMaterials.size()];
            if (rand() % 4 != 0) chunk.nodes[i].materialIdx = airMat;
        }
    }
    
    void buildTerrain1(Chunk& chunk) {

        int airMat = materialList.registerMaterial(Material(make_float3(.0f, .0f, .0f), true));
        int grassMat = materialList.registerMaterial(Material(make_float3(.2f, .8f, .4f), false));
        int dirtMat = materialList.registerMaterial(Material(make_float3(.5f, .4f, .3f), false));
        int stoneMat = materialList.registerMaterial(Material(make_float3(.2f, .2f, .25f), false));

        chunk.setNodeMaterial(0, grassMat);
        splitVoxelIntoTerrain(chunk, 0, 0);
        
        printf("Number of nodes: %d\n", chunk.getNodeAmount());
        printf("Chunk Byte Size in Mb: %d\n", (int)(chunk.getNodeAmount() * sizeof(Node) / 1000000));
    }

    void splitVoxelIntoTerrain(Chunk& chunk, int nodeIdx, int depth) {

        if (depth >= 4) return;

        int firstChildNode = chunk.splitNode(nodeIdx);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {

                int lowerChildIdx = firstChildNode + i + j * 4;
                int upperChildIdx = lowerChildIdx + 2;

                if (sin(nodeIdx) > 0.0f) {
                    chunk.setNodeMaterial(upperChildIdx, (rand() % 3) + 1);
                    splitVoxelIntoTerrain(chunk, upperChildIdx, depth + 1);
                }
                else {
                    chunk.setNodeMaterial(upperChildIdx, 0);
                    chunk.setNodeMaterial(lowerChildIdx, (rand() % 3) + 1);
                    splitVoxelIntoTerrain(chunk, lowerChildIdx, depth + 1);
                }
                
            }
        }

    }

public:
    CVRApp(Window window)
        : Application(window), displayer(&this->window) {}
};