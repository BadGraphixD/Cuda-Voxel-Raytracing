@echo off

cd build
cmake --build . --config Release
start "" bin/Release/cuda-voxel-raytracing.exe