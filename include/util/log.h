#pragma once

#include <cuda_runtime.h>

#define cudaErrchk(code) { gpuAssert((code), __FILE__, __LINE__); }
#define debugKernelLaunch cudaErrchk(cudaDeviceSynchronize()); cudaErrchk(cudaGetLastError());

void gpuAssert(cudaError_t code, const char* file, int line);

class Log {
public:
	static void Error(const char* msg);
	static void Warning(const char* msg);
	static void Message(const char* msg);
private:
	Log() {}
};