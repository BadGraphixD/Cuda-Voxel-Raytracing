#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "util/log.h"

enum BufferLocation {
    HOST, DEVICE
};

template<class T>
class Buffer {
private:
    BufferLocation location;
    unsigned int size;
    T* ptr = nullptr;

public:
	Buffer() = default;
	__host__ void init(BufferLocation location, unsigned int size = 1) {
		this->location = location;
		this->size = size;

		switch (location) {
		case BufferLocation::HOST:
			ptr = new T[size];
			break;
		case BufferLocation::DEVICE:
			cudaMalloc(&ptr, size * sizeof(T));
			break;
		}
	}
    __host__ void destroy() {
		if (ptr == nullptr) return;

		switch (location) {
		case BufferLocation::HOST:
			delete[] ptr;
			break;
		case BufferLocation::DEVICE:
			cudaFree(ptr);
			break;
		}
	}

	__host__ void copyInto(Buffer buffer) const {
		if (getByteSize() != buffer.getByteSize()) {
			Log::Error("Can't copy into buffer with different size!");
			return;
		}

		copyInto(buffer.getPtr(), buffer.location);
	}

	__host__ void copyInto(void* dest, BufferLocation destLoc) const {
		cudaMemcpyKind memcpyKind;

		if (location == BufferLocation::HOST && destLoc == BufferLocation::HOST) memcpyKind = cudaMemcpyHostToHost;
		else if (location == BufferLocation::HOST && destLoc == BufferLocation::DEVICE) memcpyKind = cudaMemcpyHostToDevice;
		else if (location == BufferLocation::DEVICE && destLoc == BufferLocation::HOST) memcpyKind = cudaMemcpyDeviceToHost;
		else if (location == BufferLocation::DEVICE && destLoc == BufferLocation::DEVICE) memcpyKind = cudaMemcpyDeviceToDevice;

		cudaMemcpy(dest, ptr, getByteSize(), memcpyKind);
	}

	__host__ void copyFrom(Buffer buffer) const {
		if (getByteSize() != buffer.getByteSize()) {
			Log::Error("Can't copy into buffer with different size!");
			return;
		}

		copyFrom(buffer.getPtr(), buffer.location);
	}

	__host__ void copyFrom(void* src, BufferLocation srcLoc) const {
		cudaMemcpyKind memcpyKind;

		if (location == BufferLocation::HOST && srcLoc == BufferLocation::HOST) memcpyKind = cudaMemcpyHostToHost;
		else if (location == BufferLocation::HOST && srcLoc == BufferLocation::DEVICE) memcpyKind = cudaMemcpyDeviceToHost;
		else if (location == BufferLocation::DEVICE && srcLoc == BufferLocation::HOST) memcpyKind = cudaMemcpyHostToDevice;
		else if (location == BufferLocation::DEVICE && srcLoc == BufferLocation::DEVICE) memcpyKind = cudaMemcpyDeviceToDevice;

		cudaMemcpy(ptr, src, getByteSize(), memcpyKind);
	}
	
	__host__ void fill(const int& value) {
		if (location == BufferLocation::HOST) memset(ptr, value, getByteSize());
		else cudaMemset(ptr, value, getByteSize());
	}

	__host__ __device__ inline void set(const unsigned int& idx, const T& value) {
		ptr[idx] = value;
	}

	__host__ __device__ inline T get(const unsigned int& idx = 0) const {
		return ptr[idx];
	}

	__host__ __device__ inline T* getPtr() const {
		return ptr;
	}

	__host__ __device__ inline T* getPtrOf(const unsigned int& idx = 0) const {
		return &ptr[idx];
	}

	__host__ __device__ inline unsigned int getSize() const {
		return size;
	}

	__host__ __device__ inline size_t getByteSize() const {
		return size * sizeof(T);
	}
    
};

class BufferManager {
private:
	std::vector<void*> buffers;

public:
	BufferManager() = default;

	template<class T>
	void addBuffer(Buffer<T>* buffer, BufferLocation location, unsigned int size = 1) {
		buffer->init(location, size);
		buffers.push_back((void*)buffer);
	}
	
	void destroyBuffers();
};
