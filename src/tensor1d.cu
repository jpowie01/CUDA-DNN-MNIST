#include "tensor1d.h"

__global__
void kAdd(int *a, int *b, int N) {
    int i = blockIdx.x;
    if (i < N) {
        a[i] += b[i];
    }
}

Tensor1D::Tensor1D(int size, int* hostData) {
    this->size = size;
    cudaMalloc((void **)&(this->devData), this->size*sizeof(int));
    cudaMemcpy(devData, hostData, this->size*sizeof(int), cudaMemcpyHostToDevice);
}

Tensor1D::~Tensor1D() {
    cudaFree(this->devData);
}

int* Tensor1D::getDeviceData() {
    return this->devData;
}

int* Tensor1D::fetchDataFromDevice() {
    int* hostData = (int*)malloc(this->size*sizeof(int));
    cudaMemcpy(hostData, this->devData, this->size*sizeof(int), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor1D::add(Tensor1D* tensor) {
    kAdd<<<this->size, 1>>>(this->getDeviceData(), tensor->getDeviceData(), this->size);
}
