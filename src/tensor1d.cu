#include "tensor1d.h"

__global__
void kAdd(float *a, float *b, int N) {
    int i = blockIdx.x;
    if (i < N) {
        a[i] += b[i];
    }
}

Tensor1D::Tensor1D(int size, float* hostData) {
    this->size = size;
    cudaMalloc((void **)&(this->devData), this->size*sizeof(float));
    cudaMemcpy(devData, hostData, this->size*sizeof(float), cudaMemcpyHostToDevice);
}

Tensor1D::~Tensor1D() {
    cudaFree(this->devData);
}

float* Tensor1D::getDeviceData() {
    return this->devData;
}

float* Tensor1D::fetchDataFromDevice() {
    float* hostData = (float*)malloc(this->size*sizeof(float));
    cudaMemcpy(hostData, this->devData, this->size*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor1D::add(Tensor1D* tensor) {
    kAdd<<<this->size, 1>>>(this->getDeviceData(), tensor->getDeviceData(), this->size);
}
