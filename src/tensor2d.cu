#include "tensor2d.h"

__global__
void kAdd(int *a, int *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeY + x] += b[y*sizeY + x];
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, int** hostData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(int));
    cudaMemcpy(devData, *hostData, this->sizeX*this->sizeY*sizeof(int), cudaMemcpyHostToDevice);
}

Tensor2D::~Tensor2D() {
    cudaFree(this->devData);
}

int* Tensor2D::getDeviceData() {
    return this->devData;
}

int** Tensor2D::fetchDataFromDevice() {
    int** hostData = new int*[this->sizeY];
    *hostData = new int[this->sizeY * this->sizeX];
    for (int i = 1; i < this->sizeY; i++) hostData[i] = hostData[i-1] + this->sizeX;

    cudaMemcpy(*hostData, this->devData, this->sizeX*this->sizeY*sizeof(int), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor2D::add(Tensor2D* tensor) {
    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}
