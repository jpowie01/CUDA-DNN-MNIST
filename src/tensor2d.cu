#include "tensor2d.h"

__global__
void kAdd(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeY + x] += b[y*sizeY + x];
    }
}

__global__ void kMultiply(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bX && row < aY) {
        float sum = 0.0f;
        for (int i = 0; i < aX; i++) {
            sum += a[row*aX+i] * b[i*bX+col];
        }
        c[row*bX+col] = sum;
    }
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float** hostData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    cudaMalloc((void **)&(this->devData), this->sizeX*this->sizeY*sizeof(float));
    cudaMemcpy(devData, *hostData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyHostToDevice);
}

Tensor2D::Tensor2D(int sizeX, int sizeY, float* devData) {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->devData = devData;
}

Tensor2D::~Tensor2D() {
    cudaFree(this->devData);
}

float* Tensor2D::getDeviceData() {
    return this->devData;
}

float** Tensor2D::fetchDataFromDevice() {
    float** hostData = new float*[this->sizeY];
    *hostData = new float[this->sizeY * this->sizeX];
    for (int i = 1; i < this->sizeY; i++) hostData[i] = hostData[i-1] + this->sizeX;

    cudaMemcpy(*hostData, this->devData, this->sizeX*this->sizeY*sizeof(float), cudaMemcpyDeviceToHost);
    return hostData;
}

void Tensor2D::add(Tensor2D* tensor) {
    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kAdd<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

Tensor2D* Tensor2D::multiply(Tensor2D* tensor) {
    float* output;
    cudaMalloc((void **)&(output), this->sizeY*tensor->sizeX*sizeof(float));

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kMultiply<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output);

    return new Tensor2D(tensor->sizeX, this->sizeY, output);
}
