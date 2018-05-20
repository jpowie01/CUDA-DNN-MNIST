#include "tensor2d.h"

__global__
void kAdd(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeY + x] += b[y*sizeY + x];
    }
}

__global__
void kSubtract(float *a, float *b, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeY + x] -= b[y*sizeY + x];
    }
}

__global__
void kScale(float *a, float factor, int sizeX, int sizeY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        a[y*sizeY + x] *= factor;
    }
}

__global__
void kMultiply(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
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

__global__
void kMultiplyByTransposition(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bY && row < aY) {
        float sum = 0.0f;
        for (int i = 0; i < aX; i++) {
            sum += a[row*aX+i] * b[col*bX+i];
        }
        c[row*bY+col] = sum;
    }
}

__global__
void kTransposeAndMultiply(float *a, int aX, int aY, float* b, int bX, int bY, float *c)
{
    // TODO: This implementation is very basic. Please improve me!
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < bX && row < aX) {
        float sum = 0.0f;
        for (int i = 0; i < bY; i++) {
            sum += a[i*aX+row] * b[i*bX+col];
        }
        c[row*bX+col] = sum;
    }
}

__global__
void kMeanX(float* a, int aX, int aY, float* b)
{
    // TODO: Check if this implementation is fine... May be broken :/
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < aX) {
        float sum = 0.0;
        for (int i = 0; i < aY; i++) {
            sum += a[i*aX + col];
        }
        b[col] = sum / aY;
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

void Tensor2D::subtract(Tensor2D* tensor) {
    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kSubtract<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), tensor->getDeviceData(), this->sizeX, this->sizeY);
}

void Tensor2D::scale(float factor) {
    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((this->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kScale<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), factor, this->sizeX, this->sizeY);
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

Tensor2D* Tensor2D::multiplyByTransposition(Tensor2D* tensor) {
    float* output;
    cudaMalloc((void **)&(output), this->sizeY*tensor->sizeY*sizeof(float));

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeY + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kMultiplyByTransposition<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output);

    return new Tensor2D(this->sizeY, tensor->sizeY, output);
}

Tensor2D* Tensor2D::transposeAndMultiply(Tensor2D* tensor) {
    float* output;
    cudaMalloc((void **)&(output), this->sizeX*tensor->sizeX*sizeof(float));

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((tensor->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (this->sizeX + threadsPerBlock.y)/threadsPerBlock.y);
    kTransposeAndMultiply<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, tensor->getDeviceData(), tensor->sizeX, tensor->sizeY, output);

    return new Tensor2D(tensor->sizeX, this->sizeX, output);
}

Tensor2D* Tensor2D::meanX() {
    float* output;
    cudaMalloc((void **)&(output), this->sizeX*sizeof(float));

    int threadsPerBlock = 64;  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    int numBlocks = (this->sizeX + threadsPerBlock)/threadsPerBlock;
    kMeanX<<<numBlocks, threadsPerBlock>>>(this->getDeviceData(), this->sizeX, this->sizeY, output);

    return new Tensor2D(this->sizeX, 1, output);
}
