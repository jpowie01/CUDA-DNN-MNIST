#include "relu.h"

__global__
void kReLuForward(float *a, int sizeX, int sizeY, float* b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        if (a[y*sizeX + x] < 0.0) {
            b[y*sizeX + x] = 0;
        } else {
            b[y*sizeX + x] = a[y*sizeX + x];
        }
    }
}

__global__
void kReLuBackward(float *a, int sizeX, int sizeY, float* b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < sizeX && y < sizeY) {
        if (a[y*sizeX + x] < 0.0) {
            b[y*sizeX + x] = 0;
        } else {
            b[y*sizeX + x] = a[y*sizeX + x];
        }
    }
}

ReLuLayer::ReLuLayer(int inputOutput) {
    this->input = this->output = inputOutput;
    this->weights = NULL;
    this->bias = NULL;
    this->deltaWeights = NULL;
    this->deltaBias = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;
}

Tensor2D* ReLuLayer::forward(Tensor2D* data) {
    this->inputData = data;

    if (!this->outputForward) {
        this->outputForward = new Tensor2D(data->getSize(X), data->getSize(Y));
    }

    dim3 threadsPerBlock(Configuration::reLuBlockSize, Configuration::reLuBlockSize);
    dim3 numBlocks((data->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                   (data->getSize(Y) + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuForward<<<numBlocks, threadsPerBlock>>>(data->getDeviceData(), data->getSize(X), data->getSize(Y), this->outputForward->getDeviceData());
    return this->outputForward;
}
 
Tensor2D* ReLuLayer::backward(Tensor2D* gradients) {
    dim3 threadsPerBlock(Configuration::reLuBlockSize, Configuration::reLuBlockSize);
    dim3 numBlocks((gradients->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                   (gradients->getSize(Y) + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuBackward<<<numBlocks, threadsPerBlock>>>(gradients->getDeviceData(), gradients->getSize(X), gradients->getSize(Y), gradients->getDeviceData());
    return new Tensor2D(gradients->getSize(X), gradients->getSize(Y), gradients->getDeviceData());
}
