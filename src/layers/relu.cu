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
        this->outputForward = new Tensor2D(data->sizeX, data->sizeY);
    }

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((data->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (data->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuForward<<<numBlocks, threadsPerBlock>>>(data->getDeviceData(), data->sizeX, data->sizeY, this->outputForward->getDeviceData());
    return this->outputForward;
}
 
Tensor2D* ReLuLayer::backward(Tensor2D* gradients) {
    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((gradients->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (gradients->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuBackward<<<numBlocks, threadsPerBlock>>>(gradients->getDeviceData(), gradients->sizeX, gradients->sizeY, gradients->getDeviceData());
    return new Tensor2D(gradients->sizeX, gradients->sizeY, gradients->getDeviceData());
}
