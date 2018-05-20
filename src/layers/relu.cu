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
            b[y*sizeX + x] = a[y*sizeX + x];  // TODO: Shouldn't it be 1?
        }
    }
}

ReLuLayer::ReLuLayer(int inputOutput) {
    this->input = this->output = inputOutput;

    // TODO: Remove this by making ReLu Layer an Activation Layer
    this->weights = new Tensor2D(0, 0, (float*)NULL);
    this->bias = new Tensor2D(0, 0, (float*)NULL);
    this->deltaWeights = new Tensor2D(0, 0, (float*)NULL);
    this->deltaBias = new Tensor2D(0, 0, (float*)NULL);
}

Tensor2D* ReLuLayer::forward(Tensor2D* data) {
    // TODO: Check if I really need to create new Tensor here
    float* output;
    cudaMalloc((void **)&(output), data->sizeX*data->sizeY*sizeof(float));

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((data->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (data->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuForward<<<numBlocks, threadsPerBlock>>>(data->getDeviceData(), data->sizeX, data->sizeY, output);

    return new Tensor2D(data->sizeX, data->sizeY, output);
}
 
Tensor2D* ReLuLayer::backward(Tensor2D* gradients) {
    // TODO: Check if I really need to create new Tensor here
    float* output;
    cudaMalloc((void **)&(output), gradients->sizeX*gradients->sizeY*sizeof(float));

    dim3 threadsPerBlock(16, 16);  // TODO: Extract this somewhere else, so we'll be able to easily change it during experiments
    dim3 numBlocks((gradients->sizeX + threadsPerBlock.x)/threadsPerBlock.x,
                   (gradients->sizeY + threadsPerBlock.y)/threadsPerBlock.y);
    kReLuBackward<<<numBlocks, threadsPerBlock>>>(gradients->getDeviceData(), gradients->sizeX, gradients->sizeY, output);

    return new Tensor2D(gradients->sizeX, gradients->sizeY, output);
}
