#include "relu.cuh"

__global__
void kReLu(float *A, int aX, int aY, float* B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < aX && y < aY) {
        if (A[y*aX + x] < 0.0) {
            B[y*aX + x] = 0;
        } else {
            B[y*aX + x] = A[y*aX + x];
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
    kReLu<<<numBlocks, threadsPerBlock>>>(
        data->getDeviceData(), data->getSize(X), data->getSize(Y),
        this->outputForward->getDeviceData()
    );
    return this->outputForward;
}
 
Tensor2D* ReLuLayer::backward(Tensor2D* gradients) {
    dim3 threadsPerBlock(Configuration::reLuBlockSize, Configuration::reLuBlockSize);
    dim3 numBlocks((gradients->getSize(X) + threadsPerBlock.x)/threadsPerBlock.x,
                   (gradients->getSize(Y) + threadsPerBlock.y)/threadsPerBlock.y);
    kReLu<<<numBlocks, threadsPerBlock>>>(
        gradients->getDeviceData(), gradients->getSize(X), gradients->getSize(Y),
        gradients->getDeviceData()
    );
    return new Tensor2D(gradients->getSize(X), gradients->getSize(Y), gradients->getDeviceData());
}
