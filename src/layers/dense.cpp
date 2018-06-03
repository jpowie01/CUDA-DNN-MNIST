#include "dense.hpp"

DenseLayer::DenseLayer(int input, int output) {
    this->input = input;
    this->output = output;

    // Prepare place for initial weights on CPU
    float** initialWeigths = new float*[output];
    *initialWeigths = new float[input * output];
    for (int i = 1; i < output; i++) initialWeigths[i] = initialWeigths[i-1] + input;

    // Fill weights with some float numbers
    float minWeight = -1.0f / sqrt(input);
    float maxWeight = 1.0f / sqrt(input);
    for (int y = 0; y < output; y++) {
        for (int x = 0; x < input; x++) {
            initialWeigths[y][x] = randomFloat(minWeight, maxWeight);
        }
    }
    this->weights = new Tensor2D(output, input, initialWeigths);
    this->deltaWeights = NULL;

    // Prepare place for initial bias on CPU
    float* initialBias = new float[output];
    
    // Fill weights with some float numbers
    for (int x = 0; x < output; x++) {
        initialBias[x] = 0;
    }
    this->bias = new Tensor1D(output, initialBias);
    this->deltaBias = NULL;

    // Prepare output for forward and backprop
    this->outputForward = NULL;
    this->outputBackward = NULL;

    // Clean memory
    delete[] initialWeigths;
    delete[] initialBias;
}

Tensor2D* DenseLayer::forward(Tensor2D* data) {
    // Save this data - will be needed for backpropagation
    this->inputData = data;
    if (!this->outputForward) {
        this->outputForward = new Tensor2D(this->weights->getSize(X), this->inputData->getSize(Y));
    }

    // Calculate on GPU: Y = x * W + b
    this->inputData->multiply(this->weights, this->outputForward);
    this->outputForward->add(this->bias);

    DEBUG_PRINT("=== Layer %d ===\n", this);
    DEBUG_PRINT("Input Data = X: %d Y: %d\n", this->inputData->getSize(X), this->inputData->getSize(Y));
    DEBUG_PRINT("Weights = X: %d Y: %d\n", this->weights->getSize(X), this->weights->getSize(Y));
    DEBUG_PRINT("Bias = X: %d\n", this->bias->getSize());
    DEBUG_PRINT("Output = X: %d Y: %d\n", this->outputForward->getSize(X), this->outputForward->getSize(Y));
    return this->outputForward;
}

Tensor2D* DenseLayer::backward(Tensor2D* gradients) {
    if (!this->deltaWeights) {
        this->deltaWeights = new Tensor2D(gradients->getSize(X), this->inputData->getSize(X));
    }
    if (!this->deltaBias) {
        this->deltaBias = new Tensor1D(gradients->getSize(X));
    }
    this->inputData->transposeAndMultiply(gradients, this->deltaWeights);
    gradients->meanX(this->deltaBias);

    DEBUG_PRINT("\n=== Layer %d ===\n", this);
    DEBUG_PRINT("Input data = X: %d Y: %d\n", this->inputData->getSize(X), this->inputData->getSize(Y));
    DEBUG_PRINT("Gradients = X: %d Y: %d\n", gradients->getSize(X), gradients->getSize(Y));
    DEBUG_PRINT("Weights = X: %d Y: %d\n", this->weights->getSize(X), this->weights->getSize(Y));
    DEBUG_PRINT("Delta Weights (%d) = X: %d Y: %d\n", this->deltaWeights, this->deltaWeights->getSize(X), this->deltaWeights->getSize(Y));
    DEBUG_PRINT("Bias = X: %d\n", this->bias->getSize());
    DEBUG_PRINT("Delta Bias (%d) = X: %d\n", this->deltaBias, this->deltaBias->getSize());

    if (!this->outputBackward) {
        this->outputBackward = new Tensor2D(this->weights->getSize(Y), gradients->getSize(Y));
    }
    gradients->multiplyByTransposition(this->weights, this->outputBackward);
    DEBUG_PRINT("Output = X: %d Y: %d\n", this->outputBackward->getSize(X), this->outputBackward->getSize(Y));
    return this->outputBackward;
}
