#include "dense.h"

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

    // Clean memory
    delete[] initialWeigths;
    delete[] initialBias;
}

Tensor2D* DenseLayer::forward(Tensor2D* data) {
    // Save this data - will be needed for backpropagation
    this->inputData = data;

    // Calculate on GPU: Y = x * W + b
    Tensor2D* output = this->inputData->multiply(this->weights);
    output->add(this->bias);

    DEBUG_PRINT("=== Layer %d ===\n", this);
    DEBUG_PRINT("Input Data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    DEBUG_PRINT("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    DEBUG_PRINT("Bias = X: %d\n", this->bias->size);
    DEBUG_PRINT("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);
    return output;
}

Tensor2D* DenseLayer::backward(Tensor2D* gradients, bool firstLayer) {
    if (this->deltaWeights) {
        delete this->deltaWeights;
    }
    if (this->deltaBias) {
        delete this->deltaBias;
    }
    this->deltaWeights = this->inputData->transposeAndMultiply(gradients);
    this->deltaBias = gradients->meanX();

    DEBUG_PRINT("\n=== Layer %d ===\n", this);
    DEBUG_PRINT("Input data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    DEBUG_PRINT("Gradients = X: %d Y: %d\n", gradients->sizeX, gradients->sizeY);
    DEBUG_PRINT("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    DEBUG_PRINT("Delta Weights (%d) = X: %d Y: %d\n", this->deltaWeights, this->deltaWeights->sizeX, this->deltaWeights->sizeY);
    DEBUG_PRINT("Bias = X: %d\n", this->bias->size);
    DEBUG_PRINT("Delta Bias (%d) = X: %d\n", this->deltaBias, this->deltaBias->size);

    if (firstLayer) {
        delete gradients;
        return NULL;
    }

    Tensor2D* output = gradients->multiplyByTransposition(this->weights);
    DEBUG_PRINT("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);

    delete gradients;
    delete this->inputData;
    return output;
}
