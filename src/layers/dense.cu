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

    // Prepare place for initial bias on CPU
    float* initialBias = new float[output];
    
    // Fill weights with some float numbers
    for (int x = 0; x < output; x++) {
        initialBias[x] = 0;
    }
    this->bias = new Tensor1D(output, initialBias);
}

Tensor2D* DenseLayer::forward(Tensor2D* data) {
    // Save this data - will be needed for backpropagation
    this->inputData = data;

    Tensor2D* output = this->inputData->multiply(this->weights);
    output->add(this->bias);

    // TODO: Remove me or wrap with DEBUG flag.
    /*
    printf("\n=== Layer %d ===\n", this);
    printf("Input Data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    printf("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    printf("Bias = X: %d Y: %d\n", this->bias->sizeX, this->bias->sizeY);
    printf("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);
    */
    return output;
}

Tensor2D* DenseLayer::backward(Tensor2D* gradients) {
    this->deltaWeights = this->inputData->transposeAndMultiply(gradients);
    this->deltaBias = gradients->meanX();
    Tensor2D* output = gradients->multiplyByTransposition(this->weights);

    // TODO: Remove me or wrap with DEBUG flag.
    /*
    printf("\n=== Layer %d ===\n", this);
    printf("Input data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    printf("Gradients = X: %d Y: %d\n", gradients->sizeX, gradients->sizeY);
    printf("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    printf("Delta Weights (%d) = X: %d Y: %d\n", this->deltaWeights, this->deltaWeights->sizeX, this->deltaWeights->sizeY);
    printf("Bias = X: %d Y: %d\n", this->bias->sizeX, this->bias->sizeY);
    printf("Delta Bias (%d) = X: %d Y: %d\n", this->deltaBias, this->deltaBias->sizeX, this->deltaBias->sizeY);
    printf("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);
    */
    return output;
}
