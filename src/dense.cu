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
    float maxWeight = -1.0f / sqrt(input);
    for (int y = 0; y < output; y++) {
        for (int x = 0; x < input; x++) {
            initialWeigths[y][x] = randomFloat(minWeight, maxWeight);
        }
    }
    this->weights = new Tensor2D(output, input, initialWeigths);

    // Prepare place for initial bias on CPU
    float** initialBias = new float*[16];  // TODO: Take it as a batch size
    *initialBias = new float[16 * output];
    for (int i = 1; i < 16; i++) initialBias[i] = initialBias[i-1] + output;
    
    // Fill weights with some float numbers
    for (int batch = 0; batch < 16; batch++) {  // TODO: Take it as a batch size
        for (int x = 0; x < output; x++) {
            initialBias[batch][x] = 0;
        }
    }
    this->bias = new Tensor2D(output, 16, initialBias);  // TODO: Take it as a batch size
}

Tensor2D* DenseLayer::forward(Tensor2D* data) {
    // Save this data - will be needed for backpropagation
    this->inputData = data;

    Tensor2D* output = this->inputData->multiply(this->weights);
    output->add(this->bias);

    // TODO: Remove me or wrap with DEBUG flag.
    printf("Input Data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    printf("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    printf("Bias = X: %d Y: %d\n", this->bias->sizeX, this->bias->sizeY);
    printf("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);

    return output;
}

Tensor2D* DenseLayer::backward(Tensor2D* gradients) {
    Tensor2D* deltaWeights = this->inputData->transposeAndMultiply(gradients);
    Tensor2D* deltaBias = gradients->meanX();
    Tensor2D* output = gradients->multiplyByTransposition(this->weights);

    printf("Input data = X: %d Y: %d\n", this->inputData->sizeX, this->inputData->sizeY);
    printf("Gradients = X: %d Y: %d\n", gradients->sizeX, gradients->sizeY);
    printf("Weights = X: %d Y: %d\n", this->weights->sizeX, this->weights->sizeY);
    printf("Delta Weights = X: %d Y: %d\n", deltaWeights->sizeX, deltaWeights->sizeY);
    printf("Bias = X: %d Y: %d\n", this->bias->sizeX, this->bias->sizeY);
    printf("Delta Bias = X: %d Y: %d\n", deltaBias->sizeX, deltaBias->sizeY);
    printf("Output = X: %d Y: %d\n", output->sizeX, output->sizeY);

    return output;
}
