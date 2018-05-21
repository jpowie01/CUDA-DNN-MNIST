#include "sgd.h"

SGDOptimizer::SGDOptimizer(float learningRate) {
    this->learningRate = learningRate;
}

void SGDOptimizer::optimize(Layer* layer) {
    // Scale deltas with learning rate
    layer->deltaWeights->scale(this->learningRate);
    layer->deltaBias->scale(this->learningRate);

    // Update weights by subtracting deltas
    layer->weights->subtract(layer->deltaWeights);
    layer->bias->subtract(layer->deltaBias);
}
