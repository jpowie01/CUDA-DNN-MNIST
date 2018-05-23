#include "sgd.h"

SGDOptimizer::SGDOptimizer(float learningRate) {
    this->learningRate = learningRate;
}

void SGDOptimizer::optimize(Layer* layer) {
    // Scale deltas with learning rate
    if (layer->deltaWeights) {
        layer->deltaWeights->scale(this->learningRate);
    }
    if (layer->deltaBias) {
        layer->deltaBias->scale(this->learningRate);
    }

    // Update weights by subtracting deltas
    if (layer->weights) {
        layer->weights->subtract(layer->deltaWeights);
    }
    if (layer->bias) {
        layer->bias->subtract(layer->deltaBias);
    }
}
