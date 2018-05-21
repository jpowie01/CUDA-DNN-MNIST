#include "sgd.h"

SGDOptimizer::SGDOptimizer(float learningRate) {
    this->learningRate = learningRate;
}

void SGDOptimizer::optimize(Layer* layer) {
    // Scale deltas with learning rate
    if (layer->deltaWeights->getDeviceData()) {
        layer->deltaWeights->scale(this->learningRate);
    }
    if (layer->deltaBias->getDeviceData()) {
        layer->deltaBias->scale(this->learningRate);
    }

    // Update weights by subtracting deltas
    if (layer->weights->getDeviceData()) {
        layer->weights->subtract(layer->deltaWeights);
    }
    if (layer->bias->getDeviceData()) {
        layer->bias->subtract(layer->deltaBias);
    }
}
