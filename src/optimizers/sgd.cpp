#include "sgd.hpp"

SGDOptimizer::SGDOptimizer(float learningRate) {
    this->learningRate = learningRate;
}

void SGDOptimizer::optimize(Layer* layer) {
    // Scale deltas with learning rate
    if (layer->getDeltaWeights()) {
        layer->getDeltaWeights()->scale(this->learningRate);
    }
    if (layer->getDeltaBias()) {
        layer->getDeltaBias()->scale(this->learningRate);
    }

    // Update weights by subtracting deltas
    if (layer->getWeights()) {
        layer->getWeights()->subtract(layer->getDeltaWeights());
    }
    if (layer->getBias()) {
        layer->getBias()->subtract(layer->getDeltaBias());
    }
}
