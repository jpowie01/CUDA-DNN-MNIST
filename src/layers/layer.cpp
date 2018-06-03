#include "layer.hpp"

Tensor2D* Layer::getWeights() {
    return this->weights;
}

Tensor1D* Layer::getBias() {
    return this->bias;
}

Tensor2D* Layer::getDeltaWeights() {
    return this->deltaWeights;
}

Tensor1D* Layer::getDeltaBias() {
    return this->deltaBias;
}
