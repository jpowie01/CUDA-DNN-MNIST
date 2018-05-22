#pragma once
#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor1d.h"
#include "../tensor/tensor2d.h"

class Layer {
public:
    // TODO: Make it private!
    Tensor2D* weights;
    Tensor1D* bias;
    Tensor2D* deltaWeights;
    Tensor1D* deltaBias;

    virtual Tensor2D* forward(Tensor2D* data) = 0;
    virtual Tensor2D* backward(Tensor2D* gradients, bool firstLayer = false) = 0;
};

#endif  /* !BASE_LAYER_H */
