#pragma once
#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"

class Layer {
public:
    // TODO: Make it private!
    Tensor2D* weights;
    Tensor2D* bias;
    Tensor2D* deltaWeights;
    Tensor2D* deltaBias;

    virtual Tensor2D* forward(Tensor2D* data) = 0;
    virtual Tensor2D* backward(Tensor2D* gradients) = 0;
};

#endif  /* !BASE_LAYER_H */
