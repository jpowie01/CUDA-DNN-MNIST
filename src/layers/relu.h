#pragma once
#ifndef RELU_H
#define RELU_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor1d.h"
#include "../tensor/tensor2d.h"
#include "layer.h"


class ReLuLayer: public Layer {
private:
    int input;
    int output;
    
    Tensor2D* inputData;
    Tensor2D* outputForward;
    Tensor2D* outputBackward;

public:
    ReLuLayer(int inputOutput);

    Tensor2D* forward(Tensor2D* data);
    Tensor2D* backward(Tensor2D* gradients);
};

#endif  /* !RELU_H */
