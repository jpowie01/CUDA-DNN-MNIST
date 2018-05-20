#pragma once
#ifndef RELU_H
#define RELU_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"
#include "base.h"


class ReLuLayer: public Layer {
private:
    int input;
    int output;
    
    Tensor2D* inputData;

public:
    ReLuLayer(int inputOutput);

    Tensor2D* forward(Tensor2D* data);
    Tensor2D* backward(Tensor2D* gradients);
};

#endif  /* !RELU_H */
