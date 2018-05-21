#pragma once
#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor1d.h"
#include "../tensor/tensor2d.h"
#include "../utils.h"
#include "base.h"


class DenseLayer: public Layer {
private:
    int input;
    int output;
    
    Tensor2D* inputData;

public:
    DenseLayer(int input, int output);

    Tensor2D* forward(Tensor2D* data);
    Tensor2D* backward(Tensor2D* gradients);
};

#endif  /* !DENSE_H */
