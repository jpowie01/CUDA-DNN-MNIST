#pragma once
#ifndef DENSE_H
#define DENSE_H

#include <stdio.h>
#include <cmath>

#include "tensor2d.h"
#include "utils.h"


class DenseLayer {
private:
    int input;
    int output;
    
    Tensor2D* inputData;
    Tensor2D* weights;
    Tensor2D* bias;

public:
    DenseLayer(int input, int output);

    Tensor2D* forward(Tensor2D* data);
    Tensor2D* backward(Tensor2D* gradients);
};

#endif  /* !DENSE_H */
