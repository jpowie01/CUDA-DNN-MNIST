#pragma once
#ifndef DENSE_HPP
#define DENSE_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor1d.cuh"
#include "../tensor/tensor2d.cuh"
#include "../utils.hpp"
#include "layer.hpp"


class DenseLayer: public Layer {
private:
    int input;
    int output;
    
    Tensor2D* inputData;
    Tensor2D* outputForward;
    Tensor2D* outputBackward;

public:
    DenseLayer(int input, int output);

    Tensor2D* forward(Tensor2D* data);
    Tensor2D* backward(Tensor2D* gradients);
};

#endif  /* !DENSE_HPP */
