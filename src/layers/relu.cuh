#pragma once
#ifndef RELU_HPP
#define RELU_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor1d.cuh"
#include "../tensor/tensor2d.cuh"
#include "layer.hpp"


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

#endif  /* !RELU_HPP */
