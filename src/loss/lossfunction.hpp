#pragma once
#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor2d.cuh"

class LossFunction {
public:
    virtual Tensor2D* calculate(Tensor2D* networkOutput, Tensor2D* labels, Tensor2D* output) = 0;
    virtual float getLoss(Tensor2D* networkOutput, Tensor2D* labels) = 0;
    virtual float getAccuracy(Tensor2D* networkOutput, Tensor2D* labels) = 0;
};

#endif  /* !LOSS_FUNCTION_HPP */
