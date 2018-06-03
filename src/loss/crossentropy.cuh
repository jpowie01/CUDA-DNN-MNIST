#pragma once
#ifndef CROSSENTROPY_HPP
#define CROSSENTROPY_HPP

#include <cstdio>
#include <cmath>

#include "../tensor/tensor2d.cuh"
#include "lossfunction.hpp"

#define VERY_SMALL_NUMBER 1e-10


class CrossEntropyLoss: public LossFunction {
private:

public:
    CrossEntropyLoss();

    Tensor2D* calculate(Tensor2D* networkOutput, Tensor2D* labels, Tensor2D* output);
    float getLoss(Tensor2D* networkOutput, Tensor2D* labels);
    float getAccuracy(Tensor2D* networkOutput, Tensor2D* labels);
};

#endif  /* !CROSSENTROPY_HPP */
