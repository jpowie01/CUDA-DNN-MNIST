#pragma once
#ifndef BASE_LOSS_H
#define BASE_LOSS_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"

class LossFunction {
public:
    virtual Tensor2D* calculate(Tensor2D* output, Tensor2D* labels) = 0;
};

#endif  /* !BASE_LOSS_H */
