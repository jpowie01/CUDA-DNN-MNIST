#pragma once
#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"
#include "base.h"


class CrossEntropyLoss: public LossFunction {
private:

public:
    CrossEntropyLoss();

    Tensor2D* calculate(Tensor2D* output, Tensor2D* labels);
    float getLoss(Tensor2D* networkOutput, Tensor2D* labels);
};

#endif  /* !CROSSENTROPY_H */
