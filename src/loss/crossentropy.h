#pragma once
#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <stdio.h>
#include <cmath>

#include "../tensor/tensor2d.h"
#include "lossfunction.h"


class CrossEntropyLoss: public LossFunction {
private:

public:
    CrossEntropyLoss();

    Tensor2D* calculate(Tensor2D* networkOutput, Tensor2D* labels, Tensor2D* output);
    float getLoss(Tensor2D* networkOutput, Tensor2D* labels);
    float getAccuracy(Tensor2D* networkOutput, Tensor2D* labels);
};

#endif  /* !CROSSENTROPY_H */
