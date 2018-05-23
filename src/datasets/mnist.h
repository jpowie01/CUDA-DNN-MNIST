#pragma once
#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>

#include "../tensor/tensor2d.h"
#include "../utils.h"


enum DataSetType {
    TRAIN,
    TEST
};


class MNISTDataSet {
private:
    float** images;
    float** labels;
    int size;

public:
    MNISTDataSet(DataSetType type = TRAIN);
    
    int getSize();
    void shuffle();
    Tensor2D* getBatchOfImages(int index, int size);
    Tensor2D* getBatchOfLabels(int index, int size);
};

#endif  /* !MNIST_H */
