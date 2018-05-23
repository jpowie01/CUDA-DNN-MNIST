#pragma once
#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>

#include "../tensor/tensor2d.h"


class MNISTDataSet {
private:
    unsigned char* bufferImages;
    unsigned char* bufferLabels;
    int size;

public:
    MNISTDataSet();
    
    int getSize();
    Tensor2D* getBatchOfImages(int index, int size);
    Tensor2D* getBatchOfLabels(int index, int size);
};

#endif  /* !MNIST_H */
